from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import tiktoken

from services.account_service import account_service
from services.config import config
from services.image_backends import get_image_backend_specs, is_image_backend_fallback_error
from services.image_backends.base import ImageBackendSpec
from services.openai_backend_api import OpenAIBackendAPI
from utils.helper import IMAGE_MODELS, extract_image_from_message_content, is_transient_connection_error_message
from utils.log import logger


class ImageGenerationError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = 502,
        error_type: str = "server_error",
        code: str | None = "upstream_error",
        param: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.code = code
        self.param = param
        self.detail = detail or {}

    def to_openai_error(self) -> dict[str, Any]:
        return {
            "error": {
                "message": str(self),
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


def is_token_invalid_error(message: str) -> bool:
    text = str(message or "").lower()
    return (
        "token_invalidated" in text
        or "token_revoked" in text
        or "authentication token has been invalidated" in text
        or "invalidated oauth token" in text
    )


def image_stream_error_message(message: str) -> str:
    text = str(message or "")
    if is_transient_connection_error_message(text):
        return "upstream image connection failed, please retry later"
    return text or "image generation failed"


def is_upstream_image_connection_error(message: str) -> bool:
    return image_stream_error_message(message) == "upstream image connection failed, please retry later"


def _short_text(value: object, limit: int = 500) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _image_no_data_detail(
    *,
    conversation_id: str,
    file_ids: list[str],
    sediment_ids: list[str],
    message: str,
    last_event_type: str,
    tool_invoked: object,
    turn_use_case: str,
    resolution_errors: list[str],
) -> dict[str, Any]:
    return {
        "reason": "image_generation_finished_without_image_data",
        "conversation_id": conversation_id,
        "file_ids": file_ids,
        "sediment_ids": sediment_ids,
        "last_event": last_event_type,
        "tool_invoked": tool_invoked,
        "turn_use_case": turn_use_case,
        "message": _short_text(message),
        "resolution_errors": [_short_text(item, 800) for item in resolution_errors],
    }


def _image_no_data_message(detail: dict[str, Any]) -> str:
    parts = ["image generation finished without image data"]
    if detail.get("conversation_id"):
        parts.append(f"conversation_id={detail['conversation_id']}")
    if detail.get("file_ids"):
        parts.append(f"file_ids={detail['file_ids']}")
    if detail.get("sediment_ids"):
        parts.append(f"sediment_ids={detail['sediment_ids']}")
    if detail.get("last_event"):
        parts.append(f"last_event={detail['last_event']}")
    if detail.get("tool_invoked") is not None:
        parts.append(f"tool_invoked={detail['tool_invoked']}")
    if detail.get("turn_use_case"):
        parts.append(f"turn_use_case={detail['turn_use_case']}")
    if detail.get("message"):
        parts.append(f"message={detail['message']}")
    resolution_errors = detail.get("resolution_errors")
    if isinstance(resolution_errors, list) and resolution_errors:
        parts.append(f"resolution_errors={'; '.join(str(item) for item in resolution_errors)}")
    return "; ".join(parts)


def encode_images(images: Iterable[tuple[bytes, str, str]]) -> list[str]:
    return [base64.b64encode(data).decode("ascii") for data, _, _ in images if data]


def save_image_bytes(image_data: bytes, base_url: str | None = None) -> str:
    config.cleanup_old_images()
    file_hash = hashlib.md5(image_data).hexdigest()
    filename = f"{int(time.time())}_{file_hash}.png"
    relative_dir = Path(time.strftime("%Y"), time.strftime("%m"), time.strftime("%d"))
    file_path = config.images_dir / relative_dir / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(image_data)
    return f"{(base_url or config.base_url)}/images/{relative_dir.as_posix()}/{filename}"


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and str(item.get("type") or "") in {"text", "input_text", "output_text"}:
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return ""


def normalize_messages(messages: object, system: Any = None) -> list[dict[str, Any]]:
    normalized = []
    if config.global_system_prompt:
        normalized.append({"role": "system", "content": config.global_system_prompt})
    system_text = message_text(system)
    if system_text:
        normalized.append({"role": "system", "content": system_text})
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role", "user")
            content = message.get("content", "")
            text = message_text(content)
            images: list[tuple[bytes, str]] = []
            if role == "user":
                images.extend(extract_image_from_message_content(content))
                if isinstance(content, list):
                    for part in content:
                        if not isinstance(part, dict) or part.get("type") != "image":
                            continue
                        data = part.get("data")
                        if isinstance(data, (bytes, bytearray)):
                            images.append((bytes(data), str(part.get("mime") or "image/png")))
            if images:
                parts: list[Any] = []
                if text:
                    parts.append({"type": "text", "text": text})
                for data, mime in images:
                    parts.append({"type": "image", "data": data, "mime": mime})
                normalized.append({"role": role, "content": parts})
            else:
                normalized.append({"role": role, "content": text})
    return normalized


def prompt_with_global_system(prompt: str) -> str:
    return f"{config.global_system_prompt}\n\n{prompt}" if config.global_system_prompt else prompt


def assistant_history_text(messages: list[dict[str, Any]]) -> str:
    return "".join(str(item.get("content") or "") for item in messages if item.get("role") == "assistant")


def assistant_history_messages(messages: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("content") or "") for item in messages if item.get("role") == "assistant" and item.get("content")]


def build_image_prompt(prompt: str, size: str | None) -> str:
    if not size:
        return prompt
    if size not in {"1:1", "16:9", "9:16", "4:3", "3:4"}:
        return f"{prompt.strip()}\n\n输出图片，宽高比为 {size}。"
    hint = {
        "1:1": "输出为 1:1 正方形构图，主体居中，适合正方形画幅。",
        "16:9": "输出为 16:9 横屏构图，适合宽画幅展示。",
        "9:16": "输出为 9:16 竖屏构图，适合竖版画幅展示。",
        "4:3": "输出为 4:3 比例，兼顾宽度与高度，适合展示画面细节。",
        "3:4": "输出为 3:4 比例，纵向构图，适合人物肖像或竖向场景。",
    }[size]
    return f"{prompt.strip()}\n\n{hint}"


def image_backend_prompt(prompt: str, size: str | None, backend_spec: ImageBackendSpec) -> str:
    if backend_spec.use_legacy_prompt_hint:
        return build_image_prompt(prompt, size)
    return prompt.strip()


def encoding_for_model(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        try:
            return tiktoken.get_encoding("o200k_base")
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")


def count_message_tokens(messages: list[dict[str, Any]], model: str) -> int:
    encoding = encoding_for_model(model)
    total = 0
    for message in messages:
        total += 3
        for key, value in message.items():
            if not isinstance(value, str):
                continue
            total += len(encoding.encode(value))
            if key == "name":
                total += 1
    return total + 3


def count_text_tokens(text: str, model: str) -> int:
    return len(encoding_for_model(model).encode(text))


def format_image_result(
    items: list[dict[str, Any]],
    prompt: str,
    response_format: str,
    base_url: str | None = None,
    created: int | None = None,
    message: str = "",
) -> dict[str, Any]:
    data: list[dict[str, Any]] = []
    for item in items:
        b64_json = str(item.get("b64_json") or "").strip()
        if not b64_json:
            continue
        revised_prompt = str(item.get("revised_prompt") or prompt).strip() or prompt
        if response_format == "b64_json":
            data.append({
                "b64_json": b64_json,
                "url": save_image_bytes(base64.b64decode(b64_json), base_url),
                "revised_prompt": revised_prompt,
            })
        else:
            data.append({
                "url": save_image_bytes(base64.b64decode(b64_json), base_url),
                "revised_prompt": revised_prompt,
            })
    result: dict[str, Any] = {"created": created or int(time.time()), "data": data}
    if message and not data:
        result["message"] = message
    return result


@dataclass
class ConversationRequest:
    model: str = "auto"
    prompt: str = ""
    messages: list[dict[str, Any]] | None = None
    images: list[str] | None = None
    n: int = 1
    size: str | None = None
    response_format: str = "b64_json"
    base_url: str | None = None
    message_as_error: bool = False
    stream_progress: bool = False


@dataclass
class ConversationState:
    text: str = ""
    conversation_id: str = ""
    file_ids: list[str] = field(default_factory=list)
    sediment_ids: list[str] = field(default_factory=list)
    blocked: bool = False
    tool_invoked: bool | None = None
    turn_use_case: str = ""


@dataclass
class ImageOutput:
    kind: str
    model: str
    index: int
    total: int
    created: int = field(default_factory=lambda: int(time.time()))
    text: str = ""
    upstream_event_type: str = ""
    data: list[dict[str, Any]] = field(default_factory=list)

    def to_chunk(self) -> dict[str, Any]:
        chunk: dict[str, Any] = {
            "object": "image.generation.chunk",
            "created": self.created,
            "model": self.model,
            "index": self.index,
            "total": self.total,
            "progress_text": self.text,
            "upstream_event_type": self.upstream_event_type,
            "data": [],
        }
        if self.kind == "message":
            chunk.update({
                "object": "image.generation.message",
                "message": self.text,
            })
            chunk.pop("progress_text", None)
            chunk.pop("upstream_event_type", None)
        elif self.kind == "result":
            chunk.update({
                "object": "image.generation.result",
                "data": self.data,
            })
            chunk.pop("progress_text", None)
            chunk.pop("upstream_event_type", None)
        return chunk


def assistant_message_text(message: dict[str, Any]) -> str:
    content = message.get("content") or {}
    parts = content.get("parts") or []
    if not isinstance(parts, list):
        return ""
    return "".join(part for part in parts if isinstance(part, str))


def strip_history(text: str, history_text: str = "") -> str:
    text = str(text or "")
    history_text = str(history_text or "")
    while history_text and text.startswith(history_text):
        text = text[len(history_text):]
    return text


def assistant_text(event: dict[str, Any], current_text: str = "", history_text: str = "") -> str:
    for candidate in (event, event.get("v")):
        if not isinstance(candidate, dict):
            continue
        message = candidate.get("message")
        if not isinstance(message, dict):
            continue
        role = str((message.get("author") or {}).get("role") or "").strip().lower()
        if role != "assistant":
            continue
        text = assistant_message_text(message)
        if text:
            return strip_history(text, history_text)
    return apply_text_patch(event, current_text, history_text)


def event_assistant_text(event: dict[str, Any], history_text: str = "") -> str:
    for candidate in (event, event.get("v")):
        if not isinstance(candidate, dict):
            continue
        message = candidate.get("message")
        if isinstance(message, dict) and (message.get("author") or {}).get("role") == "assistant":
            return strip_history(assistant_message_text(message), history_text)
    return ""


def apply_text_patch(event: dict[str, Any], current_text: str = "", history_text: str = "") -> str:
    if event.get("p") == "/message/content/parts/0":
        return apply_patch_op(event, current_text, history_text)

    operations = event.get("v")
    if isinstance(operations, str) and current_text and not event.get("p") and not event.get("o"):
        return current_text + operations

    if event.get("o") == "patch" and isinstance(operations, list):
        text = current_text
        for item in operations:
            if isinstance(item, dict):
                text = apply_text_patch(item, text, history_text)
        return text

    if not isinstance(operations, list):
        return current_text

    text = current_text
    for item in operations:
        if isinstance(item, dict):
            text = apply_text_patch(item, text, history_text)
    return text


def apply_patch_op(operation: dict[str, Any], current_text: str, history_text: str = "") -> str:
    op = operation.get("o")
    value = str(operation.get("v") or "")
    if op == "append":
        return current_text + value
    if op == "replace":
        return strip_history(value, history_text)
    return current_text


def add_unique(values: list[str], candidates: list[str]) -> None:
    for candidate in candidates:
        if candidate and candidate not in values:
            values.append(candidate)


def extract_conversation_ids(payload: str) -> tuple[str, list[str], list[str]]:
    conversation_match = re.search(r'"conversation_id"\s*:\s*"([^"]+)"', payload)
    conversation_id = conversation_match.group(1) if conversation_match else ""
    file_ids = re.findall(r"(file[-_][A-Za-z0-9]+)", payload)
    sediment_ids = re.findall(r"sediment://([A-Za-z0-9_-]+)", payload)
    return conversation_id, file_ids, sediment_ids


def is_image_tool_event(event: dict[str, Any]) -> bool:
    value = event.get("v")
    message = event.get("message") or (value.get("message") if isinstance(value, dict) else None)
    if not isinstance(message, dict):
        return False
    metadata = message.get("metadata") or {}
    author = message.get("author") or {}
    return author.get("role") == "tool" and metadata.get("async_task_type") == "image_gen"


def update_conversation_state(state: ConversationState, payload: str, event: dict[str, Any] | None = None) -> None:
    conversation_id, file_ids, sediment_ids = extract_conversation_ids(payload)
    if conversation_id and not state.conversation_id:
        state.conversation_id = conversation_id
    if isinstance(event, dict) and is_image_tool_event(event):
        add_unique(state.file_ids, file_ids)
        add_unique(state.sediment_ids, sediment_ids)
    if not isinstance(event, dict):
        return
    state.conversation_id = str(event.get("conversation_id") or state.conversation_id)
    value = event.get("v")
    if isinstance(value, dict):
        state.conversation_id = str(value.get("conversation_id") or state.conversation_id)
    if event.get("type") == "moderation":
        moderation = event.get("moderation_response")
        if isinstance(moderation, dict) and moderation.get("blocked") is True:
            state.blocked = True
    if event.get("type") == "server_ste_metadata":
        metadata = event.get("metadata")
        if isinstance(metadata, dict):
            if isinstance(metadata.get("tool_invoked"), bool):
                state.tool_invoked = metadata["tool_invoked"]
            state.turn_use_case = str(metadata.get("turn_use_case") or state.turn_use_case)


def conversation_base_event(event_type: str, state: ConversationState, **extra: Any) -> dict[str, Any]:
    return {
        "type": event_type,
        "text": state.text,
        "conversation_id": state.conversation_id,
        "file_ids": list(state.file_ids),
        "sediment_ids": list(state.sediment_ids),
        "blocked": state.blocked,
        "tool_invoked": state.tool_invoked,
        "turn_use_case": state.turn_use_case,
        **extra,
    }


def iter_conversation_payloads(payloads: Iterator[str], history_text: str = "",
                               history_messages: list[str] | None = None) -> Iterator[dict[str, Any]]:
    state = ConversationState()
    history_messages = history_messages or []
    history_index = 0
    for payload in payloads:
        # print(f"[upstream_sse] {payload}", flush=True)
        if not payload:
            continue
        if payload == "[DONE]":
            yield conversation_base_event("conversation.done", state, done=True)
            break
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            update_conversation_state(state, payload)
            yield conversation_base_event("conversation.raw", state, payload=payload)
            continue
        if not isinstance(event, dict):
            yield conversation_base_event("conversation.event", state, raw=event)
            continue
        update_conversation_state(state, payload, event)
        if history_index < len(history_messages) and event_assistant_text(event, history_text) == history_messages[history_index]:
            history_index += 1
            state.text = ""
            continue
        next_text = assistant_text(event, state.text, history_text)
        if next_text != state.text:
            delta = next_text[len(state.text):] if next_text.startswith(state.text) else next_text
            state.text = next_text
            yield conversation_base_event("conversation.delta", state, raw=event, delta=delta)
            continue
        yield conversation_base_event("conversation.event", state, raw=event)


def conversation_events(
    backend: OpenAIBackendAPI,
    messages: list[dict[str, Any]] | None = None,
    model: str = "auto",
    prompt: str = "",
    images: list[str] | None = None,
    size: str | None = None,
    image_backend: ImageBackendSpec | None = None,
) -> Iterator[dict[str, Any]]:
    normalized = normalize_messages(messages or ([{"role": "user", "content": prompt}] if prompt else []))
    image_model = str(model or "").strip() in IMAGE_MODELS
    history_text = "" if image_model else assistant_history_text(normalized)
    history_messages = [] if image_model else assistant_history_messages(normalized)
    final_prompt = prompt_with_global_system(prompt) if image_model else prompt
    if image_model:
        payloads = backend.stream_image_conversation(
            prompt=final_prompt,
            model=model,
            images=images or [],
            slug_mode=(image_backend.slug_mode if image_backend else "native"),
        )
    else:
        payloads = backend.stream_conversation(
            messages=normalized,
            model=model,
            prompt=final_prompt,
            images=None,
            system_hints=None,
        )
    yield from iter_conversation_payloads(payloads, history_text, history_messages)


def text_backend() -> OpenAIBackendAPI:
    return OpenAIBackendAPI(access_token=account_service.get_text_access_token())


def stream_text_deltas(backend: OpenAIBackendAPI, request: ConversationRequest) -> Iterator[str]:
    attempted_tokens: set[str] = set()
    token = getattr(backend, "access_token", "")
    emitted = False
    while True:
        if token and token in attempted_tokens:
            raise RuntimeError("no available text account")
        if token:
            attempted_tokens.add(token)
        try:
            active_backend = OpenAIBackendAPI(access_token=token)
            for event in conversation_events(active_backend, messages=request.messages, model=request.model, prompt=request.prompt):
                if event.get("type") != "conversation.delta":
                    continue
                delta = str(event.get("delta") or "")
                if delta:
                    emitted = True
                    yield delta
            account_service.mark_text_used(token)
            return
        except Exception as exc:
            error_message = str(exc)
            if token and not emitted and is_token_invalid_error(error_message):
                account_service.remove_invalid_token(token, "text_stream")
                token = account_service.get_text_access_token(attempted_tokens)
                if token:
                    continue
            if is_transient_connection_error_message(error_message):
                logger.warning({
                    "event": "text_stream_connection_interrupted",
                    "request_token": token,
                    "emitted": emitted,
                    "error": _short_text(error_message, 800),
                })
                if emitted:
                    return
                token = account_service.get_text_access_token(attempted_tokens)
                if token:
                    continue
                raise RuntimeError("upstream text connection failed, please retry later") from exc
            raise


def collect_text(backend: OpenAIBackendAPI, request: ConversationRequest) -> str:
    return "".join(stream_text_deltas(backend, request))


def stream_image_outputs(
        backend: OpenAIBackendAPI,
        request: ConversationRequest,
        index: int = 1,
        total: int = 1,
        image_backend: ImageBackendSpec | None = None,
) -> Iterator[ImageOutput]:
    active_image_backend = image_backend or ImageBackendSpec(name="native")
    last: dict[str, Any] = {}
    request_prompt = image_backend_prompt(request.prompt, request.size, active_image_backend)
    stream_interruption_error = ""
    try:
        for event in conversation_events(
                backend,
                prompt=request_prompt,
                model=request.model,
                images=request.images or [],
                size=None,
                image_backend=active_image_backend,
        ):
            last = event
            if event.get("type") == "conversation.delta":
                yield ImageOutput(
                    kind="progress",
                    model=request.model,
                    index=index,
                    total=total,
                    text=str(event.get("delta") or ""),
                    upstream_event_type="conversation.delta",
                )
                continue
            if event.get("type") == "conversation.event":
                raw = event.get("raw")
                raw_type = str(raw.get("type") or "") if isinstance(raw, dict) else ""
                yield ImageOutput(
                    kind="progress",
                    model=request.model,
                    index=index,
                    total=total,
                    upstream_event_type=raw_type,
                )
    except Exception as exc:
        stream_interruption_error = str(exc)
        has_resolution_anchor = bool(
            last.get("conversation_id")
            or last.get("file_ids")
            or last.get("sediment_ids")
        )
        if not is_upstream_image_connection_error(stream_interruption_error) or not has_resolution_anchor:
            raise

    conversation_id = str(last.get("conversation_id") or "")
    file_ids = [str(item) for item in last.get("file_ids") or []]
    sediment_ids = [str(item) for item in last.get("sediment_ids") or []]
    message = str(last.get("text") or "").strip()
    raw = last.get("raw")
    last_event_type = str(raw.get("type") or "") if isinstance(raw, dict) else str(last.get("type") or "")
    is_text_response = last.get("tool_invoked") is False or last.get("turn_use_case") == "text"
    logger.info({
        "event": "image_stream_resolve_start",
        "conversation_id": conversation_id,
        "file_ids": file_ids,
        "sediment_ids": sediment_ids,
        "tool_invoked": last.get("tool_invoked"),
        "turn_use_case": last.get("turn_use_case"),
        "stream_interrupted": bool(stream_interruption_error),
    })
    if stream_interruption_error:
        logger.warning({
            "event": "image_stream_interrupted_resolving",
            "conversation_id": conversation_id,
            "file_ids": file_ids,
            "sediment_ids": sediment_ids,
            "error": _short_text(stream_interruption_error, 800),
        })
    if message and not file_ids and not sediment_ids and (last.get("blocked") or is_text_response):
        yield ImageOutput(kind="message", model=request.model, index=index, total=total, text=message)
        return

    image_urls: list[str] = []
    resolution_errors: list[str] = []
    resolver = getattr(backend, "iter_conversation_image_resolution", None) if request.stream_progress else None
    if callable(resolver):
        for resolve_event in resolver(conversation_id, file_ids, sediment_ids, progress_text=message):
            event_type = str(resolve_event.get("type") or "")
            if event_type == "image.resolve.done":
                image_urls = [str(item) for item in resolve_event.get("urls") or []]
                resolution_errors = [str(item) for item in resolve_event.get("errors") or []]
                break
            if resolve_event.get("error"):
                resolution_errors.append(str(resolve_event.get("error")))
            yield ImageOutput(
                kind="progress",
                model=request.model,
                index=index,
                total=total,
                text=str(resolve_event.get("progress_text") or ""),
                upstream_event_type=event_type,
            )
    else:
        image_urls = backend.resolve_conversation_image_urls(conversation_id, file_ids, sediment_ids)
        resolution_errors = [str(item) for item in getattr(backend, "last_image_resolution_errors", [])]
    if image_urls:
        image_items = [
            {"b64_json": base64.b64encode(image_data).decode("ascii")}
            for image_data in backend.download_image_bytes(image_urls)
        ]
        data = format_image_result(
            image_items,
            request_prompt,
            request.response_format,
            request.base_url,
            int(time.time()),
        )["data"]
        if data:
            yield ImageOutput(kind="result", model=request.model, index=index, total=total, data=data)
        return

    if message and not file_ids and not sediment_ids and not resolution_errors:
        yield ImageOutput(kind="message", model=request.model, index=index, total=total, text=message)
        return

    detail = _image_no_data_detail(
        conversation_id=conversation_id,
        file_ids=file_ids,
        sediment_ids=sediment_ids,
        message=message,
        last_event_type=last_event_type,
        tool_invoked=last.get("tool_invoked"),
        turn_use_case=str(last.get("turn_use_case") or ""),
        resolution_errors=resolution_errors,
    )
    raise ImageGenerationError(_image_no_data_message(detail), detail=detail)


def stream_image_outputs_with_pool(request: ConversationRequest) -> Iterator[ImageOutput]:
    if str(request.model or "").strip() not in IMAGE_MODELS:
        raise ImageGenerationError("unsupported image model,supported models: " + ", ".join(IMAGE_MODELS))

    emitted = False
    last_error = ""
    for index in range(1, request.n + 1):
        attempted_tokens: set[str] = set()
        while True:
            try:
                token = account_service.get_available_access_token(excluded_tokens=attempted_tokens)
            except RuntimeError as exc:
                if last_error and not emitted and attempted_tokens:
                    raise ImageGenerationError(image_stream_error_message(last_error)) from exc
                if emitted:
                    return
                raise ImageGenerationError(image_stream_error_message(str(exc) or "image generation failed")) from exc

            attempted_tokens.add(token)
            emitted_for_token = False
            terminal_emitted_for_token = False
            returned_message = False
            returned_result = False
            backend_fallbacks: list[dict[str, str]] = []
            backend_specs = get_image_backend_specs()
            try:
                for backend_index, backend_spec in enumerate(backend_specs):
                    logger.info({
                        "event": "image_backend_attempt",
                        "backend": backend_spec.name,
                        "model": request.model,
                        "size": request.size,
                        "index": index,
                        "total": request.n,
                        "fallback_count": len(backend_fallbacks),
                    })
                    backend = OpenAIBackendAPI(access_token=token)
                    try:
                        for output in stream_image_outputs(backend, request, index, request.n, image_backend=backend_spec):
                            if output.kind == "message" and request.message_as_error:
                                raise ImageGenerationError(
                                    output.text or "Image generation was rejected by upstream policy.",
                                    status_code=400,
                                    error_type="invalid_request_error",
                                    code="content_policy_violation",
                                )
                            emitted = True
                            emitted_for_token = True
                            terminal_emitted_for_token = terminal_emitted_for_token or output.kind in {"message", "result"}
                            returned_message = output.kind == "message"
                            returned_result = returned_result or output.kind == "result"
                            yield output
                        logger.info({
                            "event": "image_backend_success",
                            "backend": backend_spec.name,
                            "model": request.model,
                            "size": request.size,
                            "token_attempted": token,
                            "fell_back": bool(backend_fallbacks),
                            "fallbacks": backend_fallbacks,
                        })
                        break
                    except ImageGenerationError as exc:
                        error_message = str(exc)
                        if is_image_backend_fallback_error(error_message) and backend_index < len(backend_specs) - 1:
                            backend_fallbacks.append({"backend": backend_spec.name, "reason": error_message})
                            logger.warning({
                                "event": "image_backend_fallback",
                                "backend": backend_spec.name,
                                "next_backend": backend_specs[backend_index + 1].name,
                                "error": error_message,
                                "model": request.model,
                                "size": request.size,
                            })
                            continue
                        raise
                    except Exception as exc:
                        error_message = str(exc)
                        if not emitted_for_token and is_token_invalid_error(error_message):
                            raise
                        if is_image_backend_fallback_error(image_stream_error_message(error_message)) and backend_index < len(backend_specs) - 1:
                            backend_fallbacks.append({"backend": backend_spec.name, "reason": error_message})
                            logger.warning({
                                "event": "image_backend_fallback",
                                "backend": backend_spec.name,
                                "next_backend": backend_specs[backend_index + 1].name,
                                "error": error_message,
                                "model": request.model,
                                "size": request.size,
                            })
                            continue
                        raise
                else:
                    raise ImageGenerationError("image generation failed")
                if returned_message or not returned_result:
                    account_service.mark_image_result(token, False)
                    return
                account_service.mark_image_result(token, True)
                break
            except ImageGenerationError:
                account_service.mark_image_result(token, False)
                raise
            except Exception as exc:
                account_service.mark_image_result(token, False)
                last_error = str(exc)
                logger.warning({"event": "image_stream_fail", "request_token": token, "error": last_error})
                if not emitted_for_token and is_token_invalid_error(last_error):
                    account_service.remove_invalid_token(token, "image_stream")
                    continue
                if not terminal_emitted_for_token and is_upstream_image_connection_error(last_error):
                    continue
                raise ImageGenerationError(image_stream_error_message(last_error)) from exc

    if not emitted:
        raise ImageGenerationError(image_stream_error_message(last_error))


def stream_image_chunks(outputs: Iterable[ImageOutput]) -> Iterator[dict[str, Any]]:
    for output in outputs:
        yield output.to_chunk()


def collect_image_outputs(outputs: Iterable[ImageOutput]) -> dict[str, Any]:
    created = None
    data: list[dict[str, Any]] = []
    message = ""
    progress_parts: list[str] = []
    for output in outputs:
        created = created or output.created
        if output.kind == "progress" and output.text:
            progress_parts.append(output.text)
        elif output.kind == "message":
            message = output.text
        elif output.kind == "result":
            data.extend(output.data)

    result: dict[str, Any] = {"created": created or int(time.time()), "data": data}
    if not data:
        text = message or "".join(progress_parts).strip()
        if text:
            result["message"] = text
    return result
