from __future__ import annotations

import json
from typing import Any

from curl_cffi import requests
from fastapi import HTTPException

from services.config import config
from services.proxy_service import proxy_settings

DEFAULT_REVIEW_PROMPT = "判断用户请求是否允许。只回答 ALLOW 或 REJECT。"


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_text(item) for item in value)
    if isinstance(value, dict):
        return "\n".join(_text(value.get(key)) for key in ("text", "input_text", "content", "input", "instructions", "system", "prompt"))
    return ""


def request_text(*values: object) -> str:
    return "\n".join(part for value in values if (part := _text(value).strip()))


def _response_excerpt(value: object, limit: int = 1200) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, default=str)
    else:
        text = str(value or "")
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _review_response_content(data: Any) -> str:
    if not isinstance(data, dict):
        raise ValueError(f"AI review returned non-object JSON: {_response_excerpt(data)}")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        upstream_error = data.get("error")
        message = ""
        if isinstance(upstream_error, dict):
            message = str(upstream_error.get("message") or upstream_error.get("code") or "").strip()
        elif upstream_error:
            message = str(upstream_error).strip()
        suffix = f": {message}" if message else ""
        raise ValueError(f"AI review response missing choices{suffix}; body={_response_excerpt(data)}")

    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError(f"AI review choice is not an object: {_response_excerpt(first)}")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError(f"AI review choice missing message: {_response_excerpt(first)}")
    content = message.get("content")
    if content is None:
        raise ValueError(f"AI review message missing content: {_response_excerpt(message)}")
    return str(content)


def check_request(text: str) -> None:
    text = str(text or "")
    if not text:
        return
    for word in config.sensitive_words:
        if word in text:
            raise HTTPException(status_code=400, detail={"error": "检测到敏感词，拒绝本次任务"})
    review = config.ai_review
    if not review.get("enabled"):
        return
    base_url = str(review.get("base_url") or "").strip().rstrip("/")
    api_key = str(review.get("api_key") or "").strip()
    model = str(review.get("model") or "").strip()
    if not base_url or not api_key or not model:
        raise HTTPException(status_code=400, detail={"error": "ai review config is incomplete"})
    prompt = str(review.get("prompt") or DEFAULT_REVIEW_PROMPT).strip()
    content = f"{prompt}\n\n用户请求:\n{text}\n\n只回答 ALLOW 或 REJECT。"
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0},
            timeout=60,
            **proxy_settings.build_session_kwargs(),
        )
        try:
            data = response.json()
        except Exception as exc:
            raise ValueError(
                f"AI review returned non-JSON response: status={response.status_code}, "
                f"body={_response_excerpt(response.text)}"
            ) from exc
        if not 200 <= int(response.status_code) < 300:
            raise ValueError(f"AI review HTTP {response.status_code}: body={_response_excerpt(data)}")
        result = _review_response_content(data).strip().lower()
    except Exception as exc:
        raise HTTPException(status_code=502, detail={"error": f"ai review failed: {exc}"}) from exc
    if result.startswith(("allow", "pass", "true", "yes", "通过", "允许", "安全")):
        return
    raise HTTPException(status_code=400, detail={"error": "AI 审核未通过，拒绝本次任务"})
