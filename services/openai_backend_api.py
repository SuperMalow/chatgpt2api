import base64
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from curl_cffi import requests
from PIL import Image

from services.account_service import account_service
from services.config import config
from services.proxy_service import proxy_settings
from utils.helper import ensure_ok, is_transient_connection_error_message, iter_sse_payloads, new_uuid
from utils.log import logger
from utils.pow import build_legacy_requirements_token, build_proof_token, parse_pow_resources
from utils.turnstile import solve_turnstile_token


class InvalidAccessTokenError(RuntimeError):
    pass


@dataclass
class ChatRequirements:
    """保存一次对话请求所需的 sentinel token。"""
    token: str
    proof_token: str = ""
    turnstile_token: str = ""
    so_token: str = ""
    raw_finalize: Optional[Dict[str, Any]] = None


DEFAULT_CLIENT_VERSION = "prod-be885abbfcfe7b1f511e88b3003d9ee44757fbad"
DEFAULT_CLIENT_BUILD_NUMBER = "5955942"
DEFAULT_POW_SCRIPT = "https://chatgpt.com/backend-api/sentinel/sdk.js"
CODEX_IMAGE_MODEL = "codex-gpt-image-2"


class OpenAIBackendAPI:
    """ChatGPT Web 后端封装。

    说明：
    - 传入 `access_token` 时，聊天和模型列表都会走已登录链路
      例如 `/backend-api/sentinel/chat-requirements`、`/backend-api/conversation`
    - 不传 `access_token` 时，会走未登录链路
      例如 `/backend-anon/sentinel/chat-requirements`、`/backend-anon/conversation`
    - `stream_conversation()` 是底层统一流式入口
    - 协议兼容转换放在 `services.protocol`
    """

    def __init__(self, access_token: str = "") -> None:
        """初始化后端客户端。

        参数：
        - `access_token`：可选。传入后表示使用已登录链路；不传则使用未登录链路。
        """
        self.base_url = "https://chatgpt.com"
        self.client_version = DEFAULT_CLIENT_VERSION
        self.client_build_number = DEFAULT_CLIENT_BUILD_NUMBER
        self.access_token = access_token
        self.fp = self._build_fp()
        self.user_agent = self.fp["user-agent"]
        self.device_id = self.fp["oai-device-id"]
        self.session_id = self.fp["oai-session-id"]
        self.pow_script_sources: list[str] = []
        self.pow_data_build = ""
        self.last_image_resolution_errors: list[str] = []
        self.session = requests.Session(**proxy_settings.build_session_kwargs(
            impersonate=self.fp["impersonate"],
            verify=True,
        ))
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Origin": self.base_url,
            "Referer": self.base_url + "/",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-US;q=0.7",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Priority": "u=1, i",
            "Sec-Ch-Ua": self.fp["sec-ch-ua"],
            "Sec-Ch-Ua-Arch": '"x86"',
            "Sec-Ch-Ua-Bitness": '"64"',
            "Sec-Ch-Ua-Full-Version": '"143.0.3650.96"',
            "Sec-Ch-Ua-Full-Version-List": '"Microsoft Edge";v="143.0.3650.96", "Chromium";v="143.0.7499.147", "Not A(Brand";v="24.0.0.0"',
            "Sec-Ch-Ua-Mobile": self.fp["sec-ch-ua-mobile"],
            "Sec-Ch-Ua-Model": '""',
            "Sec-Ch-Ua-Platform": self.fp["sec-ch-ua-platform"],
            "Sec-Ch-Ua-Platform-Version": '"19.0.0"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "OAI-Device-Id": self.device_id,
            "OAI-Session-Id": self.session_id,
            "OAI-Language": "zh-CN",
            "OAI-Client-Version": self.client_version,
            "OAI-Client-Build-Number": self.client_build_number,
        })
        if self.access_token:
            self.session.headers["Authorization"] = f"Bearer {self.access_token}"

    def _build_fp(self) -> Dict[str, str]:
        account = account_service.get_account(self.access_token) if self.access_token else {}
        account = account if isinstance(account, dict) else {}
        raw_fp = account.get("fp")
        fp = {str(k).lower(): str(v) for k, v in raw_fp.items()} if isinstance(raw_fp, dict) else {}
        for key in (
                "user-agent",
                "impersonate",
                "oai-device-id",
                "oai-session-id",
                "sec-ch-ua",
                "sec-ch-ua-mobile",
                "sec-ch-ua-platform",
        ):
            value = str(account.get(key) or "").strip()
            if value:
                fp[key] = value
        fp.setdefault(
            "user-agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0",
        )
        fp.setdefault("impersonate", "edge101")
        fp.setdefault("oai-device-id", new_uuid())
        fp.setdefault("oai-session-id", new_uuid())
        fp.setdefault("sec-ch-ua", '"Microsoft Edge";v="143", "Chromium";v="143", "Not A(Brand";v="24"')
        fp.setdefault("sec-ch-ua-mobile", "?0")
        fp.setdefault("sec-ch-ua-platform", '"Windows"')
        return fp

    def _headers(self, path: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """构造请求头，并补上 web 端要求的 target path/route。"""
        headers = dict(self.session.headers)
        headers["X-OpenAI-Target-Path"] = path
        headers["X-OpenAI-Target-Route"] = path
        if extra:
            headers.update(extra)
        return headers

    @staticmethod
    def _is_retryable_request_error(exc: Exception) -> bool:
        text = str(exc or "").lower()
        return (
            isinstance(
                exc,
                (
                    requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ProxyError,
                    requests.exceptions.SSLError,
                    requests.exceptions.ConnectTimeout,
                    requests.exceptions.ReadTimeout,
                ),
            )
            or is_transient_connection_error_message(text)
        )

    def _request_with_local_retry(
            self,
            method: str,
            url: str,
            *,
            retry_attempts: int = 3,
            retry_delay: float = 0.8,
            **kwargs,
    ):
        last_error: Exception | None = None
        for attempt in range(1, max(1, retry_attempts) + 1):
            try:
                return self.session.request(method.upper(), url, **kwargs)
            except Exception as exc:
                last_error = exc
                if attempt >= retry_attempts or not self._is_retryable_request_error(exc):
                    raise
                logger.warning({
                    "event": "upstream_request_retry",
                    "method": method.upper(),
                    "url": url,
                    "attempt": attempt,
                    "retry_attempts": retry_attempts,
                    "error": str(exc),
                })
                time.sleep(retry_delay * attempt)
        if last_error is not None:
            raise last_error

    @staticmethod
    def _extract_quota_and_restore_at(limits_progress: list[Any]) -> tuple[int, str | None, bool]:
        for item in limits_progress:
            if isinstance(item, dict) and item.get("feature_name") == "image_gen":
                return int(item.get("remaining") or 0), str(item.get("reset_after") or "") or None, False
        return 0, None, True

    def _get_me(self, timeout: float = 20) -> Dict[str, Any]:
        path = "/backend-api/me"
        response = self._request_with_local_retry(
            "GET",
            self.base_url + path,
            headers=self._headers(path),
            timeout=timeout,
        )
        if response.status_code != 200:
            if response.status_code == 401:
                raise InvalidAccessTokenError(f"{path} failed: HTTP {response.status_code}")
            raise RuntimeError(f"{path} failed: HTTP {response.status_code}")
        return response.json()

    def _get_conversation_init(self, timeout: float = 20) -> Dict[str, Any]:
        path = "/backend-api/conversation/init"
        response = self._request_with_local_retry(
            "POST",
            self.base_url + path,
            headers=self._headers(path, {"Content-Type": "application/json"}),
            json={
                "gizmo_id": None,
                "requested_default_model": None,
                "conversation_id": None,
                "timezone_offset_min": -480,
            },
            timeout=timeout,
        )
        if response.status_code != 200:
            if response.status_code == 401:
                raise InvalidAccessTokenError(f"{path} failed: HTTP {response.status_code}")
            raise RuntimeError(f"{path} failed: HTTP {response.status_code}")
        return response.json()

    def _get_default_account(self, timeout: float = 20) -> Dict[str, Any]:
        route = "/backend-api/accounts/check/v4-2023-04-27"
        response = self._request_with_local_retry(
            "GET",
            self.base_url + route + "?timezone_offset_min=-480",
            headers=self._headers(route),
            timeout=timeout,
        )
        if response.status_code != 200:
            if response.status_code == 401:
                raise InvalidAccessTokenError(f"{route} failed: HTTP {response.status_code}")
            raise RuntimeError(f"/backend-api/accounts/check failed: HTTP {response.status_code}")
        payload = response.json()
        logger.debug({"event": "backend_user_info_account_payload", "account_payload": payload})
        return ((payload.get("accounts") or {}).get("default") or {}).get("account") or {}

    def get_user_info(self, timeout: float | None = None) -> Dict[str, Any]:
        """获取当前 token 的账号信息。"""
        if not self.access_token:
            raise RuntimeError("access_token is required")
        logger.debug({"event": "backend_user_info_start"})
        request_timeout = timeout or 20
        with ThreadPoolExecutor(max_workers=3) as executor:
            me_future = executor.submit(self._get_me, request_timeout)
            init_future = executor.submit(self._get_conversation_init, request_timeout)
            account_future = executor.submit(self._get_default_account, request_timeout)
            me_payload, init_payload, default_account = me_future.result(), init_future.result(), account_future.result()

        plan_type = str(default_account.get("plan_type") or "free")

        limits_progress = init_payload.get("limits_progress")
        limits_progress = limits_progress if isinstance(limits_progress, list) else []
        quota, restore_at, image_quota_unknown = self._extract_quota_and_restore_at(limits_progress)
        result = {
            "email": me_payload.get("email"),
            "user_id": me_payload.get("id"),
            "type": plan_type,
            "quota": quota,
            "image_quota_unknown": image_quota_unknown,
            "limits_progress": limits_progress,
            "default_model_slug": init_payload.get("default_model_slug"),
            "restore_at": restore_at,
            "status": "正常" if image_quota_unknown and plan_type.lower() != "free" else ("限流" if quota == 0 else "正常"),
        }
        logger.debug({
            "event": "backend_user_info_result",
            "email": result.get("email"),
            "user_id": result.get("user_id"),
            "type": result.get("type"),
            "quota": result.get("quota"),
            "image_quota_unknown": result.get("image_quota_unknown"),
            "default_model_slug": result.get("default_model_slug"),
            "restore_at": result.get("restore_at"),
            "status": result.get("status"),
        })
        return result

    def get_quota_info(self, current_account: dict[str, Any] | None = None, timeout: float | None = None) -> Dict[str, Any]:
        """只刷新额度相关字段，避免账号池批量刷新时重复请求完整账号资料。"""
        if not self.access_token:
            raise RuntimeError("access_token is required")
        account = current_account if isinstance(current_account, dict) else {}
        request_timeout = timeout or 20
        init_payload = self._get_conversation_init(request_timeout)
        plan_type = str(account.get("type") or "free")
        limits_progress = init_payload.get("limits_progress")
        limits_progress = limits_progress if isinstance(limits_progress, list) else []
        quota, restore_at, image_quota_unknown = self._extract_quota_and_restore_at(limits_progress)
        return {
            "email": account.get("email"),
            "user_id": account.get("user_id"),
            "type": plan_type,
            "quota": quota,
            "image_quota_unknown": image_quota_unknown,
            "limits_progress": limits_progress,
            "default_model_slug": init_payload.get("default_model_slug"),
            "restore_at": restore_at,
            "status": "正常" if image_quota_unknown and plan_type.lower() != "free" else ("限流" if quota == 0 else "正常"),
        }

    def _bootstrap_headers(self) -> Dict[str, str]:
        """构造首页预热请求头。"""
        return {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Sec-Ch-Ua": self.session.headers["Sec-Ch-Ua"],
            "Sec-Ch-Ua-Mobile": self.session.headers["Sec-Ch-Ua-Mobile"],
            "Sec-Ch-Ua-Platform": self.session.headers["Sec-Ch-Ua-Platform"],
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
        }

    def _build_requirements(self, data: Dict[str, Any], source_p: str = "") -> ChatRequirements:
        """把 sentinel 响应整理成后续对话需要的 token 集合。"""
        if (data.get("arkose") or {}).get("required"):
            raise RuntimeError("chat requirements requires arkose token, which is not implemented")

        proof_token = ""
        proof_info = data.get("proofofwork") or {}
        if proof_info.get("required"):
            proof_token = build_proof_token(
                proof_info.get("seed", ""),
                proof_info.get("difficulty", ""),
                self.user_agent,
                script_sources=self.pow_script_sources,
                data_build=self.pow_data_build,
            )

        turnstile_token = ""
        turnstile_info = data.get("turnstile") or {}
        if turnstile_info.get("required") and turnstile_info.get("dx"):
            turnstile_token = solve_turnstile_token(turnstile_info["dx"], source_p) or ""

        return ChatRequirements(
            token=data.get("token", ""),
            proof_token=proof_token,
            turnstile_token=turnstile_token,
            so_token=data.get("so_token", ""),
            raw_finalize=data,
        )

    def _conversation_headers(self, path: str, requirements: ChatRequirements) -> Dict[str, str]:
        """根据当前 requirements 构造对话 SSE 请求头。"""
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "OpenAI-Sentinel-Chat-Requirements-Token": requirements.token,
        }
        if requirements.proof_token:
            headers["OpenAI-Sentinel-Proof-Token"] = requirements.proof_token
        if requirements.turnstile_token:
            headers["OpenAI-Sentinel-Turnstile-Token"] = requirements.turnstile_token
        if requirements.so_token:
            headers["OpenAI-Sentinel-SO-Token"] = requirements.so_token
        return self._headers(path, headers)

    def _api_messages_to_conversation_messages(self, messages: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """把标准 chat messages 转成 web conversation 所需的 messages。"""
        conversation_messages = []
        for item in messages:
            role = item.get("role", "user")
            content = item.get("content", "")
            if isinstance(content, str):
                conversation_messages.append({
                    "id": new_uuid(),
                    "author": {"role": role},
                    "content": {"content_type": "text", "parts": [content]},
                })
                continue
            if not isinstance(content, list):
                raise RuntimeError("only string or list message content is supported")
            text_parts: list[str] = []
            image_inputs: list[tuple[bytes, str]] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = str(part.get("type") or "")
                if part_type == "text":
                    text_parts.append(str(part.get("text") or ""))
                elif part_type == "image":
                    data = part.get("data")
                    mime = str(part.get("mime") or "image/png")
                    if isinstance(data, (bytes, bytearray)):
                        image_inputs.append((bytes(data), mime))
            if not image_inputs:
                conversation_messages.append({
                    "id": new_uuid(),
                    "author": {"role": role},
                    "content": {"content_type": "text", "parts": ["".join(text_parts)]},
                })
                continue
            if not self.access_token:
                raise RuntimeError("authenticated upstream account required for image input")
            uploaded: list[Dict[str, Any]] = []
            for idx, (data, mime) in enumerate(image_inputs, start=1):
                ext_part = mime.split("/", 1)[1].split("+")[0] if "/" in mime else "png"
                extension = "jpg" if ext_part == "jpeg" else (ext_part or "png")
                b64 = base64.b64encode(data).decode("ascii")
                uploaded.append(self._upload_image(f"data:{mime};base64,{b64}", f"image_{idx}.{extension}"))
            parts: list[Any] = []
            for ref in uploaded:
                parts.append({
                    "content_type": "image_asset_pointer",
                    "asset_pointer": f"file-service://{ref['file_id']}",
                    "width": ref["width"],
                    "height": ref["height"],
                    "size_bytes": ref["file_size"],
                })
            text = "".join(text_parts)
            if text:
                parts.append(text)
            conversation_messages.append({
                "id": new_uuid(),
                "author": {"role": role},
                "content": {"content_type": "multimodal_text", "parts": parts},
                "metadata": {
                    "attachments": [{
                        "id": ref["file_id"],
                        "mimeType": ref["mime_type"],
                        "name": ref["file_name"],
                        "size": ref["file_size"],
                        "width": ref["width"],
                        "height": ref["height"],
                    } for ref in uploaded],
                },
            })
        return conversation_messages

    def _conversation_payload(self, messages: list[Dict[str, Any]], model: str, timezone: str) -> Dict[str, Any]:
        """把标准 messages 构造成 web 对话请求体。"""
        return {
            "action": "next",
            "messages": self._api_messages_to_conversation_messages(messages),
            "model": model,
            "parent_message_id": new_uuid(),
            "conversation_mode": {"kind": "primary_assistant"},
            "conversation_origin": None,
            "force_paragen": False,
            "force_paragen_model_slug": "",
            "force_rate_limit": False,
            "force_use_sse": True,
            "history_and_training_disabled": True,
            "reset_rate_limits": False,
            "suggestions": [],
            "supported_encodings": [],
            "system_hints": [],
            "timezone": timezone,
            "timezone_offset_min": -480,
            "variant_purpose": "comparison_implicit",
            "websocket_request_id": new_uuid(),
            "client_contextual_info": {
                "is_dark_mode": False,
                "time_since_loaded": 120,
                "page_height": 900,
                "page_width": 1400,
                "pixel_ratio": 2,
                "screen_height": 1440,
                "screen_width": 2560,
            },
        }

    def _image_model_slug(self, model: str, *, mode: str = "native") -> str:
        """把标准图片模型名映射到底层 model slug。"""
        model = str(model or "").strip()
        if not model:
            return "auto"
        if mode == "conversation" and model == "gpt-image-2":
            return "gpt-5-3"
        if model in {"gpt-image-2", CODEX_IMAGE_MODEL}:
            return model
        return "auto"

    def _image_headers(self, path: str, requirements: ChatRequirements, conduit_token: str = "", accept: str = "*/*") -> \
            Dict[str, str]:
        """构造图片链路请求头。"""
        headers = {
            "Content-Type": "application/json",
            "Accept": accept,
            "OpenAI-Sentinel-Chat-Requirements-Token": requirements.token,
        }
        if requirements.proof_token:
            headers["OpenAI-Sentinel-Proof-Token"] = requirements.proof_token
        if conduit_token:
            headers["X-Conduit-Token"] = conduit_token
        if accept == "text/event-stream":
            headers["X-Oai-Turn-Trace-Id"] = new_uuid()
        return self._headers(path, headers)

    def _prepare_image_conversation(
            self,
            prompt: str,
            requirements: ChatRequirements,
            model: str,
            *,
            slug_mode: str = "native",
    ) -> str:
        """为图片生成准备 conduit token。"""
        path = "/backend-api/f/conversation/prepare"
        payload = {
            "action": "next",
            "fork_from_shared_post": False,
            "parent_message_id": new_uuid(),
            "model": self._image_model_slug(model, mode=slug_mode),
            "client_prepare_state": "success",
            "timezone_offset_min": -480,
            "timezone": "Asia/Shanghai",
            "conversation_mode": {"kind": "primary_assistant"},
            "system_hints": ["picture_v2"],
            "partial_query": {
                "id": new_uuid(),
                "author": {"role": "user"},
                "content": {"content_type": "text", "parts": [prompt]},
            },
            "supports_buffering": True,
            "supported_encodings": ["v1"],
            "client_contextual_info": {"app_name": "chatgpt.com"},
        }
        response = self._request_with_local_retry(
            "POST",
            self.base_url + path,
            headers=self._image_headers(path, requirements),
            json=payload,
            timeout=60,
        )
        ensure_ok(response, path)
        return response.json().get("conduit_token", "")

    def _decode_image_base64(self, image: str) -> bytes:
        """把 base64 图片字符串或本地路径解码成二进制。"""
        if (
                image
                and len(image) < 512
                and not image.startswith("data:")
                and "\n" not in image
                and "\r" not in image
        ):
            file_path = Path(os.path.expanduser(image))
            if file_path.exists() and file_path.is_file():
                return file_path.read_bytes()
        payload = image.split(",", 1)[1] if image.startswith("data:") and "," in image else image
        return base64.b64decode(payload)

    def _upload_image(self, image: str, file_name: str = "image.png") -> Dict[str, Any]:
        """上传一张 base64 图片，返回底层文件元数据。"""
        data = self._decode_image_base64(image)
        if (
                image
                and len(image) < 512
                and not image.startswith("data:")
                and "\n" not in image
                and "\r" not in image
        ):
            candidate_path = Path(os.path.expanduser(image))
            if candidate_path.exists() and candidate_path.is_file():
                file_name = candidate_path.name
        image = Image.open(BytesIO(data))
        width, height = image.size
        mime_type = Image.MIME.get(image.format, "image/png")
        path = "/backend-api/files"
        response = self._request_with_local_retry(
            "POST",
            self.base_url + path,
            headers=self._headers(path, {"Content-Type": "application/json", "Accept": "application/json"}),
            json={"file_name": file_name, "file_size": len(data), "use_case": "multimodal", "width": width,
                  "height": height},
            timeout=60,
        )
        ensure_ok(response, path)
        upload_meta = response.json()
        time.sleep(0.5)
        response = self._request_with_local_retry(
            "PUT",
            upload_meta["upload_url"],
            headers={
                "Content-Type": mime_type,
                "x-ms-blob-type": "BlockBlob",
                "x-ms-version": "2020-04-08",
                "Origin": self.base_url,
                "Referer": self.base_url + "/",
                "User-Agent": self.user_agent,
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.8",
            },
            data=data,
            timeout=120,
        )
        ensure_ok(response, "image_upload")
        path = f"/backend-api/files/{upload_meta['file_id']}/uploaded"
        response = self._request_with_local_retry(
            "POST",
            self.base_url + path,
            headers=self._headers(path, {"Content-Type": "application/json", "Accept": "application/json"}),
            data="{}",
            timeout=60,
        )
        ensure_ok(response, path)
        return {
            "file_id": upload_meta["file_id"],
            "file_name": file_name,
            "file_size": len(data),
            "mime_type": mime_type,
            "width": width,
            "height": height,
        }

    def _start_image_generation(
            self,
            prompt: str,
            requirements: ChatRequirements,
            conduit_token: str,
            model: str,
            references: Optional[list[Dict[str, Any]]] = None,
            *,
            slug_mode: str = "native",
    ) -> requests.Response:
        """启动图片生成或编辑的 SSE 请求。"""
        references = references or []
        parts = [{
            "content_type": "image_asset_pointer",
            "asset_pointer": f"file-service://{item['file_id']}",
            "width": item["width"],
            "height": item["height"],
            "size_bytes": item["file_size"],
        } for item in references]
        parts.append(prompt)
        content = {"content_type": "multimodal_text", "parts": parts} if references else {"content_type": "text",
                                                                                          "parts": [prompt]}
        metadata = {
            "developer_mode_connector_ids": [],
            "selected_github_repos": [],
            "selected_all_github_repos": False,
            "system_hints": ["picture_v2"],
            "serialization_metadata": {"custom_symbol_offsets": []},
        }
        if references:
            metadata["attachments"] = [{
                "id": item["file_id"],
                "mimeType": item["mime_type"],
                "name": item["file_name"],
                "size": item["file_size"],
                "width": item["width"],
                "height": item["height"],
            } for item in references]
        payload = {
            "action": "next",
            "messages": [{
                "id": new_uuid(),
                "author": {"role": "user"},
                "create_time": time.time(),
                "content": content,
                "metadata": metadata,
            }],
            "parent_message_id": new_uuid(),
            "model": self._image_model_slug(model, mode=slug_mode),
            "client_prepare_state": "sent",
            "timezone_offset_min": -480,
            "timezone": "Asia/Shanghai",
            "conversation_mode": {"kind": "primary_assistant"},
            "enable_message_followups": True,
            "system_hints": ["picture_v2"],
            "supports_buffering": True,
            "supported_encodings": ["v1"],
            "client_contextual_info": {
                "is_dark_mode": False,
                "time_since_loaded": 1200,
                "page_height": 1072,
                "page_width": 1724,
                "pixel_ratio": 1.2,
                "screen_height": 1440,
                "screen_width": 2560,
                "app_name": "chatgpt.com",
            },
            "paragen_cot_summary_display_override": "allow",
            "force_parallel_switch": "auto",
        }
        path = "/backend-api/f/conversation"
        response = self._request_with_local_retry(
            "POST",
            self.base_url + path,
            headers=self._image_headers(path, requirements, conduit_token, "text/event-stream"),
            json=payload,
            timeout=300,
            stream=True,
        )
        ensure_ok(response, path)
        return response

    def _get_conversation(self, conversation_id: str, timeout: float = 60) -> Dict[str, Any]:
        """获取完整 conversation 详情。"""
        path = f"/backend-api/conversation/{conversation_id}"
        response = self._request_with_local_retry(
            "GET",
            self.base_url + path,
            headers=self._headers(path, {"Accept": "application/json"}),
            timeout=timeout,
        )
        ensure_ok(response, path)
        return response.json()

    @staticmethod
    def _extract_message_text(message: Dict[str, Any]) -> str:
        content = message.get("content") or {}
        parts = content.get("parts") or []
        if not isinstance(parts, list):
            return ""
        text_parts: list[str] = []
        for part in parts:
            if isinstance(part, str):
                text_parts.append(part)
                continue
            if not isinstance(part, dict):
                continue
            for key in ("text", "caption", "alt_text"):
                text = str(part.get(key) or "")
                if text:
                    text_parts.append(text)
                    break
        return "".join(text_parts).strip()

    def _extract_image_progress_text(self, data: Dict[str, Any]) -> str:
        """从完整 conversation 里提取最新 assistant 进度文本。"""
        mapping = data.get("mapping") or {}
        records: list[tuple[float, str]] = []
        for node in mapping.values():
            message = (node or {}).get("message") or {}
            author = message.get("author") or {}
            if author.get("role") != "assistant":
                continue
            text = self._extract_message_text(message)
            if text:
                records.append((float(message.get("create_time") or 0), text))
        if not records:
            return ""
        return sorted(records, key=lambda item: item[0])[-1][1]

    def _extract_image_tool_records(self, data: Dict[str, Any]) -> list[Dict[str, Any]]:
        """从 conversation 明细里提取图片工具输出记录。"""
        mapping = data.get("mapping") or {}
        file_pat = re.compile(r"file-service://([A-Za-z0-9_-]+)")
        sed_pat = re.compile(r"sediment://([A-Za-z0-9_-]+)")
        records = []
        for message_id, node in mapping.items():
            message = (node or {}).get("message") or {}
            author = message.get("author") or {}
            metadata = message.get("metadata") or {}
            content = message.get("content") or {}
            if author.get("role") != "tool":
                continue
            if metadata.get("async_task_type") != "image_gen":
                continue
            if content.get("content_type") != "multimodal_text":
                continue
            file_ids, sediment_ids = [], []
            for part in content.get("parts") or []:
                text = (part.get("asset_pointer") or "") if isinstance(part, dict) else (
                    part if isinstance(part, str) else "")
                for hit in file_pat.findall(text):
                    if hit not in file_ids:
                        file_ids.append(hit)
                for hit in sed_pat.findall(text):
                    if hit not in sediment_ids:
                        sediment_ids.append(hit)
            records.append(
                {"message_id": message_id, "create_time": message.get("create_time") or 0, "file_ids": file_ids,
                 "sediment_ids": sediment_ids})
        return sorted(records, key=lambda item: item["create_time"])

    def _poll_image_results(self, conversation_id: str, timeout_secs: float = 120.0) -> tuple[list[str], list[str]]:
        """轮询 conversation，直到拿到图片文件 id 或超时。"""
        start = time.time()
        attempt = 0
        logger.info({"event": "image_poll_start", "conversation_id": conversation_id, "timeout_secs": timeout_secs})
        while time.time() - start < timeout_secs:
            attempt += 1
            conversation = self._get_conversation(conversation_id)
            file_ids, sediment_ids = [], []
            for record in self._extract_image_tool_records(conversation):
                for file_id in record["file_ids"]:
                    if file_id not in file_ids:
                        file_ids.append(file_id)
                for sediment_id in record["sediment_ids"]:
                    if sediment_id not in sediment_ids:
                        sediment_ids.append(sediment_id)
            logger.debug({"event": "image_poll_check", "conversation_id": conversation_id, "attempt": attempt,
                          "file_ids": file_ids, "sediment_ids": sediment_ids})
            if file_ids:
                logger.info({"event": "image_poll_hit", "conversation_id": conversation_id, "file_ids": file_ids,
                             "sediment_ids": sediment_ids})
                return file_ids, sediment_ids
            if sediment_ids:
                logger.info({"event": "image_poll_hit", "conversation_id": conversation_id, "file_ids": [],
                             "sediment_ids": sediment_ids})
                return [], sediment_ids
            logger.debug({"event": "image_poll_wait", "conversation_id": conversation_id,
                          "elapsed_secs": round(time.time() - start, 1)})
            time.sleep(4)
        logger.info({"event": "image_poll_timeout", "conversation_id": conversation_id, "timeout_secs": timeout_secs})
        return [], []

    def _poll_image_results_stream(
            self,
            conversation_id: str,
            timeout_secs: float = 120.0,
            progress_text: str = "",
    ) -> Iterator[Dict[str, Any]]:
        """轮询 conversation，并在每次等待/进度变化时产出流式保活事件。"""
        start = time.time()
        attempt = 0
        last_progress_text = str(progress_text or "").strip()
        last_error = ""
        logger.info({"event": "image_poll_stream_start", "conversation_id": conversation_id,
                     "timeout_secs": timeout_secs})
        while time.time() - start < timeout_secs:
            attempt += 1
            elapsed_secs = round(time.time() - start, 1)
            yield {
                "type": "image.poll.wait",
                "conversation_id": conversation_id,
                "attempt": attempt,
                "elapsed_secs": elapsed_secs,
                "progress_text": "",
            }
            try:
                conversation = self._get_conversation(conversation_id, timeout=45)
            except Exception as exc:
                last_error = str(exc)
                logger.debug({"event": "image_poll_stream_check_failed", "conversation_id": conversation_id,
                              "attempt": attempt, "error": last_error})
                time.sleep(4)
                continue

            file_ids, sediment_ids = [], []
            for record in self._extract_image_tool_records(conversation):
                for file_id in record["file_ids"]:
                    if file_id not in file_ids:
                        file_ids.append(file_id)
                for sediment_id in record["sediment_ids"]:
                    if sediment_id not in sediment_ids:
                        sediment_ids.append(sediment_id)

            current_progress_text = self._extract_image_progress_text(conversation)
            if current_progress_text and current_progress_text != last_progress_text:
                if last_progress_text and current_progress_text.startswith(last_progress_text):
                    delta = current_progress_text[len(last_progress_text):]
                else:
                    delta = current_progress_text
                last_progress_text = current_progress_text
                if delta:
                    yield {
                        "type": "image.poll.progress",
                        "conversation_id": conversation_id,
                        "attempt": attempt,
                        "elapsed_secs": round(time.time() - start, 1),
                        "progress_text": delta,
                    }

            logger.debug({"event": "image_poll_stream_check", "conversation_id": conversation_id, "attempt": attempt,
                          "file_ids": file_ids, "sediment_ids": sediment_ids})
            if file_ids:
                logger.info({"event": "image_poll_stream_hit", "conversation_id": conversation_id, "file_ids": file_ids,
                             "sediment_ids": sediment_ids})
                yield {"type": "image.poll.result", "file_ids": file_ids, "sediment_ids": sediment_ids}
                return
            if sediment_ids:
                logger.info({"event": "image_poll_stream_hit", "conversation_id": conversation_id, "file_ids": [],
                             "sediment_ids": sediment_ids})
                yield {"type": "image.poll.result", "file_ids": [], "sediment_ids": sediment_ids}
                return
            time.sleep(4)
        logger.info({"event": "image_poll_stream_timeout", "conversation_id": conversation_id,
                     "timeout_secs": timeout_secs, "last_error": last_error})
        yield {"type": "image.poll.timeout", "conversation_id": conversation_id, "progress_text": "",
               "error": last_error}

    def _get_file_download_url(self, file_id: str) -> str:
        """获取文件下载地址。"""
        path = f"/backend-api/files/{file_id}/download"
        response = self._request_with_local_retry(
            "GET",
            self.base_url + path,
            headers=self._headers(path, {"Accept": "application/json"}),
            timeout=60,
        )
        ensure_ok(response, path)
        data = response.json()
        return data.get("download_url") or data.get("url") or ""

    def _get_attachment_download_url(self, conversation_id: str, attachment_id: str) -> str:
        """通过 conversation 附件接口获取下载地址。"""
        path = f"/backend-api/conversation/{conversation_id}/attachment/{attachment_id}/download"
        response = self._request_with_local_retry(
            "GET",
            self.base_url + path,
            headers=self._headers(path, {"Accept": "application/json"}),
            timeout=60,
        )
        ensure_ok(response, path)
        data = response.json()
        return data.get("download_url") or data.get("url") or ""

    def _resolve_image_urls(self, conversation_id: str, file_ids: list[str], sediment_ids: list[str]) -> list[str]:
        """把图片结果 id 解析成可下载 URL。"""
        urls = []
        skip_patterns = {"file_upload"}
        for file_id in file_ids:
            if file_id in skip_patterns:
                logger.debug({
                    "event": "image_file_id_skipped",
                    "source": "file",
                    "conversation_id": conversation_id,
                    "id": file_id,
                })
                continue
            try:
                url = self._get_file_download_url(file_id)
            except Exception as exc:
                logger.debug({
                    "event": "image_download_url_failed",
                    "source": "file",
                    "conversation_id": conversation_id,
                    "id": file_id,
                    "error": repr(exc),
                })
                self.last_image_resolution_errors.append(f"file {file_id} download url failed: {exc!r}")
                continue
            if url:
                urls.append(url)
            else:
                self.last_image_resolution_errors.append(f"file {file_id} download url was empty")
                logger.debug({
                    "event": "image_download_url_empty",
                    "source": "file",
                    "conversation_id": conversation_id,
                    "id": file_id,
                })
        if urls or not conversation_id:
            logger.debug({
                "event": "image_urls_resolved",
                "conversation_id": conversation_id,
                "file_ids": file_ids,
                "sediment_ids": sediment_ids,
                "urls": urls,
            })
            return urls
        for sediment_id in sediment_ids:
            try:
                url = self._get_attachment_download_url(conversation_id, sediment_id)
            except Exception as exc:
                logger.debug({
                    "event": "image_download_url_failed",
                    "source": "sediment",
                    "conversation_id": conversation_id,
                    "id": sediment_id,
                    "error": repr(exc),
                })
                self.last_image_resolution_errors.append(f"sediment {sediment_id} download url failed: {exc!r}")
                continue
            if url:
                urls.append(url)
            else:
                self.last_image_resolution_errors.append(f"sediment {sediment_id} download url was empty")
                logger.debug({
                    "event": "image_download_url_empty",
                    "source": "sediment",
                    "conversation_id": conversation_id,
                    "id": sediment_id,
                })
        logger.debug({
            "event": "image_urls_resolved",
            "conversation_id": conversation_id,
            "file_ids": file_ids,
            "sediment_ids": sediment_ids,
            "urls": urls,
        })
        return urls

    def resolve_conversation_image_urls(
            self,
            conversation_id: str,
            file_ids: list[str],
            sediment_ids: list[str],
            poll: bool = True,
    ) -> list[str]:
        self.last_image_resolution_errors = []
        file_ids = [item for item in file_ids if item != "file_upload"]
        sediment_ids = list(sediment_ids)
        if poll and conversation_id and not file_ids and not sediment_ids:
            logger.info({"event": "image_resolve_poll_needed", "conversation_id": conversation_id})
            polled_file_ids, polled_sediment_ids = self._poll_image_results(conversation_id,
                                                                            config.image_poll_timeout_secs)
            file_ids.extend(item for item in polled_file_ids if item and item not in file_ids)
            sediment_ids.extend(item for item in polled_sediment_ids if item and item not in sediment_ids)
            if not file_ids and not sediment_ids:
                self.last_image_resolution_errors.append(
                    f"image polling timed out after {config.image_poll_timeout_secs}s without file ids or sediment ids"
                )
        return self._resolve_image_urls(conversation_id, file_ids, sediment_ids)

    def iter_conversation_image_resolution(
            self,
            conversation_id: str,
            file_ids: list[str],
            sediment_ids: list[str],
            poll: bool = True,
            progress_text: str = "",
    ) -> Iterator[Dict[str, Any]]:
        """流式解析图片 URL，轮询期间持续产出进度/保活事件。"""
        self.last_image_resolution_errors = []
        file_ids = [item for item in file_ids if item != "file_upload"]
        sediment_ids = list(sediment_ids)
        if poll and conversation_id and not file_ids and not sediment_ids:
            logger.info({"event": "image_resolve_poll_stream_needed", "conversation_id": conversation_id})
            for event in self._poll_image_results_stream(
                    conversation_id,
                    config.image_poll_timeout_secs,
                    progress_text,
            ):
                if event.get("type") == "image.poll.result":
                    polled_file_ids = [str(item) for item in event.get("file_ids") or []]
                    polled_sediment_ids = [str(item) for item in event.get("sediment_ids") or []]
                    file_ids.extend(item for item in polled_file_ids if item and item not in file_ids)
                    sediment_ids.extend(item for item in polled_sediment_ids if item and item not in sediment_ids)
                    break
                if event.get("type") == "image.poll.timeout":
                    self.last_image_resolution_errors.append(
                        f"image polling timed out after {config.image_poll_timeout_secs}s without file ids or sediment ids"
                    )
                yield event
        yield {
            "type": "image.resolve.done",
            "urls": self._resolve_image_urls(conversation_id, file_ids, sediment_ids),
            "file_ids": file_ids,
            "sediment_ids": sediment_ids,
            "errors": list(self.last_image_resolution_errors),
        }

    def download_image_bytes(self, urls: list[str]) -> list[bytes]:
        images = []
        for url in urls:
            response = self._request_with_local_retry("GET", url, timeout=120)
            ensure_ok(response, "image_download")
            images.append(response.content)
        return images

    def stream_conversation(
            self,
            messages: Optional[list[Dict[str, Any]]] = None,
            model: str = "auto",
            prompt: str = "",
            images: Optional[list[str]] = None,
            system_hints: Optional[list[str]] = None,
    ) -> Iterator[str]:
        system_hints = system_hints or []
        if "picture_v2" in system_hints:
            yield from self._stream_picture_conversation(prompt, model, images or [])
            return

        normalized = messages or [{"role": "user", "content": prompt}]
        self._bootstrap()
        requirements = self._get_chat_requirements()
        path, timezone = self._chat_target()
        payload = self._conversation_payload(normalized, model, timezone)
        response = self._request_with_local_retry(
            "POST",
            self.base_url + path,
            headers=self._conversation_headers(path, requirements),
            json=payload,
            timeout=300,
            stream=True,
        )
        ensure_ok(response, path)
        try:
            yield from iter_sse_payloads(response)
        finally:
            response.close()

    def _stream_picture_conversation(
            self,
            prompt: str,
            model: str,
            images: list[str],
            *,
            slug_mode: str = "native",
    ) -> Iterator[str]:
        if not self.access_token:
            raise RuntimeError("access_token is required for image endpoints")
        references = [self._upload_image(image, f"image_{idx}.png") for idx, image in enumerate(images, start=1)]
        self._bootstrap()
        requirements = self._get_chat_requirements()
        conduit_token = self._prepare_image_conversation(prompt, requirements, model, slug_mode=slug_mode)
        response = self._start_image_generation(prompt, requirements, conduit_token, model, references, slug_mode=slug_mode)
        try:
            yield from iter_sse_payloads(response)
        finally:
            response.close()

    def stream_image_conversation(
            self,
            prompt: str,
            model: str,
            images: list[str],
            *,
            slug_mode: str = "native",
    ) -> Iterator[str]:
        yield from self._stream_picture_conversation(prompt, model, images, slug_mode=slug_mode)

    def _bootstrap(self) -> None:
        """预热首页，并提取 PoW 相关脚本引用。"""
        response = self._request_with_local_retry(
            "GET",
            self.base_url + "/",
            headers=self._bootstrap_headers(),
            timeout=30,
        )
        ensure_ok(response, "bootstrap")
        self.pow_script_sources, self.pow_data_build = parse_pow_resources(response.text)
        if not self.pow_script_sources:
            self.pow_script_sources = [DEFAULT_POW_SCRIPT]

    def _get_chat_requirements(self) -> ChatRequirements:
        """获取当前模式对话所需的 sentinel token。"""
        path = "/backend-api/sentinel/chat-requirements" if self.access_token else "/backend-anon/sentinel/chat-requirements"
        context = "auth_chat_requirements" if self.access_token else "noauth_chat_requirements"
        body = {"p": build_legacy_requirements_token(self.user_agent, self.pow_script_sources, self.pow_data_build)}
        response = self._request_with_local_retry(
            "POST",
            self.base_url + path,
            headers=self._headers(path, {"Content-Type": "application/json"}),
            json=body,
            timeout=30,
        )
        ensure_ok(response, context)
        requirements = self._build_requirements(response.json(), "" if self.access_token else body["p"])
        if not requirements.token:
            message = "missing auth chat requirements token" if self.access_token else "missing chat requirements token"
            raise RuntimeError(f"{message}: {requirements.raw_finalize}")
        return requirements

    def _chat_target(self) -> tuple[str, str]:
        if self.access_token:
            return "/backend-api/conversation", "Asia/Shanghai"
        return "/backend-anon/conversation", "America/Los_Angeles"

    def list_models(self) -> Dict[str, Any]:
        """返回当前模式下可用模型，格式对齐 OpenAI `/v1/models`。"""
        self._bootstrap()
        path = "/backend-api/models?history_and_training_disabled=false" if self.access_token else (
            "/backend-anon/models?iim=false&is_gizmo=false"
        )
        route = "/backend-api/models" if self.access_token else "/backend-anon/models"
        context = "auth_models" if self.access_token else "anon_models"
        response = self._request_with_local_retry(
            "GET",
            self.base_url + path,
            headers=self._headers(route),
            timeout=30,
        )
        ensure_ok(response, context)
        data = []
        seen = set()
        for item in response.json().get("models", []):
            if not isinstance(item, dict):
                continue
            slug = str(item.get("slug", "")).strip()
            if not slug or slug in seen:
                continue
            seen.add(slug)
            data.append({
                "id": slug,
                "object": "model",
                "created": int(item.get("created") or 0),
                "owned_by": str(item.get("owned_by") or "chatgpt"),
                "permission": [],
                "root": slug,
                "parent": None,
            })
        data.sort(key=lambda item: item["id"])
        return {"object": "list", "data": data}
