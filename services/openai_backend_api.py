import base64
import json
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import tiktoken
from curl_cffi import requests
from PIL import Image

from services.account_service import account_service
from services.config import config
from services.proxy_service import proxy_settings
from utils.helper import build_chat_image_markdown_content, ensure_ok, new_uuid, parse_sse_lines
from utils.log import logger
from utils.pow import build_legacy_requirements_token, build_proof_token, parse_pow_resources
from utils.turnstile import solve_turnstile_token


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
CODEX_RESPONSE_MODEL = "gpt-5.4"


class OpenAIBackendAPI:
    """ChatGPT Web 后端封装。

    说明：
    - 传入 `access_token` 时，聊天和模型列表都会走已登录链路
      例如 `/backend-api/sentinel/chat-requirements`、`/backend-api/conversation`
    - 不传 `access_token` 时，会走未登录链路
      例如 `/backend-anon/sentinel/chat-requirements`、`/backend-anon/conversation`
    - 对外统一调用 `list_models()`、`chat_completions(...)`
    - 图片相关接口 `images_generations(...)`、`images_edits(...)` 目前只支持登录态
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

    def _build_requirements_token(self) -> str:
        """生成 sentinel 接口需要的旧版 requirements token。"""
        return build_legacy_requirements_token(
            self.user_agent,
            script_sources=self.pow_script_sources,
            data_build=self.pow_data_build,
        )

    def _get_token_info(self) -> Dict[str, Any]:
        """从 access token 的 JWT payload 中提取后续请求可能会用到的信息。"""
        if not self.access_token:
            return {}
        parts = self.access_token.split(".")
        if len(parts) < 2:
            return {}
        payload = parts[1]
        padding = "=" * (-len(payload) % 4)
        try:
            decoded = base64.urlsafe_b64decode(payload + padding).decode("utf-8")
            data = json.loads(decoded)
        except Exception:
            return {}
        auth_data = data.get("https://api.openai.com/auth") or {}
        return auth_data

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
            content = item.get("content", "")
            if not isinstance(content, str):
                raise RuntimeError("only string message content is supported")
            conversation_messages.append({
                "id": new_uuid(),
                "author": {"role": item.get("role", "user")},
                "content": {"content_type": "text", "parts": [content]},
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

    def _normalize_models(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """把 web models 响应整理成 OpenAI `/v1/models` 风格结构。"""
        data = []
        seen = set()
        for item in payload.get("models", []):
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

    def _build_image_prompt(self, prompt: str, size: str | None) -> str:
        """把标准图片 prompt 和宽高比转成底层图片生成 prompt。"""
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

    def _image_model_slug(self, model: str) -> str:
        """把标准图片模型名映射到底层 model slug。"""
        model = str(model or "").strip()
        if not model:
            return "auto"
        if model in {"gpt-image-1", "gpt-image-2", "gpt-image"}:
            return "gpt-5-3"
        if model == CODEX_IMAGE_MODEL:
            return model
        return "auto"

    def _is_codex_image_model(self, model: str) -> bool:
        return model == CODEX_IMAGE_MODEL

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

    def _prepare_image_conversation(self, prompt: str, requirements: ChatRequirements, model: str) -> str:
        """为图片生成准备 conduit token。"""
        path = "/backend-api/f/conversation/prepare"
        payload = {
            "action": "next",
            "fork_from_shared_post": False,
            "parent_message_id": new_uuid(),
            "model": self._image_model_slug(model),
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
        response = self.session.post(
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

    def _image_to_data_url(self, image: str) -> str:
        """把本地图片路径或 base64 图片统一转成 data URL。"""
        data = self._decode_image_base64(image)
        try:
            opened = Image.open(BytesIO(data))
            mime_type = Image.MIME.get(opened.format, "image/png")
        except Exception:
            mime_type = "image/png"
        return f"data:{mime_type};base64,{base64.b64encode(data).decode()}"

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
        response = self.session.post(
            self.base_url + path,
            headers=self._headers(path, {"Content-Type": "application/json", "Accept": "application/json"}),
            json={"file_name": file_name, "file_size": len(data), "use_case": "multimodal", "width": width,
                  "height": height},
            timeout=60,
        )
        ensure_ok(response, path)
        upload_meta = response.json()
        time.sleep(0.5)
        response = self.session.put(
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
        response = self.session.post(
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

    def _start_image_generation(self, prompt: str, requirements: ChatRequirements, conduit_token: str, model: str,
                                references: Optional[list[Dict[str, Any]]] = None) -> requests.Response:
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
            "model": self._image_model_slug(model),
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
        response = self.session.post(
            self.base_url + path,
            headers=self._image_headers(path, requirements, conduit_token, "text/event-stream"),
            json=payload,
            timeout=300,
            stream=True,
        )
        ensure_ok(response, path)
        return response

    def _parse_image_sse(self, response: requests.Response) -> Dict[str, Any]:
        """从图片 SSE 里提取 conversation_id、file_ids、sediment_ids。"""
        conversation_id = ""
        file_ids: list[str] = []
        sediment_ids: list[str] = []
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="ignore")
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            if not conversation_id:
                match = re.search(r'"conversation_id"\s*:\s*"([^"]+)"', payload)
                if match:
                    conversation_id = match.group(1)
            for file_id in re.findall(r"(file[-_][A-Za-z0-9]+)", payload):
                if file_id not in file_ids:
                    file_ids.append(file_id)
            for sediment_id in re.findall(r"sediment://([A-Za-z0-9_-]+)", payload):
                if sediment_id not in sediment_ids:
                    sediment_ids.append(sediment_id)
        return {"conversation_id": conversation_id, "file_ids": file_ids, "sediment_ids": sediment_ids}

    def _get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """获取完整 conversation 详情。"""
        path = f"/backend-api/conversation/{conversation_id}"
        response = self.session.get(self.base_url + path, headers=self._headers(path, {"Accept": "application/json"}),
                                    timeout=60)
        ensure_ok(response, path)
        return response.json()

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
        last_sediment_ids: list[str] = []
        while time.time() - start < timeout_secs:
            conversation = self._get_conversation(conversation_id)
            file_ids, sediment_ids = [], []
            for record in self._extract_image_tool_records(conversation):
                for file_id in record["file_ids"]:
                    if file_id not in file_ids:
                        file_ids.append(file_id)
                for sediment_id in record["sediment_ids"]:
                    if sediment_id not in sediment_ids:
                        sediment_ids.append(sediment_id)
            if file_ids:
                return file_ids, sediment_ids
            if sediment_ids:
                return [], sediment_ids
            time.sleep(4)
        return [], last_sediment_ids

    def _get_file_download_url(self, file_id: str) -> str:
        """获取文件下载地址。"""
        path = f"/backend-api/files/{file_id}/download"
        response = self.session.get(self.base_url + path, headers=self._headers(path, {"Accept": "application/json"}),
                                    timeout=60)
        ensure_ok(response, path)
        data = response.json()
        return data.get("download_url") or data.get("url") or ""

    def _get_attachment_download_url(self, conversation_id: str, attachment_id: str) -> str:
        """通过 conversation 附件接口获取下载地址。"""
        path = f"/backend-api/conversation/{conversation_id}/attachment/{attachment_id}/download"
        response = self.session.get(self.base_url + path, headers=self._headers(path, {"Accept": "application/json"}),
                                    timeout=60)
        ensure_ok(response, path)
        data = response.json()
        return data.get("download_url") or data.get("url") or ""

    def _save_image_bytes(self, image_data: bytes) -> str:
        file_name = f"{int(time.time())}_{new_uuid().replace('-', '')}.png"
        relative_dir = Path(time.strftime("%Y"), time.strftime("%m"), time.strftime("%d"))
        file_path = config.images_dir / relative_dir / file_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(image_data)
        return f"{config.base_url}/images/{relative_dir.as_posix()}/{file_name}"

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
                continue
            if url:
                urls.append(url)
            else:
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
                continue
            if url:
                urls.append(url)
            else:
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

    def _image_response(self, urls: list[str], response_format: str) -> Dict[str, Any]:
        """把图片结果整理成 OpenAI `/v1/images/*` 风格结构。"""
        if response_format not in {"url", "b64_json"}:
            raise ValueError("response_format must be 'url' or 'b64_json'")
        data = []
        for url in urls:
            response = self.session.get(url, timeout=120)
            ensure_ok(response, "image_download")
            if response_format == "b64_json":
                data.append({"b64_json": base64.b64encode(response.content).decode()})
            else:
                data.append({"url": self._save_image_bytes(response.content)})
        return {"created": int(time.time()), "data": data}

    def _run_image_task(self, prompt: str, model: str, size: str | None, images: Optional[list[str]] = None,
                        response_format: str = "url") -> Dict[str, Any]:
        """执行图片生成或图片编辑主流程。"""
        if not self.access_token:
            raise RuntimeError("access_token is required for image endpoints")
        logger.debug({
            "event": "image_task_start",
            "prompt": prompt,
            "model": model,
            "size": size,
            "image_count": len(images or []),
        })
        references = [self._upload_image(image, f"image_{idx}.png") for idx, image in enumerate(images or [], start=1)]
        logger.debug({"event": "image_references_uploaded", "references": references})
        self._bootstrap()
        requirements = self._get_auth_chat_requirements()
        logger.debug({
            "event": "image_requirements_ready",
            "token_present": bool(requirements.token),
            "proof_token_present": bool(requirements.proof_token),
            "turnstile_token_present": bool(requirements.turnstile_token),
            "so_token_present": bool(requirements.so_token),
            "raw_finalize": requirements.raw_finalize,
        })
        final_prompt = self._build_image_prompt(prompt, size)
        logger.debug({"event": "image_final_prompt", "final_prompt": final_prompt})
        conduit_token = self._prepare_image_conversation(final_prompt, requirements, model)
        logger.debug({"event": "image_conduit_ready", "conduit_token_present": bool(conduit_token)})
        sse = self._start_image_generation(final_prompt, requirements, conduit_token, model, references)
        sse_result = self._parse_image_sse(sse)
        logger.debug({"event": "image_sse_result", "sse_result": sse_result})
        conversation_id = sse_result["conversation_id"]
        file_ids = list(sse_result["file_ids"])
        sediment_ids = list(sse_result["sediment_ids"])
        invalid_file_id_patterns = {"file_upload"}
        file_ids = [fid for fid in file_ids if fid not in invalid_file_id_patterns]
        if conversation_id and not file_ids and not sediment_ids:
            polled_file_ids, polled_sediment_ids = self._poll_image_results(conversation_id)
            file_ids.extend([item for item in polled_file_ids if item not in file_ids])
            sediment_ids.extend([item for item in polled_sediment_ids if item not in sediment_ids])
            logger.debug({
                "event": "image_polled_result",
                "conversation_id": conversation_id,
                "file_ids": polled_file_ids,
                "sediment_ids": polled_sediment_ids,
            })
            try:
                conversation = self._get_conversation(conversation_id)
                logger.debug({
                    "event": "image_conversation_snapshot",
                    "conversation_id": conversation_id,
                    "conversation": conversation,
                })
            except Exception as exc:
                logger.debug({
                    "event": "image_conversation_snapshot_failed",
                    "conversation_id": conversation_id,
                    "error": repr(exc),
                })
        logger.debug({
            "event": "image_resolved_ids",
            "conversation_id": conversation_id,
            "file_ids": file_ids,
            "sediment_ids": sediment_ids,
        })
        urls = self._resolve_image_urls(conversation_id, file_ids, sediment_ids)
        logger.debug({"event": "image_final_urls", "conversation_id": conversation_id, "urls": urls})
        if not urls:
            raise RuntimeError(
                "no downloadable image result found; "
                f"conversation_id={conversation_id}, file_ids={file_ids}, sediment_ids={sediment_ids}"
            )
        return self._image_response(urls, response_format)

    def _build_codex_response_input(self, prompt: str, images: Optional[list[str]] = None) -> list[Dict[str, Any]]:
        if not images:
            return [{"role": "user", "content": prompt}]
        content = [{"type": "input_text", "text": prompt}]
        for image in images:
            content.append({"type": "input_image", "image_url": self._image_to_data_url(image)})
        return [{"role": "user", "content": content}]

    def _collect_codex_events(self, prompt: str, model: str = CODEX_IMAGE_MODEL,
                              images: Optional[list[str]] = None) -> list[Dict[str, Any]]:
        events = list(self.responses(
            input=self._build_codex_response_input(prompt, images),
            model=model,
            stream=True,
        ))
        logger.debug({"event": "codex_responses_events_collected", "event_count": len(events)})
        return events

    def _codex_image_response(self, events: list[Dict[str, Any]], response_format: str) -> Dict[str, Any]:
        if response_format not in {"url", "b64_json"}:
            raise ValueError("response_format must be 'url' or 'b64_json'")
        image_item = {}
        response_payload = {}
        for event in events:
            if not isinstance(event, dict):
                continue
            if event.get("type") == "response.output_item.done":
                item = event.get("item") or {}
                if item.get("type") == "image_generation_call" and item.get("result"):
                    image_item = item
            elif event.get("type") == "response.completed":
                response_payload = event.get("response") or {}
        image_b64 = image_item.get("result", "")
        if not image_b64:
            raise RuntimeError("codex responses did not return image base64 result")
        data = []
        if response_format == "b64_json":
            data.append({"b64_json": image_b64})
        else:
            data.append({"url": self._save_image_bytes(base64.b64decode(image_b64))})
        return {
            "created": response_payload.get("created_at") or int(time.time()),
            "data": data,
            "model": response_payload.get("model") or CODEX_RESPONSE_MODEL,
            "usage": response_payload.get("usage"),
            "tool_usage": response_payload.get("tool_usage"),
            "response_id": response_payload.get("id"),
            "status": response_payload.get("status"),
            "revised_prompt": image_item.get("revised_prompt"),
            "size": image_item.get("size"),
            "output_format": image_item.get("output_format"),
            "response": response_payload,
        }

    def _run_codex_image_task(self, prompt: str, response_format: str = "url",
                              images: Optional[list[str]] = None) -> Dict[str, Any]:
        logger.debug({"event": "codex_image_task_start", "prompt": prompt, "image_count": len(images or [])})
        events = self._collect_codex_events(prompt=prompt, model=CODEX_IMAGE_MODEL, images=images)
        result = self._codex_image_response(events, response_format)
        logger.debug({
            "event": "codex_image_task_done",
            "response_id": result.get("response_id"),
            "status": result.get("status"),
            "usage": result.get("usage"),
            "tool_usage": result.get("tool_usage"),
        })
        return result

    def _extract_text_from_events(self, events: list[Dict[str, Any]]) -> str:
        """从 SSE 事件中提取最终 assistant 文本。"""
        for event in reversed(events):
            message = event.get("message") or {}
            if (message.get("author") or {}).get("role") != "assistant":
                continue
            text = self._text_from_message(message)
            if text.strip():
                return text
        return ""

    @staticmethod
    def _strip_history_prefix(text: str, history_text: str) -> str:
        history_text = str(history_text or "")
        text = str(text or "")
        if history_text and text.startswith(history_text):
            return text[len(history_text):]
        return text

    def _normalize_assistant_text(self, text: str, history_text: str) -> str:
        return self._strip_history_prefix(text, history_text)

    def _text_from_message(self, message: Dict[str, Any]) -> str:
        """从单条 message 结构中提取文本。"""
        content = message.get("content") or {}
        parts = content.get("parts") or []
        if not isinstance(parts, list):
            return ""
        texts = [part for part in parts if isinstance(part, str)]
        if texts:
            return "".join(texts)
        return ""

    @staticmethod
    def _append_unique(values: list[str], candidates: list[str]) -> None:
        for candidate in candidates:
            if candidate and candidate not in values:
                values.append(candidate)

    @staticmethod
    def _extract_image_stream_ids(payload: str) -> tuple[list[str], list[str]]:
        file_ids = re.findall(r"(file[-_][A-Za-z0-9]+)", payload)
        sediment_ids = re.findall(r"sediment://([A-Za-z0-9_-]+)", payload)
        return file_ids, sediment_ids

    @staticmethod
    def _extract_image_stream_conversation_id(payload: str) -> str:
        match = re.search(r'"conversation_id"\s*:\s*"([^"]+)"', payload)
        return match.group(1) if match else ""

    def _next_image_stream_text(self, event: Dict[str, Any], current_text: str) -> str:
        for candidate in (event, event.get("v")):
            if not isinstance(candidate, dict):
                continue
            message = candidate.get("message")
            if not isinstance(message, dict):
                continue
            role = str((message.get("author") or {}).get("role") or "").strip().lower()
            if role == "user":
                continue
            text = self._text_from_message(message)
            if text:
                return text
        return self._apply_text_patch(event, current_text)

    def stream_image_chat_completions(
        self,
        prompt: str,
        model: str = "gpt-image-2",
        size: str | None = None,
        images: Optional[list[str]] = None,
    ) -> Iterator[Dict[str, Any]]:
        if not self.access_token:
            raise RuntimeError("access_token is required for image endpoints")

        completion_id = f"chatcmpl-{new_uuid()}"
        created = int(time.time())
        current_text = ""
        sent_role = False
        conversation_id = ""
        file_ids: list[str] = []
        sediment_ids: list[str] = []

        references = [self._upload_image(image, f"image_{idx}.png") for idx, image in enumerate(images or [], start=1)]
        self._bootstrap()
        requirements = self._get_auth_chat_requirements()
        final_prompt = self._build_image_prompt(prompt, size)
        conduit_token = self._prepare_image_conversation(final_prompt, requirements, model)
        sse = self._start_image_generation(final_prompt, requirements, conduit_token, model, references)
        try:
            for raw_line in sse.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="ignore") if isinstance(raw_line, bytes) else str(raw_line)
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload:
                    continue
                if payload == "[DONE]":
                    break

                if not conversation_id:
                    conversation_id = self._extract_image_stream_conversation_id(payload)
                new_file_ids, new_sediment_ids = self._extract_image_stream_ids(payload)
                self._append_unique(file_ids, new_file_ids)
                self._append_unique(sediment_ids, new_sediment_ids)

                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue

                conversation_id = str(event.get("conversation_id") or conversation_id)
                value = event.get("v")
                if isinstance(value, dict):
                    conversation_id = str(value.get("conversation_id") or conversation_id)

                next_text = self._next_image_stream_text(event, current_text)
                if next_text == current_text:
                    yield {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": None,
                        }],
                        "upstream_event": event,
                    }
                    continue

                delta = next_text[len(current_text):] if next_text.startswith(current_text) else next_text
                current_text = next_text
                if not sent_role:
                    sent_role = True
                    yield {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": delta},
                            "finish_reason": None,
                        }],
                        "upstream_event": event,
                    }
                    continue

                yield {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": delta},
                        "finish_reason": None,
                    }],
                    "upstream_event": event,
                }
        finally:
            sse.close()

        invalid_file_id_patterns = {"file_upload"}
        file_ids = [fid for fid in file_ids if fid not in invalid_file_id_patterns]
        if conversation_id and not file_ids and not sediment_ids:
            polled_file_ids, polled_sediment_ids = self._poll_image_results(conversation_id)
            self._append_unique(file_ids, polled_file_ids)
            self._append_unique(sediment_ids, polled_sediment_ids)

        urls = self._resolve_image_urls(conversation_id, file_ids, sediment_ids)
        if not urls:
            raise RuntimeError(
                "no downloadable image result found; "
                f"conversation_id={conversation_id}, file_ids={file_ids}, sediment_ids={sediment_ids}"
            )

        image_content = build_chat_image_markdown_content(self._image_response(urls, "b64_json"))
        if not sent_role:
            sent_role = True
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": image_content},
                    "finish_reason": None,
                }],
            }
        else:
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": image_content},
                    "finish_reason": None,
                }],
            }
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }

    def _apply_text_patch(self, event: Dict[str, Any], current_text: str, history_text: str = "") -> str:
        """从 patch 事件里恢复最新文本。"""
        operations = event.get("v")
        if not isinstance(operations, list):
            return current_text
        text = current_text
        for item in operations:
            if not isinstance(item, dict):
                continue
            if item.get("p") != "/message/content/parts/0":
                continue
            if item.get("o") == "append":
                text += str(item.get("v", ""))
            if item.get("o") == "replace":
                text = self._normalize_assistant_text(str(item.get("v", "")), history_text)
        return text

    def _next_assistant_text(self, event: Dict[str, Any], current_text: str, history_text: str = "") -> str:
        """从 SSE 事件中推导当前 assistant 全量文本。"""
        message = event.get("message")
        if isinstance(message, dict) and (message.get("author") or {}).get("role") == "assistant":
            text = self._text_from_message(message)
            if text:
                return self._normalize_assistant_text(text, history_text)

        value = event.get("v")
        if isinstance(value, dict):
            message = value.get("message")
            if isinstance(message, dict) and (message.get("author") or {}).get("role") == "assistant":
                text = self._text_from_message(message)
                if text:
                    return self._normalize_assistant_text(text, history_text)

        return self._apply_text_patch(event, current_text, history_text)

    def _encoding_for_model(self, model: str):
        """按模型选择 tokenizer，失败时回退到通用编码。"""
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            try:
                return tiktoken.get_encoding("o200k_base")
            except KeyError:
                return tiktoken.get_encoding("cl100k_base")

    def _count_message_tokens(self, messages: list[Dict[str, Any]], model: str) -> int:
        """估算标准 chat messages 的 token 数。"""
        encoding = self._encoding_for_model(model)
        tokens_per_message = 3
        total = 0
        for message in messages:
            total += tokens_per_message
            for key, value in message.items():
                if not isinstance(value, str):
                    continue
                total += len(encoding.encode(value))
                if key == "name":
                    total += 1
        return total + 3

    def _count_text_tokens(self, text: str, model: str) -> int:
        """估算单段文本的 token 数。"""
        return len(self._encoding_for_model(model).encode(text))

    def _extract_message_text(self, content: Any) -> str:
        """从 OpenAI/Anthropic 风格的 content 字段里提取纯文本。"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if not isinstance(item, dict):
                    raise RuntimeError("only string or text content blocks are supported")
                block_type = item.get("type")
                if block_type in {"text", "input_text", "output_text"}:
                    parts.append(str(item.get("text", "")))
                    continue
                raise RuntimeError(f"unsupported content block type: {block_type}")
            return "".join(parts)
        if content is None:
            return ""
        raise RuntimeError("only string or text content blocks are supported")

    def _normalize_messages(self, messages: list[Dict[str, Any]], system: Any = None) -> list[Dict[str, Any]]:
        """把 OpenAI/Anthropic 风格的消息统一整理成标准 chat messages。"""
        normalized = []
        if system is not None:
            system_text = self._extract_message_text(system)
            if system_text:
                normalized.append({"role": "system", "content": system_text})

        for item in messages:
            normalized.append({
                "role": item.get("role", "user"),
                "content": self._extract_message_text(item.get("content", "")),
            })
        return normalized

    def _assistant_history_text(self, messages: list[Dict[str, Any]]) -> str:
        """获取输入历史里所有 assistant 文本拼接结果。"""
        parts = []
        for message in messages:
            if message.get("role") != "assistant":
                continue
            content = message.get("content", "")
            if isinstance(content, str) and content:
                parts.append(content)
        return "".join(parts)

    def _assistant_history_messages(self, messages: list[Dict[str, Any]]) -> list[str]:
        parts = []
        for message in messages:
            if message.get("role") != "assistant":
                continue
            content = message.get("content", "")
            if isinstance(content, str) and content:
                parts.append(content)
        return parts

    def _event_assistant_text(self, event: Dict[str, Any], history_text: str = "") -> str:
        message = event.get("message")
        if isinstance(message, dict) and (message.get("author") or {}).get("role") == "assistant":
            return self._normalize_assistant_text(self._text_from_message(message), history_text)

        value = event.get("v")
        if isinstance(value, dict):
            message = value.get("message")
            if isinstance(message, dict) and (message.get("author") or {}).get("role") == "assistant":
                return self._normalize_assistant_text(self._text_from_message(message), history_text)

        return ""

    def _last_event(self, events: list[Dict[str, Any]]) -> Dict[str, Any]:
        """返回最后一个非终止事件，方便排查问题。"""
        for event in reversed(events):
            if not event.get("done"):
                return event
        return {}

    def _stream_events(self, path: str, requirements: ChatRequirements, payload: Dict[str, Any]) -> Iterator[
        Dict[str, Any]]:
        """向 conversation 接口发起请求，并逐条产出 SSE 事件。"""
        response = self.session.post(
            self.base_url + path,
            headers=self._conversation_headers(path, requirements),
            json=payload,
            timeout=300,
            stream=True,
        )
        ensure_ok(response, path)
        yield from parse_sse_lines(response)

    def _bootstrap(self) -> None:
        """预热首页，并提取 PoW 相关脚本引用。"""
        response = self.session.get(
            self.base_url + "/",
            headers=self._bootstrap_headers(),
            timeout=30,
        )
        ensure_ok(response, "bootstrap")
        self.pow_script_sources, self.pow_data_build = parse_pow_resources(response.text)
        if not self.pow_script_sources:
            self.pow_script_sources = [DEFAULT_POW_SCRIPT]

    def _get_chat_requirements(self, authenticated: bool) -> ChatRequirements:
        """获取当前模式对话所需的 sentinel token。"""
        if authenticated and not self.access_token:
            raise RuntimeError("access_token is required for auth chat")
        path = "/backend-api/sentinel/chat-requirements" if authenticated else "/backend-anon/sentinel/chat-requirements"
        context = "auth_chat_requirements" if authenticated else "noauth_chat_requirements"
        body = {"p": self._build_requirements_token()}
        response = self.session.post(
            self.base_url + path,
            headers=self._headers(path, {"Content-Type": "application/json"}),
            json=body,
            timeout=30,
        )
        ensure_ok(response, context)
        requirements = self._build_requirements(response.json(), "" if authenticated else body["p"])
        if not requirements.token:
            message = "missing auth chat requirements token" if authenticated else "missing chat requirements token"
            raise RuntimeError(f"{message}: {requirements.raw_finalize}")
        return requirements

    def _get_auth_chat_requirements(self) -> ChatRequirements:
        return self._get_chat_requirements(authenticated=True)

    def _get_anon_chat_requirements(self) -> ChatRequirements:
        return self._get_chat_requirements(authenticated=False)

    def _get_models_raw(self, authenticated: bool) -> Dict[str, Any]:
        """获取当前模式模型列表原始响应。"""
        if authenticated and not self.access_token:
            raise RuntimeError("access_token is required for auth models")
        self._bootstrap()
        path = "/backend-api/models?history_and_training_disabled=false" if authenticated else (
            "/backend-anon/models?iim=false&is_gizmo=false"
        )
        route = "/backend-api/models" if authenticated else "/backend-anon/models"
        context = "auth_models" if authenticated else "anon_models"
        response = self.session.get(
            self.base_url + path,
            headers=self._headers(route),
            timeout=30,
        )
        ensure_ok(response, context)
        return response.json()

    def _chat_target(self) -> tuple[str, str]:
        if self.access_token:
            return "/backend-api/conversation", "Asia/Shanghai"
        return "/backend-anon/conversation", "America/Los_Angeles"

    def _complete_chat(self, messages: list[Dict[str, Any]], model: str) -> Dict[str, Any]:
        self._bootstrap()
        requirements = self._get_chat_requirements(authenticated=bool(self.access_token))
        path, timezone = self._chat_target()
        events = list(self._stream_events(path, requirements, self._conversation_payload(messages, model, timezone)))
        history_assistant_text = self._assistant_history_text(messages)
        return {
            "requirements": requirements,
            "prepare": {},
            "events": events,
            "last_event": self._last_event(events),
            "text": self._strip_history_prefix(self._extract_text_from_events(events), history_assistant_text),
        }

    def list_models(self) -> Dict[str, Any]:
        """返回当前模式下可用模型，格式对齐 OpenAI `/v1/models`。"""
        return self._normalize_models(self._get_models_raw(authenticated=bool(self.access_token)))

    def images_generations(self, prompt: str, model: str = "gpt-image-2", size: str | None = None,
                           response_format: str = "url") -> Dict[str, Any]:
        """返回 OpenAI `/v1/images/generations` 风格结果。"""
        if self._is_codex_image_model(model):
            return self._run_codex_image_task(prompt, response_format=response_format)
        return self._run_image_task(prompt, model, size, response_format=response_format)

    def images_edits(self, image: str | list[str], prompt: str, model: str = "gpt-image-2", size: str | None = None,
                     response_format: str = "url") -> Dict[str, Any]:
        """返回 OpenAI `/v1/images/edits` 风格结果。"""
        images = [image] if isinstance(image, str) else image
        if not images:
            raise ValueError("image is required for image edits")
        if self._is_codex_image_model(model):
            return self._run_codex_image_task(prompt, response_format=response_format, images=images)
        return self._run_image_task(prompt, model, size, images=images, response_format=response_format)

    def _chat_completion_response(self, model: str, messages: list[Dict[str, Any]], text: str) -> Dict[str, Any]:
        """把对话结果整理成 OpenAI `/v1/chat/completions` 风格结构。"""
        prompt_tokens = self._count_message_tokens(messages, model)
        completion_tokens = self._count_text_tokens(text, model)
        return {
            "id": f"chatcmpl-{new_uuid()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def _stream_chat_completions(self, messages: list[Dict[str, Any]], model: str = "auto") -> Iterator[Dict[str, Any]]:
        """返回 OpenAI `/v1/chat/completions` 风格的流式 chunk。"""
        completion_id = f"chatcmpl-{new_uuid()}"
        created = int(time.time())
        history_assistant_text = self._assistant_history_text(messages)
        history_assistant_messages = self._assistant_history_messages(messages)
        history_assistant_index = 0
        current_text = ""
        sent_role = False
        self._bootstrap()
        requirements = self._get_chat_requirements(authenticated=bool(self.access_token))
        path, timezone = self._chat_target()
        payload = self._conversation_payload(messages, model, timezone)

        for event in self._stream_events(path, requirements, payload):
            if event.get("done"):
                break
            event_assistant_text = self._event_assistant_text(event, history_assistant_text)
            if (
                history_assistant_index < len(history_assistant_messages)
                and event_assistant_text
                and event_assistant_text == history_assistant_messages[history_assistant_index]
            ):
                history_assistant_index += 1
                current_text = ""
                continue
            next_text = self._next_assistant_text(event, current_text, history_assistant_text)
            if next_text == current_text:
                continue
            delta = next_text[len(current_text):] if next_text.startswith(current_text) else next_text
            current_text = next_text
            if not sent_role:
                sent_role = True
                yield {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": delta},
                        "finish_reason": None,
                    }],
                }
                continue
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": delta},
                    "finish_reason": None,
                }],
            }

        if not sent_role:
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }],
            }
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }

    def _anthropic_message_response(self, model: str, messages: list[Dict[str, Any]], text: str) -> Dict[str, Any]:
        """把对话结果整理成 Anthropic `/v1/messages` 风格结构。"""
        prompt_tokens = self._count_message_tokens(messages, model)
        completion_tokens = self._count_text_tokens(text, model)
        return {
            "id": f"msg_{new_uuid()}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{
                "type": "text",
                "text": text,
            }],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
            },
        }

    def _stream_anthropic_messages(self, messages: list[Dict[str, Any]], model: str = "auto") -> Iterator[
        Dict[str, Any]]:
        """返回 Anthropic `/v1/messages` 风格的流式事件。"""
        message_id = f"msg_{new_uuid()}"
        created = int(time.time())
        current_text = ""

        yield {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": self._count_message_tokens(messages, model),
                    "output_tokens": 0,
                },
            },
        }
        yield {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "text",
                "text": "",
            },
        }

        for chunk in self._stream_chat_completions(messages, model):
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            text_delta = delta.get("content", "")
            if text_delta:
                current_text += text_delta
                yield {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "text_delta",
                        "text": text_delta,
                    },
                }

            if choice.get("finish_reason"):
                yield {
                    "type": "content_block_stop",
                    "index": 0,
                }
                yield {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                    },
                    "usage": {
                        "output_tokens": self._count_text_tokens(current_text, model),
                    },
                }
                break

        yield {
            "type": "message_stop",
            "created": created,
        }

    def _iter_response_events(self, response: requests.Response) -> Iterator[Dict[str, Any]]:
        """按 Responses 接口事件格式直通输出，不额外包装标准事件。"""
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="ignore")
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            yield json.loads(payload)

    def responses(
            self,
            input: str | list[Dict[str, Any]],
            model: str = CODEX_RESPONSE_MODEL,
            tools: Optional[list[Dict[str, Any]]] = None,
            instructions: str = "you are a helpful assistant",
            tool_choice: str = "auto",
            stream: bool = False,
            store: bool = False,
    ) -> Dict[str, Any] | Iterator[Dict[str, Any]]:
        """返回 `/v1/responses` 风格结果，底层走 `/backend-api/codex/responses`。"""
        if not self.access_token:
            raise RuntimeError("access_token is required for responses")
        path = "/backend-api/codex/responses"
        token_info = self._get_token_info()
        input_items = [{"role": "user", "content": input}] if isinstance(input, str) else input
        effective_model = CODEX_RESPONSE_MODEL if self._is_codex_image_model(model) else model
        effective_tools = tools
        if self._is_codex_image_model(model) and not effective_tools:
            effective_tools = [{"type": "image_generation", "output_format": "png"}]
        payload = {
            "model": effective_model,
            "input": input_items,
            "tools": effective_tools or [],
            "instructions": instructions,
            "tool_choice": tool_choice,
            "stream": stream,
            "store": store,
        }
        logger.debug({
            "event": "responses_start",
            "model": model,
            "effective_model": effective_model,
            "stream": stream,
            "input_count": len(input_items),
            "tool_count": len(payload["tools"]),
            "token_info": token_info,
        })
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "user-agent": (
                "codex-tui/0.122.0 (Manjaro 26.1.0-pre; x86_64) "
                "vscode/3.0.12 (codex-tui; 0.122.0)"
            ),
            "version": "0.122.0",
            "originator": "codex_cli_rs",
            "session_id": "test-session",
            "accept": "text/event-stream" if stream else "application/json",
            "Content-Type": "application/json",
        }
        if token_info.get("chatgpt_account_id"):
            headers["chatgpt-account-id"] = token_info["chatgpt_account_id"]
        response = self.session.post(
            self.base_url + path,
            headers=headers,
            json=payload,
            timeout=300,
            stream=stream,
        )
        if response.status_code >= 400:
            logger.debug({
                "event": "responses_failed",
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.text,
            })
        ensure_ok(response, path)
        if stream:
            return self._iter_response_events(response)
        return response.json()

    def chat_completions(
            self,
            messages: list[Dict[str, Any]],
            model: str = "auto",
            stream: bool = False,
    ) -> Dict[str, Any] | Iterator[Dict[str, Any]]:
        """返回 OpenAI `/v1/chat/completions` 风格结果，支持 stream 模式。"""
        normalized_messages = self._normalize_messages(messages)
        if stream:
            return self._stream_chat_completions(normalized_messages, model)
        result = self._complete_chat(normalized_messages, model)
        return self._chat_completion_response(model, normalized_messages, result["text"])

    def messages(self, messages: list[Dict[str, Any]], model: str = "auto", stream: bool = False,
                 system: Any = None) -> Dict[str, Any] | Iterator[Dict[str, Any]]:
        """返回 Anthropic `/v1/messages` 风格结果，支持 stream 模式。"""
        normalized_messages = self._normalize_messages(messages, system)
        if stream:
            return self._stream_anthropic_messages(normalized_messages, model)
        result = self._complete_chat(normalized_messages, model)
        return self._anthropic_message_response(model, normalized_messages, result["text"])
