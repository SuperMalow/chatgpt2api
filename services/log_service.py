from __future__ import annotations

import hashlib
import json
import itertools
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse

from services.config import DATA_DIR
from utils.helper import anthropic_sse_stream, sse_json_stream

LOG_TYPE_CALL = "call"
LOG_TYPE_ACCOUNT = "account"
DEFAULT_LOG_PAGE_SIZE = 20
MAX_LOG_PAGE_SIZE = 100


class LogService:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _legacy_id(raw_line: str, line_number: int) -> str:
        payload = f"{line_number}:{raw_line}".encode("utf-8", errors="ignore")
        return hashlib.sha1(payload).hexdigest()[:24]

    def _parse_line(self, raw_line: str, line_number: int) -> dict[str, Any] | None:
        try:
            item = json.loads(raw_line)
        except Exception:
            return None
        if not isinstance(item, dict):
            return None
        parsed = dict(item)
        parsed["id"] = str(parsed.get("id") or self._legacy_id(raw_line, line_number))
        return parsed

    @staticmethod
    def _serialize_item(item: dict[str, Any]) -> str:
        return json.dumps(item, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _matches_filters(item: dict[str, Any], *, type: str = "", start_date: str = "", end_date: str = "") -> bool:
        t = str(item.get("time") or "")
        day = t[:10]
        if type and item.get("type") != type:
            return False
        if start_date and day < start_date:
            return False
        if end_date and day > end_date:
            return False
        return True

    def add(self, type: str, summary: str = "", detail: dict[str, Any] | None = None, **data: Any) -> None:
        item = {
            "id": uuid4().hex,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": type,
            "summary": summary,
            "detail": detail or data,
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(self._serialize_item(item) + "\n")

    @staticmethod
    def _normalize_page(page: int, page_size: int, max_page_size: int | None = MAX_LOG_PAGE_SIZE) -> tuple[int, int]:
        page_size = max(1, int(page_size or DEFAULT_LOG_PAGE_SIZE))
        if max_page_size is not None:
            page_size = min(max_page_size, page_size)
        page = max(1, int(page or 1))
        return page, page_size

    def _iter_items(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file):
                item = self._parse_line(raw_line.rstrip("\r\n"), line_number)
                if item is not None:
                    yield item

    def _count_matches(self, *, type: str = "", start_date: str = "", end_date: str = "") -> int:
        total = 0
        for item in self._iter_items():
            if self._matches_filters(item, type=type, start_date=start_date, end_date=end_date):
                total += 1
        return total

    def _collect_page_items(
        self,
        *,
        type: str = "",
        start_date: str = "",
        end_date: str = "",
        start: int = 0,
        page_size: int = DEFAULT_LOG_PAGE_SIZE,
    ) -> list[dict[str, Any]]:
        end = start + page_size
        recent_items: deque[dict[str, Any]] = deque(maxlen=end)
        for item in self._iter_items():
            if not self._matches_filters(item, type=type, start_date=start_date, end_date=end_date):
                continue
            recent_items.append(item)
        return list(reversed(recent_items))[start:end]

    def _list_page(
        self,
        *,
        type: str = "",
        start_date: str = "",
        end_date: str = "",
        page: int = 1,
        page_size: int = DEFAULT_LOG_PAGE_SIZE,
        max_page_size: int | None = MAX_LOG_PAGE_SIZE,
    ) -> dict[str, Any]:
        page, page_size = self._normalize_page(page, page_size, max_page_size)
        if not self.path.exists():
            return {"items": [], "total": 0, "page": 1, "page_size": page_size, "pages": 1}

        total = self._count_matches(type=type, start_date=start_date, end_date=end_date)
        pages = max(1, (total + page_size - 1) // page_size)
        safe_page = min(page, pages)
        items = self._collect_page_items(
            type=type,
            start_date=start_date,
            end_date=end_date,
            start=(safe_page - 1) * page_size,
            page_size=page_size,
        )
        return {
            "items": items,
            "total": total,
            "page": safe_page,
            "page_size": page_size,
            "pages": pages,
        }

    def list_page(
        self,
        type: str = "",
        start_date: str = "",
        end_date: str = "",
        page: int = 1,
        page_size: int = DEFAULT_LOG_PAGE_SIZE,
    ) -> dict[str, Any]:
        return self._list_page(
            type=type,
            start_date=start_date,
            end_date=end_date,
            page=page,
            page_size=page_size,
            max_page_size=MAX_LOG_PAGE_SIZE,
        )

    def list(self, type: str = "", start_date: str = "", end_date: str = "", limit: int = 200) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return self._list_page(
            type=type,
            start_date=start_date,
            end_date=end_date,
            page=1,
            page_size=limit,
            max_page_size=None,
        )["items"]

    def delete(self, ids: list[str]) -> dict[str, int]:
        target_ids = {str(item or "").strip() for item in ids if str(item or "").strip()}
        if not self.path.exists() or not target_ids:
            return {"removed": 0}
        lines = self.path.read_text(encoding="utf-8").splitlines()
        kept_lines: list[str] = []
        removed = 0
        for line_number, raw_line in enumerate(lines):
            item = self._parse_line(raw_line, line_number)
            if item is None:
                kept_lines.append(raw_line)
                continue
            if str(item.get("id") or "") in target_ids:
                removed += 1
                continue
            kept_lines.append(self._serialize_item(item))
        content = "\n".join(kept_lines)
        if content:
            content += "\n"
        self.path.write_text(content, encoding="utf-8")
        return {"removed": removed}


log_service = LogService(DATA_DIR / "logs.jsonl")


def _collect_urls(value: object) -> list[str]:
    urls: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "url" and isinstance(item, str):
                urls.append(item)
            elif key == "urls" and isinstance(item, list):
                urls.extend(str(url) for url in item if isinstance(url, str))
            else:
                urls.extend(_collect_urls(item))
    elif isinstance(value, list):
        for item in value:
            urls.extend(_collect_urls(item))
    return urls


def _request_excerpt(text: object, limit: int = 1000) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _image_error_response(exc: Exception) -> JSONResponse:
    message = str(exc)
    if "no available image quota" in message.lower():
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "message": "no available image quota",
                    "type": "insufficient_quota",
                    "param": None,
                    "code": "insufficient_quota",
                }
            },
        )
    if hasattr(exc, "to_openai_error") and hasattr(exc, "status_code"):
        return JSONResponse(status_code=int(exc.status_code), content=exc.to_openai_error())
    return JSONResponse(
        status_code=502,
        content={
            "error": {
                "message": message,
                "type": "server_error",
                "param": None,
                "code": "upstream_error",
            }
        },
    )


def _http_exception_message(exc: HTTPException) -> str:
    detail = exc.detail
    if isinstance(detail, dict) and isinstance(detail.get("error"), str):
        return detail["error"]
    return str(detail)


def _http_exception_detail(exc: HTTPException) -> dict[str, Any] | None:
    detail = exc.detail
    return detail if isinstance(detail, dict) else None


def _next_item(items):
    try:
        return True, next(items)
    except StopIteration:
        return False, None


@dataclass
class LoggedCall:
    identity: dict[str, object]
    endpoint: str
    model: str
    summary: str
    started: float = field(default_factory=time.time)
    request_text: str = ""

    async def run(self, handler, *args, sse: str = "openai"):
        from services.protocol.conversation import ImageGenerationError

        try:
            result = await run_in_threadpool(handler, *args)
        except ImageGenerationError as exc:
            self.log("调用失败", status="failed", error=str(exc), error_detail=exc.detail)
            return _image_error_response(exc)
        except HTTPException as exc:
            self.log("调用失败", status="failed", error=_http_exception_message(exc), error_detail=_http_exception_detail(exc))
            raise
        except Exception as exc:
            self.log("调用失败", status="failed", error=str(exc))
            raise HTTPException(status_code=502, detail={"error": str(exc)}) from exc

        if isinstance(result, dict):
            self.log("调用完成", result)
            return result

        sender = anthropic_sse_stream if sse == "anthropic" else sse_json_stream
        try:
            has_first, first = await run_in_threadpool(_next_item, result)
        except ImageGenerationError as exc:
            self.log("调用失败", status="failed", error=str(exc), error_detail=exc.detail)
            return _image_error_response(exc)
        except HTTPException as exc:
            self.log("调用失败", status="failed", error=_http_exception_message(exc), error_detail=_http_exception_detail(exc))
            raise
        except Exception as exc:
            self.log("调用失败", status="failed", error=str(exc))
            raise HTTPException(status_code=502, detail={"error": str(exc)}) from exc
        if not has_first:
            self.log("流式调用结束")
            return StreamingResponse(sender(()), media_type="text/event-stream")
        return StreamingResponse(sender(self.stream(itertools.chain([first], result))), media_type="text/event-stream")

    def stream(self, items):
        urls: list[str] = []
        failed = False
        try:
            for item in items:
                urls.extend(_collect_urls(item))
                yield item
        except Exception as exc:
            failed = True
            error_detail = getattr(exc, "detail", None)
            self.log(
                "流式调用失败",
                status="failed",
                error=str(exc),
                error_detail=error_detail if isinstance(error_detail, dict) else None,
                urls=urls,
            )
            raise
        finally:
            if not failed:
                self.log("流式调用结束", urls=urls)

    def log(self, suffix: str, result: object = None, status: str = "success", error: str = "",
            urls: list[str] | None = None, error_detail: dict[str, Any] | None = None) -> None:
        detail = {
            "key_id": self.identity.get("id"),
            "key_name": self.identity.get("name"),
            "role": self.identity.get("role"),
            "endpoint": self.endpoint,
            "model": self.model,
            "started_at": datetime.fromtimestamp(self.started).strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_ms": int((time.time() - self.started) * 1000),
            "status": status,
        }
        request_excerpt = _request_excerpt(self.request_text)
        if request_excerpt:
            detail["request_text"] = request_excerpt
        if error:
            detail["error"] = error
        if error_detail:
            detail["error_detail"] = error_detail
        collected_urls = [*(urls or []), *_collect_urls(result)]
        if collected_urls:
            detail["urls"] = list(dict.fromkeys(collected_urls))
        log_service.add(LOG_TYPE_CALL, f"{self.summary}{suffix}", detail)
