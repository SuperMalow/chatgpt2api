from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.support import require_identity, resolve_image_base_url
from services.content_filter import check_request
from services.image_task_events import image_task_event_service
from services.image_task_service import image_task_service
from services.log_service import LoggedCall


class ImageGenerationTaskRequest(BaseModel):
    client_task_id: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    model: str = "gpt-image-2"
    size: str | None = None
    conversation_id: str = ""
    turn_id: str = ""


def _parse_task_ids(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _sse_message(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def filter_or_log(call: LoggedCall, text: str) -> None:
    try:
        await run_in_threadpool(check_request, text)
    except HTTPException as exc:
        error = exc.detail.get("error") if isinstance(exc.detail, dict) else None
        call.log(
            "调用失败",
            status="failed",
            error=str(error or exc.detail),
            error_detail=exc.detail if isinstance(exc.detail, dict) else None,
        )
        raise


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/image-tasks")
    async def list_image_tasks(
        ids: str = Query(default=""),
        authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        return await run_in_threadpool(image_task_service.list_tasks, identity, _parse_task_ids(ids))

    @router.get("/api/image-tasks/events")
    async def stream_image_task_events(
        conversation_id: str = Query(..., min_length=1),
        authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        subscriber_id, queue = image_task_event_service.subscribe(conversation_id)

        async def event_stream():
            try:
                snapshot = await run_in_threadpool(image_task_service.list_tasks_for_conversation, identity, conversation_id)
                for task in snapshot.get("items") or []:
                    status = str(task.get("status") or "queued").strip() or "queued"
                    yield _sse_message(
                        f"task.{status}",
                        {
                            "event": f"task.{status}",
                            "conversation_id": task.get("conversation_id") or conversation_id,
                            "turn_id": task.get("turn_id") or "",
                            "task_id": task.get("id") or "",
                            "status": status,
                            "task": task,
                        },
                    )
                while True:
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=20.0)
                    except TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    event_name = str(item.get("event") or "task.updated")
                    yield _sse_message(event_name, item)
            finally:
                image_task_event_service.unsubscribe(conversation_id, subscriber_id)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.post("/api/image-tasks/generations")
    async def create_generation_task(
        body: ImageGenerationTaskRequest,
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        await filter_or_log(LoggedCall(identity, "/api/image-tasks/generations", body.model, "文生图任务", request_text=body.prompt), body.prompt)
        try:
            return await run_in_threadpool(
                image_task_service.submit_generation,
                identity,
                client_task_id=body.client_task_id,
                prompt=body.prompt,
                model=body.model,
                size=body.size,
                base_url=resolve_image_base_url(request),
                conversation_id=body.conversation_id,
                turn_id=body.turn_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc

    @router.post("/api/image-tasks/edits")
    async def create_edit_task(
        request: Request,
        authorization: str | None = Header(default=None),
        image: list[UploadFile] | None = File(default=None),
        image_list: list[UploadFile] | None = File(default=None, alias="image[]"),
        client_task_id: str = Form(...),
        prompt: str = Form(...),
        model: str = Form(default="gpt-image-2"),
        size: str | None = Form(default=None),
        conversation_id: str = Form(default=""),
        turn_id: str = Form(default=""),
    ):
        identity = require_identity(authorization)
        await filter_or_log(LoggedCall(identity, "/api/image-tasks/edits", model, "图生图任务", request_text=prompt), prompt)
        uploads = [*(image or []), *(image_list or [])]
        if not uploads:
            raise HTTPException(status_code=400, detail={"error": "image file is required"})
        images: list[tuple[bytes, str, str]] = []
        for upload in uploads:
            image_data = await upload.read()
            if not image_data:
                raise HTTPException(status_code=400, detail={"error": "image file is empty"})
            images.append((image_data, upload.filename or "image.png", upload.content_type or "image/png"))
        try:
            return await run_in_threadpool(
                image_task_service.submit_edit,
                identity,
                client_task_id=client_task_id,
                prompt=prompt,
                model=model,
                size=size,
                base_url=resolve_image_base_url(request),
                images=images,
                conversation_id=conversation_id,
                turn_id=turn_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc

    return router
