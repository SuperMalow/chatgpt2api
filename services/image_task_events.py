from __future__ import annotations

import asyncio
from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass
class _Subscriber:
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue[dict[str, Any]]


class ImageTaskEventService:
    def __init__(self) -> None:
        self._lock = Lock()
        self._subscribers: dict[str, dict[int, _Subscriber]] = {}
        self._next_id = 1

    def subscribe(self, conversation_id: str) -> tuple[int, asyncio.Queue[dict[str, Any]]]:
        normalized = str(conversation_id or "").strip()
        if not normalized:
            raise ValueError("conversation_id is required")
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        with self._lock:
            subscriber_id = self._next_id
            self._next_id += 1
            bucket = self._subscribers.setdefault(normalized, {})
            bucket[subscriber_id] = _Subscriber(loop=loop, queue=queue)
        return subscriber_id, queue

    def unsubscribe(self, conversation_id: str, subscriber_id: int) -> None:
        normalized = str(conversation_id or "").strip()
        if not normalized or not subscriber_id:
            return
        with self._lock:
            bucket = self._subscribers.get(normalized)
            if not bucket:
                return
            bucket.pop(subscriber_id, None)
            if not bucket:
                self._subscribers.pop(normalized, None)

    def publish(self, conversation_id: str, event: dict[str, Any]) -> None:
        normalized = str(conversation_id or "").strip()
        if not normalized:
            return
        with self._lock:
            subscribers = list((self._subscribers.get(normalized) or {}).values())
        for subscriber in subscribers:
            subscriber.loop.call_soon_threadsafe(subscriber.queue.put_nowait, dict(event))


image_task_event_service = ImageTaskEventService()
