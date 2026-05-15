from __future__ import annotations

import json
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.log_service import LOG_TYPE_ACCOUNT, LOG_TYPE_CALL, LogService


def make_service() -> LogService:
    root = Path("/private/tmp") / f"chatgpt2api-log-pagination-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    return LogService(root / "logs.jsonl")


def write_logs(service: LogService, items: list[dict[str, object]]) -> None:
    service.path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in items),
        encoding="utf-8",
    )


class LogPaginationTests(unittest.TestCase):
    def test_list_page_returns_requested_page_with_total(self) -> None:
        service = make_service()
        write_logs(
            service,
            [
                {"id": f"log-{index}", "time": f"2026-05-16 12:00:0{index}", "type": LOG_TYPE_CALL, "summary": str(index), "detail": {}}
                for index in range(5)
            ],
        )

        result = service.list_page(page=2, page_size=2)

        self.assertEqual([item["id"] for item in result["items"]], ["log-2", "log-1"])
        self.assertEqual(result["total"], 5)
        self.assertEqual(result["page"], 2)
        self.assertEqual(result["page_size"], 2)
        self.assertEqual(result["pages"], 3)

    def test_list_page_filters_before_pagination(self) -> None:
        service = make_service()
        write_logs(
            service,
            [
                {"id": "call-old", "time": "2026-05-14 12:00:00", "type": LOG_TYPE_CALL, "summary": "", "detail": {}},
                {"id": "account-old", "time": "2026-05-15 12:00:00", "type": LOG_TYPE_ACCOUNT, "summary": "", "detail": {}},
                {"id": "call-new", "time": "2026-05-16 12:00:00", "type": LOG_TYPE_CALL, "summary": "", "detail": {}},
                {"id": "account-new", "time": "2026-05-16 13:00:00", "type": LOG_TYPE_ACCOUNT, "summary": "", "detail": {}},
            ],
        )

        result = service.list_page(type=LOG_TYPE_ACCOUNT, start_date="2026-05-16", page=1, page_size=1)

        self.assertEqual([item["id"] for item in result["items"]], ["account-new"])
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["pages"], 1)

    def test_legacy_list_keeps_limit_behavior(self) -> None:
        service = make_service()
        write_logs(
            service,
            [
                {"id": f"log-{index}", "time": f"2026-05-16 12:00:0{index}", "type": LOG_TYPE_CALL, "summary": str(index), "detail": {}}
                for index in range(3)
            ],
        )

        items = service.list(limit=2)

        self.assertEqual([item["id"] for item in items], ["log-2", "log-1"])

    def test_legacy_list_limit_is_not_capped_by_api_page_size(self) -> None:
        service = make_service()
        write_logs(
            service,
            [
                {"id": f"log-{index}", "time": f"2026-05-16 12:{index:02d}:00", "type": LOG_TYPE_CALL, "summary": str(index), "detail": {}}
                for index in range(120)
            ],
        )

        self.assertEqual(len(service.list(limit=120)), 120)

    def test_list_page_does_not_read_entire_file_into_memory(self) -> None:
        service = make_service()
        write_logs(
            service,
            [
                {"id": f"log-{index}", "time": f"2026-05-16 12:00:0{index}", "type": LOG_TYPE_CALL, "summary": str(index), "detail": {}}
                for index in range(3)
            ],
        )

        original_read_text = Path.read_text

        def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
            if path == service.path:
                raise AssertionError("list_page should stream log lines instead of read_text().splitlines()")
            return original_read_text(path, *args, **kwargs)

        with mock.patch.object(Path, "read_text", guarded_read_text):
            result = service.list_page(page=1, page_size=2)

        self.assertEqual([item["id"] for item in result["items"]], ["log-2", "log-1"])


if __name__ == "__main__":
    unittest.main()
