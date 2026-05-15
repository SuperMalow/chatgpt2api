from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.account_service import AccountService
from services.log_service import LOG_TYPE_ACCOUNT, LogService
from services.storage.json_storage import JSONStorageBackend


class AccountOperationLogTests(unittest.TestCase):
    def make_service(self) -> tuple[AccountService, LogService]:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        base_path = Path(tmp_dir.name)
        service = AccountService(JSONStorageBackend(base_path / "accounts.json"))
        logs = LogService(base_path / "logs.jsonl")
        return service, logs

    def test_update_account_records_changed_fields_in_account_log(self) -> None:
        service, logs = self.make_service()

        with mock.patch("services.account_service.log_service", logs):
            service.add_accounts(["token-secret-123"])
            updated = service.update_account(
                "token-secret-123",
                {
                    "type": "pro",
                    "status": "禁用",
                    "quota": 7,
                },
            )

        self.assertIsNotNone(updated)
        entries = [item for item in logs.list(type=LOG_TYPE_ACCOUNT) if item["summary"] == "更新账号"]
        self.assertEqual(len(entries), 1)
        detail = entries[0]["detail"]

        self.assertEqual(detail["action"], "update_account")
        self.assertIn("token", detail)
        self.assertNotIn("token-secret-123", json.dumps(detail, ensure_ascii=False))
        self.assertEqual(
            detail["changes"],
            {
                "type": {"before": "free", "after": "pro"},
                "status": {"before": "正常", "after": "禁用"},
                "quota": {"before": 0, "after": 7},
            },
        )

    def test_refresh_accounts_records_refresh_result_in_account_log(self) -> None:
        service, logs = self.make_service()

        def fake_remote_info(
            token: str,
            timeout: float | None = None,
            quota_only_if_possible: bool = False,
        ) -> dict:
            return {
                "email": "refresh@example.com",
                "user_id": "user-refresh",
                "type": "free",
                "quota": 4,
                "image_quota_unknown": False,
                "limits_progress": [],
                "default_model_slug": "gpt-5",
                "restore_at": None,
                "status": "正常",
            }

        with (
            mock.patch("services.account_service.log_service", logs),
            mock.patch.object(service, "_fetch_remote_info_result", side_effect=fake_remote_info),
        ):
            service.add_accounts(["token-refresh-123"])
            result = service.refresh_accounts(["token-refresh-123"], include_items=False)

        self.assertEqual(result["refreshed"], 1)
        entries = [item for item in logs.list(type=LOG_TYPE_ACCOUNT) if item["summary"].startswith("批量刷新账号")]
        self.assertEqual(len(entries), 1)
        detail = entries[0]["detail"]

        self.assertEqual(detail["action"], "refresh_accounts")
        self.assertEqual(detail["requested"], 1)
        self.assertEqual(detail["refreshed"], 1)
        self.assertEqual(detail["errors"], 0)
        self.assertEqual(len(detail["requested_tokens"]), 1)
        self.assertEqual(len(detail["refreshed_tokens"]), 1)
        self.assertEqual(detail["failed_items"], [])
        self.assertNotIn("token-refresh-123", json.dumps(detail, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
