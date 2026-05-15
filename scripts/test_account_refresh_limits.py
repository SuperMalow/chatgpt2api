from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import accounts as accounts_api
from services.config import config


class AccountRefreshLimitApiTests(unittest.TestCase):
    def make_client(self) -> TestClient:
        app = FastAPI()
        app.include_router(accounts_api.create_router())
        return TestClient(app)

    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {config.auth_key}"}

    def test_refresh_rejects_empty_token_list_instead_of_refreshing_all_accounts(self) -> None:
        client = self.make_client()

        with mock.patch.object(accounts_api, "account_service") as service:
            response = client.post(
                "/api/accounts/refresh?include_items=false",
                json={"access_tokens": []},
                headers=self.auth_headers(),
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["error"], "access_tokens is required")
        service.list_tokens.assert_not_called()
        service.refresh_accounts.assert_not_called()

    def test_refresh_rejects_more_than_50_tokens(self) -> None:
        client = self.make_client()
        tokens = [f"token-{index}" for index in range(51)]

        with mock.patch.object(accounts_api, "account_service") as service:
            response = client.post(
                "/api/accounts/refresh?include_items=false",
                json={"access_tokens": tokens},
                headers=self.auth_headers(),
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["error"], "一次最多只能刷新 50 个账号")
        service.refresh_accounts.assert_not_called()

    def test_refresh_accepts_50_tokens(self) -> None:
        client = self.make_client()
        tokens = [f"token-{index}" for index in range(50)]

        with mock.patch.object(accounts_api, "account_service") as service:
            service.refresh_accounts.return_value = {"refreshed": 50, "errors": []}
            response = client.post(
                "/api/accounts/refresh?include_items=false",
                json={"access_tokens": tokens},
                headers=self.auth_headers(),
            )

        self.assertEqual(response.status_code, 200)
        service.refresh_accounts.assert_called_once_with(
            tokens,
            include_items=False,
            quota_only_if_possible=True,
        )


if __name__ == "__main__":
    unittest.main()
