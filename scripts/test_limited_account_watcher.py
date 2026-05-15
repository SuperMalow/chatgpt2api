from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api import support


class LimitedAccountWatcherTests(unittest.TestCase):
    def test_watcher_uses_latest_refresh_interval_each_cycle(self) -> None:
        class FakeConfig:
            refresh_account_interval_minute = 5

        class FakeStopEvent:
            def __init__(self) -> None:
                self.waits: list[int] = []

            def is_set(self) -> bool:
                return len(self.waits) >= 2

            def wait(self, timeout: int) -> bool:
                self.waits.append(timeout)
                FakeConfig.refresh_account_interval_minute = 2
                return False

        class SynchronousThread:
            def __init__(self, target, name: str, daemon: bool) -> None:
                self.target = target
                self.name = name
                self.daemon = daemon

            def start(self) -> None:
                self.target()

        stop_event = FakeStopEvent()

        with (
            mock.patch.object(support, "config", FakeConfig),
            mock.patch.object(support, "Thread", SynchronousThread),
            mock.patch.object(support.account_service, "list_limited_tokens", return_value=[]),
        ):
            support.start_limited_account_watcher(stop_event)

        self.assertEqual(stop_event.waits, [300, 120])

    def test_watcher_refreshes_limited_accounts_in_batches_and_rotates(self) -> None:
        class FakeConfig:
            refresh_account_interval_minute = 1
            limited_account_refresh_batch_size = 2

        class FakeStopEvent:
            def __init__(self) -> None:
                self.waits: list[int] = []

            def is_set(self) -> bool:
                return len(self.waits) >= 2

            def wait(self, timeout: int) -> bool:
                self.waits.append(timeout)
                return False

        class SynchronousThread:
            def __init__(self, target, name: str, daemon: bool) -> None:
                self.target = target
                self.name = name
                self.daemon = daemon

            def start(self) -> None:
                self.target()

        stop_event = FakeStopEvent()
        refresh_calls: list[list[str]] = []
        limited_tokens = [f"token-{index}" for index in range(5)]

        with (
            mock.patch.object(support, "config", FakeConfig),
            mock.patch.object(support, "Thread", SynchronousThread),
            mock.patch.object(support.account_service, "list_limited_tokens", return_value=limited_tokens),
            mock.patch.object(support.account_service, "refresh_accounts", side_effect=lambda tokens: refresh_calls.append(tokens)),
        ):
            support.start_limited_account_watcher(stop_event)

        self.assertEqual(refresh_calls, [["token-0", "token-1"], ["token-2", "token-3"]])


if __name__ == "__main__":
    unittest.main()
