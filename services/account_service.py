from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from threading import Condition, Lock
from typing import Any
from datetime import datetime

from services.config import config
from services.log_service import (
    LOG_TYPE_ACCOUNT,
    log_service,
)
from services.storage.base import StorageBackend
from utils.helper import anonymize_token


MAX_ACCOUNT_PAGE_SIZE = 50


def _positive_int_env(name: str, default: int, minimum: int = 1, maximum: int | None = None) -> int:
    try:
        value = int(os.getenv(name, ""))
    except (TypeError, ValueError):
        value = default
    value = max(minimum, value)
    return min(value, maximum) if maximum is not None else value


def _positive_float_env(name: str, default: float, minimum: float = 1.0) -> float:
    try:
        value = float(os.getenv(name, ""))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


ACCOUNT_REFRESH_MAX_WORKERS = _positive_int_env("ACCOUNT_REFRESH_MAX_WORKERS", 16, 1, 64)
ACCOUNT_REFRESH_REQUEST_TIMEOUT = _positive_float_env("ACCOUNT_REFRESH_REQUEST_TIMEOUT", 12.0, 3.0)


class AccountService:
    """账号池服务，使用 token -> account 的 dict 保存账号。"""

    def __init__(self, storage_backend: StorageBackend):
        self.storage = storage_backend
        self._lock = Lock()
        self._image_slot_condition = Condition(self._lock)
        self._index = 0
        self._accounts = self._load_accounts()
        self._image_inflight: dict[str, int] = {}

    def _load_accounts(self) -> dict[str, dict]:
        accounts = self.storage.load_accounts()
        return {
            normalized["access_token"]: normalized
            for item in accounts
            if (normalized := self._normalize_account(item)) is not None
        }

    def _save_accounts(self) -> None:
        self.storage.save_accounts(list(self._accounts.values()))

    @staticmethod
    def _build_account_changes(before: dict | None, after: dict | None, fields: set[str]) -> dict[str, dict[str, Any]]:
        if after is None:
            return {}
        source = before or {}
        changes: dict[str, dict[str, Any]] = {}
        for field in sorted(fields):
            before_value = source.get(field)
            after_value = after.get(field)
            if before_value != after_value:
                changes[field] = {
                    "before": before_value,
                    "after": after_value,
                }
        return changes

    @staticmethod
    def _is_image_account_available(account: dict) -> bool:
        if not isinstance(account, dict):
            return False
        if account.get("status") in {"禁用", "限流", "异常"}:
            return False
        if bool(account.get("image_quota_unknown")):
            return True
        return int(account.get("quota") or 0) > 0

    def _normalize_account(self, item: dict) -> dict | None:
        if not isinstance(item, dict):
            return None
        access_token = item.get("access_token") or ""
        if not access_token:
            return None
        normalized = dict(item)
        normalized["access_token"] = access_token
        normalized["type"] = normalized.get("type") or "free"
        normalized["status"] = normalized.get("status") or "正常"
        normalized["quota"] = max(0, int(normalized.get("quota") if normalized.get("quota") is not None else 0))
        normalized["image_quota_unknown"] = bool(normalized.get("image_quota_unknown"))
        normalized["email"] = normalized.get("email") or None
        normalized["user_id"] = normalized.get("user_id") or None
        limits_progress = normalized.get("limits_progress")
        normalized["limits_progress"] = limits_progress if isinstance(limits_progress, list) else []
        normalized["default_model_slug"] = normalized.get("default_model_slug") or None
        normalized["restore_at"] = normalized.get("restore_at") or None
        normalized["success"] = int(normalized.get("success") or 0)
        normalized["fail"] = int(normalized.get("fail") or 0)
        normalized["last_used_at"] = normalized.get("last_used_at")
        return normalized

    def list_tokens(self) -> list[str]:
        with self._lock:
            return list(self._accounts)

    def _list_ready_candidate_tokens(self, excluded_tokens: set[str] | None = None) -> list[str]:
        excluded = set(excluded_tokens or set())
        return [
            token
            for item in self._accounts.values()
            if self._is_image_account_available(item)
               and (token := item.get("access_token") or "")
               and token not in excluded
        ]

    def _list_available_candidate_tokens(self, excluded_tokens: set[str] | None = None) -> list[str]:
        max_concurrency = max(1, int(config.image_account_concurrency or 1))
        return [
            token
            for token in self._list_ready_candidate_tokens(excluded_tokens)
            if int(self._image_inflight.get(token, 0)) < max_concurrency
        ]

    def _acquire_next_candidate_token(self, excluded_tokens: set[str] | None = None) -> str:
        with self._image_slot_condition:
            while True:
                if not self._list_ready_candidate_tokens(excluded_tokens):
                    raise RuntimeError("no available image quota")
                tokens = self._list_available_candidate_tokens(excluded_tokens)
                if tokens:
                    access_token = tokens[self._index % len(tokens)]
                    self._index += 1
                    self._image_inflight[access_token] = int(self._image_inflight.get(access_token, 0)) + 1
                    return access_token
                self._image_slot_condition.wait(timeout=1.0)

    def release_image_slot(self, access_token: str) -> None:
        if not access_token:
            return
        with self._image_slot_condition:
            current_inflight = int(self._image_inflight.get(access_token, 0))
            if current_inflight <= 1:
                self._image_inflight.pop(access_token, None)
            else:
                self._image_inflight[access_token] = current_inflight - 1
            self._image_slot_condition.notify_all()

    def get_available_access_token(self, excluded_tokens: set[str] | None = None) -> str:
        attempted_tokens: set[str] = set(excluded_tokens or set())
        last_error: Exception | None = None
        while True:
            try:
                access_token = self._acquire_next_candidate_token(excluded_tokens=attempted_tokens)
            except RuntimeError as exc:
                if last_error is not None:
                    raise RuntimeError(str(last_error)) from last_error
                raise
            attempted_tokens.add(access_token)
            try:
                account = self.fetch_remote_info(access_token, "get_available_access_token")
            except Exception as exc:
                last_error = exc
                self.release_image_slot(access_token)
                continue
            if self._is_image_account_available(account or {}):
                return access_token
            self.release_image_slot(access_token)

    def get_text_access_token(self, excluded_tokens: set[str] | None = None) -> str:
        excluded = set(excluded_tokens or set())
        with self._lock:
            candidates = [
                token
                for account in self._accounts.values()
                if account.get("status") not in {"禁用", "异常"}
                   and (token := account.get("access_token") or "")
                   and token not in excluded
            ]
            if not candidates:
                return ""
            access_token = candidates[self._index % len(candidates)]
            self._index += 1
            return access_token

    def mark_text_used(self, access_token: str) -> None:
        if not access_token:
            return
        with self._lock:
            current = self._accounts.get(access_token)
            if current is None:
                return
            next_item = dict(current)
            next_item["last_used_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            account = self._normalize_account(next_item)
            if account is None:
                return
            self._accounts[access_token] = account
            self._save_accounts()

    def remove_invalid_token(self, access_token: str, event: str) -> bool:
        if not config.auto_remove_invalid_accounts:
            self.update_account(access_token, {"status": "异常", "quota": 0})
            return False
        removed = bool(self.delete_accounts([access_token])["removed"])
        if removed:
            log_service.add(LOG_TYPE_ACCOUNT, "自动移除异常账号",
                            {"source": event, "token": anonymize_token(access_token)})
        elif access_token:
            self.update_account(access_token, {"status": "异常", "quota": 0})
        return removed

    def get_account(self, access_token: str) -> dict | None:
        if not access_token:
            return None
        with self._lock:
            account = self._accounts.get(access_token)
            return dict(account) if account else None

    def list_accounts(self) -> list[dict]:
        with self._lock:
            return [dict(item) for item in self._accounts.values()]

    @staticmethod
    def _display_account_type(account: dict) -> str:
        return str(account.get("type") or "Free")

    @staticmethod
    def _is_unlimited_image_quota_account(account: dict) -> bool:
        return str(account.get("type") or "").lower() in {"pro", "prolite"}

    @staticmethod
    def _format_compact_quota(value: int) -> str:
        if value >= 1000:
            formatted = f"{value / 1000:.1f}".rstrip("0").rstrip(".")
            return f"{formatted}k"
        return str(value)

    def _build_account_summary(self, accounts: list[dict]) -> dict[str, Any]:
        active_accounts = [item for item in accounts if item.get("status") == "正常"]
        quota_unlimited = any(self._is_unlimited_image_quota_account(item) for item in active_accounts)
        quota_unknown = (not quota_unlimited) and any(bool(item.get("image_quota_unknown")) for item in active_accounts)
        quota_value = sum(max(0, int(item.get("quota") or 0)) for item in active_accounts)
        if quota_unlimited:
            quota_display = "∞"
        elif quota_unknown:
            quota_display = "未知"
        else:
            quota_display = self._format_compact_quota(quota_value)

        return {
            "total": len(accounts),
            "active": sum(1 for item in accounts if item.get("status") == "正常"),
            "limited": sum(1 for item in accounts if item.get("status") == "限流"),
            "abnormal": sum(1 for item in accounts if item.get("status") == "异常"),
            "disabled": sum(1 for item in accounts if item.get("status") == "禁用"),
            "quota": quota_value,
            "quota_display": quota_display,
            "quota_unknown": quota_unknown,
            "quota_unlimited": quota_unlimited,
        }

    def get_account_summary(self) -> dict[str, Any]:
        with self._lock:
            accounts = [dict(item) for item in self._accounts.values()]
        return self._build_account_summary(accounts)

    def get_account_quota_summary(self) -> dict[str, Any]:
        summary = self.get_account_summary()
        return {
            "total": summary["total"],
            "active": summary["active"],
            "limited": summary["limited"],
            "abnormal": summary["abnormal"],
            "disabled": summary["disabled"],
            "quota": summary["quota"],
            "quota_display": summary["quota_display"],
            "quota_unknown": summary["quota_unknown"],
            "quota_unlimited": summary["quota_unlimited"],
        }

    def list_accounts_page(
        self,
        page: int = 1,
        page_size: int = 20,
        query: str = "",
        account_type: str = "all",
        status: str = "all",
    ) -> dict[str, Any]:
        with self._lock:
            accounts = [dict(item) for item in self._accounts.values()]

        normalized_query = str(query or "").strip().lower()
        normalized_type = str(account_type or "all").strip()
        normalized_status = str(status or "all").strip()

        filtered = []
        for account in accounts:
            if normalized_status != "all" and account.get("status") != normalized_status:
                continue
            if normalized_type != "all" and self._display_account_type(account) != normalized_type:
                continue
            if normalized_query:
                haystack = " ".join(
                    str(value or "").lower()
                    for value in (
                        account.get("email"),
                        account.get("user_id"),
                        account.get("access_token"),
                        account.get("type"),
                        account.get("status"),
                        account.get("default_model_slug"),
                    )
                )
                if normalized_query not in haystack:
                    continue
            filtered.append(account)

        page_size = min(MAX_ACCOUNT_PAGE_SIZE, max(1, int(page_size or 20)))
        total = len(filtered)
        pages = max(1, (total + page_size - 1) // page_size)
        page = min(max(1, int(page or 1)), pages)
        start = (page - 1) * page_size
        types = sorted({self._display_account_type(account) for account in accounts})

        return {
            "items": filtered[start : start + page_size],
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": pages,
            "summary": self._build_account_summary(accounts),
            "types": types,
        }

    def list_account_tokens(self) -> list[str]:
        return self.list_tokens()

    def list_limited_tokens(self) -> list[str]:
        with self._lock:
            return [
                token
                for item in self._accounts.values()
                if item.get("status") == "限流"
                   and (token := item.get("access_token") or "")
            ]

    def add_accounts(self, tokens: list[str], include_items: bool = True) -> dict:
        tokens = list(dict.fromkeys(token for token in tokens if token))
        if not tokens:
            result = {"added": 0, "skipped": 0}
            if include_items:
                result["items"] = self.list_accounts()
            return result

        with self._lock:
            added = 0
            skipped = 0
            for access_token in tokens:
                current = self._accounts.get(access_token)
                if current is None:
                    added += 1
                    current = {}
                else:
                    skipped += 1
                account = self._normalize_account(
                    {
                        **current,
                        "access_token": access_token,
                        "type": str(current.get("type") or "free"),
                    }
                )
                if account is not None:
                    self._accounts[access_token] = account
            self._save_accounts()
            log_service.add(LOG_TYPE_ACCOUNT, f"新增 {added} 个账号，跳过 {skipped} 个",
                            {"added": added, "skipped": skipped})
            result = {"added": added, "skipped": skipped}
            if include_items:
                result["items"] = [dict(item) for item in self._accounts.values()]
        return result

    def delete_accounts(self, tokens: list[str], include_items: bool = True) -> dict:
        target_set = set(token for token in tokens if token)
        if not target_set:
            result = {"removed": 0}
            if include_items:
                result["items"] = self.list_accounts()
            return result
        with self._lock:
            removed = sum(self._accounts.pop(token, None) is not None for token in target_set)
            for token in target_set:
                self._image_inflight.pop(token, None)
            if removed:
                if self._accounts:
                    self._index %= len(self._accounts)
                else:
                    self._index = 0
                self._save_accounts()
                log_service.add(LOG_TYPE_ACCOUNT, f"删除 {removed} 个账号", {"removed": removed})
            result = {"removed": removed}
            if include_items:
                result["items"] = [dict(item) for item in self._accounts.values()]
        return result

    def update_account(self, access_token: str, updates: dict) -> dict | None:
        if not access_token:
            return None
        with self._lock:
            current = dict(self._accounts.get(access_token) or {})
            account, changed = self._apply_account_update_locked(access_token, updates)
            if changed:
                self._save_accounts()
            if account is not None:
                log_service.add(LOG_TYPE_ACCOUNT, "更新账号",
                                {
                                    "action": "update_account",
                                    "token": anonymize_token(access_token),
                                    "status": account.get("status"),
                                    "changes": self._build_account_changes(current, account, set(updates)),
                                })
            return account

    def _apply_account_update_locked(self, access_token: str, updates: dict) -> tuple[dict | None, bool]:
        current = self._accounts.get(access_token)
        if current is None:
            return None, False
        account = self._normalize_account({**current, **updates, "access_token": access_token})
        if account is None:
            return None, False
        if account.get("status") == "限流" and config.auto_remove_rate_limited_accounts:
            removed = self._accounts.pop(access_token, None) is not None
            self._image_inflight.pop(access_token, None)
            if removed:
                log_service.add(LOG_TYPE_ACCOUNT, "自动移除限流账号", {"token": anonymize_token(access_token)})
            return None, removed
        self._accounts[access_token] = account
        return dict(account), True

    def _apply_invalid_token_locked(self, access_token: str, event: str) -> bool:
        if not access_token:
            return False
        if config.auto_remove_invalid_accounts:
            removed = self._accounts.pop(access_token, None) is not None
            self._image_inflight.pop(access_token, None)
            if removed:
                log_service.add(LOG_TYPE_ACCOUNT, "自动移除异常账号",
                                {"source": event, "token": anonymize_token(access_token)})
            return removed
        account, changed = self._apply_account_update_locked(access_token, {"status": "异常", "quota": 0})
        if changed and account is not None:
            log_service.add(LOG_TYPE_ACCOUNT, "标记异常账号",
                            {"source": event, "token": anonymize_token(access_token)})
        return changed

    def mark_image_result(self, access_token: str, success: bool) -> dict | None:
        if not access_token:
            return None
        self.release_image_slot(access_token)
        with self._lock:
            current = self._accounts.get(access_token)
            if current is None:
                return None
            next_item = dict(current)
            next_item["last_used_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            image_quota_unknown = bool(next_item.get("image_quota_unknown"))
            if success:
                next_item["success"] = int(next_item.get("success") or 0) + 1
                if not image_quota_unknown:
                    next_item["quota"] = max(0, int(next_item.get("quota") or 0) - 1)
                if not image_quota_unknown and next_item["quota"] == 0:
                    next_item["status"] = "限流"
                    next_item["restore_at"] = next_item.get("restore_at") or None
                elif next_item.get("status") == "限流":
                    next_item["status"] = "正常"
            else:
                next_item["fail"] = int(next_item.get("fail") or 0) + 1
            account = self._normalize_account(next_item)
            if account is None:
                return None
            if account.get("status") == "限流" and config.auto_remove_rate_limited_accounts:
                self._accounts.pop(access_token, None)
                self._save_accounts()
                log_service.add(LOG_TYPE_ACCOUNT, "自动移除限流账号", {"token": anonymize_token(access_token)})
                return None
            self._accounts[access_token] = account
            self._save_accounts()
            return dict(account)
        return None

    @staticmethod
    def _can_refresh_quota_only(account: dict | None) -> bool:
        if not isinstance(account, dict):
            return False
        return bool(account.get("email") or account.get("user_id")) and bool(account.get("type"))

    def _fetch_remote_info_result(
        self,
        access_token: str,
        timeout: float | None = None,
        quota_only_if_possible: bool = False,
    ) -> dict[str, Any]:
        if not access_token:
            raise ValueError("access_token is required")
        from services.openai_backend_api import OpenAIBackendAPI
        current_account = self.get_account(access_token)
        backend = OpenAIBackendAPI(access_token)
        if quota_only_if_possible and self._can_refresh_quota_only(current_account):
            return backend.get_quota_info(current_account, timeout=timeout)
        return backend.get_user_info(timeout=timeout)

    def fetch_remote_info(self, access_token: str, event: str = "fetch_remote_info") -> dict[str, Any] | None:
        try:
            from services.openai_backend_api import InvalidAccessTokenError
            result = self._fetch_remote_info_result(access_token, quota_only_if_possible=False)
        except InvalidAccessTokenError:
            self.remove_invalid_token(access_token, event)
            raise
        return self.update_account(access_token, result)

    def refresh_accounts(
        self,
        access_tokens: list[str],
        include_items: bool = True,
        quota_only_if_possible: bool = True,
    ) -> dict[str, Any]:
        access_tokens = list(dict.fromkeys(token for token in access_tokens if token))
        if not access_tokens:
            result: dict[str, Any] = {"refreshed": 0, "errors": []}
            if include_items:
                result["items"] = self.list_accounts()
            return result

        refreshed = 0
        errors = []
        updates: list[tuple[str, dict[str, Any]]] = []
        invalid_tokens: list[str] = []
        max_workers = min(ACCOUNT_REFRESH_MAX_WORKERS, len(access_tokens))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._fetch_remote_info_result,
                    token,
                    ACCOUNT_REFRESH_REQUEST_TIMEOUT,
                    quota_only_if_possible,
                ): token
                for token in access_tokens
            }
            for future in as_completed(futures):
                token = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    from services.openai_backend_api import InvalidAccessTokenError
                    if isinstance(exc, InvalidAccessTokenError):
                        invalid_tokens.append(token)
                    token_ref = anonymize_token(token)
                    errors.append({"token": token_ref, "error": str(exc).replace(token, token_ref)})
                    continue
                updates.append((token, result))

        changed = False
        refreshed_tokens: list[str] = []
        with self._lock:
            for token, result in updates:
                account, did_change = self._apply_account_update_locked(token, result)
                changed = changed or did_change
                if account is not None:
                    refreshed += 1
                    refreshed_tokens.append(anonymize_token(token))
            for token in invalid_tokens:
                changed = self._apply_invalid_token_locked(token, "refresh_accounts") or changed
            if changed:
                self._save_accounts()
            if include_items:
                items = [dict(item) for item in self._accounts.values()]

        if refreshed or errors:
            log_service.add(
                LOG_TYPE_ACCOUNT,
                f"批量刷新账号：成功 {refreshed} 个，失败 {len(errors)} 个",
                {
                    "requested": len(access_tokens),
                    "refreshed": refreshed,
                    "errors": len(errors),
                    "workers": max_workers,
                    "action": "refresh_accounts",
                    "requested_tokens": [anonymize_token(token) for token in access_tokens],
                    "refreshed_tokens": refreshed_tokens,
                    "failed_items": errors,
                },
            )

        result: dict[str, Any] = {
            "refreshed": refreshed,
            "errors": errors,
        }
        if include_items:
            result["items"] = items
        return result


account_service = AccountService(config.get_storage_backend())
