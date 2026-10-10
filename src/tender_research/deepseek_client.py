"""DeepSeek 服务适配；凭据不进入请求日志、实验记录或异常正文。"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from .model_service import ProviderError

BASE_URL = "https://api.deepseek.com"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def get_key(root: Path) -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        path = root / ".secrets/deepseek_api_key"
        if not path.exists():
            raise ProviderError("missing_credential")
        if path.stat().st_mode & 0o077:
            raise ProviderError("credential_permissions_too_open")
        key = path.read_text().strip()
    if not key or not re.fullmatch(r"sk-[A-Za-z0-9_-]+", key):
        raise ProviderError("invalid_credential_format")
    return key


class DeepSeekClient:
    provider_id = "deepseek"

    def __init__(self, root: Path, timeout: float = 120):
        self.key, self.timeout = get_key(root), timeout
        self.opener = urllib.request.build_opener(NoRedirect())

    def request(self, path: str, payload: dict | None = None) -> dict:
        if path not in ("/models", "/user/balance", "/chat/completions"):
            raise ProviderError("unsupported_endpoint")
        data = None if payload is None else json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        req = urllib.request.Request(BASE_URL + path, data=data, method="GET" if data is None else "POST",
                                     headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                raw = response.read(10 * 1024 * 1024)
            return json.loads(raw)
        except urllib.error.HTTPError as exc:
            # 不持久化供应商异常正文，也不自动重试可能计费的请求。
            category = {401:"authentication_failed",402:"insufficient_balance",403:"access_denied",429:"rate_limited"}.get(exc.code,"http_error")
            raise ProviderError(category,exc.code) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise ProviderError("connection_or_timeout") from None
        except (ValueError, TypeError):
            raise ProviderError("invalid_response") from None

    def health(self) -> dict:
        result = {"provider":"deepseek","base_url":BASE_URL,"checked_at":datetime.now(UTC).isoformat()}
        try:
            models = self.request("/models")
            result.update(status="available", models=[r["id"] for r in models.get("data",[]) if isinstance(r.get("id"),str)])
        except ProviderError as exc:
            result.update(status="unavailable",error_category=exc.category,http_status=exc.http_status)
        return result

    def complete(self, messages: list[dict], model: str, *, max_tokens: int = 4096, thinking: bool = False) -> dict:
        if model not in ("deepseek-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"):
            raise ProviderError("unsupported_model")
        if not 1 <= max_tokens <= 32768:
            raise ProviderError("output_budget_exceeded")
        payload = {"model":model,"messages":messages,"max_tokens":max_tokens,"stream":False,
                   "response_format":{"type":"json_object"},"thinking":{"type":"enabled" if thinking else "disabled"}}
        if not thinking:
            payload["temperature"] = 0
        else:
            payload["reasoning_effort"] = "high"
        started = time.perf_counter(); response = self.request("/chat/completions",payload)
        try:
            choice = response["choices"][0]
            if choice["finish_reason"] != "stop":
                # 仅保留允许的原因及数值用量，不保存截断正文或思考文本。
                raw_usage = response.get("usage", {})
                if not isinstance(raw_usage, dict):
                    raw_usage = {}
                usage = {k: v for k, v in raw_usage.items() if k in (
                    "prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")
                    and isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 100_000_000}
                reason = choice["finish_reason"] if choice["finish_reason"] in ("length", "content_filter", "tool_calls", "insufficient_system_resource") else "other"
                raise ProviderError("incomplete_generation", telemetry={"requested_model": model,
                    "finish_reason": reason, "usage": usage, "elapsed_seconds": round(time.perf_counter()-started, 3)})
            content = choice["message"]["content"]
            if not isinstance(content,str) or self.key in content or re.search(r"sk-[A-Za-z0-9_-]{20,}",content):
                raise ProviderError("unsafe_or_invalid_content")
            parsed = json.loads(content)
            if not isinstance(parsed,dict):
                raise ProviderError("json_object_required")
        except (KeyError,IndexError,TypeError,ValueError):
            raise ProviderError("invalid_structured_generation") from None
        return {"provider":"deepseek","requested_model":model,"response_model":response.get("model"),
                "response_id":response.get("id"),"created":response.get("created"),"usage":response.get("usage",{}),
                "finish_reason":choice["finish_reason"],"elapsed_seconds":round(time.perf_counter()-started,3),
                "request_sha256":hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
                "response_sha256":hashlib.sha256(content.encode()).hexdigest(),"output":parsed}


if __name__ == "__main__":
    from .paths import REPO
    print(json.dumps(DeepSeekClient(REPO).health(), ensure_ascii=False))
