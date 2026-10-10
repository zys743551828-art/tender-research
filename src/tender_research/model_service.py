"""厂商无关的模型服务边界；当前仅实现和注册 DeepSeek 适配器。"""
from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Protocol


class ProviderError(RuntimeError):
    """可记录的统一错误；不携带凭据或供应商原始响应正文。"""
    def __init__(self, category: str, http_status: int | None = None, *, telemetry: dict | None = None):
        self.category, self.http_status = category, http_status
        self.telemetry = telemetry
        super().__init__(f"Model service {category}" + (f" HTTP {http_status}" if http_status else ""))


class ModelProvider(Protocol):
    provider_id: str

    def health(self) -> dict:
        """返回 provider、status、models 及可记录的连接信息，不含凭据。"""
        ...

    def complete(self, messages: list[dict], model: str, *, max_tokens: int = 4096,
                 thinking: bool = False) -> dict:
        """返回统一响应元数据与 JSON output；不支持的能力必须显式拒绝。"""
        ...


ProviderFactory = Callable[[Path], ModelProvider]


class ProviderRegistry:
    def __init__(self):
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, provider_id: str, factory: ProviderFactory):
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", provider_id) or not callable(factory):
            raise ValueError("厂商编号或适配器工厂不合法")
        if provider_id in self._factories:
            raise ValueError("已注册的厂商适配器不能静默覆盖")
        self._factories[provider_id] = factory

    def create(self, provider_id: str, root: Path) -> ModelProvider:
        if provider_id not in self._factories:
            raise ProviderError("unsupported_provider")
        adapter = self._factories[provider_id](root)
        if adapter.provider_id != provider_id:
            raise ProviderError("provider_identity_mismatch")
        return adapter

    def registered(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def _deepseek(root: Path) -> ModelProvider:
    # 延迟加载，注册和读取接口不触发凭据访问或网络请求。
    from .deepseek_client import DeepSeekClient
    return DeepSeekClient(root)


_registry = ProviderRegistry()
_registry.register("deepseek", _deepseek)


def create_provider(root: Path, provider_id: str = "deepseek") -> ModelProvider:
    return _registry.create(provider_id, root)


def available_adapters() -> tuple[str, ...]:
    return _registry.registered()
