from src.plugin_api import HookContext
from src.plugin_api.providers import register_provider as register_provider_class


def _load_provider() -> type:
    from ..provider.civitai_provider import CivitaiProvider

    return CivitaiProvider


def register_provider(context: HookContext) -> HookContext:
    return register_provider_class(context, "civitai", "CivitAI", _load_provider)
