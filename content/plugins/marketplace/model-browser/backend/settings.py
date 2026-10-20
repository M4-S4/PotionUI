from typing import Any
from urllib.parse import urlsplit

from . import PLUGIN_ID

CORE_USER_NSFW_SETTING = "media_nsfw_filter_mode"
USER_NSFW_MODES = ("show", "blur", "hide")


class _NoStoredSettings:
    def get_plugin_settings(self, plugin_id: str) -> list:
        return []


def get_setting(key: str) -> Any:
    from src.plugin_api import plugin_settings

    values = plugin_settings(PLUGIN_ID)
    if key not in values:
        raise KeyError(f"model-browser has no setting {key!r}")
    value = values[key]
    if key == "nsfw_max_level":
        return int(value)
    if key == "civitai_api_base":
        base = str(value).strip().rstrip("/")
        try:
            if base.startswith("https://") and urlsplit(base).netloc:
                return base
        except ValueError:
            pass
        return plugin_settings(PLUGIN_ID, _NoStoredSettings())[key]
    return value


def user_nsfw_mode(user_id: str) -> str:
    from src.plugin_api import SettingRepository

    value = SettingRepository().get_user_setting_by_key(user_id, CORE_USER_NSFW_SETTING)
    value = str(value).strip().lower() if value is not None else ""
    return value if value in USER_NSFW_MODES else "blur"
