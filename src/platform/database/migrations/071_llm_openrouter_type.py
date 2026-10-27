from urllib.parse import urlsplit

from src.platform.database.database import db

_NAME = "071_llm_openrouter_type"
_HOST = "openrouter.ai"
_FROM_TYPES = ("openai", "openai-compatible")


def _host(base_url):
    text = (base_url or "").strip()
    if not text:
        return None
    if "://" not in text:
        text = f"//{text}"
    try:
        host = urlsplit(text).hostname
    except ValueError:
        return None
    return host.rstrip(".") if host else None


def up():
    changed = 0
    with db.get_cursor() as cursor:
        marks = ", ".join("?" for _ in _FROM_TYPES)
        cursor.execute(f"SELECT id, base_url FROM llm_configurations WHERE type IN ({marks})", _FROM_TYPES)
        for config_id, base_url in cursor.fetchall():
            if _host(base_url) != _HOST:
                continue
            cursor.execute("UPDATE llm_configurations SET type = 'openrouter' WHERE id = ?", (config_id,))
            changed += 1

    print(f"Migration {_NAME}: {changed} configuration(s) on {_HOST} set to openrouter")
