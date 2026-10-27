from urllib.parse import urlsplit

from src.platform.database.database import db

_NAME = "070_llm_openai_compatible"
_HOSTED_HOST = "api.openai.com"


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
        cursor.execute("SELECT id, base_url FROM llm_configurations WHERE type = 'openai'")
        for config_id, base_url in cursor.fetchall():
            host = _host(base_url)
            if host is None or host == _HOSTED_HOST:
                continue
            cursor.execute("UPDATE llm_configurations SET type = 'openai-compatible' WHERE id = ?", (config_id,))
            changed += 1

    print(f"Migration {_NAME}: {changed} configuration(s) on a server other than api.openai.com set to openai-compatible")
