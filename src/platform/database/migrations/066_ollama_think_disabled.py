import json

from src.platform.database.database import db

_NAME = "066_ollama_think_disabled"


def _decode(raw):
    if raw is None or raw == "":
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if value is None:
        return {}
    return value if isinstance(value, dict) else None


def up():
    changed = 0
    with db.get_cursor() as cursor:
        cursor.execute("SELECT id, provider_options FROM llm_configurations WHERE type = 'ollama'")
        for config_id, raw in cursor.fetchall():
            options = _decode(raw)
            if options is None or options.get("think") is not None:
                continue
            options["think"] = False
            cursor.execute(
                "UPDATE llm_configurations SET provider_options = ? WHERE id = ?",
                (json.dumps(options), config_id),
            )
            changed += 1

    print(f"Migration {_NAME}: {changed} ollama configuration(s) set to thinking disabled")
