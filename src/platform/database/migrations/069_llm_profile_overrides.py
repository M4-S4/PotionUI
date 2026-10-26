import json

from src.platform.database.database import db

_NAME = "069_llm_profile_overrides"
_OLD_DEFAULT_TEMPERATURE = 0.7
_MIN_CONTEXT = 1024
_CHILDREN = ("user_llms", "user_group_llms")
_COLUMNS = (
    "id", "name", "type", "enabled", "base_url", "api_key", "model", "system_message", "temperature", "max_tokens",
    "timeout", "created_at", "updated_at", "supports_vision", "provider_options", "disable_system_prompt",
    "memory_reflection", "show_thinking", "profile_overrides",
)

_CREATE = """
    CREATE TABLE llm_configurations_069 (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        enabled BOOLEAN NOT NULL DEFAULT 1,
        base_url TEXT NOT NULL,
        api_key TEXT,
        model TEXT NOT NULL,
        system_message TEXT NOT NULL,
        temperature REAL,
        max_tokens INTEGER,
        timeout INTEGER NOT NULL DEFAULT 30,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        supports_vision BOOLEAN NOT NULL DEFAULT 0,
        provider_options TEXT DEFAULT NULL,
        disable_system_prompt BOOLEAN NOT NULL DEFAULT 0,
        memory_reflection BOOLEAN NOT NULL DEFAULT 1,
        show_thinking BOOLEAN NOT NULL DEFAULT 0,
        profile_overrides TEXT
    )
"""

_INDEXES = (
    "CREATE INDEX idx_llm_configurations_enabled ON llm_configurations(enabled)",
    "CREATE INDEX idx_llm_configurations_vision ON llm_configurations(supports_vision)",
)

_TRIGGER = """
    CREATE TRIGGER update_llm_configurations_updated_at
    AFTER UPDATE ON llm_configurations
    FOR EACH ROW
    BEGIN
        UPDATE llm_configurations SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END
"""


def up():
    with db.get_cursor() as cursor:
        info = _table_info(cursor)
        rebuilt = info["temperature"]["notnull"] or info["max_tokens"]["notnull"]
        if rebuilt:
            _rebuild(cursor, set(info))
        elif "profile_overrides" not in info:
            cursor.execute("ALTER TABLE llm_configurations ADD COLUMN profile_overrides TEXT")
        moved, skipped = _move_provider_options(cursor)

    print(
        f"Migration {_NAME}: sampling columns allow NULL"
        f"{' and rows still at the old default temperature were cleared' if rebuilt else ''}, "
        f"{moved} configuration(s) had provider options moved into profile overrides, {skipped} value(s) dropped as unusable"
    )


def _table_info(cursor):
    cursor.execute("PRAGMA table_info(llm_configurations)")
    return {row["name"]: row for row in cursor.fetchall()}


def _rebuild(cursor, existing):
    shared = [name for name in _COLUMNS if name in existing]
    listed = ", ".join(shared)
    selected = ", ".join(
        f"CASE WHEN temperature = {_OLD_DEFAULT_TEMPERATURE} THEN NULL ELSE temperature END" if name == "temperature" else name
        for name in shared
    )
    cursor.execute("BEGIN IMMEDIATE")
    cursor.execute("DROP TABLE IF EXISTS llm_configurations_069")
    for child in _CHILDREN:
        cursor.execute(f"DROP TABLE IF EXISTS {child}_069")
    cursor.execute(_CREATE)
    cursor.execute(f"INSERT INTO llm_configurations_069 ({listed}) SELECT {selected} FROM llm_configurations")
    for child in _CHILDREN:
        cursor.execute(f"CREATE TABLE {child}_069 AS SELECT * FROM {child}")
    cursor.execute("DROP TABLE llm_configurations")
    cursor.execute("ALTER TABLE llm_configurations_069 RENAME TO llm_configurations")
    for statement in _INDEXES:
        cursor.execute(statement)
    cursor.execute(_TRIGGER)
    for child in _CHILDREN:
        cursor.execute(f"INSERT OR IGNORE INTO {child} SELECT * FROM {child}_069")
        cursor.execute(f"DROP TABLE {child}_069")


def _decode(raw):
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _size(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if value >= _MIN_CONTEXT else None


def _move_provider_options(cursor):
    cursor.execute("SELECT id, type, provider_options, profile_overrides FROM llm_configurations")
    moved = skipped = 0
    for row in cursor.fetchall():
        options = _decode(row["provider_options"])
        overrides = _decode(row["profile_overrides"])
        if options is None or overrides is None:
            continue
        keys = [key for key in ("context_window", "num_ctx", "force_prompt_tools") if key in options]
        if row["type"] != "ollama" and "num_ctx" in keys:
            keys.remove("num_ctx")
        if not keys:
            continue
        gained = _overrides_from(row["type"], options)
        skipped += sum(
            1 for key in ("context_window", "num_ctx") if key in keys and options[key] is not None and _size(options[key]) is None
        )
        for key in keys:
            options.pop(key)
        merged = {**gained, **overrides}
        cursor.execute(
            "UPDATE llm_configurations SET provider_options = ?, profile_overrides = ? WHERE id = ?",
            (json.dumps(options) if options else None, json.dumps(merged, sort_keys=True) if merged else None, row["id"]),
        )
        moved += 1
    return moved, skipped


def _overrides_from(config_type, options):
    gained = {}
    size = _size(options.get("context_window"))
    if config_type == "ollama":
        size = size or _size(options.get("num_ctx"))
        if size:
            gained["context_window"] = size
            gained["context_request"] = size
    elif size:
        gained["context_window"] = size
    if options.get("force_prompt_tools") is True:
        gained["prompt_tools"] = True
    return gained
