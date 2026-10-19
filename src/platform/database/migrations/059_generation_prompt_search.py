from src.platform.database.database import db

_TABLE = (
    "CREATE TABLE IF NOT EXISTS generation_prompt_text ("
    "id INTEGER PRIMARY KEY, "
    "generation_id TEXT NOT NULL UNIQUE REFERENCES generations (id) ON DELETE CASCADE ON UPDATE CASCADE, "
    "text TEXT)"
)

_TRIGGER_NAMES = (
    "generation_prompt_text_insert",
    "generation_prompt_text_update",
    "generation_prompt_text_delete",
)


def _prompt_text(row: str) -> str:
    sources = " UNION ALL ".join(
        f"SELECT jt.value FROM json_tree(CASE WHEN json_valid({row}.{column}) THEN {row}.{column} ELSE 'null' END) jt "
        "WHERE jt.type = 'text' AND LOWER(jt.key) LIKE '%prompt%' AND LOWER(jt.key) NOT LIKE '%negative%'"
        for column in ("form_data", "prompt_state")
    )
    return f"(SELECT group_concat(value, char(31)) FROM ({sources}))"


def _upsert(row: str) -> str:
    return (
        f"INSERT INTO generation_prompt_text (generation_id, text) VALUES ({row}.id, {_prompt_text(row)}) "
        "ON CONFLICT (generation_id) DO UPDATE SET text = excluded.text WHERE text IS NOT excluded.text;"
    )


def _generation_triggers():
    return (
        f"CREATE TRIGGER IF NOT EXISTS generation_prompt_text_insert AFTER INSERT ON generations BEGIN {_upsert('NEW')} END",
        "CREATE TRIGGER IF NOT EXISTS generation_prompt_text_update AFTER UPDATE OF form_data, prompt_state ON generations "
        f"WHEN OLD.form_data IS NOT NEW.form_data OR OLD.prompt_state IS NOT NEW.prompt_state BEGIN {_upsert('NEW')} END",
        "CREATE TRIGGER IF NOT EXISTS generation_prompt_text_delete AFTER DELETE ON generations BEGIN "
        "DELETE FROM generation_prompt_text WHERE generation_id = OLD.id; END",
    )


def up():
    with db.get_cursor() as cursor:
        cursor.execute(_TABLE)
        for statement in _generation_triggers():
            cursor.execute(statement)
        cursor.execute(
            f"INSERT INTO generation_prompt_text (generation_id, text) SELECT g.id, {_prompt_text('g')} "
            "FROM generations g WHERE true ON CONFLICT (generation_id) DO NOTHING"
        )

    print("Migration 059_generation_prompt_search: applied")


def down():
    with db.get_cursor() as cursor:
        for name in _TRIGGER_NAMES:
            cursor.execute(f"DROP TRIGGER IF EXISTS {name}")
        cursor.execute("DROP TABLE IF EXISTS generation_prompt_text")

    print("Migration 059_generation_prompt_search: dropped the prompt search table")
