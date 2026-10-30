from src.platform.database.database import db

_NAME = "072_llm_trace_wire"

_COLUMNS = (
    ("adapter", "TEXT"),
    ("request_body", "TEXT"),
    ("response_raw", "TEXT"),
    ("usage", "TEXT"),
    ("cost", "REAL"),
    ("error", "TEXT"),
)


def up():
    added = []
    with db.get_cursor() as cursor:
        cursor.execute("PRAGMA table_info(chat_llm_call_traces)")
        existing = {row[1] for row in cursor.fetchall()}
        for name, column_type in _COLUMNS:
            if name not in existing:
                cursor.execute(f"ALTER TABLE chat_llm_call_traces ADD COLUMN {name} {column_type}")
                added.append(name)

    print(f"Migration {_NAME}: {len(added)} trace column(s) added")
