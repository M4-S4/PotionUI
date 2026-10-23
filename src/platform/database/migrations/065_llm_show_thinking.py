from src.platform.database.database import db

_NAME = "065_llm_show_thinking"


def up():
    with db.get_cursor() as cursor:
        cursor.execute("PRAGMA table_info(llm_configurations)")
        existing = {row[1] for row in cursor.fetchall()}
        if "show_thinking" not in existing:
            cursor.execute(
                "ALTER TABLE llm_configurations ADD COLUMN show_thinking BOOLEAN NOT NULL DEFAULT 0"
            )

    print(f"Migration {_NAME}: applied")
