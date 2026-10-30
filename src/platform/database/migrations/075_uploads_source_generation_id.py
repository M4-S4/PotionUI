from src.platform.database.database import db

_NAME = "075_uploads_source_generation_id"


def up():
    with db.get_cursor() as cursor:
        cursor.execute("PRAGMA table_info(uploads)")
        existing = {row[1] for row in cursor.fetchall()}
        if "source_generation_id" not in existing:
            cursor.execute("ALTER TABLE uploads ADD COLUMN source_generation_id TEXT")

    print(f"Migration {_NAME}: applied")
