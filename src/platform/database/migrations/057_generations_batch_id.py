from src.platform.database.database import db


def up():
    with db.get_cursor() as cursor:
        cursor.execute("PRAGMA table_info(generations)")
        existing = {row[1] for row in cursor.fetchall()}
        if "batch_id" not in existing:
            cursor.execute("ALTER TABLE generations ADD COLUMN batch_id TEXT")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_generations_batch_id ON generations (batch_id)")

    print("Migration 057_generations_batch_id: applied")
