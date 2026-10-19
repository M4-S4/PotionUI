from src.platform.database.database import db


def up():
    with db.get_cursor() as cursor:
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS tab_drafts ("
            "user_id TEXT NOT NULL REFERENCES users (id) ON DELETE CASCADE, "
            "tab_id TEXT NOT NULL, "
            "doc TEXT NOT NULL, "
            "rev INTEGER NOT NULL DEFAULT 1, "
            "variable_count INTEGER NOT NULL DEFAULT 0, "
            "created_at TEXT NOT NULL, "
            "updated_at TEXT NOT NULL, "
            "PRIMARY KEY (user_id, tab_id))"
        )
        cursor.execute("PRAGMA table_info(tab_drafts)")
        if "variable_count" not in {row[1] for row in cursor.fetchall()}:
            cursor.execute("ALTER TABLE tab_drafts ADD COLUMN variable_count INTEGER NOT NULL DEFAULT 0")
            cursor.execute(
                "UPDATE tab_drafts SET variable_count = "
                "(SELECT COUNT(*) FROM json_each(tab_drafts.doc, '$.variables'))"
            )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tab_drafts_updated_at ON tab_drafts (updated_at)")

    print("Migration 060_tab_drafts: applied")


def down():
    with db.get_cursor() as cursor:
        cursor.execute("DROP INDEX IF EXISTS idx_tab_drafts_updated_at")
        cursor.execute("DROP TABLE IF EXISTS tab_drafts")

    print("Migration 060_tab_drafts: dropped tab_drafts")
