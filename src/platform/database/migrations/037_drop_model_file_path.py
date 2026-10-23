from src.platform.database.database import db

_NAME = "037_drop_model_file_path"


def up():
    with db.get_cursor() as cursor:
        cursor.execute("DROP INDEX IF EXISTS idx_models_file_path")
        if _has_column(cursor, "models", "file_path"):
            cursor.execute("ALTER TABLE models DROP COLUMN file_path")
    print(f"Migration {_NAME}: dropped models.file_path and its index")


def _has_column(cursor, table, column) -> bool:
    cursor.execute(f"PRAGMA table_info({table})")
    return any(row["name"] == column for row in cursor.fetchall())
