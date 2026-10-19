from src.platform.database.database import db

_INDEXES = (
    ("idx_generations_created_id", "(created_at DESC, id DESC)"),
    ("idx_generations_status_created_id", "(status, created_at DESC, id DESC)"),
    ("idx_generations_user_mode", "(user_id, mode, created_at DESC, id DESC)"),
    ("idx_generations_user_preset", "(user_id, preset_id, created_at DESC, id DESC)"),
)


def up():
    with db.get_cursor() as cursor:
        for name, columns in _INDEXES:
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {name} ON generations {columns}")

    print("Migration 058_history_indexes: applied")


def down():
    with db.get_cursor() as cursor:
        for name, _columns in _INDEXES:
            cursor.execute(f"DROP INDEX IF EXISTS {name}")

    print("Migration 058_history_indexes: dropped the added indexes")
