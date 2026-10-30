from src.platform.database.database import db

_NAME = "073_media_tool_params"

_CREATE = (
    "CREATE TABLE IF NOT EXISTS media_tool_params ("
    "user_id TEXT NOT NULL, "
    "tool_id TEXT NOT NULL, "
    "params_json TEXT NOT NULL, "
    "updated_at TEXT NOT NULL, "
    "PRIMARY KEY (user_id, tool_id), "
    "FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE)"
)


def up():
    with db.get_cursor() as cursor:
        cursor.execute(_CREATE)

    print(f"Migration {_NAME}: media_tool_params created")
