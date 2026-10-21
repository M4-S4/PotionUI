from src.platform.database.database import db
from src.platform.database.rows import now_iso
from src.platform.util.ids import generate_ulid

_SETTINGS = (
    ("preset_picker_pinned", "Presets pinned to the top of the preset dropdown, per user."),
    ("preset_picker_recent", "Presets most recently chosen in the preset dropdown, newest first, per user."),
)


def up():
    with db.get_cursor() as cursor:
        for key, description in _SETTINGS:
            cursor.execute("SELECT 1 FROM settings WHERE key = ?", (key,))
            if cursor.fetchone() is not None:
                print(f"Migration 063_preset_picker_state: {key} already present")
                continue
            now = now_iso()
            cursor.execute(
                """
                INSERT INTO settings (id, key, value, value_type, description, type, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'USER', ?, ?)
                """,
                (generate_ulid(), key, "[]", "json", description, now, now),
            )
            print(f"Migration 063_preset_picker_state: seeded {key}")
