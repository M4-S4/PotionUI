from src.platform.database.database import db
from src.platform.database.rows import now_iso
from src.platform.util.ids import generate_ulid

_NAME = "062_public_base_url"

_KEY = "public_base_url"
_DESCRIPTION = (
    "Public address of this instance (for example https://potion.example.com), "
    "used to build links to generations for external clients. Empty uses the address the client connected to."
)


def up():
    with db.get_cursor() as cursor:
        cursor.execute("SELECT 1 FROM settings WHERE key = ?", (_KEY,))
        if cursor.fetchone() is not None:
            print(f"Migration {_NAME}: setting already present")
            return
        now = now_iso()
        cursor.execute(
            """
            INSERT INTO settings (id, key, value, value_type, description, type, created_at, updated_at)
            VALUES (?, ?, '', 'string', ?, 'SYSTEM', ?, ?)
            """,
            (generate_ulid(), _KEY, _DESCRIPTION, now, now),
        )
    print(f"Migration {_NAME}: seeded {_KEY}")
