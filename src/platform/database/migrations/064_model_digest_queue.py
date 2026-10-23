from src.platform.database.database import db
from src.platform.database.rows import now_iso
from src.platform.util.ids import generate_ulid

_NAME = "064_model_digest_queue"

_CREATE_QUEUE = """
    CREATE TABLE IF NOT EXISTS model_digest_queue (
        location_id TEXT PRIMARY KEY REFERENCES model_locations(id) ON DELETE CASCADE,
        priority INTEGER NOT NULL DEFAULT 0,
        reason TEXT NOT NULL,
        size INTEGER NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0,
        last_error TEXT,
        not_before REAL,
        queued_at TIMESTAMP NOT NULL
    )
"""

_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_model_locations_size ON model_locations (size)",
    "CREATE INDEX IF NOT EXISTS idx_models_file_size ON models (file_size)",
    "CREATE INDEX IF NOT EXISTS idx_model_digest_queue_order ON model_digest_queue (priority DESC, size ASC)",
)

_LEVEL_DESCRIPTION = "How much of this computer model indexing may use: low, normal or max"
_PAUSED_DESCRIPTION = "Whether the model digest worker was paused by an administrator"
_POLICY_DESCRIPTION = "Whether new model files are read in the background: unset, background or on_demand"


def up():
    with db.get_cursor() as cursor:
        if "quick_sig" not in _column_names(cursor, "model_locations"):
            cursor.execute("ALTER TABLE model_locations ADD COLUMN quick_sig TEXT")
        cursor.execute(_CREATE_QUEUE)
        for statement in _INDEXES:
            cursor.execute(statement)
        seeded = _seed_settings(cursor)
    print(f"Migration {_NAME}: digest queue ready, {seeded} setting(s) seeded")


def _column_names(cursor, table):
    cursor.execute(f"PRAGMA table_info({table})")
    return {row["name"] for row in cursor.fetchall()}


def _has_digested_models(cursor) -> bool:
    cursor.execute("SELECT 1 FROM models WHERE sha256 IS NOT NULL LIMIT 1")
    return cursor.fetchone() is not None


def _seed_settings(cursor) -> int:
    policy = "background" if _has_digested_models(cursor) else "unset"
    wanted = (
        ("model_indexing_level", "low", "string", _LEVEL_DESCRIPTION),
        ("model_indexing_paused", "false", "boolean", _PAUSED_DESCRIPTION),
        ("model_digest_policy", policy, "string", _POLICY_DESCRIPTION),
    )
    seeded = 0
    for key, value, value_type, description in wanted:
        cursor.execute("SELECT 1 FROM settings WHERE key = ?", (key,))
        if cursor.fetchone() is not None:
            continue
        now = now_iso()
        cursor.execute(
            """
            INSERT INTO settings (id, key, value, value_type, description, type, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'SYSTEM', ?, ?)
            """,
            (generate_ulid(), key, value, value_type, description, now, now),
        )
        seeded += 1
    return seeded
