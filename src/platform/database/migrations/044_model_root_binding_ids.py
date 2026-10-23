from src.platform.database.database import db
from src.platform.filesystem.model_roots import binding_subdir_key
from src.platform.util.ids import generate_ulid

_NAME = "044_model_root_binding_ids"

_LOCATION_COLUMNS = "id, model_id, root_id, model_type, rel_path, rel_key, size, mtime_ns, sha256, status, seen_at"

_CREATE_BINDINGS = (
    "CREATE TABLE model_root_bindings_new ("
    "id TEXT PRIMARY KEY, "
    "root_id TEXT NOT NULL REFERENCES model_roots(id) ON DELETE CASCADE, "
    "model_type TEXT NOT NULL, "
    "subdir TEXT NOT NULL, "
    "subdir_key TEXT NOT NULL, "
    "position INTEGER NOT NULL, "
    "is_write INTEGER NOT NULL DEFAULT 0, "
    "scan_headers INTEGER NOT NULL DEFAULT 0, "
    "UNIQUE (root_id, model_type, subdir_key))"
)

_CREATE_LOCATIONS = (
    "CREATE TABLE model_locations_new ("
    "id TEXT PRIMARY KEY, "
    "model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE, "
    "root_id TEXT NOT NULL REFERENCES model_roots(id) ON DELETE CASCADE, "
    "model_type TEXT NOT NULL, "
    "rel_path TEXT NOT NULL, "
    "rel_key TEXT NOT NULL, "
    "size INTEGER, "
    "mtime_ns INTEGER, "
    "sha256 TEXT, "
    "status TEXT NOT NULL DEFAULT 'present', "
    "seen_at TIMESTAMP, "
    "binding_id TEXT NOT NULL REFERENCES model_root_bindings(id) ON DELETE CASCADE, "
    "UNIQUE (binding_id, rel_key))"
)

_CREATE_OLD_BINDINGS = (
    "CREATE TABLE model_root_bindings_old ("
    "root_id TEXT NOT NULL REFERENCES model_roots(id) ON DELETE CASCADE, "
    "model_type TEXT NOT NULL, "
    "subdir TEXT NOT NULL, "
    "position INTEGER NOT NULL, "
    "is_write INTEGER NOT NULL DEFAULT 0, "
    "scan_headers INTEGER NOT NULL DEFAULT 0, "
    "PRIMARY KEY (root_id, model_type))"
)

_CREATE_OLD_LOCATIONS = (
    "CREATE TABLE model_locations_old ("
    "id TEXT PRIMARY KEY, "
    "model_id TEXT NOT NULL REFERENCES models(id) ON DELETE CASCADE, "
    "root_id TEXT NOT NULL REFERENCES model_roots(id) ON DELETE CASCADE, "
    "model_type TEXT NOT NULL, "
    "rel_path TEXT NOT NULL, "
    "rel_key TEXT NOT NULL, "
    "size INTEGER, "
    "mtime_ns INTEGER, "
    "sha256 TEXT, "
    "status TEXT NOT NULL DEFAULT 'present', "
    "seen_at TIMESTAMP, "
    "UNIQUE (root_id, model_type, rel_key))"
)


def up():
    with db.get_cursor() as cursor:
        if "id" in _column_names(cursor, "model_root_bindings"):
            _add_layout_profile(cursor)
            print(f"Migration {_NAME}: bindings already have ids")
            return
        dropped = _rebuild_forward(cursor)
        _add_layout_profile(cursor)
    print(f"Migration {_NAME}: bindings and locations are keyed by binding id, {dropped} orphan location(s) dropped")


def _rebuild_forward(cursor) -> int:
    cursor.execute("SELECT id, case_insensitive FROM model_roots")
    case_insensitive = {row["id"]: bool(row["case_insensitive"]) for row in cursor.fetchall()}

    cursor.execute("DROP TABLE IF EXISTS model_root_bindings_new")
    cursor.execute("DROP TABLE IF EXISTS model_locations_new")
    cursor.execute(_CREATE_BINDINGS)
    cursor.execute("SELECT root_id, model_type, subdir, position, is_write, scan_headers FROM model_root_bindings")
    rows = [
        (
            generate_ulid(),
            row["root_id"],
            row["model_type"],
            row["subdir"],
            binding_subdir_key(row["subdir"], case_insensitive=case_insensitive.get(row["root_id"], False)),
            row["position"],
            row["is_write"],
            row["scan_headers"],
        )
        for row in cursor.fetchall()
    ]
    cursor.executemany(
        "INSERT INTO model_root_bindings_new "
        "(id, root_id, model_type, subdir, subdir_key, position, is_write, scan_headers) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    cursor.execute("DROP TABLE model_root_bindings")
    cursor.execute("ALTER TABLE model_root_bindings_new RENAME TO model_root_bindings")
    _create_binding_indexes(cursor)

    cursor.execute(_CREATE_LOCATIONS)
    columns = ", ".join(f"ml.{name.strip()}" for name in _LOCATION_COLUMNS.split(","))
    cursor.execute(
        f"INSERT INTO model_locations_new ({_LOCATION_COLUMNS}, binding_id) "
        f"SELECT {columns}, b.id FROM model_locations ml "
        "JOIN model_root_bindings b ON b.root_id = ml.root_id AND b.model_type = ml.model_type"
    )
    cursor.execute("SELECT COUNT(*) AS n FROM model_locations")
    before = cursor.fetchone()["n"]
    cursor.execute("SELECT COUNT(*) AS n FROM model_locations_new")
    after = cursor.fetchone()["n"]
    cursor.execute("DROP TABLE model_locations")
    cursor.execute("ALTER TABLE model_locations_new RENAME TO model_locations")
    _create_location_indexes(cursor)
    return before - after


def _create_binding_indexes(cursor) -> None:
    cursor.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_binding_type_position ON model_root_bindings (model_type, position)"
    )
    cursor.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_binding_write_root ON model_root_bindings (model_type) "
        "WHERE is_write = 1"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_bindings_root_type ON model_root_bindings (root_id, model_type)"
    )


def _create_location_indexes(cursor) -> None:
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_locations_model ON model_locations (model_id)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_model_locations_root_type ON model_locations (root_id, model_type)"
    )


def _add_layout_profile(cursor) -> None:
    if "layout_profile" not in _column_names(cursor, "model_roots"):
        cursor.execute("ALTER TABLE model_roots ADD COLUMN layout_profile TEXT")


def _column_names(cursor, table):
    cursor.execute(f"PRAGMA table_info({table})")
    return {row["name"] for row in cursor.fetchall()}
