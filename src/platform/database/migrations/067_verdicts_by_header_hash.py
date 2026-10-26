from src.platform.database.database import db

_NAME = "067_verdicts_by_header_hash"

_CREATE_VERDICTS = """
    CREATE TABLE model_header_verdicts_rebuilt (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sha256 TEXT UNIQUE,
        header_hash TEXT UNIQUE,
        format TEXT NOT NULL,
        status TEXT NOT NULL,
        model_type TEXT,
        family TEXT,
        variant TEXT,
        components TEXT NOT NULL DEFAULT '[]',
        transformer_extractable INTEGER NOT NULL DEFAULT 0,
        classifier TEXT,
        registry_fingerprint TEXT NOT NULL,
        reason TEXT,
        classified_at TIMESTAMP NOT NULL
    )
"""

_COPY_VERDICTS = """
    INSERT INTO model_header_verdicts_rebuilt
        (sha256, format, status, model_type, family, variant, components, transformer_extractable,
         classifier, registry_fingerprint, reason, classified_at)
    SELECT sha256, format, status, model_type, family, variant, components, transformer_extractable,
           classifier, registry_fingerprint, reason, classified_at
    FROM model_header_verdicts
"""

_CREATE_VIEW = """
    CREATE VIEW model_verdicts AS
    SELECT m.id AS model_id, v.format, v.status, v.model_type, v.family, v.variant, v.components,
           v.transformer_extractable, v.classifier, v.registry_fingerprint, v.reason, v.classified_at
    FROM models m
    JOIN model_header_verdicts v ON v.id = COALESCE(
        (SELECT hv.id
         FROM model_locations ml
         JOIN model_header_verdicts hv ON hv.header_hash = ml.header_hash
         WHERE ml.model_id = m.id
         ORDER BY (ml.status = 'present') DESC, ml.id
         LIMIT 1),
        (SELECT sv.id FROM model_header_verdicts sv WHERE sv.sha256 = m.sha256)
    )
"""

_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_model_locations_header_hash ON model_locations (header_hash)",
)


def up():
    with db.get_cursor() as cursor:
        cursor.execute("DROP VIEW IF EXISTS model_verdicts")
        if "header_hash" not in _column_names(cursor, "model_locations"):
            cursor.execute("ALTER TABLE model_locations ADD COLUMN header_hash TEXT")
        carried = 0
        if "header_hash" not in _column_names(cursor, "model_header_verdicts"):
            cursor.execute("DROP TABLE IF EXISTS model_header_verdicts_rebuilt")
            cursor.execute(_CREATE_VERDICTS)
            cursor.execute(_COPY_VERDICTS)
            carried = cursor.rowcount
            cursor.execute("DROP TABLE model_header_verdicts")
            cursor.execute("ALTER TABLE model_header_verdicts_rebuilt RENAME TO model_header_verdicts")
        for statement in _INDEXES:
            cursor.execute(statement)
        cursor.execute(_CREATE_VIEW)
    print(f"Migration {_NAME}: verdicts can be keyed by header hash, {carried} existing verdict(s) kept by digest")


def _column_names(cursor, table):
    cursor.execute(f"PRAGMA table_info({table})")
    return {row["name"] for row in cursor.fetchall()}
