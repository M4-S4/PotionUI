import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from src.platform.database import get_cursor

_CHUNK = 500

ASSERTION_SOURCE_RANK = {"download": 1, "recipe": 2, "admin": 3}

_VERDICT_COLUMNS = (
    "format, status, model_type, family, variant, components, "
    "transformer_extractable, classifier, registry_fingerprint, reason, classified_at"
)
_PREFIXED_VERDICT_COLUMNS = ", ".join(f"v.{name.strip()}" for name in _VERDICT_COLUMNS.split(","))

def _chunks(values: Sequence[Any]) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), _CHUNK):
        yield values[start:start + _CHUNK]

def verdict_is_current(row: Dict[str, Any], fingerprint: str) -> bool:
    if row["registry_fingerprint"] == fingerprint:
        return True
    return row["status"] != "undecided" and not row["transformer_extractable"]

class ModelTypeRepository:

    def get_header_verdicts(self, header_hashes: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        unique = sorted({value for value in header_hashes if value})
        found: Dict[str, Dict[str, Any]] = {}
        if not unique:
            return found
        with get_cursor() as cursor:
            for chunk in _chunks(unique):
                placeholders = ",".join("?" for _ in chunk)
                cursor.execute(
                    f"SELECT header_hash, {_VERDICT_COLUMNS} FROM model_header_verdicts "
                    f"WHERE header_hash IN ({placeholders})",
                    tuple(chunk),
                )
                for row in cursor.fetchall():
                    found[row["header_hash"]] = dict(row)
        return found

    def get_verdicts(self, model_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        unique = sorted({value for value in model_ids if value})
        found: Dict[str, Dict[str, Any]] = {}
        if not unique:
            return found
        with get_cursor() as cursor:
            for chunk in _chunks(unique):
                placeholders = ",".join("?" for _ in chunk)
                cursor.execute(
                    f"SELECT model_id, {_VERDICT_COLUMNS} FROM model_verdicts WHERE model_id IN ({placeholders})",
                    tuple(chunk),
                )
                for row in cursor.fetchall():
                    found[row["model_id"]] = dict(row)
        return found

    def get_verdicts_by_sha(self, shas: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        unique = sorted({value for value in shas if value})
        found: Dict[str, Dict[str, Any]] = {}
        if not unique:
            return found
        with get_cursor() as cursor:
            for chunk in _chunks(unique):
                placeholders = ",".join("?" for _ in chunk)
                cursor.execute(
                    f"SELECT m.sha256 AS sha256, {_PREFIXED_VERDICT_COLUMNS} "
                    f"FROM model_verdicts v JOIN models m ON m.id = v.model_id WHERE m.sha256 IN ({placeholders})",
                    tuple(chunk),
                )
                for row in cursor.fetchall():
                    found[row["sha256"]] = dict(row)
        return found

    def settled_models(self, model_ids: Iterable[str], fingerprint: str) -> Set[str]:
        return {
            model_id
            for model_id, row in self.get_verdicts(model_ids).items()
            if verdict_is_current(row, fingerprint)
        }

    def upsert_verdict(
        self,
        *,
        header_hash: str,
        format: str,
        status: str,
        model_type: Optional[str],
        family: Optional[str],
        variant: Optional[str],
        components: Sequence[str],
        transformer_extractable: bool,
        classifier: Optional[str],
        registry_fingerprint: str,
        reason: Optional[str],
        classified_at: str,
    ) -> None:
        with get_cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO model_header_verdicts (header_hash, {_VERDICT_COLUMNS})
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(header_hash) DO UPDATE SET
                    format = excluded.format,
                    status = excluded.status,
                    model_type = excluded.model_type,
                    family = excluded.family,
                    variant = excluded.variant,
                    components = excluded.components,
                    transformer_extractable = excluded.transformer_extractable,
                    classifier = excluded.classifier,
                    registry_fingerprint = excluded.registry_fingerprint,
                    reason = excluded.reason,
                    classified_at = excluded.classified_at
                """,
                (
                    header_hash, format, status, model_type, family, variant, json.dumps(list(components)),
                    int(transformer_extractable), classifier, registry_fingerprint, reason, classified_at,
                ),
            )

    def get_assertion_rows(self, sha256: str) -> Dict[str, Dict[str, Any]]:
        with get_cursor() as cursor:
            cursor.execute(
                "SELECT sha256, model_type, source, set_by, set_at FROM model_type_assertions WHERE sha256 = ?",
                (sha256,),
            )
            return {row["source"]: dict(row) for row in cursor.fetchall()}

    def get_assertions(self, shas: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        unique = sorted({sha for sha in shas if sha})
        if not unique:
            return {}
        found: Dict[str, Dict[str, Any]] = {}
        with get_cursor() as cursor:
            for chunk in _chunks(unique):
                placeholders = ",".join("?" for _ in chunk)
                cursor.execute(
                    "SELECT sha256, model_type, source, set_by, set_at FROM model_type_assertions "
                    f"WHERE sha256 IN ({placeholders})",
                    tuple(chunk),
                )
                for row in cursor.fetchall():
                    best = found.get(row["sha256"])
                    if best is None or ASSERTION_SOURCE_RANK.get(row["source"], 0) > ASSERTION_SOURCE_RANK.get(
                        best["source"], 0
                    ):
                        found[row["sha256"]] = dict(row)
        return found

    def put_assertion(self, sha256: str, model_type: str, source: str, set_by: Optional[str], set_at: str) -> None:
        with get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO model_type_assertions (sha256, model_type, source, set_by, set_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(sha256, source) DO UPDATE SET "
                "model_type = excluded.model_type, set_by = excluded.set_by, set_at = excluded.set_at",
                (sha256, model_type, source, set_by, set_at),
            )

    def delete_assertion(self, sha256: str, source: str) -> bool:
        with get_cursor() as cursor:
            cursor.execute("DELETE FROM model_type_assertions WHERE sha256 = ? AND source = ?", (sha256, source))
            return cursor.rowcount > 0

    def models_by_ids(self, model_ids: Iterable[str]) -> List[Dict[str, Any]]:
        unique = sorted({model_id for model_id in model_ids if model_id})
        if not unique:
            return []
        rows: List[Dict[str, Any]] = []
        with get_cursor() as cursor:
            for chunk in _chunks(unique):
                placeholders = ",".join("?" for _ in chunk)
                cursor.execute(
                    "SELECT id, filename, sha256, model_type, type_source, is_directory FROM models "
                    f"WHERE id IN ({placeholders})",
                    tuple(chunk),
                )
                rows.extend(dict(row) for row in cursor.fetchall())
        return rows

    def present_copies(self, model_ids: Iterable[str]) -> List[Dict[str, Any]]:
        unique = sorted({model_id for model_id in model_ids if model_id})
        if not unique:
            return []
        rows: List[Dict[str, Any]] = []
        with get_cursor() as cursor:
            for chunk in _chunks(unique):
                placeholders = ",".join("?" for _ in chunk)
                cursor.execute(
                    f"""
                    SELECT ml.model_id AS model_id, ml.root_id AS root_id, ml.model_type AS model_type,
                           ml.rel_path AS rel_path, mrb.scan_headers AS scan_headers
                    FROM model_locations ml
                    JOIN model_root_bindings mrb ON mrb.id = ml.binding_id
                    WHERE ml.status = 'present' AND ml.model_id IN ({placeholders})
                    """,
                    tuple(chunk),
                )
                rows.extend(dict(row) for row in cursor.fetchall())
        return rows

    def identities_for_filenames(self, filenames: Iterable[str]) -> Dict[Tuple[str, str], str]:
        unique = sorted({name for name in filenames if name})
        if not unique:
            return {}
        found: Dict[Tuple[str, str], str] = {}
        with get_cursor() as cursor:
            for chunk in _chunks(unique):
                placeholders = ",".join("?" for _ in chunk)
                cursor.execute(
                    f"SELECT id, model_type, filename FROM models WHERE filename IN ({placeholders})",
                    tuple(chunk),
                )
                for row in cursor.fetchall():
                    found[(row["model_type"], row["filename"])] = row["id"]
        return found

    def apply_types(self, updates: Sequence[Tuple[str, str, str]]) -> None:
        if not updates:
            return
        with get_cursor() as cursor:
            cursor.executemany(
                "UPDATE models SET model_type = ?, type_source = ? WHERE id = ?",
                [(model_type, source, model_id) for model_id, model_type, source in updates],
            )

    def count_needs_type(self) -> int:
        with get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS n FROM models WHERE model_type = 'undefined'")
            return cursor.fetchone()["n"]

    def count_header_classified(self) -> int:
        with get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS n FROM models WHERE type_source = 'header'")
            return cursor.fetchone()["n"]
