import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.features.models.repository import model_repo
from src.platform.util.ids import generate_ulid
from src.platform.database import get_cursor

_LOCATION_COLUMNS = (
    "id, model_id, root_id, model_type, rel_path, rel_key, size, mtime_ns, sha256, status, seen_at, binding_id"
)

_MODEL_CHUNK = 500

DIGEST_STALE = "stale"
DIGEST_RECORDED = "recorded"
DIGEST_CONFLICT = "conflict"
DIGEST_DUPLICATE = "duplicate"
DIGEST_MERGED = "merged"


@dataclass(frozen=True)
class DigestApplication:
    outcome: str
    model_id: Optional[str] = None
    model_filename: Optional[str] = None
    owner_model_id: Optional[str] = None
    owner_filename: Optional[str] = None
    merged_from: Optional[str] = None


def _pick_winner(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    return min(rows, key=lambda row: (row["position"], len(row["rel_path"]), row["rel_path"]))


class ModelLocationsRepository:

    def get(self, binding_id: str, rel_key: str) -> Optional[Dict[str, Any]]:
        with get_cursor() as cursor:
            cursor.execute(
                f"SELECT {_LOCATION_COLUMNS} FROM model_locations WHERE binding_id = ? AND rel_key = ?",
                (binding_id, rel_key),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_for_binding(self, binding_id: str) -> List[Dict[str, Any]]:
        with get_cursor() as cursor:
            cursor.execute(
                f"SELECT {_LOCATION_COLUMNS} FROM model_locations WHERE binding_id = ?",
                (binding_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def list_for_root_type(self, root_id: str, model_type: str) -> List[Dict[str, Any]]:
        with get_cursor() as cursor:
            cursor.execute(
                f"SELECT {_LOCATION_COLUMNS} FROM model_locations WHERE root_id = ? AND model_type = ?",
                (root_id, model_type),
            )
            return [dict(row) for row in cursor.fetchall()]

    def type_mismatches(self, binding_type: str, rel_paths: Sequence[str]) -> List[Dict[str, Any]]:
        unique = sorted({path for path in rel_paths if path})
        if not unique:
            return []
        found: List[Dict[str, Any]] = []
        with get_cursor() as cursor:
            for start in range(0, len(unique), 500):
                chunk = unique[start:start + 500]
                placeholders = ",".join("?" for _ in chunk)
                cursor.execute(
                    "SELECT ml.rel_path AS rel_path, ml.size AS size, m.id AS model_id, "
                    "m.model_type AS model_type, m.filename AS filename, m.file_size AS file_size "
                    "FROM model_locations ml JOIN models m ON m.id = ml.model_id "
                    "WHERE ml.status = 'present' AND ml.model_type = ? AND m.model_type != ? "
                    f"AND ml.rel_path IN ({placeholders})",
                    (binding_type, binding_type, *chunk),
                )
                found.extend(dict(row) for row in cursor.fetchall())
        return found

    def list_for_model(self, model_id: str) -> List[Dict[str, Any]]:
        with get_cursor() as cursor:
            cursor.execute(
                f"SELECT {_LOCATION_COLUMNS} FROM model_locations WHERE model_id = ?",
                (model_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def upsert(
        self,
        *,
        model_id: str,
        binding_id: str,
        root_id: str,
        model_type: str,
        rel_path: str,
        rel_key: str,
        size: Optional[int],
        mtime_ns: Optional[int],
        sha256: Optional[str],
        status: str,
        seen_at: str,
        quick_sig: Optional[str] = None,
    ) -> str:
        location_id = generate_ulid()
        with get_cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO model_locations
                    (id, model_id, binding_id, root_id, model_type, rel_path, rel_key, size, mtime_ns, sha256,
                     status, seen_at, quick_sig)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(binding_id, rel_key) DO UPDATE SET
                    model_id = excluded.model_id,
                    rel_path = excluded.rel_path,
                    size = excluded.size,
                    mtime_ns = excluded.mtime_ns,
                    sha256 = excluded.sha256,
                    quick_sig = excluded.quick_sig,
                    status = excluded.status,
                    seen_at = excluded.seen_at
                """,
                (
                    location_id, model_id, binding_id, root_id, model_type, rel_path, rel_key,
                    size, mtime_ns, sha256, status, seen_at, quick_sig,
                ),
            )
            cursor.execute(
                "SELECT id FROM model_locations WHERE binding_id = ? AND rel_key = ?",
                (binding_id, rel_key),
            )
            row = cursor.fetchone()
            return row["id"] if row else location_id

    def confirm_present(
        self,
        binding_id: str,
        touched: Sequence[str],
        adopted: Sequence[Tuple[str, int]],
        seen_at: str,
    ) -> None:
        if not touched and not adopted:
            return
        with get_cursor() as cursor:
            cursor.executemany(
                "UPDATE model_locations SET status = 'present', mtime_ns = ?, seen_at = ? "
                "WHERE binding_id = ? AND rel_key = ? AND status = 'present'",
                [(mtime_ns, seen_at, binding_id, rel_key) for rel_key, mtime_ns in adopted],
            )
            cursor.executemany(
                "UPDATE model_locations SET status = 'present', seen_at = ? "
                "WHERE binding_id = ? AND rel_key = ? AND status = 'present'",
                [(seen_at, binding_id, rel_key) for rel_key in touched],
            )

    def mark_missing_for_binding(
        self, binding_id: str, present_rel_keys: Iterable[str], seen_at: str
    ) -> int:
        with get_cursor() as cursor:
            cursor.execute(
                "UPDATE model_locations SET status = 'missing', seen_at = ? "
                "WHERE binding_id = ? AND status != 'missing' "
                "AND rel_key NOT IN (SELECT value FROM json_each(?))",
                (seen_at, binding_id, json.dumps(list(present_rel_keys))),
            )
            return cursor.rowcount

    def delete_missing_for_roots(self, root_ids: Sequence[str]) -> int:
        if not root_ids:
            return 0
        with get_cursor() as cursor:
            placeholders = ",".join("?" for _ in root_ids)
            cursor.execute(
                f"DELETE FROM model_locations WHERE status = 'missing' AND root_id IN ({placeholders})",
                tuple(root_ids),
            )
            return cursor.rowcount

    def delete(self, location_id: str) -> bool:
        with get_cursor() as cursor:
            cursor.execute("DELETE FROM model_locations WHERE id = ?", (location_id,))
            return cursor.rowcount > 0

    def list_present_by_size(self, size: int) -> List[Dict[str, Any]]:
        with get_cursor() as cursor:
            cursor.execute(
                f"SELECT {_LOCATION_COLUMNS}, quick_sig FROM model_locations WHERE size = ? AND status = 'present'",
                (size,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def list_missing_digested_by_size(self, size: int) -> List[Dict[str, Any]]:
        with get_cursor() as cursor:
            cursor.execute(
                f"SELECT {_LOCATION_COLUMNS}, quick_sig FROM model_locations "
                "WHERE size = ? AND status = 'missing' AND sha256 IS NOT NULL",
                (size,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_by_id(self, location_id: str) -> Optional[Dict[str, Any]]:
        with get_cursor() as cursor:
            cursor.execute(
                f"SELECT {_LOCATION_COLUMNS}, quick_sig FROM model_locations WHERE id = ?", (location_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def apply_digest_and_dequeue(
        self, location_id: str, sha256: str, quick_sig: Optional[str], file_size: Optional[int]
    ) -> DigestApplication:
        with get_cursor() as cursor:
            cursor.execute("SELECT model_id, status, sha256 FROM model_locations WHERE id = ?", (location_id,))
            location = cursor.fetchone()
            if location is None or location["status"] != "present" or location["sha256"] is not None:
                cursor.execute("DELETE FROM model_digest_queue WHERE location_id = ?", (location_id,))
                return DigestApplication(DIGEST_STALE)
            cursor.execute(
                "UPDATE model_locations SET sha256 = ?, quick_sig = ? WHERE id = ?", (sha256, quick_sig, location_id)
            )
            result = self._settle_model_digest(cursor, location_id, location["model_id"], sha256, file_size)
            cursor.execute("DELETE FROM model_digest_queue WHERE location_id = ?", (location_id,))
            return result

    @staticmethod
    def _settle_model_digest(
        cursor, location_id: str, model_id: str, sha256: str, file_size: Optional[int]
    ) -> DigestApplication:
        cursor.execute("SELECT filename, sha256 FROM models WHERE id = ?", (model_id,))
        model = cursor.fetchone()
        if model is None or model["sha256"] == sha256:
            return DigestApplication(DIGEST_RECORDED, model_id)
        filename = model["filename"]
        cursor.execute(
            "SELECT COUNT(*) AS n FROM model_locations WHERE model_id = ? AND status = 'present'", (model_id,)
        )
        if model["sha256"] is not None and cursor.fetchone()["n"] > 1:
            cursor.execute("UPDATE model_locations SET status = 'conflict' WHERE id = ?", (location_id,))
            return DigestApplication(DIGEST_CONFLICT, model_id, filename)
        if model_repo.update_digest(model_id, sha256=sha256, file_size=file_size, cursor=cursor):
            return DigestApplication(DIGEST_RECORDED, model_id, filename)
        cursor.execute("SELECT id, filename, model_type FROM models WHERE sha256 = ? AND id != ?", (sha256, model_id))
        owner = cursor.fetchone()
        if owner is not None and ModelLocationsRepository._merge_into_owner(
            cursor, model_id, model, owner, file_size
        ):
            return DigestApplication(DIGEST_MERGED, owner["id"], filename, merged_from=model_id)
        return DigestApplication(
            DIGEST_DUPLICATE, model_id, filename,
            owner["id"] if owner is not None else None, owner["filename"] if owner is not None else None,
        )

    @staticmethod
    def _merge_into_owner(cursor, model_id: str, model, owner, file_size: Optional[int]) -> bool:
        if model["sha256"] is not None:
            return False
        cursor.execute("SELECT COUNT(*) AS n FROM model_locations WHERE model_id = ?", (model_id,))
        if cursor.fetchone()["n"] != 1:
            return False
        cursor.execute(
            "SELECT COUNT(*) AS n FROM model_locations WHERE model_id = ? AND status = 'present'", (owner["id"],)
        )
        if cursor.fetchone()["n"] != 0:
            return False
        return model_repo.absorb(cursor, model_id, owner["id"], model["filename"], file_size)

    def list_present_for_roots(
        self, root_ids: Sequence[str], model_ids: Optional[Sequence[str]] = None
    ) -> List[Dict[str, Any]]:
        if not root_ids:
            return []
        placeholders = ",".join("?" for _ in root_ids)
        select = f"""
                SELECT ml.id, ml.model_id, ml.binding_id, ml.root_id, ml.model_type, ml.rel_path, ml.rel_key,
                       ml.size, ml.mtime_ns, ml.sha256, ml.status,
                       mrb.position AS position, mrb.subdir AS subdir
                FROM model_locations ml
                JOIN model_root_bindings mrb ON mrb.id = ml.binding_id
                WHERE ml.status = 'present' AND ml.root_id IN ({placeholders})
                """
        with get_cursor() as cursor:
            if model_ids is None:
                cursor.execute(select, tuple(root_ids))
                return [dict(row) for row in cursor.fetchall()]
            rows: List[Dict[str, Any]] = []
            wanted = sorted({model_id for model_id in model_ids if model_id})
            for start in range(0, len(wanted), _MODEL_CHUNK):
                chunk = wanted[start:start + _MODEL_CHUNK]
                marks = ",".join("?" for _ in chunk)
                cursor.execute(f"{select} AND ml.model_id IN ({marks})", (*root_ids, *chunk))
                rows.extend(dict(row) for row in cursor.fetchall())
            return rows

    def winners_by_model(
        self, root_ids: Sequence[str], model_ids: Optional[Sequence[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        by_model: Dict[str, List[Dict[str, Any]]] = {}
        for row in self.list_present_for_roots(root_ids, model_ids):
            by_model.setdefault(row["model_id"], []).append(row)
        return {model_id: _pick_winner(rows) for model_id, rows in by_model.items()}

    def list_conflicts(self, limit: int = 200) -> List[Dict[str, Any]]:
        with get_cursor() as cursor:
            cursor.execute(
                "SELECT ml.id, ml.model_id, ml.binding_id, ml.root_id, ml.model_type, ml.rel_path, ml.rel_key, "
                "ml.size, ml.mtime_ns, ml.sha256, ml.status, ml.seen_at, mr.label AS root_label "
                "FROM model_locations ml JOIN model_roots mr ON mr.id = ml.root_id "
                "WHERE ml.status = 'conflict' ORDER BY ml.seen_at DESC LIMIT ?",
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def list_duplicate_models(self, limit: int = 200) -> List[Dict[str, Any]]:
        from pathlib import PurePosixPath

        with get_cursor() as cursor:
            cursor.execute(
                "SELECT ml.model_id, ml.binding_id, ml.model_type, ml.root_id, ml.rel_path, "
                "mr.label AS root_label, mrb.position AS position "
                "FROM model_locations ml "
                "JOIN model_roots mr ON mr.id = ml.root_id "
                "JOIN model_root_bindings mrb ON mrb.id = ml.binding_id "
                "WHERE ml.status = 'present' AND ml.model_id IN ("
                "  SELECT model_id FROM model_locations WHERE status = 'present' "
                "  GROUP BY model_id HAVING COUNT(*) > 1"
                ") ORDER BY ml.model_id, mrb.position",
                (),
            )
            rows = [dict(row) for row in cursor.fetchall()]

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["model_id"], []).append(row)

        entries = []
        for group in grouped.values():
            winner = _pick_winner(group)
            filename = PurePosixPath(winner["rel_path"]).name
            entries.append({
                "model_type": group[0]["model_type"],
                "filename": filename,
                "copies": [
                    {
                        "root_label": row["root_label"],
                        "rel_path": row["rel_path"],
                        "winner": row is winner,
                    }
                    for row in group
                ],
            })

        entries.sort(key=lambda entry: len(entry["copies"]), reverse=True)
        return entries[:limit]

    def aggregate_by_binding(self) -> Dict[str, Dict[str, int]]:
        with get_cursor() as cursor:
            cursor.execute(
                """
                SELECT binding_id, COUNT(*) AS indexed_files, COALESCE(SUM(size), 0) AS size_bytes
                FROM model_locations
                WHERE status = 'present'
                GROUP BY binding_id
                """
            )
            return {
                row["binding_id"]: {
                    "indexed_files": row["indexed_files"],
                    "size_bytes": row["size_bytes"],
                }
                for row in cursor.fetchall()
            }

    def indexed_files_by_root(self) -> Dict[str, int]:
        with get_cursor() as cursor:
            cursor.execute(
                "SELECT root_id, COUNT(*) AS indexed_files FROM model_locations "
                "WHERE status = 'present' GROUP BY root_id"
            )
            return {row["root_id"]: row["indexed_files"] for row in cursor.fetchall()}

    def delete_for_root_and_type(self, root_id: str, model_type: str) -> None:
        with get_cursor() as cursor:
            cursor.execute(
                "DELETE FROM model_locations WHERE root_id = ? AND model_type = ?",
                (root_id, model_type),
            )

    def rehome(self, src_root_id: str, model_type: str, dst_root_id: str, dst_rel_prefix: str = "") -> int:
        from pathlib import PurePosixPath

        with get_cursor() as cursor:
            cursor.execute(
                "SELECT id FROM model_root_bindings WHERE root_id = ? AND model_type = ? ORDER BY position LIMIT 1",
                (dst_root_id, model_type),
            )
            destination = cursor.fetchone()
            if destination is None:
                return 0
            dst_binding_id = destination["id"]

            cursor.execute(
                f"SELECT {_LOCATION_COLUMNS} FROM model_locations WHERE root_id = ? AND model_type = ?",
                (src_root_id, model_type),
            )
            rows = [dict(row) for row in cursor.fetchall()]

            moved = 0
            for row in rows:
                if dst_rel_prefix:
                    new_rel_path = str(PurePosixPath(dst_rel_prefix) / row["rel_path"])
                    new_rel_key = str(PurePosixPath(dst_rel_prefix) / row["rel_key"])
                else:
                    new_rel_path = row["rel_path"]
                    new_rel_key = row["rel_key"]

                cursor.execute(
                    "SELECT id FROM model_locations WHERE binding_id = ? AND rel_key = ?",
                    (dst_binding_id, new_rel_key),
                )
                conflict = cursor.fetchone()
                if conflict is not None:
                    cursor.execute("DELETE FROM model_locations WHERE id = ?", (row["id"],))
                else:
                    cursor.execute(
                        "UPDATE model_locations SET root_id = ?, binding_id = ?, rel_path = ?, rel_key = ?, "
                        "mtime_ns = NULL WHERE id = ?",
                        (dst_root_id, dst_binding_id, new_rel_path, new_rel_key, row["id"]),
                    )
                moved += 1
            return moved
