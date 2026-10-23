from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.platform.database import get_cursor

PRIORITY_BACKGROUND = 0
PRIORITY_CHOSEN = 1
PRIORITY_TWIN = 2
PRIORITY_DEMAND = 3

MAX_ATTEMPTS = 3

_CHUNK = 500

_JOB_COLUMNS = (
    "q.location_id AS location_id, q.priority AS priority, q.reason AS reason, q.attempts AS attempts, "
    "q.last_error AS last_error, q.not_before AS not_before, "
    "ml.model_id AS model_id, ml.root_id AS root_id, ml.model_type AS model_type, ml.rel_path AS rel_path, "
    "ml.binding_id AS binding_id, ml.size AS size, ml.mtime_ns AS mtime_ns, "
    "m.filename AS filename, m.is_directory AS is_directory"
)

_PENDING = "ml.status = 'present' AND ml.sha256 IS NULL"


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _pending_filter(query: str, model_type: Optional[str], root_id: Optional[str]):
    clauses = [_PENDING]
    params: List[Any] = []
    needle = (query or "").strip()
    if needle:
        clauses.append("m.filename LIKE ? ESCAPE '\\'")
        params.append(f"%{_escape_like(needle)}%")
    if model_type:
        clauses.append("m.model_type = ?")
        params.append(model_type)
    if root_id:
        clauses.append("ml.root_id = ?")
        params.append(root_id)
    return " AND ".join(clauses), tuple(params)


def _chunks(values: Sequence[Any]) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), _CHUNK):
        yield values[start:start + _CHUNK]


class DigestQueueRepository:

    def enqueue(self, location_ids: Sequence[str], priority: int, reason: str, queued_at: str) -> int:
        unique = sorted({location_id for location_id in location_ids if location_id})
        added = 0
        with get_cursor() as cursor:
            for chunk in _chunks(unique):
                placeholders = ",".join("?" for _ in chunk)
                cursor.execute(
                    f"""
                    INSERT INTO model_digest_queue (location_id, priority, reason, size, queued_at)
                    SELECT ml.id, ?, ?, COALESCE(ml.size, 0), ?
                    FROM model_locations ml
                    WHERE {_PENDING} AND ml.id IN ({placeholders})
                    ON CONFLICT(location_id) DO UPDATE SET
                        reason = CASE WHEN excluded.priority > model_digest_queue.priority
                                      THEN excluded.reason ELSE model_digest_queue.reason END,
                        priority = MAX(model_digest_queue.priority, excluded.priority),
                        not_before = CASE WHEN excluded.priority > model_digest_queue.priority
                                          THEN NULL ELSE model_digest_queue.not_before END
                    """,
                    (priority, reason, queued_at, *chunk),
                )
                added += max(cursor.rowcount, 0)
        return added

    def enqueue_all_pending(self, priority: int, reason: str, queued_at: str) -> int:
        with get_cursor() as cursor:
            cursor.execute(
                f"""
                INSERT OR IGNORE INTO model_digest_queue (location_id, priority, reason, size, queued_at)
                SELECT ml.id, ?, ?, COALESCE(ml.size, 0), ?
                FROM model_locations ml
                WHERE {_PENDING}
                """,
                (priority, reason, queued_at),
            )
            return max(cursor.rowcount, 0)

    def remove(self, location_ids: Sequence[str]) -> int:
        unique = sorted({location_id for location_id in location_ids if location_id})
        removed = 0
        with get_cursor() as cursor:
            for chunk in _chunks(unique):
                placeholders = ",".join("?" for _ in chunk)
                cursor.execute(f"DELETE FROM model_digest_queue WHERE location_id IN ({placeholders})", tuple(chunk))
                removed += cursor.rowcount
        return removed

    def delete_stale(self) -> List[str]:
        with get_cursor() as cursor:
            cursor.execute(
                """
                SELECT q.location_id AS location_id FROM model_digest_queue q
                LEFT JOIN model_locations ml ON ml.id = q.location_id
                WHERE ml.id IS NULL OR ml.status != 'present' OR ml.sha256 IS NOT NULL
                """
            )
            stale = [row["location_id"] for row in cursor.fetchall()]
        self.remove(stale)
        return stale

    def get(self, location_id: str) -> Optional[Dict[str, Any]]:
        with get_cursor() as cursor:
            cursor.execute(
                f"""
                SELECT {_JOB_COLUMNS}
                FROM model_digest_queue q
                JOIN model_locations ml ON ml.id = q.location_id
                JOIN models m ON m.id = ml.model_id
                WHERE q.location_id = ?
                """,
                (location_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def next_jobs(self, limit: int, now: float, exclude: Iterable[str] = ()) -> List[Dict[str, Any]]:
        excluded = sorted(set(exclude))
        exclusion = f"AND q.location_id NOT IN ({','.join('?' for _ in excluded)})" if excluded else ""
        with get_cursor() as cursor:
            cursor.execute(
                f"""
                SELECT {_JOB_COLUMNS}
                FROM model_digest_queue q
                JOIN model_locations ml ON ml.id = q.location_id
                JOIN models m ON m.id = ml.model_id
                WHERE {_PENDING} AND q.attempts < ? AND (q.not_before IS NULL OR q.not_before <= ?) {exclusion}
                ORDER BY q.priority DESC, q.size ASC, q.queued_at ASC, q.location_id ASC
                LIMIT ?
                """,
                (MAX_ATTEMPTS, now, *excluded, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    def record_failure(self, location_id: str, error: str, now: float, retry_delays: Sequence[float]) -> int:
        with get_cursor() as cursor:
            cursor.execute("SELECT attempts FROM model_digest_queue WHERE location_id = ?", (location_id,))
            row = cursor.fetchone()
            if row is None:
                return 0
            attempts = row["attempts"] + 1
            not_before = None
            if attempts < MAX_ATTEMPTS and retry_delays:
                not_before = now + retry_delays[min(attempts - 1, len(retry_delays) - 1)]
            cursor.execute(
                "UPDATE model_digest_queue SET attempts = ?, last_error = ?, not_before = ? WHERE location_id = ?",
                (attempts, error[:500], not_before, location_id),
            )
            return attempts

    def defer(self, location_id: str, not_before: float) -> None:
        with get_cursor() as cursor:
            cursor.execute("UPDATE model_digest_queue SET not_before = ? WHERE location_id = ?", (not_before, location_id))

    def retry_failed(self, location_ids: Optional[Sequence[str]] = None) -> int:
        with get_cursor() as cursor:
            if location_ids is None:
                cursor.execute(
                    "UPDATE model_digest_queue SET attempts = 0, last_error = NULL, not_before = NULL WHERE attempts >= ?",
                    (MAX_ATTEMPTS,),
                )
                return cursor.rowcount
            retried = 0
            for chunk in _chunks(sorted(set(location_ids))):
                placeholders = ",".join("?" for _ in chunk)
                cursor.execute(
                    "UPDATE model_digest_queue SET attempts = 0, last_error = NULL, not_before = NULL "
                    f"WHERE attempts >= ? AND location_id IN ({placeholders})",
                    (MAX_ATTEMPTS, *chunk),
                )
                retried += cursor.rowcount
            return retried

    def list_failed(self, limit: int = 50) -> List[Dict[str, Any]]:
        with get_cursor() as cursor:
            cursor.execute(
                f"""
                SELECT {_JOB_COLUMNS}
                FROM model_digest_queue q
                JOIN model_locations ml ON ml.id = q.location_id
                JOIN models m ON m.id = ml.model_id
                WHERE q.attempts >= ?
                ORDER BY q.queued_at ASC, q.location_id ASC
                LIMIT ?
                """,
                (MAX_ATTEMPTS, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    def queue_stats(self) -> Dict[str, int]:
        with get_cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    SUM(CASE WHEN q.attempts < ? THEN 1 ELSE 0 END) AS queued,
                    COALESCE(SUM(CASE WHEN q.attempts < ? THEN ml.size ELSE 0 END), 0) AS queued_bytes,
                    SUM(CASE WHEN q.attempts >= ? THEN 1 ELSE 0 END) AS failed,
                    COALESCE(SUM(CASE WHEN q.attempts >= ? THEN ml.size ELSE 0 END), 0) AS failed_bytes
                FROM model_digest_queue q
                JOIN model_locations ml ON ml.id = q.location_id
                WHERE ml.status = 'present' AND ml.sha256 IS NULL
                """,
                (MAX_ATTEMPTS, MAX_ATTEMPTS, MAX_ATTEMPTS, MAX_ATTEMPTS),
            )
            row = cursor.fetchone()
            return {
                "queued": row["queued"] or 0,
                "queued_bytes": row["queued_bytes"] or 0,
                "failed": row["failed"] or 0,
                "failed_bytes": row["failed_bytes"] or 0,
            }

    def digest_stats(self) -> Dict[str, int]:
        with get_cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    SUM(CASE WHEN ml.sha256 IS NULL THEN 1 ELSE 0 END) AS pending,
                    COALESCE(SUM(CASE WHEN ml.sha256 IS NULL THEN ml.size ELSE 0 END), 0) AS pending_bytes,
                    SUM(CASE WHEN ml.sha256 IS NOT NULL THEN 1 ELSE 0 END) AS ready,
                    COALESCE(SUM(CASE WHEN ml.sha256 IS NOT NULL THEN ml.size ELSE 0 END), 0) AS ready_bytes
                FROM model_locations ml
                WHERE ml.status = 'present'
                """
            )
            row = cursor.fetchone()
            return {
                "pending": row["pending"] or 0,
                "pending_bytes": row["pending_bytes"] or 0,
                "ready": row["ready"] or 0,
                "ready_bytes": row["ready_bytes"] or 0,
            }

    def attempts_among(self, location_ids: Sequence[str]) -> Dict[str, int]:
        unique = sorted({location_id for location_id in location_ids if location_id})
        attempts: Dict[str, int] = {}
        with get_cursor() as cursor:
            for chunk in _chunks(unique):
                placeholders = ",".join("?" for _ in chunk)
                cursor.execute(
                    f"SELECT location_id, attempts FROM model_digest_queue WHERE location_id IN ({placeholders})",
                    tuple(chunk),
                )
                attempts.update({row["location_id"]: row["attempts"] for row in cursor.fetchall()})
        return attempts

    def waiting_among(self, location_ids: Sequence[str], now: Optional[float] = None) -> List[str]:
        unique = sorted({location_id for location_id in location_ids if location_id})
        waiting: List[str] = []
        ready = "" if now is None else "AND (q.not_before IS NULL OR q.not_before <= ?)"
        extra = () if now is None else (now,)
        with get_cursor() as cursor:
            for chunk in _chunks(unique):
                placeholders = ",".join("?" for _ in chunk)
                cursor.execute(
                    f"""
                    SELECT q.location_id AS location_id FROM model_digest_queue q
                    JOIN model_locations ml ON ml.id = q.location_id
                    WHERE {_PENDING} AND q.attempts < ? {ready} AND q.location_id IN ({placeholders})
                    """,
                    (MAX_ATTEMPTS, *extra, *chunk),
                )
                waiting.extend(row["location_id"] for row in cursor.fetchall())
        return waiting

    def list_pending(
        self,
        query: str = "",
        model_type: Optional[str] = None,
        root_id: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        where, params = _pending_filter(query, model_type, root_id)
        with get_cursor() as cursor:
            cursor.execute(
                f"""
                SELECT ml.id AS location_id, ml.model_id AS model_id, m.filename AS filename,
                       m.model_type AS model_type, COALESCE(ml.size, 0) AS size, r.label AS root_label,
                       ml.rel_path AS rel_path, (q.location_id IS NOT NULL AND q.attempts < ?) AS queued
                FROM model_locations ml
                JOIN models m ON m.id = ml.model_id
                JOIN model_roots r ON r.id = ml.root_id
                LEFT JOIN model_digest_queue q ON q.location_id = ml.id
                WHERE {where}
                ORDER BY lower(m.filename), ml.id
                LIMIT ? OFFSET ?
                """,
                (MAX_ATTEMPTS, *params, limit, offset),
            )
            return [{**dict(row), "queued": bool(row["queued"])} for row in cursor.fetchall()]

    def count_pending(
        self, query: str = "", model_type: Optional[str] = None, root_id: Optional[str] = None
    ) -> Dict[str, int]:
        where, params = _pending_filter(query, model_type, root_id)
        with get_cursor() as cursor:
            cursor.execute(
                f"""
                SELECT COUNT(*) AS total, COALESCE(SUM(ml.size), 0) AS total_bytes
                FROM model_locations ml JOIN models m ON m.id = ml.model_id
                WHERE {where}
                """,
                params,
            )
            row = cursor.fetchone()
            return {"total": row["total"] or 0, "total_bytes": row["total_bytes"] or 0}

    def pending_by_type(self, root_id: Optional[str] = None) -> Dict[str, int]:
        where, params = _pending_filter("", None, root_id)
        with get_cursor() as cursor:
            cursor.execute(
                f"""
                SELECT m.model_type AS model_type, COUNT(*) AS n
                FROM model_locations ml JOIN models m ON m.id = ml.model_id
                WHERE {where}
                GROUP BY m.model_type
                """,
                params,
            )
            return {row["model_type"]: row["n"] for row in cursor.fetchall()}

    def pending_location_ids(
        self,
        query: str = "",
        model_type: Optional[str] = None,
        root_id: Optional[str] = None,
        exclude_model_ids: Sequence[str] = (),
    ) -> List[str]:
        where, params = _pending_filter(query, model_type, root_id)
        skip = {model_id for model_id in exclude_model_ids if model_id}
        with get_cursor() as cursor:
            cursor.execute(
                f"SELECT ml.id AS id, ml.model_id AS model_id FROM model_locations ml JOIN models m ON m.id = ml.model_id WHERE {where}",
                params,
            )
            return [row["id"] for row in cursor.fetchall() if row["model_id"] not in skip]

    def pending_location_ids_for_models(self, model_ids: Sequence[str]) -> List[str]:
        unique = sorted({model_id for model_id in model_ids if model_id})
        ids: List[str] = []
        with get_cursor() as cursor:
            for chunk in _chunks(unique):
                placeholders = ",".join("?" for _ in chunk)
                cursor.execute(
                    f"SELECT ml.id AS id FROM model_locations ml WHERE {_PENDING} AND ml.model_id IN ({placeholders})",
                    tuple(chunk),
                )
                ids.extend(row["id"] for row in cursor.fetchall())
        return ids

    def clear_background(self) -> int:
        with get_cursor() as cursor:
            cursor.execute("DELETE FROM model_digest_queue WHERE priority <= ?", (PRIORITY_BACKGROUND,))
            return cursor.rowcount

    def pending_locations_for_model(self, model_id: str) -> List[Dict[str, Any]]:
        with get_cursor() as cursor:
            cursor.execute(
                """
                SELECT ml.id AS location_id, ml.model_id AS model_id, ml.root_id AS root_id,
                       ml.model_type AS model_type, ml.rel_path AS rel_path, ml.binding_id AS binding_id,
                       ml.size AS size, ml.mtime_ns AS mtime_ns, m.filename AS filename,
                       m.is_directory AS is_directory
                FROM model_locations ml JOIN models m ON m.id = ml.model_id
                WHERE ml.model_id = ? AND ml.status = 'present' AND ml.sha256 IS NULL
                ORDER BY ml.id
                """,
                (model_id,),
            )
            return [dict(row) for row in cursor.fetchall()]
