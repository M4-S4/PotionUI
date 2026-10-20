import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from src.features.tab_drafts.errors import TabDraftLimitReached
from src.features.tab_drafts.records import TabDraft, TabDraftSummary
from src.platform.database.rows import dt_column, dt_iso, now_utc
from src.platform.database import get_cursor

Mutation = Callable[[Dict[str, Any]], Dict[str, Any]]


def _to_draft(row) -> TabDraft:
    return TabDraft(
        user_id=row["user_id"],
        tab_id=row["tab_id"],
        doc=json.loads(row["doc"]),
        rev=row["rev"],
        created_at=dt_column(row["created_at"]),
        updated_at=dt_column(row["updated_at"]),
    )


class TabDraftRepository:
    def get(self, user_id: str, tab_id: str) -> Optional[TabDraft]:
        with get_cursor() as cursor:
            cursor.execute(
                "SELECT user_id, tab_id, doc, rev, created_at, updated_at "
                "FROM tab_drafts WHERE user_id = ? AND tab_id = ?",
                (user_id, tab_id),
            )
            row = cursor.fetchone()
        return _to_draft(row) if row else None

    def list_summaries(self, user_id: str) -> List[TabDraftSummary]:
        with get_cursor() as cursor:
            cursor.execute(
                "SELECT tab_id, rev, updated_at, variable_count "
                "FROM tab_drafts WHERE user_id = ? ORDER BY updated_at DESC, tab_id",
                (user_id,),
            )
            rows = cursor.fetchall()
        return [
            TabDraftSummary(
                tab_id=row["tab_id"],
                variable_count=row["variable_count"] or 0,
                rev=row["rev"],
                updated_at=dt_column(row["updated_at"]),
            )
            for row in rows
        ]

    def replace(
        self,
        user_id: str,
        tab_id: str,
        doc: Dict[str, Any],
        now: Optional[datetime] = None,
        max_rows: Optional[int] = None,
    ) -> TabDraft:
        return self.mutate(user_id, tab_id, lambda _current: doc, now=now, max_rows=max_rows)

    def mutate(
        self,
        user_id: str,
        tab_id: str,
        mutation: Mutation,
        now: Optional[datetime] = None,
        create: bool = True,
        max_rows: Optional[int] = None,
    ) -> Optional[TabDraft]:
        from src.platform.database.database import db

        stamp = dt_iso(now or now_utc())
        with db.get_connection() as conn:
            previous_isolation = conn.isolation_level
            conn.isolation_level = None
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                cursor.execute(
                    "SELECT doc, rev, created_at FROM tab_drafts WHERE user_id = ? AND tab_id = ?",
                    (user_id, tab_id),
                )
                row = cursor.fetchone()
                if row is None and not create:
                    cursor.execute("ROLLBACK")
                    return None
                if row is None and max_rows is not None:
                    cursor.execute("SELECT COUNT(*) FROM tab_drafts WHERE user_id = ?", (user_id,))
                    if cursor.fetchone()[0] >= max_rows:
                        raise TabDraftLimitReached(max_rows)
                current = json.loads(row["doc"]) if row else {}
                rev = (row["rev"] if row else 0) + 1
                created_at = row["created_at"] if row else stamp
                doc = mutation(current)
                cursor.execute(
                    "INSERT INTO tab_drafts (user_id, tab_id, doc, rev, variable_count, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT (user_id, tab_id) DO UPDATE SET "
                    "doc = excluded.doc, rev = excluded.rev, "
                    "variable_count = excluded.variable_count, updated_at = excluded.updated_at",
                    (
                        user_id,
                        tab_id,
                        json.dumps(doc, separators=(",", ":")),
                        rev,
                        len(doc.get("variables") or {}),
                        created_at,
                        stamp,
                    ),
                )
                cursor.execute("COMMIT")
            except BaseException:
                cursor.execute("ROLLBACK")
                raise
            finally:
                conn.isolation_level = previous_isolation
        return TabDraft(
            user_id=user_id,
            tab_id=tab_id,
            doc=doc,
            rev=rev,
            created_at=dt_column(created_at),
            updated_at=dt_column(stamp),
        )

    def delete(self, user_id: str, tab_id: str) -> bool:
        with get_cursor() as cursor:
            cursor.execute(
                "DELETE FROM tab_drafts WHERE user_id = ? AND tab_id = ?", (user_id, tab_id)
            )
            return cursor.rowcount > 0

    def delete_not_updated_since(self, cutoff: datetime) -> int:
        with get_cursor() as cursor:
            cursor.execute("DELETE FROM tab_drafts WHERE updated_at < ?", (dt_iso(cutoff),))
            return cursor.rowcount
