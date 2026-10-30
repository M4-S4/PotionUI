import json
from typing import Any, Dict, Iterable, Optional

from src.features.media_tools.records import MediaToolParams
from src.platform.database import get_cursor
from src.platform.database.rows import now_iso


class MediaToolParamsRepository:

    def get(self, user_id: str, tool_id: str) -> Optional[Dict[str, Any]]:
        with get_cursor() as cursor:
            cursor.execute(
                "SELECT * FROM media_tool_params WHERE user_id = ? AND tool_id = ?",
                (user_id, tool_id),
            )
            row = cursor.fetchone()
        return MediaToolParams.from_row(row).params if row else None

    def get_many(self, user_id: str, tool_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        ids = list(tool_ids)
        if not ids:
            return {}
        placeholders = ", ".join("?" for _ in ids)
        with get_cursor() as cursor:
            cursor.execute(
                f"SELECT * FROM media_tool_params WHERE user_id = ? AND tool_id IN ({placeholders})",
                (user_id, *ids),
            )
            rows = cursor.fetchall()
        records = [MediaToolParams.from_row(row) for row in rows]
        return {record.tool_id: record.params for record in records}

    def save(self, user_id: str, tool_id: str, params: Dict[str, Any]) -> None:
        with get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO media_tool_params (user_id, tool_id, params_json, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT (user_id, tool_id) DO UPDATE SET "
                "params_json = excluded.params_json, updated_at = excluded.updated_at",
                (user_id, tool_id, json.dumps(params), now_iso()),
            )
