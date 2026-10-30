import json
from dataclasses import dataclass, field
from typing import Any, Dict

from src.platform.database.rows import row_get


@dataclass
class MediaToolParams:
    user_id: str
    tool_id: str
    params: Dict[str, Any] = field(default_factory=dict)
    updated_at: str = ""

    @classmethod
    def from_row(cls, row) -> "MediaToolParams":
        try:
            params = json.loads(row["params_json"])
        except (TypeError, ValueError):
            params = {}
        return cls(
            user_id=row["user_id"],
            tool_id=row["tool_id"],
            params=params if isinstance(params, dict) else {},
            updated_at=row_get(row, "updated_at") or "",
        )
