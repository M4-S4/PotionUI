from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict

from src.platform.database.rows import dt_iso


@dataclass
class TabDraft:
    user_id: str
    tab_id: str
    doc: Dict[str, Any] = field(default_factory=dict)
    rev: int = 0
    created_at: datetime = None
    updated_at: datetime = None

    @property
    def variables(self) -> Dict[str, Any]:
        value = self.doc.get("variables")
        return value if isinstance(value, dict) else {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tab_id": self.tab_id,
            "doc": self.doc,
            "rev": self.rev,
            "created_at": dt_iso(self.created_at),
            "updated_at": dt_iso(self.updated_at),
        }


@dataclass
class TabDraftSummary:
    tab_id: str
    variable_count: int
    rev: int
    updated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tab_id": self.tab_id,
            "variable_count": self.variable_count,
            "rev": self.rev,
            "updated_at": dt_iso(self.updated_at),
        }
