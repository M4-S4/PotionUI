from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional

from src.platform.database.rows import dt_iso


@dataclass
class Workspace:
    """Model representing a saved workspace (tab layout configuration)."""
    id: str
    user_id: str
    name: str
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'data': self.data,
            'created_at': dt_iso(self.created_at),
            'updated_at': dt_iso(self.updated_at)
        }
