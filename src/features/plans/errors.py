from typing import Any, Dict, List, Optional

from src.platform.http.base_controller import PassThrough
from src.platform.http.errors import DomainError


class PlanError(DomainError):
    pass


class LimitExceeded(PassThrough):
    error = "limit_exceeded"
    status_code = 403

    def __init__(self, refusals: List[Dict[str, Any]], point: str, contact_line: Optional[str]):
        self.refusals = refusals
        self.point = point
        self.contact_line = contact_line or ""
        super().__init__(refusals[0]["message"])

    @property
    def primary(self) -> Dict[str, Any]:
        return self.refusals[0]

    @property
    def kinds(self) -> List[str]:
        return [refusal["kind"] for refusal in self.refusals]

    def payload(self) -> Dict[str, Any]:
        primary = self.primary
        return {
            "success": False,
            "error": self.error,
            "code": primary["code"],
            "message": primary["message"],
            "kind": primary["kind"],
            "label": primary["label"],
            "format": primary["format"],
            "point": self.point,
            "used": primary["used"],
            "limit": primary["limit"],
            "incoming": primary.get("incoming"),
            "needed": primary.get("needed"),
            "remaining": primary.get("remaining"),
            "percent": primary["percent"],
            "resets_at": primary["resets_at"],
            "contact_line": self.contact_line,
            "kinds": [dict(refusal) for refusal in self.refusals],
        }
