from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ReplaceTabDraftRequest(BaseModel):
    doc: Dict[str, Any]


class TabDraftOperationsRequest(BaseModel):
    ops: List[Any]
    base_rev: Optional[int] = None
