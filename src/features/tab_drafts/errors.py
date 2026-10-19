from typing import Any, Dict, List, Optional


class TabDraftError(Exception):
    pass


class TabDraftNotFound(TabDraftError):
    pass


class TabDraftTooLarge(TabDraftError):
    def __init__(self, size: int, limit: int):
        super().__init__(f"The draft is {size} bytes; the limit is {limit}.")
        self.size = size
        self.limit = limit


class TabDraftLimitReached(TabDraftError):
    def __init__(self, limit: int):
        super().__init__(f"A user can keep at most {limit} tab drafts.")
        self.limit = limit


class TabDraftInvalid(TabDraftError):
    def __init__(self, message: str, operations: Optional[List[Dict[str, Any]]] = None):
        super().__init__(message)
        self.operations = operations or []
