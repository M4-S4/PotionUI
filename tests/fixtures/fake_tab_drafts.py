import copy
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.features.llm.tools.base import ToolContext
from src.features.tab_drafts.manager import TabDraftManager
from src.features.tab_drafts.records import TabDraft, TabDraftSummary
from src.platform.database.rows import now_utc

TAB_ID = "tab-1"


class MemoryTabDraftRepository:
    def __init__(self):
        self.rows: Dict[tuple, TabDraft] = {}

    def get(self, user_id: str, tab_id: str) -> Optional[TabDraft]:
        row = self.rows.get((user_id, tab_id))
        return copy.deepcopy(row) if row else None

    def list_summaries(self, user_id: str) -> List[TabDraftSummary]:
        return [
            TabDraftSummary(
                tab_id=row.tab_id,
                variable_count=len(row.variables),
                rev=row.rev,
                updated_at=row.updated_at,
            )
            for (owner, _), row in sorted(self.rows.items())
            if owner == user_id
        ]

    def replace(self, user_id, tab_id, doc, now=None, max_rows=None):
        return self.mutate(user_id, tab_id, lambda _current: doc, now=now, max_rows=max_rows)

    def mutate(self, user_id, tab_id, mutation, now=None, create=True, max_rows=None):
        existing = self.rows.get((user_id, tab_id))
        if existing is None and not create:
            return None
        current = copy.deepcopy(existing.doc) if existing else {}
        doc = mutation(current)
        stamp = now or now_utc()
        draft = TabDraft(
            user_id=user_id,
            tab_id=tab_id,
            doc=copy.deepcopy(doc),
            rev=(existing.rev if existing else 0) + 1,
            created_at=existing.created_at if existing else stamp,
            updated_at=stamp,
        )
        self.rows[(user_id, tab_id)] = draft
        return copy.deepcopy(draft)

    def delete(self, user_id: str, tab_id: str) -> bool:
        return self.rows.pop((user_id, tab_id), None) is not None

    def delete_not_updated_since(self, cutoff: datetime) -> int:
        stale = [key for key, row in self.rows.items() if row.updated_at < cutoff]
        for key in stale:
            del self.rows[key]
        return len(stale)


class RecordingNotifier:
    def __init__(self):
        self.sent: List[tuple] = []

    def schedule_send(self, user_id, message):
        self.sent.append((user_id, message))


def make_manager():
    notifier = RecordingNotifier()
    return TabDraftManager(MemoryTabDraftRepository(), notifier=notifier), notifier


def seed(manager: TabDraftManager, user_id: str, variables: List[Dict[str, Any]], tab_id: str = TAB_ID) -> None:
    doc = {"variables": {v["name"]: {k: val for k, val in v.items() if k != "name"} for v in variables}}
    manager.repository.mutate(user_id, tab_id, lambda _current: doc)


def drafts_context(
    session_metadata: Optional[dict] = None, user_id: str = "user-1", manager=None, chat_session: bool = True
) -> ToolContext:
    metadata = copy.deepcopy(session_metadata or {})
    manager = manager or make_manager()[0]
    form_state = metadata.get("form_state")
    if isinstance(form_state, dict) and "variables" in form_state:
        variables = form_state.pop("variables")
        form_state.setdefault("tab_id", TAB_ID)
        if variables:
            seed(manager, user_id, variables)
    return ToolContext(user_id=user_id, session_metadata=metadata, tab_drafts=manager, chat_session=chat_session)
