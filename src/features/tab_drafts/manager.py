import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from src.features.tab_drafts import validation
from src.features.tab_drafts.errors import TabDraftInvalid, TabDraftNotFound
from src.features.tab_drafts.records import TabDraft, TabDraftSummary
from src.features.tab_drafts.repository import TabDraftRepository
from src.platform.database.rows import now_utc

logger = logging.getLogger(__name__)

RETENTION_DAYS = 30


class TabDraftManager:
    def __init__(
        self,
        repository: TabDraftRepository,
        clock: Callable[[], datetime] = now_utc,
        notifier: Optional[Any] = None,
    ):
        self.repository = repository
        self.clock = clock
        self.notifier = notifier

    def announce(self, draft: TabDraft) -> None:
        if self.notifier is None:
            return
        try:
            self.notifier.schedule_send(
                draft.user_id,
                {
                    "type": "tab_variables_changed",
                    "tab_id": draft.tab_id,
                    "doc": draft.doc,
                    "rev": draft.rev,
                },
            )
        except Exception:
            logger.warning("Could not push tab draft %s to its user", draft.tab_id, exc_info=True)

    def get(self, user_id: str, tab_id: str) -> TabDraft:
        draft = self.find(user_id, tab_id)
        if draft is None:
            raise TabDraftNotFound(tab_id)
        return draft

    def find(self, user_id: str, tab_id: str) -> Optional[TabDraft]:
        return self.repository.get(user_id, validation.check_tab_id(tab_id))

    def variables(self, user_id: str, tab_id: str) -> Dict[str, Any]:
        draft = self.find(user_id, tab_id)
        return draft.variables if draft else {}

    def list(self, user_id: str) -> List[TabDraftSummary]:
        return self.repository.list_summaries(user_id)

    def replace(self, user_id: str, tab_id: str, doc: Any) -> TabDraft:
        validation.check_tab_id(tab_id)
        normalized = validation.normalize_doc(doc)
        return self.repository.replace(
            user_id, tab_id, normalized, now=self.clock(), max_rows=validation.MAX_DRAFTS_PER_USER
        )

    def apply_operations(self, user_id: str, tab_id: str, operations: Any) -> TabDraft:
        validation.check_tab_id(tab_id)
        parsed, failures = validation.parse_operations(operations)
        if failures:
            raise TabDraftInvalid("Some operations are invalid.", failures)

        def mutate(current: Dict[str, Any]) -> Dict[str, Any]:
            merged, rejected = validation.apply_operations(current, parsed)
            if rejected:
                raise TabDraftInvalid("Some operations are invalid.", rejected)
            validation.check_doc_size(merged)
            return merged

        return self.repository.mutate(
            user_id, tab_id, mutate, now=self.clock(), max_rows=validation.MAX_DRAFTS_PER_USER
        )

    def delete(self, user_id: str, tab_id: str) -> None:
        if not self.repository.delete(user_id, validation.check_tab_id(tab_id)):
            raise TabDraftNotFound(tab_id)

    def sweep(self, now: Optional[datetime] = None) -> int:
        cutoff = (now or self.clock()) - timedelta(days=RETENTION_DAYS)
        return self.repository.delete_not_updated_since(cutoff)
