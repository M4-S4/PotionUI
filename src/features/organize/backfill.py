import logging
from typing import Any, Dict, List

from src.features.organize import errors
from src.features.organize.evaluator import evaluate
from src.features.organize.jobs import BackfillJob
from src.features.organize.manager import (
    BACKFILL_CHUNK,
    NOTICE_JOB_FINISHED,
    NOTICE_JOB_PROGRESS,
    OrganizeManager,
    Viewer,
    viewer_of,
)

logger = logging.getLogger(__name__)


class BackfillManager:

    def __init__(self, organize: OrganizeManager):
        self.o = organize

    def start(self, user: Any, rule_id: str) -> Dict[str, Any]:
        viewer = viewer_of(user)
        rule = self.o.require_rule(viewer.id, rule_id)
        if self.o.is_paused_for(viewer.id):
            raise errors.organize_paused()
        running = self.o.jobs.active_for_rule(viewer.id, rule_id)
        if running is not None:
            raise errors.OrganizeError("job_running", "This rule is already being applied", 409, {"job_id": running.id})
        restricted = self.o.c.visibility.is_restricted(viewer.id)
        reason = self.o.blocking_reason(rule, viewer.id, viewer.is_admin, restricted)
        if reason:
            raise errors.OrganizeError("rule_not_runnable", "Fix what this rule needs before applying it", 422,
                                       {"reason": reason})
        job = self.o.jobs.create(viewer.id, rule.id, rule.name)
        self.o.executor.submit(self._run, job, viewer)
        return job.to_dict()

    def _run(self, job: BackfillJob, viewer: Viewer) -> None:
        try:
            self._fill(job, viewer)
        except Exception:
            logger.exception("Auto-organize backfill %s failed", job.id)
            job.fail("Applying the rule stopped because of an error")
            if job.run_id:
                self.o.c.runs.finish_run(viewer.id, job.run_id, "failed")

    def _fill(self, job: BackfillJob, viewer: Viewer) -> None:
        if job.status == "cancelled":
            return
        rule = self.o.c.rules.get(viewer.id, job.rule_id)
        if rule is None:
            job.fail("The rule was deleted")
            return
        job.start()
        restricted = self.o.c.visibility.is_restricted(viewer.id)
        ids, _approximate = self.o.matching_ids(viewer, rule.subject, rule.match, rule.conditions, None)
        handled = self.o.c.rules.handled_ids(rule.id, rule.subject, ids)
        ids = [i for i in ids if i not in handled]
        job.total = len(ids)
        run = self.o.c.runs.create_run(rule.id, rule.name, viewer.id, rule.subject, "backfill")
        job.run_id = run.id
        status = "completed"
        for start in range(0, len(ids), BACKFILL_CHUNK):
            if job.cancel_requested or self.o.is_paused_for(viewer.id):
                status = "cancelled"
                break
            chunk = ids[start:start + BACKFILL_CHUNK]
            blocked = False
            for item in self.o.load_items(rule.subject, viewer.id, chunk, viewer.is_admin, restricted):
                if not evaluate(self.o.c.registry, rule.match, rule.conditions, item):
                    continue
                result = self.o.apply(rule, item, lambda: run, bump=False)
                if result is None:
                    blocked = True
                    break
                if result:
                    job.applied += 1
            if rule.subject == "generation":
                self.o.c.writer.bump_history(viewer.id)
            job.processed = min(job.total, start + len(chunk))
            self.o.notify(user_id=viewer.id, level="info", title="Auto-organize", message=f"Applying \"{rule.name}\"",
                         type=NOTICE_JOB_PROGRESS, metadata=job.to_dict(), transient=True, show_toast=False)
            if blocked:
                status = "failed"
                job.error = "The collection this rule adds to no longer exists"
                break
        self.o.c.runs.finish_run(viewer.id, run.id, status)
        job.finish(status)
        if status != "failed":
            self.o.notify(user_id=viewer.id, level="success", title="Auto-organize",
                         message=self._finished_message(viewer.id, run.id, job.applied),
                         type=NOTICE_JOB_FINISHED, metadata=job.to_dict())

    def _finished_message(self, user_id: str, run_id: str, applied: int) -> str:
        if not applied:
            return "Nothing new to add"
        summary = self.o.c.runs.run_change_summary(user_id, [run_id]).get(run_id, [])
        targets = [row["target_name"] for row in summary if row["target_type"] in ("collection", "model_collection")]
        noun = "item" if applied == 1 else "items"
        if targets:
            return f"Added {applied} {noun} to {', '.join(dict.fromkeys(targets))}"
        return f"Organized {applied} {noun}"

    def list_jobs(self, user: Any, active: bool) -> List[Dict[str, Any]]:
        return [job.to_dict() for job in self.o.jobs.for_user(viewer_of(user).id, active)]

    def get_job(self, user: Any, job_id: str) -> Dict[str, Any]:
        job = self.o.jobs.get(viewer_of(user).id, job_id)
        if job is None:
            raise errors.OrganizeError("job_not_found", "Job not found", 404)
        return job.to_dict()

    def cancel_job(self, user: Any, job_id: str) -> Dict[str, Any]:
        job = self.o.jobs.get(viewer_of(user).id, job_id)
        if job is None:
            raise errors.OrganizeError("job_not_found", "Job not found", 404)
        job.request_cancel()
        return job.to_dict()
