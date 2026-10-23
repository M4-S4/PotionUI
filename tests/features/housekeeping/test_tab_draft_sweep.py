from unittest.mock import Mock

from src.features.housekeeping.tasks import prune_tab_drafts
from src.features.housekeeping.worker import build_tasks


class TestPruneTabDrafts:

    def test_it_reports_how_many_drafts_the_sweep_removed(self):
        manager = Mock()
        manager.sweep.return_value = 3

        assert prune_tab_drafts(manager) == {"rows_removed": 3, "errors": []}

    def test_a_failing_sweep_is_reported_not_raised(self):
        manager = Mock()
        manager.sweep.side_effect = RuntimeError("locked")

        result = prune_tab_drafts(manager)

        assert result["rows_removed"] == 0
        assert result["errors"] == ["locked"]


class TestWorkerTask:

    def test_the_sweep_is_a_housekeeping_task_with_no_retention_setting(self):
        manager = Mock()
        manager.sweep.return_value = 2
        tasks = build_tasks(Mock(), Mock(), Mock(), Mock(), manager)
        task = next(t for t in tasks if t.name == "tab_drafts")

        assert task.setting_key is None
        assert task.run(0) == {"rows_removed": 2, "errors": []}
        manager.sweep.assert_called_once_with()
