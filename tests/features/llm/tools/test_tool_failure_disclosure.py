from unittest.mock import Mock

import pytest

from src.features.llm.tools.base import BaseTool, ToolContext
from src.features.llm.tools.builtin.active_models_tool import GetActiveModelsTool
from src.features.llm.tools.builtin.phrasebook_tool import ListPhrasebookCategoriesTool
from src.features.llm.tools.builtin.segments_tool import ListSegmentCategoriesTool

SECRET = "connection to /srv/app/storage/db.sqlite refused"


def _raising_repo():
    repo = Mock()
    repo.get_all.side_effect = RuntimeError(SECRET)
    return repo


def _segments(is_admin):
    return ListSegmentCategoriesTool(), ToolContext(
        user_id="u1", is_admin=is_admin, segment_category_repository=_raising_repo(),
    )


def _phrasebook(is_admin):
    return ListPhrasebookCategoriesTool(), ToolContext(
        user_id="u1", is_admin=is_admin, phrasebook_category_repository=_raising_repo(),
    )


def _active_models(is_admin):
    return GetActiveModelsTool(), ToolContext(
        user_id="u1", is_admin=is_admin, model_index_manager=Mock(),
        session_metadata={"form_state": {"form_data": {"a": 1}}},
    )


BUILDERS = [_segments, _phrasebook, _active_models]


@pytest.fixture(autouse=True)
def _break_active_models_metadata(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError(SECRET)

    monkeypatch.setattr("src.features.llm.tools.builtin.active_models_tool.build_model_field_metadata", boom)


@pytest.mark.asyncio
@pytest.mark.parametrize("builder", BUILDERS)
async def test_regular_user_never_sees_exception_text(builder, run_tool):
    tool, context = builder(False)

    result = await run_tool(tool, context)

    assert result.success is False
    assert "refused" not in result.error
    assert "/srv/app" not in result.error
    assert "unexpectedly" in result.error


@pytest.mark.asyncio
@pytest.mark.parametrize("builder", BUILDERS)
async def test_admin_sees_exception_text_with_paths_scrubbed(builder, run_tool):
    tool, context = builder(True)

    result = await run_tool(tool, context)

    assert "refused" in result.error
    assert "/srv/app" not in result.error


class _ExplodingTool(BaseTool):
    @property
    def name(self):
        return "explode"

    @property
    def description(self):
        return "Always raises."

    @property
    def parameters(self):
        return {"type": "object", "properties": {}, "required": []}

    @property
    def requires_approval(self):
        return True

    async def execute(self, context, **kwargs):
        raise RuntimeError(SECRET)

    async def execute_confirmed(self, context, **kwargs):
        raise RuntimeError(SECRET)


@pytest.mark.asyncio
async def test_executor_hides_a_raising_tool_from_regular_users(run_tool, confirm_tool):
    context = ToolContext(user_id="u1")

    result = await run_tool(_ExplodingTool(), context)
    confirmed = await confirm_tool(_ExplodingTool(), context)

    for outcome in (result, confirmed):
        assert outcome.success is False
        assert "refused" not in outcome.error
        assert "/srv/app" not in outcome.error


@pytest.mark.asyncio
async def test_executor_keeps_the_scrubbed_detail_for_admins(run_tool, confirm_tool):
    context = ToolContext(user_id="a1", is_admin=True)

    result = await run_tool(_ExplodingTool(), context)
    confirmed = await confirm_tool(_ExplodingTool(), context)

    for outcome in (result, confirmed):
        assert "refused" in outcome.error
        assert "/srv/app" not in outcome.error
