import pytest

from src.features.llm.tools.base import ToolContext
from src.features.llm.tools.builtin.condition_text import condition_tag_error, ensure_condition_tags
from types import SimpleNamespace

from src.features.llm.tools.builtin.manage_prompts_tool import AddPromptTool, EditPromptTool
from src.features.llm.tools.builtin.segments_tool import (
    CreateSavedSegmentTool,
    CreateSegmentTemplateTool,
    UpdateSavedSegmentTool,
    UpdateSegmentTemplateTool,
)

GOOD = "woman, <<if $age > 65>>grey hair<<else>>auburn hair<<end>>"


@pytest.mark.parametrize(
    "text,kind",
    [
        ("woman, <<if $age > 65>>grey hair", "never reaches <<end>>"),
        ("woman <<else>> grey<<end>>", "no open <<if"),
        ("<<if $a > 1>>x<<if $b > 1>>y<<end>><<end>>", "inside another condition"),
        ("<<if $age>65>>x<<end>>", "cannot read"),
        ("<<if $age > 1 and $b = \"x\" or $c < 2>>x<<end>>", "cannot read"),
    ],
)
def test_malformed_tags_get_a_teaching_error(text, kind):
    error = condition_tag_error([text])
    assert kind in error
    assert "<<if $age > 65>>" in error
    assert "Nothing was proposed" in error


@pytest.mark.parametrize("text", [GOOD, "plain text", "", "<lora:x:1> <<hello>> <Picture 1>", None])
def test_valid_or_tagless_text_passes(text):
    assert condition_tag_error([text]) == ""
    ensure_condition_tags([text])


def test_ensure_raises_value_error_for_the_approval_tool_to_surface():
    with pytest.raises(ValueError, match="malformed condition tags"):
        ensure_condition_tags(["<<if $a > 1>>x"])


class _Sentinel:
    def __getattr__(self, name):
        raise AssertionError(f"touched {name} before validating condition tags")


BAD = "a <<if $x = \"y\">>b"


def _segment_ctx():
    return ToolContext(
        user_id="u", saved_segment_repository=_Sentinel(), segment_category_repository=_Sentinel(), plugin_registry=_Sentinel()
    )


def _template_ctx():
    return ToolContext(user_id="u", segment_template_repository=_Sentinel(), plugin_registry=_Sentinel())


def _prompt_ctx():
    return ToolContext(user_id="u", prompt_database=_Sentinel())


CALL_SITES = {
    "saved_create_content": (CreateSavedSegmentTool, _segment_ctx, dict(name="n", category_id="c", content=BAD)),
    "saved_create_prefix": (CreateSavedSegmentTool, _segment_ctx, dict(name="n", category_id="c", content="ok", prefix="<<end>>")),
    "saved_create_suffix": (CreateSavedSegmentTool, _segment_ctx, dict(name="n", category_id="c", content="ok", suffix=BAD)),
    "template_create": (CreateSegmentTemplateTool, _template_ctx, dict(name="n", segments=[{"content": "ok"}, {"content": BAD}])),
    "template_create_prefix": (CreateSegmentTemplateTool, _template_ctx, dict(name="n", segments=[{"content": "ok", "prefix": BAD}])),
    "add_prompt": (AddPromptTool, _prompt_ctx, dict(segments=[{"content": BAD}])),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("site", sorted(CALL_SITES))
async def test_create_sites_refuse_before_touching_anything(site):
    tool_cls, make_ctx, kwargs = CALL_SITES[site]
    result = await tool_cls().execute(make_ctx(), **kwargs)
    assert not result.success
    assert "malformed condition tags" in result.error


def _existing_segment():
    return SimpleNamespace(
        id="sid", name="n", category_id="c", content="ok", chips={}, resources={}, enabled=True,
        color=None, description=None, prefix=None, suffix=None, tags=[],
    )


def _existing_template():
    seg = SimpleNamespace(content="ok", prefix=None, suffix=None, model_dump=lambda mode="json": {"content": "ok"})
    return SimpleNamespace(id="tid", name="n", description=None, tags=[], segments=[seg])


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["content", "prefix", "suffix"])
async def test_update_saved_segment_refuses_before_writing(field):
    tool = UpdateSavedSegmentTool()
    tool._existing = lambda context, kwargs: _existing_segment()
    result = await tool.execute(_segment_ctx(), segment_id="sid", **{field: BAD})
    assert not result.success
    assert "malformed condition tags" in result.error


@pytest.mark.asyncio
async def test_update_template_refuses_before_writing():
    tool = UpdateSegmentTemplateTool()
    tool._existing = lambda context, kwargs: _existing_template()
    result = await tool.execute(_template_ctx(), template_id="tid", segments=[{"content": BAD}])
    assert not result.success
    assert "malformed condition tags" in result.error


@pytest.mark.asyncio
async def test_edit_prompt_refuses_before_writing(monkeypatch):
    from src.features.llm.tools.builtin import manage_prompts_tool as module

    existing = SimpleNamespace(
        segments=[], variables=None, name=None, usage_hint=None, nsfw=False,
        **{f: None for f in (
            "source_provider", "source_id", "source_url", "source_group_id", "model_id", "model_name",
            "base_model", "cfg_scale", "steps", "sampler", "width", "height",
        )},
        **{f: 0 for f in ("heart_count", "like_count", "laugh_count", "cry_count", "comment_count")},
        tags=[], metadata={},
    )
    monkeypatch.setattr(module, "_require_prompt", lambda context, prompt_id: existing)
    result = await EditPromptTool().execute(_prompt_ctx(), prompt_id="p", segments=[{"content": BAD}])
    assert not result.success
    assert "malformed condition tags" in result.error
