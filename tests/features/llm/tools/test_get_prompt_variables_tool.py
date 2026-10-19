import json

import pytest

from tests.fixtures.fake_tab_drafts import drafts_context

from src.features.llm.tools.base import ToolContext
from src.features.llm.tools.builtin.prompt_variables_tool import (
    DEFAULT_OPTION_LIMIT,
    MAX_LIST_LIMIT,
    MAX_OPTION_LIMIT,
    GetPromptVariablesTool,
    ManagePromptVariablesTool,
)


def _ctx(variables):
    return drafts_context({"form_state": {"variables": variables}}, user_id="u")


def _choice(name, options, **extra):
    return {"name": name, "type": "choice", "options": options, "mode": "shuffle", **extra}


def _outfit_options(count=40):
    return [(f"option {i:02d} " + "long outfit description " * 8).strip() for i in range(count)]


def _fixture():
    variables = [_choice("outfit_kind", _outfit_options())]
    variables += [_choice(f"mood_{i}", ["noir", "sunlit", "dusk"]) for i in range(6)]
    variables += [
        {"name": f"num_{i}", "type": "number", "mode": "shuffle", "min": 0, "max": 100, "step": 5}
        for i in range(5)
    ]
    variables += [{"name": f"txt_{i}", "type": "text", "value": f"fragment {i}"} for i in range(8)]
    return variables


async def _run(variables, **kwargs):
    result = await GetPromptVariablesTool().execute(_ctx(variables), **kwargs)
    return result, (json.loads(result.data) if result.success else None)


class TestSchema:
    def test_name_and_read_only(self):
        tool = GetPromptVariablesTool()
        assert tool.name == "get_prompt_variables"
        assert tool.requires_approval is False
        assert tool.modes == ["generation"]

    def test_names_is_an_optional_array(self):
        schema = GetPromptVariablesTool().parameters
        assert schema["required"] == []
        assert schema["properties"]["names"]["type"] == "array"

    def test_description_is_templated_for_the_write_tool(self):
        assert "{{#if get_prompt_variables}}" in ManagePromptVariablesTool().description


class TestList:
    @pytest.mark.asyncio
    async def test_lists_every_variable_as_a_summary_row(self):
        _, data = await _run(_fixture())
        assert data["total"] == 20
        assert data["count"] == 20
        assert data["has_more"] is False
        rows = {r["name"]: r for r in data["variables"]}
        assert rows["outfit_kind"] == {"name": "outfit_kind", "type": "choice", "mode": "shuffle", "option_count": 40}
        assert rows["num_0"]["count"] == 21 and rows["num_0"]["step"] == "5"
        assert rows["txt_3"] == {"name": "txt_3", "type": "text", "length": 10}

    @pytest.mark.asyncio
    async def test_summary_rows_never_carry_option_lists(self):
        result, data = await _run(_fixture())
        assert "options" not in data["variables"][0]
        assert "option 05" not in result.data

    @pytest.mark.asyncio
    async def test_pinned_value_is_reported(self):
        _, data = await _run([_choice("m", ["a", "b", "c"], mode="pin", pinnedIndex=2)])
        assert data["variables"][0]["pinned_value"] == "c"

    @pytest.mark.asyncio
    async def test_search_matches_names_and_option_texts(self):
        _, by_name = await _run(_fixture(), search="OUTFIT")
        assert [r["name"] for r in by_name["variables"]] == ["outfit_kind"]
        _, by_option = await _run(_fixture(), search="sunlit")
        assert by_option["total"] == 6
        _, by_text = await _run(_fixture(), search="fragment 4")
        assert [r["name"] for r in by_text["variables"]] == ["txt_4"]

    @pytest.mark.asyncio
    async def test_paging_reports_total_and_has_more(self):
        _, first = await _run(_fixture(), limit=8)
        assert (first["count"], first["total"], first["offset"], first["has_more"]) == (8, 20, 0, True)
        _, last = await _run(_fixture(), limit=8, offset=16)
        assert (last["count"], last["has_more"]) == (4, False)
        names = [r["name"] for r in first["variables"]] + [
            r["name"] for r in (await _run(_fixture(), limit=8, offset=8))[1]["variables"]
        ] + [r["name"] for r in last["variables"]]
        assert len(set(names)) == 20

    @pytest.mark.asyncio
    async def test_limit_is_capped(self):
        _, data = await _run(_fixture(), limit=100000)
        assert data["limit"] == MAX_LIST_LIMIT

    @pytest.mark.asyncio
    async def test_bad_paging_values_are_an_error(self):
        result, _ = await _run(_fixture(), limit="many")
        assert not result.success and "integers" in result.error

    @pytest.mark.asyncio
    async def test_no_variables_is_a_note_not_an_error(self):
        result, data = await _run([])
        assert result.success and data["total"] == 0 and "no prompt variables" in data["note"]

    @pytest.mark.asyncio
    async def test_missing_tab_id_is_an_error_that_names_the_fix(self):
        result = await GetPromptVariablesTool().execute(drafts_context({}, user_id="u", chat_session=False))
        assert not result.success and "tab_id" in result.error and "list_tab_drafts" in result.error

    @pytest.mark.asyncio
    async def test_a_tab_without_a_draft_has_no_variables_and_a_hint(self):
        result, data = await _run([])
        assert result.success
        assert data["variables"] == [] and data["total"] == 0
        assert "no prompt variables" in data["note"]


class TestDetail:
    @pytest.mark.asyncio
    async def test_returns_every_option_with_its_index(self):
        _, data = await _run(_fixture(), names=["outfit_kind"])
        detail = data["variables"][0]
        assert detail["options_total"] == 40
        assert [o["index"] for o in detail["options"]] == list(range(40))
        assert detail["options"][2]["text"] == _outfit_options()[2]
        assert detail["has_more"] is False and data["has_more"] is False

    @pytest.mark.asyncio
    async def test_options_beyond_the_old_cut_are_readable(self):
        _, data = await _run(_fixture(), names=["outfit_kind"], search="option 39")
        assert [o["index"] for o in data["variables"][0]["options"]] == [39]

    @pytest.mark.asyncio
    async def test_option_conditions_pinned_index_and_readers(self):
        variables = [
            _choice("music", ["hip hop", "classical"], mode="pin", pinnedIndex=1),
            _choice("dance", [{"text": "breaking", "when": {"var": "music", "values": ["hip hop"]}}, "salsa"]),
        ]
        _, data = await _run(variables, names=["music", "dance"])
        music, dance = data["variables"]
        assert music["pinned_index"] == 1 and music["pinned_value"] == "classical"
        assert music["read_by_conditions_of"] == ["dance"]
        assert dance["options"][0] == {"index": 0, "text": "breaking", "when": {"var": "music", "values": ["hip hop"]}}
        assert dance["resolves_after"] == ["music"]

    @pytest.mark.asyncio
    async def test_several_names_keep_the_requested_order(self):
        _, data = await _run(_fixture(), names=["txt_2", "mood_1"])
        assert [d["name"] for d in data["variables"]] == ["txt_2", "mood_1"]
        assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_name_forms_with_dollar_and_braces_resolve(self):
        _, data = await _run(_fixture(), names=["${mood_1}", "$mood_2", "mood_1"])
        assert [d["name"] for d in data["variables"]] == ["mood_1", "mood_2"]

    @pytest.mark.asyncio
    async def test_single_name_string_is_accepted(self):
        _, data = await _run(_fixture(), name="mood_1")
        assert data["variables"][0]["name"] == "mood_1"

    @pytest.mark.asyncio
    async def test_options_page_with_offset_and_limit(self):
        _, first = await _run(_fixture(), names=["outfit_kind"], limit=15)
        detail = first["variables"][0]
        assert len(detail["options"]) == 15 and detail["has_more"] and first["has_more"]
        _, last = await _run(_fixture(), names=["outfit_kind"], limit=15, offset=30)
        tail = last["variables"][0]
        assert [o["index"] for o in tail["options"]] == list(range(30, 40))
        assert tail["has_more"] is False
        assert tail["options_total"] == 40

    @pytest.mark.asyncio
    async def test_option_limit_is_capped(self):
        _, data = await _run([_choice("big", [f"o{i}" for i in range(500)])], names=["big"], limit=100000)
        detail = data["variables"][0]
        assert detail["limit"] == MAX_OPTION_LIMIT
        assert len(detail["options"]) == MAX_OPTION_LIMIT and detail["has_more"]

    @pytest.mark.asyncio
    async def test_default_option_page_size(self):
        _, data = await _run([_choice("big", [f"o{i}" for i in range(500)])], names=["big"])
        assert len(data["variables"][0]["options"]) == DEFAULT_OPTION_LIMIT

    @pytest.mark.asyncio
    async def test_number_detail_has_the_grid_summary(self):
        variables = [{"name": "age", "type": "number", "mode": "pin", "min": 18, "max": 90, "step": 1, "value": 30}]
        _, data = await _run(variables, names=["age"])
        assert data["variables"][0] == {
            "name": "age", "type": "number", "mode": "pin", "min": "18", "max": "90",
            "step": "1", "count": 73, "decimals": 0, "pinned_value": "30",
        }

    @pytest.mark.asyncio
    async def test_number_decimals_follow_the_step(self):
        variables = [{"name": "cfg", "type": "number", "mode": "shuffle", "min": 1, "max": 2, "step": 0.25}]
        _, data = await _run(variables, names=["cfg"])
        detail = data["variables"][0]
        assert (detail["decimals"], detail["count"], detail["step"]) == (2, 5, "0.25")
        assert "pinned_value" not in detail

    @pytest.mark.asyncio
    async def test_invalid_number_is_flagged(self):
        _, data = await _run([{"name": "bad", "type": "number", "min": 5, "max": 1}], names=["bad"])
        assert data["variables"][0]["invalid"] is True

    @pytest.mark.asyncio
    async def test_text_detail_returns_the_full_value(self):
        long_value = "word " * 400
        _, data = await _run([{"name": "t", "type": "text", "value": long_value}], names=["t"])
        assert data["variables"][0]["value"] == long_value
        assert data["variables"][0]["length"] == len(long_value.strip())


class TestNotFound:
    @pytest.mark.asyncio
    async def test_unknown_name_is_an_error_with_suggestions(self):
        result, _ = await _run(_fixture(), names=["outfit"])
        assert not result.success
        assert "'outfit' not found" in result.error
        assert "did you mean: outfit_kind" in result.error
        assert "Variables on this form:" in result.error

    @pytest.mark.asyncio
    async def test_typo_suggests_the_close_name(self):
        result, _ = await _run(_fixture(), names=["moood_1"])
        assert "mood_1" in result.error

    @pytest.mark.asyncio
    async def test_no_suggestion_still_lists_the_available_names(self):
        result, _ = await _run([_choice("alpha", ["a"])], names=["zzzz"])
        assert "not found" in result.error and "did you mean" not in result.error
        assert "alpha" in result.error

    @pytest.mark.asyncio
    async def test_partial_hit_returns_found_and_reports_missing(self):
        result, data = await _run(_fixture(), names=["mood_1", "outfit"])
        assert result.success
        assert [d["name"] for d in data["variables"]] == ["mood_1"]
        assert data["not_found"] == [{"name": "outfit", "suggestions": ["outfit_kind"]}]


class TestReadsTheSessionsVariables:
    @pytest.mark.asyncio
    async def test_does_not_mutate_the_form_state(self):
        variables = _fixture()
        before = json.dumps(variables, sort_keys=True)
        await _run(variables, names=["outfit_kind"])
        await _run(variables)
        assert json.dumps(variables, sort_keys=True) == before

    @pytest.mark.asyncio
    async def test_third_option_index_is_what_manage_prompt_variables_accepts(self):
        variables = [_choice("outfit_kind", _outfit_options())]
        _, data = await _run(variables, names=["outfit_kind"])
        third = data["variables"][0]["options"][2]
        write = await ManagePromptVariablesTool().execute(
            _ctx(variables),
            operations=[{"op": "set", "name": "outfit_kind", "type": "choice",
                         "options": _outfit_options(), "mode": "pin", "pinned_index": third["index"]}],
        )
        assert write.success
        assert third["text"] in json.loads(write.data)["proposed_changes"][0]["new_value"]


class TestCompleteDraft:
    @pytest.mark.asyncio
    async def test_complete_snapshot_reports_nothing_missing(self):
        _, data = await _run(_fixture(), names=["outfit_kind"])
        assert "options_missing" not in data["variables"][0]
        assert "omitted_variables" not in data


class TestFullSnapshotOptionsAreWritable:
    @pytest.mark.asyncio
    async def test_condition_on_an_option_past_the_old_twelve_option_cut_is_accepted(self):
        variables = [_choice("outfit_kind", _outfit_options())]
        target = _outfit_options()[30]
        result = await ManagePromptVariablesTool().execute(
            _ctx(variables),
            operations=[{"op": "set", "name": "shoes", "type": "choice", "options": [
                {"text": "boots", "when": {"var": "outfit_kind", "values": [target]}}, "sneakers"]}],
        )
        assert result.success, result.error
        assert "warnings" not in json.loads(result.data)

    @pytest.mark.asyncio
    async def test_pinning_the_thirtieth_option_is_accepted(self):
        variables = [_choice("outfit_kind", _outfit_options())]
        result = await ManagePromptVariablesTool().execute(
            _ctx(variables),
            operations=[{"op": "set", "name": "outfit_kind", "type": "choice",
                         "options": _outfit_options(), "mode": "pin", "pinned_index": 29}],
        )
        assert result.success, result.error
        assert _outfit_options()[29] in json.loads(result.data)["proposed_changes"][0]["new_value"]
