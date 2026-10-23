import json
import time

import pytest

from src.features.llm.tools.base import ToolContext
from src.features.llm.tools.builtin.condition_tool import WriteConditionTool, build_condition
from src.features.prompt import conditions

VARIABLES = {
    "age": {"type": "number", "min": 18, "max": 90, "step": 1, "mode": "shuffle"},
    "mood": {"type": "choice", "options": ["noir", "sunlit", "foggy dawn"], "mode": "shuffle"},
    "record": {"type": "choice", "options": ['12" vinyl', "tape > CD", "C:\\music"], "mode": "shuffle"},
    "subject": {"type": "text", "value": "a woman"},
}


class _Drafts:
    def __init__(self):
        self.calls = []

    def variables(self, user_id, tab_id):
        self.calls.append((user_id, tab_id))
        return VARIABLES


def _chat_context(drafts=None):
    return ToolContext(user_id="u1", session_metadata={"form_state": {"tab_id": "tab-1"}}, tab_drafts=drafts or _Drafts())


def _test(var, op, **extra):
    return {"var": var, "op": op, **extra}


def _render(text, **values):
    rendered, errors = conditions.render_conditions(text, {k: str(v) for k, v in values.items()})
    assert errors == []
    return rendered


async def _run(context, **kwargs):
    result = await WriteConditionTool().execute(context, **kwargs)
    return result, (json.loads(result.data) if result.success else None)


@pytest.mark.asyncio
async def test_builds_if_else_if_otherwise_on_a_choice_variable():
    drafts = _Drafts()
    result, payload = await _run(
        _chat_context(drafts),
        branches=[
            {"tests": [_test("mood", "=", value="noir")], "text": "rain-soaked neon"},
            {"tests": [_test("mood", "in", values=["sunlit", "foggy dawn"])], "text": "warm backlight"},
        ],
        otherwise="overcast sky",
    )
    assert result.success
    assert payload["text"] == (
        '<<if $mood = "noir">>rain-soaked neon<<else if $mood in "sunlit", "foggy dawn">>warm backlight'
        "<<else>>overcast sky<<end>>"
    )
    assert payload["variables_checked"] is True
    assert drafts.calls == [("u1", "tab-1")]
    assert _render(payload["text"], mood="sunlit") == "warm backlight"


@pytest.mark.asyncio
async def test_number_tests_take_bare_numbers_and_ranges():
    _, payload = await _run(
        _chat_context(),
        branches=[
            {"tests": [_test("age", ">", value=65)], "text": "grey hair"},
            {"tests": [_test("age", "in", values=[18, 30]), _test("mood", "is not empty")], "text": "youthful"},
            {"tests": [_test("age", "=", value="40")], "text": "forty"},
        ],
    )
    assert payload["text"] == (
        "<<if $age > 65>>grey hair<<else if $age in 18..30 and $mood is not empty>>youthful"
        "<<else if $age = 40>>forty<<end>>"
    )
    assert _render(payload["text"], age=70, mood="noir") == "grey hair"
    assert _render(payload["text"], age=65, mood="noir") == ""


@pytest.mark.asyncio
async def test_values_with_quotes_backslashes_and_angle_brackets_are_escaped():
    _, payload = await _run(
        _chat_context(),
        branches=[
            {"tests": [_test("record", "=", value='12" vinyl')], "text": "turntable"},
            {"tests": [_test("record", "in", values=["tape > CD", "C:\\music"])], "text": "boombox"},
        ],
    )
    assert payload["text"] == (
        '<<if $record = "12\\" vinyl">>turntable<<else if $record in "tape > CD", "C:\\\\music">>boombox<<end>>'
    )
    assert _render(payload["text"], record='12" vinyl') == "turntable"
    assert _render(payload["text"], record="C:\\music") == "boombox"


@pytest.mark.asyncio
async def test_variable_mistakes_are_refused_with_the_fix():
    result, _ = await _run(
        _chat_context(),
        branches=[{"tests": [_test("mood", "=", value="Noir"), _test("agee", ">", value=3)], "text": "x"}],
    )
    assert not result.success
    assert result.error.startswith("Nothing was built: ")
    assert 'did you mean "noir"?' in result.error
    assert "did you mean $age?" in result.error
    assert result.error.endswith("Fix these and call write_condition again.")


@pytest.mark.parametrize(
    "branches,otherwise,expected",
    [
        ([{"tests": [_test("age", "=", value="sixty")], "text": "x"}], None, "`=` on $age needs a number in value"),
        ([{"tests": [_test("age", "in", values=[30, 18])], "text": "x"}], None, "put the smaller number first"),
        ([{"tests": [_test("age", "~", value=1)], "text": "x"}], None, "op '~' is not one of"),
        ([{"tests": [_test("bad name", "=", value=1)], "text": "x"}], None, "is not a variable name"),
        ([{"tests": [], "text": "x"}], None, "branch 1 needs at least one test"),
        ([{"tests": [_test("mood", "=", value="noir")]}], None, "branch 1 needs text"),
        ([{"tests": [_test("mood", "=", value="noir")], "text": 'a <<if $age > 2>>b<<end>>'}], None, "conditions can't nest"),
        ([{"tests": [_test("mood", "=", value="noir")], "text": "x"}], "<<end>>", "otherwise text contains condition tags"),
        ([{"tests": [_test("mood", "=", value=" ")], "text": "x"}], None, "needs one non-empty text value"),
        ([{"tests": [_test("mood", "in", values=[])], "text": "x"}], None, "needs values"),
        ([], None, "branches needs at least one branch"),
        (
            [{"tests": [_test("age", ">", value=1)] * (conditions.MAX_TESTS + 1), "text": "x"}],
            None,
            f"the limit is {conditions.MAX_TESTS}",
        ),
        (
            [{"tests": [_test("age", ">", value=n)], "text": "x"} for n in range(conditions.MAX_BRANCHES)],
            "y",
            f"at most {conditions.MAX_BRANCHES} branches",
        ),
    ],
)
def test_bad_arguments_name_the_branch_and_the_fix(branches, otherwise, expected):
    text, problems = build_condition(branches, otherwise, VARIABLES)
    assert text is None
    assert any(expected in problem for problem in problems), problems


def test_without_variables_a_number_value_still_builds_a_number_test():
    text, problems = build_condition(
        [{"tests": [_test("$age", ">", value=65), _test("${mood}", "==", value="noir")], "join": "or", "text": "x"}],
        None,
        None,
    )
    assert problems == []
    assert text == '<<if $age > 65 or $mood = "noir">>x<<end>>'


@pytest.mark.asyncio
async def test_check_returns_the_branches_of_valid_text():
    _, payload = await _run(_chat_context(), text='portrait, <<if $mood = "noir" and $age >= 60>>rain<<else>>sun<<end>>')
    assert payload == {
        "valid": True,
        "variables_checked": True,
        "text": 'portrait, <<if $mood = "noir" and $age >= 60>>rain<<else>>sun<<end>>',
        "branches": [[{"when": '$mood = "noir" and $age >= 60', "text": "rain"}, {"when": "otherwise", "text": "sun"}]],
    }


@pytest.mark.asyncio
async def test_check_reports_problems_and_a_corrected_text():
    _, payload = await _run(_chat_context(), text='portrait, <<if $mood == "Noir">>rain<<else>>sun<<end>>')
    assert payload["valid"] is False
    assert payload["problems"] == ['`<<if $mood == "Noir">>`: use = for equals, not ==']
    assert payload["corrected_text"] == 'portrait, <<if $mood = "Noir">>rain<<else>>sun<<end>>'
    assert 'did you mean "noir"?' in payload["corrected_text_problems"][0]


@pytest.mark.asyncio
async def test_check_of_text_without_a_condition_says_so():
    _, payload = await _run(_chat_context(), text="just a portrait")
    assert payload["valid"] is False
    assert "has no condition" in payload["problems"][0]


@pytest.mark.asyncio
async def test_outside_chat_it_reads_the_tab_by_id_and_builds_unchecked_without_one():
    drafts = _Drafts()
    by_id = ToolContext(user_id="u2", chat_session=False, tab_drafts=drafts)
    result, payload = await _run(by_id, tab_id="tab-9", branches=[{"tests": [_test("mood", "=", value="Noir")], "text": "x"}])
    assert not result.success and 'did you mean "noir"?' in result.error
    assert drafts.calls == [("u2", "tab-9")]

    _, payload = await _run(
        ToolContext(user_id="u2", chat_session=False, tab_drafts=drafts),
        branches=[{"tests": [_test("mood", "=", value="Noir")], "text": "x"}],
    )
    assert payload["variables_checked"] is False
    assert payload["variables_note"].startswith("No Generate tab is selected. Pass tab_id")
    assert payload["text"] == '<<if $mood = "Noir">>x<<end>>'


@pytest.mark.asyncio
async def test_in_chat_without_an_open_tab_the_note_says_so():
    _, payload = await _run(
        ToolContext(user_id="u1", tab_drafts=_Drafts()), text='<<if $mood = "noir">>x<<end>>'
    )
    assert payload["variables_checked"] is False
    assert payload["variables_note"].startswith("No Generate tab is open")


@pytest.mark.parametrize("huge", ["1e999999999", "-1e999999999", "1e-999999999", "9" * 30, "0." + "0" * 30 + "1", 1e300])
def test_a_huge_or_tiny_number_is_refused_before_it_is_written_out(huge, monkeypatch):
    written = conditions.format_number

    def guarded(value):
        assert abs(value.adjusted()) <= conditions.MAX_NUMBER_CHARS, "a huge number was written out"
        return written(value)

    monkeypatch.setattr(conditions, "format_number", guarded)
    for test in (_test("age", ">", value=huge), _test("age", "in", values=[huge, 50]), _test("age", "in", values=[1, huge])):
        start = time.perf_counter()
        text, problems = build_condition([{"tests": [test], "text": "x"}], None, VARIABLES)
        assert time.perf_counter() - start < 0.05
        assert text is None
        assert any("needs" in problem for problem in problems), problems


@pytest.mark.asyncio
async def test_results_list_at_most_six_problems():
    many = "".join(f'<<if $mood = "n{i}">>x<<end>>' for i in range(20))
    _, payload = await _run(_chat_context(), text=many)
    assert len(payload["problems"]) == 6
    result, _ = await _run(
        _chat_context(), branches=[{"tests": [_test("mood", "=", value=f"n{i}") for i in range(20)], "text": "x"}]
    )
    assert result.error.count("is not an option of $mood") == 6


def test_schema_is_small_and_lists_every_operator():
    tool = WriteConditionTool()
    schema = tool.to_schema()
    assert len(json.dumps(schema)) < 2000
    ops = schema["function"]["parameters"]["properties"]["branches"]["items"]["properties"]["tests"]["items"]["properties"]["op"]["enum"]
    assert set(ops) == {"=", "!=", "<", "<=", ">", ">=", "in", "not in", "contains", "not contains", "is empty", "is not empty"}
    assert tool.requires_approval is False
    assert tool.modes == ["generation"]


JUNK = ("<<if " * (conditions.MAX_SCAN_CHARS // 5 + 1))[: conditions.MAX_SCAN_CHARS]


def test_thirty_two_junk_bodies_are_refused_without_parsing_them(monkeypatch):
    monkeypatch.setattr(conditions, "parse_template", lambda text: pytest.fail("parsed a body"))
    branches = [{"tests": [_test("mood", "=", value="noir")], "text": JUNK} for _ in range(conditions.MAX_BRANCHES - 1)]
    start = time.perf_counter()
    text, problems = build_condition(branches, JUNK, VARIABLES)
    assert time.perf_counter() - start < 0.3
    assert text is None
    assert problems[0] == "branch 1 text contains condition tags; conditions can't nest, add another branch instead"
    assert problems[-1] == f"the branch texts add up to more than {conditions.MAX_SCAN_CHARS:,} characters"


def test_too_many_branches_stop_before_any_branch_is_read(monkeypatch):
    monkeypatch.setattr("src.features.llm.tools.builtin.condition_tool._predicate", lambda *a: pytest.fail("read a test"))
    text, problems = build_condition([{"tests": [], "text": "x"}] * 10_000, None, VARIABLES)
    assert text is None
    assert problems == [f"a condition has at most {conditions.MAX_BRANCHES} branches, otherwise included; this one has 10000"]


def test_a_branch_with_thousands_of_tests_reads_only_up_to_the_limit():
    tests = [_test("age", ">", value=n) for n in range(200_000)]
    start = time.perf_counter()
    text, problems = build_condition([{"tests": tests, "text": "x"}], None, VARIABLES)
    assert time.perf_counter() - start < 0.3
    assert text is None
    assert problems[0] == f"branch 1 has 200000 tests; the limit is {conditions.MAX_TESTS}"


@pytest.mark.parametrize(
    "branches,expected",
    [
        ([{"tests": [_test("mood", "=", value="noir")], "text": "x" * (conditions.MAX_SCAN_CHARS + 1)}],
         f"branch 1 text is longer than {conditions.MAX_SCAN_CHARS:,} characters"),
        ([{"tests": [_test("mood", "=", value="x" * (conditions.MAX_STRING_CHARS + 1))], "text": "x"}],
         f"a value holds at most {conditions.MAX_STRING_CHARS} characters"),
        ([{"tests": [_test("mood", "in", values=["x" * 1500] * 12)] * 3, "text": "x"}],
         f"its tests are longer than {conditions.MAX_HEAD_CHARS:,} characters"),
        ([{"tests": [_test("mood", "in", values=["a"] * 100_000)], "text": "x"}],
         f"takes at most {conditions.MAX_VALUES} values"),
    ],
)
def test_oversized_parts_are_named(branches, expected):
    text, problems = build_condition(branches, None, None)
    assert text is None
    assert any(expected in problem for problem in problems), problems


@pytest.mark.asyncio
async def test_a_check_too_long_to_parse_twice_lists_problems_without_a_correction():
    _, payload = await _run(_chat_context(), text="x " * 6000 + '<<if $mood == "noir">>rain<<end>>')
    assert payload["problems"] == ['`<<if $mood == "noir">>`: use = for equals, not ==']
    assert "corrected_text" not in payload


@pytest.mark.asyncio
async def test_a_check_of_a_huge_text_is_refused_with_the_reason():
    _, payload = await _run(_chat_context(), text="<<if<<end>> " * 40000)
    assert payload["valid"] is False
    assert payload["problems"][0].startswith("the text is 480,000 characters; conditions are read only in the first 20,000")


def test_a_built_condition_longer_than_the_window_names_the_branch_texts():
    text, problems = build_condition(
        [{"tests": [_test("mood", "=", value="noir")], "text": "x" * (conditions.MAX_SCAN_CHARS - 10)}], "y" * 5, VARIABLES
    )
    assert text is None
    assert problems == [
        f"the built condition is {conditions.MAX_SCAN_CHARS + 31:,} characters; a segment's conditions are read only "
        f"in its first {conditions.MAX_SCAN_CHARS:,}, so shorten the branch texts"
    ]
