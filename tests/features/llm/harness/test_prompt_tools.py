import pytest

from src.features.llm.harness import prompt_tools
from src.features.llm.harness.adapter import ToolSpec
from src.features.llm.harness.messages import ToolCallPart
from src.features.llm.harness.prompt_tools import CallItem, PromptToolStage, TextItem
from tests.features.llm.harness.history import tool_schemas

COMPLEX = [
    {
        "type": "function",
        "function": {
            "name": "set_things",
            "description": "Set things",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"enum": ["a", "b"]},
                    "count": {"type": "integer", "minimum": 3},
                    "items": {"type": "array", "minItems": 2, "items": {"type": "string", "minLength": 5}},
                    "nested": {"type": "object", "properties": {"k": {"type": "boolean"}}, "required": ["k"]},
                    "either": {"anyOf": [{"type": "number", "maximum": -1, "minimum": 0}, {"type": "string"}]},
                },
                "required": ["mode", "count", "items", "nested", "either"],
            },
        },
    },
    {"type": "function", "function": {"name": "needs_unsynthesisable", "description": "x", "parameters": {"required": ["ghost"]}}},
]


def specs(schemas):
    return tuple(ToolSpec.from_schema(schema) for schema in schemas)


SIMPLE_INLINE = (
    '\n\nYou have access to the following tools. To call one, respond with EXACTLY one <tool_call>{"name": "...", '
    '"arguments": {...}}</tool_call> block and nothing else \u2014 for example <tool_call>{"name": "get_form_state", '
    '"arguments": {"scope": "..."}}</tool_call>. Do not wrap a call in a <tool_action> tag or a code fence; only a '
    '<tool_call> block runs.\n- get_form_state: Read the current form (arguments schema: {"type": "object", '
    '"properties": {"scope": {"type": "string", "description": "what to read"}}, "required": ["scope"]})'
)

COMPLEX_EXAMPLE = (
    '<tool_call>{"name": "set_things", "arguments": {"mode": "a", "count": 3, "items": [".....", "....."], '
    '"nested": {"k": false}, "either": -1}}</tool_call>'
)

UNSYNTHESISABLE_EXAMPLE = '<tool_call>{"name": "<tool name>", "arguments": {...}}</tool_call>'

SIMPLE_SYSTEM = (
    "\n\n## Available Tools\n\n"
    "You have the following tools available. To call a tool, output a "
    "`<tool_call>` XML block with a JSON body containing `name` and `arguments`.\n\n"
    "Example:\n"
    '<tool_call>{"name": "tool_name", "arguments": {"arg1": "value1"}}</tool_call>\n\n'
    "You may call multiple tools in a single response. "
    "After each tool call you will receive the result in the next message. "
    "When you have enough information, respond normally without any tool_call blocks.\n\n"
    "### Tools\n\n"
    "**get_form_state**: Read the current form\n"
    '  Parameters: {"properties":{"scope":{"description":"what to read","type":"string"}},'
    '"required":["scope"],"type":"object"}\n'
)


def test_the_system_tool_text_lists_every_tool_with_its_full_schema_sorted():
    assert prompt_tools.render_tools_text(specs(tool_schemas())) == SIMPLE_SYSTEM
    text = prompt_tools.render_tools_text(specs(COMPLEX))
    assert text.startswith(SIMPLE_SYSTEM.split("**get_form_state**")[0])
    assert '**set_things**: Set things\n  Parameters: {"properties":{"count":{"minimum":3,"type":"integer"}' in text
    assert text.count("**needs_unsynthesisable**: x\n  Parameters: {\"required\":[\"ghost\"]}") == 1


def test_the_inline_tool_text_names_a_real_example_built_from_the_offered_tools():
    assert prompt_tools.render_inline_tools_text(specs(tool_schemas())) == SIMPLE_INLINE
    assert COMPLEX_EXAMPLE in prompt_tools.render_inline_tools_text(specs(COMPLEX))
    assert UNSYNTHESISABLE_EXAMPLE in prompt_tools.render_inline_tools_text(specs(COMPLEX[1:]))


def test_no_tools_render_no_inline_text():
    assert prompt_tools.render_inline_tools_text(()) == ""


def test_an_assistant_turn_with_calls_is_text_the_model_could_have_written():
    call = ToolCallPart(call_id="c", name="get_form_state", args={"scope": "all"})

    assert prompt_tools.assistant_text("", [call]) == (
        '<tool_call>{"name": "get_form_state", "arguments": {"scope": "all"}}</tool_call>'
    )
    assert prompt_tools.assistant_text("Looking.", [call, call]).count("<tool_call>") == 2
    assert prompt_tools.assistant_text("Looking.", [call]).startswith("Looking.\n<tool_call>")


def test_tool_results_come_back_as_labelled_user_text():
    assert prompt_tools.result_text("get_form_state", '{"ok": true}') == '[Tool Result: get_form_state]\n{"ok": true}'
    assert prompt_tools.result_text("", "x") == "[Tool Result: tool]\nx"


def feed_all(chunks):
    stage = PromptToolStage()
    items = []
    for chunk in chunks:
        items.extend(stage.feed(chunk))
    items.extend(stage.finish())
    return items


def text_of(items):
    return "".join(item.text for item in items if isinstance(item, TextItem))


def calls_of(items):
    return [item for item in items if isinstance(item, CallItem)]


BLOCK = '<tool_call>{"name": "t", "arguments": {"a": 1}}</tool_call>'


def every_split(text):
    yield [text]
    for cut in range(1, len(text)):
        yield [text[:cut], text[cut:]]
    yield list(text)


def test_a_block_decodes_to_a_call_whichever_way_the_stream_is_chunked():
    raw = "before " + BLOCK + " after"

    for chunks in every_split(raw):
        items = feed_all(chunks)
        assert text_of(items) == "before  after", chunks
        assert [(c.name, c.args) for c in calls_of(items)] == [("t", {"a": 1})], chunks


def test_text_is_forwarded_as_it_arrives_when_no_tag_is_in_sight():
    stage = PromptToolStage()

    assert stage.feed("hello ") == [TextItem("hello ")]
    assert stage.feed("world <") == [TextItem("world ")]
    assert stage.feed("b>") == [TextItem("<b>")]


def test_text_inside_an_open_block_is_withheld_until_it_closes():
    stage = PromptToolStage()

    assert stage.feed('<tool_call>{"name": "t",') == []
    assert stage.feed(' "arguments": {}}') == []
    assert [type(i) for i in stage.feed("</tool_call>x")] == [CallItem, TextItem]


def test_two_blocks_in_one_chunk_are_two_calls_in_order():
    items = feed_all([BLOCK + "\n" + BLOCK.replace('"t"', '"u"')])

    assert [c.name for c in calls_of(items)] == ["t", "u"]


@pytest.mark.parametrize(
    "block, name, args",
    [
        ('<tool_call>{"function": {"name": "t", "arguments": {"a": 1}}}</tool_call>', "t", {"a": 1}),
        ('<tool_call>{"name": "t", "parameters": {"a": 1}}</tool_call>', "t", {"a": 1}),
        ('<tool_call>{"name": "t"}</tool_call>', "t", {}),
        ('<tool_call>{"name": "t", "arguments": "{\\"a\\": 1}"}</tool_call>', "t", {"a": 1}),
        ('<tool_call>{"name": "t", "arguments": "{oops"}</tool_call>', "t", None),
        ('<tool_call>  {"name": <|"|>t<|"|>, "arguments": {}}  </tool_call>', "t", {}),
    ],
)
def test_the_accepted_call_shapes(block, name, args):
    (call,) = calls_of(feed_all([block]))

    assert (call.name, call.args) == (name, args)


@pytest.mark.parametrize(
    "block",
    [
        '<tool_call>{"name": </tool_call>',
        "<tool_call>not json</tool_call>",
        '<tool_call>{"arguments": {}}</tool_call>',
        "<tool_call>[1, 2]</tool_call>",
        '<tool_call>{"name": ""}</tool_call>',
    ],
)
def test_blocks_that_are_not_calls_come_back_as_the_text_they_were(block):
    items = feed_all([block])

    assert calls_of(items) == []
    assert text_of(items) == block


def test_an_unterminated_block_is_released_as_text_at_the_end():
    items = feed_all(['ok <tool_call>{"name": "t"'])

    assert calls_of(items) == [] and text_of(items) == 'ok <tool_call>{"name": "t"'


def test_the_raw_argument_text_is_kept_for_a_call_whose_arguments_do_not_parse():
    (call,) = calls_of(feed_all(['<tool_call>{"name": "t", "arguments": "{oops"}</tool_call>']))

    assert call.raw == "{oops"


def test_a_registered_tool_action_tag_is_ordinary_text_here():
    raw = '<tool_action type="get_form_state"></tool_action>'

    assert text_of(feed_all([raw])) == raw
