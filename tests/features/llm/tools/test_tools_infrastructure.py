import pytest
from typing import Any, Dict

from src.features.llm.harness.call_filter import ToolCallTextFilter
from src.features.llm.tools.base import (
    BaseTool, ToolApprovalPreview, ToolResult, ToolSource, ToolContext, ToolExecution,
    serialize_approval_preview,
)
from src.features.llm.tools.registry import ToolRegistry
from src.features.llm.tools.executor import ToolExecutor, bound_tool_result_content


class EchoTool(BaseTool):
    """Simple tool that echoes the 'message' argument back."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echoes a message back."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Message to echo"}
            },
            "required": ["message"],
        }

    async def execute(self, context: ToolContext, **kwargs) -> ToolResult:
        message = kwargs.get("message", "")
        return ToolResult(success=True, data=f"echo: {message}")


class FailingTool(BaseTool):
    """Tool that always raises an exception."""

    @property
    def name(self) -> str:
        return "fail"

    @property
    def description(self) -> str:
        return "Always fails."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, context: ToolContext, **kwargs) -> ToolResult:
        raise RuntimeError("intentional failure")


def make_context(user_id: str = "user-1") -> ToolContext:
    return ToolContext(user_id=user_id)


class TestSerializeApprovalPreview:
    def test_none_serializes_to_none(self):
        assert serialize_approval_preview(None) is None

    def test_legacy_only_preview_has_none_for_new_fields(self):
        preview = ToolApprovalPreview(action="Remove", target="from category camera", items=["a", "b"])
        serialized = serialize_approval_preview(preview)
        assert serialized["action"] == "Remove"
        assert serialized["target"] == "from category camera"
        assert serialized["items"] == ["a", "b"]
        assert serialized["kind"] is None
        assert serialized["summary"] is None
        assert serialized["fields"] is None
        assert serialized["text_blocks"] is None
        assert serialized["rows"] is None

    def test_round_trips_every_new_field(self):
        preview = ToolApprovalPreview(
            action="Start generation",
            target="sdxl/base",
            kind="generation",
            summary="a red fox in snow",
            fields=[{"label": "Steps", "value": "30", "old": "20"}],
            text_blocks=[{"label": "Prompt", "text": "a red fox in snow"}],
            rows=[{"range": "0:00–0:04.5", "text": "sunrise"}],
        )
        serialized = serialize_approval_preview(preview)
        assert serialized["kind"] == "generation"
        assert serialized["summary"] == "a red fox in snow"
        assert serialized["fields"] == [{"label": "Steps", "value": "30", "old": "20"}]
        assert serialized["text_blocks"] == [{"label": "Prompt", "text": "a red fox in snow"}]
        assert serialized["rows"] == [{"range": "0:00–0:04.5", "text": "sunrise"}]


class TestToolExecution:
    def test_fields(self):
        result = ToolResult(success=True, data="ok")
        exec_ = ToolExecution(tool_name="echo", arguments={"message": "hi"}, result=result, duration_ms=42)
        assert exec_.tool_name == "echo"
        assert exec_.arguments == {"message": "hi"}
        assert exec_.result is result
        assert exec_.duration_ms == 42


class TestBaseTool:
    @pytest.mark.asyncio
    async def test_execute_returns_result(self):
        tool = EchoTool()
        ctx = make_context()
        result = await tool.execute(ctx, message="hello")
        assert result.success is True
        assert result.data == "echo: hello"

    def test_to_schema_structure(self):
        tool = EchoTool()
        schema = tool.to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "echo"
        assert schema["function"]["description"] == "Echoes a message back."
        assert "parameters" in schema["function"]


    def test_is_available_defaults_to_true_regardless_of_form_state(self):
        tool = EchoTool()
        assert tool.is_available(None) is True
        assert tool.is_available({}) is True
        assert tool.is_available({"video_director": {"active": False}}) is True


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = EchoTool()
        registry.register(tool)
        assert registry.get("echo") is tool

    def test_get_missing_returns_none(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_get_all(self):
        registry = ToolRegistry()
        echo = EchoTool()
        fail = FailingTool()
        registry.register(echo)
        registry.register(fail)
        all_tools = registry.get_all()
        assert len(all_tools) == 2
        assert echo in all_tools
        assert fail in all_tools

    def test_replace_existing_tool(self):
        registry = ToolRegistry()
        tool1 = EchoTool()
        tool2 = EchoTool()
        registry.register(tool1)
        registry.register(tool2)
        assert registry.get("echo") is tool2

    def test_get_schemas(self):
        registry = ToolRegistry()
        registry.register(EchoTool())
        schemas = registry.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "echo"

    def test_get_schemas_empty(self):
        registry = ToolRegistry()
        assert registry.get_schemas() == []

    def test_register_tracks_source_and_unregister_source(self):
        registry = ToolRegistry()
        registry.register(EchoTool(), source="plugin-x")
        registry.register(FailingTool())  # builtin source
        removed = registry.unregister_source("plugin-x")
        assert removed == 1
        assert registry.get("echo") is None
        assert registry.get("fail") is not None

    def test_unregister_source_no_match(self):
        registry = ToolRegistry()
        registry.register(EchoTool())
        assert registry.unregister_source("nope") == 0
        assert registry.get("echo") is not None

    def _mode(self, mode_id="generation", tool_names=None):
        from src.features.chat.modes import ChatMode
        return ChatMode(id=mode_id, name=mode_id, tool_names=tool_names or [])

    def test_get_for_mode_includes_global_tools(self):
        registry = ToolRegistry()
        registry.register(EchoTool())  # modes = None -> global
        tools = registry.get_for_mode(self._mode("anything"))
        assert [t.name for t in tools] == ["echo"]

    def test_get_for_mode_includes_declared_mode(self):
        class GenTool(EchoTool):
            modes = ["generation"]
        registry = ToolRegistry()
        registry.register(GenTool())
        assert len(registry.get_for_mode(self._mode("generation"))) == 1
        assert len(registry.get_for_mode(self._mode("other"))) == 0

    def test_get_for_mode_includes_mode_tool_names(self):
        class GenTool(EchoTool):
            modes = ["generation"]
        registry = ToolRegistry()
        registry.register(GenTool())
        # 'other' mode borrows the tool explicitly via tool_names
        mode = self._mode("other", tool_names=["echo"])
        assert [t.name for t in registry.get_for_mode(mode)] == ["echo"]

    def test_get_tool_hints_text(self):
        """get_tool_hints_text should return only the hints portion."""
        class HintedTool(EchoTool):
            @property
            def hint(self) -> str:
                return "Use when user asks to echo."

        registry = ToolRegistry()
        registry.register(HintedTool())
        hints = registry.get_tool_hints_text()

        assert hints == "- echo: Use when user asks to echo."

    def test_get_tool_hints_text_empty(self):
        registry = ToolRegistry()
        assert registry.get_tool_hints_text() == ""


class TestToolRegistryFiltered:
    """Tests for the optional `names` parameter on registry methods."""

    def _make_hinted_tool(self, name: str, hint: str):
        """Create a tool with a custom name and hint."""
        class _Tool(EchoTool):
            @property
            def name(self_inner) -> str:
                return name

            @property
            def hint(self_inner) -> str:
                return hint
        return _Tool()

    def test_get_schemas_with_names_filter(self):
        registry = ToolRegistry()
        registry.register(EchoTool())
        registry.register(FailingTool())
        schemas = registry.get_schemas(names=["echo"])
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "echo"

    def test_get_schemas_with_empty_names_returns_nothing(self):
        registry = ToolRegistry()
        registry.register(EchoTool())
        schemas = registry.get_schemas(names=[])
        assert len(schemas) == 0

    def test_get_schemas_none_returns_all(self):
        registry = ToolRegistry()
        registry.register(EchoTool())
        registry.register(FailingTool())
        schemas = registry.get_schemas(names=None)
        assert len(schemas) == 2

    def test_get_schemas_nonexistent_name(self):
        registry = ToolRegistry()
        registry.register(EchoTool())
        schemas = registry.get_schemas(names=["nonexistent"])
        assert len(schemas) == 0

    def test_get_tool_hints_text_filtered(self):
        registry = ToolRegistry()
        tool_a = self._make_hinted_tool("tool_a", "Hint A")
        tool_b = self._make_hinted_tool("tool_b", "Hint B")
        registry.register(tool_a)
        registry.register(tool_b)
        hints = registry.get_tool_hints_text(names=["tool_a"])
        assert "tool_a: Hint A" in hints
        assert "tool_b" not in hints

    def test_get_tool_hints_text_none_returns_all(self):
        registry = ToolRegistry()
        tool_a = self._make_hinted_tool("tool_a", "Hint A")
        tool_b = self._make_hinted_tool("tool_b", "Hint B")
        registry.register(tool_a)
        registry.register(tool_b)
        hints = registry.get_tool_hints_text(names=None)
        assert "tool_a" in hints
        assert "tool_b" in hints

    def _make_conditional_tool(self, name, description):
        class _Tool(BaseTool):
            @property
            def name(self_inner):
                return name

            @property
            def description(self_inner):
                return description

            @property
            def parameters(self_inner):
                return {"type": "object", "properties": {}}

            async def execute(self_inner, context, **kwargs):
                return ToolResult(success=True, data="")

        return _Tool()

    def test_get_schemas_resolves_description_cross_reference_when_present(self):
        registry = ToolRegistry()
        registry.register(self._make_conditional_tool(
            "run_it", "Do it.{{#if helper}} Call helper first.{{/if}}"
        ))
        registry.register(self._make_conditional_tool("helper", "A helper."))
        schema = registry.get_schemas(names=["run_it", "helper"])[0]
        assert "Call helper first." in schema["function"]["description"]

    def test_get_schemas_drops_description_cross_reference_when_absent(self):
        registry = ToolRegistry()
        registry.register(self._make_conditional_tool(
            "run_it", "Do it.{{#if helper}} Call helper first.{{/if}}"
        ))
        schema = registry.get_schemas(names=["run_it"])[0]
        assert schema["function"]["description"] == "Do it."
        assert "helper" not in schema["function"]["description"]

    def test_get_schemas_none_resolves_against_all_registered(self):
        registry = ToolRegistry()
        registry.register(self._make_conditional_tool(
            "run_it", "Do it.{{#if helper}} Call helper first.{{/if}}"
        ))
        registry.register(self._make_conditional_tool("helper", "A helper."))
        schema = next(s for s in registry.get_schemas(names=None) if s["function"]["name"] == "run_it")
        # helper is registered, so its cross-reference survives.
        assert "Call helper first." in schema["function"]["description"]


class TestXmlToolCallParsing:
    def test_parse_xml_tool_call_with_name_format(self):
        from src.features.llm.tools.workflow import parse_xml_tool_calls as _parse_xml_tool_calls
        content = '<tool_call>\n{"name": "echo", "arguments": {"message": "hi"}}\n</tool_call>'
        calls = _parse_xml_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "echo"
        assert calls[0]["function"]["arguments"] == {"message": "hi"}

    def test_parse_xml_tool_call_with_function_format(self):
        from src.features.llm.tools.workflow import parse_xml_tool_calls as _parse_xml_tool_calls
        content = '<tool_call>{"function": {"name": "echo", "arguments": {}}}</tool_call>'
        calls = _parse_xml_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "echo"

    def test_parse_multiple_xml_tool_calls(self):
        from src.features.llm.tools.workflow import parse_xml_tool_calls as _parse_xml_tool_calls
        content = (
            '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
            'some text'
            '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
        )
        calls = _parse_xml_tool_calls(content)
        assert len(calls) == 2

    def test_parse_invalid_json_skipped(self):
        from src.features.llm.tools.workflow import parse_xml_tool_calls as _parse_xml_tool_calls
        content = '<tool_call>not valid json</tool_call>'
        calls = _parse_xml_tool_calls(content)
        assert len(calls) == 0

    def test_parse_multiple_xml_tool_calls_preserves_order(self):
        from src.features.llm.tools.workflow import parse_xml_tool_calls as _parse_xml_tool_calls
        content = (
            '<tool_call>{"name": "a", "arguments": {"n": 1}}</tool_call>'
            'some text between'
            '<tool_call>{"name": "b", "arguments": {"n": 2}}</tool_call>'
            '<tool_call>{"name": "c", "arguments": {"n": 3}}</tool_call>'
        )
        calls = _parse_xml_tool_calls(content)
        assert [c["function"]["name"] for c in calls] == ["a", "b", "c"]

    def test_parse_xml_tool_call_with_mangled_quote_tokens(self):
        """The same tokenizer artifact tool_call_rescue demangles for
        <tool_action> tags shows up inside a well-formed <tool_call> block
        too -- the complete call must still parse, not be dropped."""
        from src.features.llm.tools.workflow import parse_xml_tool_calls as _parse_xml_tool_calls
        content = (
            '<tool_call>{<|"|>name<|"|>: <|"|>echo<|"|>, <|"|>arguments<|"|>: '
            '{<|"|>message<|"|>: <|"|>hi<|"|>}}</tool_call>'
        )
        calls = _parse_xml_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "echo"
        assert calls[0]["function"]["arguments"] == {"message": "hi"}

    def test_parse_xml_tool_call_accepts_parameters_key(self):
        """A model that writes 'parameters' (the schema's own field name)
        instead of 'arguments' must still dispatch with the real payload,
        not an empty arguments dict."""
        from src.features.llm.tools.workflow import parse_xml_tool_calls as _parse_xml_tool_calls
        content = '<tool_call>{"name": "echo", "parameters": {"message": "hi"}}</tool_call>'
        calls = _parse_xml_tool_calls(content)
        assert len(calls) == 1
        assert calls[0]["function"]["arguments"] == {"message": "hi"}

    def test_parse_no_xml_returns_empty(self):
        from src.features.llm.tools.workflow import parse_xml_tool_calls as _parse_xml_tool_calls
        content = 'just regular text'
        calls = _parse_xml_tool_calls(content)
        assert len(calls) == 0

    def test_strip_tool_call_xml(self):
        from src.features.llm.tools.workflow import strip_tool_call_xml
        content = 'hello <tool_call>{"name": "x"}</tool_call> world'
        assert strip_tool_call_xml(content) == 'hello  world'

    def test_strip_tool_call_xml_multiline(self):
        from src.features.llm.tools.workflow import strip_tool_call_xml
        content = 'before\n<tool_call>\n{"name": "x"}\n</tool_call>\nafter'
        assert strip_tool_call_xml(content) == 'before\n\nafter'

    def test_strip_preserves_clean_content(self):
        from src.features.llm.tools.workflow import strip_tool_call_xml
        content = 'no xml here'
        assert strip_tool_call_xml(content) == 'no xml here'


class ApprovalTool(BaseTool):
    @property
    def name(self) -> str:
        return "approval_tool"

    @property
    def description(self) -> str:
        return "A tool that needs approval."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}

    @property
    def requires_approval(self) -> bool:
        return True

    async def execute(self, context: ToolContext, **kwargs) -> ToolResult:
        return ToolResult(
            success=True,
            data=f"Preview: would apply value={kwargs.get('value', '')}",
        )

    async def execute_confirmed(self, context: ToolContext, **kwargs) -> ToolResult:
        return ToolResult(
            success=True,
            data=f"Applied: value={kwargs.get('value', '')}",
        )


class ApprovalToolWithFailingPreview(ApprovalTool):
    @property
    def name(self) -> str:
        return "fail_preview_tool"

    async def execute(self, context: ToolContext, **kwargs) -> ToolResult:
        return ToolResult(success=False, data="", error="preview failed")


class TestBaseToolApproval:
    def test_requires_approval_default_false(self):
        assert EchoTool().requires_approval is False

    def test_requires_approval_overridden_true(self):
        assert ApprovalTool().requires_approval is True

    @pytest.mark.asyncio
    async def test_execute_confirmed_raises_by_default(self):
        tool = EchoTool()
        ctx = make_context()
        with pytest.raises(NotImplementedError):
            await tool.execute_confirmed(ctx)

    @pytest.mark.asyncio
    async def test_execute_confirmed_works_when_overridden(self):
        tool = ApprovalTool()
        ctx = make_context()
        result = await tool.execute_confirmed(ctx, value="test")
        assert result.success is True
        assert "Applied" in result.data


class TestToolExecutorAllowedTools:
    def _make_executor(self):
        registry = ToolRegistry()
        registry.register(EchoTool())
        registry.register(FailingTool())
        return ToolExecutor(tool_registry=registry), None

    @pytest.mark.asyncio
    async def test_execute_tool_rejects_disabled_tool(self):
        executor, _ = self._make_executor()
        result, pending = await executor._execute_tool(
            "echo", make_context(), {"message": "hi"}, allowed_tools=["fail"]
        )
        assert result.success is False
        assert "do not call it again" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_allows_enabled_tool(self):
        executor, _ = self._make_executor()
        result, pending = await executor._execute_tool(
            "echo", make_context(), {"message": "hi"}, allowed_tools=["echo"]
        )
        assert result.success is True
        assert result.data == "echo: hi"

    @pytest.mark.asyncio
    async def test_rejection_message_lists_available_tools(self):
        executor, _ = self._make_executor()
        result, _ = await executor._execute_tool(
            "echo", make_context(), {"message": "hi"}, allowed_tools=["fail", "other"]
        )
        assert result.success is False
        assert result.error == "Tool 'echo' is unavailable; do not call it again. Use: fail, other."

    @pytest.mark.asyncio
    async def test_rejection_message_when_no_tools_available(self):
        executor, _ = self._make_executor()
        result, _ = await executor._execute_tool(
            "echo", make_context(), {"message": "hi"}, allowed_tools=[]
        )
        assert result.success is False
        assert result.error == "No tools are available in this conversation; answer without calling tools."

    @pytest.mark.asyncio
    async def test_execute_tool_none_allows_all(self):
        executor, _ = self._make_executor()
        result, pending = await executor._execute_tool(
            "echo", make_context(), {"message": "hi"}, allowed_tools=None
        )
        assert result.success is True


class TestToolExecutorConfirmed:
    def _make_executor(self, tool: BaseTool):
        registry = ToolRegistry()
        registry.register(tool)
        return ToolExecutor(tool_registry=registry), None

    @pytest.mark.asyncio
    async def test_execute_tool_confirmed_success(self):
        executor, _ = self._make_executor(ApprovalTool())
        result = await executor.execute_tool_confirmed(
            "approval_tool", make_context(), {"value": "hello"}
        )
        assert result.success is True
        assert "Applied" in result.data

    @pytest.mark.asyncio
    async def test_execute_tool_confirmed_unknown_tool(self):
        executor, _ = self._make_executor(ApprovalTool())
        result = await executor.execute_tool_confirmed(
            "nonexistent", make_context(), {}
        )
        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_confirmed_non_approval_tool(self):
        executor, _ = self._make_executor(EchoTool())
        result = await executor.execute_tool_confirmed(
            "echo", make_context(), {"message": "hi"}
        )
        assert result.success is False
        assert "does not require approval" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_confirmed_execute_confirmed_raises(self):
        """If execute_confirmed raises, the error is caught and returned."""
        class BrokenConfirmTool(ApprovalTool):
            @property
            def name(self) -> str:
                return "broken_confirm"

            async def execute_confirmed(self, context: ToolContext, **kwargs) -> ToolResult:
                raise RuntimeError("confirm boom")

        executor, _ = self._make_executor(BrokenConfirmTool())
        result = await executor.execute_tool_confirmed("broken_confirm", make_context(), {})
        assert result.success is False
        assert "confirm boom" not in result.error
        assert "broken_confirm" in result.error
        assert result.error != "Confirmed execution failed: confirm boom"


class TestStreamToolCallFilter:
    def _make(self):
        return ToolCallTextFilter()

    def test_ordinary_text_forwards_immediately_with_no_holdback(self):
        f = self._make()
        assert f.feed("Hel") == [("text", "Hel")]
        assert f.feed("lo!") == [("text", "lo!")]
        assert f.flush() == ""

    def test_complete_block_in_one_chunk_is_suppressed_and_returned_whole(self):
        f = self._make()
        call = '<tool_call>{"name": "echo", "arguments": {}}</tool_call>'
        segments = f.feed(f"before {call} after")
        assert segments == [("text", "before "), ("block", call), ("text", " after")]
        assert not f.suppressing

    def test_open_tag_split_across_two_chunks_is_never_forwarded(self):
        f = self._make()
        first = f.feed("hi <tool_c")
        assert first == [("text", "hi ")]
        second = f.feed('all>{"name": "echo", "arguments": {}}</tool_call> done')
        assert second == [
            ("block", '<tool_call>{"name": "echo", "arguments": {}}</tool_call>'),
            ("text", " done"),
        ]

    def test_unclosed_block_is_never_returned_and_flush_stays_hidden(self):
        f = self._make()
        segments = f.feed('before <tool_call>{"name": "echo"')
        assert segments == [("text", "before ")]
        assert f.suppressing is True
        assert f.flush() == ""

    def test_multiple_blocks_in_one_chunk_return_in_order(self):
        f = self._make()
        call_a = '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        call_b = '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
        segments = f.feed(f"{call_a}mid{call_b}")
        assert segments == [("block", call_a), ("text", "mid"), ("block", call_b)]

    def test_trailing_angle_bracket_in_prose_is_not_held_back_forever(self):
        f = self._make()
        assert f.feed("compare a < b here") == [("text", "compare a < b here")]

    def test_tool_action_naming_a_registered_tool_is_suppressed_as_a_block(self):
        f = ToolCallTextFilter({"propose_form_changes"})
        block = '<tool_action type="propose_form_changes">\n{"ops": []}\n</tool_action>'
        segments = f.feed(f"before {block} after")
        assert segments == [("text", "before "), ("block", block), ("text", " after")]
        assert not f.suppressing

    def test_tool_action_for_an_unregistered_type_is_forwarded_as_text(self):
        f = ToolCallTextFilter({"propose_form_changes"})
        block = '<tool_action type="update_segment" segment_index="0">a lone hiker</tool_action>'
        assert f.feed(block) == [("text", '<tool_action type="update_segment" segment_index="0">'), ("text", "a lone hiker</tool_action>")]
        assert not f.suppressing

    def test_add_and_delete_segment_tags_are_forwarded_as_text(self):
        f = ToolCallTextFilter({"propose_form_changes"})
        add = '<tool_action type="add_segment" segment_id="a" list="negative">blurry</tool_action>'
        delete = '<tool_action type="delete_segment" segment_index="1" segment_id="b"></tool_action>'
        segments = f.feed(add + delete)
        assert [kind for kind, _ in segments] == ["text"] * len(segments)
        assert "".join(text for _, text in segments) == add + delete
        assert not f.suppressing

    def test_tool_action_open_tag_split_across_chunks_is_held_back_until_type_known(self):
        f = ToolCallTextFilter({"propose_form_changes"})
        first = f.feed('before <tool_action type="prop')
        assert first == [("text", "before ")]
        second = f.feed('ose_form_changes">{"ops": []}</tool_action> after')
        assert second == [
            ("block", '<tool_action type="propose_form_changes">{"ops": []}</tool_action>'),
            ("text", " after"),
        ]

    def test_unclosed_tool_action_open_tag_stays_hidden(self):
        f = ToolCallTextFilter({"propose_form_changes"})
        segments = f.feed('before <tool_action type="propose_form_changes"')
        assert segments == [("text", "before ")]
        assert f.suppressing is True
        assert f.flush() == ""


class TestToolResultBound:
    def test_an_oversized_result_is_cut_with_a_teaching_marker(self):
        bounded = bound_tool_result_content("x" * 20000)
        assert len(bounded) < 9000
        assert bounded.startswith("x" * 100)
        assert "[Result truncated at 8000 characters." in bounded
        assert "Refine the call" in bounded

    def test_a_small_result_is_untouched(self):
        assert bound_tool_result_content("echo: hi") == "echo: hi"
