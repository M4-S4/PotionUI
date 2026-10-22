"""Tests for the generation mode's code-owned system prompt template."""

from src.features.chat.modes import ChatModeRegistry, build_generation_mode
from src.features.prompt.conditions import MAX_TESTS


class TestMemoryPromptBullet:
    def _resolve(self, allowed):
        registry = ChatModeRegistry()
        mode = build_generation_mode()
        return registry.resolve_system_prompt(mode, "- some_tool: hint", allowed)

    def test_update_over_append_instruction_present_when_both_tools_allowed(self):
        prompt = self._resolve(["write_memory", "update_memory"])
        assert "Save the pattern, not the instance" in prompt
        assert "or update_memory when a note in your context already covers it" in prompt
        assert "{{#if" not in prompt
        assert "{{/if" not in prompt

    def test_update_over_append_instruction_absent_without_update_memory(self):
        prompt = self._resolve(["write_memory"])
        assert "Save the pattern, not the instance" in prompt
        assert "update_memory" not in prompt
        assert "{{#if" not in prompt
        assert "{{/if" not in prompt

    def test_memory_bullet_absent_without_write_memory(self):
        prompt = self._resolve(["update_memory"])
        assert "Save the pattern, not the instance" not in prompt
        assert "update_memory" not in prompt
        assert "{{#if" not in prompt
        assert "{{/if" not in prompt


class TestPromptVariablesGuidance:
    def _resolve(self, allowed):
        registry = ChatModeRegistry()
        mode = build_generation_mode()
        return registry.resolve_system_prompt(mode, "- some_tool: hint", allowed)

    def test_manage_prompt_variables_instruction_present_when_allowed(self):
        prompt = self._resolve(["manage_prompt_variables"])
        assert "manage_prompt_variables first" in prompt
        assert "{{#if" not in prompt
        assert "{{/if" not in prompt

    def test_manage_prompt_variables_instruction_absent_when_not_allowed(self):
        prompt = self._resolve([])
        assert "manage_prompt_variables first" not in prompt
        # The base sentence about reusing an existing variable still stands.
        assert "named ${...} placeholders" in prompt
        assert "{{#if" not in prompt
        assert "{{/if" not in prompt

    def test_read_tool_instruction_present_when_allowed(self):
        prompt = self._resolve(["get_prompt_variables"])
        assert "call get_prompt_variables for its exact options" in prompt
        assert "{{#if" not in prompt
        assert "{{/if" not in prompt

    def test_read_tool_instruction_absent_when_not_allowed(self):
        prompt = self._resolve(["manage_prompt_variables", "get_form_state"])
        assert "get_prompt_variables" not in prompt

    def test_write_tool_description_points_at_the_read_tool_only_when_allowed(self):
        from src.features.llm.tools.builtin.prompt_variables_tool import ManagePromptVariablesTool

        text = ManagePromptVariablesTool().description
        assert "{{#if get_prompt_variables}}" in text
        assert "get_form_state first" not in text


class TestConditionGrammarGuidance:
    def _resolve(self, allowed):
        registry = ChatModeRegistry()
        return registry.resolve_system_prompt(build_generation_mode(), "- some_tool: hint", allowed)

    def test_grammar_is_taught_without_any_tool(self):
        prompt = self._resolve([])
        assert "<<if TEST>>text<<else if TEST>>text<<else>>text<<end>>" in prompt
        assert "<<if $age > 65>>grey hair<<else if $age in 18..30>>youthful skin<<end>>" in prompt
        assert '<<if $mood = "noir">>rain-soaked neon<<else if $mood in "sunlit", "golden hour">>' in prompt
        assert '<<if $record = "12\\" vinyl" and $era is not empty>>' in prompt
        assert "never both" in prompt
        assert "never sits inside another one" in prompt
        assert f"Join up to {MAX_TESTS} tests" in prompt
        assert "Never {% if %}, ==, &&, elif or endif." in prompt
        assert "{{#if" not in prompt
        assert "write_condition" not in prompt
        assert "manage_prompt_variables" not in prompt

    def test_every_example_parses_and_teaches_only_valid_grammar(self):
        from src.features.prompt import conditions

        prompt = self._resolve([])
        section = prompt.split("## Prompt conditions", 1)[1].split("## ", 1)[0]
        examples = [line for line in section.splitlines() if line.startswith("<<if $")]
        assert len(examples) == 3
        for example in examples:
            parsed = conditions.parse_template(example)
            assert parsed is not None and not parsed.errors and len(parsed.blocks) == 1

    def test_tool_lines_appear_only_when_offered(self):
        prompt = self._resolve(["write_condition", "manage_prompt_variables", "get_current_segments"])
        assert "with write_condition and paste its text verbatim" in prompt
        assert "An option's `when` in manage_prompt_variables is different" in prompt
        assert "PROMPT STATE lists the tab's variables with their exact values." in prompt
        assert "PROMPT STATE lists" not in self._resolve([])

    @staticmethod
    def _section(prompt):
        return "## Prompt conditions" + prompt.split("## Prompt conditions", 1)[1].split("\n## ", 1)[0]

    def test_the_prompts_mode_teaches_the_same_guide_while_a_prompt_tool_is_offered(self):
        from src.features.chat.modes import build_prompts_mode

        registry = ChatModeRegistry()
        for tool in ("get_prompt", "add_prompt", "edit_prompt"):
            prompt = registry.resolve_system_prompt(build_prompts_mode(), "", [tool])
            assert self._section(prompt) == self._section(self._resolve([]))
            assert "{{#" not in prompt
        assert "## Prompt conditions" not in registry.resolve_system_prompt(
            build_prompts_mode(), "", ["list_prompts", "search_model_prompts", "delete_prompt"]
        )
