import json

import pytest

from src.features.llm.tools.operation_args import collapse_repeats, operations_from

EXAMPLE = '<tool_call>{"name": "demo", "arguments": {"operations": [{"op": "go"}]}}</tool_call>'
OP = {"op": "go", "name": "x"}
EMPTY = "nothing to do"


class TestOperationsFrom:
    def test_a_list_passes_through_untouched(self):
        operations = [OP, "stray"]
        assert operations_from(operations, EXAMPLE, EMPTY) == (operations, None)

    def test_a_single_object_is_wrapped(self):
        assert operations_from(OP, EXAMPLE, EMPTY) == ([OP], None)

    def test_a_json_string_array_is_decoded(self):
        assert operations_from(json.dumps([OP, OP]), EXAMPLE, EMPTY) == ([OP, OP], None)

    def test_a_json_string_object_is_decoded_and_wrapped(self):
        assert operations_from(json.dumps(OP), EXAMPLE, EMPTY) == ([OP], None)

    def test_surrounding_whitespace_is_ignored(self):
        assert operations_from("\n  " + json.dumps([OP]) + "  \n", EXAMPLE, EMPTY) == ([OP], None)

    def test_a_python_literal_string_is_decoded(self):
        assert operations_from("[{'op': 'go', 'name': 'x'}]", EXAMPLE, EMPTY) == ([OP], None)

    def test_bare_object_keys_are_decoded(self):
        assert operations_from('[{op: "go", name: "x"}]', EXAMPLE, EMPTY) == ([OP], None)

    def test_mangled_quote_tokens_are_decoded(self):
        text = '[{"op": <|"|>go<|"|>, "name": "x"}]'
        assert operations_from(text, EXAMPLE, EMPTY) == ([OP], None)

    @pytest.mark.parametrize(
        "text",
        ['[{"op": "go", "name": "x"', "make the variables moodier", "   ", '"[1, 2]"', "42", "null"],
    )
    def test_text_that_is_not_an_array_gets_one_error_with_the_example(self, text):
        operations, problem = operations_from(text, EXAMPLE, EMPTY)
        assert operations is None
        assert problem.count(EXAMPLE) == 1
        assert "ARRAY" in problem

    def test_a_cut_off_array_says_why_it_could_not_be_read(self):
        _operations, problem = operations_from('[{"op": "go", "name": "x"', EXAMPLE, EMPTY)
        assert "could not be read" in problem

    @pytest.mark.parametrize("value, type_name", [(5, "int"), (2.5, "float"), (True, "bool")])
    def test_other_types_get_one_error_naming_the_type(self, value, type_name):
        operations, problem = operations_from(value, EXAMPLE, EMPTY)
        assert operations is None
        assert f"got {type_name}" in problem
        assert problem.count(EXAMPLE) == 1


    @pytest.mark.parametrize("value", [None, [], {}, "", "[]", " [ ] ", 0, False])
    def test_nothing_to_apply_gets_the_empty_error(self, value):
        assert operations_from(value, EXAMPLE, EMPTY) == (None, EMPTY)

    def test_an_object_without_fields_is_still_one_operation(self):
        assert operations_from("{}", EXAMPLE, EMPTY) == ([{}], None)


class TestCollapseRepeats:
    def test_distinct_messages_are_untouched(self):
        assert collapse_repeats(["a", "b"]) == ["a", "b"]

    def test_repeats_become_one_message_with_a_count(self):
        assert collapse_repeats(["a"] * 412) == ["a (×412)"]

    def test_order_of_first_appearance_is_kept(self):
        assert collapse_repeats(["b", "a", "b", "c", "a", "b"]) == ["b (×3)", "a (×2)", "c"]

    def test_nothing_in_nothing_out(self):
        assert collapse_repeats([]) == []
