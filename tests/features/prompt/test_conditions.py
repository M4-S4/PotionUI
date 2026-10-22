import json
import random
import time
from pathlib import Path

import pytest

from src.features.prompt import conditions as cond
from src.features.prompt.conditions import (
    SENTINEL_CLOSE,
    SENTINEL_OPEN,
    choose_branch,
    parse_condition,
    parse_template,
    render_conditions,
    splice,
)
from src.features.prompt.expander import expand_prompts
from src.platform.resources.prompt_variables import MAX_OPTION_CHARS


def kinds(errors):
    return [e.kind for e in errors]


class TestParseCondition:
    def test_single_test_shape(self):
        parsed = parse_condition('$age > 65')
        assert parsed.joiner == "and"
        (test,) = parsed.tests
        assert (test.var, test.op, test.kind) == ("age", ">", "number")

    def test_joiner_is_recorded(self):
        assert parse_condition('$a = 1 or $b = 2').joiner == "or"
        assert parse_condition('$a = 1 and $b = 2').joiner == "and"

    def test_string_escapes_are_decoded(self):
        parsed = parse_condition(r'$q = "a \"b\" \\ c"')
        assert parsed.tests[0].strings == ('a "b" \\ c',)

    def test_variable_name_cap(self):
        assert parse_condition("$" + "a" * 60 + " = 1") is not None
        assert parse_condition("$" + "a" * 61 + " = 1") is None

    def test_number_literal_length_cap(self):
        assert parse_condition("$a = " + "1" * 20) is not None
        assert parse_condition("$a = " + "1" * 21) is None

    def test_string_length_cap(self):
        assert parse_condition('$a = "' + "x" * cond.MAX_STRING_CHARS + '"') is not None
        assert parse_condition('$a = "' + "x" * (cond.MAX_STRING_CHARS + 1) + '"') is None
        assert parse_condition('$a = ""') is None

    def test_tests_per_head_cap(self):
        def head(count):
            return " and ".join(["$a = 1"] * count)

        assert len(parse_condition(head(cond.MAX_TESTS)).tests) == cond.MAX_TESTS
        assert parse_condition(head(cond.MAX_TESTS + 1)) is None
        assert len(parse_condition(head(cond.MAX_TESTS).replace(" and ", " or ")).tests) == cond.MAX_TESTS
        assert parse_condition(head(cond.MAX_TESTS + 1).replace(" and ", " or ")) is None

    @pytest.mark.parametrize("head", [
        "", "$a", "$a =", "$a = ", "a = 1", "$a  1", "$a = 1 and", "$a = 1 xor $b = 2",
        "$a = 1andd $b = 2", "$a=1", "$a =1", "$a = 1,", "$a in 1", "$a in 1..", "$a in ..2",
        "$a contains 5", "$a contains \"x\",\"y\"", "$a is", "$a is nothing", "$a = 1.", "$a = .5",
        "$a = +5", "$a = 1e3", '$a = "x', '$a = "x\\n"', "$a in \"x\" \"y\"",
    ])
    def test_rejected_heads(self, head):
        assert parse_condition(head) is None

    @pytest.mark.parametrize("head", [
        "$a = 1", "$a = -1.5", "$a != 2", "$a <= 3", "$a >= 3", "$a in 1..2", "$a not in -2..2.5",
        '$a in "x"', '$a not in "x", "y"', '$a contains "x"', '$a not contains "x"',
        "$a is empty", "$a is not empty", "$a = 1   and\t$b = 2",
    ])
    def test_accepted_heads(self, head):
        assert parse_condition(head) is not None


class TestBlockStructure:
    def test_no_marker_returns_none(self):
        assert parse_template("plain text <lora:x:1>") is None
        assert parse_template("<<hello>>") is None

    def test_skeleton_holds_a_sentinel_per_block(self):
        parsed = parse_template('a <<if $x = "y">>B<<end>> c <<if $x = "n">>D<<end>>')
        assert parsed.skeleton == f"a {SENTINEL_OPEN}0{SENTINEL_CLOSE} c {SENTINEL_OPEN}1{SENTINEL_CLOSE}"
        assert [b.index for b in parsed.blocks] == [0, 1]

    def test_branches_in_order(self):
        parsed = parse_template('<<if $x = "a">>A<<else if $x = "b">>B<<else>>Z<<end>>')
        (block,) = parsed.blocks
        assert [(b.kind, b.body) for b in block.branches] == [("if", "A"), ("elseif", "B"), ("else", "Z")]

    def test_reads_collects_every_variable(self):
        parsed = parse_template('<<if $a = 1 and $b = 2>>x<<else if $c = 3>>y<<end>>')
        assert parsed.reads() == {"a", "b", "c"}

    def test_choose_branch_reports_the_index(self):
        parsed = parse_template('<<if $x = "a">>A<<else if $x = "b">>B<<else>>Z<<end>>')
        block = parsed.blocks[0]
        assert choose_branch(block, {"x": "b"}) == (1, "B")
        assert choose_branch(block, {"x": "q"}) == (2, "Z")

    def test_no_branch_chosen(self):
        block = parse_template('<<if $x = "a">>A<<end>>').blocks[0]
        assert choose_branch(block, {"x": "q"}) == (None, "")

    def test_user_supplied_sentinel_characters_are_removed(self):
        parsed = parse_template(f'{SENTINEL_OPEN}7{SENTINEL_CLOSE}<<if $x = "y">>A<<end>>')
        assert parsed.skeleton == f"7{SENTINEL_OPEN}0{SENTINEL_CLOSE}"

    def test_branch_limit(self):
        def block(branches):
            return "<<if $x = 1>>a" + "".join(f"<<else if $x = {n}>>b" for n in range(branches - 1)) + "<<end>>"

        parsed = parse_template(block(cond.MAX_BRANCHES))
        assert len(parsed.blocks) == 1
        assert parsed.errors == ()
        parsed = parse_template(block(cond.MAX_BRANCHES + 1))
        assert parsed.blocks == ()
        assert "limit" in kinds(parsed.errors)
        assert "<<" not in parsed.skeleton

    def test_branch_limit_counts_else(self):
        def block(branches):
            return (
                "<<if $x = 1>>a"
                + "".join(f"<<else if $x = {n}>>b" for n in range(branches - 2))
                + "<<else>>c<<end>>"
            )

        assert len(parse_template(block(cond.MAX_BRANCHES)).blocks) == 1
        assert parse_template(block(cond.MAX_BRANCHES + 1)).blocks == ()

    def test_block_limit(self):
        one = '[<<if $x = "y">>A<<end>>]'
        parsed = parse_template(one * cond.MAX_BLOCKS)
        assert len(parsed.blocks) == cond.MAX_BLOCKS
        assert parsed.errors == ()
        over = one * (cond.MAX_BLOCKS + 1)
        parsed = parse_template(over)
        assert len(parsed.blocks) == cond.MAX_BLOCKS
        assert "limit" in kinds(parsed.errors)
        text, _ = render_conditions(over, {"x": "y"})
        assert text == "[A]" * (cond.MAX_BLOCKS + 1)
        assert "<<" not in text

    def test_head_length_limit(self):
        def block(length):
            head = '$x in "v", "w"'
            return "<<if " + head + " " * (length - len(head)) + ">>A<<else>>B<<end>>"

        assert render_conditions(block(cond.MAX_HEAD_CHARS), {"x": "v"})[0] == "A"
        text, errors = render_conditions(block(cond.MAX_HEAD_CHARS + 1), {"x": "v"})
        assert "malformed_tag" in kinds(errors)
        assert "<<if" not in text

    def test_scan_limit_strips_tags_past_the_limit(self):
        pad = "x" * (20_010)
        parsed = parse_template(pad + '<<if $x = "y">>A<<end>>')
        assert parsed.blocks == ()
        assert parsed.skeleton == pad + ' $x = "y">>A'
        assert kinds(parsed.errors) == ["limit"]

    def test_block_straddling_the_scan_limit_is_dropped(self):
        pad = "x" * (19_995)
        parsed = parse_template('<<if $x = "y">>' + pad + "<<end>>")
        assert parsed.blocks == ()
        assert sorted(kinds(parsed.errors)) == ["limit", "unclosed"]
        assert "<<" not in parsed.skeleton


def _option(length, lead="a "):
    text = (lead + "long hooded cloak woven from silver thread and moonlight, trimmed with fox fur, " * 40)
    return text[: length - 1] + "s"


def _quoted(value):
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


class TestLongChoiceOptions:
    def test_value_cap_holds_the_longest_choice_option(self):
        assert cond.MAX_STRING_CHARS == MAX_OPTION_CHARS

    def test_head_cap_holds_the_longest_escaped_value(self):
        head = "$" + "n" * cond.MAX_NAME_CHARS + " not in " + _quoted('"' * cond.MAX_STRING_CHARS)
        assert len(head) <= cond.MAX_HEAD_CHARS
        assert parse_condition(head).tests[0].strings == ('"' * cond.MAX_STRING_CHARS,)

    @pytest.mark.parametrize("length", [81, 200, MAX_OPTION_CHARS])
    def test_editor_switch_on_a_long_option_round_trips(self, length):
        options = [_option(length), _option(length, "b "), _option(length, 'c "quoted" \\ ')]
        text = (
            f"<<if $look = {_quoted(options[0])}>>A"
            f"<<else if $look = {_quoted(options[1])}>>B"
            f"<<else if $look = {_quoted(options[2])}>>C<<else>>D<<end>>"
        )
        (block,) = parse_template(text).blocks
        assert [b.cond.tests[0].strings for b in block.branches[:3]] == [(o,) for o in options]
        assert [render_conditions(text, {"look": o}) for o in options] == [("A", []), ("B", []), ("C", [])]
        assert render_conditions(text, {"look": "plain"}) == ("D", [])

    def test_an_option_over_the_cap_is_a_bad_head(self):
        option = _option(cond.MAX_STRING_CHARS + 1)
        text, errors = render_conditions(f"<<if $look = {_quoted(option)}>>A<<else>>B<<end>>", {"look": option})
        assert (text, kinds(errors)) == ("B", ["bad_head"])

    def test_corpus_holds_the_editor_output_for_long_options(self):
        corpus = Path(__file__).resolve().parents[2] / "fixtures" / "prompt_conditions" / "evaluate.json"
        cases = {c["name"]: c for c in json.loads(corpus.read_text(encoding="utf-8"))["cases"]}
        lengths = [
            len(cases[name]["values"]["look"])
            for name in (
                "editor_switch_option_81_chars",
                "editor_switch_option_200_chars",
                "editor_switch_option_at_value_cap",
            )
        ]
        assert lengths == [81, 200, MAX_OPTION_CHARS]


class TestSplice:
    def test_nonempty_body_replaces_sentinel_verbatim(self):
        assert splice(f"a {SENTINEL_OPEN}0{SENTINEL_CLOSE} c", [" B "]) == "a  B  c"

    def test_unknown_sentinel_index_is_a_hole(self):
        assert splice(f"a {SENTINEL_OPEN}9{SENTINEL_CLOSE} c", ["x"]) == "a c"

    def test_sentinel_characters_in_a_body_are_dropped(self):
        assert splice(f"{SENTINEL_OPEN}0{SENTINEL_CLOSE}", [f"a{SENTINEL_OPEN}b"]) == "ab"

    def test_repeated_sentinel_fills_every_copy(self):
        assert splice(f"{SENTINEL_OPEN}0{SENTINEL_CLOSE}-{SENTINEL_OPEN}0{SENTINEL_CLOSE}", ["z"]) == "z-z"

    def test_tabs_around_a_hole_are_trimmed(self):
        assert splice(f"a\t \t{SENTINEL_OPEN}0{SENTINEL_CLOSE}\t b", [""]) == "a b"


ALPHABET = [
    "<<if", "<<else", "<<else if", "<<end>>", "<<end", ">>", "<<", "$", "$x", "$age", "\"", "'", "{", "}",
    "|", "%", "\n", " ", "=", "!=", ">", "<", "in", "and", "or", "1", "..", ",", "\\", "a", "b", "empty", "is",
    "", "",
]


class TestRobustness:
    def test_fuzz_never_raises_and_skeleton_and_bodies_hold_no_tags(self):
        rng = random.Random(1234)
        for _ in range(2000):
            text = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 40)))
            rendered, _ = render_conditions(text, {"x": "1", "age": "5"})
            parsed = parse_template(text)
            if parsed is None:
                assert rendered == text
                continue
            assert cond._tokenize(parsed.skeleton)[0] == []
            for block in parsed.blocks:
                for branch in block.branches:
                    assert cond._tokenize(branch.body)[0] == []
            assert SENTINEL_OPEN not in rendered and SENTINEL_CLOSE not in rendered

    def test_well_formed_blocks_never_survive_rendering(self):
        rng = random.Random(99)
        for _ in range(500):
            body = "".join(rng.choice(["a", " ", ",", "b", "\n"]) for _ in range(rng.randint(0, 6)))
            text = f'<<if $x = "{rng.choice("yn")}">>{body}<<else>>z<<end>>'
            rendered, errors = render_conditions(text, {"x": "y"})
            assert errors == []
            assert "<<" not in rendered

    @pytest.mark.parametrize("unit", [
        "<<if ", "<<if $x = \"", "<<if $x = \"a>>", "<<else if $a in \"x\",", "<<", "<<<<<<if $x = 1>>",
        "<<if $x = 1>>a<<else>>", "<<if $x = 1>>", "<<end>>", "<<else>>", "$x = 1 and ",
    ])
    def test_adversarial_input_is_fast(self, unit):
        text = unit * (100_000 // len(unit))
        start = time.perf_counter()
        rendered, _ = render_conditions(text, {"x": "1"})
        elapsed = time.perf_counter() - start
        parsed = parse_template(text)
        assert isinstance(rendered, str)
        if parsed is not None:
            assert len(parsed.blocks) <= cond.MAX_BLOCKS
            assert cond._tokenize(parsed.skeleton)[0] == []
        assert elapsed < 2

    @pytest.mark.parametrize("shape", ["widest_block", "most_blocks"])
    def test_a_template_at_the_caps_expands_quickly(self, shape):
        if shape == "widest_block":
            head = " and ".join(["$a = 1"] * (cond.MAX_TESTS - 1) + ["$a = 2"])
            branches = "".join(f"<<else if {head}>>b{n}" for n in range(cond.MAX_BRANCHES - 2))
            template = f"<<if {head}>>a{branches}<<else>>z<<end>>"
            expected = "z"
        else:
            template = "[<<if $a = 1>>A<<else>>B<<end>>]" * cond.MAX_BLOCKS
            expected = "[A]" * cond.MAX_BLOCKS
        assert len(parse_template(template).blocks) == (1 if shape == "widest_block" else cond.MAX_BLOCKS)
        warnings = []
        start = time.perf_counter()
        expanded = expand_prompts(template, "", count=1, base_seed=7, variables={"a": "1"}, on_warning=warnings.append)
        elapsed = time.perf_counter() - start
        assert warnings == []
        assert expanded[0].positive == expected
        assert elapsed < 1

    def test_deep_nesting_attempt_is_fast_and_clean(self):
        text = '<<if $x = 1>>' * 5000 + "y" + "<<end>>" * 5000
        start = time.perf_counter()
        rendered, errors = render_conditions(text, {"x": "1"})
        assert time.perf_counter() - start < 2
        assert "<<" not in rendered
        assert "nested" in kinds(errors)

    def test_huge_numeric_values_do_not_blow_up(self):
        rendered, _ = render_conditions('<<if $x > 1>>big<<else>>small<<end>>', {"x": "9" * 100000})
        assert rendered == "small"

    def test_no_tag_text_is_not_parsed(self):
        assert parse_template("a" * 100_000) is None
