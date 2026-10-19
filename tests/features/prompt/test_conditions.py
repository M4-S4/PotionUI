import random
import time

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
        assert parse_condition('$a = "' + "x" * 80 + '"') is not None
        assert parse_condition('$a = "' + "x" * 81 + '"') is None
        assert parse_condition('$a = ""') is None

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
        ok = "<<if $x = 1>>a" + "".join(f"<<else if $x = {n}>>b" for n in range(7)) + "<<end>>"
        assert len(parse_template(ok).blocks) == 1
        too_many = "<<if $x = 1>>a" + "".join(f"<<else if $x = {n}>>b" for n in range(8)) + "<<end>>"
        parsed = parse_template(too_many)
        assert parsed.blocks == ()
        assert "limit" in kinds(parsed.errors)
        assert "<<" not in parsed.skeleton

    def test_branch_limit_counts_else(self):
        ok = "<<if $x = 1>>a" + "".join(f"<<else if $x = {n}>>b" for n in range(6)) + "<<else>>c<<end>>"
        assert len(parse_template(ok).blocks) == 1
        over = "<<if $x = 1>>a" + "".join(f"<<else if $x = {n}>>b" for n in range(7)) + "<<else>>c<<end>>"
        assert parse_template(over).blocks == ()

    def test_block_limit(self):
        one = '[<<if $x = "y">>A<<end>>]'
        parsed = parse_template(one * 32)
        assert len(parsed.blocks) == 32
        parsed = parse_template(one * (33))
        assert len(parsed.blocks) == 32
        assert "limit" in kinds(parsed.errors)
        text, _ = render_conditions(one * (33), {"x": "y"})
        assert text == "[A]" * (33)
        assert "<<" not in text

    def test_head_length_limit(self):
        ok = "<<if $x in " + ",".join('"' + "v" * 17 + '"' for _ in range(12)) + ">>A<<else>>B<<end>>"
        assert render_conditions(ok, {"x": "v" * 17})[0] == "A"
        long_head = "<<if $x in " + ",".join('"' + "v" * 20 + '"' for _ in range(12)) + ">>A<<else>>B<<end>>"
        text, errors = render_conditions(long_head, {"x": "v" * 20})
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
            assert len(parsed.blocks) <= 32
            assert cond._tokenize(parsed.skeleton)[0] == []
        assert elapsed < 2

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
