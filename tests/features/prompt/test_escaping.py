import time

import pytest

from src.features.prompt.escaping import protect_literals, restore_literals
from src.features.prompt.expander import MALFORMED_TEMPLATE_WARNING, expand_prompts


def one(template, variables=None, negative=""):
    return expand_prompts(template, negative, count=1, base_seed=3, variables=variables)[0]


class TestProtectLiterals:
    @pytest.mark.parametrize("text", [
        "{a|b}", "${x}", "${x=1}", "${x:fallback}", "{2$$a|b|c}", "{2-3$$ and $$a|b|c}",
        "%{wrap$$inner}", "__wild__", "{1::a|2::b}", "plain",
    ])
    def test_grammar_is_left_untouched(self, text):
        assert protect_literals(text) == text

    @pytest.mark.parametrize("text", ["$100", "100%", "$", "%", "$$", "%%", "a $ b", "5% off $5", "100%{a|b}", "#", "#5", "#art #photo"])
    def test_literal_symbols_are_replaced_and_roundtrip(self, text):
        protected = protect_literals(text)
        assert "$" not in protected and "%" not in protected and "#" not in protected
        assert restore_literals(protected) == text

    def test_double_dollar_outside_braces_is_literal(self):
        assert "$$" not in protect_literals("a $$ b")


class TestLiteralSymbolsExpand:
    def test_dollar_amount_does_not_disable_choices_and_variables(self):
        r = one("a $100 bill, {red|red} dress, ${c} hat", {"c": "green"})
        assert r.positive == "a $100 bill, red dress, green hat"

    def test_percent_does_not_disable_choices_and_variables(self):
        r = one("100% cotton, {red|red} dress, ${c} hat", {"c": "green"})
        assert r.positive == "100% cotton, red dress, green hat"

    @pytest.mark.parametrize("text", ["$", "%", "$ x", "x $", "% x", "x %", "$$", "a $$ b", "%%", "$5%"])
    def test_lone_symbols_pass_through(self, text):
        assert one(text).positive == text.strip()

    def test_symbols_in_negative_prompt(self):
        assert one("x", negative="$ {a|a} %").negative == "$ a %"

    def test_variable_values_keep_literal_symbols(self):
        assert one("cost ${p}", {"p": "$5 or 10%"}).positive == "cost $5 or 10%"

    def test_symbols_inside_choice_options(self):
        assert one("{$5|$5} {10%|10%}").positive == "$5 10%"

    def test_percent_before_choice_group_is_literal(self):
        assert one("100%{a|a}").positive == "100%a"


class TestGrammarStillWorks:
    def test_variable_assignment_and_use(self):
        assert one("${x=big} a ${x} cat $5").positive == "a big cat $5"

    def test_counted_choice(self):
        assert one("{2$$ and $$a|b|c} $5").positive.count(" and ") == 1

    def test_weights(self):
        assert one("{1::a|0::b} 100%").positive == "a 100%"

    def test_wrap_command(self):
        assert one("%{[$$x]} $1").positive == "[x] $1"

    def test_wildcard_text_survives(self):
        assert "__foo__" in one("__foo__ $1").positive


class TestMalformedWarning:
    def test_malformed_template_reports_once_per_generation(self):
        seen = []
        out = expand_prompts("a {broken", "", count=4, base_seed=1, on_warning=seen.append)
        assert seen == [MALFORMED_TEMPLATE_WARNING]
        assert all(e.positive == "a {broken" for e in out)

    def test_malformed_negative_reports(self):
        seen = []
        expand_prompts("ok", "bad {", count=1, base_seed=1, on_warning=seen.append)
        assert seen == [MALFORMED_TEMPLATE_WARNING]

    def test_valid_template_with_symbols_reports_nothing(self):
        seen = []
        expand_prompts("$5 {a|b} 100%", "", count=3, base_seed=1, on_warning=seen.append)
        assert seen == []

    def test_no_callback_is_fine(self):
        assert one("a {broken").positive == "a {broken"


class TestHashIsLiteral:
    def test_number_sign_before_digit_keeps_the_rest(self):
        assert one("issue #5 {a|a}").positive == "issue #5 a"

    def test_hashtags(self):
        assert one("{a|a} #art #photo").positive == "a #art #photo"

    @pytest.mark.parametrize("text", ["#", "# x", "x #", "#x", "##", "x#y"])
    def test_edges_pass_through(self, text):
        assert one(text).positive == text.strip()

    def test_mixed_with_variables_and_symbols(self):
        r = one("#1 ${c} $5 {x|x} 100% #end", {"c": "green"})
        assert r.positive == "#1 green $5 x 100% #end"

    def test_hash_in_variable_value_and_negative(self):
        r = one("tag ${t}", {"t": "#cool"}, negative="#bad {a|a}")
        assert (r.positive, r.negative) == ("tag #cool", "#bad a")

    def test_hash_does_not_warn(self):
        seen = []
        expand_prompts("#1 {a|b}", "", count=2, base_seed=1, on_warning=seen.append)
        assert seen == []


class TestMalformedWithLiterals:
    def test_text_is_returned_exactly_and_warns_once(self):
        seen = []
        template = "$5 100% #1 {broken"
        out = expand_prompts(template, "", count=3, base_seed=1, on_warning=seen.append)
        assert [e.positive for e in out] == [template] * 3
        assert not any(ch in e.positive for e in out for ch in "\ue000\ue001\ue002")
        assert seen == [MALFORMED_TEMPLATE_WARNING]


class TestLinearTime:
    @pytest.mark.parametrize(
        "text",
        ["%{" * 50000, "${" * 50000, "{" * 100000, "%{a$$" * 20000, "$$" * 50000],
        ids=["pct-brace", "dollar-brace", "brace", "pct-brace-dollar", "dollar"],
    )
    def test_adversarial_input_is_fast(self, text):
        start = time.perf_counter()
        protect_literals(text)
        assert time.perf_counter() - start < 3.0
