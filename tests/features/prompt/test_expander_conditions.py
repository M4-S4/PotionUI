import dataclasses
import random

import pytest

from src.features.prompt import expander
from src.features.prompt.expander import CONDITION_WARNING, MALFORMED_TEMPLATE_WARNING, expand_prompts
from tests.features.prompt.test_expander import FakePluginRegistry

DRESS = 'woman, <<if $age > 65>>grey hair<<else>>{freckles|scar}<<end>>, {red|blue|green|black} dress, ${age} years'


def texts(expanded):
    return [e.positive for e in expanded]


def legacy_expand(positive, negative, count, base_seed, variables=None):
    ctx = expander._base_context(variables)
    out = []
    for i in range(count):
        seeded = dataclasses.replace(ctx, rand=random.Random(base_seed + i))
        p = expander._sample_one(seeded, positive).strip()
        n = expander._sample_one(seeded, negative).strip()
        out.append((p, n))
    return out


NO_TAG_TEMPLATES = [
    "a {red|blue|green} dress, {x|y|z} hat, ${hair}",
    "price $100, 100% cotton, #tag, {a|b}",
    "{2$$ and $$a|b|c|d}, ${hair}",
    "<<hello>> <lora:x:1> <Picture 1> {a|b}",
    "(red:1.2) [x] BREAK {0.3::rain|0.7::sun}",
    "{unbalanced brace, ${hair}",
    "",
]


class TestNoTagRegression:
    @pytest.mark.parametrize("template", NO_TAG_TEMPLATES)
    def test_byte_identical_to_the_pre_conditions_loop(self, template):
        variables = {"hair": "{red|blue}"}
        for seed in range(51):
            got = expand_prompts(template, "{n1|n2}, bad", count=1, base_seed=seed, variables=variables)[0]
            assert (got.positive, got.negative) == legacy_expand(template, "{n1|n2}, bad", 1, seed, variables)[0]

    def test_batch_is_identical(self):
        got = expand_prompts("{a|b|c|d}", "{x|y}", count=8, base_seed=40, variables={"v": "1"})
        want = legacy_expand("{a|b|c|d}", "{x|y}", 8, 40, {"v": "1"})
        assert [(e.positive, e.negative) for e in got] == want

    def test_no_condition_warning_without_tags(self):
        warnings = []
        expand_prompts("a {b|c", "", count=1, base_seed=0, on_warning=warnings.append)
        assert warnings == [MALFORMED_TEMPLATE_WARNING]


class TestSentinelStability:
    @pytest.mark.parametrize("seed", range(51))
    def test_flipping_a_variable_changes_only_the_condition_text(self, seed):
        old = expand_prompts(DRESS, "", count=1, base_seed=seed, variables={"age": "70"})[0].positive
        young = expand_prompts(DRESS, "", count=1, base_seed=seed, variables={"age": "30"})[0].positive
        assert old.startswith("woman, grey hair, ")
        assert young.split(", ")[1] in ("freckles", "scar")
        assert old.split(", ")[2] == young.split(", ")[2]
        assert old.endswith("70 years") and young.endswith("30 years")

    def test_dress_colour_matches_a_template_without_the_condition_group(self):
        for seed in range(6):
            with_tags = expand_prompts(DRESS, "", count=1, base_seed=seed, variables={"age": "70"})[0].positive
            plain = expand_prompts(
                "woman, X, {red|blue|green|black} dress, ${age} years", "", count=1, base_seed=seed,
                variables={"age": "70"},
            )[0].positive
            assert with_tags.split(", ")[2] == plain.split(", ")[2]

    def test_same_seed_same_output(self):
        a = expand_prompts(DRESS, "", count=5, base_seed=9, variables={"age": "{20|70}"})
        b = expand_prompts(DRESS, "", count=5, base_seed=9, variables={"age": "{20|70}"})
        assert texts(a) == texts(b)

    def test_image_i_matches_a_single_expansion_at_that_seed(self):
        template = '<<if $age >= 40>>{old|elder}<<else>>{young|kid}<<end>> {a|b}'
        batch = expand_prompts(template, "", count=6, base_seed=20, variables={"age": "{18|60}"})
        for i, item in enumerate(batch):
            single = expand_prompts(template, "", count=1, base_seed=20 + i, variables={"age": "{18|60}"})
            assert single[0].positive == item.positive


class TestPerImageBinding:
    def test_condition_and_use_agree_on_the_same_roll(self):
        batch = expand_prompts(
            'x <<if $age >= 40>>old<<else>>young<<end>> ${age}', "",
            count=40, base_seed=100, variables={"age": "{18|30|45|60}"},
        )
        seen = set()
        for item in batch:
            _, label, age = item.positive.split(" ")
            assert (label == "old") == (int(age) >= 40)
            seen.add(label)
        assert seen == {"old", "young"}

    def test_every_use_of_a_read_variable_shares_one_roll(self):
        batch = expand_prompts(
            '<<if $m = "a">>A<<else>>B<<end>> ${m} ${m} ${m}', "", count=30, base_seed=0, variables={"m": "{a|b|c}"},
        )
        for item in batch:
            marker, *uses = item.positive.split(" ")
            assert len(set(uses)) == 1
            assert marker == ("A" if uses[0] == "a" else "B")

    def test_unread_variable_keeps_independent_rolls_per_use(self):
        batch = expand_prompts(
            '<<if $m = "a">>A<<end>> ${n} ${n}', "", count=40, base_seed=0, variables={"m": "a", "n": "{p|q|r|s}"},
        )
        assert any(len(set(item.positive.split(" ")[1:])) > 1 for item in batch)

    def test_roll_is_a_function_of_seed_and_name_only(self):
        template = '<<if $m = "a">>A<<else>>B<<end>> {p|q|r}'
        for seed in range(30):
            a = expand_prompts(template, "", count=1, base_seed=seed, variables={"m": "{a|b}", "z": "{1|2}"})
            b = expand_prompts(template, "", count=1, base_seed=seed, variables={"m": "{a|b}"})
            assert texts(a) == texts(b)

    def test_undefined_variable_takes_otherwise(self):
        got = expand_prompts('<<if $x = "y">>A<<else>>fallback<<end>>', "", count=1, base_seed=0)[0]
        assert got.positive == "fallback"

    def test_negative_channel_reads_the_same_roll(self):
        got = expand_prompts(
            '<<if $m = "a">>A<<else>>B<<end>>', '<<if $m = "a">>na<<else>>nb<<end>>',
            count=30, base_seed=0, variables={"m": "{a|b}"},
        )
        for item in got:
            assert (item.positive == "A") == (item.negative == "na")


class TestBodies:
    def test_branch_body_expands_groups_and_variables(self):
        got = expand_prompts('<<if $x = "y">>{a|a} ${x}<<end>>, end', "", count=1, base_seed=5, variables={"x": "y"})[0]
        assert got.positive == "a y, end"

    def test_literal_dollar_percent_hash_survive_inside_branches_and_around(self):
        got = expand_prompts('a $100 bill, <<if $x = "y">>100% #1<<end>>', "", count=1, base_seed=0, variables={"x": "y"})[0]
        assert got.positive == "a $100 bill, 100% #1"

    def test_malformed_group_inside_a_branch_uses_the_body_verbatim_and_warns_once(self):
        warnings = []
        got = expand_prompts('<<if $x = "y">>{a|b<<end>>', "", count=3, base_seed=0, variables={"x": "y"}, on_warning=warnings.append)
        assert texts(got) == ["{a|b"] * 3
        assert warnings == [MALFORMED_TEMPLATE_WARNING]

    def test_empty_result_cleanup_is_applied_after_sampling(self):
        got = expand_prompts('a, <<if $x = "n">>{p|q}<<end>>, b', "", count=1, base_seed=0, variables={"x": "y"})[0]
        assert got.positive == "a, b"

    def test_group_that_samples_to_empty_in_a_branch_counts_as_empty(self):
        got = expand_prompts('a, <<if $x = "y">>{|}<<end>>, b', "", count=1, base_seed=0, variables={"x": "y"})[0]
        assert got.positive == "a, b"


class TestWarnings:
    def test_malformed_tag_warns_once_for_the_whole_generation(self):
        warnings = []
        got = expand_prompts('a <<if $x = "y">>B', "n <<end>>", count=4, base_seed=0, variables={"x": "y"}, on_warning=warnings.append)
        assert warnings == [CONDITION_WARNING]
        assert {e.positive for e in got} == {"a B"}
        assert {e.negative for e in got} == {"n"}

    def test_bad_head_warns(self):
        warnings = []
        got = expand_prompts('<<if $x == "y">>A<<else>>B<<end>>', "", count=1, base_seed=0, variables={"x": "y"}, on_warning=warnings.append)
        assert got[0].positive == "B"
        assert warnings == [CONDITION_WARNING]

    def test_clean_tags_do_not_warn(self):
        warnings = []
        expand_prompts('<<if $x = "y">>A<<end>>', "", count=2, base_seed=0, variables={"x": "y"}, on_warning=warnings.append)
        assert warnings == []

    def test_never_raises_on_hostile_input(self):
        rng = random.Random(5)
        pieces = ["<<if $a = 1>>", "<<else>>", "<<end>>", "${a}", "{x|y}", "$", "%", "{", "}", "\n", "<<", ">>", "#"]
        for _ in range(300):
            template = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 15)))
            expand_prompts(template, template, count=2, base_seed=1, variables={"a": "{1|2}"})


class TestHooks:
    def test_pre_hook_sees_tags_and_may_add_them(self):
        seen = []

        def handler(context):
            if context.data["phase"] == "pre":
                seen.append(context.data["positive"])
                context.data["positive"] = context.data["positive"] + ' <<if $x = "y">>added<<end>>'
            return context

        got = expand_prompts(
            '<<if $x = "y">>A<<end>>', "", count=1, base_seed=0, variables={"x": "y"},
            plugin_registry=FakePluginRegistry(handler),
        )
        assert seen == ['<<if $x = "y">>A<<end>>']
        assert got[0].positive == "A added"

    def test_post_hook_sees_the_final_text(self):
        seen = []

        def handler(context):
            if context.data["phase"] == "post":
                seen.append(context.data["positive"])
            return context

        expand_prompts(
            'a <<if $x = "y">>B<<end>>', "", count=1, base_seed=0, variables={"x": "y"},
            plugin_registry=FakePluginRegistry(handler),
        )
        assert seen == ["a B"]
