"""Tests for the shared prompt-variable plain-language renderer."""

from src.platform.resources.prompt_variables import (
    SUMMARY_READ_HINT,
    render_prompt_variable_lines,
    render_prompt_variable_summary,
)


def _choice(name, options, **extra):
    return {"name": name, "type": "choice", "options": options, **extra}


class TestRenderPromptVariableLines:
    def test_absent_or_malformed_returns_empty(self):
        assert render_prompt_variable_lines(None) == []
        assert render_prompt_variable_lines("nope") == []
        assert render_prompt_variable_lines({}) == []
        assert render_prompt_variable_lines([]) == []
        assert render_prompt_variable_lines([123, "x", None]) == []

    def test_text_variable_reports_length_and_preview(self):
        lines = render_prompt_variable_lines([{"name": "subject", "type": "text", "value": "a red fox"}])
        assert lines == ['subject: text, 9 chars ("a red fox")']

    def test_text_variable_empty(self):
        assert render_prompt_variable_lines([{"name": "subject", "type": "text"}]) == ["subject: text, empty"]

    def test_long_text_is_clipped_but_its_length_is_exact(self):
        line = render_prompt_variable_lines([{"name": "v", "type": "text", "value": "x" * 500}])[0]
        assert "500 chars" in line
        assert "x" * 100 not in line
        assert len(line) < 100

    def test_choice_shuffle_with_last_roll(self):
        lines = render_prompt_variable_lines([_choice("mood", ["noir", "sunlit"], mode="shuffle", lastRoll="sunlit")])
        assert lines == ["mood: choice, shuffles each generation, 2 options (noir, sunlit); last roll: sunlit"]

    def test_choice_default_mode_is_shuffle(self):
        lines = render_prompt_variable_lines([_choice("mood", ["a", "b"])])
        assert lines == ["mood: choice, shuffles each generation, 2 options (a, b)"]

    def test_single_option_is_singular(self):
        assert render_prompt_variable_lines([_choice("mood", ["a"])]) == [
            "mood: choice, shuffles each generation, 1 option (a)"
        ]

    def test_choice_pin_names_the_pinned_option(self):
        lines = render_prompt_variable_lines([_choice("mood", ["noir", "sunlit"], mode="pin", pinnedIndex=1)])
        assert lines == ["mood: choice, pinned to sunlit, 2 options (noir, sunlit)"]

    def test_choice_pin_out_of_range_falls_back(self):
        lines = render_prompt_variable_lines([_choice("mood", ["noir"], mode="pin", pinnedIndex=9)])
        assert lines == ["mood: choice, pinned, 1 option (noir)"]

    def test_pinned_value_is_clipped(self):
        line = render_prompt_variable_lines(
            [_choice("mood", ["a", "z" * 200], mode="pin", pinnedIndex=1)]
        )[0]
        assert "pinned to " + "z" * 40 + "…" in line
        assert "z" * 41 not in line

    def test_choice_per_image(self):
        lines = render_prompt_variable_lines([_choice("mood", ["a", "b"], mode="per-image")])
        assert lines == ["mood: choice, re-rolls independently per image, 2 options (a, b)"]

    def test_choice_with_no_valid_options_skipped(self):
        lines = render_prompt_variable_lines([
            _choice("empty", ["", "  "]),
            {"name": "ok", "type": "text", "value": "x"},
        ])
        assert lines == ['ok: text, 1 chars ("x")']

    def test_nameless_entries_skipped(self):
        lines = render_prompt_variable_lines([
            {"type": "text", "value": "x"},
            {"name": "  ", "type": "text", "value": "y"},
            {"name": "keep", "type": "text", "value": "z"},
        ])
        assert lines == ['keep: text, 1 chars ("z")']

    def test_variable_count_is_uncapped(self):
        variables = [{"name": f"v{i}", "type": "text", "value": str(i)} for i in range(40)]
        assert len(render_prompt_variable_lines(variables)) == 40

    def test_many_options_show_the_count_and_only_the_first_few(self):
        options = [f"opt{i}" for i in range(20)]
        line = render_prompt_variable_lines([_choice("big", options, mode="per-image")])[0]
        assert "20 options (first 3: opt0, opt1, opt2)" in line
        assert "opt3" not in line

    def test_long_options_never_appear_in_full(self):
        options = [f"option {i:02d} " + "long outfit description " * 8 for i in range(40)]
        line = render_prompt_variable_lines([_choice("outfit_kind", options)])[0]
        assert "40 options" in line
        assert options[0] not in line
        assert "option 03" not in line
        assert len(line) < 200

    def test_summary_size_does_not_grow_with_option_count(self):
        small = render_prompt_variable_lines([_choice("v", [f"o{i}" for i in range(5)])])[0]
        large = render_prompt_variable_lines([_choice("v", [f"o{i}" for i in range(500)])])[0]
        assert len(large) - len(small) <= 3

    def test_conditioned_options_are_counted_and_dependencies_named(self):
        lines = render_prompt_variable_lines([
            _choice("music", ["hip hop", "classical"]),
            _choice("dance", [{"text": "breaking", "when": {"var": "music", "values": ["hip hop"]}}, "salsa"]),
        ])
        assert lines == [
            "music: choice, shuffles each generation, 2 options (hip hop, classical)",
            "dance: choice, resolves after $music, shuffles each generation, 2 options, 1 conditional (breaking, salsa)",
        ]


class TestRenderPromptVariableSummary:
    def test_ends_with_the_read_tool_pointer(self):
        lines = render_prompt_variable_summary([_choice("mood", ["a", "b"])])
        assert lines[:-1] == render_prompt_variable_lines([_choice("mood", ["a", "b"])])
        assert lines[-1] == SUMMARY_READ_HINT
        assert "get_prompt_variables" in lines[-1]

    def test_no_variables_means_no_pointer(self):
        assert render_prompt_variable_summary([]) == []
        assert render_prompt_variable_summary(None) == []

    def test_only_one_pointer_line_for_many_variables(self):
        lines = render_prompt_variable_summary([_choice(f"v{i}", ["a", "b"]) for i in range(10)])
        assert len(lines) == 11
        assert sum(1 for line in lines if "get_prompt_variables" in line) == 1


class TestTruncatedSnapshotSummary:
    def test_marker_adds_the_omitted_line_before_the_pointer(self):
        lines = render_prompt_variable_summary([_choice("a", ["x", "y"]), {"truncated": True, "omitted": 4}])
        assert len(lines) == 3
        assert lines[1].startswith("4 more variables not sent with this snapshot")
        assert lines[2] == SUMMARY_READ_HINT

    def test_singular_noun_for_one(self):
        lines = render_prompt_variable_summary([_choice("a", ["x"]), {"truncated": True, "omitted": 1}])
        assert lines[1].startswith("1 more variable not sent")

    def test_option_total_overrides_the_sent_count(self):
        line = render_prompt_variable_lines([_choice("big", ["a", "b", "c", "d"], optionTotal=250)])[0]
        assert "250 options (first 3: a, b, c)" in line

    def test_marker_alone_still_reports_the_omitted_variables(self):
        lines = render_prompt_variable_summary([{"truncated": True, "omitted": 3}])
        assert lines[0].startswith("3 more variables not sent") and lines[-1] == SUMMARY_READ_HINT


class TestTwentyVariableFixture:
    OPTIONS = [
        f"option {i:02d}: a deliberately long outfit description with layered fabrics, trims and accessories"
        for i in range(40)
    ]

    def fixture(self):
        variables = [_choice("outfit_kind", self.OPTIONS)]
        variables += [_choice(f"mood_{i}", ["noir", "sunlit", "dusk", "misty", "neon"]) for i in range(6)]
        variables += [
            {"name": f"num_{i}", "type": "number", "mode": "shuffle", "min": 0, "max": 100, "step": 5}
            for i in range(5)
        ]
        variables += [
            {"name": f"txt_{i}", "type": "text", "value": "a reusable fragment of prompt text " * 4}
            for i in range(8)
        ]
        return variables

    def test_summary_keeps_the_outfit_kind_count_without_its_options(self):
        lines = render_prompt_variable_summary(self.fixture())
        assert len(lines) == 21
        outfit = next(line for line in lines if line.startswith("outfit_kind:"))
        assert "40 options (first 3:" in outfit
        assert "option 02" in outfit
        assert "option 03" not in outfit
        assert max(len(line) for line in lines) < 200

    def test_whole_summary_stays_small(self):
        assert sum(len(line) for line in render_prompt_variable_summary(self.fixture())) < 2200


class TestSharedHelpers:
    def test_valid_options_normalizes_mixed_string_and_object_entries(self):
        from src.platform.resources.prompt_variables import valid_options

        options = valid_options([
            "salsa",
            {"text": "breaking", "when": {"var": "music", "values": ["hip hop"]}},
            {"text": "  "},
            "",
        ])
        assert options == [
            {"text": "salsa", "when": None},
            {"text": "breaking", "when": {"var": "music", "values": ["hip hop"]}},
        ]

    def test_variable_dependencies_dedupes_and_preserves_order(self):
        from src.platform.resources.prompt_variables import variable_dependencies

        var = {
            "type": "choice",
            "options": [
                {"text": "a", "when": {"var": "music", "values": ["x"]}},
                {"text": "b", "when": {"var": "era", "values": ["y"]}},
                {"text": "c", "when": {"var": "music", "values": ["z"]}},
            ],
        }
        assert variable_dependencies(var) == ["music", "era"]

    def test_is_downstream_detects_transitive_chain(self):
        from src.platform.resources.prompt_variables import is_downstream

        variables = {
            "music": {"type": "choice", "options": ["hip hop"]},
            "dance": {
                "type": "choice",
                "options": [{"text": "a", "when": {"var": "music", "values": ["hip hop"]}}],
            },
            "era": {
                "type": "choice",
                "options": [{"text": "b", "when": {"var": "dance", "values": ["a"]}}],
            },
        }
        assert is_downstream("era", "music", variables) is True
        assert is_downstream("music", "era", variables) is False

    def test_format_option_appends_when_clause(self):
        from src.platform.resources.prompt_variables import format_option

        assert format_option({"text": "breaking", "when": None}) == "breaking"
        assert format_option(
            {"text": "breaking", "when": {"var": "music", "values": ["hip hop", "latin"]}}
        ) == "breaking (when $music = hip hop, latin)"


class TestNameError:
    def test_valid_names_pass(self):
        from src.platform.resources.prompt_variables import name_error

        assert name_error("mood") is None
        assert name_error("_mood2") is None

    def test_invalid_names_are_rejected(self):
        from src.platform.resources.prompt_variables import name_error

        assert "not a valid variable name" in name_error("1mood")
        assert "not a valid variable name" in name_error("mo od")
        assert "not a valid variable name" in name_error("m" * 61)


class TestValidateCondition:
    def test_valid_condition_normalizes(self):
        from src.platform.resources.prompt_variables import validate_condition

        variables = {"scene": {"type": "choice", "options": ["day", "night"]}}
        normalized, error = validate_condition(
            "mood", {"var": "scene", "values": ["day"]}, variables
        )
        assert error is None
        assert normalized == {"var": "scene", "values": ["day"]}

    def test_self_reference_is_rejected(self):
        from src.platform.resources.prompt_variables import validate_condition

        _, error = validate_condition("mood", {"var": "mood", "values": ["x"]}, {"mood": {}})
        assert "cannot reference" in error

    def test_reference_to_non_choice_variable_is_rejected(self):
        from src.platform.resources.prompt_variables import validate_condition

        variables = {"scene": {"type": "text", "value": "day"}}
        _, error = validate_condition("mood", {"var": "scene", "values": ["day"]}, variables)
        assert "is not a choice variable" in error

    def test_values_not_among_referenced_options_are_rejected(self):
        from src.platform.resources.prompt_variables import validate_condition

        variables = {"scene": {"type": "choice", "options": ["day", "night"]}}
        _, error = validate_condition(
            "mood", {"var": "scene", "values": ["dusk"]}, variables
        )
        assert "are not options of $scene" in error


class TestValidateVariablesMap:
    def test_none_is_valid(self):
        from src.platform.resources.prompt_variables import validate_variables_map

        assert validate_variables_map(None) == []

    def test_non_dict_is_rejected(self):
        from src.platform.resources.prompt_variables import validate_variables_map

        assert validate_variables_map([1, 2]) != []

    def test_valid_text_and_choice_map_has_no_errors(self):
        from src.platform.resources.prompt_variables import validate_variables_map

        variables = {
            "mood": {"type": "text", "value": "noir"},
            "scene": {
                "type": "choice", "mode": "pin", "pinnedIndex": 0,
                "options": ["day", "night"],
            },
        }
        assert validate_variables_map(variables) == []

    def test_bad_name_is_reported(self):
        from src.platform.resources.prompt_variables import validate_variables_map

        errors = validate_variables_map({"1bad": {"type": "text", "value": "x"}})
        assert any("not a valid variable name" in e for e in errors)

    def test_choice_with_no_options_is_rejected(self):
        from src.platform.resources.prompt_variables import validate_variables_map

        errors = validate_variables_map({"scene": {"type": "choice", "options": []}})
        assert any("needs at least one non-empty option" in e for e in errors)

    def test_invalid_mode_is_rejected(self):
        from src.platform.resources.prompt_variables import validate_variables_map

        errors = validate_variables_map({
            "scene": {"type": "choice", "mode": "random", "options": ["day"]}
        })
        assert any("invalid mode" in e for e in errors)

    def test_pinned_index_out_of_range_is_rejected(self):
        from src.platform.resources.prompt_variables import validate_variables_map

        errors = validate_variables_map({
            "scene": {"type": "choice", "mode": "pin", "pinnedIndex": 5, "options": ["day"]}
        })
        assert any("pinnedIndex must be a valid index" in e for e in errors)

    def test_unknown_definition_type_is_rejected(self):
        from src.platform.resources.prompt_variables import validate_variables_map

        errors = validate_variables_map({"mood": {"type": "date", "value": 1}})
        assert any("Use text, choice or number" in e for e in errors)

    def test_condition_referencing_a_variable_outside_the_map_is_rejected(self):
        from src.platform.resources.prompt_variables import validate_variables_map

        errors = validate_variables_map({
            "scene": {
                "type": "choice", "mode": "shuffle", "options": [
                    {"text": "x", "when": {"var": "missing", "values": ["y"]}}
                ],
            },
        })
        assert any("is not a choice variable" in e for e in errors)

    def test_choice_options_and_condition_values_are_uncapped(self):
        from src.platform.resources.prompt_variables import validate_variables_map

        scenes = [f"scene {i}" for i in range(40)]
        variables = {
            "scene": {"type": "choice", "mode": "shuffle", "options": scenes},
            "outfit": {
                "type": "choice", "mode": "shuffle", "options": [
                    {"text": f"outfit {i}", "when": {"var": "scene", "values": scenes[:20]}}
                    for i in range(40)
                ],
            },
        }
        assert validate_variables_map(variables) == []

    def test_many_variables_validate(self):
        from src.platform.resources.prompt_variables import validate_variables_map

        variables = {f"v{i}": {"type": "text", "value": "x"} for i in range(30)}
        assert validate_variables_map(variables) == []


class TestBuildVariablesForSubmit:
    def test_absent_or_malformed_returns_empty_map(self):
        from src.platform.resources.prompt_variables import build_variables_for_submit

        assert build_variables_for_submit(None) == {}
        assert build_variables_for_submit("nope") == {}
        assert build_variables_for_submit([123, "x", None]) == {}

    def test_text_variable_sends_its_value(self):
        from src.platform.resources.prompt_variables import build_variables_for_submit

        wire_map = build_variables_for_submit([{"name": "subject", "type": "text", "value": "a red fox"}])
        assert wire_map == {"subject": "a red fox"}

    def test_blank_text_variable_is_omitted(self):
        from src.platform.resources.prompt_variables import build_variables_for_submit

        wire_map = build_variables_for_submit([{"name": "subject", "type": "text", "value": ""}])
        assert wire_map == {}

    def test_pin_variable_sends_pinned_option(self):
        from src.platform.resources.prompt_variables import build_variables_for_submit

        wire_map = build_variables_for_submit([{
            "name": "mood", "type": "choice", "options": ["noir", "sunlit"],
            "mode": "pin", "pinnedIndex": 1,
        }])
        assert wire_map == {"mood": "sunlit"}

    def test_per_image_variable_sends_the_choice_group_template(self):
        from src.platform.resources.prompt_variables import build_variables_for_submit

        wire_map = build_variables_for_submit([{
            "name": "mood", "type": "choice", "options": ["noir", "sunlit"], "mode": "per-image",
        }])
        assert wire_map == {"mood": "{noir|sunlit}"}

    def test_shuffle_picks_one_option_deterministically(self):
        from src.platform.resources.prompt_variables import build_variables_for_submit

        wire_map = build_variables_for_submit(
            [{"name": "mood", "type": "choice", "options": ["noir", "sunlit"], "mode": "shuffle"}],
            random_fn=lambda: 0.0,
        )
        assert wire_map == {"mood": "noir"}

        wire_map = build_variables_for_submit(
            [{"name": "mood", "type": "choice", "options": ["noir", "sunlit"], "mode": "shuffle"}],
            random_fn=lambda: 0.99,
        )
        assert wire_map == {"mood": "sunlit"}

    def test_shuffle_resolves_music_before_dance_and_filters_to_eligible_options(self):
        from src.platform.resources.prompt_variables import build_variables_for_submit

        variables = [
            {"name": "dance", "type": "choice", "mode": "shuffle", "options": [
                {"text": "breaking", "when": {"var": "music", "values": ["hip hop"]}},
                {"text": "waltz", "when": {"var": "music", "values": ["classical"]}},
                "freestyle",
            ]},
            {"name": "music", "type": "choice", "mode": "shuffle", "options": ["hip hop", "classical", "latin"]},
        ]
        wire_map = build_variables_for_submit(variables, random_fn=lambda: 0.0)
        assert wire_map["music"] == "hip hop"
        assert wire_map["dance"] in ("breaking", "freestyle")

    def test_shuffle_omits_variable_with_no_eligible_option(self):
        from src.platform.resources.prompt_variables import build_variables_for_submit

        variables = [
            {"name": "music", "type": "choice", "mode": "shuffle", "options": ["latin"]},
            {"name": "dance", "type": "choice", "mode": "shuffle", "options": [
                {"text": "waltz", "when": {"var": "music", "values": ["classical"]}},
            ]},
        ]
        wire_map = build_variables_for_submit(variables, random_fn=lambda: 0.0)
        assert "dance" not in wire_map

    def test_pin_wins_regardless_of_its_own_condition(self):
        from src.platform.resources.prompt_variables import build_variables_for_submit

        variables = [
            {"name": "music", "type": "choice", "mode": "shuffle", "options": ["latin"]},
            {"name": "dance", "type": "choice", "mode": "pin", "pinnedIndex": 0, "options": [
                {"text": "breaking", "when": {"var": "music", "values": ["hip hop"]}},
                {"text": "waltz", "when": {"var": "music", "values": ["classical"]}},
            ]},
        ]
        wire_map = build_variables_for_submit(variables)
        assert wire_map["dance"] == "breaking"

    def test_per_image_ignores_conditions_and_wires_every_option(self):
        from src.platform.resources.prompt_variables import build_variables_for_submit

        variables = [
            {"name": "music", "type": "choice", "mode": "shuffle", "options": ["latin"]},
            {"name": "dance", "type": "choice", "mode": "per-image", "options": [
                {"text": "breaking", "when": {"var": "music", "values": ["hip hop"]}},
                {"text": "waltz", "when": {"var": "music", "values": ["classical"]}},
            ]},
        ]
        wire_map = build_variables_for_submit(variables)
        assert wire_map["dance"] == "{breaking|waltz}"
