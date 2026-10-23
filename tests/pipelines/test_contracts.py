import pytest

from src.pipelines.contracts import PIPE_FAMILY_TITLES, BasePipe, PipeConfigSpec, PipeOutput, resolve_display_title


class TestResolveDisplayTitle:
    def test_override_wins_regardless_of_name(self):
        assert resolve_display_title("generator", "Custom step name") == "Custom step name"

    @pytest.mark.parametrize(
        "pipe_name,expected",
        [
            ("checkpoint_loader", "Loading model"),
            ("model_loader", "Loading model"),
            ("generator", "Generating"),
            ("upscaler", "Upscaling"),
        ],
    )
    def test_known_family_fallback(self, pipe_name, expected):
        assert resolve_display_title(pipe_name) == expected

    def test_variant_suffix_is_stripped_before_family_lookup(self):
        assert resolve_display_title("interpolator/rife") == "Interpolating frames"

    def test_unknown_pipe_name_is_cleaned_up(self):
        assert resolve_display_title("my_custom_plugin_step") == "My custom plugin step"

    def test_missing_name_falls_back_to_processing(self):
        assert resolve_display_title(None) == "Processing"

    def test_every_family_title_is_non_empty(self):
        assert all(title for title in PIPE_FAMILY_TITLES.values())


class _SpecOnlyPipe(BasePipe):
    name = "spec_only"
    description = "pipe whose defaults come from configuration() alone"

    def process(self, pipe_input, generation_outputs):
        return PipeOutput(output={})

    @classmethod
    def inputs(cls):
        return []

    @classmethod
    def outputs(cls):
        return []

    @classmethod
    def configuration(cls):
        return [
            PipeConfigSpec("steps", int, 20),
            PipeConfigSpec("loras", list, [{"weight": 1.0}]),
            PipeConfigSpec("model", dict, None, required=True),
        ]


class TestDerivedDefaultConfig:
    def test_defaults_are_the_spec_defaults(self):
        assert _SpecOnlyPipe.get_default_config() == {
            "steps": 20,
            "loras": [{"weight": 1.0}],
            "model": None,
        }

    def test_mutating_a_returned_default_does_not_leak(self):
        first = _SpecOnlyPipe.get_default_config()
        first["loras"][0]["weight"] = 0.1
        first["loras"].append({"weight": 2.0})

        assert _SpecOnlyPipe.get_default_config()["loras"] == [{"weight": 1.0}]
