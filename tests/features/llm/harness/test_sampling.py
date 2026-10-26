import pytest

from src.features.llm.harness.adapter import Settings, ToolChoice
from src.features.llm.harness.sampling import reasoning_from_options, resolve_settings
from tests.fixtures.llm_harness import scripted_profile
from tests.fixtures.llm_profiles import make_config


def resolved(config_fields=None, profile_fields=None, request=None):
    return resolve_settings(make_config(**(config_fields or {})), scripted_profile(**(profile_fields or {})), request or Settings())


def test_the_configuration_supplies_temperature_and_max_tokens_when_the_request_has_none():
    settings = resolved({"temperature": 0.4, "max_tokens": 900})

    assert (settings.temperature, settings.max_tokens) == (0.4, 900)


def test_provider_options_beat_the_configuration_and_the_request_beats_both():
    config = {"temperature": 0.4, "max_tokens": 900, "provider_options": {"temperature": 0.9, "top_p": 0.8, "max_tokens": 700}}

    from_options = resolved(config)
    from_request = resolved(config, request=Settings(temperature=0.1, top_p=0.5, max_tokens=300))

    assert (from_options.temperature, from_options.top_p, from_options.max_tokens) == (0.9, 0.8, 700)
    assert (from_request.temperature, from_request.top_p, from_request.max_tokens) == (0.1, 0.5, 300)


def test_a_request_value_of_zero_is_a_value():
    assert resolved({"temperature": 0.7}, request=Settings(temperature=0.0)).temperature == 0.0


def test_the_profile_recommendation_fills_only_what_nothing_else_sets():
    profile = {"recommended_sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20.0, "min_p": 0.0}}

    bare = resolved({"temperature": 0.7}, profile)
    tuned = resolved({"temperature": 0.7, "provider_options": {"top_p": 0.5, "min_p": 0.1}}, profile, Settings(top_k=5))

    assert (bare.temperature, bare.top_p, bare.top_k, bare.min_p) == (0.7, 0.95, 20, 0.0)
    assert isinstance(bare.top_k, int)
    assert (tuned.top_p, tuned.top_k, tuned.min_p) == (0.5, 5, 0.1)


def test_nothing_is_invented_without_a_recommendation():
    settings = resolved()

    assert (settings.top_p, settings.top_k, settings.min_p) == (None, None, None)


def test_parameters_the_api_rejects_are_removed_after_every_layer_has_spoken():
    profile = {"sampling_rejected": ["temperature", "top_p", "top_k"], "recommended_sampling": {"top_k": 20.0}}

    settings = resolved({"temperature": 0.7, "provider_options": {"top_p": 0.8}}, profile, Settings(top_k=3, min_p=0.1))

    assert (settings.temperature, settings.top_p, settings.top_k, settings.min_p) == (None, None, None, 0.1)


def test_other_settings_pass_through_untouched():
    request = Settings(tool_choice=ToolChoice("required"), response_schema={"type": "object"}, cache_key="k", keep_model_loaded=True)

    settings = resolved(request=request)

    assert settings.tool_choice == ToolChoice("required")
    assert (settings.response_schema, settings.cache_key, settings.keep_model_loaded) == ({"type": "object"}, "k", True)


@pytest.mark.parametrize(
    "options, expected",
    [
        ({"reasoning_effort": "high"}, "high"),
        ({"reasoning_effort": "xhigh"}, "xhigh"),
        ({"reasoning_effort": "none"}, False),
        ({"reasoning_effort": None}, None),
        ({"reasoning_effort": "  "}, None),
        ({"think": True}, True),
        ({"think": False}, False),
        ({"think": "medium"}, "medium"),
        ({"think": None}, None),
        ({"thinking": True}, True),
        ({"thinking": False}, False),
        ({"thinking": "yes"}, None),
        ({"reasoning_effort": "low", "think": True}, "low"),
        ({}, None),
        (None, None),
    ],
)
def test_provider_options_map_to_one_reasoning_setting(options, expected):
    assert reasoning_from_options(options) == expected


def test_the_request_reasoning_beats_provider_options_even_when_it_switches_thinking_off():
    config = {"provider_options": {"reasoning_effort": "high"}}

    assert resolved(config).reasoning == "high"
    assert resolved(config, request=Settings(reasoning=False)).reasoning is False
    assert resolved(config, request=Settings(reasoning="low")).reasoning == "low"
    assert resolved({"provider_options": {"reasoning_effort": "none"}}).reasoning is False
