import copy
from collections import Counter

import pytest

from src.features.llm.harness.profile import LAYERED_FIELDS, CapabilityProfile, ProfileSource
from src.features.llm.harness.profile_seed import CORE_ROOT, SeedError, SeedKey, SeedOrigin, SeedTable
from tests.fixtures.llm_profiles import shipped_mapping, shipped_seed, tiny_seed

CATALOG_OWNED = {"context_window", "max_output_tokens", "prices"}


def profile_for(adapter: str, model: str, server: str = None) -> CapabilityProfile:
    layer = shipped_seed().layer(SeedKey(adapter, model, server))
    return CapabilityProfile(adapter=adapter, model=model).layered([layer])


def shipped() -> dict:
    return shipped_mapping()


def test_the_shipped_seed_loads_and_every_row_cites_known_sources():
    table = shipped_seed()

    assert len(table.rows) > 10
    assert table.catalog
    for row in table.rows:
        assert row.sources
        assert set(row.sources) <= set(table.sources)


def test_every_source_except_the_catalog_one_is_cited_by_some_row():
    table = shipped_seed()
    cited = {name for row in table.rows for name in row.sources}

    assert set(table.sources) - cited == {"models-dev"}


def test_every_source_is_a_public_url():
    assert all(url.startswith("https://") for url in shipped_seed().sources.values())


def test_the_base_row_states_every_profile_field():
    assert set(shipped_seed().base) == set(LAYERED_FIELDS)


def test_curated_rows_leave_catalog_data_to_the_catalog():
    for row in shipped_seed().rows:
        assert not (set(row.fields) & CATALOG_OWNED), row.id


def test_an_unknown_model_resolves_to_the_conservative_base():
    profile = profile_for("mystery-adapter", "mystery-model")

    assert profile.tools is True
    assert profile.prompt_tools is False
    assert profile.tool_choice == frozenset({"auto"})
    assert profile.reasoning.value == "none"
    assert profile.replay.value == "send_all"
    assert profile.cache.value == "none"
    assert profile.context_window is None
    assert set(profile.sources) == set(LAYERED_FIELDS)
    assert set(profile.sources.values()) == {ProfileSource.SEED}


def test_the_catalog_overlays_the_base_and_curated_rows_overlay_the_catalog():
    profile = profile_for("anthropic-messages", "claude-opus-5-5")

    assert profile.context_window == 1_000_000
    assert profile.prices is not None and profile.prices.input == 4
    assert profile.sampling_rejected == frozenset({"temperature", "top_p", "top_k"})


@pytest.mark.parametrize(
    "model, forced_allowed",
    [
        ("claude-opus-5-5", False),
        ("claude-sonnet-5-5", False),
        ("claude-fable-5-1", False),
        ("claude-mythos-5-1", False),
        ("claude-opus-5", True),
        ("claude-sonnet-5", True),
        ("claude-haiku-5-5", True),
        ("claude-fable-5", True),
    ],
)
def test_anthropic_forced_tool_choice_is_banned_only_on_the_models_that_reject_it(model, forced_allowed):
    profile = profile_for("anthropic-messages", model)

    assert ("required" in profile.tool_choice) is forced_allowed
    assert ("named" in profile.tool_choice_with_reasoning) is forced_allowed
    assert "auto" in profile.tool_choice and "none" in profile.tool_choice


@pytest.mark.parametrize(
    "model, minimum",
    [
        ("claude-opus-5-5", 512),
        ("claude-haiku-5-5", 512),
        ("claude-fable-5-1", 512),
        ("claude-sonnet-5", 1024),
        ("claude-mythos-preview", 2048),
    ],
)
def test_anthropic_cache_minimums_follow_the_model(model, minimum):
    profile = profile_for("anthropic-messages", model)

    assert profile.cache.value == "explicit"
    assert profile.cache_min_tokens == minimum


@pytest.mark.parametrize(
    "model, can_disable",
    [
        ("claude-opus-5-5", False),
        ("claude-sonnet-5-5", False),
        ("claude-fable-5-1", False),
        ("claude-opus-5", True),
        ("claude-sonnet-5", True),
        ("claude-haiku-5-5", True),
    ],
)
def test_anthropic_thinking_switch_exists_only_where_the_api_accepts_disabled(model, can_disable):
    profile = profile_for("anthropic-messages", model)

    assert profile.reasoning.value == "effort"
    assert (profile.reasoning_toggle == "thinking.type") is can_disable
    assert profile.interleaved_reasoning is True


def test_older_claude_models_get_the_adapter_baseline_without_adaptive_thinking():
    profile = profile_for("anthropic-messages", "claude-sonnet-4-5")

    assert profile.reasoning.value == "budget"
    assert profile.sampling_rejected == frozenset()
    assert profile.parallel_tools is True
    assert profile.tool_result_media is True
    assert profile.cache.value == "explicit"


def test_openai_reasoning_models_use_responses_defaults():
    profile = profile_for("openai-responses", "gpt-5.6-sol")

    assert profile.reasoning.value == "effort"
    assert profile.reasoning_toggle == "reasoning.effort"
    assert "max" in profile.reasoning_levels
    assert profile.cache.value == "automatic"
    assert profile.cache_min_tokens == 1024
    assert profile.strict_tools is True
    assert profile.sampling_rejected == frozenset({"temperature"})


def test_openai_models_before_the_new_cache_minimum_leave_it_unset():
    assert profile_for("openai-responses", "gpt-5.4").cache_min_tokens is None


def test_gemini_three_and_four_reason_with_levels_that_cannot_be_switched_off():
    profile = profile_for("gemini-interactions", "gemini-3.5-flash")

    assert profile.reasoning.value == "effort"
    assert profile.reasoning_toggle is None
    assert profile.reasoning_levels[0] == "minimal"
    assert profile.cache_min_tokens == 4096
    assert profile.parallel_tools is True


@pytest.mark.parametrize("model", ["deepseek-flash", "deepseek-v4-pro", "deepseek-v4-flash"])
def test_deepseek_needs_reasoning_replayed_and_bans_forced_tool_choice_while_thinking(model):
    profile = profile_for("openai-chat", model)

    assert profile.replay.value == "send_all"
    assert profile.reasoning_field == "reasoning_content"
    assert profile.reasoning_toggle == "thinking.type"
    assert profile.reasoning_levels == ("low", "high", "max")
    assert profile.tool_choice_with_reasoning == frozenset({"auto", "none"})
    assert profile.tool_choice == frozenset({"auto", "none", "required", "named"})
    assert "temperature" in profile.sampling_ignored_with_reasoning


def test_deepseek_pro_also_ignores_top_p_while_thinking_but_flash_does_not():
    assert profile_for("openai-chat", "deepseek-v4-pro").sampling_ignored_with_reasoning == frozenset(
        {"temperature", "top_p"}
    )
    assert profile_for("openai-chat", "deepseek-flash").sampling_ignored_with_reasoning == frozenset({"temperature"})


def test_deepseek_rows_do_not_leak_to_other_openai_compatible_models():
    profile = profile_for("openai-chat", "llama-3.3-70b")

    assert profile.reasoning.value == "none"
    assert profile.tool_choice == frozenset({"auto"})
    assert profile.reasoning_field is None


def test_vllm_servers_read_and_write_the_reasoning_field_named_reasoning():
    assert profile_for("openai-chat", "qwen3-32b", "vllm").reasoning_field == "reasoning"
    assert profile_for("openai-chat", "qwen3-32b", "VLLM ").reasoning_field == "reasoning"
    assert profile_for("openai-chat", "qwen3-32b", "llama.cpp").reasoning_field is None
    assert profile_for("openai-chat", "qwen3-32b").reasoning_field is None


@pytest.mark.parametrize(
    "model, replay",
    [
        ("qwen3:8b", "drop_earlier"),
        ("qwen3.5:9b", "drop_earlier"),
        ("qwen3.6:27b", "drop_earlier"),
        ("hf.co/Qwen/qwen3:latest", "drop_earlier"),
        ("qwen3.8:14b", "send_all"),
        ("gpt-oss:20b", "drop_earlier"),
        ("glm-5.3", "send_all"),
        ("kimi-k2.6", "send_all"),
        ("minimax-m2.7", "send_all"),
        ("llama3.2:3b", "send_all"),
    ],
)
def test_ollama_family_replay_defaults_follow_what_the_template_does(model, replay):
    assert profile_for("ollama-chat", model).replay.value == replay


def test_ollama_reasoning_controls_per_family():
    qwen = profile_for("ollama-chat", "qwen3.6:27b")
    oss = profile_for("ollama-chat", "gpt-oss:20b")
    minimax = profile_for("ollama-chat", "minimax-m2.7")

    assert (qwen.reasoning.value, qwen.reasoning_toggle) == ("toggle", "think")
    assert (oss.reasoning.value, oss.reasoning_levels, oss.reasoning_toggle) == ("effort", ("low", "medium", "high"), None)
    assert (minimax.reasoning.value, minimax.interleaved_reasoning) == ("always_on", True)
    assert qwen.reasoning_field == "thinking"


def test_ollama_baseline_has_no_tool_choice_no_ids_and_a_local_cache():
    profile = profile_for("ollama-chat", "llama3.2:3b")

    assert profile.tool_choice == frozenset({"auto"})
    assert profile.tool_id_rule.value == "mint"
    assert profile.cache.value == "local_prefix"
    assert profile.parallel_tools is True


def test_ollama_gemma_three_sees_images_but_does_not_reason_and_gemma_four_reasons():
    assert profile_for("ollama-chat", "gemma3:12b").vision is True
    assert profile_for("ollama-chat", "gemma3:12b").reasoning.value == "none"
    assert profile_for("ollama-chat", "gemma2:9b").vision is False
    assert profile_for("ollama-chat", "gemma4:31b").reasoning.value == "toggle"


def test_ollama_context_window_is_never_taken_from_the_catalog():
    assert profile_for("ollama-chat", "gpt-oss:20b").context_window is None


def test_the_native_engine_renders_tools_into_the_prompt():
    profile = profile_for("native", "some-checkpoint")

    assert profile.prompt_tools is True
    assert profile.tool_id_rule.value == "mint"
    assert profile.reasoning_toggle == "chat_template_kwargs.enable_thinking"
    assert profile.replay.value == "drop_earlier"


@pytest.mark.parametrize(
    "adapter, model, expected",
    [
        ("ollama-chat", "qwen3:8b", {"temperature": 0.6, "top_p": 0.95, "top_k": 20.0, "min_p": 0.0}),
        ("ollama-chat", "qwen3.5:9b", {"temperature": 1.0, "top_p": 0.95, "top_k": 20.0, "min_p": 0.0}),
        ("ollama-chat", "qwen3.6:27b", {"temperature": 1.0, "top_p": 0.95, "top_k": 20.0, "min_p": 0.0}),
        ("ollama-chat", "qwen3.8:14b", {"temperature": 1.0, "top_p": 0.95, "top_k": 20.0, "min_p": 0.0}),
        ("ollama-chat", "glm-5.3", {"temperature": 1.0, "top_p": 0.95}),
        ("ollama-chat", "kimi-k2.6", {"temperature": 1.0}),
        ("ollama-chat", "minimax-m2.7", {"temperature": 1.0, "top_p": 0.95, "top_k": 40.0}),
        ("ollama-chat", "gemma3:12b", {"temperature": 1.0, "top_p": 0.95, "top_k": 64.0}),
        ("ollama-chat", "gemma4:31b", {"temperature": 1.0, "top_p": 0.95, "top_k": 64.0}),
        ("gemini-interactions", "gemini-3.5-flash", {"temperature": 1.0}),
        ("ollama-chat", "llama3.2:3b", {}),
        ("gemini-interactions", "gemini-2.5-pro", {}),
        ("anthropic-messages", "claude-opus-5-5", {}),
    ],
)
def test_recommended_sampling_follows_the_model_card(adapter, model, expected):
    assert profile_for(adapter, model).recommended_sampling == expected


def test_every_recommended_sampling_value_passes_the_override_validation():
    from src.features.llm.harness.profile_schema import validate_overrides

    for row in shipped_seed().rows:
        if "recommended_sampling" in row.fields:
            validate_overrides({"recommended_sampling": row.fields["recommended_sampling"]})


def test_only_anthropic_profiles_are_reasoning_prefix_bound():
    assert profile_for("anthropic-messages", "claude-opus-5-5").reasoning_prefix_bound is True
    assert profile_for("anthropic-messages", "claude-sonnet-4-5").reasoning_prefix_bound is True
    for adapter, model in (
        ("openai-responses", "gpt-5.6"),
        ("gemini-interactions", "gemini-3.5-flash"),
        ("openai-chat", "deepseek-v4-pro"),
        ("ollama-chat", "qwen3:8b"),
        ("native", "x"),
    ):
        assert profile_for(adapter, model).reasoning_prefix_bound is False


def test_only_deepseek_requires_reasoning_replayed_when_tools_are_present():
    assert profile_for("openai-chat", "deepseek-v4-pro").replay_required_with_tools is True
    assert profile_for("openai-chat", "deepseek-flash").replay_required_with_tools is True
    for adapter, model in (
        ("anthropic-messages", "claude-opus-5-5"),
        ("openai-responses", "gpt-5.6"),
        ("openai-chat", "llama-3.3-70b"),
        ("ollama-chat", "qwen3:8b"),
    ):
        assert profile_for(adapter, model).replay_required_with_tools is False


def test_matching_is_case_insensitive():
    assert profile_for("anthropic-messages", "Claude-Opus-5-5").tool_choice == frozenset({"auto", "none"})


def test_patterns_written_with_capitals_still_match_lowercased_model_names():
    table = tiny_seed({"id": "capitals", "models": ["^Claude-OPUS"], "fields": {"vision": True}})

    assert table.layer(SeedKey("any", "claude-opus-5")).fields["vision"] is True
    assert table.layer(SeedKey("any", "CLAUDE-Opus-5")).fields["vision"] is True
    assert table.layer(SeedKey("any", "gpt-5")).fields["vision"] is False


def test_applied_rows_are_listed_in_application_order():
    applied = shipped_seed().applied(SeedKey("anthropic-messages", "claude-sonnet-5"))

    assert [(item.id, item.origin) for item in applied] == [
        ("base", SeedOrigin(CORE_ROOT, "profile_base.yml")),
        ("catalog:claude-sonnet-5", SeedOrigin("marketplace", "anthropic.yml")),
        ("anthropic-messages", SeedOrigin("marketplace", "anthropic.yml")),
        ("anthropic-5x", SeedOrigin("marketplace", "anthropic.yml")),
        ("anthropic-sonnet-5-cache-minimum", SeedOrigin("marketplace", "anthropic.yml")),
    ]


def test_a_layer_is_the_same_object_for_the_same_key():
    key = SeedKey("ollama-chat", "qwen3:8b")

    assert shipped_seed().layer(key) is shipped_seed().layer(key)


class TestRejectedSeedTables:
    def table(self, mutate):
        data = shipped()
        mutate(data)
        return data

    def test_an_unknown_top_level_key(self):
        with pytest.raises(SeedError, match="unknown keys"):
            SeedTable.from_mapping(self.table(lambda data: data.update(extra=1)))

    def test_a_wrong_version(self):
        with pytest.raises(SeedError, match="version"):
            SeedTable.from_mapping(self.table(lambda data: data.update(version=2)))

    def test_a_base_that_misses_a_field(self):
        with pytest.raises(SeedError, match="missing"):
            SeedTable.from_mapping(self.table(lambda data: data["base"]["fields"].pop("replay")))

    def test_a_row_with_an_unknown_profile_field(self):
        with pytest.raises(SeedError, match="unknown profile field"):
            SeedTable.from_mapping(self.table(lambda data: data["rows"][0]["fields"].update(bogus=1)))

    def test_a_row_with_a_wrong_typed_value(self):
        with pytest.raises(SeedError, match="row 'anthropic-messages'"):
            SeedTable.from_mapping(self.table(lambda data: data["rows"][0]["fields"].update(parallel_tools="yes")))

    def test_a_row_with_an_invalid_enum_value(self):
        with pytest.raises(SeedError):
            SeedTable.from_mapping(self.table(lambda data: data["rows"][0]["fields"].update(cache="sometimes")))

    def test_a_row_that_cites_no_source(self):
        with pytest.raises(SeedError, match="cite at least one source"):
            SeedTable.from_mapping(self.table(lambda data: data["rows"][0].update(source=[])))

    def test_a_row_that_cites_an_unknown_source(self):
        with pytest.raises(SeedError, match="unknown sources"):
            SeedTable.from_mapping(self.table(lambda data: data["rows"][0].update(source=["nowhere"])))

    def test_a_bad_model_pattern(self):
        with pytest.raises(SeedError, match="bad model pattern"):
            SeedTable.from_mapping(self.table(lambda data: data["rows"][0].update(models=["(unclosed"])))

    def test_duplicate_row_ids(self):
        with pytest.raises(SeedError, match="duplicate row ids"):
            SeedTable.from_mapping(self.table(lambda data: data["rows"][1].update(id=data["rows"][0]["id"])))

    def test_duplicate_catalog_entries(self):
        with pytest.raises(SeedError, match="duplicate catalog"):
            SeedTable.from_mapping(self.table(lambda data: data["catalog"].append(copy.deepcopy(data["catalog"][0]))))

    def test_an_empty_fields_mapping(self):
        with pytest.raises(SeedError, match="non-empty"):
            SeedTable.from_mapping(self.table(lambda data: data["rows"][0].update(fields={})))

    def test_a_row_that_sets_the_server_it_is_selected_by(self):
        with pytest.raises(SeedError, match="cannot set it"):
            SeedTable.from_mapping(self.table(lambda data: data["rows"][0]["fields"].update(server="vllm")))

    def test_a_catalog_entry_that_sets_the_server(self):
        with pytest.raises(SeedError, match="cannot set it"):
            SeedTable.from_mapping(self.table(lambda data: data["catalog"][0]["fields"].update(server="vllm")))

    def test_an_unknown_row_key(self):
        with pytest.raises(SeedError, match="unknown keys"):
            SeedTable.from_mapping(self.table(lambda data: data["rows"][0].update(note="x")))


def test_catalog_entries_are_unique_per_adapter_and_model():
    counts = Counter((entry.adapter, entry.model) for entry in shipped_seed().catalog)

    assert max(counts.values()) == 1
