import asyncio
import logging
from datetime import datetime, timezone

import pytest

from src.features.llm.harness import codec
from src.features.llm.harness.profile import ProfileSource, ReplayPolicy
from src.features.llm.harness.profiles import (
    DISCOVERY_HIT_TTL_SECONDS,
    DISCOVERY_MISS_TTL_SECONDS,
    LayeredProfileResolver,
)
from src.features.llm.model_listing import ModelListingError
from src.platform.database.rows import dt_iso
from tests.fixtures.llm_profiles import FakeClock, ProbeAdapter, adapter_for, make_config, tiny_seed

MOMENT = datetime(2026, 10, 27, 9, 12, tzinfo=timezone.utc)


def build(adapter=None, seed=None, clock=None):
    adapter = adapter or ProbeAdapter()
    clock = clock or FakeClock()
    resolver = LayeredProfileResolver(adapter_for(adapter), seed or tiny_seed(), now=lambda: MOMENT, clock=clock)
    return resolver, adapter, clock


def test_seed_only_profile_carries_the_adapter_and_model():
    resolver, adapter, _ = build()

    profile = resolver.resolve(make_config(model="Some-Model"))

    assert (profile.adapter, profile.model) == ("probe", "Some-Model")
    assert set(profile.sources.values()) == {ProfileSource.SEED}


def test_layers_apply_seed_then_discovered_then_override():
    seed = tiny_seed({"id": "all", "fields": {"context_window": 8192, "replay": "drop_earlier", "vision": True}})
    resolver, adapter, _ = build(ProbeAdapter(fields={"context_window": 65536, "replay": "preserve"}), seed)
    config = make_config(profile_overrides={"replay": "never"})

    asyncio.run(resolver.refresh(config))
    profile = resolver.resolve(config)

    assert profile.context_window == 65536
    assert profile.replay is ReplayPolicy.NEVER
    assert profile.vision is True
    assert profile.sources["context_window"] is ProfileSource.DISCOVERED
    assert profile.sources["replay"] is ProfileSource.OVERRIDE
    assert profile.sources["vision"] is ProfileSource.SEED


def test_an_override_beats_a_discovered_value_and_removing_it_restores_the_discovered_one():
    resolver, _, _ = build(ProbeAdapter(fields={"context_window": 65536}))
    asyncio.run(resolver.refresh(make_config()))

    assert resolver.resolve(make_config(profile_overrides={"context_window": 4096})).context_window == 4096
    assert resolver.resolve(make_config(profile_overrides=None)).context_window == 65536


def test_discovery_is_not_used_before_the_first_refresh():
    resolver, adapter, _ = build(ProbeAdapter(fields={"context_window": 65536}))

    assert resolver.resolve(make_config()).context_window is None
    assert adapter.calls == []


def test_ensure_discovers_once_and_then_serves_the_cache():
    resolver, adapter, _ = build(ProbeAdapter(fields={"context_window": 65536}))
    config = make_config()

    first = asyncio.run(resolver.ensure(config))
    second = asyncio.run(resolver.ensure(config))

    assert first.context_window == second.context_window == 65536
    assert adapter.calls == ["cfg-1"]


def test_refresh_always_rediscovers():
    resolver, adapter, _ = build(ProbeAdapter(fields={"context_window": 65536}))
    config = make_config()

    asyncio.run(resolver.refresh(config))
    adapter.fields = {"context_window": 131072}
    refreshed = asyncio.run(resolver.refresh(config))

    assert refreshed.context_window == 131072
    assert adapter.calls == ["cfg-1", "cfg-1"]


def test_a_hit_expires_after_its_ttl_and_ensure_discovers_again():
    resolver, adapter, clock = build(ProbeAdapter(fields={"context_window": 65536}))
    config = make_config()
    asyncio.run(resolver.ensure(config))

    clock.advance(DISCOVERY_HIT_TTL_SECONDS - 1)
    asyncio.run(resolver.ensure(config))
    assert len(adapter.calls) == 1

    clock.advance(2)
    asyncio.run(resolver.ensure(config))
    assert len(adapter.calls) == 2


def test_a_failed_discovery_is_remembered_for_the_short_ttl_and_never_breaks_the_profile():
    adapter = ProbeAdapter(error=ModelListingError("Couldn't reach it: connection refused."))
    resolver, _, clock = build(adapter)
    config = make_config()

    profile = asyncio.run(resolver.ensure(config))
    asyncio.run(resolver.ensure(config))

    assert profile.context_window is None
    assert len(adapter.calls) == 1
    assert resolver.explain(config).discovery_error == "Couldn't reach it: connection refused."

    clock.advance(DISCOVERY_MISS_TTL_SECONDS + 1)
    asyncio.run(resolver.ensure(config))
    assert len(adapter.calls) == 2


def test_an_unexpected_discovery_error_is_reported_without_leaking_details():
    resolver, _, _ = build(ProbeAdapter(error=RuntimeError("secret internals")))
    config = make_config()

    asyncio.run(resolver.refresh(config))

    assert resolver.explain(config).discovery_error == "Discovery failed unexpectedly."


def test_a_discovery_that_hangs_times_out(monkeypatch):
    class Hanging(ProbeAdapter):
        async def discover(self, config):
            await asyncio.Event().wait()

    monkeypatch.setattr("src.features.llm.harness.profiles.DISCOVERY_TIMEOUT_SECONDS", 0.01)
    resolver, _, _ = build(Hanging())
    config = make_config()

    asyncio.run(resolver.refresh(config))

    assert resolver.explain(config).discovery_error == "Discovery timed out."


def test_a_failed_refresh_keeps_the_previous_discovery_and_reports_the_error():
    adapter = ProbeAdapter(fields={"context_window": 65536})
    resolver, _, _ = build(adapter)
    config = make_config()
    asyncio.run(resolver.refresh(config))

    adapter.error = ModelListingError("rejected the credentials (HTTP 401).")
    profile = asyncio.run(resolver.refresh(config))

    assert profile.context_window == 65536
    report = resolver.explain(config)
    assert report.discovery_error == "rejected the credentials (HTTP 401)."
    assert report.discovered_at is not None


def test_a_successful_refresh_clears_an_earlier_failure():
    adapter = ProbeAdapter(error=ModelListingError("down"))
    resolver, _, _ = build(adapter)
    config = make_config()
    asyncio.run(resolver.refresh(config))

    adapter.error = None
    adapter.fields = {"context_window": 2048}
    asyncio.run(resolver.refresh(config))

    report = resolver.explain(config)
    assert report.discovery_error is None
    assert resolver.resolve(config).context_window == 2048


def test_a_discovered_layer_with_an_unknown_field_is_rejected_whole():
    resolver, _, _ = build(ProbeAdapter(fields={"context_window": 65536, "temperature_default": 0.7}))
    config = make_config()

    profile = asyncio.run(resolver.refresh(config))

    assert profile.context_window is None
    error = resolver.explain(config).discovery_error
    assert "probe adapter returned an invalid profile" in error and "temperature_default" in error


def test_a_discovered_layer_with_a_wrong_typed_value_is_rejected_whole():
    resolver, _, _ = build(ProbeAdapter(fields={"context_window": "big"}))
    config = make_config()

    asyncio.run(resolver.refresh(config))

    assert "invalid profile" in resolver.explain(config).discovery_error


def test_discovery_is_cached_per_adapter_server_and_model():
    resolver, adapter, _ = build(ProbeAdapter(fields={"context_window": 65536}))
    asyncio.run(resolver.ensure(make_config(id="a", model="m1", base_url="http://host:1/")))
    asyncio.run(resolver.ensure(make_config(id="b", model="m1", base_url="http://host:1")))
    asyncio.run(resolver.ensure(make_config(id="c", model="m2", base_url="http://host:1")))
    asyncio.run(resolver.ensure(make_config(id="d", model="m1", base_url="http://other:1")))

    assert adapter.calls == ["a", "c", "d"]


def vllm_seed():
    return tiny_seed({"id": "vllm", "servers": ["vllm"], "fields": {"reasoning_field": "reasoning"}})


def test_a_discovered_server_selects_seed_rows_and_is_part_of_the_profile():
    resolver, _, _ = build(ProbeAdapter(fields={"server": "vllm"}), vllm_seed())
    config = make_config()

    assert resolver.resolve(config).reasoning_field is None
    profile = asyncio.run(resolver.refresh(config))

    assert profile.server == "vllm"
    assert profile.reasoning_field == "reasoning"
    assert profile.sources["server"] is ProfileSource.DISCOVERED
    assert profile.sources["reasoning_field"] is ProfileSource.SEED


def test_an_admin_override_of_the_server_selects_seed_rows_without_discovery():
    resolver, _, _ = build(seed=vllm_seed())

    profile = resolver.resolve(make_config(profile_overrides={"server": "vLLM"}))

    assert profile.server == "vLLM"
    assert profile.reasoning_field == "reasoning"
    assert profile.sources["server"] is ProfileSource.OVERRIDE


def test_the_override_of_the_server_beats_the_discovered_one():
    resolver, _, _ = build(ProbeAdapter(fields={"server": "llama.cpp"}), vllm_seed())
    config = make_config(profile_overrides={"server": "vllm"})

    profile = asyncio.run(resolver.refresh(config))

    assert profile.server == "vllm" and profile.reasoning_field == "reasoning"


def test_an_unknown_server_matches_no_server_rows():
    resolver, _, _ = build(ProbeAdapter(fields={"server": "llama.cpp"}), vllm_seed())

    profile = asyncio.run(resolver.refresh(make_config()))

    assert profile.server == "llama.cpp" and profile.reasoning_field is None


def test_explain_reports_the_server_row_in_the_seed_layer():
    resolver, _, _ = build(ProbeAdapter(fields={"server": "vllm"}), vllm_seed())
    config = make_config()
    asyncio.run(resolver.refresh(config))

    fields = resolver.explain(config).fields

    assert fields["reasoning_field"].seed == "reasoning" and fields["reasoning_field"].source is ProfileSource.SEED
    assert fields["server"].discovered == "vllm" and fields["server"].seed is None


def test_a_stored_override_that_no_longer_validates_is_skipped_and_logged_once(caplog):
    resolver, _, _ = build()
    config = make_config(id="logged", profile_overrides={"replay": "sometimes", "vision": True, "bogus": 1})

    with caplog.at_level(logging.WARNING, logger="src.features.llm.harness.profiles"):
        profile = resolver.resolve(config)
        resolver.resolve(config)

    assert profile.vision is True
    assert profile.replay is ReplayPolicy.SEND_ALL
    messages = [record.getMessage() for record in caplog.records]
    assert sum("replay" in message for message in messages) == 1
    assert any("bogus" in message for message in messages)


def test_explain_reports_value_source_and_each_layer_per_field():
    seed = tiny_seed({"id": "all", "fields": {"context_window": 32768, "replay": "send_all"}})
    resolver, _, _ = build(ProbeAdapter(fields={"context_window": 65536}), seed)
    config = make_config(model="qwen3.8:14b", profile_overrides={"replay": "preserve"})
    asyncio.run(resolver.refresh(config))

    report = resolver.explain(config)
    encoded = codec.to_json(report)

    assert encoded["config_id"] == "cfg-1" and encoded["adapter"] == "probe" and encoded["model"] == "qwen3.8:14b"
    assert encoded["discovered_at"] == dt_iso(MOMENT)
    assert encoded["discovery_error"] is None
    assert encoded["fields"]["context_window"] == {
        "value": 65536, "source": "discovered", "seed": 32768, "discovered": 65536, "override": None,
    }
    assert encoded["fields"]["replay"] == {
        "value": "preserve", "source": "override", "seed": "send_all", "discovered": None, "override": "preserve",
    }
    assert encoded["fields"]["vision"]["source"] == "seed"
    assert encoded["verified"] is None


def test_explain_lists_every_editable_and_read_only_field_in_schema_order():
    from src.features.llm.harness.profile_schema import PROFILE_FIELDS

    resolver, _, _ = build()

    report = resolver.explain(make_config())

    assert list(report.fields) == [spec.name for spec in PROFILE_FIELDS]


def test_explain_serialises_sets_sorted_and_enums_by_value():
    resolver, _, _ = build(seed=tiny_seed(tool_choice=["required", "auto"]))

    encoded = codec.to_json(resolver.explain(make_config()))

    assert encoded["fields"]["tool_choice"]["value"] == ["auto", "required"]
    assert encoded["fields"]["reasoning"]["value"] == "none"


def test_an_unsupported_type_surfaces_the_lookup_error():
    def unsupported(config):
        raise ValueError(f"Unsupported LLM type: {config.type}")

    resolver = LayeredProfileResolver(unsupported, tiny_seed())

    with pytest.raises(ValueError, match="Unsupported LLM type: nope"):
        resolver.resolve(make_config(type="nope"))
