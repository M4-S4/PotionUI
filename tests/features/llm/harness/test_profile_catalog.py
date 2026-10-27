import json
from pathlib import Path

import pytest
import yaml

from scripts import llm_profiles_refresh
from src.features.llm.harness.profile_catalog import build_catalog, refresh_seed, render_seed
from src.features.llm.harness.profile_seed import SeedKey, SeedOrigin, SeedTable, parse_seed_file
from tests.fixtures.llm_profiles import MODELS_DEV_PROVIDERS, combined_seed_text, shipped_roots

FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "llm_models_dev" / "api.json"


@pytest.fixture(scope="module")
def models_dev() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def catalog(models_dev) -> dict:
    return {(entry["adapter"], entry["model"]): entry["fields"] for entry in build_catalog(models_dev, MODELS_DEV_PROVIDERS)}


@pytest.fixture
def seed_copy(tmp_path) -> Path:
    path = tmp_path / "profiles.yml"
    path.write_text(combined_seed_text(), encoding="utf-8", newline="\n")
    return path


def test_only_chat_models_of_the_mapped_providers_are_kept(catalog):
    assert sorted(catalog) == [
        ("anthropic-messages", "claude-haiku-5-5"),
        ("anthropic-messages", "claude-opus-5-5"),
        ("anthropic-messages", "claude-sonnet-4-5"),
        ("gemini-interactions", "gemini-3.5-flash"),
        ("openai-chat", "deepseek-v4-flash-vision-exp"),
        ("openai-chat", "deepseek-v4-pro"),
        ("openai-responses", "gpt-5.4-pro"),
        ("openai-responses", "gpt-5.6"),
    ]


def test_entries_follow_provider_then_model_order(models_dev):
    order = [(entry["adapter"], entry["model"]) for entry in build_catalog(models_dev, MODELS_DEV_PROVIDERS)]
    providers = list(MODELS_DEV_PROVIDERS.values())

    assert order == sorted(order, key=lambda item: (providers.index(item[0]), item[1]))


def test_limits_prices_and_vision_are_mapped(catalog):
    assert catalog[("anthropic-messages", "claude-opus-5-5")] == {
        "context_window": 1_000_000,
        "max_output_tokens": 128_000,
        "tools": True,
        "reasoning": "effort",
        "reasoning_levels": ["low", "medium", "high", "xhigh", "max"],
        "vision": True,
        "sampling_rejected": ["temperature"],
        "prices": {"input": 4, "output": 20, "cached_input": 0.2, "cache_write": 5},
    }


def test_reasoning_options_map_to_the_control_kind(catalog):
    assert catalog[("anthropic-messages", "claude-haiku-5-5")]["reasoning"] == "effort"
    assert catalog[("anthropic-messages", "claude-sonnet-4-5")]["reasoning"] == "budget"
    assert "reasoning_levels" not in catalog[("anthropic-messages", "claude-sonnet-4-5")]
    assert catalog[("openai-responses", "gpt-5.6")]["reasoning_levels"] == [
        "none", "low", "medium", "high", "xhigh", "max",
    ]
    assert catalog[("openai-responses", "gpt-5.4-pro")]["reasoning_levels"] == ["medium", "high", "xhigh"]
    assert catalog[("gemini-interactions", "gemini-3.5-flash")]["reasoning_levels"][0] == "minimal"


def test_interleaved_reasoning_and_its_field_come_from_the_provider(catalog):
    fields = catalog[("openai-chat", "deepseek-v4-pro")]

    assert fields["interleaved_reasoning"] is True
    assert fields["reasoning_field"] == "reasoning_content"
    assert "interleaved_reasoning" not in catalog[("anthropic-messages", "claude-opus-5-5")]


def test_image_input_decides_vision(catalog):
    assert catalog[("openai-chat", "deepseek-v4-pro")]["vision"] is False
    assert catalog[("openai-chat", "deepseek-v4-flash-vision-exp")]["vision"] is True


def test_a_model_that_rejects_temperature_says_so_and_others_do_not(catalog):
    assert catalog[("openai-responses", "gpt-5.6")]["sampling_rejected"] == ["temperature"]
    assert "sampling_rejected" not in catalog[("anthropic-messages", "claude-sonnet-4-5")]


def test_prices_without_both_input_and_output_are_left_out_and_bad_numbers_are_ignored():
    entries = build_catalog(
        {
            "openai": {
                "models": {
                    "free": {"modalities": {"input": ["text"], "output": ["text"]}, "cost": {"input": 1}},
                    "negative": {
                        "modalities": {"input": ["text"], "output": ["text"]},
                        "cost": {"input": 1, "output": 2, "cache_read": -5, "reasoning": True},
                    },
                    "odd": "not a model",
                    "bare": {"modalities": {"input": ["text"], "output": ["text"]}, "reasoning": True},
                }
            }
        },
        MODELS_DEV_PROVIDERS,
    )
    by_model = {entry["model"]: entry["fields"] for entry in entries}

    assert "prices" not in by_model["free"]
    assert by_model["negative"]["prices"] == {"input": 1, "output": 2}
    assert by_model["bare"] == {"vision": False}
    assert "odd" not in by_model


def test_missing_providers_are_skipped():
    assert build_catalog({}, MODELS_DEV_PROVIDERS) == []
    assert build_catalog({"anthropic": {}}, MODELS_DEV_PROVIDERS) == []


SHIPPED_FILES = [path for root in shipped_roots() for path in sorted(root.path.glob("*.yml"))]
CATALOG_FILES = [path for path in SHIPPED_FILES if "models_dev" in path.read_text(encoding="utf-8")]


@pytest.mark.parametrize("path", CATALOG_FILES, ids=lambda path: path.name)
def test_every_shipped_catalog_is_in_the_format_the_script_writes(path):
    text = path.read_text(encoding="utf-8")

    assert render_seed(text, yaml.safe_load(text)["catalog"]) == text


def test_the_files_without_a_catalog_declare_no_providers():
    others = [path for path in SHIPPED_FILES if path not in CATALOG_FILES]

    assert others
    for path in others:
        assert not yaml.safe_load(path.read_text(encoding="utf-8")).get("catalog")


def test_the_providers_each_shipped_file_declares_cover_the_catalog_adapters():
    declared = {}
    for path in CATALOG_FILES:
        declared.update(yaml.safe_load(path.read_text(encoding="utf-8"))["models_dev"])

    assert declared == MODELS_DEV_PROVIDERS


def test_refreshing_rewrites_only_the_catalog_and_keeps_the_curated_part_byte_for_byte(models_dev):
    text = combined_seed_text()
    head = text[: text.index("\ncatalog:") + 1]

    refreshed = refresh_seed(text, models_dev)

    assert refreshed.startswith(head)
    assert refreshed != text
    assert len(yaml.safe_load(refreshed)["catalog"]) == 8


def test_refreshing_twice_changes_nothing_more(models_dev):
    once = refresh_seed(combined_seed_text(), models_dev)

    assert refresh_seed(once, models_dev) == once


def test_a_refreshed_seed_resolves_catalog_data_into_profiles(models_dev):
    refreshed = parse_seed_file(yaml.safe_load(refresh_seed(combined_seed_text(), models_dev)), SeedOrigin("test", "refreshed.yml"), strict=True)
    table = SeedTable.assemble(SeedTable.load_base(), [refreshed])

    layer = table.layer(SeedKey("openai-responses", "gpt-5.6"))

    assert layer.fields["context_window"] == 1_050_000
    assert layer.fields["prices"]["output"] == 20


def test_an_empty_catalog_renders_as_an_empty_list():
    assert render_seed("version: 1\ncatalog: []\n", []) == "version: 1\ncatalog: []\n"
    assert render_seed("version: 1\ncatalog:\n  - x\n", []) == "version: 1\ncatalog: []\n"


@pytest.mark.parametrize("text", ["version: 1\n", "catalog: []\ncatalog: []\n"])
def test_a_seed_without_exactly_one_catalog_key_is_refused(text):
    with pytest.raises(ValueError, match="exactly one"):
        render_seed(text, [])


def test_a_catalog_the_seed_loader_would_reject_is_never_written(models_dev):
    broken = {"anthropic": {"models": {"m": {"modalities": {"input": ["text"], "output": ["text"]}, "reasoning_options": []}}}}
    entries = build_catalog(broken, MODELS_DEV_PROVIDERS)
    entries[0]["fields"]["not_a_field"] = 1
    bad = render_seed(combined_seed_text(), entries)

    with pytest.raises(Exception, match="not_a_field"):
        parse_seed_file(yaml.safe_load(bad), SeedOrigin("test", "bad.yml"), strict=True)


class TestScript:
    def test_it_rewrites_the_seed_from_a_local_file_without_touching_the_network(self, seed_copy, monkeypatch, capsys):
        monkeypatch.setattr(llm_profiles_refresh.httpx, "get", lambda *args, **kwargs: pytest.fail("network used"))

        code = llm_profiles_refresh.main(["--input", str(FIXTURE), "--seed", str(seed_copy)])

        assert code == 0
        assert "catalog rewritten" in capsys.readouterr().out
        assert len(yaml.safe_load(seed_copy.read_text(encoding="utf-8"))["catalog"]) == 8

    def test_a_second_run_reports_the_catalog_is_up_to_date(self, seed_copy, capsys):
        llm_profiles_refresh.main(["--input", str(FIXTURE), "--seed", str(seed_copy)])
        before = seed_copy.read_bytes()

        code = llm_profiles_refresh.main(["--input", str(FIXTURE), "--seed", str(seed_copy)])

        assert code == 0
        assert "up to date" in capsys.readouterr().out
        assert seed_copy.read_bytes() == before

    def test_check_fails_on_a_stale_catalog_and_writes_nothing(self, seed_copy, capsys):
        before = seed_copy.read_bytes()

        code = llm_profiles_refresh.main(["--input", str(FIXTURE), "--seed", str(seed_copy), "--check"])

        assert code == 1
        assert "stale" in capsys.readouterr().out
        assert seed_copy.read_bytes() == before

    def test_check_passes_on_a_fresh_catalog(self, seed_copy):
        llm_profiles_refresh.main(["--input", str(FIXTURE), "--seed", str(seed_copy)])

        assert llm_profiles_refresh.main(["--input", str(FIXTURE), "--seed", str(seed_copy), "--check"]) == 0

    def test_it_fetches_the_url_when_no_file_is_given(self, seed_copy, monkeypatch):
        calls = []

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return json.loads(FIXTURE.read_text(encoding="utf-8"))

        def fake_get(url, **kwargs):
            calls.append((url, kwargs["follow_redirects"]))
            return Response()

        monkeypatch.setattr(llm_profiles_refresh.httpx, "get", fake_get)

        assert llm_profiles_refresh.main(["--seed", str(seed_copy), "--url", "https://models.example/api.json"]) == 0
        assert calls == [("https://models.example/api.json", True)]
