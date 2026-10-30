import json

import pytest

from src.features.llm.harness.adapters.native import NativeAdapter
from src.features.llm.harness.profile import ProfileSource
from src.features.llm.model_listing import DiscoveredWindow
from src.features.llm.native.runtime import read_checkpoint_context_window
from tests.features.llm.harness.history import make_config


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("files", [
    {"config.json": {"max_position_embeddings": 40960}},
    {"config.json": {"text_config": {"max_position_embeddings": 40960}}},
    {"config.json": {"max_position_embeddings": 40960}, "tokenizer_config.json": {"model_max_length": 1024}},
    {"config.json": {"hidden_size": 8}, "tokenizer_config.json": {"model_max_length": 40960}},
])
def test_the_trained_window_is_read_preferring_config_json(tmp_path, files):
    for name, payload in files.items():
        write_json(tmp_path / name, payload)

    assert read_checkpoint_context_window(str(tmp_path)) == DiscoveredWindow(tokens=40960, kind="trained")


def test_the_unbounded_tokenizer_sentinel_is_ignored(tmp_path):
    write_json(tmp_path / "tokenizer_config.json", {"model_max_length": int(1e30)})

    assert read_checkpoint_context_window(str(tmp_path)) is None


def test_missing_or_broken_files_yield_nothing(tmp_path):
    assert read_checkpoint_context_window(str(tmp_path)) is None
    (tmp_path / "config.json").write_text("{not json", encoding="utf-8")
    write_json(tmp_path / "tokenizer_config.json", ["x"])

    assert read_checkpoint_context_window(str(tmp_path)) is None


@pytest.mark.asyncio
async def test_discovery_reports_the_checkpoints_window_from_local_files(tmp_path):
    write_json(tmp_path / "config.json", {"max_position_embeddings": 32768})
    adapter = NativeAdapter()
    adapter.runtime._resolve_model = lambda name: (str(tmp_path), False)

    layer = await adapter.discover(make_config("native", model="qwen"))

    assert layer.source is ProfileSource.DISCOVERED
    assert layer.fields["context_window"] == 32768
    assert layer.fields["prompt_tools"] is True


@pytest.mark.asyncio
async def test_an_adopted_text_encoder_or_an_unresolvable_name_reports_no_window(tmp_path):
    write_json(tmp_path / "config.json", {"max_position_embeddings": 32768})
    adapter = NativeAdapter()
    adapter.runtime._resolve_model = lambda name: (str(tmp_path), True)
    assert "context_window" not in (await adapter.discover(make_config("native", model="te"))).fields

    def unresolvable(name):
        raise ValueError("no such checkpoint")

    adapter.runtime._resolve_model = unresolvable
    assert "context_window" not in (await adapter.discover(make_config("native", model="missing"))).fields
