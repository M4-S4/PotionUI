import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.features.llm.harness.profile_seed import (
    BASE_PATH,
    SeedKey,
    SeedOrigin,
    SeedRoot,
    SeedTable,
    default_seed,
    local_root,
    marketplace_root,
)
from src.features.llm.harness.profile_seed_catalog import SeedCatalog
from src.features.llm.harness.profiles import LayeredProfileResolver
from tests.fixtures.llm_profiles import FakeClock, ProbeAdapter, adapter_for, make_config, shipped_seed

GOLDEN = Path(__file__).resolve().parents[3] / "fixtures" / "llm_goldens" / "profile_resolution.json"
ADAPTERS = ["anthropic-messages", "openai-responses", "gemini-interactions", "openai-chat", "ollama-chat", "native"]
SERVERS = [None, "vllm"]
MODELS = [
    "claude-opus-5-5-x", "claude-sonnet-5-x", "claude-sonnet-5", "claude-sonnet-5-5-x", "claude-fable-5-1-x", "claude-fable-5",
    "claude-mythos-preview", "claude-mythos-5-1-x", "claude-haiku-5-x", "claude-3-x",
    "gpt-5-test", "gpt-5.6-test", "gpt-6-test", "o3-test", "gpt-4o-test", "gemini-3-test", "gemini-4-test", "gemini-2-test",
    "deepseek-v4-pro-test", "deepseek-flash-test", "deepseek-v4-flash-test", "deepseek-chat", "openrouter/deepseek-v4-pro-test",
    "qwen3-test", "qwen3.5-test", "qwen3.6-test", "qwen3.8-test", "qwen2-test", "gpt-oss-test", "glm-4.7-test", "kimi-k2-test",
    "minimax-m2-test", "gemma3-test", "gemma4-test", "gemma2-test", "llama-test", "some-model",
]


def doc(sources=None, rows=None, catalog=None):
    return {
        "version": 1,
        "sources": sources or {"doc": "https://example.test/doc"},
        "rows": [dict(row, source=row.get("source", ["doc"])) for row in rows or []],
        "catalog": catalog or [],
    }


def write(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else yaml.safe_dump(data, sort_keys=False), encoding="utf-8", newline="\n")
    return path


def row(row_id, **fields):
    return {"id": row_id, "adapters": ["probe"], "fields": fields}


def fields_of(table, model="m", adapter="probe", server=None):
    return table.layer(SeedKey(adapter, model, server)).fields


@pytest.fixture
def content(tmp_path) -> Path:
    return tmp_path / "llm-profiles"


def test_the_shipped_resolution_is_unchanged_for_every_curated_row():
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    table = shipped_seed()
    base = table.layer(SeedKey("other", "some-model")).fields
    seen = 0
    for adapter in ADAPTERS:
        for server in SERVERS:
            for model in MODELS:
                fields = table.layer(SeedKey(adapter, model, server)).fields
                changed = {name: repr(value) for name, value in fields.items() if repr(base[name]) != repr(value)}
                assert changed == golden.get(f"{adapter}|{server}|{model}", {}), (adapter, server, model)
                seen += 1
    assert seen == len(ADAPTERS) * len(SERVERS) * len(MODELS)


def test_a_server_scoped_row_still_wins_over_a_row_from_a_later_root(content):
    write(content / "marketplace" / "a.yml", doc(rows=[{"id": "by-server", "adapters": ["probe"], "servers": ["vllm"], "fields": {"max_images": 7}}]))
    write(content / "local" / "b.yml", doc(rows=[row("by-adapter", max_images=3)]))

    table = SeedTable.scan([marketplace_root(content), local_root(content)])

    assert fields_of(table, server="vllm")["max_images"] == 7
    assert fields_of(table)["max_images"] == 3


def test_core_carries_no_deepseek_row_and_the_plugin_root_provides_it():
    core = default_seed()
    plugin_ids = {item.id for item in shipped_seed().rows} - {item.id for item in core.rows}

    assert not [item for item in core.rows if "deepseek" in item.id]
    assert not [item for item in core.catalog if "deepseek" in item.model]
    assert plugin_ids == {"deepseek", "deepseek-v4-pro-ignores-top-p-while-reasoning"}
    assert core.layer(SeedKey("openai-chat", "deepseek-v4-pro")).fields["replay_required_with_tools"] is False
    assert shipped_seed().layer(SeedKey("openai-chat", "deepseek-v4-pro")).fields["replay_required_with_tools"] is True


def test_the_plugin_rows_report_the_root_and_file_they_came_from():
    applied = shipped_seed().applied(SeedKey("openai-chat", "deepseek-v4-pro"))
    origins = {item.id: item.origin for item in applied}

    assert origins["deepseek"] == SeedOrigin("plugin:deepseek-provider", "deepseek.yml")
    assert origins["catalog:deepseek-v4-pro"] == SeedOrigin("plugin:deepseek-provider", "deepseek.yml")


class TestPrecedence:
    def test_a_local_row_replaces_the_marketplace_row_with_the_same_id_without_merging(self, content):
        write(content / "marketplace" / "a.yml", doc(rows=[row("r", vision=True, max_images=4)]))
        write(content / "local" / "mine.yml", doc(rows=[row("r", max_images=9)]))

        table = SeedTable.scan([marketplace_root(content), local_root(content)])

        assert fields_of(table)["max_images"] == 9
        assert fields_of(table)["vision"] is False
        assert [item.origin for item in table.applied(SeedKey("probe", "m")) if item.id == "r"] == [SeedOrigin("local", "mine.yml")]

    def test_the_replaced_row_keeps_its_place_in_the_order(self, content):
        write(content / "marketplace" / "a.yml", doc(rows=[row("first", max_images=1), row("second", max_images=2)]))
        write(content / "local" / "mine.yml", doc(rows=[row("first", max_images=5)]))

        table = SeedTable.scan([marketplace_root(content), local_root(content)])

        assert [item.id for item in table.rows] == ["first", "second"]
        assert fields_of(table)["max_images"] == 2

    def test_a_plugin_root_sits_between_the_marketplace_and_local(self, content, tmp_path):
        write(content / "marketplace" / "a.yml", doc(rows=[row("r", max_images=1)]))
        write(tmp_path / "plug" / "p.yml", doc(rows=[row("r", max_images=2)]))
        write(content / "local" / "l.yml", doc(rows=[row("x", vision=True)]))

        table = SeedTable.scan([marketplace_root(content), SeedRoot("plugin:p", tmp_path / "plug"), local_root(content)])

        assert fields_of(table)["max_images"] == 2
        assert fields_of(table)["vision"] is True

    def test_a_local_catalog_entry_replaces_the_marketplace_one(self, content):
        entry = {"adapter": "probe", "model": "M", "fields": {"context_window": 1000}}
        write(content / "marketplace" / "a.yml", doc(catalog=[entry]))
        write(content / "local" / "l.yml", doc(catalog=[dict(entry, fields={"max_output_tokens": 50})]))

        table = SeedTable.scan([marketplace_root(content), local_root(content)])

        assert fields_of(table, "m")["max_output_tokens"] == 50
        assert fields_of(table, "m")["context_window"] is None


class TestBrokenFiles:
    def test_a_broken_local_file_is_skipped_and_named_while_everything_else_loads(self, content, caplog):
        write(content / "marketplace" / "a.yml", doc(rows=[row("good", max_images=4)]))
        bad = write(content / "local" / "bad.yml", "version: 1\nrows: [unclosed\n")
        write(content / "local" / "fine.yml", doc(rows=[row("mine", vision=True)]))

        with caplog.at_level(logging.WARNING):
            table = SeedTable.scan([marketplace_root(content), local_root(content)])

        assert fields_of(table)["max_images"] == 4
        assert fields_of(table)["vision"] is True
        assert any(bad.as_posix() in message for message in caplog.messages)
        assert any(bad.as_posix() in problem for problem in table.problems)

    def test_a_wrong_version_or_unknown_key_rejects_only_that_file(self, content):
        write(content / "local" / "old.yml", dict(doc(rows=[row("a", vision=True)]), version=9))
        write(content / "local" / "extra.yml", dict(doc(rows=[row("b", vision=True)]), surprise=1))
        write(content / "local" / "ok.yml", doc(rows=[row("c", max_images=2)]))

        table = SeedTable.scan([local_root(content)])

        assert [item.id for item in table.rows] == ["c"]
        assert len(table.problems) == 2

    def test_a_bad_row_is_rejected_by_itself_and_the_other_rows_of_the_file_load(self, content, caplog):
        path = write(
            content / "local" / "mixed.yml",
            doc(rows=[row("bad", not_a_field=1), row("good", max_images=3), {"id": "nosource", "source": [], "fields": {"vision": True}}]),
        )

        with caplog.at_level(logging.WARNING):
            table = SeedTable.scan([local_root(content)])

        assert [item.id for item in table.rows] == ["good"]
        assert sum(path.as_posix() in message for message in caplog.messages) == 2

    def test_a_row_citing_a_source_another_file_defines_is_rejected(self, content):
        write(content / "marketplace" / "a.yml", doc(sources={"elsewhere": "https://example.test/e"}, rows=[dict(row("x", vision=True), source=["elsewhere"])]))
        write(content / "local" / "b.yml", doc(rows=[dict(row("y", vision=True), source=["elsewhere"])]))

        table = SeedTable.scan([marketplace_root(content), local_root(content)])

        assert [item.id for item in table.rows] == ["x"]

    def test_a_non_mapping_file_and_a_missing_root_do_not_break_loading(self, content):
        write(content / "local" / "list.yml", "- just\n- a list\n")

        table = SeedTable.scan([marketplace_root(content), local_root(content)])

        assert table.rows == ()
        assert len(table.problems) == 1

    def test_a_broken_base_still_fails_loudly(self, tmp_path):
        broken = write(tmp_path / "base.yml", {"version": 1, "base": {"fields": {"tools": True}}})

        with pytest.raises(Exception, match="missing"):
            SeedTable.scan([], broken)

    def test_the_shipped_files_load_without_a_single_problem(self):
        assert shipped_seed().problems == ()
        assert default_seed().problems == ()
        assert BASE_PATH.exists()


class TestSeedCatalog:
    def registry(self, *manifests):
        return SimpleNamespace(get_enabled_plugins=lambda: list(manifests))

    def manifest(self, plugin_dir: Path):
        return SimpleNamespace(id="plug", plugin_dir=str(plugin_dir), llm_profiles=[{"path": "profiles"}])

    def test_plugin_rows_load_only_while_the_plugin_is_enabled(self, content, tmp_path):
        write(tmp_path / "plug" / "profiles" / "p.yml", doc(rows=[row("from-plugin", vision=True)]))
        enabled = [self.manifest(tmp_path / "plug")]
        clock = FakeClock()
        catalog = SeedCatalog(content, SimpleNamespace(get_enabled_plugins=lambda: enabled), clock=clock)

        assert fields_of(catalog.table())["vision"] is True
        enabled.clear()
        clock.now += 3
        assert fields_of(catalog.table())["vision"] is False

    def test_a_plugin_root_escaping_its_directory_is_ignored(self, content, tmp_path):
        write(tmp_path / "outside" / "p.yml", doc(rows=[row("sneaky", vision=True)]))
        manifest = SimpleNamespace(id="plug", plugin_dir=str(tmp_path / "plug"), llm_profiles=[{"path": "../outside"}])
        (tmp_path / "plug").mkdir()

        table = SeedCatalog(content, self.registry(manifest)).table()

        assert table.rows == ()

    def test_an_edited_or_added_local_file_is_picked_up_without_a_restart(self, content):
        clock = FakeClock()
        catalog = SeedCatalog(content, clock=clock)
        assert fields_of(catalog.table())["vision"] is False

        path = write(content / "local" / "mine.yml", doc(rows=[row("mine", vision=True)]))
        clock.now += 3
        assert fields_of(catalog.table())["vision"] is True

        write(path, doc(rows=[row("mine", max_images=6)]))
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
        clock.now += 3
        assert fields_of(catalog.table())["max_images"] == 6

    def test_the_files_are_not_looked_at_again_inside_the_recheck_window(self, content):
        clock = FakeClock()
        catalog = SeedCatalog(content, clock=clock)
        first = catalog.table()

        write(content / "local" / "mine.yml", doc(rows=[row("mine", vision=True)]))
        clock.now += 1.9
        assert catalog.table() is first
        assert fields_of(catalog.table())["vision"] is False

        clock.now += 0.2
        assert fields_of(catalog.table())["vision"] is True

    def test_the_table_is_reused_while_nothing_changed(self, content):
        write(content / "marketplace" / "a.yml", doc(rows=[row("r", vision=True)]))
        catalog = SeedCatalog(content)

        assert catalog.table() is catalog.table()

    def test_a_resolver_given_the_catalog_sees_local_rows(self, content):
        write(content / "local" / "mine.yml", doc(rows=[row("mine", max_images=8)]))
        resolver = LayeredProfileResolver(adapter_for(ProbeAdapter(name="probe")), SeedCatalog(content))

        assert resolver.resolve(make_config(model="m")).max_images == 8
