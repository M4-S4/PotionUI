from src.features.media_index.hooks import MEDIA_INDEX_HOOKS
from src.features.media_index.indexer import PASS_CLIP_EMBED, PASS_TAGS
from src.platform.plugins.hooks import HookChain, hooks_registry
from tests.features.media_index.test_indexer import FakeTagger, IndexerTestBase


class ChainPlugins:
    def __init__(self):
        self.chain = HookChain()
        self.events = []
        self.chain.register(MEDIA_INDEX_HOOKS.after_tags_written, "probe", self._record)

    def _record(self, context):
        self.events.append(dict(context.data))
        return context

    def execute_hook(self, hook, initial_data=None, context=None):
        return self.chain.execute(hook, context=context, initial_data=initial_data)


class TestAfterTagsWritten(IndexerTestBase):
    def _wired(self, **kwargs):
        manager = self._indexer(**kwargs)
        manager.plugins = ChainPlugins()
        return manager

    def test_fires_once_per_generation_with_its_tagged_files(self):
        gen_a = self.create_test_generation("gen_a", self.user_id)
        gen_b = self.create_test_generation("gen_b", self.user_id)
        self._make_file("a1", gen_a)
        self._make_file("a2", gen_a)
        self._make_file("b1", gen_b)
        manager = self._wired()
        for generation in ("gen_a", "gen_b"):
            manager.on_generation_complete(generation, "completed")

        manager.process_pending(PASS_TAGS, batch_size=10)

        events = {e["generation_id"]: sorted(e["file_ids"]) for e in manager.plugins.events}
        assert events == {"gen_a": ["a1", "a2"], "gen_b": ["b1"]}
        assert len(manager.plugins.events) == 2

    def test_does_not_fire_when_tagging_fails(self):
        gen = self.create_test_generation("gen1", self.user_id)
        self._make_file("f1", gen)
        manager = self._wired(tagger=FakeTagger(error=RuntimeError("no model")))
        manager.on_generation_complete("gen1", "completed")

        manager.process_pending(PASS_TAGS)

        assert manager.plugins.events == []

    def test_other_passes_do_not_fire(self):
        gen = self.create_test_generation("gen1", self.user_id)
        self._make_file("f1", gen)
        manager = self._wired()
        manager.repository.enqueue_files(["f1"], PASS_CLIP_EMBED)

        manager.process_pending(PASS_CLIP_EMBED)

        assert manager.plugins.events == []

    def test_a_missing_plugin_registry_is_fine(self):
        gen = self.create_test_generation("gen1", self.user_id)
        self._make_file("f1", gen)
        manager = self._indexer()
        manager.on_generation_complete("gen1", "completed")

        assert manager.process_pending(PASS_TAGS) == {"processed": 1, "failed": 0}

    def test_a_raising_hook_chain_does_not_fail_the_drain(self):
        gen = self.create_test_generation("gen1", self.user_id)
        self._make_file("f1", gen)
        manager = self._indexer()

        class Exploding:
            def execute_hook(self, *args, **kwargs):
                raise RuntimeError("boom")

        manager.plugins = Exploding()
        manager.on_generation_complete("gen1", "completed")

        assert manager.process_pending(PASS_TAGS) == {"processed": 1, "failed": 0}

    def test_the_hook_is_in_the_catalog(self):
        spec = hooks_registry.get("media_index.after_tags_written")

        assert spec is not None and "generation_id" in spec.payload
