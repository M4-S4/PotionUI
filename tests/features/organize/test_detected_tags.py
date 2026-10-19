from typing import Any, Dict, List, Optional

import pytest

from src.features.media_index.indexer import PASS_TAGS, MediaIndexer
from src.features.media_index.repository import MediaIndexRepository
from src.features.media_index.tagger import SystemTagPrediction, TaggingResult
from src.features.organize.evaluator import evaluate, sql_filter
from src.platform.plugins.hooks import HookChain
from tests.features.media_index.test_indexer import (
    FakeFileStore,
    FakeGalleryPromptStore,
    FakeGalleryStore,
    FakeTextEmbedder,
    FakeVisionEmbedder,
)
from tests.features.organize.conftest import collection_action, rule_body

DOG = [{"fact": "detected_tags", "operator": "has", "value": ["dog"]}]


class ScriptedTagger:
    provenance = "model-a"

    def __init__(self, tags_by_path: Optional[Dict[str, List[str]]] = None, default: Optional[List[str]] = None):
        self.tags_by_path = tags_by_path or {}
        self.default = default if default is not None else ["dog"]

    def tag_image_file(self, path: str) -> TaggingResult:
        names = next((v for k, v in self.tags_by_path.items() if k in path), self.default)
        return TaggingResult(
            tags=[SystemTagPrediction(name, "general", 0.9) for name in names],
            ratings={"general": 0.9, "explicit": 0.01},
        )


class ChainPlugins:
    def __init__(self, chain: HookChain):
        self.chain = chain

    def execute_hook(self, hook: str, initial_data: Optional[Dict[str, Any]] = None, context=None):
        return self.chain.execute(hook, context=context, initial_data=initial_data)


@pytest.fixture
def chain(worker):
    hook_chain = HookChain()
    worker.subscribe(hook_chain)
    return hook_chain


@pytest.fixture
def indexer(chain):
    def build(tagger=None, plugins=True):
        return MediaIndexer(
            repository=MediaIndexRepository(),
            tagger_provider=tagger or ScriptedTagger(),
            file_service=FakeFileStore(),
            vision_embedder=FakeVisionEmbedder(),
            gallery_vector_store=FakeGalleryStore(),
            text_embedding_provider=FakeTextEmbedder(),
            gallery_prompt_vector_store=FakeGalleryPromptStore(),
            plugins=ChainPlugins(chain) if plugins else None,
        )

    return build


def tag_pass(indexer, generation_id):
    indexer.on_generation_complete(generation_id, "completed")
    return indexer.process_generation(generation_id, PASS_TAGS)


def write_tags(seed, generation_id, *names, category="general", rating=None):
    file_id = seed.rows("SELECT file_id FROM generation_files WHERE generation_id = ?", (generation_id,))[0]["file_id"]
    for name in names:
        seed._exec(
            "INSERT INTO media_system_tags (id, file_id, generation_id, tag, category, confidence, provenance) "
            "VALUES (?, ?, ?, ?, ?, 0.9, 'model-a')",
            (f"{file_id}-{name}", file_id, generation_id, name, category),
        )


def dog_rule(manager, user, collection_id):
    return manager.create_rule(user, rule_body(conditions=DOG, actions=[collection_action(collection_id)]))


def test_a_completed_generation_is_filed_once_the_tagger_writes_dog(seed, manager, worker, chain, indexer):
    user = seed.user("u1")
    dogs = seed.collection("u1")
    dog_rule(manager, user, dogs)
    generation = seed.generation("u1")
    chain.execute("generation.after_complete", initial_data={"generation_id": generation, "status": "completed"})
    worker.drain()
    assert seed.members(dogs) == set()

    tag_pass(indexer(), generation)
    worker.drain()

    assert seed.members(dogs) == {generation}


def test_other_detected_tags_do_not_file_the_generation(seed, manager, worker, indexer):
    user = seed.user("u1")
    dogs = seed.collection("u1")
    dog_rule(manager, user, dogs)
    generation = seed.generation("u1")

    tag_pass(indexer(ScriptedTagger(default=["cat"])), generation)
    worker.drain()

    assert seed.members(dogs) == set()


def test_a_rerun_files_nothing_new_and_logs_one_run(seed, manager, worker, indexer):
    user = seed.user("u1")
    dogs = seed.collection("u1")
    dog_rule(manager, user, dogs)
    generation = seed.generation("u1")
    index = indexer()

    tag_pass(index, generation)
    worker.drain()
    seed._exec("UPDATE media_index_queue SET status = 'pending'")
    index.process_generation(generation, PASS_TAGS)
    manager.handle_event("generation_detected_tags_changed", {"generation_id": generation})
    worker.drain()

    assert seed.members(dogs) == {generation}
    runs = manager.activity(user, "generation", None, None, 50)["runs"]
    assert [(r["kind"], r["applied"]) for r in runs] == [("live", 1)]
    assert seed.rows("SELECT COUNT(*) AS c FROM organize_applications")[0]["c"] == 1


def test_a_later_tag_event_does_not_refile_an_item_the_user_removed(seed, manager, worker, indexer):
    user = seed.user("u1")
    dogs = seed.collection("u1")
    dog_rule(manager, user, dogs)
    generation = seed.generation("u1")
    index = indexer()
    tag_pass(index, generation)
    worker.drain()
    seed._exec("DELETE FROM collection_generations WHERE generation_id = ?", (generation,))

    manager.handle_event("generation_detected_tags_changed", {"generation_id": generation})

    assert seed.members(dogs) == set()
    assert seed.rows("SELECT COUNT(*) AS c FROM organize_applications")[0]["c"] == 1


def test_rules_only_see_their_owners_items(seed, manager, worker, indexer):
    owner = seed.user("u1")
    other = seed.user("u2")
    mine = seed.collection("u1")
    theirs = seed.collection("u2")
    dog_rule(manager, owner, mine)
    dog_rule(manager, other, theirs)
    generation = seed.generation("u1")

    tag_pass(indexer(), generation)
    worker.drain()

    assert seed.members(mine) == {generation}
    assert seed.members(theirs) == set()


def test_without_a_hook_chain_nothing_extra_runs(seed, manager, worker, indexer):
    user = seed.user("u1")
    dogs = seed.collection("u1")
    dog_rule(manager, user, dogs)
    generation = seed.generation("u1")

    result = tag_pass(indexer(plugins=False), generation)
    worker.drain()

    assert result["processed"] == 1
    assert worker.pending() == 0
    assert seed.members(dogs) == set()


def test_a_failed_tagger_run_announces_nothing(seed, manager, worker, indexer):
    user = seed.user("u1")
    dog_rule(manager, user, seed.collection("u1"))
    generation = seed.generation("u1")

    class Broken(ScriptedTagger):
        def tag_image_file(self, path):
            raise RuntimeError("model missing")

    result = tag_pass(indexer(Broken()), generation)

    assert result == {"processed": 0, "failed": 1}
    assert worker.pending() == 0


def test_a_negated_detected_rule_waits_for_the_tags_instead_of_filing_on_completion(seed, manager, worker, chain, indexer):
    user = seed.user("u1")
    safe = seed.collection("u1")
    manager.create_rule(user, rule_body(
        conditions=[{"fact": "detected_tags", "operator": "has_not", "value": ["dog"]}],
        actions=[collection_action(safe)],
    ))
    with_dog = seed.generation("u1")
    without = seed.generation("u1")
    for generation in (with_dog, without):
        chain.execute("generation.after_complete", initial_data={"generation_id": generation, "status": "completed"})
    worker.drain()
    assert seed.members(safe) == set()

    tag_pass(indexer(ScriptedTagger(tags_by_path={}, default=["dog"])), with_dog)
    tag_pass(indexer(ScriptedTagger(default=["cat"])), without)
    worker.drain()

    assert seed.members(safe) == {without}


def test_backfill_files_existing_tagged_generations(seed, manager, backfill):
    user = seed.user("u1")
    dogs = seed.collection("u1")
    tagged = seed.generation("u1")
    other = seed.generation("u1")
    untagged = seed.generation("u1")
    write_tags(seed, tagged, "dog", "grass")
    write_tags(seed, other, "cat")
    rule = dog_rule(manager, user, dogs)

    job = backfill.start(user, rule["id"])

    assert backfill.get_job(user, job["id"])["applied"] == 1
    assert seed.members(dogs) == {tagged}
    assert untagged not in seed.members(dogs)


def test_preview_counts_detected_matches(seed, manager):
    user = seed.user("u1")
    dogs = seed.collection("u1")
    write_tags(seed, seed.generation("u1"), "dog")
    write_tags(seed, seed.generation("u1"), "dog")
    write_tags(seed, seed.generation("u1"), "cat")
    seed.generation("u1")
    seed.user("u2")
    write_tags(seed, seed.generation("u2"), "dog")

    preview = manager.preview(user, {
        "subject": "generation", "conditions": DOG, "actions": [collection_action(dogs)],
    })

    assert (preview["matched"], preview["would_change"]) == (2, 2)


def test_ratings_are_not_detected_tags(seed, manager):
    seed.user("u1")
    generation = seed.generation("u1")
    write_tags(seed, generation, "dog", category="rating")

    where, params, complete = sql_filter(manager.c.registry, "generation", "all", DOG)
    ids = manager.c.items.candidate_ids("generation", "u1", where, params, False, False)

    assert complete
    assert ids == []
    assert manager.c.items.generations("u1", [generation])[0]["detected_tags"] == []


def test_negated_operator_matches_sql_and_evaluator(seed, manager):
    seed.user("u1")
    with_dog = seed.generation("u1")
    without = seed.generation("u1")
    write_tags(seed, with_dog, "Dog")
    conditions = [{"fact": "detected_tags", "operator": "has_not", "value": ["dog"]}]

    where, params, complete = sql_filter(manager.c.registry, "generation", "all", conditions)
    ids = manager.c.items.candidate_ids("generation", "u1", where, params, False, False)
    loaded = manager.load_items("generation", "u1", [with_dog, without], False, False)

    assert complete
    assert set(ids) == {without}
    assert {i.item_id for i in loaded if evaluate(manager.c.registry, "all", conditions, i)} == {without}


def test_the_vocabulary_lists_only_the_users_detected_tags_most_used_first(seed, manager):
    user = seed.user("u1")
    seed.user("u2")
    first, second = seed.generation("u1"), seed.generation("u1")
    write_tags(seed, first, "dog", "grass")
    write_tags(seed, second, "dog")
    write_tags(seed, seed.generation("u2"), "secret")
    write_tags(seed, seed.generation("u1"), "nsfw", category="rating")

    assert [o["value"] for o in manager.fact_options(user, "detected_tags", "generation", "", 50)] == ["dog", "grass"]
    assert [o["value"] for o in manager.fact_options(user, "detected_tags", "generation", "gra", 50)] == ["grass"]


def test_the_catalog_tells_the_editor_when_a_fact_can_first_be_known(seed, manager):
    user = seed.user("u1")

    facts = {f["key"]: f for f in manager.catalog(user, "generation")["facts"]}

    assert facts["detected_tags"]["triggers"] == ["detected_tags_changed"]
    assert "item_created" in facts["tags"]["triggers"]


def test_the_fact_is_offered_for_generations_only(seed, manager):
    user = seed.user("u1")

    generation = {f["key"] for f in manager.catalog(user, "generation")["facts"]}
    upload = {f["key"] for f in manager.catalog(user, "upload")["facts"]}

    assert "detected_tags" in generation and "detected_tags" not in upload
