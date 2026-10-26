from dataclasses import replace
from typing import Any, Dict, List

import pytest

from src.features.llm.harness import codec
from src.features.llm.harness.adapters.gemini_interactions import ADAPTER_NAME, GeminiInteractionsAdapter, target_origin
from src.features.llm.harness.messages import Message, Role
from tests.features.llm.harness import cassettes

SCENARIOS = cassettes.scenario_dirs(ADAPTER_NAME)
EXPECTED = {"thinking_parallel_tools", "approval_pause", "in_band_error"}


def test_every_planned_scenario_is_present():
    assert {path.name for path in SCENARIOS} == EXPECTED


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda path: path.name)
def test_scenario_replays_through_the_adapter(scenario):
    cassettes.run_scenario(GeminiInteractionsAdapter(), scenario)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda path: path.name)
def test_scenario_is_declared_hand_built_with_its_source(scenario):
    for step in cassettes.load_scenario(scenario):
        assert step.meta["source"] == cassettes.HAND_BUILT
        assert step.meta["doc_url"].startswith("https://ai.google.dev/")
        assert step.meta["adapter"] == ADAPTER_NAME


def same_origin(stored: Message, request) -> Message:
    return replace(stored, origin=target_origin(request.config))


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda path: path.name)
def test_stored_steps_and_rebuilt_parts_send_the_same_bytes(scenario):
    adapter = GeminiInteractionsAdapter()
    history: List[Message] = []
    carried: Dict[str, Any] = {}
    for step in cassettes.load_scenario(scenario):
        carried = {**carried, **{key: value for key, value in step.input.items() if key != "messages"}}
        history = history + codec.from_json(List[Message], step.input.get("messages", []))
        profile = cassettes.profile_for(step.meta)
        request = cassettes.request_from(step.meta, carried, history)
        replayed = [same_origin(m, request) if m.role is Role.ASSISTANT else m for m in history]
        wire = adapter.render(cassettes.request_from(step.meta, carried, replayed), profile)
        assert wire.content() == cassettes.request_bytes(step.request["body"]), step.directory
        if step.step is not None:
            history = history + [codec.from_json(Message, step.step["message"])]
