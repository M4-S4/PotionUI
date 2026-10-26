from dataclasses import replace
from typing import Any, Dict, List

import pytest

from src.features.llm.harness import codec
from src.features.llm.harness.adapters.gemini_interactions import ADAPTER_NAME, GeminiInteractionsAdapter
from src.features.llm.harness.messages import Message, Role
from tests.features.llm.harness import cassettes

SCENARIOS = cassettes.scenario_dirs(ADAPTER_NAME)
PLANNED = {"thinking_parallel_tools", "approval_pause", "in_band_error"}


def test_every_planned_scenario_is_present():
    assert PLANNED <= {path.name for path in SCENARIOS}


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda path: path.name)
def test_scenario_replays_through_the_adapter(scenario):
    cassettes.run_scenario(GeminiInteractionsAdapter(), scenario)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda path: path.name)
def test_scenario_declares_where_it_came_from(scenario):
    for step in cassettes.load_scenario(scenario):
        assert step.meta["adapter"] == ADAPTER_NAME
        assert step.meta["source"] in {cassettes.HAND_BUILT, cassettes.RECORDED}
        if step.meta["source"] == cassettes.HAND_BUILT:
            assert step.meta["doc_url"].startswith("https://ai.google.dev/")


def foreign_credential(stored: Message) -> Message:
    return replace(stored, origin=replace(stored.origin, credential="another-key"))


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda path: path.name)
def test_history_rebuilt_from_parts_sends_the_same_bytes_as_the_stored_steps(scenario):
    adapter = GeminiInteractionsAdapter()
    history: List[Message] = []
    carried: Dict[str, Any] = {}
    for step in cassettes.load_scenario(scenario):
        carried = {**carried, **{key: value for key, value in step.input.items() if key != "messages"}}
        history = history + codec.from_json(List[Message], step.input.get("messages", []))
        profile = cassettes.profile_for(step.meta)
        rebuilt = [foreign_credential(m) if m.role is Role.ASSISTANT else m for m in history]
        wire = adapter.render(cassettes.request_from(step.meta, carried, rebuilt), profile)
        assert wire.content() == cassettes.request_bytes(step.request["body"]), step.directory
        if step.step is not None:
            history = history + [codec.from_json(Message, step.step["message"])]
