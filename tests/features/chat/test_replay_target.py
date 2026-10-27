from types import SimpleNamespace

import pytest

from src.features.chat.conversation import ConversationRunner
from src.features.llm.harness.build import user_message
from tests.fixtures.llm_engine import scripted_config, scripted_gateway
from tests.fixtures.llm_harness import ScriptedAdapter


def runner_for(gateway):
    return ConversationRunner(SimpleNamespace(llm_service=gateway))


@pytest.mark.asyncio
async def test_history_replays_for_the_same_origin_and_profile_the_send_uses():
    gateway = scripted_gateway(ScriptedAdapter(), scripted_config(api_key="sk-x", profile_overrides={"context_window": 8192}))

    prepared = await gateway.prepare("cfg-scripted", [user_message("hi")])
    origin, profile = runner_for(gateway)._replay_target(SimpleNamespace(llm_config_id="cfg-scripted"))

    assert (origin, profile) == (prepared.origin, prepared.profile)
    assert origin.credential is not None and profile.context_window == 8192


@pytest.mark.parametrize("config, error", [
    (scripted_config(id="other"), "not found"),
    (scripted_config(enabled=False), "disabled"),
    (scripted_config(type="nope"), "Unsupported LLM type"),
])
@pytest.mark.asyncio
async def test_a_config_the_send_would_refuse_fails_the_replay_target_the_same_way(config, error):
    gateway = scripted_gateway(ScriptedAdapter(), config)
    runner = runner_for(gateway)

    with pytest.raises(ValueError, match=error):
        await gateway.prepare("cfg-scripted", [user_message("hi")])
    with pytest.raises(ValueError, match=error):
        runner._replay_target(SimpleNamespace(llm_config_id="cfg-scripted"))
