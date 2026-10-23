from unittest.mock import Mock

from src.features.chat.context_builder import ChatContextBuilder
from src.features.llm.tools.base import ToolServices
from tests.fixtures.fake_tab_drafts import make_manager, seed


def _builder(manager):
    runtime = Mock()
    runtime.tool_services = ToolServices(tab_drafts=manager)
    return ChatContextBuilder(runtime)


def _variables(builder, user_id, form_state):
    reader = builder.build_resource_context(user_id, "generation", form_state).prompt_variables
    return None if reader is None else reader()


def test_the_resource_context_reads_the_users_draft_for_the_open_tab():
    manager, _ = make_manager()
    seed(manager, "u1", [{"name": "mood", "type": "text", "value": "noir"}])

    assert _variables(_builder(manager), "u1", {"tab_id": "tab-1"}) == {"mood": {"type": "text", "value": "noir"}}


def test_another_users_resource_context_never_reads_it():
    manager, _ = make_manager()
    seed(manager, "u1", [{"name": "mood", "type": "text", "value": "noir"}])

    assert _variables(_builder(manager), "u2", {"tab_id": "tab-1"}) == {}


def test_without_a_tab_id_or_a_store_there_is_no_reader():
    manager, _ = make_manager()

    assert _variables(_builder(manager), "u1", {}) is None
    assert _variables(_builder(manager), "u1", None) is None
    assert _variables(_builder(None), "u1", {"tab_id": "tab-1"}) is None


def test_a_malformed_tab_id_reads_as_no_variables():
    manager, _ = make_manager()

    assert _variables(_builder(manager), "u1", {"tab_id": "../etc"}) == {}
