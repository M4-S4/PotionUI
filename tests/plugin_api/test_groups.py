"""Tests for the `src.plugin_api.groups` surface."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.features.user_groups.hooks import USER_GROUP_HOOKS
from src.features.user_groups.records import UserGroup, UserGroupModel
from src.platform.security.user import AccountType, User
from src.plugin_api.groups import (
    assign_models_to_group,
    effective_content_policy,
    get_user_groups,
    group_member_ids,
    list_groups,
)


def _group(group_id, name, **fields):
    defaults = dict(id=group_id, name=name, is_system=False, content_policy=None)
    defaults.update(fields)
    return SimpleNamespace(**defaults)


def _container(**fields):
    return SimpleNamespace(**fields)


def _user(account_type):
    return User(id="u1", username="someone", email="someone@example.org", password_hash="x", account_type=account_type)


def _assigning_repository(existing=()):
    now = datetime(2026, 10, 13, tzinfo=timezone.utc)

    def assign_model_to_group(group_id, model_id):
        if model_id in existing:
            return None
        return UserGroupModel(id=f"a-{model_id}", group_id=group_id, model_id=model_id, assigned_at=now, updated_at=now)

    return Mock(
        get_group_by_id=Mock(return_value=UserGroup(id="g1", name="Artists", created_at=now, updated_at=now)),
        assign_model_to_group=Mock(side_effect=assign_model_to_group),
    )


def _plugins(blocked=False):
    def execute_hook(hook_name, context=None, initial_data=None):
        return Mock(data={**(initial_data or {}), "blocked": blocked}), []

    return Mock(execute_hook=Mock(side_effect=execute_hook))


def test_list_groups_answers_plain_dicts():
    repository = Mock(get_all_groups=Mock(return_value=[
        _group("g1", "All Users", is_system=True), _group("g2", "Artists", content_policy="blur"),
    ]))

    with patch("src.plugin_api.groups.get_container", return_value=_container(user_group_repository=repository)):
        result = list_groups()

    assert result == [
        {"id": "g1", "name": "All Users", "is_system": True, "content_policy": None},
        {"id": "g2", "name": "Artists", "is_system": False, "content_policy": "blur"},
    ]


def test_get_user_groups_answers_the_users_groups():
    repository = Mock(get_user_groups=Mock(return_value=[_group("g2", "Artists")]))

    with patch("src.plugin_api.groups.get_container", return_value=_container(user_group_repository=repository)):
        result = get_user_groups("user-1")
        assert get_user_groups("") == []

    assert [group["id"] for group in result] == ["g2"]
    repository.get_user_groups.assert_called_once_with("user-1")


def test_group_member_ids_deduplicates():
    members = [SimpleNamespace(user_id="u1"), SimpleNamespace(user_id="u2"), SimpleNamespace(user_id="u1")]
    repository = Mock(get_group_members=Mock(return_value=members))

    with patch("src.plugin_api.groups.get_container", return_value=_container(user_group_repository=repository)):
        assert group_member_ids("g1") == ["u1", "u2"]
        assert group_member_ids("") == []


def test_effective_content_policy_is_the_resolved_mode():
    resolver = Mock(resolve=Mock(return_value=SimpleNamespace(mode="blur")))

    with patch("src.plugin_api.groups.get_container", return_value=_container(content_policy_resolver=resolver)):
        assert effective_content_policy("user-1") == "blur"
        effective_content_policy("")

    assert [c.args for c in resolver.resolve.call_args_list] == [("user-1",), (None,)]


def test_assign_models_to_group_fires_the_group_hooks_in_the_admins_name_and_skips_models_already_there():
    repository = _assigning_repository(existing={"m-old"})
    plugins = _plugins()
    container = _container(user_group_repository=repository, plugin_registry=plugins)

    with patch("src.plugin_api.groups.get_container", return_value=container):
        assigned = assign_models_to_group("g1", ["m-new", "m-old", ""], _user(AccountType.ADMIN))

    assert assigned == ["m-new"]
    assert [c.args for c in repository.assign_model_to_group.call_args_list] == [("g1", "m-new"), ("g1", "m-old")]
    fired = [(c.args[0], c.kwargs["initial_data"]) for c in plugins.execute_hook.call_args_list]
    assert fired == [
        (USER_GROUP_HOOKS.before_assign_resource, {"group_id": "g1", "resource_type": "model", "resource_id": "m-new", "admin_id": "u1"}),
        (USER_GROUP_HOOKS.after_assign_resource, {"group_id": "g1", "resource_type": "model", "resource_id": "m-new", "assignment_id": "a-m-new"}),
        (USER_GROUP_HOOKS.before_assign_resource, {"group_id": "g1", "resource_type": "model", "resource_id": "m-old", "admin_id": "u1"}),
    ]


def test_assign_models_to_group_honours_another_plugins_veto():
    repository = _assigning_repository()
    container = _container(user_group_repository=repository, plugin_registry=_plugins(blocked=True))

    with patch("src.plugin_api.groups.get_container", return_value=container):
        assigned = assign_models_to_group("g1", ["m1"], _user(AccountType.ADMIN))

    assert assigned == []
    repository.assign_model_to_group.assert_not_called()


def test_assign_models_to_group_refuses_a_non_admin_actor_even_with_nothing_to_assign():
    repository = _assigning_repository()
    container = _container(user_group_repository=repository, plugin_registry=_plugins())

    with patch("src.plugin_api.groups.get_container", return_value=container):
        with pytest.raises(ValueError):
            assign_models_to_group("g1", [], _user(AccountType.USER))

    repository.assign_model_to_group.assert_not_called()


def test_assign_models_to_group_refuses_an_unknown_group():
    repository = Mock(get_group_by_id=Mock(return_value=None), assign_model_to_group=Mock())
    container = _container(user_group_repository=repository, plugin_registry=_plugins())

    with patch("src.plugin_api.groups.get_container", return_value=container):
        with pytest.raises(ValueError):
            assign_models_to_group("missing", ["m1"], _user(AccountType.ADMIN))

    repository.assign_model_to_group.assert_not_called()
