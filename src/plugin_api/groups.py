"""User groups and the content policy that applies to a user.

Groups are `{"id", "name", "is_system", "content_policy"}` dicts.
`effective_content_policy` is the NSFW policy the application enforces for a
user ("allowed", "blur" or "blocked"): the instance setting made stricter by
their groups. `assign_models_to_group` makes catalog models visible to every
member of a group, in an admin's name and through the group operation and its
hooks.
"""

from typing import Any, Dict, List, Optional

from src.platform.plugins.runtime_registries import get_container
from src.platform.security.user import User

__all__ = [
    "assign_models_to_group",
    "effective_content_policy",
    "get_user_groups",
    "group_member_ids",
    "list_groups",
]


def _group(group: Any) -> Dict[str, Any]:
    return {
        "id": group.id,
        "name": group.name,
        "is_system": bool(getattr(group, "is_system", False)),
        "content_policy": getattr(group, "content_policy", None),
    }


def list_groups() -> List[Dict[str, Any]]:
    """Every user group, in the application's order (by name)."""
    return [_group(group) for group in get_container().user_group_repository.get_all_groups()]


def get_user_groups(user_id: str) -> List[Dict[str, Any]]:
    """The groups `user_id` belongs to."""
    if not user_id:
        return []
    return [_group(group) for group in get_container().user_group_repository.get_user_groups(user_id)]


def group_member_ids(group_id: str) -> List[str]:
    """The ids of a group's members, without duplicates; none for an unknown group."""
    if not group_id:
        return []
    members = get_container().user_group_repository.get_group_members(group_id)
    return list(dict.fromkeys(member.user_id for member in members if member.user_id))


def effective_content_policy(user_id: Optional[str]) -> str:
    """The content policy enforced for `user_id`: "allowed", "blur" or
    "blocked". Without a user, the instance policy."""
    return get_container().content_policy_resolver.resolve(user_id or None).mode


def assign_models_to_group(group_id: str, model_ids: List[str], acting_user: User) -> List[str]:
    """Make models visible to every member of a group, in `acting_user`'s name
    and through the application's group operation (`require_admin`, the
    `user_groups.before_assign_resource` veto, `after_assign_resource`); returns
    the ids newly assigned. `ValueError` when `acting_user` is not an admin or
    the group is unknown."""
    from src.features.user_groups.operations.resources import assign_models

    container = get_container()
    assigned = assign_models(
        container.user_group_repository,
        container.plugin_registry,
        group_id,
        [model_id for model_id in model_ids if model_id],
        acting_user,
    )
    return [assignment.model_id for assignment in assigned]
