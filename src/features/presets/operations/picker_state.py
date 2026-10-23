import threading
from typing import Dict, Iterable, List, Set

from src.features.presets.collaborators import PresetCollaborators
from src.features.presets.exceptions import PresetNotFoundException
from src.features.presets.operations.query import list_presets
from src.platform.security.user import User

PINNED_KEY = "preset_picker_pinned"
RECENT_KEY = "preset_picker_recent"
PINNED_LIMIT = 50
RECENT_LIMIT = 8

_locks_guard = threading.Lock()
_user_locks: Dict[str, threading.Lock] = {}


def _user_lock(user: User) -> threading.Lock:
    with _locks_guard:
        return _user_locks.setdefault(str(user.id), threading.Lock())


def _visible_ids(collaborators: PresetCollaborators, user: User) -> Set[str]:
    return {preset["id"] for preset in list_presets(collaborators, user)}


def _stored(collaborators: PresetCollaborators, user: User, key: str) -> List[str]:
    value = collaborators.settings.get_setting(key, default=[], user_id=user.id)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _kept(ids: Iterable[str], visible: Set[str]) -> List[str]:
    seen: Set[str] = set()
    kept: List[str] = []
    for preset_id in ids:
        if preset_id in visible and preset_id not in seen:
            seen.add(preset_id)
            kept.append(preset_id)
    return kept


def _state(collaborators: PresetCollaborators, user: User, visible: Set[str]) -> Dict[str, List[str]]:
    return {
        "pinned": _kept(_stored(collaborators, user, PINNED_KEY), visible),
        "recent": _kept(_stored(collaborators, user, RECENT_KEY), visible),
    }


def _require_visible(collaborators: PresetCollaborators, user: User, preset_id: str) -> Set[str]:
    visible = _visible_ids(collaborators, user)
    if preset_id not in visible:
        raise PresetNotFoundException(preset_id)
    return visible


def get_picker_state(collaborators: PresetCollaborators, user: User) -> Dict[str, List[str]]:
    return _state(collaborators, user, _visible_ids(collaborators, user))


def pin_preset(collaborators: PresetCollaborators, user: User, preset_id: str) -> Dict[str, List[str]]:
    visible = _require_visible(collaborators, user, preset_id)
    with _user_lock(user):
        pinned = _kept([*_stored(collaborators, user, PINNED_KEY), preset_id], visible)[-PINNED_LIMIT:]
        collaborators.settings.set_setting(PINNED_KEY, pinned, user_id=user.id)
        return _state(collaborators, user, visible)


def unpin_preset(collaborators: PresetCollaborators, user: User, preset_id: str) -> Dict[str, List[str]]:
    with _user_lock(user):
        stored = _stored(collaborators, user, PINNED_KEY)
        if preset_id in stored:
            collaborators.settings.set_setting(PINNED_KEY, [item for item in stored if item != preset_id], user_id=user.id)
    return get_picker_state(collaborators, user)


def push_recent_preset(collaborators: PresetCollaborators, user: User, preset_id: str) -> Dict[str, List[str]]:
    visible = _require_visible(collaborators, user, preset_id)
    with _user_lock(user):
        recent = _kept([preset_id, *_stored(collaborators, user, RECENT_KEY)], visible)[:RECENT_LIMIT]
        collaborators.settings.set_setting(RECENT_KEY, recent, user_id=user.id)
        return _state(collaborators, user, visible)
