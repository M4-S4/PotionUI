from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterable, Iterator, Optional, Set


class FormLookups:
    def __init__(self) -> None:
        self._configurations: Dict[str, Dict[str, Any]] = {}
        self._filenames: Dict[str, bool] = {}

    def configuration(self, preset_id: str) -> Dict[str, Any]:
        if preset_id not in self._configurations:
            try:
                from src.features.presets.repository import preset_repo
                self._configurations[preset_id] = preset_repo.get_preset_configuration(preset_id)
            except Exception:
                return {}
        return self._configurations[preset_id]

    def existing_filenames(self, candidates: Iterable[str]) -> Set[str]:
        wanted = list(dict.fromkeys(candidates))
        missing = [name for name in wanted if name not in self._filenames]
        if missing:
            try:
                from src.features.models.repository import model_repo
                found = model_repo.existing_filenames(missing)
            except Exception:
                return set()
            for name in missing:
                self._filenames[name] = name in found
        return {name for name in wanted if self._filenames[name]}


_current: ContextVar[Optional[FormLookups]] = ContextVar("form_lookups", default=None)


def current_form_lookups() -> FormLookups:
    return _current.get() or FormLookups()


@contextmanager
def form_lookups_scope() -> Iterator[FormLookups]:
    lookups = FormLookups()
    token = _current.set(lookups)
    try:
        yield lookups
    finally:
        _current.reset(token)
