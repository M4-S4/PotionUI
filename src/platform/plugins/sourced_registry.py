from typing import ClassVar, Protocol


class DuplicateRegistrationError(ValueError):
    pass


class _Sourced(Protocol):
    source: str


class SourcedRegistry[T: _Sourced]:
    key_attr: ClassVar[str]
    label: ClassVar[str]
    duplicate_error: ClassVar[type[DuplicateRegistrationError]] = DuplicateRegistrationError

    def __init__(self) -> None:
        self._entries: dict[str, T] = {}

    def _duplicate_message(self, key: str) -> str:
        return f"{self.label} already registered: '{key}'"

    def register(self, entry: T) -> None:
        key = getattr(entry, self.key_attr)
        if key in self._entries:
            raise self.duplicate_error(self._duplicate_message(key))
        self._entries[key] = entry

    def unregister_source(self, source: str) -> int:
        stale = [key for key, entry in self._entries.items() if entry.source == source]
        for key in stale:
            del self._entries[key]
        return len(stale)

    def get(self, key: str) -> T | None:
        return self._entries.get(key)

    def all(self) -> list[T]:
        return list(self._entries.values())
