from __future__ import annotations


class ModelToolsError(Exception):
    pass


class StreamMismatchError(ModelToolsError):
    pass


class WriteCancelledError(ModelToolsError):
    pass


class UnsupportedFamilyError(ModelToolsError):
    pass


class AlreadyQuantizedError(ModelToolsError):
    pass


class UnsupportedFormatError(ModelToolsError):
    pass


class SourceFormatError(ModelToolsError):
    pass


class BakeError(ModelToolsError):
    def __init__(self, message: str, unmatched: dict[str, list[str]] | None = None) -> None:
        super().__init__(message)
        self.unmatched = unmatched or {}


class MergeError(ModelToolsError):
    pass
