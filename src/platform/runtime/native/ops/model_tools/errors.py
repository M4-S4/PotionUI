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


class SourceFormatError(ModelToolsError):
    pass
