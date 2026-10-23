from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class DomainError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400, **extra: Any):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.extra = extra

    def payload(self) -> Dict[str, Any]:
        return {"success": False, "error": self.code, "message": self.message, **self.extra}


class NotFoundError(DomainError, ValueError):
    def __init__(self, code: str, message: str, **extra: Any):
        super().__init__(code, message, 404, **extra)


class AccessDeniedError(DomainError, ValueError):
    def __init__(self, code: str, message: str, **extra: Any):
        super().__init__(code, message, 403, **extra)


class InvalidInputError(DomainError, ValueError):
    def __init__(self, code: str, message: str, **extra: Any):
        super().__init__(code, message, 400, **extra)


class ConflictError(DomainError, ValueError):
    def __init__(self, code: str, message: str, **extra: Any):
        super().__init__(code, message, 409, **extra)


@dataclass(frozen=True)
class DomainFailure:
    code: str
    message: str
    status_code: int
    detail: Dict[str, Any] = field(default_factory=dict)


def domain_failure(exc: BaseException) -> Optional[DomainFailure]:
    if not isinstance(exc, DomainError):
        return None
    return DomainFailure(exc.code, exc.message, exc.status_code, exc.payload())
