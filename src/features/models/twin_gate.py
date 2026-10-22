import threading
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

from src.features.models.file_digest import quick_signature


@dataclass(frozen=True)
class Twin:
    path: str
    size: int
    location_id: Optional[str] = None
    model_id: Optional[str] = None
    sha256: Optional[str] = None
    signature: Optional[str] = None


@dataclass(frozen=True)
class GateResult:
    equal: Tuple[Twin, ...] = ()
    different: Tuple[Twin, ...] = ()
    unknown: Tuple[Twin, ...] = ()

    @property
    def contested(self) -> bool:
        return bool(self.equal) or any(twin.sha256 for twin in self.unknown)

    @property
    def awaited_locations(self) -> Tuple[str, ...]:
        return tuple(twin.location_id for twin in self.equal if twin.location_id and not twin.sha256)


def evaluate(signature: Optional[str], twins: Sequence[Twin]) -> GateResult:
    equal, different, unknown = [], [], []
    for twin in twins:
        if signature is None or twin.signature is None:
            unknown.append(twin)
        elif twin.signature == signature:
            equal.append(twin)
        else:
            different.append(twin)
    return GateResult(tuple(equal), tuple(different), tuple(unknown))


class SignatureCache:
    def __init__(self, compute: Callable[[str, Optional[int]], Optional[str]] = quick_signature):
        self._compute = compute
        self._lock = threading.Lock()
        self._values: Dict[str, Optional[str]] = {}

    def of(self, path: str, size: Optional[int] = None) -> Optional[str]:
        with self._lock:
            if path in self._values:
                return self._values[path]
        value = self._compute(path, size)
        with self._lock:
            self._values[path] = value
        return value
