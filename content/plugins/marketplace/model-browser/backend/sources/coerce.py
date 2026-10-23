import math
import re
from typing import Any, Iterable, List, Optional

SHA256 = re.compile(r"[0-9a-fA-F]{64}")


def text(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def int_value(value: Any, *, whole_floats_only: bool = False) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and (not whole_floats_only or value.is_integer()):
        return int(value)
    return None


def as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def strings(values: Any, *, strip: bool = True) -> List[str]:
    if not isinstance(values, list):
        return []
    if strip:
        return list(dict.fromkeys(value.strip() for value in values if isinstance(value, str) and value.strip()))
    return list(dict.fromkeys(value for value in values if isinstance(value, str) and value))


def clean_list(values: Iterable[Any]) -> List[str]:
    return sorted({str(value).strip() for value in values or () if str(value).strip()})


def sha256_hex(value: Any) -> Optional[str]:
    return value.lower() if isinstance(value, str) and SHA256.fullmatch(value) else None
