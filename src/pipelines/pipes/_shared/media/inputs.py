from pathlib import Path
from typing import Any, List


def as_flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def media_paths(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    return [str(item) for item in value if item]
