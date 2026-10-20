import copy
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import yaml

_FAST_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

_CACHE_LIMIT = 256
_cache: "OrderedDict[str, tuple[int, int, Any]]" = OrderedDict()
_cache_lock = threading.Lock()


def load_yaml(source: Any) -> Any:
    text = source.read() if hasattr(source, "read") else source
    try:
        return yaml.load(text, Loader=_FAST_LOADER)
    except yaml.YAMLError:
        if _FAST_LOADER is yaml.SafeLoader:
            raise
        return yaml.load(text, Loader=yaml.SafeLoader)


def load_yaml_file(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return load_yaml(handle)


def load_yaml_cached(path: str | Path) -> Any:
    resolved = str(Path(path).resolve())
    stat = Path(resolved).stat()
    with _cache_lock:
        entry = _cache.get(resolved)
        if entry is not None and entry[0] == stat.st_mtime_ns and entry[1] == stat.st_size:
            _cache.move_to_end(resolved)
            return copy.deepcopy(entry[2])
    data = load_yaml_file(resolved)
    with _cache_lock:
        _cache[resolved] = (stat.st_mtime_ns, stat.st_size, data)
        _cache.move_to_end(resolved)
        while len(_cache) > _CACHE_LIMIT:
            _cache.popitem(last=False)
    return copy.deepcopy(data)


def clear_yaml_cache() -> None:
    with _cache_lock:
        _cache.clear()
