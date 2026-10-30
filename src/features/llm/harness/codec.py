import collections.abc
import dataclasses
import json
import types
import typing
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, Tuple, Type, TypeVar, Union

T = TypeVar("T")

TAG = "type"


def to_json(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        encoded: Dict[str, Any] = {}
        tag = getattr(type(value), "TYPE", None)
        if tag is not None:
            encoded[TAG] = tag
        floats = _float_fields(type(value))
        for item in dataclasses.fields(value):
            field_value = getattr(value, item.name)
            if item.name in floats and isinstance(field_value, int) and not isinstance(field_value, bool):
                field_value = float(field_value)
            encoded[item.name] = to_json(field_value)
        return encoded
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (frozenset, set)):
        return sorted(to_json(item) for item in value)
    if isinstance(value, (tuple, list)):
        return [to_json(item) for item in value]
    if isinstance(value, collections.abc.Mapping):
        return {_key(key): to_json(item) for key, item in value.items()}
    if isinstance(value, (bytes, bytearray)):
        raise TypeError("bytes are not JSON; store base64 text instead")
    return value


def _key(key: Any) -> str:
    if isinstance(key, Enum):
        return key.value
    if not isinstance(key, str):
        raise TypeError(f"JSON object keys must be strings, got {type(key).__name__}")
    return key


def dumps(value: Any) -> str:
    return json.dumps(to_json(value), ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def loads(hint: Type[T], text: str) -> T:
    return from_json(hint, json.loads(text))


@lru_cache(maxsize=None)
def _hints(cls: type) -> Dict[str, Any]:
    return typing.get_type_hints(cls)


@lru_cache(maxsize=None)
def _init_fields(cls: type) -> Tuple[str, ...]:
    return tuple(item.name for item in dataclasses.fields(cls) if item.init)


@lru_cache(maxsize=None)
def _float_fields(cls: type) -> frozenset:
    names = set()
    for name, hint in _hints(cls).items():
        members = typing.get_args(hint) if typing.get_origin(hint) in (Union, types.UnionType) else (hint,)
        if float in members and int not in members:
            names.add(name)
    return frozenset(names)


def field_from_json(cls: type, name: str, data: Any) -> Any:
    return from_json(_hints(cls)[name], data)


def from_json(hint: Any, data: Any) -> Any:
    if hint is Any:
        return data
    origin = typing.get_origin(hint)
    if origin is Union or origin is types.UnionType:
        return _union(typing.get_args(hint), data)
    if origin is tuple:
        args = typing.get_args(hint)
        items = _sequence(data, hint)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(from_json(args[0], item) for item in items)
        if len(args) != len(items):
            raise ValueError(f"expected {len(args)} items for {hint}, got {len(items)}")
        return tuple(from_json(arg, item) for arg, item in zip(args, items))
    if origin in (frozenset, set):
        (arg,) = typing.get_args(hint)
        return frozenset(from_json(arg, item) for item in _sequence(data, hint))
    if origin is list:
        (arg,) = typing.get_args(hint)
        return [from_json(arg, item) for item in _sequence(data, hint)]
    if origin in (dict, collections.abc.Mapping, typing.Mapping):
        key_hint, value_hint = typing.get_args(hint)
        if not isinstance(data, dict):
            raise ValueError(f"expected an object for {hint}, got {type(data).__name__}")
        return {from_json(key_hint, key): from_json(value_hint, item) for key, item in data.items()}
    if isinstance(hint, type) and issubclass(hint, Enum):
        return hint(data)
    if isinstance(hint, type) and dataclasses.is_dataclass(hint):
        return _dataclass(hint, data)
    return _scalar(hint, data)


def _sequence(data: Any, hint: Any) -> list:
    if not isinstance(data, list):
        raise ValueError(f"expected a list for {hint}, got {type(data).__name__}")
    return data


def _union(members: Tuple[Any, ...], data: Any) -> Any:
    if data is None:
        if type(None) in members:
            return None
        raise ValueError(f"null is not allowed for {members}")
    candidates = [member for member in members if member is not type(None)]
    if len(candidates) == 1:
        return from_json(candidates[0], data)
    tagged = {
        member.TYPE: member
        for member in candidates
        if isinstance(member, type) and dataclasses.is_dataclass(member) and hasattr(member, "TYPE")
    }
    if tagged:
        if not isinstance(data, dict) or data.get(TAG) not in tagged:
            raise ValueError(f"unknown {TAG} {data.get(TAG) if isinstance(data, dict) else data!r}")
        return _dataclass(tagged[data[TAG]], data)
    for member in candidates:
        if member in (bool, int, float, str) and _is_scalar(member, data):
            return data
    raise ValueError(f"{data!r} matches none of {candidates}")


def _is_scalar(hint: type, data: Any) -> bool:
    if hint is bool:
        return isinstance(data, bool)
    if hint is int:
        return isinstance(data, int) and not isinstance(data, bool)
    if hint is float:
        return isinstance(data, (int, float)) and not isinstance(data, bool)
    return isinstance(data, hint)


def _scalar(hint: Any, data: Any) -> Any:
    if hint in (bool, int, float, str):
        if not _is_scalar(hint, data):
            raise ValueError(f"expected {hint.__name__}, got {type(data).__name__}")
        return float(data) if hint is float else data
    if hint is type(None):
        if data is not None:
            raise ValueError("expected null")
        return None
    return data


def _dataclass(cls: type, data: Any) -> Any:
    if not isinstance(data, dict):
        raise ValueError(f"expected an object for {cls.__name__}, got {type(data).__name__}")
    tag = getattr(cls, "TYPE", None)
    names = _init_fields(cls)
    allowed = set(names) | ({TAG} if tag is not None else set())
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    if tag is not None and data.get(TAG) != tag:
        raise ValueError(f"{cls.__name__} expects {TAG} {tag!r}, got {data.get(TAG)!r}")
    hints = _hints(cls)
    return cls(**{name: from_json(hints[name], data[name]) for name in names if name in data})
