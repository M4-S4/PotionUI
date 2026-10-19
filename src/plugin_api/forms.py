from functools import lru_cache
from typing import FrozenSet

__all__ = ["core_field_type_names"]


@lru_cache(maxsize=1)
def core_field_type_names() -> FrozenSet[str]:
    from src.features.fields.builtin import register_builtin_fields
    from src.platform.plugins.field_types import FieldTypeRegistry

    registry = FieldTypeRegistry()
    register_builtin_fields(registry)
    return frozenset(d.type_name for d in registry.all() if not d.container)
