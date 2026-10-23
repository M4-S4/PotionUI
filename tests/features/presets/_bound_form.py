from __future__ import annotations

from typing import Any, Dict

from src.features.forms.binding import bind_form


def bound_form_data(template: Any, mode: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    bound = bind_form(
        template, mode, form_name=None, raw_form_data=raw,
        user_id=None, storage_dir=None,
    )
    return dict(bound.values)
