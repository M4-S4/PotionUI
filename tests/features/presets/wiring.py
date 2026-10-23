from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from unittest.mock import Mock

import pytest

from src.features.presets import PresetTemplateLoader
from src.features.presets.processor import PresetProcessor
from src.platform.templating.processor import TemplateProcessor

from ._bound_form import bound_form_data


@lru_cache(maxsize=None)
def _loader(root: str) -> PresetTemplateLoader:
    loader = PresetTemplateLoader([root])
    loader.load_presets()
    return loader


def find_template(*, preset_id: Optional[str] = None, suffix: Optional[str] = None, root: str = "content/presets"):
    loader = _loader(root)
    for preset in loader.presets:
        if preset_id is not None and preset.id == preset_id:
            return preset
        if suffix is not None and Path(preset.path).as_posix().endswith(suffix):
            return preset
    pytest.fail(f"preset {preset_id or suffix} not present under {root}")


def render(
    template: Any,
    mode: str,
    form_data: Dict[str, Any],
    prompts: Iterable[Dict[str, Any]] = (),
) -> List[Dict[str, Any]]:
    processor = PresetProcessor(
        template_processor=TemplateProcessor(settings=Mock()),
        settings=Mock(),
        preset_template_loader=Mock(),
    )
    generation_data = {
        "prompts": list(prompts),
        "mode": mode,
        "form_data": bound_form_data(template, mode, form_data),
    }
    return processor.process(template, generation_data)


def pipe(pipes: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    return next(p for p in pipes if p.get("id") == name or p["name"] == name)
