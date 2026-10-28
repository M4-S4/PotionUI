from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.features.presets.templates import FieldTemplate, PresetTemplate, default_form_name

MODEL_FIELD_TYPES = frozenset({"model", "models"})
LORA_PICKER_FIELD_TYPE = "lora_picker"
MEDIA_FIELD_TYPES = frozenset({"image", "video", "audio", "media", "file"})
MEDIA_EXTENSIONS = {"image": "png", "video": "mp4", "audio": "wav", "media": "png", "file": "bin"}
TEXT_FIELD_TYPES = frozenset({"string", "textbox"})


def requires_by_reaction(field: FieldTemplate) -> bool:
    for reaction in field.reactions or []:
        validation = (reaction.get("then") or {}).get("update_validation") or {}
        if validation.get("required") is True:
            return True
    return False


def collect_named_fields(fields: List[FieldTemplate], form_data: Dict[str, Any]) -> None:
    for field in fields:
        if field.name and field.default is not None:
            form_data[field.name] = field.default
        if isinstance(field.children, list):
            collect_named_fields(field.children, form_data)


def inject_unresolvable_defaults(fields: List[FieldTemplate], form_data: Dict[str, Any], tag: str) -> None:
    for field in fields:
        name = field.name
        if name and name not in form_data:
            if field.type in MODEL_FIELD_TYPES:
                form_data[name] = f"/{tag}/models/{name}.safetensors"
            elif field.type == LORA_PICKER_FIELD_TYPE:
                form_data[name] = []
            elif field.type in MEDIA_FIELD_TYPES and field.required:
                placeholder = f"/{tag}/media/{name}.{MEDIA_EXTENSIONS.get(field.type, 'bin')}"
                multi = bool((field.configuration or {}).get("multi"))
                form_data[name] = [placeholder] if multi else placeholder
            elif (field.type in TEXT_FIELD_TYPES or field.type == "tags") and (field.required or requires_by_reaction(field)):
                form_data[name] = f"{tag} placeholder text"
        if isinstance(field.children, list):
            inject_unresolvable_defaults(field.children, form_data, tag)


def default_form_fields(preset: PresetTemplate, mode: str) -> List[FieldTemplate]:
    mode_data = preset.modes[mode]
    form_name = default_form_name(mode_data)
    if form_name is None:
        return []
    return next(f for f in mode_data.forms if f.name == form_name).fields


def build_default_form_data(preset: PresetTemplate, mode: str, tag: str) -> Tuple[Dict[str, Any], List[FieldTemplate]]:
    fields = default_form_fields(preset, mode)
    form_data: Dict[str, Any] = {}
    collect_named_fields(fields, form_data)
    inject_unresolvable_defaults(fields, form_data, tag)
    return form_data, fields
