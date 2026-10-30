from typing import Any, Dict, List, Optional

from src.platform.media_tools.params import ANCHORS

from .base_field import BaseField
from .specs import FieldConfigSpec, FieldExampleSpec, FieldValidationSpec
from .tool_values import coerce_field_value

DEFAULT_ANCHOR = "center"


class Anchor(BaseField):
    @classmethod
    def description(cls) -> str:
        return "Three by three picker for a position inside a frame"

    def output(self, field, preset_id: str = None) -> Dict[str, Any]:
        field_info = self.get_field_info(field)
        schema = self.create_base_schema(field_info)
        schema.setdefault('default', DEFAULT_ANCHOR)
        return schema

    def input(self, field_name: str, value: Any, validation_rules: Optional[Dict[str, Any]] = None) -> Any:
        return coerce_field_value({"type": "anchor"}, value)

    @classmethod
    def configuration(cls) -> List[FieldConfigSpec]:
        return [
            FieldConfigSpec(
                name="tooltip",
                param_type=str,
                default=None,
                description="Help text shown in an info tooltip next to the label",
                example="Where the image sits on the canvas"
            ),
        ]

    @classmethod
    def validation_rules(cls) -> List[FieldValidationSpec]:
        return [
            FieldValidationSpec(
                rule_name="required",
                description=f"Whether a position must be chosen; the value is one of {', '.join(ANCHORS)}",
                param_type=bool,
                example=True
            ),
        ]

    @classmethod
    def examples(cls) -> List[FieldExampleSpec]:
        return [
            FieldExampleSpec(
                title="Image Position",
                description="A position picker that starts in the middle",
                yaml_config="""type: anchor
name: anchor
label: Anchor
default: center""",
                rendered_output={
                    "type": "anchor",
                    "name": "anchor",
                    "title": "Anchor",
                    "default": "center"
                },
                frontend_preview={
                    "type": "anchor",
                    "name": "preview_anchor",
                    "title": "Anchor",
                    "default": "top_left"
                }
            ),
        ]
