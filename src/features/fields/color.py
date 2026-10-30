from typing import Any, Dict, List, Optional

from src.platform.media_tools.params import TRANSPARENT

from .base_field import BaseField
from .specs import FieldConfigSpec, FieldExampleSpec, FieldValidationSpec
from .tool_values import coerce_field_value


class Color(BaseField):
    @classmethod
    def description(cls) -> str:
        return "Colour swatch with a hex box and an optional transparent choice"

    def output(self, field, preset_id: str = None) -> Dict[str, Any]:
        field_info = self.get_field_info(field)
        schema = self.create_base_schema(field_info)
        schema['allow_transparent'] = field_info['configuration'].get('allow_transparent') is True
        return schema

    def input(self, field_name: str, value: Any, validation_rules: Optional[Dict[str, Any]] = None) -> Any:
        allow = (validation_rules or {}).get('allow_transparent') is True
        return coerce_field_value({"type": "color", "allow_transparent": allow}, value)

    @classmethod
    def configuration(cls) -> List[FieldConfigSpec]:
        return [
            FieldConfigSpec(
                name="allow_transparent",
                param_type=bool,
                default=False,
                description=f"Offer a Transparent choice; the value is then `{TRANSPARENT}` instead of a hex colour",
                example=True
            ),
            FieldConfigSpec(
                name="tooltip",
                param_type=str,
                default=None,
                description="Help text shown in an info tooltip next to the label",
                example="Colour of the empty canvas"
            ),
        ]

    @classmethod
    def validation_rules(cls) -> List[FieldValidationSpec]:
        return [
            FieldValidationSpec(
                rule_name="required",
                description="Whether a colour must be chosen; the value is #RRGGBB",
                param_type=bool,
                example=True
            ),
        ]

    @classmethod
    def examples(cls) -> List[FieldExampleSpec]:
        return [
            FieldExampleSpec(
                title="Fill Colour",
                description="A colour that may also be left transparent",
                yaml_config="""type: color
name: fill
label: Fill
default: transparent
configuration:
  allow_transparent: true""",
                rendered_output={
                    "type": "color",
                    "name": "fill",
                    "title": "Fill",
                    "allow_transparent": True,
                    "default": "transparent"
                },
                frontend_preview={
                    "type": "color",
                    "name": "preview_fill",
                    "title": "Fill",
                    "allow_transparent": True,
                    "default": "#00b140"
                }
            ),
        ]
