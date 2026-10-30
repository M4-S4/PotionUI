from typing import Any, Dict, List, Optional

from .base_field import BaseField
from .specs import FieldConfigSpec, FieldExampleSpec, FieldValidationSpec
from .tool_values import coerce_field_value, normalize_options


class Segmented(BaseField):
    @classmethod
    def description(cls) -> str:
        return "Segmented control for choosing one of a few options shown side by side"

    def output(self, field, preset_id: str = None) -> Dict[str, Any]:
        field_info = self.get_field_info(field)
        schema = self.create_base_schema(field_info)
        config = field_info['configuration']

        schema['options'] = normalize_options(config.get('options'))
        if config.get('fit') is True:
            schema['fit'] = True
        return schema

    def input(self, field_name: str, value: Any, validation_rules: Optional[Dict[str, Any]] = None) -> Any:
        options = normalize_options((validation_rules or {}).get('options'))
        return coerce_field_value({"type": "select", "options": options}, value)

    @classmethod
    def configuration(cls) -> List[FieldConfigSpec]:
        return [
            FieldConfigSpec(
                name="options",
                param_type=list,
                default=[],
                description="The segments, each a plain value or a mapping with `value`, an optional `label` and an optional `disabled`",
                required=True,
                example=[
                    {"label": "Pad", "value": "pad"},
                    {"label": "Cover", "value": "cover"},
                ]
            ),
            FieldConfigSpec(
                name="fit",
                param_type=bool,
                default=False,
                description="Size the control to its content instead of stretching it across the row",
                example=True
            ),
            FieldConfigSpec(
                name="tooltip",
                param_type=str,
                default=None,
                description="Help text shown in an info tooltip next to the label",
                example="How a clip of another size meets the frame"
            ),
        ]

    @classmethod
    def validation_rules(cls) -> List[FieldValidationSpec]:
        return [
            FieldValidationSpec(
                rule_name="required",
                description="Whether a segment must be chosen",
                param_type=bool,
                example=True
            ),
        ]

    @classmethod
    def examples(cls) -> List[FieldExampleSpec]:
        return [
            FieldExampleSpec(
                title="Fit Mode",
                description="Two segments with a default",
                yaml_config="""type: segmented
name: fit
label: Fit
default: pad
configuration:
  options:
    - label: Pad
      value: pad
    - label: Cover
      value: cover""",
                rendered_output={
                    "type": "segmented",
                    "name": "fit",
                    "title": "Fit",
                    "options": [
                        {"label": "Pad", "value": "pad"},
                        {"label": "Cover", "value": "cover"}
                    ],
                    "default": "pad"
                },
                frontend_preview={
                    "type": "segmented",
                    "name": "preview_fit",
                    "title": "Fit",
                    "options": [
                        {"label": "Pad", "value": "pad"},
                        {"label": "Cover", "value": "cover"}
                    ],
                    "default": "pad"
                }
            ),
        ]
