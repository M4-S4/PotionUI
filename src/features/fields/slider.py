from typing import Dict, Any, List

from .base_field import BaseField
from .specs import FieldConfigSpec, FieldValidationSpec, FieldExampleSpec


class Slider(BaseField):
    """Slider field for numeric input with min/max ranges"""

    WIDGET = 'slider'
    MIN_EXAMPLE = 0
    MAX_EXAMPLE = 100
    STEP_EXAMPLE = 0.1
    TOOLTIP_EXAMPLE = "Adjust the strength of the effect"

    def output(self, field, preset_id: str = None) -> Dict[str, Any]:
        """Transform slider field data to frontend format"""
        field_info = self.get_field_info(field)
        schema = self.create_base_schema(field_info)

        config = field_info['configuration']
        schema['minimum'] = config.get('min', 0)
        schema['maximum'] = config.get('max', 100)
        schema['step'] = config.get('step', 1)

        # Pass through tooltip if present in configuration
        if config.get('tooltip'):
            schema['tooltip'] = config['tooltip']

        # Also preserve the configuration object with tooltip for frontend access
        schema['configuration'] = config

        return schema

    @classmethod
    def configuration(cls) -> List[FieldConfigSpec]:
        """Return specification of configuration parameters this field accepts"""
        return [
            FieldConfigSpec(
                name="min",
                param_type=float,
                default=0,
                description=f"Minimum value for the {cls.WIDGET}",
                example=cls.MIN_EXAMPLE
            ),
            FieldConfigSpec(
                name="max",
                param_type=float,
                default=100,
                description=f"Maximum value for the {cls.WIDGET}",
                example=cls.MAX_EXAMPLE
            ),
            FieldConfigSpec(
                name="step",
                param_type=float,
                default=1,
                description=f"Step increment for the {cls.WIDGET}",
                example=cls.STEP_EXAMPLE
            ),
            FieldConfigSpec(
                name="tooltip",
                param_type=str,
                default="",
                description=f"Tooltip text displayed next to the {cls.WIDGET} value",
                example=cls.TOOLTIP_EXAMPLE
            ),
        ]

    @classmethod
    def validation_rules(cls) -> List[FieldValidationSpec]:
        """Return specification of validation rules this field supports"""
        base_rules = super().validation_rules()
        slider_rules = [
            FieldValidationSpec(
                rule_name="min",
                description="Minimum allowed value",
                param_type=float,
                example=cls.MIN_EXAMPLE
            ),
            FieldValidationSpec(
                rule_name="max",
                description="Maximum allowed value",
                param_type=float,
                example=cls.MAX_EXAMPLE
            ),
        ]
        return base_rules + slider_rules

    @classmethod
    def examples(cls) -> List[FieldExampleSpec]:
        """Return example configurations for this field"""
        return [
            FieldExampleSpec(
                title="Basic Slider",
                description="Simple numeric slider with range 0-100",
                yaml_config="""type: slider
name: strength
label: Strength
configuration:
  min: 0
  max: 100
  step: 1
default: 50""",
                rendered_output={
                    "type": "slider",
                    "name": "strength",
                    "title": "Strength",
                    "minimum": 0,
                    "maximum": 100,
                    "step": 1,
                    "default": 50
                },
                frontend_preview={
                    "type": "slider",
                    "name": "preview_strength",
                    "title": "Strength",
                    "minimum": 0,
                    "maximum": 100,
                    "step": 1,
                    "default": 50
                }
            ),
            FieldExampleSpec(
                title="Decimal Slider",
                description="Slider with decimal values for fine control",
                yaml_config="""type: slider
name: cfg_scale
label: CFG Scale
configuration:
  min: 1.0
  max: 20.0
  step: 0.5
default: 7.5""",
                rendered_output={
                    "type": "slider",
                    "name": "cfg_scale",
                    "title": "CFG Scale",
                    "minimum": 1.0,
                    "maximum": 20.0,
                    "step": 0.5,
                    "default": 7.5
                },
                frontend_preview={
                    "type": "slider",
                    "name": "preview_cfg_scale",
                    "title": "CFG Scale",
                    "minimum": 1.0,
                    "maximum": 20.0,
                    "step": 0.5,
                    "default": 7.5
                }
            ),
        ]