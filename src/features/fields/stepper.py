from typing import List

from .slider import Slider
from .specs import FieldExampleSpec


class Stepper(Slider):
    """Compact -/+ stepper for small-range numeric fields (e.g. batch count).

    Numeric semantics and config keys (`min`/`max`/`step`) match `Slider` -
    only the frontend widget differs: a bordered [-][value][+] control
    instead of a range track.
    """

    WIDGET = 'stepper'
    MIN_EXAMPLE = 1
    MAX_EXAMPLE = 10
    STEP_EXAMPLE = 1
    TOOLTIP_EXAMPLE = "Number of images to generate"

    @classmethod
    def examples(cls) -> List[FieldExampleSpec]:
        return [
            FieldExampleSpec(
                title="Compact Stepper",
                description="Small integer range rendered as a [-][value][+] control",
                yaml_config="""type: stepper
name: quantity
label: Images
configuration:
  min: 1
  max: 10
  step: 1
default: 1""",
                rendered_output={
                    "type": "stepper",
                    "name": "quantity",
                    "title": "Images",
                    "minimum": 1,
                    "maximum": 10,
                    "step": 1,
                    "default": 1
                },
                frontend_preview={
                    "type": "stepper",
                    "name": "preview_quantity",
                    "title": "Images",
                    "minimum": 1,
                    "maximum": 10,
                    "step": 1,
                    "default": 1
                }
            ),
        ]
