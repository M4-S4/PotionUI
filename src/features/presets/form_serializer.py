from typing import Dict, Any, Optional

from src.features.presets import PresetTemplateLoader
from src.platform.templating import TemplateProcessor
from src.features.fields.field_factory import FieldFactory

class PresetFormSerializer:
    """
    Serializer class for processing preset form configurations and converting them to JSON schema.
    This class handles the transformation of form fields, options, and other form-related data.
    """

    def __init__(self, preset_loader: PresetTemplateLoader, template_processor: TemplateProcessor = None):
        self.preset_loader = preset_loader
        self.template_processor = template_processor
        self.field_factory = FieldFactory(preset_loader, template_processor)

    def process_form_fields(
        self,
        form_config,
        preset_id: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process form configuration to create JSON schema.

        `overrides` (admin per-field form overrides for the relevant
        mode, `{field_name: {default?, editable?, visible?}}`) is applied
        to the loaded field tree before it is rendered, so hidden/locked
        fields never reach `_process_field_recursive`
        (which builds `properties`/`required`). See
        src/features/presets/form_overrides.py.
        """
        # If form_config is None or doesn't have fields, return empty schema
        if not form_config or not hasattr(form_config, 'fields'):
            return {
                'type': 'object',
                'properties': {},
                'required': []
            }

        # Get fields from form_config
        fields = form_config.fields

        if overrides:
            preset_template = self.preset_loader.load_preset_by_id(preset_id)
            if preset_template:
                from src.features.presets.form_overrides import apply_overrides_to_fields
                fields = apply_overrides_to_fields(fields, overrides, preset_template)

        # Create schema
        schema = {
            'type': 'object',
            'properties': {},
            'required': []
        }

        # Process all fields recursively
        for field in fields:
            self._process_field_recursive(field, schema, preset_id)

        return schema

    def _process_field_recursive(self, field, schema_obj: Dict[str, Any], preset_id: str = None):
        """Process a field and its children recursively"""
        # Handle both FieldTemplate objects and dictionaries
        if hasattr(field, 'type'):
            field_type = field.type
            field_name = field.name
            required = field.required
        else:
            field_type = field.get('type')
            field_name = field.get('name')
            required = field.get('required', False)

        # For tabs field, use 'tabs' as the name if it doesn't have one
        if field_type == 'tabs' and not field_name:
            field_name = 'tabs'

        # Process all fields, including container fields
        if field_name:
            # Process the field using the field factory
            schema_obj['properties'][field_name] = self.field_factory.map_field(field, preset_id)
            if required:
                schema_obj['required'].append(field_name)


    def get_json_schema_type(self, field_type: str) -> str:
        """Convert field type to JSON schema type"""
        type_mapping = {
            'textbox': 'string',
            'select': 'string',
            'slider': 'number',
            'number': 'number',
            'stepper': 'number',
            'checkbox_group': 'array',
            'button': 'null',
        }
        return type_mapping.get(field_type, 'string')