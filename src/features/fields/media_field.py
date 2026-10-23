import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base_field import BaseField
from .media_input import process_media_input, echo_configured_constraints, MAX_MEDIA_LABEL_LENGTH
from .specs import FieldConfigSpec, FieldValidationSpec


class MediaField(BaseField):
    KIND = 'media'
    SUPPORTED_FORMATS: frozenset = frozenset()
    MIME_TYPES: Dict[str, str] = {}
    DEFAULT_MAX_SIZE = 10 * 1024 * 1024
    DEFAULT_MIME = 'application/octet-stream'
    ACCEPT_FALLBACK = '*/*'
    MAX_LABEL_LENGTH = MAX_MEDIA_LABEL_LENGTH
    HAS_DIMENSIONS = False
    HAS_DURATION = False
    CHECK_LEGACY_LIMITS = True
    MAX_ITEMS_EXAMPLE = 5
    FORMATS_EXAMPLE: List[str] = []
    VALIDATION_EXAMPLE: Dict[str, Any] = {}
    EXTRA_CONFIG: Tuple[FieldConfigSpec, ...] = ()

    @classmethod
    def default_formats(cls) -> List[str]:
        return sorted(cls.SUPPORTED_FORMATS)

    def input(self, field_name: str, value: Any, validation_rules: Optional[Dict[str, Any]] = None) -> Any:
        return process_media_input(
            field_name, value, validation_rules,
            validate_legacy=self._validate_legacy,
            process_legacy=self._process_legacy,
        )

    def output(self, field, preset_id: str = None) -> Dict[str, Any]:
        field_info = self.get_field_info(field)
        schema = self.create_base_schema(field_info)
        config = field_info['configuration']

        schema['accept'] = self._get_accept_string(config)
        schema['multiple'] = config.get('multi', False)

        max_items = config.get('max_items')
        if max_items is not None:
            schema['max_items'] = max_items

        if 'allow_inpaint' in config:
            schema.setdefault('configuration', {})['allow_inpaint'] = config['allow_inpaint']

        echo_configured_constraints(schema, config)

        validation = field_info.get('validation', {})
        if validation:
            schema['validation'] = self._validation_schema(validation)

        config_validation = config.get('validation', {})
        if config_validation and 'validation' not in schema:
            schema['validation'] = self._validation_schema(config_validation)

        return schema

    def _validation_schema(self, source: Dict[str, Any]) -> Dict[str, Any]:
        result = {'maxSize': source.get('max_size', self.DEFAULT_MAX_SIZE)}
        if self.HAS_DIMENSIONS:
            result['maxWidth'] = source.get('max_width')
            result['maxHeight'] = source.get('max_height')
        if self.HAS_DURATION:
            result['maxDuration'] = source.get('max_duration')
        result['formats'] = source.get('formats', self.default_formats())
        return result

    def _validate_legacy(self, item: Dict[str, Any], rules: Dict[str, Any]) -> List[str]:
        kind = self.KIND
        errors = []

        if not isinstance(item, dict):
            errors.append(f"Invalid {kind} data format")
            return errors

        if 'data' not in item:
            errors.append(f"Missing {kind} data")

        if 'type' not in item:
            errors.append(f"Missing {kind} type")

        max_size = rules.get('max_size', self.DEFAULT_MAX_SIZE)
        if 'size' in item and item['size'] > max_size:
            errors.append(f"{kind.capitalize()} size exceeds maximum allowed size of {max_size / (1024 * 1024):.1f}MB")

        if 'name' in item:
            ext = Path(item['name']).suffix.lower()
            allowed_formats = rules.get('formats', self.default_formats())
            if ext not in allowed_formats:
                errors.append(f"Unsupported {kind} format. Allowed: {', '.join(allowed_formats)}")

        if self.CHECK_LEGACY_LIMITS and self.HAS_DIMENSIONS and 'width' in item and 'height' in item:
            if 'max_width' in rules and item['width'] > rules['max_width']:
                errors.append(f"{kind.capitalize()} width exceeds maximum of {rules['max_width']}px")

            if 'max_height' in rules and item['height'] > rules['max_height']:
                errors.append(f"{kind.capitalize()} height exceeds maximum of {rules['max_height']}px")

        if self.CHECK_LEGACY_LIMITS and self.HAS_DURATION and 'duration' in item and 'max_duration' in rules:
            if item['duration'] > rules['max_duration']:
                errors.append(f"{kind.capitalize()} duration exceeds maximum of {rules['max_duration']} seconds")

        return errors

    def _process_legacy(self, item: Dict[str, Any]) -> Dict[str, Any]:
        base64_data = item['data']
        if base64_data.startswith('data:'):
            base64_data = base64_data.split(',', 1)[1]

        try:
            raw = base64.b64decode(base64_data)
        except Exception as e:
            raise ValueError(f"Invalid base64 {self.KIND} data: {str(e)}")

        return {
            'data': raw,
            'name': item.get('name', self.KIND),
            'type': item.get('type', self.DEFAULT_MIME),
            'size': len(raw),
            'original_data': item
        }

    def _get_accept_string(self, configuration: Dict[str, Any]) -> str:
        formats = configuration.get('formats', self.default_formats())
        mime_types = dict.fromkeys(self.MIME_TYPES[fmt] for fmt in formats if fmt in self.MIME_TYPES)
        return ','.join(mime_types) if mime_types else self.ACCEPT_FALLBACK

    @classmethod
    def configuration(cls) -> List[FieldConfigSpec]:
        kind = cls.KIND
        return [
            FieldConfigSpec(
                name="multi",
                param_type=bool,
                default=False,
                description=f"Allow multiple {kind} uploads",
                example=True
            ),
            FieldConfigSpec(
                name="max_items",
                param_type=int,
                default=None,
                description="Maximum number of items allowed when `multi` is enabled",
                example=cls.MAX_ITEMS_EXAMPLE
            ),
            FieldConfigSpec(
                name="formats",
                param_type=list,
                default=cls.default_formats(),
                description=f"Allowed {kind} file formats",
                example=cls.FORMATS_EXAMPLE
            ),
            FieldConfigSpec(
                name="validation",
                param_type=dict,
                default={},
                description=f"{kind.capitalize()} validation rules (legacy, prefer field-level validation)",
                example=cls.VALIDATION_EXAMPLE
            ),
            *cls.EXTRA_CONFIG,
        ]

    @classmethod
    def validation_rules(cls) -> List[FieldValidationSpec]:
        return super().validation_rules() + cls.media_rules()

    @classmethod
    def media_rules(cls) -> List[FieldValidationSpec]:
        kind = cls.KIND
        rules = [
            FieldValidationSpec(
                rule_name="max_size",
                description="Maximum file size in bytes",
                param_type=int,
                example=cls.DEFAULT_MAX_SIZE
            ),
        ]
        if cls.HAS_DURATION:
            rules.append(FieldValidationSpec(
                rule_name="max_duration",
                description=f"Maximum {kind} duration in seconds",
                param_type=int,
                example=300
            ))
        if cls.HAS_DIMENSIONS:
            rules.append(FieldValidationSpec(
                rule_name="max_width",
                description=f"Maximum {kind} width in pixels",
                param_type=int,
                example=1920
            ))
            rules.append(FieldValidationSpec(
                rule_name="max_height",
                description=f"Maximum {kind} height in pixels",
                param_type=int,
                example=1080
            ))
        rules.append(FieldValidationSpec(
            rule_name="formats",
            description="List of allowed file extensions",
            param_type=list,
            example=cls.FORMATS_EXAMPLE
        ))
        return rules
