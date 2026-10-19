"""Tests for FieldTypeRegistry (src/platform/plugins/field_types.py)."""

import unittest

from src.platform.plugins.field_types import (
    FieldTypeDefinition,
    FieldTypeRegistry,
)


class TestFieldTypeRegistry(unittest.TestCase):

    def setUp(self):
        self.registry = FieldTypeRegistry()

    def test_get_unknown_type_returns_default_never_none(self):
        definition = self.registry.get('does_not_exist')

        self.assertIsNotNone(definition)
        self.assertIsNone(definition.schema_cls)
        self.assertIsNone(definition.options_provider)

    def test_frontend_manifest_shape(self):
        self.registry.register(FieldTypeDefinition(
            type_name='select',
            options_provider=lambda config: [],
            frontend_component='core:SelectField',
            container=False,
            source='core',
        ))
        self.registry.register(FieldTypeDefinition(
            type_name='tabs',
            frontend_component='core:TabsField',
            container=True,
            source='core',
        ))

        manifest = self.registry.frontend_manifest()
        by_type = {entry['type']: entry for entry in manifest}

        self.assertEqual(by_type['select'], {
            'type': 'select',
            'component': 'core:SelectField',
            'has_options': True,
            'container': False,
            'source': 'core',
            'axis_editor': '',
            'configuration_schema': [],
        })
        self.assertEqual(by_type['tabs'], {
            'type': 'tabs',
            'component': 'core:TabsField',
            'has_options': False,
            'container': True,
            'source': 'core',
            'axis_editor': '',
            'configuration_schema': [],
        })

    def test_frontend_manifest_carries_a_plugin_axis_editor(self):
        self.registry.register(FieldTypeDefinition(
            type_name='palette',
            frontend_component='plugin:colors:PaletteField',
            source='colors',
            axis_editor='plugin:colors:PaletteAxis',
        ))

        entry = next(e for e in self.registry.frontend_manifest() if e['type'] == 'palette')

        self.assertEqual(entry['axis_editor'], 'plugin:colors:PaletteAxis')

    def test_frontend_manifest_configuration_schema_self_describes_a_real_field_type(self):
        """A registered type with a real `schema_cls` (not the bare
        placeholder definitions above) surfaces its `configuration()`
        `FieldConfigSpec`s so a frontend field component can discover its
        own knobs (e.g. MediaLoaderField's `accepted_types`) without the
        backend and frontend hardcoding the same key names twice."""
        from src.features.fields.media import Media

        self.registry.register(FieldTypeDefinition(
            type_name='media',
            schema_cls=Media,
            frontend_component='core:MediaLoaderField',
        ))

        manifest = self.registry.frontend_manifest()
        entry = next(e for e in manifest if e['type'] == 'media')

        names = {spec['name'] for spec in entry['configuration_schema']}
        self.assertIn('accepted_types', names)
        self.assertIn('max_resolution', names)
        self.assertIn('max_total_video_duration_seconds', names)

        accepted_types_spec = next(s for s in entry['configuration_schema'] if s['name'] == 'accepted_types')
        self.assertEqual(accepted_types_spec['param_type'], 'list')
        self.assertEqual(accepted_types_spec['example'], ['image', 'video'])

    def test_configuration_schema_param_type_is_a_readable_string_not_a_class_repr(self):
        """`FieldConfigSpec.param_type` is a Python `type` (e.g. `float`); the
        manifest must serialize it to its short name so a field catalogue page
        can render it directly, not as `<class 'float'>`."""
        from src.features.fields.slider import Slider

        self.registry.register(FieldTypeDefinition(
            type_name='slider',
            schema_cls=Slider,
            frontend_component='core:SliderField',
        ))

        manifest = self.registry.frontend_manifest()
        entry = next(e for e in manifest if e['type'] == 'slider')
        by_name = {spec['name']: spec for spec in entry['configuration_schema']}

        self.assertEqual(set(by_name), {'min', 'max', 'step', 'tooltip'})
        self.assertEqual(by_name['min']['param_type'], 'float')
        self.assertEqual(by_name['max']['param_type'], 'float')
        self.assertEqual(by_name['step']['param_type'], 'float')
        self.assertEqual(by_name['tooltip']['param_type'], 'str')
        for spec in by_name.values():
            self.assertNotIn('<class', spec['param_type'])


if __name__ == '__main__':
    unittest.main()
