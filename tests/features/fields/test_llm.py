import unittest
from unittest.mock import Mock, patch
from src.features.fields.llm import LLMField


class TestLLMField(unittest.TestCase):

    def setUp(self):
        self.preset_loader = Mock()
        self.field = LLMField(self.preset_loader)

    def test_output_schema_full_field(self):
        """Test output schema generation with all components enabled"""
        field_config = {
            'type': 'llm',
            'name': 'llm_field',
            'label': 'LLM Configuration',
            'configuration': {
                'show_llm_select': True,
                'show_prompt': True
            }
        }

        schema = self.field.output(field_config)

        self.assertEqual(schema['type'], 'llm')
        self.assertEqual(schema['name'], 'llm_field')
        self.assertEqual(schema['title'], 'LLM Configuration')
        self.assertIn('llm_options', schema)
        # LLM options are populated client-side from the API.
        self.assertEqual(schema['llm_options'], [])
        self.assertEqual(schema['default_llm_id'], '')
        self.assertEqual(schema['configuration']['show_llm_select'], True)
        self.assertEqual(schema['configuration']['show_prompt'], True)

    def test_output_schema_llm_only(self):
        """Test output schema with only LLM select enabled"""
        field_config = {
            'type': 'llm',
            'name': 'llm_select',
            'label': 'Select LLM',
            'configuration': {
                'show_llm_select': True,
                'show_prompt': False
            }
        }

        schema = self.field.output(field_config)

        self.assertIn('llm_options', schema)
        self.assertEqual(schema['configuration']['show_llm_select'], True)
        self.assertEqual(schema['configuration']['show_prompt'], False)

    def test_output_default_llm_id_none_when_not_allow_empty(self):
        """default_llm_id is None when empty selection is disallowed"""
        field_config = {
            'type': 'llm',
            'name': 'llm_field',
            'configuration': {'show_llm_select': True, 'allow_empty': False}
        }

        schema = self.field.output(field_config)

        self.assertIsNone(schema['default_llm_id'])


    def test_output_with_tooltip(self):
        """Test that tooltip is included in schema when provided"""
        field_config = {
            'type': 'llm',
            'name': 'llm_field',
            'label': 'LLM Configuration',
            'configuration': {
                'show_llm_select': True,
                'tooltip': 'This is helpful information about the LLM field'
            }
        }

        schema = self.field.output(field_config)

        self.assertEqual(schema['tooltip'], 'This is helpful information about the LLM field')

    def test_output_without_tooltip(self):
        """Test that tooltip is not included in schema when not provided"""
        field_config = {
            'type': 'llm',
            'name': 'llm_field',
            'label': 'LLM Configuration',
            'configuration': {
                'show_llm_select': True
            }
        }

        schema = self.field.output(field_config)

        self.assertNotIn('tooltip', schema)


if __name__ == '__main__':
    unittest.main()
