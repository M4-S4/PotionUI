import unittest
from unittest.mock import Mock

from src.features.fields.header import Header


class TestHeader(unittest.TestCase):

    def setUp(self):
        self.preset_loader = Mock()
        self.header_field = Header(self.preset_loader)

    def test_output_basic_header(self):
        """Test output transformation for basic header field"""
        field = {
            'type': 'header',
            'name': 'section_header',
            'label': 'Advanced Settings',
            'description': 'Configure advanced options'
        }

        result = self.header_field.output(field)

        self.assertEqual(result['type'], 'header')
        self.assertNotIn('name', result)  # Headers don't have names
        self.assertEqual(result['title'], 'Advanced Settings')
        self.assertEqual(result['description'], 'Configure advanced options')

    def test_output_minimal_header(self):
        """Test output transformation for minimal header field"""
        field = {
            'type': 'header',
            'label': 'Simple Header'
        }

        result = self.header_field.output(field)

        self.assertEqual(result['type'], 'header')
        self.assertNotIn('name', result)  # Headers don't have names
        self.assertEqual(result['title'], 'Simple Header')
        self.assertEqual(result['description'], '')
        self.assertNotIn('default', result)

    def test_output_with_variant_configuration(self):
        """Test output transformation with variant configuration"""
        field = {
            'type': 'header',
            'label': 'Large Header',
            'configuration': {
                'variant': 'h2'
            }
        }

        result = self.header_field.output(field)

        self.assertEqual(result['type'], 'header')
        self.assertNotIn('name', result)
        self.assertEqual(result['title'], 'Large Header')
        # Configuration should be preserved in output
        # (checked through base_schema behavior)

    def test_output_with_style_configuration(self):
        """Test output transformation with custom style"""
        field = {
            'type': 'header',
            'label': 'Styled Header',
            'configuration': {
                'variant': 'h4',
                'style': {
                    'color': '#0066cc',
                    'fontWeight': 'bold',
                    'marginTop': '20px'
                }
            }
        }

        result = self.header_field.output(field)

        self.assertEqual(result['type'], 'header')
        self.assertNotIn('name', result)
        self.assertEqual(result['title'], 'Styled Header')

    def test_output_with_object_field(self):
        """Test output transformation with object-style field"""
        mock_field = Mock()
        mock_field.type = 'header'
        mock_field.name = 'object_header'
        mock_field.label = 'Object Header'
        mock_field.description = 'An object-style header'
        mock_field.default = None
        mock_field.required = False
        mock_field.configuration = {'variant': 'h3'}

        result = self.header_field.output(mock_field)

        self.assertEqual(result['type'], 'header')
        self.assertNotIn('name', result)  # Name should be removed for headers
        self.assertEqual(result['title'], 'Object Header')
        self.assertEqual(result['description'], 'An object-style header')


    def test_name_removal_from_output(self):
        """Test that name is always removed from output schema"""
        # Test with name explicitly set
        field_with_name = {
            'type': 'header',
            'name': 'should_be_removed',
            'label': 'Header With Name'
        }

        result = self.header_field.output(field_with_name)
        self.assertNotIn('name', result)

        # Test without name
        field_without_name = {
            'type': 'header',
            'label': 'Header Without Name'
        }

        result = self.header_field.output(field_without_name)
        self.assertNotIn('name', result)

    def test_edge_cases(self):
        """Test edge cases for header field"""
        # Empty label
        field = {
            'type': 'header',
            'label': ''
        }
        result = self.header_field.output(field)
        self.assertEqual(result['type'], 'header')
        self.assertEqual(result['title'], '')

        # No label at all (using name)
        field = {
            'type': 'header',
            'name': 'header_name'
        }
        result = self.header_field.output(field)
        self.assertEqual(result['type'], 'header')
        self.assertNotIn('name', result)  # Still removed even when used as title


if __name__ == '__main__':
    unittest.main()
