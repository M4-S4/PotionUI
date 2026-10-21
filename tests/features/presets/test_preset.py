import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.features.presets import PresetTemplateLoader, PresetProcessor
from src.features.presets.templates import PresetTemplate, PipeTemplate, FieldTemplate, FormTemplate, ModeTemplate
from src.pipelines.models import BaseModel
from src.platform.templating import TemplateProcessor
from src.platform.settings.settings import Settings


class TestPresetTemplateLoader(unittest.TestCase):
    
    def setUp(self):
        self.loader = PresetTemplateLoader("/path/to/presets")
    
    def test_init(self):
        self.assertEqual(self.loader.preset_files_path, Path("/path/to/presets"))
        self.assertEqual(self.loader.presets, [])
    
    @patch.object(PresetTemplateLoader, '_load_preset_file')
    @patch('src.features.presets.loader.Path')
    def test_load_presets(self, mock_path_class, mock_load_preset_file):
        preset = PresetTemplate(
            id='preset1',
            name='Test Preset',
            version='1.0',
            modes={},
            path='presets/test/test/v1/test',
            vars={},
            description='Test preset',
            tags=['test']
        )

        mock_load_preset_file.return_value = preset

        # Create mock path instances
        mock_base_path = MagicMock()
        mock_base_path.exists.return_value = True
        mock_base_path.rglob.return_value = [Path('/path/to/preset.yml')]

        # Make Path() return our mock when called with list items
        def path_side_effect(p):
            mock_p = MagicMock()
            mock_p.exists.return_value = True
            mock_p.rglob.return_value = [Path('/path/to/preset.yml')]
            return mock_p

        mock_path_class.side_effect = path_side_effect

        # Re-init loader to use mocked Path
        self.loader.preset_files_paths = [mock_base_path]

        self.loader.load_presets()

        # Presets are now stored as a list, not keyed by sha256
        self.assertIsInstance(self.loader.presets, list)
        self.assertEqual(len(self.loader.presets), 1)
        self.assertEqual(self.loader.presets[0], preset)
    
    def test_clear_cache(self):
        self.loader.presets = ['preset1', 'preset2']
        self.loader.clear_cache()
        self.assertEqual(self.loader.presets, [])
    
    def test_get_preset_by_name(self):
        preset1 = PresetTemplate(
            id='preset1',
            name='Preset 1',
            version='1.0',
            modes={},
            path='presets/test/test/v1/test',
            vars={},
            description='Test preset',
            tags=['test']
        )

        preset2 = PresetTemplate(
            id='preset2',
            name='Preset 2',
            version='1.0',
            modes={},
            path='presets/test/test/v1/test',
            vars={},
            description='Test preset',
            tags=['test']
        )

        self.loader.presets = [preset1, preset2]
        self.loader._loaded = True

        result = self.loader.get_preset_by_name('Preset 1')
        self.assertEqual(result, preset1)

        result = self.loader.get_preset_by_name('Preset 2')
        self.assertEqual(result, preset2)

        result = self.loader.get_preset_by_name('Nonexistent')
        self.assertIsNone(result)
    
    def test_load_preset_by_id(self):
        preset1 = PresetTemplate(
            id='preset_id_1',
            name='Preset 1',
            version='1.0',
            modes={},
            path='presets/test/test/v1/test',
            vars={},
            description='Test preset',
            tags=['test']
        )

        preset2 = PresetTemplate(
            id='preset_id_2',
            name='Preset 2',
            version='1.0',
            modes={},
            path='presets/test/test/v1/test',
            vars={},
            description='Test preset',
            tags=['test']
        )

        self.loader.presets = [preset1, preset2]
        self.loader.presets_by_id = {'preset_id_1': preset1, 'preset_id_2': preset2}
        self.loader._loaded = True

        result = self.loader.load_preset_by_id('preset_id_1')
        self.assertEqual(result, preset1)

        result = self.loader.load_preset_by_id('preset_id_2')
        self.assertEqual(result, preset2)

        result = self.loader.load_preset_by_id('nonexistent_id')
        self.assertIsNone(result)


class TestPresetProcessor(unittest.TestCase):
    
    def setUp(self):
        self.template_processor = Mock(spec=TemplateProcessor)
        self.settings = Mock(spec=Settings)
        self.preset_template_loader = Mock(spec=PresetTemplateLoader)

        self.processor = PresetProcessor(
            self.template_processor,
            self.settings,
            self.preset_template_loader
        )
    
    def test_process_value_string(self):
        # Test plain string
        result = self.processor.process_value("plain text", {})
        self.assertEqual(result, "plain text")
        
        # @object:/@dict: directives are deleted (spec §1/§6): exact `{{ expr }}`
        # scalars return native values directly via dot access instead. An
        # exact-expression scalar delegates to the template processor and
        # passes its native (non-stringified) value through untouched.
        context = {'form': {'field': 42}}
        self.template_processor.process_template.return_value = 42
        result = self.processor.process_value("{{ form.field }}", context)
        self.template_processor.process_template.assert_called_once_with("{{ form.field }}", context)
        self.assertEqual(result, 42)
        self.assertIsInstance(result, int)

        # Test template string
        self.template_processor.process_template.reset_mock()
        self.template_processor.process_template.return_value = "processed template"
        result = self.processor.process_value("{{variable}}", {})
        self.template_processor.process_template.assert_called_once_with("{{variable}}", {})
        self.assertEqual(result, "processed template")
    
    def test_process_value_dict(self):
        input_dict = {
            'key1': 'value1',
            'key2': '{{template}}',
            'key3': {'nested': 'value'}
        }
        context = {}
        
        self.template_processor.process_template.return_value = "processed"
        
        result = self.processor.process_value(input_dict, context)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['key1'], 'value1')
        self.assertEqual(result['key2'], 'processed')
        self.assertIsInstance(result['key3'], dict)
    
    def test_process_value_list(self):
        input_list = ['value1', '{{template}}', {'key': 'value'}]
        context = {}
        
        self.template_processor.process_template.return_value = "processed"
        
        result = self.processor.process_value(input_list, context)
        
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], 'value1')
        self.assertEqual(result[1], 'processed')
        self.assertIsInstance(result[2], dict)
    
    @patch.object(PresetProcessor, 'process_value')
    def test_process_new_format(self, mock_process_value):
        # Setup mock to return values as-is
        mock_process_value.side_effect = lambda value, context, *args, **kwargs: value

        # Setup settings
        self.settings.get_setting.side_effect = lambda key: 'test_key' if key in ['civitai_api_key', 'hf_api_key'] else None

        # Setup preset template loader
        self.preset_template_loader.preset_files_path = '/path/to/presets'
        
        # Create test preset with new format (ModeTemplate)
        form1 = FormTemplate(
            name='custom',
            fields=[
                FieldTemplate(
                    name='prompt',
                    type='text',
                    label='Prompt',
                    default=''
                )
            ]
        )
        
        form2 = FormTemplate(
            name='advanced',
            fields=[
                FieldTemplate(
                    name='steps',
                    type='number',
                    label='Steps',
                    default=20
                )
            ]
        )
        
        mode = ModeTemplate(
            forms=[form1, form2],
            pipes=[
                PipeTemplate(
                    name='generator',
                    enabled=True,
                    configuration={'sampler': 'euler'}
                )
            ]
        )
        
        preset_template = PresetTemplate(
            id='test_preset',
            name='Test Preset',
            version='1.0',
            modes={'txt2img': mode},
            path='presets/author/model/v1/preset',
            vars={},
            description='Test preset',
            tags=['test']
        )
        
        generation_data = {
            'prompt': 'test prompt',
            'negative_prompt': 'test negative',
            'seed': 12345,
            'num_images': 1,
            'width': 512,
            'height': 512,
            'mode': 'txt2img',
            'form_data': {'prompt': 'custom prompt', 'steps': 30},
            'generation_settings': {},
            'image': None,
            'mask': None,
            'numpy': None
        }
        
        result = self.processor.process(preset_template, generation_data)
        
        # Verify result structure
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        
        # Check pipe
        self.assertEqual(result[0]['name'], 'generator')
        self.assertEqual(result[0]['config'], {'sampler': 'euler'})
        
        # Verify context was built correctly: `form` is the bound form_data
        # dict directly (spec §2 deletes `input.*`/`input.forms` entirely).
        calls = mock_process_value.call_args_list
        found_form_context = False
        for call in calls:
            if len(call[0]) > 1:
                context = call[0][1]
                if 'form' in context:
                    self.assertEqual(context['form'], {'prompt': 'custom prompt', 'steps': 30})
                    found_form_context = True
                    break
        self.assertTrue(found_form_context, "expected at least one process_value call with a 'form' context")
    


class TestPresetTemplateMedia(unittest.TestCase):
    MEDIA = {
        "cover": "public/cover.png",
        "gallery": [{"src": "public/examples/a.png", "seed": 7, "mode": "txt2img"}],
    }

    def _template(self, media=None):
        return PresetTemplate(
            id="p1", name="P", version="1.0.0", path="/tmp/p", modes={}, media=media
        )

    def test_media_defaults_to_none(self):
        self.assertIsNone(self._template().media)

    def test_media_set_when_provided(self):
        self.assertEqual(self._template(self.MEDIA).media, self.MEDIA)


if __name__ == '__main__':
    unittest.main()