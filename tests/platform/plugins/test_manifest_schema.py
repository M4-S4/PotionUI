"""Tests for the canonical plugin manifest schema (PluginManifestSchema)"""

import unittest

from pydantic import ValidationError

from src.platform.plugins.manifest import PluginCategory, PluginManifestSchema


class TestPluginManifestSchema(unittest.TestCase):
    """Validate PluginManifestSchema accepts the canonical shape and rejects legacy ones"""

    def _minimal(self, **overrides):
        data = {
            'id': 'test-plugin',
            'name': 'Test Plugin',
            'version': '1.0.0',
            'description': 'A test plugin',
            'author': 'Test Author',
            'type': 'backend-only',
        }
        data.update(overrides)
        return data

    def test_minimal_manifest_is_valid(self):
        schema = PluginManifestSchema.model_validate(self._minimal())
        self.assertEqual(schema.id, 'test-plugin')
        self.assertEqual(schema.hooks.backend, [])
        self.assertEqual(schema.dependencies.python, [])

    def test_missing_required_field_raises(self):
        data = self._minimal()
        del data['author']
        with self.assertRaises(ValidationError):
            PluginManifestSchema.model_validate(data)

    def test_unknown_top_level_key_raises(self):
        data = self._minimal(unexpected_key="boom")
        with self.assertRaises(ValidationError):
            PluginManifestSchema.model_validate(data)

    def test_canonical_hooks_format_accepted(self):
        data = self._minimal(hooks={
            'backend': [{'hook': 'generation.before_start', 'handler': 'hooks.mod.fn'}],
            'frontend': [{'hook': 'workbench.actions', 'component': 'Action.js', 'order': 5}],
        })
        schema = PluginManifestSchema.model_validate(data)
        self.assertEqual(len(schema.hooks.backend), 1)
        self.assertEqual(schema.hooks.backend[0].hook, 'generation.before_start')
        self.assertEqual(len(schema.hooks.frontend), 1)
        self.assertEqual(schema.hooks.frontend[0].order, 5)

    def test_legacy_flat_hooks_format_rejected(self):
        data = self._minimal(hooks={'generation.before_start': 'hooks.mod.fn'})
        with self.assertRaises(ValidationError):
            PluginManifestSchema.model_validate(data)

    def test_backend_hook_remote_flag_parses_true(self):
        data = self._minimal(hooks={
            'backend': [{'hook': 'prompt.transform', 'handler': 'hooks.mod.fn', 'remote': True}],
        })
        schema = PluginManifestSchema.model_validate(data)
        self.assertTrue(schema.hooks.backend[0].remote)

    def test_backend_hook_remote_flag_defaults_false(self):
        data = self._minimal(hooks={
            'backend': [{'hook': 'generation.before_start', 'handler': 'hooks.mod.fn'}],
        })
        schema = PluginManifestSchema.model_validate(data)
        self.assertFalse(schema.hooks.backend[0].remote)

    def test_backend_hook_remote_flag_rejects_non_bool(self):
        data = self._minimal(hooks={
            'backend': [{'hook': 'prompt.transform', 'handler': 'hooks.mod.fn', 'remote': 'yes-please'}],
        })
        with self.assertRaises(ValidationError):
            PluginManifestSchema.model_validate(data)

    def test_canonical_dependencies_format_accepted(self):
        data = self._minimal(dependencies={
            'python': ['numpy>=1.24.0'],
            'binaries': ['ffmpeg'],
        })
        schema = PluginManifestSchema.model_validate(data)
        self.assertEqual(schema.dependencies.python, ['numpy>=1.24.0'])
        self.assertEqual(schema.dependencies.binaries, ['ffmpeg'])

    def test_legacy_dependencies_list_format_rejected(self):
        data = self._minimal(dependencies=['numpy==1.24.0'])
        with self.assertRaises(ValidationError):
            PluginManifestSchema.model_validate(data)

    def test_legacy_dependencies_categorized_backend_frontend_rejected(self):
        data = self._minimal(dependencies={'backend': ['ffmpeg'], 'frontend': []})
        with self.assertRaises(ValidationError):
            PluginManifestSchema.model_validate(data)

    def test_inert_sections_accepted(self):
        data = self._minimal(
            field_types=[{'type': 'custom_field', 'component': 'CustomField.js'}],
            renderers=[{'kind': 'history.artifact', 'key': 'my_artifact', 'component': 'MyArtifact.svelte'}],
            contributions=[{'slot': 'admin.tabs', 'component': 'MyTab.svelte', 'label': 'My Tab'}],
            provides_hooks=['my_plugin.custom_event'],
        )
        schema = PluginManifestSchema.model_validate(data)
        self.assertEqual(schema.field_types[0].type, 'custom_field')
        self.assertEqual(schema.renderers[0].kind, 'history.artifact')
        self.assertEqual(schema.contributions[0].slot, 'admin.tabs')
        self.assertEqual(schema.provides_hooks, ['my_plugin.custom_event'])

    def test_documentation_category_metadata_is_optional_and_validated(self):
        schema = PluginManifestSchema.model_validate(self._minimal(docs=[{
            'title': 'Model Inference',
            'path': 'docs/model-inference.md',
            'audience': 'developer',
            'category': 'Presets / Models',
            'category_order': 10,
        }]))

        entry = schema.docs[0]
        self.assertEqual(entry.category, 'Presets / Models')
        self.assertEqual(entry.category_order, 10)

        uncategorized = PluginManifestSchema.model_validate(self._minimal(docs=[{
            'title': 'Usage',
            'path': 'docs/README.md',
        }])).docs[0]
        self.assertIsNone(uncategorized.category)
        self.assertIsNone(uncategorized.category_order)

        with self.assertRaises(ValidationError):
            PluginManifestSchema.model_validate(self._minimal(docs=[{
                'title': 'Usage',
                'path': 'docs/README.md',
                'unknown_category_field': True,
            }]))

    def test_provides_hooks_object_form_accepted(self):
        data = self._minimal(provides_hooks=[
            'my_plugin.simple_event',
            {
                'name': 'my_plugin.custom_event',
                'description': 'Fires on a custom condition',
                'payload': {'value': {'type': 'int', 'description': 'The value'}},
                'mutable': ['value'],
                'use_when': ['Adjust the value before it is used'],
                'example': 'hooks.backend: [{hook: my_plugin.custom_event, handler: mod.fn}]',
            },
        ])
        schema = PluginManifestSchema.model_validate(data)
        self.assertEqual(schema.provides_hooks[0], 'my_plugin.simple_event')
        entry = schema.provides_hooks[1]
        self.assertEqual(entry.name, 'my_plugin.custom_event')
        self.assertEqual(entry.payload, {'value': {'type': 'int', 'description': 'The value'}})
        self.assertEqual(entry.mutable, ['value'])

    def test_provides_hooks_object_form_rejects_unknown_key(self):
        data = self._minimal(provides_hooks=[
            {'name': 'my_plugin.custom_event', 'unexpected': 'boom'},
        ])
        with self.assertRaises(ValidationError):
            PluginManifestSchema.model_validate(data)

    def test_category_defaults_to_other_when_omitted(self):
        schema = PluginManifestSchema.model_validate(self._minimal())
        self.assertEqual(schema.category, PluginCategory.OTHER)
        self.assertEqual(schema.category.value, "other")

    def test_category_accepts_valid_value(self):
        schema = PluginManifestSchema.model_validate(self._minimal(category="backends"))
        self.assertEqual(schema.category, PluginCategory.BACKENDS)

    def test_pipes_and_pages_and_api(self):
        data = self._minimal(
            pipes=[{'path': 'backend/pipes/foo', 'register_as': 'foo'}],
            pages=[{'route': 'foo', 'component': 'FooPage.js', 'label': 'Foo'}],
            api={'module': 'backend/api.py'},
        )
        schema = PluginManifestSchema.model_validate(data)
        self.assertEqual(schema.pipes[0].path, 'backend/pipes/foo')
        self.assertEqual(schema.pages[0].route, 'foo')
        self.assertEqual(schema.api.module, 'backend/api.py')

    def test_chat_extensions_accepted(self):
        data = self._minimal(
            chat_modes=[{
                'id': 'dataset',
                'name': 'Dataset Mode',
                'system_prompt': 'You build datasets. {{TOOL_HINTS}}',
                'tools': ['list_models'],
                'default_route_prefixes': ['/plugins/dataset'],
                'llm_options': {'think': False},
            }],
            tools=[{'class': 'tools.mod:MyTool', 'modes': ['dataset']}],
            resources=[{'namespace': 'datasets', 'provider': 'res.mod:MyProvider'}],
        )
        schema = PluginManifestSchema.model_validate(data)
        self.assertEqual(schema.chat_modes[0].id, 'dataset')
        self.assertEqual(schema.chat_modes[0].llm_options, {'think': False})
        self.assertFalse(schema.chat_modes[0].admin_only)
        self.assertEqual(schema.tools[0].tool_class, 'tools.mod:MyTool')
        self.assertEqual(schema.tools[0].model_dump(by_alias=True)['class'], 'tools.mod:MyTool')
        self.assertIsNone(schema.resources[0].modes)
        self.assertEqual(schema.resources[0].namespace, 'datasets')

    def test_chat_mode_admin_only_accepted(self):
        data = self._minimal(chat_modes=[{
            'id': 'importer',
            'name': 'Importer Mode',
            'system_prompt': 'Admin-only workflow import assistant.',
            'admin_only': True,
        }])
        schema = PluginManifestSchema.model_validate(data)
        self.assertTrue(schema.chat_modes[0].admin_only)

    def test_chat_mode_requires_exactly_one_prompt_source(self):
        # neither prompt source
        data = self._minimal(chat_modes=[{'id': 'x', 'name': 'X'}])
        with self.assertRaises(ValidationError):
            PluginManifestSchema.model_validate(data)
        # both prompt sources
        data = self._minimal(chat_modes=[{
            'id': 'x', 'name': 'X',
            'system_prompt': 'inline', 'system_prompt_file': 'prompts/mode.md',
        }])
        with self.assertRaises(ValidationError):
            PluginManifestSchema.model_validate(data)

    def test_chat_mode_prompt_file_form_accepted(self):
        data = self._minimal(chat_modes=[{
            'id': 'x', 'name': 'X', 'system_prompt_file': 'prompts/mode.md',
        }])
        schema = PluginManifestSchema.model_validate(data)
        self.assertEqual(schema.chat_modes[0].system_prompt_file, 'prompts/mode.md')

    def test_chat_extensions_reject_unknown_keys(self):
        for section in (
            {'chat_modes': [{'id': 'x', 'name': 'X', 'system_prompt': 'p', 'boom': 1}]},
            {'tools': [{'class': 'm:C', 'boom': 1}]},
            {'resources': [{'namespace': 'n', 'provider': 'm:C', 'boom': 1}]},
        ):
            with self.assertRaises(ValidationError):
                PluginManifestSchema.model_validate(self._minimal(**section))

    def test_automation_templates_accepted(self):
        schema = PluginManifestSchema.model_validate(self._minimal(automation_templates=[{
            'id': 'clear-runtime-memory',
            'title': 'Clear runtime memory',
            'description': 'Release memory held by this runtime.',
            'category': 'system',
            'icon': 'trash',
            'tags': ['memory', 'gpu'],
            'path': 'automations/clear-runtime-memory.json',
        }]))

        template = schema.automation_templates[0]
        self.assertEqual(template.id, 'clear-runtime-memory')
        self.assertEqual(template.path, 'automations/clear-runtime-memory.json')
        self.assertEqual(template.tags, ['memory', 'gpu'])

    def test_automation_template_id_and_shape_are_strict(self):
        for template in (
            {'id': 'Uppercase', 'title': 'Bad', 'path': 'bad.json'},
            {'id': 'has spaces', 'title': 'Bad', 'path': 'bad.json'},
            {'id': 'valid', 'title': '', 'path': 'bad.json'},
            {'id': 'valid', 'title': 'Bad', 'path': ''},
            {'id': 'valid', 'title': 'Bad', 'path': 'bad.json', 'unexpected': True},
        ):
            with self.assertRaises(ValidationError):
                PluginManifestSchema.model_validate(
                    self._minimal(automation_templates=[template])
                )

    def test_prompt_importer_accepted(self):
        schema = PluginManifestSchema.model_validate(self._minimal(prompt_importers=[{
            'id': 'community',
            'label': 'Community import',
            'component': 'ImportModal.svelte',
            'backend': 'importers:CommunityImporter',
        }]))

        importer = schema.prompt_importers[0]
        self.assertEqual(importer.id, 'community')
        self.assertEqual(importer.backend, 'importers:CommunityImporter')

    def test_prompt_importer_requires_backend_and_component(self):
        for entry in (
            {'id': 'community', 'label': 'Community', 'backend': 'importers:X'},
            {'id': 'community', 'label': 'Community', 'component': 'Modal.svelte'},
            {'id': 'community', 'label': 'Community', 'component': 'Modal.svelte', 'backend': 'importers:X', 'unexpected': True},
        ):
            with self.assertRaises(ValidationError):
                PluginManifestSchema.model_validate(self._minimal(prompt_importers=[entry]))

    def test_phrasebook_op_accepted_with_optional_component(self):
        schema = PluginManifestSchema.model_validate(self._minimal(phrasebook_ops=[
            {'id': 'shout', 'label': 'Shout', 'component': 'ShoutModal.svelte', 'backend': 'ops:ShoutOperation'},
            {'id': 'quiet', 'label': 'Quiet', 'backend': 'ops:QuietOperation'},
        ]))

        shout, quiet = schema.phrasebook_ops
        self.assertEqual(shout.component, 'ShoutModal.svelte')
        self.assertEqual(shout.backend, 'ops:ShoutOperation')
        self.assertIsNone(quiet.component)

    def test_phrasebook_op_requires_backend_and_forbids_extras(self):
        for entry in (
            {'id': 'shout', 'label': 'Shout', 'component': 'Modal.svelte'},
            {'id': 'shout', 'label': 'Shout', 'backend': 'ops:X', 'unexpected': True},
        ):
            with self.assertRaises(ValidationError):
                PluginManifestSchema.model_validate(self._minimal(phrasebook_ops=[entry]))

    def _tool(self, **overrides):
        entry = {
            'id': 'test-plugin.loop',
            'label': 'Loop',
            'accepts': ['video'],
            'component': 'LoopPreview.svelte',
        }
        entry.update(overrides)
        return entry

    def _validate_tool(self, **overrides):
        return PluginManifestSchema.model_validate(self._minimal(media_tools=[self._tool(**overrides)]))

    def test_history_tools_key_is_gone(self):
        with self.assertRaises(ValidationError):
            PluginManifestSchema.model_validate(self._minimal(history_tools=[
                {'id': 'thing', 'label': 'Thing', 'category': 'analyze', 'component': 'Thing.svelte'},
            ]))

    def test_media_tool_component_only_entry_accepted(self):
        schema = self._validate_tool(selection={'min': 1, 'max': 12}, scopes=['history', 'library'])

        tool = schema.media_tools[0]
        self.assertEqual(tool.id, 'test-plugin.loop')
        self.assertEqual(tool.component, 'LoopPreview.svelte')
        self.assertIsNone(tool.python)
        self.assertEqual(tool.selection.min, 1)
        self.assertEqual(tool.selection.max, 12)
        self.assertEqual(tool.scopes, ['history', 'library'])

    def test_media_tool_defaults(self):
        tool = self._validate_tool().media_tools[0]

        self.assertEqual(tool.icon, 'extension')
        self.assertEqual(tool.description, '')
        self.assertEqual(tool.produces, 'same')
        self.assertEqual(tool.cardinality, 'one_to_one')
        self.assertEqual(tool.scopes, ['library', 'history', 'field'])
        self.assertFalse(tool.preview)
        self.assertEqual(tool.params, [])
        self.assertIsNone(tool.selection)

    def test_media_tool_python_entry_with_params_accepted(self):
        schema = self._validate_tool(
            python='tools/loop.py:LoopTool',
            component=None,
            preview=True,
            params=[
                {'name': 'style', 'type': 'select', 'options': ['forward', 'pingpong'], 'default': 'pingpong'},
                {'name': 'repeats', 'type': 'integer', 'min': 1, 'max': 8, 'default': 2},
            ],
        )

        tool = schema.media_tools[0]
        self.assertEqual(tool.python, 'tools/loop.py:LoopTool')
        self.assertTrue(tool.preview)
        self.assertEqual([p.name for p in tool.params], ['style', 'repeats'])

    def test_media_tool_needs_python_or_component(self):
        with self.assertRaises(ValidationError):
            self._validate_tool(component=None)

    def test_media_tool_id_must_carry_the_plugin_prefix(self):
        for bad_id in ('loop', 'other-plugin.loop', 'test-plugin.'):
            with self.assertRaises(ValidationError):
                self._validate_tool(id=bad_id)

    def test_media_tool_ids_must_be_unique(self):
        with self.assertRaises(ValidationError):
            PluginManifestSchema.model_validate(self._minimal(media_tools=[self._tool(), self._tool()]))

    def test_media_tool_python_path_must_stay_inside_the_plugin(self):
        for bad in ('../escape.py:Tool', 'tools/../escape.py:Tool', 'tools/loop.py', 'tools/loop:Tool'):
            with self.assertRaises(ValidationError):
                self._validate_tool(python=bad)

    def test_media_tool_rejects_unknown_kind_scope_and_cardinality(self):
        for override in (
            {'accepts': ['mesh']},
            {'accepts': []},
            {'scopes': ['sidebar']},
            {'scopes': []},
            {'cardinality': 'many_to_many'},
            {'produces': 'mesh'},
        ):
            with self.assertRaises(ValidationError):
                self._validate_tool(**override)

    def test_media_tool_selection_bounds_validated(self):
        for selection in ({'min': 0}, {'min': 5, 'max': 2}):
            with self.assertRaises(ValidationError):
                self._validate_tool(selection=selection)

    def test_media_tool_preview_needs_python(self):
        with self.assertRaises(ValidationError):
            self._validate_tool(preview=True)

    def test_media_tool_params_validated(self):
        for params in (
            [{'name': 'mode', 'type': 'select'}],
            [{'name': 'n', 'type': 'integer', 'min': 5, 'max': 1}],
            [{'name': 'n', 'type': 'vector'}],
            [{'name': 'n', 'type': 'integer'}, {'name': 'n', 'type': 'float'}],
            [{'name': 'Bad Name', 'type': 'string'}],
        ):
            with self.assertRaises(ValidationError):
                self._validate_tool(python='tools/loop.py:LoopTool', params=params)

    def test_media_tool_manifest_dump_keeps_only_declared_param_keys(self):
        schema = self._validate_tool(
            python='tools/loop.py:LoopTool',
            params=[
                {'name': 'end', 'type': 'float', 'label': 'End'},
                {'name': 'width', 'type': 'integer', 'default': None},
            ],
            requires={'has_audio': True},
        )

        dumped = schema.media_tools[0].model_dump()

        self.assertEqual(dumped['params'][0], {'name': 'end', 'type': 'float', 'label': 'End'})
        self.assertEqual(dumped['params'][1], {'name': 'width', 'type': 'integer', 'default': None})
        self.assertEqual(dumped['requires'], {'has_audio': True})

    def test_a1111_style_component_tool_is_accepted(self):
        schema = PluginManifestSchema.model_validate(self._minimal(
            id='a1111-metadata-export',
            media_tools=[{
                'id': 'a1111-metadata-export.export-zip',
                'label': 'Download with A1111 metadata',
                'icon': 'download',
                'accepts': ['image'],
                'selection': {'min': 1},
                'scopes': ['history'],
                'component': 'A1111ExportTool.js',
            }],
        ))

        tool = schema.media_tools[0]
        self.assertEqual(tool.component, 'A1111ExportTool.js')
        self.assertEqual(tool.scopes, ['history'])
        self.assertEqual(tool.selection.min, 1)

if __name__ == '__main__':
    unittest.main()


class TestDefaultEnabledFlag(unittest.TestCase):
    _base = {
        'id': 'test-plugin',
        'name': 'Test Plugin',
        'version': '1.0.0',
        'description': 'A test plugin',
        'author': 'Test Author',
        'type': 'backend-only',
    }

    def test_defaults_to_false(self):
        self.assertFalse(PluginManifestSchema.model_validate(self._base).default_enabled)

    def test_true_is_parsed(self):
        schema = PluginManifestSchema.model_validate({**self._base, 'default_enabled': True})
        self.assertTrue(schema.default_enabled)

    def test_non_boolean_is_rejected(self):
        with self.assertRaises(ValidationError):
            PluginManifestSchema.model_validate({**self._base, 'default_enabled': 'sometimes'})
