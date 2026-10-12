from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.features.backends.backend_registry import BackendRegistry
from src.features.presets import routes as routes_module
from src.features.presets.dto import PresetInfo
from src.features.presets.routes import PresetController


def _registry(*backends):
    registry = object.__new__(BackendRegistry)
    registry.get_available_backends = lambda: list(backends)
    return registry


def _backend(engine, supports):
    return SimpleNamespace(engine=engine, supports_prompt_batch=supports)


class TestEngineSupport:
    def test_true_when_every_enabled_backend_of_the_engine_batches(self):
        registry = _registry(_backend('native', True), _backend('native', True), _backend('other', False))

        assert registry.engine_supports_prompt_batch('native') is True

    def test_false_when_any_backend_of_the_engine_cannot(self):
        registry = _registry(_backend('native', True), _backend('native', False))

        assert registry.engine_supports_prompt_batch('native') is False

    def test_false_when_no_backend_is_enabled_for_the_engine(self):
        assert _registry(_backend('native', True)).engine_supports_prompt_batch('comfyui') is False

    def test_false_without_an_engine(self):
        assert _registry(_backend('native', True)).engine_supports_prompt_batch(None) is False


class _Operations:
    def __init__(self, listed, detail):
        self._listed, self._detail = listed, detail

    def list_presets(self, collaborators, user, include_uninstalled=False):
        return [dict(entry) for entry in self._listed]

    def get_preset(self, collaborators, preset_id):
        return dict(self._detail)


@pytest.fixture
def controller(monkeypatch):
    listed = [{'id': 'a', 'engine': 'native'}, {'id': 'b', 'engine': 'comfyui'}]
    monkeypatch.setattr(routes_module, 'operations', _Operations(listed, {'id': 'a', 'engine': 'native'}))
    registry = Mock()
    registry.engine_supports_prompt_batch.side_effect = lambda engine: engine == 'native'
    return PresetController(Mock(), registry)


class TestPresetPayload:
    def test_the_dto_declares_the_field_off_by_default(self):
        assert PresetInfo(id='x', name='x', version='1').supports_prompt_batch is False

    @pytest.mark.asyncio
    async def test_the_list_carries_it_per_preset(self, controller):
        result = await controller.list_presets(Mock())

        assert {entry['id']: entry['supports_prompt_batch'] for entry in result.data} == {'a': True, 'b': False}

    @pytest.mark.asyncio
    async def test_the_detail_carries_it(self, controller):
        result = await controller.get_preset('a', Mock())

        assert result.data['supports_prompt_batch'] is True
