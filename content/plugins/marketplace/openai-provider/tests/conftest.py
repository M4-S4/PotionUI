from pathlib import Path

from src.plugin_api.cloud_testing import bind_plugin_backend

bind_plugin_backend(Path(__file__).resolve().parents[1])

from .fixtures import openai_api
