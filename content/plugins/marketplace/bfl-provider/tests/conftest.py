from pathlib import Path

from src.plugin_api.cloud_testing import bind_plugin_backend

bind_plugin_backend(Path(__file__).resolve().parents[1])

import pytest

from .fixtures import bfl
from .support import build


@pytest.fixture
async def provider(bfl):
    made, http = build(bfl)
    try:
        yield made
    finally:
        await made.close()
        await http.close()
