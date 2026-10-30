from __future__ import annotations

import pytest

from src.platform.media_tools import ToolContext
from tests.platform.media_tools.media_factory import ffmpeg_available


@pytest.fixture
def need_ffmpeg():
    if not ffmpeg_available():
        pytest.skip("ffmpeg is not available")


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(tmp_path / "scratch")
