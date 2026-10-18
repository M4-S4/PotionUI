import sys

import pytest


@pytest.fixture(autouse=True)
def _restore_text_encoder_placement():
    """``NativeBackend.prepare_pipes`` configures the engine process-wide; put the
    default back so a backend test never leaks its mode or budget into another."""
    yield
    module = sys.modules.get("src.platform.runtime.native.memory.text_encoder")
    if module is not None:
        module.reset_text_encoder_placement()
