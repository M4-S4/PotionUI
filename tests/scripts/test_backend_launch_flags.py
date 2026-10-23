from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

LAUNCH_PATHS = ["api.py", "run.sh", "scripts/potionui_cli.py"]


@pytest.mark.parametrize("relative", LAUNCH_PATHS)
def test_backend_launch_has_no_request_cap(relative):
    text = (REPO_ROOT / relative).read_text(encoding="utf-8").lower()
    assert "limit_max_requests" not in text
    assert "limit-max-requests" not in text


@pytest.mark.parametrize("relative", LAUNCH_PATHS)
def test_backend_launch_keeps_concurrency_limit(relative):
    text = (REPO_ROOT / relative).read_text(encoding="utf-8")
    assert "limit_concurrency=1000" in text or "limit-concurrency" in text
