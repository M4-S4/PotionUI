import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_PROBE = r"""
import json
from src.bootstrap.app import run_migrations_sync
from src.bootstrap.container import build_container
from src.features.llm.harness.adapter import Adapter, RenderRequest
from src.features.llm.harness.messages import Message, Role, TextPart
from src.platform.database.database import db

run_migrations_sync()
rows = [
    ("hosted", "openai", "https://api.openai.com/v1"),
    ("local", "openai", "http://localhost:1234/v1"),
    ("router", "openai", "https://openrouter.ai/api/v1"),
    ("ollama", "ollama", "http://localhost:11434"),
]
with db.get_cursor() as cursor:
    for config_id, config_type, base_url in rows:
        cursor.execute(
            "INSERT INTO llm_configurations (id, name, type, base_url, model, system_message) "
            "VALUES (?, ?, ?, ?, 'model-x', 'system')",
            (config_id, config_id, config_type, base_url),
        )
    cursor.execute("DELETE FROM applied_migrations WHERE migration_name = '070_llm_openai_compatible'")
run_migrations_sync()

container = build_container()
gateway = container.llm_service
report = {}
for config_id, _, _ in rows:
    config = gateway.repository.get_configuration(config_id)
    adapter = gateway.types.require(config.type).adapter
    entry = {"type": config.type}
    if config.type.startswith("openai"):
        request = RenderRequest(
            config=config,
            system="system",
            messages=(Message(id="u1", role=Role.USER, parts=(TextPart("hi"),)),),
        )
        entry.update(
            adapter=adapter.name,
            is_adapter=isinstance(adapter, Adapter),
            url=adapter.render(request, gateway.profiles.resolve(config)).url,
        )
    report[config_id] = entry
described = {entry["type"]: entry for entry in gateway.provider_types()}
report["described"] = {
    key: {"adapter": described[key]["adapter"], "default_base_url": described[key]["default_base_url"]}
    for key in ("openai", "openai-compatible")
}
print("TYPES_REPORT=" + json.dumps(report))
"""


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("llm_types")
    paths = [str(ROOT)]
    venv = ROOT / "venv" / "lib" / "python3.12" / "site-packages"
    if venv.is_dir():
        paths.insert(0, str(venv))
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    env = {
        "POTIONUI_DB_PATH": str(workdir / "db.sqlite"),
        "PYTHONPATH": os.pathsep.join(paths),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(workdir),
    }
    for name in ("SYSTEMROOT", "TEMP", "TMP", "PATHEXT", "COMSPEC"):
        if name in os.environ:
            env[name] = os.environ[name]
    result = subprocess.run(
        [sys.executable, "-c", _PROBE], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    lines = [line for line in result.stdout.splitlines() if line.startswith("TYPES_REPORT=")]
    assert len(lines) == 1, result.stdout[-4000:]
    return json.loads(lines[0].split("=", 1)[1])


def test_a_config_on_the_hosted_api_resolves_to_the_responses_adapter(report):
    assert report["hosted"] == {
        "type": "openai",
        "adapter": "openai-responses",
        "is_adapter": True,
        "url": "https://api.openai.com/v1/responses",
    }


@pytest.mark.parametrize("config_id, url", [
    ("local", "http://localhost:1234/v1/chat/completions"),
    ("router", "https://openrouter.ai/api/v1/chat/completions"),
])
def test_a_migrated_config_resolves_to_the_chat_adapter(report, config_id, url):
    assert report[config_id] == {"type": "openai-compatible", "adapter": "openai-chat", "is_adapter": True, "url": url}


def test_other_types_are_left_alone_by_the_migration(report):
    assert report["ollama"] == {"type": "ollama"}


def test_both_types_are_offered_to_the_admin_ui(report):
    assert report["described"] == {
        "openai": {"adapter": "openai-responses", "default_base_url": "https://api.openai.com/v1"},
        "openai-compatible": {"adapter": "openai-chat", "default_base_url": None},
    }
