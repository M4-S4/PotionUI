import ast
from pathlib import Path

import pytest

from src.pipelines.outputs import ProgressGenerationOutput
from src.pipelines.pipes.media_loader.main import MediaLoaderPipe

PIPES = Path(__file__).resolve().parents[3] / "src" / "pipelines" / "pipes"


def _progress_calls(pipe_dir):
    tree = ast.parse((PIPES / pipe_dir / "main.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "ProgressGenerationOutput":
            yield {kw.arg: kw.value for kw in node.keywords}


@pytest.mark.parametrize("pipe_dir", ["media_loader", "controlnet_preprocessor"])
def test_every_fraction_the_pipe_emits_is_text_only(pipe_dir):
    with_fraction = [kw for kw in _progress_calls(pipe_dir) if "progress" in kw]
    assert with_fraction
    for kw in with_fraction:
        assert isinstance(kw.get("bar"), ast.Constant) and kw["bar"].value is False


def test_media_loader_emits_bar_false_progress(tmp_path):
    from PIL import Image
    paths = []
    for name in ("a.png", "b.png"):
        path = tmp_path / name
        Image.new("RGB", (16, 16)).save(path, format="PNG")
        paths.append({"type": "image", "path": str(path)})
    cfg = MediaLoaderPipe.get_default_config()
    cfg.update(media=paths)
    emitted = []

    MediaLoaderPipe(config=cfg).process(None, emitted.append)

    progress = [o for o in emitted if isinstance(o, ProgressGenerationOutput) and o.progress is not None]
    assert [(o.progress.current, o.progress.max) for o in progress] == [(0, 100), (0, 100), (40, 100), (100, 100)]
    assert all(o.bar is False for o in progress)
