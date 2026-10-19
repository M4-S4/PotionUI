from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.pipelines.contracts import PipeInput
from src.pipelines.pipes.model_tools.main import ModelToolsPipe
from src.platform.filesystem.model_roots import ModelRoot, TypeDir
from tests.fixtures.model_index_fixtures import make_resolver
from tests.platform.runtime.native.ops.model_tools.lora_fixtures import (
    CFG,
    DIM,
    build_moss,
    lora_pair,
    save_lora,
    save_model,
)

DIRECTORIES = {"diffusion_model": "diffusion_models", "lora": "loras", "text_encoder": "text_encoders"}


@dataclass
class Depot:
    base: Path
    resolver: object
    dirs: dict[str, Path]

    @property
    def models(self) -> Path:
        return self.dirs["diffusion_model"]

    @property
    def loras(self) -> Path:
        return self.dirs["lora"]


def make_depot(tmp_path: Path, *, read_only: bool = False) -> Depot:
    base = tmp_path / "depot"
    root = ModelRoot(
        id="r1", label="r1", path=base, kind="library", read_only=read_only,
        case_insensitive=False, state="online", state_reason=None, raw_path=str(base),
    )
    dirs: dict[str, Path] = {}
    bindings = []
    for position, (model_type, folder) in enumerate(DIRECTORIES.items()):
        path = base / folder
        path.mkdir(parents=True)
        dirs[model_type] = path
        bindings.append(
            TypeDir(
                root_id="r1", model_type=model_type, path=path, position=position,
                is_write=True, subdir=folder, binding_id=f"b-{model_type}",
            )
        )
    return Depot(base=base, resolver=make_resolver([root], bindings), dirs=dirs)


@pytest.fixture
def depot(tmp_path):
    return make_depot(tmp_path)


@pytest.fixture
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def moss(depot):
    sd = {k: v.detach().clone() for k, v in build_moss().state_dict().items()}
    path = save_model(depot.models / "moss-tiny.safetensors", sd, metadata={"license": "apache-2.0"})
    return sd, Path(path)


def lora_state(seed: int = 1):
    sd = {}
    sd.update(lora_pair("diffusion_model.layers.0.proj.o", DIM, DIM, 3, seed, alpha=6))
    sd.update(lora_pair("diffusion_model.layers.1.proj.ffn_down", DIM, CFG.ffn_dim, 2, seed + 1, alpha=2))
    return sd


@pytest.fixture
def lora_files(depot):
    first = save_lora(depot.loras / "style-a.safetensors", lora_state(1))
    second = save_lora(depot.loras / "style-b.safetensors", lora_state(5))
    return Path(first), Path(second)


def run_pipe(depot, config, *, models=True, is_admin=True, cancelled=None):
    outputs: list = []
    merged = {**ModelToolsPipe.get_default_config(), "is_admin": is_admin, **config}
    pipe = ModelToolsPipe(merged)
    service = SimpleNamespace(roots=depot.resolver) if models is True else models
    pipe.process(PipeInput(input={"MODELS": service}), outputs.append, is_cancelled=cancelled)
    return outputs


def model_files(outputs):
    from src.pipelines.outputs import ModelFileGenerationOutput

    return [o for o in outputs if isinstance(o, ModelFileGenerationOutput)]


def rows(*paths_and_strengths):
    return [{"model": str(p), "strength": s} for p, s in paths_and_strengths]


@pytest.fixture
def wrap_verify(monkeypatch):
    import src.pipelines.pipes.model_tools.runner as runner

    real_publish = runner.publish

    def install(wrapper):
        def publish(job, *args, **kwargs):
            job.verify = wrapper(job.verify)
            return real_publish(job, *args, **kwargs)

        monkeypatch.setattr(runner, "publish", publish)

    return install


def failing_verify(_inner):
    def verify(_staged):
        from src.pipelines.pipes.model_tools.errors import ModelToolsRefused

        raise ModelToolsRefused("Verification failed: forced")

    return verify
