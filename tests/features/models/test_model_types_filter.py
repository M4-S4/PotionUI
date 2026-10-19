from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from safetensors.torch import save_file

from src.features.models import availability as av
from src.features.models.quant_filter import keep_unquantized
from src.features.models.records import Model
from src.features.models.repository import ModelRepository


def registry():
    reg = Mock()
    reg.get_backends_for_engine.return_value = [Mock(backend_id="b1")]
    reg.engine_extracts_diffusion_model_from_checkpoint.return_value = False
    return reg


def write_plain(path: Path) -> Path:
    save_file({"layers.0.weight": torch.zeros(4, 4, dtype=torch.bfloat16)}, str(path))
    return path


def write_fp8(path: Path) -> Path:
    save_file(
        {
            "layers.0.weight": torch.zeros(4, 4).to(torch.float8_e4m3fn),
            "layers.0.weight_scale": torch.ones(()),
        },
        str(path),
    )
    return path


def write_scaled_marker(path: Path) -> Path:
    save_file({"layers.0.weight": torch.zeros(4, 4, dtype=torch.bfloat16), "scaled_fp8": torch.zeros(2)}, str(path))
    return path


@pytest.fixture
def files(tmp_path):
    return {
        "plain": write_plain(tmp_path / "plain.safetensors"),
        "fp8": write_fp8(tmp_path / "fp8.safetensors"),
        "marker": write_scaled_marker(tmp_path / "marker.safetensors"),
        "gguf": (tmp_path / "model.gguf"),
        "broken": (tmp_path / "broken.safetensors"),
    }


def models_from(files):
    out = []
    for index, name in enumerate(["plain", "fp8", "marker", "gguf", "broken"]):
        out.append(Model(id=f"m{index}", filename=files[name].name, model_type="diffusion_model"))
    files["gguf"].write_bytes(b"GGUF")
    files["broken"].write_bytes(b"not a header")
    return out


class TestKeepUnquantized:
    def test_only_plain_safetensors_survive(self, files):
        models = models_from(files)
        by_id = {f"m{i}": files[n] for i, n in enumerate(["plain", "fp8", "marker", "gguf", "broken"])}
        kept = keep_unquantized(models, lambda model_id: by_id[model_id])
        assert [m.id for m in kept] == ["m0"]

    def test_a_model_without_a_path_is_dropped(self, files):
        models = models_from(files)[:1]

        def missing(_model_id):
            raise FileNotFoundError("gone")

        assert keep_unquantized(models, missing) == []
        assert keep_unquantized(models, lambda _id: None) == []

    def test_a_rewritten_file_is_rechecked(self, files):
        models = models_from(files)[:1]
        target = files["plain"]
        assert [m.id for m in keep_unquantized(models, lambda _id: target)] == ["m0"]
        write_fp8(target)
        assert keep_unquantized(models, lambda _id: target) == []


class TestModelsForEngine:
    def run(self, repo, **kwargs):
        with patch.object(av, "model_availability_repo") as availability:
            availability.any_indexed.return_value = False
            return av.models_for_engine("native", registry(), model_repository=repo, **kwargs)

    def repo(self, models):
        repo = Mock()
        repo.get_all.return_value = models
        return repo

    def test_model_types_reach_the_repository_when_no_single_type_is_given(self):
        repo = self.repo([])
        self.run(repo, model_types=["diffusion_model", "unet", "text_encoder"])
        kwargs = repo.get_all.call_args.kwargs
        assert kwargs["model_types"] == ["diffusion_model", "unet", "text_encoder"]
        assert kwargs["model_type"] is None

    def test_the_single_type_keeps_its_behaviour(self):
        repo = self.repo([])
        self.run(repo, model_type="lora")
        kwargs = repo.get_all.call_args.kwargs
        assert kwargs["model_type"] == "lora"
        assert "model_types" not in kwargs

    def test_a_single_type_wins_over_the_list(self):
        repo = self.repo([])
        self.run(repo, model_type="lora", model_types=["unet", "vae"])
        assert "model_types" not in repo.get_all.call_args.kwargs

    def test_unquantized_only_filters_then_pages(self, files):
        models = models_from(files)
        plain_two = write_plain(files["plain"].parent / "plain2.safetensors")
        models.append(Model(id="m9", filename=plain_two.name, model_type="diffusion_model"))
        by_id = {"m0": files["plain"], "m1": files["fp8"], "m2": files["marker"], "m3": files["gguf"], "m4": files["broken"], "m9": plain_two}
        repo = self.repo(models)
        result = self.run(
            repo, model_types=["diffusion_model"], unquantized_only=True, path_for_model=lambda i: by_id[i],
            limit=1, offset=1,
        )
        assert [r["id"] for r in result] == ["m9"]
        kwargs = repo.get_all.call_args.kwargs
        assert "limit" not in kwargs and "offset" not in kwargs

    def test_unquantized_only_without_a_path_resolver_fails_closed(self):
        repo = self.repo([Model(id="m0", filename="x.safetensors", model_type="diffusion_model")])
        assert self.run(repo, unquantized_only=True) == []
        repo.get_all.assert_not_called()

    def test_without_the_flag_nothing_is_filtered_or_unpaged(self):
        models = [Model(id="m0", filename="x.safetensors", model_type="diffusion_model")]
        repo = self.repo(models)
        result = self.run(repo, limit=5, offset=2)
        assert [r["id"] for r in result] == ["m0"]
        assert repo.get_all.call_args.kwargs["limit"] == 5


class TestRepositoryTypeList:
    @pytest.fixture
    def repository(self, mock_db):
        return ModelRepository()

    def _put(self, repository, name, model_type):
        return repository.create(Model(filename=name, file_size=1, sha256=(name * 64)[:64], model_type=model_type))

    def test_lists_the_union_of_the_requested_types(self, repository):
        self._put(repository, "a.safetensors", "diffusion_model")
        self._put(repository, "b.safetensors", "unet")
        self._put(repository, "c.safetensors", "checkpoint")
        self._put(repository, "d.safetensors", "text_encoder")
        self._put(repository, "e.safetensors", "lora")
        found = repository.get_all(model_types=["diffusion_model", "unet", "text_encoder"])
        assert sorted(m.filename for m in found) == ["a.safetensors", "b.safetensors", "d.safetensors"]

    def test_combines_with_the_other_filters(self, repository):
        self._put(repository, "a.safetensors", "diffusion_model")
        self._put(repository, "b-extra.safetensors", "unet")
        found = repository.get_all(model_types=["diffusion_model", "unet"], search="extra")
        assert [m.filename for m in found] == ["b-extra.safetensors"]

    def test_single_type_still_wins_and_empty_list_means_every_defined_type(self, repository):
        self._put(repository, "a.safetensors", "diffusion_model")
        self._put(repository, "b.safetensors", "lora")
        assert [m.filename for m in repository.get_all(model_type="lora", model_types=["diffusion_model"])] == ["b.safetensors"]
        assert len(repository.get_all(model_types=[])) == 2
