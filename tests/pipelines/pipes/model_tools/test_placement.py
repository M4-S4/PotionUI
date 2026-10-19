from __future__ import annotations

import os
import sys
from collections import namedtuple

import pytest

from src.pipelines.pipes.model_tools import placement
from src.pipelines.pipes.model_tools.errors import ModelToolsRefused

from .conftest import make_depot

Usage = namedtuple("Usage", "total used free")


@pytest.fixture
def type_dir(tmp_path):
    path = tmp_path / "models"
    path.mkdir()
    return path


class TestContainment:
    def test_a_plain_name_beside_the_source_is_accepted(self, type_dir):
        dest = placement.contained_destination(type_dir / "sub", type_dir, "out.safetensors", "diffusion_model")
        assert dest.path == type_dir / "sub" / "out.safetensors"

    @pytest.mark.parametrize("name", ["../out.safetensors", "a/b.safetensors", "..", "sub/../../x"])
    def test_names_that_walk_out_are_refused(self, type_dir, name):
        with pytest.raises(ModelToolsRefused):
            placement.contained_destination(type_dir, type_dir, name, "diffusion_model")

    def test_a_directory_outside_the_type_dir_is_refused(self, tmp_path, type_dir):
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        with pytest.raises(ModelToolsRefused, match="outside the model folder"):
            placement.contained_destination(outside, type_dir, "out.safetensors", "diffusion_model")

    def test_a_sibling_sharing_the_prefix_is_not_inside(self, tmp_path, type_dir):
        sibling = tmp_path / "models-extra"
        sibling.mkdir()
        with pytest.raises(ModelToolsRefused, match="outside the model folder"):
            placement.contained_destination(sibling, type_dir, "out.safetensors", "diffusion_model")

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
    def test_a_symlinked_subfolder_that_escapes_is_refused(self, tmp_path, type_dir):
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        os.symlink(outside, type_dir / "link", target_is_directory=True)
        with pytest.raises(ModelToolsRefused, match="symbolic link"):
            placement.contained_destination(type_dir / "link", type_dir, "out.safetensors", "diffusion_model")


class TestFreeSpace:
    def test_enough_space_passes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(placement.shutil, "disk_usage", lambda _p: Usage(0, 0, 10 * placement.GIB))
        placement.check_free_space(tmp_path, 5 * placement.GIB)

    def test_the_margin_counts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(placement.shutil, "disk_usage", lambda _p: Usage(0, 0, 5 * placement.GIB))
        with pytest.raises(ModelToolsRefused):
            placement.check_free_space(tmp_path, 5 * placement.GIB)

    def test_refusal_names_needed_and_available(self, tmp_path, monkeypatch):
        monkeypatch.setattr(placement.shutil, "disk_usage", lambda _p: Usage(0, 0, 2 * placement.GIB))
        with pytest.raises(ModelToolsRefused) as err:
            placement.check_free_space(tmp_path, 20 * placement.GIB)
        assert "needs about 20.06 GiB" in str(err.value)
        assert "only 2.00 GiB is available" in str(err.value)


class TestLocate:
    def test_a_file_in_the_lora_folder_is_a_lora(self, tmp_path):
        depot = make_depot(tmp_path)
        lora = depot.loras / "x.safetensors"
        lora.write_bytes(b"0")
        found = placement.locate_model_file(depot.resolver, str(lora), "LoRA 1", frozenset({"lora"}))
        assert found.model_type == "lora"
        assert found.type_dir == depot.loras

    def test_a_relative_path_is_taken_from_the_working_directory(self, tmp_path, monkeypatch):
        depot = make_depot(tmp_path)
        (depot.models / "m.safetensors").write_bytes(b"0")
        monkeypatch.chdir(tmp_path)
        found = placement.locate_model_file(
            depot.resolver, os.path.join("depot", "diffusion_models", "m.safetensors"), "Model", placement.MODEL_TYPES
        )
        assert found.path == depot.models / "m.safetensors"

    def test_a_folder_is_not_a_model_file(self, tmp_path):
        depot = make_depot(tmp_path)
        (depot.models / "dir.safetensors").mkdir()
        with pytest.raises(ModelToolsRefused, match="not found"):
            placement.locate_model_file(
                depot.resolver, str(depot.models / "dir.safetensors"), "Model", placement.MODEL_TYPES
            )


class TestOverwrite:
    def test_the_destination_equal_to_an_input_is_refused_even_when_overwriting(self, tmp_path):
        depot = make_depot(tmp_path)
        source = depot.models / "m.safetensors"
        source.write_bytes(b"0")
        dest = placement.contained_destination(depot.models, depot.models, "m.safetensors", "diffusion_model")
        with pytest.raises(ModelToolsRefused, match="own input"):
            placement.check_overwrite(dest, [source], True)

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
    def test_a_dangling_symlink_counts_as_existing(self, tmp_path):
        depot = make_depot(tmp_path)
        os.symlink(tmp_path / "gone", depot.models / "out.safetensors")
        dest = placement.contained_destination(depot.models, depot.models, "out.safetensors", "diffusion_model")
        with pytest.raises(ModelToolsRefused, match="already exists"):
            placement.check_overwrite(dest, [], False)
