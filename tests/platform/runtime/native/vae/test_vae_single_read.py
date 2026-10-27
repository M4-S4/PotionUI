from __future__ import annotations

import contextlib
import dataclasses
from collections import Counter
from unittest import mock

import pytest
import torch
from safetensors import safe_open as real_safe_open
from safetensors.torch import save_file

from src.platform.runtime.native.detect.vae_detect import detect_vae_config
from src.platform.runtime.native.engine import NativeEngineLoader
from src.platform.runtime.native.errors import NativeEngineUnsupportedError
from src.platform.runtime.native.io import safetensors_loader
from src.platform.runtime.native.vae.ae_2d import AutoEncoder2D
from src.platform.runtime.native.vae import causal_3d, causal_3d_v2
from src.platform.runtime.native.vae.loader import _BARE_VAES, _load_bare_vae, build_native_vae, load_vae
from src.platform.runtime.native.vae.seedvr2_causal_video import SeedVR2CausalVideoVAE
from src.platform.runtime.native.kernels import disable_weight_init

from .test_ae_2d import _bfl_layout, _randomize_weights, _tiny_config
from .test_qwen_image21_vae import _build_qwen21


class _Reads:
    def __init__(self) -> None:
        self.opens = 0
        self.full_reads = 0
        self.tensors: Counter[str] = Counter()


@pytest.fixture
def reads(monkeypatch) -> _Reads:
    counts = _Reads()

    class _Handle:
        def __init__(self, handle) -> None:
            self._handle = handle

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *exc):
            return self._handle.__exit__(*exc)

        def keys(self):
            return self._handle.keys()

        def get_tensor(self, key):
            counts.tensors[key] += 1
            return self._handle.get_tensor(key)

    def counting_safe_open(*args, **kwargs):
        counts.opens += 1
        return _Handle(real_safe_open(*args, **kwargs))

    real_load_file = safetensors_loader.load_file

    def counting_load_file(*args, **kwargs):
        counts.full_reads += 1
        return real_load_file(*args, **kwargs)

    monkeypatch.setattr(safetensors_loader, "safe_open", counting_safe_open)
    monkeypatch.setattr(safetensors_loader, "load_file", counting_load_file)
    return counts


@pytest.fixture
def bare_key_vae_path(tmp_path):
    config = _tiny_config(vae_type="flux2_ae", latent=32, has_quant_conv=True, has_batchnorm=True)
    source = AutoEncoder2D.from_config(config, disable_weight_init)
    _randomize_weights(source)
    sd = _bfl_layout(source)
    path = tmp_path / "ae.safetensors"
    save_file({k: v.contiguous() for k, v in sd.items()}, str(path))
    return path, sd


def test_a_bare_key_vae_file_is_read_exactly_once_per_engine_load(bare_key_vae_path, reads):
    path, sd = bare_key_vae_path

    model = NativeEngineLoader(device="cpu")._load_vae(path)

    assert reads.opens + reads.full_reads == 1
    assert set(reads.tensors) == set(sd)
    assert set(reads.tensors.values()) == {1}
    assert model.module.vae_type == "flux2_ae"


def test_build_native_vae_reads_nothing_when_handed_the_state_dict(bare_key_vae_path, reads):
    path, _sd = bare_key_vae_path
    state_dict, metadata = safetensors_loader.load_torch_file(path)
    reads.opens = reads.full_reads = 0

    module = build_native_vae(path, disable_weight_init, state_dict, metadata)

    assert reads.opens == reads.full_reads == 0
    assert detect_vae_config(state_dict)["vae_type"] == module.vae_type


def test_the_standalone_loader_reads_once_and_reuses_a_passed_state_dict(bare_key_vae_path, reads):
    path, _sd = bare_key_vae_path

    load_vae(path, disable_weight_init, device="cpu")
    assert reads.full_reads == 1

    state_dict, _ = safetensors_loader.load_torch_file(path)
    reads.full_reads = 0
    load_vae(path, disable_weight_init, sd=state_dict)
    assert reads.full_reads == 0


def test_an_unrecognised_file_keeps_the_family_error_text(tmp_path):
    path = tmp_path / "junk.safetensors"
    save_file({"nothing.weight": torch.zeros(2)}, str(path))

    with pytest.raises(NativeEngineUnsupportedError, match="does not look like a Flux-family 2D VAE"):
        load_vae(path, disable_weight_init)
    with pytest.raises(NativeEngineUnsupportedError, match="matches no known native VAE family"):
        build_native_vae(path, disable_weight_init, {"nothing.weight": torch.zeros(2)}, {})


def _flux_state():
    config = _tiny_config(vae_type="flux2_ae", latent=32, has_quant_conv=True, has_batchnorm=True)
    source = AutoEncoder2D.from_config(config, disable_weight_init)
    _randomize_weights(source)
    return _bfl_layout(source), contextlib.nullcontext()


def _causal3d_state():
    with mock.patch.object(causal_3d, "_DIM", 8):
        module = causal_3d.AutoEncoderCausal3D.from_config({}, disable_weight_init)
    _randomize_weights(module)
    return module.state_dict(), mock.patch.object(causal_3d, "_DIM", 8)


def _causal3d_v2_state():
    module = causal_3d_v2.AutoEncoderCausal3D_2_2.from_config({"dim": 8, "dec_dim": 8}, disable_weight_init)
    _randomize_weights(module)
    patches = contextlib.ExitStack()
    patches.enter_context(mock.patch.object(causal_3d_v2, "_ENC_DIM", 8))
    patches.enter_context(mock.patch.object(causal_3d_v2, "_DEC_DIM", 8))
    return module.state_dict(), patches


def _qwen_image21_state():
    module = _build_qwen21()
    _randomize_weights(module)
    return module.state_dict(), contextlib.nullcontext()


def _seedvr2_state():
    tiny = {"block_out_channels": (32, 32, 32, 32), "layers_per_block": 1}
    module = SeedVR2CausalVideoVAE.from_config(tiny, disable_weight_init)
    _randomize_weights(module)
    spec = _BARE_VAES["seedvr2"]
    real = spec.detect
    tiny_spec = dataclasses.replace(spec, detect=lambda sd: {**real(sd), **tiny})
    return module.state_dict(), mock.patch.dict(_BARE_VAES, {"seedvr2": tiny_spec})


_FAMILIES = {
    "flux": _flux_state,
    "causal3d": _causal3d_state,
    "causal3d_v2": _causal3d_v2_state,
    "qwen_image21": _qwen_image21_state,
    "seedvr2": _seedvr2_state,
}


def test_the_fixture_table_covers_every_bare_vae_family():
    assert set(_FAMILIES) == set(_BARE_VAES)


@pytest.mark.parametrize("kind", sorted(_FAMILIES))
def test_a_passed_state_dict_is_never_reread_for_any_bare_vae_family(kind, tmp_path, reads):
    sd, patches = _FAMILIES[kind]()
    path = tmp_path / f"{kind}.safetensors"
    save_file({k: v.contiguous() for k, v in sd.items()}, str(path))
    state_dict, metadata = safetensors_loader.load_torch_file(path)
    reads.opens = reads.full_reads = 0

    with patches:
        direct = _load_bare_vae(kind, path, disable_weight_init, sd=state_dict)
        dispatched = build_native_vae(path, disable_weight_init, state_dict, metadata)

    assert reads.opens == reads.full_reads == 0
    assert type(direct) is type(dispatched) is _BARE_VAES[kind].module_class
