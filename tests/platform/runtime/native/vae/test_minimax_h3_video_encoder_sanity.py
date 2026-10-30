from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from src.pipelines.contracts import PipeInput
from src.pipelines.pipes.latent_upscaler.minimax_h3.main import (
    UPSCALE_NEEDS_ENCODER_MESSAGE,
    LatentUpscalerMinimaxH3Pipe,
)
from src.platform.runtime.native.detect.vae_detect import detect_minimax_h3_video_vae_config
from src.platform.runtime.native.errors import NativeEngineLoadIntegrityError
from src.platform.runtime.native.kernels import disable_weight_init
from src.platform.runtime.native.vae.loader import load_minimax_h3_video_vae
from src.platform.runtime.native.vae.minimax_h3_video import (
    ENCODER_KEY_PREFIXES,
    MiniMaxH3VideoVAE,
    VaeCannotEncodeError,
)

_LOADER = "src.platform.runtime.native.vae.loader"

_TINY = dict(
    latent_channels=4,
    block_out_channels=(8, 16),
    layers_per_block=1,
    spatial_downsample_factors=(2, 1),
    temporal_downsample_factors=(1, 2),
    norm_num_groups=2,
    decoder_num_layers=2,
    decoder_num_attention_heads=2,
    decoder_attention_head_dim=8,
    decoder_num_register_tokens=2,
    decoder_ffn_mult=2,
    clip_length=5,
    token_drop=1,
)


def _state_dict() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    module = MiniMaxH3VideoVAE.from_config(_TINY, disable_weight_init)
    with torch.no_grad():
        for p in module.parameters():
            if p.is_floating_point():
                p.normal_(std=0.02)
    return {k: v.clone() for k, v in module.state_dict().items()}


def _load(sd):
    with patch(f"{_LOADER}.detect_minimax_h3_video_vae_config", return_value=dict(_TINY)):
        return load_minimax_h3_video_vae("tiny_h3_vae.safetensors", disable_weight_init, sd=sd, metadata={})


def _encoder_keys(sd):
    return [k for k in sd if k.startswith(ENCODER_KEY_PREFIXES)]


def test_a_complete_file_loads_able_to_encode_and_logs_both_key_counts(caplog):
    sd = _state_dict()
    with caplog.at_level(logging.INFO, logger=_LOADER):
        module = _load(sd)
    assert module.can_encode is True
    n_enc = len(_encoder_keys(sd))
    n_dec = len(sd) - n_enc
    assert f"encoder keys {n_enc}/{n_enc}" in caplog.text
    assert f"decoder keys {n_dec}/{n_dec}" in caplog.text
    assert "can_encode=True" in caplog.text
    with torch.no_grad():
        assert module.encode(torch.rand(1, 3, 5, 8, 8)).shape[1] == _TINY["latent_channels"]


def test_a_decode_only_file_loads_marked_unable_to_encode(caplog):
    sd = {k: v for k, v in _state_dict().items() if not k.startswith(ENCODER_KEY_PREFIXES)}
    with caplog.at_level(logging.INFO, logger=_LOADER):
        module = _load(sd)
    assert module.can_encode is False
    assert module.encoder is None
    assert "encoder keys 0/" in caplog.text
    assert "can_encode=False" in caplog.text
    with torch.no_grad():
        frames = module.decode(torch.randn(1, 4, 5, 2, 2))
    assert torch.isfinite(frames).all()


def test_an_encoder_missing_one_tensor_is_never_left_half_initialised():
    sd = _state_dict()
    del sd["encoder.down.0.block.0.conv1.weight"]
    module = _load(sd)
    assert module.can_encode is False
    assert module.encoder is None
    assert module.quant_conv is None


def test_an_encoder_tensor_of_the_wrong_shape_counts_as_missing():
    sd = _state_dict()
    sd["encoder.conv_out.weight"] = torch.zeros(1, 1, 3, 3, 3)
    module = _load(sd)
    assert module.can_encode is False


def test_a_missing_decoder_tensor_still_fails_the_load():
    sd = _state_dict()
    del sd["decoder.proj_out.weight"]
    with pytest.raises(NativeEngineLoadIntegrityError):
        _load(sd)


def test_encoding_with_a_decode_only_vae_raises_a_clear_error():
    sd = {k: v for k, v in _state_dict().items() if not k.startswith(ENCODER_KEY_PREFIXES)}
    module = _load(sd)
    with pytest.raises(VaeCannotEncodeError, match="can only decode"):
        module.encode(torch.rand(1, 3, 5, 8, 8))


def test_detection_accepts_a_file_with_no_encoder_at_all():
    sd = {k: v for k, v in _state_dict().items() if not k.startswith(ENCODER_KEY_PREFIXES)}
    assert detect_minimax_h3_video_vae_config(sd) is not None


def test_upscale_refuses_a_decode_only_vae_before_reading_the_video():
    module = SimpleNamespace(can_encode=False)
    bundle = SimpleNamespace(
        video_vae=SimpleNamespace(module=module), upsampler=SimpleNamespace(module=None),
        audio_vae=None, dit=None,
    )
    pipe = LatentUpscalerMinimaxH3Pipe(config={**LatentUpscalerMinimaxH3Pipe.get_default_config(), "device": "cpu"})
    with patch("src.pipelines.pipes.latent_upscaler.minimax_h3.main.load_video_frames") as load:
        with pytest.raises(VaeCannotEncodeError) as raised:
            pipe.process(PipeInput(input={"model": bundle, "video": ["clip.mp4"]}), lambda o: None)
    assert str(raised.value) == UPSCALE_NEEDS_ENCODER_MESSAGE
    assert "pick the standard MiniMax-H3 video VAE for Upscale" in UPSCALE_NEEDS_ENCODER_MESSAGE
    load.assert_not_called()
