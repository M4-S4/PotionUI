---
type: model
title: MOSS-Image
family_key: moss_image
modes: [txt2img]
spec:
  arch: MOSS-Image DiT (42 blocks, hidden size 3072, fused QKV, two text-embedder blocks)
  latent: 128-channel FLUX 2 autoencoder latent (16x downscale, 2D)
  vae: flux2_ae (flux2-vae.safetensors)
  te: Qwen3.5-9B (Gated DeltaNet hybrid, int8 ConvRot)
  guidance: "true CFG with block 22 skipped on the negative pass"
  shift: "5.0, snapped to the training grid"
  engine: native
files:
  - role: dit
    dir: models/checkpoints
    note: SFT or Base, bf16 or int8 ConvRot
  - role: text_encoder
    dir: models/text_encoders
    note: Qwen3.5-9B int8 ConvRot
  - role: vae
    dir: models/vae
---

# MOSS-Image

MOSS-Image (`src/platform/runtime/native/arch/moss_image/`) is OpenMOSS's open text-to-image model. It takes natural-language prompts, tag lists and JSON layout prompts that place text and objects. The prompt is encoded by Qwen3.5-9B (`src/platform/runtime/native/text_encoders/qwen35.py`), a hybrid of Gated DeltaNet and full-attention layers.

## Files & detection

The Comfy-Org/MOSS-Image repository ships four DiT files that share one architecture and are told apart from the weights:

- `moss_image_sft_bf16.safetensors` and `moss_image_base_bf16.safetensors`: about 9.5 GB each.
- `moss_image_sft_int8_convrot_attn.safetensors` and `moss_image_base_int8_convrot_attn.safetensors`: about 5 GB each. Only the per-block projections are int8; the text embedder, modulation, norms and head stay bf16.

The text encoder is `qwen3.5_9b_int8_convrot.safetensors` (about 9 GB, int8 ConvRot). The VAE is `flux2-vae.safetensors`, the same file the Flux2 preset recommends.

## Presets & modes

The shipped **MOSS-Image** preset has `txt2img` only, with a Base speed profile (50 steps, CFG 3.0, the default) and a Custom one. The pipes are `model_loader/moss_image` and `generator/moss_image`.

The negative prompt is empty by default and only used when CFG is above 1. The Advanced tab has a Negative-branch layer skip toggle: on skips transformer block 22 on the negative pass, off runs plain CFG. The generator forces CFG-Zero\* off and never uses the step cache, because the negative pass is always degraded.

## Sampling

Resolutions are multiples of 16. The sigma shift is 5.0 unless the Advanced tab overrides it, and the default schedule (`moss_grid`) snaps the shifted ramp onto the training timestep grid, as the reference sampler does. The Sampler and Schedule pickers (family `moss_image`) list whatever the instance has registered; see [Samplers and sigma schedules](../techniques/samplers-and-schedules.md).

## Limitations

- No Turbo checkpoint, no LoRA tab, no img2img.
- The schedule follows the reference's snapped grid, so the same seed renders a different image than ComfyUI.
- The int8 checkpoints carry an int8-attention hint that is ignored; attention runs on SDPA.

## Hardware

The preset requires 12 GB of VRAM and recommends 24 GB. The DiT takes about 10.2 GB in bf16 or 5.4 GB in int8, and the int8 text encoder about 8 GB; the two are swapped between the encode and denoise phases.
