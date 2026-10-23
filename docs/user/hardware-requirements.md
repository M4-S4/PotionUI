---
title: Hardware Requirements
order: 72
---

# Hardware Requirements

PotionUI runs on a single CUDA-capable NVIDIA GPU. What you can actually generate — and how well — depends on that GPU's VRAM and, just as much, on the host's system RAM. This page gives honest numbers instead of "a GPU with enough VRAM."

## GPU generation and driver

PotionUI requires **Turing or newer GPUs** (sm_75 and above). CUDA 13 dropped support for Maxwell, Pascal, and Volta architectures, so **GTX 10-series and older cards cannot run PotionUI at all** — RTX 20xx / GTX 16xx and newer are supported.

The **NVIDIA driver must be version 580 or newer** on your host machine. Older r550 or r570 drivers fail immediately with "CUDA driver version is insufficient for CUDA runtime version." Within the CUDA 13.x series, the 580+ driver remains compatible across minor versions, so a 580.65+ driver will continue to work as CUDA 13 updates ship.

For Docker deployments, the host needs only the NVIDIA driver and `nvidia-container-toolkit` — the torch PyPI wheel carries the CUDA 13 userspace runtime, so a separate host CUDA toolkit install is not required.

## The floor

**8 GB VRAM + 16 GB system RAM** is the supported minimum. On that floor, **SDXL** is the model family PotionUI guarantees works — it runs on `diffusers` pipelines with its own VRAM-tier memory policy (aggressive sequential offload + max attention slicing below 8 GB, model offload 8–12 GB, no offload 12–16 GB, fully GPU-resident at 16 GB+ — `src/platform/runtime/model_lifecycle/memory_policy.py`). Every other native-engine family (Flux, Krea-2, Qwen-Image, Wan, LTX, Z-Image, Anima, MiniMax-H3) needs more, some considerably more — see the per-family table below. The smallest video models also run on an 8 GB card, but they need **32 GB of system RAM**, not 16 — see [Video presets by hardware tier](#video-presets-by-hardware-tier) below.

**Recommended: 16–24 GB VRAM + 32 GB system RAM.** This is what the native engine's low-VRAM design targets as "first class" (`docs/native-engine.md`), and it's comfortably above the RAM-cache floor (below) for realistic multi-model sessions.

The native engine (`src/platform/runtime/native/`) is explicitly designed for small GPUs, not just the 32 GB card it was developed on: `memory/tiering.py` places each component (DiT / text encoder / VAE) by its *actual* measured size, and a component that doesn't fit fully resident streams the overflow from pinned host RAM (partial residency) or gets quantized to fp8 on load instead of failing outright. That's what makes several families runnable — slowly — below their "comfortable" VRAM tier.

## System RAM is not optional headroom

PotionUI reserves system RAM the same way it reserves VRAM, and this bites on small boxes:

- **Before a load, the model RAM cache tries to reclaim free system RAM up to a reserve of `min(max(8 GiB, 10% of total RAM), 25% of total RAM)`** (`host_ram_reserve_gb()`, `src/platform/runtime/model_lifecycle/lifecycle.py`, applied in `_make_room_for_ram`). That's 2 GiB on an 8 GiB box, 4 GiB at 16 GiB, 8 GiB at 32 and 64 GiB, and 10% of total once total RAM exceeds 80 GiB (12.8 GiB at 128 GiB) — it scales down on small machines instead of reserving a fixed 8 GiB regardless of total RAM. It is a retention/eviction policy, not a guarantee that any particular model fits: cached entries are evicted until the reserve is met, and if pressure persists (for example when every remaining entry is leased by an active generation) the lifecycle warns and continues, so don't count on cross-generation caching surviving a genuinely tight box.
- **Low-VRAM streaming (partial residency) needs `streamed_gb + 2 GB` of free host RAM up front**, and refuses to start rather than risk an OS OOM-kill if it isn't there (`HostMemoryExhaustedError`, `_guard_host_ram_for_streaming` / `_STREAM_HOST_RESERVE_GB = 2.0`, `src/platform/runtime/native/engine.py:271,353-375`). A big model streaming several GB of overflow on a 16 GB RAM host can hit this guard even when VRAM itself would have been fine. In a memory-limited container the guard counts the file cache of the model files as used RAM: the Wan 2.1 1.3B preset was refused at a 16 GB limit because the guard did not see the 2.6 GB for streaming plus its 2 GB margin.

Practical takeaway: **16 GB system RAM is the bare minimum that keeps the app from refusing to load, not a comfortable amount.** 32 GB+ is what lets the RAM cache actually do its job (reuse a loaded checkpoint across generations instead of re-reading it from disk every time) and gives streaming headroom for the larger native families. For video the measured figures are higher: **32 GB** for the 8 GB-card presets, **64 GB** for the 12 GB-card presets and for Wan 2.2 14B, and at least **36 GB of free RAM** for LTX-2.5 Layout-To-Render (see [Video presets by hardware tier](#video-presets-by-hardware-tier)).

## Per-family verdicts

Peak-VRAM figures below marked **(32 GB-card ceiling)** are measured GPU runs on a 32 GB development card under fit-first placement (`docs/native-engine.md`) — they are *not* minimums, since a smaller card places the same model differently (more offload/streaming, lower peak, slower). Where no real-weight run exists yet, that's stated explicitly rather than guessed. The Wan, LTX-2.5 and MiniMax-H3 rows also carry figures from the hardware-tier runs of [Video presets by hardware tier](#video-presets-by-hardware-tier), which use a different method (the engine's own peak on a 12 GB card, some of them capped to 8 GB).

| Family | DiT size (bf16 unless noted) | Measured peak (32 GB-card ceiling) | 8 GB card | 12 GB card | Recommended |
|---|---|---|---|---|---|
| SDXL | few GB, single-file checkpoint | — (diffusers pipeline, not native engine) | **Works** — the supported floor | Works comfortably | 8 GB+ |
| Z-Image | ~11.5 GB bf16 (`zImage_turbo`, `docs/native-engine.md`) | CPU-validated only; no GPU e2e peak measured yet | Unvalidated | Unvalidated but the best native bet at this tier (smallest native DiT) | 12–16 GB |
| Krea-2 | 26 GB bf16 on disk (~12B DiT); fp8 quantize-at-load brings it to **~12.5 GB** (`docs/native-engine.md`) | 26 GB (streaming manual_cast tier, 1024² turbo) | No | **Viable via fp8 quantize-at-load** | 16 GB+ bf16, 12 GB fp8 |
| Flux2 / Klein | 9B params → ~18 GB bf16 estimated (params×2); fp8 checkpoint measured smaller | fp8: 28.7 GB · bf16: 28.55 GB (both include the Qwen3 TE, `docs/native-engine.md`) | No | **Viable but slow** — partial residency + fp8-auto, unmeasured at this tier | 16 GB+ |
| Qwen-Image | fp8 DiT file ~20 GB (`qwen_image_2512_fp8_e4m3fn`) | 19.37 GB (1024², true-CFG, `docs/native-engine.md`) | No | Marginal — close to the fp8 file size itself | 16 GB+ |
| Anima | not separately measured; small Qwen3-0.6B TE keeps total footprint below other families | 20.7 GB (1024², 24 steps, incl. causal-3D fp32 decode spike, `docs/native-engine.md`) | No | Marginal | 24 GB |
| Wan 2.2 (14B, dual-expert) | 14B per resident expert | 15.2 GB (33 frames, 832×480, fp8 expert pair — short low-res clip, `docs/native-engine.md`) | No | **Works, slowly** — text to video, text to video with the Lightning LoRAs (3 min 24 s for a 5 s clip at 832×480, models already in memory) and image to video (5 min 21 s, loading included) all finish on the whole card; about 35 GiB of process memory, so 64 GB of system RAM (see [Video presets by hardware tier](#video-presets-by-hardware-tier)) | 16 GB+; 12 GB + 64 GB RAM with the Wan 2.2 14B Lightning (12 GB) preset; for a small card use the 5B or 1.3B presets |
| Wan 2.2 (5B ti2v) | single dense model, coarser spatial granularity; 10.0 GB fp16 file, converted to fp8 at load on an 8 GB card | not measured on the 32 GB card; 5,588 MiB engine peak at 832×480 on a 12 GB card capped to 8 GB | **Works in 480p only, with 32 GB of system RAM** (measured at 832×480) — 1280×704 failed in the VAE decode | The preset's recommended tier (`recommended_vram_gb: 12`); not run without the cap | 8 GB + 32 GB RAM minimum, 12 GB recommended |
| Wan 2.1 (T2V 1.3B) | 2.84 GB fp16 DiT + 6.74 GB fp8 UMT5-XXL text encoder | not measured on the 32 GB card; 5,314 MiB engine peak at 832×480 on a 12 GB card capped to 8 GB | **Works with 32 GB of system RAM** — refused with 16 GB in a container | The preset's recommended tier (`recommended_vram_gb: 12`); not run without the cap | 8 GB + 32 GB RAM minimum, 12 GB recommended |
| LTX-2 / 2.3 | ~27 GB on-disk all-in-one AV checkpoint (19–22B params) | Not GPU-validated end-to-end — DiT/TE built, AV forward + golden validation still open (`docs/native-engine.md`) | No | No — LTX-2.3 22B distilled (fp8) ran out of GPU memory in the DiT on a 12 GB card, for a 5 s and a 2 s clip | Not yet recommended at any tier — validation pending |
| LTX-2.5 (22B distilled) | split files instead of an all-in-one checkpoint: int8 ConvRot DiT, Gemma4 12B int8 text encoder, video and audio VAEs — five files, about 40 GB on disk | not measured on the 32 GB card; 10,702 MiB engine peak for Layout-To-Render at 832×480, 3 s on a 12 GB card | Not measured | **Works, slowly, on the whole card** — text to video (1 s at 480×832 with sound) in 5 min 57 s and Layout-To-Render (3 s at 832×480) in 4 min 30 s, both from a cold start; Layout-To-Render needs at least 36 GB of free system RAM (see [Video presets by hardware tier](#video-presets-by-hardware-tier)) | 12 GB + at least 36 GB of free RAM, the tier that was measured |
| MiniMax-H3 | 20B (pruned) / 33B (full) params, per its own doc front matter; the 12 GB presets use an int8 ConvRot DiT of about 21 GB (FastH3: 22.13 GB) and a 15.69 GB nvfp4 text encoder | not measured on the 32 GB card; 5,097 MiB engine peak at 832×480, 124 frames on a 12 GB card, with the text encoder streamed from system RAM | No — the 12 GB presets require 12 GB | **Works with 64 GB of system RAM** — MiniMax-H3 (12 GB) and MiniMax-H3-Fast (12 GB) | 12 GB + 64 GB RAM with the 12 GB presets; the weights are territorially license-restricted (excludes EU/UK/US/KR), so read the preset description first |
| SeedVR2 (upscaler) | 3B fp8 **3.39 GB** / 3B fp16 6.78 GB / 7B fp8 **8.24 GB** / 7B fp16 16.48 GB (`content/presets/marketplace/SeedVR2/modes/upscale/tabs/generation.yml`) | Batch size auto-sizes to ~72% of free VRAM; halves and retries on OOM | 3B fp8 works | 7B fp8 works | Scales down cleanly to small cards — no fixed floor |

## Video presets by hardware tier

The Video Producer plugin ships five video presets, each set up for one size of card and one size of system RAM (`requires.min_vram_gb` and `requires.min_ram_gb`, shown as badges in the preset picker), and a recipe for each that downloads its files. One more recipe, `vp-wan22-14b-i2v`, completes the core Wan 2.1 / 2.2 preset with the 14B image-to-video experts, and the core LTX-2.5 preset has its own recipe, `ltx25-starter`; the table below also carries their 12 GB runs. Their numbers are measured, not estimated: one machine with an RTX 4070 Ti (12 GB of VRAM), one clip per run (5 seconds unless the row says otherwise), cold start included unless the row says otherwise. They are not guarantees for another machine.

| Preset | Card / system RAM | Result by container RAM limit | Time | Engine VRAM peak | Process RAM peak |
|---|---|---|---|---|---|
| Wan 2.1 T2V 1.3B (8 GB), 832×480 | 8 GB / 32 GB | Done at 32 GB and 29 GB; refused by the engine at 16 GB | 5 min 55 s (32 GB), 5 min 40 s (29 GB) | 5,314 MiB | 10.7 GiB |
| Wan 2.2 TI2V 5B (8 GB), 832×480 | 8 GB / 32 GB | Done at 32 GB and 29 GB | 4 min 51 s (32 GB), 4 min 38 s (29 GB) | 5,588 MiB | 18.6 GiB |
| Wan 2.2 TI2V 5B, 1280×704 | — | Failed at 32 GB: the VAE decode needed more than 8 GB; stopped after 38 min | — | — | 17.0 GiB |
| MiniMax-H3 (12 GB), 832×480 | 12 GB / 64 GB | Done at 32 GB (no margin); killed for lack of memory at 29 GB | 9 min 48 s | 5,097 MiB | 24.5 GiB |
| MiniMax-H3-Fast (12 GB), 832×480 | 12 GB / 64 GB | Killed for lack of memory at 32 GB; done at 62 GB | 6 min 11 s | 5,097 MiB | 25.6 GiB |
| Wan 2.2 14B Lightning (12 GB), 832×480 | 12 GB / 64 GB | Done on the whole card, no out-of-memory error | 3 min 24 s, models already in memory | 8,114 MiB | 34.7 GiB |
| Wan 2.2 14B image to video (Wan 2.1 / 2.2 preset), 832×480, 6 steps | 12 GB / 64 GB | Done on the whole card | 5 min 21 s, of which 65 s of loading | 6,938 MiB | — |
| LTX-2.5 text to video, 480×832, 1 s with sound, 8 steps | 12 GB / — | Done on the whole card | 5 min 57 s from a cold start (about 7.5 s per step) | — | — |
| LTX-2.5 Layout-To-Render, 832×480, 3 s (73 frames), 8 steps | 12 GB / at least 36 GB free | Done on the whole card | 4 min 30 s, of which 103 s of loading (about 7 s per step) | 10,702 MiB | 48,273 MB |

- **8 GB of VRAM, 32 GB of system RAM.** Wan 2.1 T2V 1.3B and Wan 2.2 TI2V 5B make text to video only, and the 5B only in 480p: its native 1280×704 does not decode inside 8 GB. With 16 GB of RAM the 1.3B was refused by the streaming guard (see above). The 5B holds more system RAM than the 1.3B because its fp16 file is read whole into RAM before the engine converts it to fp8 at load.
- **12 GB of VRAM, 64 GB of system RAM.** MiniMax-H3 (12 GB) and MiniMax-H3-Fast (12 GB) stream the 32B text encoder from system RAM instead of loading it on the card (the native backend's text encoder placement, `auto` by default, does this), and so do the DiT layers that do not fit. The base preset finished with 32 GB but without margin and the Fast one did not, so both declare 64 GB.
- **Docker Desktop on Windows.** WSL 2 gives Docker half of the host's RAM by default. On a 32 GB PC, raise it in `%UserProfile%\.wslconfig` (`[wsl2]`, `memory=28GB`), then run `wsl --shutdown`.
- **Wan 2.2 14B on a 12 GB card, with 64 GB of system RAM.** Wan 2.2 14B Lightning (12 GB) makes text to video; image to video uses the 14B image-to-video experts of the Wan 2.1 / 2.2 preset, which the `vp-wan22-14b-i2v` recipe installs. Text to video, text to video with the Lightning LoRAs and image to video all finish on the whole card, with the engine told the real 12 GB. Measured at 832×480 for a 5-second clip:
  - **Text to video without a LoRA** works: a 30-step draft took about 26 minutes on the whole card in an earlier run, and a 6-step run in a container limited to 62 GB took 5 min, with a 6,930 MiB engine peak and 34.8 GiB of process memory.
  - **With the lightx2v Lightning LoRAs:** 3 min 24 s with the models already in memory (the first run of a session also reads the 38 GB of files from disk), an 8,114 MiB engine peak and 34.7 GiB of process memory, with no out-of-memory error.
  - **Image to video** (6 steps): 5 min 21 s including 65 s of loading, with a 6,938 MiB engine peak.
  - **What changed.** Before the engine fixes of 20 October 2026 both the Lightning and the image-to-video runs failed on the whole card: the Lightning run's process was killed by a CUDA out-of-memory error that was not caught, and the image-to-video run ran out of GPU memory while encoding the first frame. The fixes return the allocator's pool to the driver when a streamed expert is torn down, evict the UMT5 text encoder before the VAE encodes, and classify the driver's out-of-memory error (`torch.AcceleratorError`) as an out-of-memory error.
  - **First generation of each recipe**, a 1-second clip: 242 s for the Lightning recipe from a cold start (its `smoke.timeout_seconds` is 600) and 171 s for the image-to-video recipe (600; its check is a short text-to-video clip).
  - **Lightning on a 1-second clip**, models already in memory: 45 s, against 175 s for a 5-second clip (56 s and 186 s before the idle expert was kept pinned between generations). Before the recipe check form was fixed on 22 October 2026, the Lightning check ran 30 steps at CFG 5 instead of the 6 steps of its speed profile, and took 534 s.
  - **Pinned memory.** The engine keeps the idle Wan expert page-locked in RAM between generations (backend setting `native_park_experts`, on by default). During a 5-second clip Windows' "Shared GPU memory" counter then reaches about 42 GB (21 GB with the setting off), within the 64 GB tier; the container's anonymous memory reads about 21 GiB instead of 34.7 GiB because page-locked memory is not counted there, and the process's RSS is unchanged (about 54 GB at its peak on a 1-second clip). Parking is skipped when both experts would pin more than 60 % of the host's RAM.
- **LTX-2.5 22B on a 12 GB card.** The LTX-2.5 preset on the whole card, Distilled speed profile (8 steps, CFG 1), from a cold start:
  - **Text to video**, 1 s at 480×832 with sound: 5 min 57 s, about 7.5 s per step. This is the first generation of the `ltx25-starter` recipe, whose `smoke.timeout_seconds` is 600.
  - **Layout-To-Render** (an IC-LoRA with 1.22 GB of LoRA deltas, a reference video and a first frame), 3 s (73 frames) at 832×480: 4 min 30 s (270 s), of which 103 s of loading, about 7 s per step, with a 10,702 MiB engine peak and 48,273 MB of process RAM at its peak. At least 36 GB of system RAM must be free. Earlier takes of the same job ran 389 to 475 s, at 11 to 24 s per step, with a 10,715 MiB peak.
  - **What changed.** The DiT placement fix of 20 October 2026: the placement now counts the IC-LoRA and reserves 1.72 GB for activations. Before it, every Layout-To-Render take on this card ran out of GPU memory in the first DiT step.
- **What does not fit on a 12 GB card yet.** LTX-2.3 22B distilled (fp8) runs out of GPU memory inside the DiT, for a 5 s and a 2 s clip.
- **How the figures were taken.** The container's RAM is limited with `--memory` equal to `--memory-swap` (no swap), at the full tier and at that tier minus 3 GB. For the 8 GB presets the engine was told the card has 8 GB (`POTIONUI_VRAM_CAP_GB=7.99`, `gpu_max_vram` 6): a 12 GB card made to report 8 GB, not a physical 8 GB card. The Wan 2.2 14B and LTX-2.5 runs are whole-card runs in Docker Desktop on WSL 2: the engine was told the card's real 12 GB (the backend's `gpu_max_vram` at 10). **Engine VRAM peak** is the high-water mark PotionUI reports for the run (`peak_vram_mb`), not the `nvidia-smi` figure, which also counts the allocator's cache; Windows' "Shared Usage" counter counts pinned memory as well. **Process RAM peak** is the container's anonymous memory (of the finished run); the cgroup's `memory.current` also counts the file cache of the model files.

## Disk space

Budget for the checkpoint files themselves plus their text encoders and VAEs — native-engine families ship DiT, text encoder, and VAE as separate files, all of which sit on disk simultaneously once downloaded:

- SDXL: a few GB per checkpoint.
- Z-Image: ~11.5 GB DiT (bf16) + a Qwen3-4B text encoder.
- Krea-2: 26 GB DiT (bf16) + a Qwen3-VL-4B text encoder + the shared Wan 2.1 causal-3D VAE.
- Flux2/Klein: ~18 GB DiT (bf16 estimate) or a smaller fp8 file + an 8B or 4B Qwen3 text encoder.
- Qwen-Image: ~20 GB DiT (fp8) + a Qwen2.5-VL-7B text encoder.
- Wan 2.1 T2V 1.3B: about 9.8 GB in all — 2.84 GB DiT (fp16) + a 6.74 GB fp8 UMT5-XXL text encoder + a 0.25 GB VAE.
- Wan 2.2 TI2V 5B: about 18.1 GB in all — 10.0 GB DiT (fp16) + the same text encoder + a 1.41 GB VAE (48 channels, not interchangeable with the Wan 2.1 VAE).
- Wan 2.2 14B: **two** DiT files (high/low-noise experts, 14B each) + UMT5-XXL + VAE — the largest native disk footprint short of LTX/MiniMax.
- LTX-2/2.3: ~27 GB all-in-one file (DiT + both VAEs + vocoder + text-embedding projection). LTX-2.5 splits into separate DiT/TE/VAE files instead: the 12 GB runs above used five files, about 40 GB together.
- MiniMax-H3: four separate files (DiT, text encoder, video VAE, audio VAE) — about 40 GB together for the 12 GB profile: a 20.97 GB int8 ConvRot DiT (Comfy-Org; Wan2GP's is 21.06 GB), a 15.69 GB nvfp4 text encoder, a 2.81 GB video VAE and a 0.61 GB audio VAE. MiniMax-H3-Fast swaps in a 22.13 GB DiT (about 41 GB together).
- SeedVR2: 3.39–16.48 GB per DiT variant (see table above) + a ~501 MB VAE.

None of these are downloaded by PotionUI itself unless you ask — see [Models](models.md) for how installation works.

## Three different numbers, three different sources

The per-family table above, a backend's configured VRAM cap, and a per-request memory estimate answer three different questions, and PotionUI never blends them into one verdict:

- **A preset's `requires:` badge** (in the picker, sourced from this page's per-family table and, for the hardware-tier video presets, from the runs above) is static, author-supplied guidance about the *model family* or the preset's memory profile — a VRAM badge and, when the preset sets `min_ram_gb`, a system RAM badge that the picker shows without comparing it to your machine. It says nothing about the specific request you're about to submit (resolution, frame count, which optional components — LoRAs, a second text encoder — you've added).
- **A backend's configured VRAM budget** (its `gpu_max_vram` setting, Admin → Backends) is the ceiling the native engine plans model placement against. It's physical device evidence bounded by an admin-set cap, not a prediction for any one request.
- **A request's memory estimate** (`POST /api/generations/memory-preview`, same payload as `/start` plus an optional `backend_id`) is a non-blocking, **checkpoint-based estimate** for the specific form you're about to submit — never called a lower bound or "at least" anywhere in the response, since runtime use can land on either side of it: it sums the on-disk size of every model reference on an active loader (checkpoint, LoRA, VAE, text encoder alike — never a component a preset pins in its own config outside a form picker, and never a Video/Music Director document's raw wire shape — the same normalized document a real generation would build from, so the active checkpoint set matches), applies a small weight-load margin, and adds a resolution/frame-count-scaled activation term. It never allocates GPU memory, never loads a model, and never enqueues anything — it only inspects the form and the model index.

The estimate is deliberately never a fit guarantee. Its response carries the evidence behind the number rather than a verdict:

- `coverage.known` / `coverage.unknown` — which referenced models had an indexed on-disk size and which didn't; an unresolvable size is skipped, so the estimate only gets more precise, never less, as more models get indexed. When the request itself couldn't be resolved (an in-progress or invalid Video/Music Director document, or any other pipeline-build failure), `coverage.active_set_resolved` is `false` and the estimate reports nothing counted — displayed as "the request could not be resolved," never "no model references."
- `coverage.uncertainty` — plain-language caveats: the activation term is a heuristic, not a measurement; quantization/dtype settings on an active loader may lower real residency below the summed weight sizes; a component a preset pins outside a form picker is never counted.
- `device` — what the routed backend's hardware actually reports, read from that backend's own `resolve_execution_device()` (never guessed from its driver name, and never a bare class tag — a native backend can be configured for `cpu` or any `cuda:N`, so only the specific instance knows which): free/total VRAM from this host's own GPU monitor when the backend is configured for a GPU AND that GPU's own stable hardware identity (a UUID, never an enumeration index or ordinal — neither is proof of the same physical card) actually matches the one the monitor is bound to (`local`); `none` when there's definite evidence of no applicable GPU — a native backend explicitly configured with no GPU (`device="cpu"`), or this host genuinely has no GPU at all; `null` numbers for a remote worker whose hardware isn't this process's to read (`remote`); or `unknown` when either side's identity couldn't be established, the two identities disagree, or the backend hasn't declared where it executes at all (e.g. a `comfyui` backend, whose server has its own hardware this host doesn't see). It is also `unknown` — never a confident `local` or `none` — when an ACTIVE stage in the request's own resolved pipeline pins its own `device` that conflicts with (or isn't a plain literal comparable to) the backend's configured one: a preset's `pipeline.yml` can override the backend's device per-stage, and the backend's own GPU reading cannot honestly be claimed to apply to a stage pinned elsewhere. A matching override, an absent one, or one on a disabled stage changes nothing.
- `budget` — the configured loading budget for the backend as a whole (its `gpu_max_vram`, bounded by the device's reported free VRAM when there is one) — configuration, not a promise about this request. Reported separately in `budget.pipe_hints_gb`: each active pipe's own per-stage `vram_limit_gb` hint (e.g. a tiled detailer's own tile budget), composed against that same backend cap individually — a preset can declare several with different values, so none of them is presented as if it were the whole request's budget.

Reading the estimate against the device/budget evidence is a judgment call for whoever is looking at it — PotionUI reports the numbers, not a pass/fail.

## See also

- [Models](models.md) — the installed-models inventory and how models connect to presets.
- [Administration](admin.md) — backends and where generations actually run.
- Per-family reference docs under `docs/models/` (in the repository) for full architecture and sampling details — this page only covers the memory/disk numbers.
