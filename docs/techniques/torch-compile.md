---
type: technique
title: Regional torch.compile
category_group: Performance
status: needs-gpu-validation
families: [all-native]
authors: []
paper: null
reference_impl: null
knobs:
  - key: native_torch_compile
    surface: admin
    default: "off"
    effect: "Compiles the repeated transformer block of a GPU-resident or streamed DiT, fp8-scaled included, for faster steps after warm-up"
related: [first-block-cache, fp8-matmul]
---

# Regional torch.compile

Compiling an entire diffusion transformer with PyTorch's `torch.compile` normally costs a
multi-minute warm-up the first time it runs, which is a poor fit for an interactive app. PotionUI
instead compiles only the *repeated transformer block* — the same block class that gets stacked N
times to build the network — and reuses one compiled artifact for every repetition. Warm-up becomes
the cost of compiling a single block's graph, not the whole model, while the surrounding control
flow (guidance branching, first-block-cache skip decisions) stays in ordinary eager Python and
never breaks the compiled graph.

Once compiled, later denoising steps that hit the same block shapes run faster because PyTorch's
Inductor backend has fused and optimized the block's operations. The compiled state is reversible:
PotionUI automatically restores the original, uncompiled blocks whenever the model leaves the GPU
(offload, unload, or a move back to CPU), so a model cached in host RAM is always the plain,
uncompiled version.

## When to use it

Best for repeated generations against the same model where the compile warm-up cost on the first
run is amortized over many subsequent steps or requests — long sessions on one preset/model rather
than one-off generations or frequent model switching. It engages when the model is fully
GPU-resident or streamed (partial residency) without stream prefetch; see the limits below.

## How to enable it

Turn on **Compile the model (torch.compile)** in Admin → Backends → *your native backend* →
Optimizations → Speed (setting `native_torch_compile`, off by default). It takes effect on the next
model load, with no restart; a model that is already loaded keeps running uncompiled until it is
loaded again. There is no preset key for this.

The setting is part of each native backend's configuration, so a local backend and each remote
native worker can differ; a remote worker receives its backend's value with every job, and the
worker's own environment never decides it.

The setting replaces the old `NATIVE_TORCH_COMPILE` environment variable, which is no longer read.
On the first boot after the update every existing native backend is seeded once from that variable
if it was set (`auto` is seeded as on); after that the variable is ignored.

## Tradeoffs and limitations

- First-generation cost: the initial forward through a newly compiled block pays the Inductor
  compile time before it becomes faster on subsequent calls.
- What compiles: fully resident DiTs (each block is wrapped), fp8-scaled and int8 tensorwise
  (`Fp8ScaledLinear`, including ConvRot and AWQ pre-scale) DiTs, and streamed DiTs under partial
  residency (each block's `forward` is compiled in place; the streamer's per-leaf cast stays inside
  the graph, so hooks on the block itself keep firing eagerly). The fp8 and streamed block forwards
  trace with zero graph breaks under `torch._dynamo.explain` on CPU.
- Still skipped automatically (eager, no error), with the reason logged at debug level:
  - `runtime-lora`: active cast-mode runtime LoRA deltas rebuild the weight per forward and would
    recompile on every strength change.
  - `nvfp4`: `Nvfp4Linear` layers dequantise through a uint8 lookup-table gather into a full
    float32 weight; not validated under compile on a GPU.
  - `fp8-scaled-mm`: the opt-in native fp8 GEMM (`native_fp8_matmul`, `torch._scaled_mm`) reads
    scale values back to the host and logs per call, which graph-breaks around every linear.
  - `streaming-hooks`: a streamed DiT with stream prefetch on installs forward hooks on every
    streamed leaf that swap `weight.data` per call; a compiled graph cannot follow that.
  - `no-triton`: Triton is not importable (typical on native Windows).
  - `quantized`: any quantisation format other than fp8-scaled.
- A DiT that flips from resident to streaming (co-tenant out-of-memory fallback) or leaves the GPU is
  restored to plain modules first; a streamed DiT is compiled again after the flip.
- Any compile failure at first-forward time degrades gracefully to eager execution (logged, not
  fatal) rather than aborting the generation.
- Applies only to the `native` engine family DiTs; SDXL runs on a separate diffusers-based stack and
  is unaffected.
- Reaches every native family, not just the resident-sample path `NativeGenerator` drives directly
  (Flux, Qwen-Image, Z-Image, Krea-2, Anima, …). The video / bring-your-own-loop families — LTX
  (`generator/txt2vid_ltx`, `generator/video_ltx`, `detailer/video_ltx`), DFR
  (`generator/dfr_video_ltx`), and MiniMax-H3 (`generator/video_minimax_h3`) — drive their own
  sampling loop and place the DiT through `place_dit_for_sequence` (`src/pipelines/pipes/_shared/
  generation/dit_placement.py`), which calls the same gate on every resident placement. Wan 2.1/2.2
  (`generator/txt2vid_wan22`, `generator/img2vid_wan22`, `generator/chain_video_wan22`) place their
  high/low-noise experts through `_ExpertRouter._place_expert`, which offers each expert to the same
  gate independently the moment it lands fully resident. SeedVR2 (`generator/seedvr2`) subclasses
  `NativeGenerator` and calls the inherited `_maybe_compile()` right after its own `_move_dit_to_gpu`,
  same as the image path.
- Steady-state speedup and output numerical drift versus eager have not been benchmarked on real
  hardware — validate wall-clock and image diffs before relying on it for production throughput.
