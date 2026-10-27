---
category: Presets / Models
category_order: 70
order: 30
---

# Model Tools

The **Model Tools** preset (`content/presets/marketplace/ModelTools/`) turns files already in the
model depot into new ones: it quantizes a model to fp8, int8, nvfp4 or int4, bakes LoRAs into a model, or merges
LoRAs into one. It has no prompt and no seed. One pipe, `model_tools`
(`src/pipelines/pipes/model_tools/`), does the work on top of the streaming primitives in
`src/platform/runtime/native/ops/model_tools/`, and the result appears as a model file card.

## Modes

| Mode | Fields | What it does |
|---|---|---|
| `quantize` | model, format, extra keep patterns | Streams the model into the chosen format; norms, embeddings, modulation and the input and output layers stay in full precision. |
| `merge_into_model` | model, LoRAs with strengths | Adds the LoRA deltas to the weights with the runtime's own key maps and delta maths. An fp8 source is expanded to bf16 first. |
| `merge_and_quantize` | model, LoRAs, format, keep patterns | Merges in full precision, then quantizes. Never merge into fp8 storage: the delta is smaller than an fp8 step and would be rounded away. |
| `merge_loras` | LoRAs, method, target rank, minimum energy, precision | Combines LoRAs into one. `concat` is exact (ranks add up); `reduce` compresses to the target rank and is refused when a layer keeps less than the minimum energy. |

DiT families and large text encoders are supported (quantize); bake and merge cover the families the
LoRA key maps know. SDXL is not supported. LoKr adapters work; LoHa, DoRA, convolution and
text-encoder LoRAs are refused with a message naming the file.

## Formats

Every format is written in the ComfyUI per-layer layout: each quantized Linear gets its codes in
`<layer>.weight`, its scales in `<layer>.weight_scale` (and `weight_scale_2` for nvfp4), and a
`<layer>.comfy_quant` JSON descriptor. The native engine loads all of them; the fast paths are the
`native_fp8_matmul` and `native_nvfp4_matmul` runtime settings, both off by default.

| Format | Descriptor | Bytes per weight | In this engine |
|---|---|---|---|
| `fp8_e4m3` (default) | `float8_e4m3fn`, one scale per tensor | 1 | Dequantized per step; `_scaled_mm` on RTX 40/50 with fp8 matmul on |
| `fp8_e5m2` | `float8_e5m2`, one scale per tensor | 1 | Always dequantized; coarser than e4m3, no speedup |
| `int8` | `int8_tensorwise`, one scale per output row | 1 | Always dequantized |
| `int8_convrot` | `int8_tensorwise` + ConvRot (Hadamard, group 256) | 1 | Always dequantized; the rotation is what ComfyUI's int8 kernels use |
| `nvfp4` | `nvfp4`, fp8 block scale per 16 + tensor scale | about 0.56 | LUT dequant per step; fp4 matmul on RTX 50 with nvfp4 matmul on |
| `int4_convrot` | `convrot_w4a4`, group 256, one scale per row | about 0.5 | Unpacked to int8 codes at load and dequantized like int8 ConvRot, so it uses int8 memory; the result card flags its quality as unverified until a GPU check |

The ConvRot writers rotate with the loader's own Hadamard (`build_convrot_hadamard` in
`src/platform/runtime/native/io/quant_formats.py`). The int4 layout (two's-complement nibbles, even column in the low
nibble, codes in [-7, 7], one F32 scale per row) follows the published ComfyUI / comfy-kitchen
`convrot_w4a4` format; no code was taken from comfy-kitchen (provenance in `vendor/NOTICE.md`).
The safetensors loader expands int4 layers to int8 ConvRot as it reads the file, so model detection
and memory estimates see the true layer width.

Text encoders take `fp8_e4m3`, `fp8_e5m2` or `nvfp4` only; the integer formats are refused for them,
and the chat provider never adopts an integer-coded text encoder. A layer whose shape a format cannot
hold stays in its source dtype: nvfp4 needs both dimensions divisible by 16, the ConvRot formats need
in-features divisible by 256. The result card reports how many layers that kept (`layers_shape_kept`).

## Who can run it, and where

- **Administrators only.** The pipe refuses unless its `is_admin` option is exactly `true`. The
  preset sets it from `{{ runtime.user.is_admin }}`, which the server resolves from the signed-in
  account; nothing in the form can set it. The preset is also never assigned to ordinary users, which
  is how presets are restricted today (users only see presets assigned to them).
- **Local native backend only.** The pipe needs the `MODELS` service's model roots to locate and
  contain every path. A remote worker has no roots, so it refuses; a ComfyUI server cannot be selected
  for a `native` preset at all.

## Where the output goes

The file is written next to its source, so it lands in the same model folder and type as the source;
merged LoRAs go to the write-enabled LoRA folder. Every path is checked with `is_within` against the
type directory it must stay inside (and again after resolving symbolic links), a read-only root is
refused, and the output can never replace one of its own inputs.

The **Output File Name** may stay empty: the default is `<name><format>`, `<name>+<lora>`,
`<name>+<lora><format>` or `<lora>+<lora>`, where `<format>` is `-fp8`, `-fp8-e5m2`, `-int8`,
`-int8-convrot`, `-nvfp4` or `-int4-convrot`, sanitized and shortened with a stable digest when long. A name
typed by the user is validated, not rewritten: path separators, `: * ? " < > |`, control characters,
leading dots, `..`, Windows reserved names and anything over 150 characters are refused. An existing
file is only replaced when **Overwrite** is ticked.

## Safety and progress

1. The source headers and the LoRA files are read and planned first; unsupported input is refused
   before any byte is written.
2. The planned output size plus a reserve of `max(2 GiB, 10% of the drive)` must fit in the free space of
   the destination folder, otherwise the run is refused with the needed size, the free space, the reserve
   and how much to free. The reserve keeps the drive from filling up completely, where consumer SSDs
   stall. Overwrite does not credit the old file, because it stays until the new one is renamed over it.
3. The sources are hashed (SHA-256) so the provenance can be stored in the file header, then the output
   is streamed tensor by tensor to a hidden staging file with per-tensor progress. The writer syncs to disk
   every 256 MiB (and drops the synced pages from the cache on Linux), so dirty data stays bounded instead
   of being flushed in one burst at the end.
4. The staged file is verified (`verify_quantized` on sampled layers for quantize, each restored through its
   format's own dequantization; the quantization error
   against the merged fp32 weights for merge and quantize; changed, finite sampled layers for a bake;
   the written tensors and kept energy for a LoRA merge) and only then renamed over the final name. A
   failed check, an error or a cancel removes the staging file, so a failed overwrite keeps the old file.

Cancelling stops the stream at the next tensor.

## The result card

The pipe emits a `ModelFileGenerationOutput`: `kind` (`model` or `lora`), `model_type`, sizes, `format`
(the quantization format, the merged dtype, or the LoRA dtype) and a `summary` with the layers quantized and kept,
the layers kept because of their shape,
the worst layers, the kept energy, the rank, the sources with their SHA-256 and strengths, the
verification and a licence note taken from the source metadata when present. A merged file is a
derivative of its inputs: check their licences before redistributing it.

## Model field options

The preset's pickers use two options of the `model` field: `model_types` (several types in one list) and
`unquantized_only` (hide files that are already quantized). See [Form fields](presets/forms.md).
