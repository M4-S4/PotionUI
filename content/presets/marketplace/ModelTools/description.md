Work on model files you already have. Nothing here runs a diffusion model: every mode streams one file in and writes a new safetensors file out, next to its source, so there is no prompt and no seed.

**Quantize** turns a model or large text encoder into scaled fp8 (the ComfyUI per-module format) and keeps norms, embeddings and other sensitive layers at full precision. **Merge into Model** bakes one or more LoRAs into a model with the same maths the runtime uses, and **Merge and Quantize** does both in one pass, merging in full precision before it quantizes. **Merge LoRAs** combines several LoRAs into one, either exactly (stacked) or reduced to a target rank with a check on how much of the change survives.

The output file lands in the same model folder as its source (merged LoRAs in the LoRA folder), is verified before it is published, and is never allowed to replace an input. Leave the file name empty to get one derived from the inputs, such as `flux-dev-fp8` or `flux-dev+style`; an existing file is only replaced when Overwrite is ticked.

DiT families and large text encoders are supported; SDXL is not. LoKr adapters work; LoHa, DoRA, convolution and text-encoder LoRAs are refused with a message naming the file. This preset is for administrators and runs only on the local native backend.
