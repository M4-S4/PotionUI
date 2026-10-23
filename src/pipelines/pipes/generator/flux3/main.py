from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from src.pipelines.contracts import IOType, PipeInput, PipeInputSpec, PipeOutputSpec, PipeConfigSpec, logger
from src.pipelines.outputs import GenerationExecutionError, ImageGenerationOutput, Icon
from src.pipelines.pipes._shared.generation.flow_generator_pipe import FlowMatchGeneratorPipe
from src.pipelines.pipes._shared.generation.generator_base import GeneratorContext
from src.pipelines.pipes._shared.generation.guidance_options import (
    apply_schedule_settings,
    schedule_settings_config_specs,
)
from src.pipelines.pipes._shared.generation.progress import ProgressEmitter, native_step_hooks
from src.pipelines.pipes._shared.imaging.alpha import flatten_onto
from src.pipelines.pipes.generator.flux3.references import (
    GRID,
    prepare_multi_reference,
    prepare_reference,
)
from src.platform.runtime.native.engine import Conditioning, NativeGenerator
from src.pipelines.pipes._shared.generation.config_specs import device_config_spec, seed_config_spec

_SIZE = re.compile(r"^\s*(\d+)\s*x\s*(\d+)\s*$")


class GeneratorFlux3Pipe(FlowMatchGeneratorPipe):
    name = "generator"
    description = "Native FLUX 3 Image Klein generator (base: true CFG, distilled: flow-map sampling)"
    family_tag = "GENERATOR FLUX 3"

    MAX_REFERENCES = 10

    @classmethod
    def configuration(cls) -> List[PipeConfigSpec]:
        return [
            PipeConfigSpec("mode", str, "txt2img", "Generation mode", required=True, choices=["txt2img", "edit"]),
            PipeConfigSpec("steps", int, 50, "Denoising steps (16 for the distilled checkpoint, 50 for base)",
                           required=False, min_value=1, max_value=100),
            PipeConfigSpec("guidance", float, 3.5, "True CFG scale for the base checkpoint; ignored by the "
                           "distilled one", required=False, min_value=0.0, max_value=30.0),
            PipeConfigSpec("shift", float, None, "Sigma-shift override (multiplicative); blank uses the "
                           "checkpoint's 6.93", required=False),
            PipeConfigSpec("sampler", str, "euler", "Sampler: any key registered on the sampler registry "
                           "(see GET /api/sampling/catalog); the distilled checkpoint requires euler",
                           required=False),
            PipeConfigSpec("resolution", str, "1024x1024", "Resolution (WxH) for txt2img", required=False),
            PipeConfigSpec("output_size", str, "", "Edit output size (WxH); blank matches the first reference",
                           required=False),
            PipeConfigSpec("quantity", int, 1, "Number of images", required=False, min_value=1, max_value=10),
            seed_config_spec(),
            device_config_spec(),
            PipeConfigSpec("preview", bool, True, "Emit live latent previews to the workbench during sampling",
                           required=False),
            *schedule_settings_config_specs(),
            PipeConfigSpec(
                "step_cache", dict, {},
                "FBCache step-skipping options: {'rel_threshold': 0.12, 'warmup_steps': 4, "
                "'max_consecutive_skips': 3}. rel_threshold<=0 is off. Ignored by the distilled checkpoint.",
                required=False,
            ),
            PipeConfigSpec(
                "guidance_options", dict, {},
                "TrueCFG options for the base checkpoint; cfg_zero_star defaults to false here because the "
                "reference pipeline has none.",
                required=False,
            ),
        ]

    @classmethod
    def inputs(cls) -> List[PipeInputSpec]:
        return [
            PipeInputSpec("model", IOType.MODEL, True, "FLUX 3 model bundle", is_array=False),
            PipeInputSpec("conditioning", IOType.CONDITIONING, True, "Encoded prompt conditioning (per image)", is_array=True),
            PipeInputSpec("seed", IOType.SEED, False, "Random seeds", is_array=True),
            cls.img2img_input_spec(),
        ]

    @classmethod
    def outputs(cls) -> List[PipeOutputSpec]:
        return [
            PipeOutputSpec("image", IOType.IMAGE, "Generated images", is_array=True),
        ]

    def build_context(self, pipe_input: PipeInput) -> GeneratorContext:
        ctx = super().build_context(pipe_input)
        apply_schedule_settings(ctx, self.config)
        gen: NativeGenerator = ctx.extra["generator"]
        settings = gen.spec.sampling_settings
        flow_map = settings.get("guidance") == "flow_map"
        ctx.extra["flow_map"] = flow_map
        if flow_map:
            self._configure_distilled(ctx)
        else:
            ctx.extra["guidance_options"] = {"cfg_zero_star": False, **(ctx.extra.get("guidance_options") or {})}
            if ctx.extra["guidance"] <= 1.0:
                logger.warning(
                    "[%s] %s is a base checkpoint but guidance is %.2f: it will render without CFG; "
                    "use the Base speed profile (CFG 3.5)", self.family_tag, gen.spec.variant, ctx.extra["guidance"],
                )
        ctx.extra["output_size"] = self._parse_size(self.config.get("output_size"))
        return ctx

    def _configure_distilled(self, ctx: GeneratorContext) -> None:
        if ctx.extra["sampler"] != "euler":
            raise GenerationExecutionError(
                f"The distilled FLUX 3 checkpoint only samples with Euler, not '{ctx.extra['sampler']}'. "
                "Pick Euler in Advanced, or load the Base checkpoint."
            )
        if ctx.extra["guidance"] != 1.0:
            logger.info("[%s] distilled: guidance ignored", self.family_tag)
        ctx.extra["guidance"] = 1.0
        ctx.extra["step_cache_options"] = None
        ctx.extra["iterate_mode"] = False

    @staticmethod
    def _parse_size(raw: Any) -> Optional[Tuple[int, int]]:
        match = _SIZE.match(str(raw or ""))
        if match is None:
            return None
        width, height = int(match.group(1)), int(match.group(2))
        return (width, height) if width > 0 and height > 0 else None

    def attach_conditioning(self, ctx: GeneratorContext, cond: dict,
                            uncond: Optional[dict]) -> Tuple[dict, Optional[dict]]:
        if ctx.extra.get("flow_map"):
            return cond, None
        return cond, uncond

    def generate_one(self, ctx: GeneratorContext, index: int, seed: int, progress: ProgressEmitter):
        if ctx.extra.get("mode") == "edit":
            return self._generate_edit(ctx, index, seed, progress)
        return super().generate_one(ctx, index, seed, progress)

    def _prepare_edit(self, ctx: GeneratorContext) -> Dict[str, Any]:
        prepared = ctx.extra.get("edit_prepared")
        if prepared is not None:
            return prepared
        images = (ctx.extra.get("images") or [])[:self.MAX_REFERENCES]
        if not images:
            raise GenerationExecutionError("FLUX 3 edit needs at least one reference image.")
        gen: NativeGenerator = ctx.extra["generator"]
        cfg = getattr(gen.dit.module, "cfg", None)
        if cfg is not None and not getattr(cfg, "has_image_cond", True):
            raise GenerationExecutionError("This FLUX 3 checkpoint has no reference-image weights, so it cannot edit.")
        sources = [flatten_onto(image) for image in images]
        for number, source in enumerate(sources, start=1):
            if min(source.size) < GRID:
                raise GenerationExecutionError(
                    f"Reference image {number} is too small ({source.width}x{source.height}); "
                    f"each side needs at least {GRID} px."
                )
        crop = None
        if len(sources) == 1:
            reference, crop = prepare_reference(sources[0])
            references = [reference]
        else:
            references = [prepare_multi_reference(source) for source in sources]
        requested = ctx.extra.get("output_size")
        if requested is not None:
            width, height = gen.snap_resolution(*requested)
            crop = None
        else:
            width, height = references[0].size
        ref_latents = [gen.encode_image(np.asarray(reference)) for reference in references]
        prepared = {"ref_latents": ref_latents, "width": width, "height": height, "crop": crop}
        ctx.extra["edit_prepared"] = prepared
        return prepared

    def _generate_edit(
        self, ctx: GeneratorContext, index: int, seed: int, progress: ProgressEmitter,
    ) -> ImageGenerationOutput:
        gen: NativeGenerator = ctx.extra["generator"]
        conditioning = ctx.extra["conditioning"]
        prepared = self._prepare_edit(ctx)
        width, height = prepared["width"], prepared["height"]

        cond_model = conditioning[index] if index < len(conditioning) else conditioning[-1]
        cond = dict(cond_model.embeds)
        uncond = dict(cond_model.n_embeds) if cond_model.n_embeds else None
        cond, uncond = self.attach_conditioning(ctx, cond, uncond)
        cond["ref_latents"] = prepared["ref_latents"]
        if uncond is not None:
            uncond["ref_latents"] = prepared["ref_latents"]

        def on_progress(_fraction: float, step_index: int, total: int) -> None:
            progress.step(step_index + 1, total, state="EDIT", icon=Icon(name="bolt", effect="pulse"))

        logger.debug("[%s] edit image %d/%d, seed %d", self.family_tag, index + 1, ctx.quantity, seed)
        latent = gen.sample(
            Conditioning(cond=cond, uncond=uncond),
            gen.latent_shape_for(width, height),
            steps=ctx.extra["steps"],
            seed=seed,
            cfg_scale=ctx.extra["guidance"],
            sampler=ctx.extra["sampler"],
            guidance_options=ctx.extra.get("guidance_options"),
            sampler_options=ctx.extra.get("sampler_options"),
            step_cache_options=ctx.extra.get("step_cache_options"),
            schedule_settings=ctx.extra.get("schedule_settings"),
            hooks=native_step_hooks(gen, progress, on_progress, preview=self.config.get("preview", True),
                                    extra=self.extra_step_hooks()),
            is_cancelled=ctx.is_cancelled,
        )
        image = Image.fromarray(gen.decode(latent)[0])
        if prepared["crop"] is not None:
            image = image.crop(prepared["crop"])
        return ImageGenerationOutput(
            image=image, temporary=True, seed=seed,
            resolution=image.size, cfg=ctx.extra["guidance"], step=ctx.extra["steps"],
        )
