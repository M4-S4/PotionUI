from typing import List

import numpy as np
from PIL import Image

from src.pipelines.contracts import (
    BasePipe,
    IOType,
    PipeConfigSpec,
    PipeInput,
    PipeInputSpec,
    PipeOutput,
    PipeOutputSpec,
)
from src.pipelines.pipes._shared.imaging.io import as_image_list
from src.pipelines.pipes._shared.inpainting.mask_utils import composite_region
from src.platform.runtime.tensors import numpy_to_pil


class SubjectCompositePipe(BasePipe):
    name = "subject_composite"
    description = "Paste the original subject back over an edited image through its matte, so the subject's pixels stay identical"

    def process(self, pipe_input: PipeInput, generation_outputs: callable) -> PipeOutput:
        generated = as_image_list(pipe_input.input.get("image"), self.name)
        cutout = as_image_list(pipe_input.input.get("cutout"), self.name)[0].convert("RGBA")
        feather = float(self.config.get("feather", 0.01))
        threshold = int(self.config.get("threshold", 128))

        original = np.array(cutout.convert("RGB"))
        matte = np.array(cutout.getchannel("A"))
        mask = np.where(matte >= threshold, 255, 0).astype(np.uint8)
        height, width = original.shape[:2]

        results = []
        for item in generated:
            rgb = item.convert("RGB")
            if rgb.size != (width, height):
                rgb = rgb.resize((width, height), Image.LANCZOS)
            results.append(numpy_to_pil(composite_region(np.array(rgb), original, mask, feather, False)))
        return PipeOutput(output={"image": results})

    @classmethod
    def inputs(cls) -> List[PipeInputSpec]:
        return [
            PipeInputSpec("image", IOType.IMAGE, True, "Edited image(s) the subject is pasted over", is_array=True),
            PipeInputSpec("cutout", IOType.IMAGE, True, "RGBA cutout of the original image, alpha = subject matte", is_array=True),
        ]

    @classmethod
    def outputs(cls) -> List[PipeOutputSpec]:
        return [
            PipeOutputSpec("image", IOType.IMAGE, "Edited image(s) at the original size with the original subject restored", is_array=True),
        ]

    @classmethod
    def configuration(cls) -> List[PipeConfigSpec]:
        return [
            PipeConfigSpec(
                name="feather",
                param_type=float,
                default=0.01,
                description="Feather outside the subject edge as a fraction of the subject's longer side",
                required=False,
                min_value=0.0,
                max_value=1.0,
            ),
            PipeConfigSpec(
                name="threshold",
                param_type=int,
                default=128,
                description="Matte value from which a pixel counts as subject",
                required=False,
                min_value=1,
                max_value=255,
            ),
        ]
