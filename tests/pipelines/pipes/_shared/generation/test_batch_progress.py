from typing import Any, Dict, List

from src.pipelines.contracts import IOType, PipeConfigSpec, PipeInput, PipeInputSpec, PipeOutputSpec
from src.pipelines.outputs import ProgressGenerationOutput
from src.pipelines.pipes._shared.generation.batch_layout import PromptBatchLayout
from src.pipelines.pipes._shared.generation.generator_base import BaseGeneratorPipe, GeneratorContext
from src.pipelines.pipes.prompt_encoder.main import PromptEncoderPipe
from src.platform.runtime.primitives.clip import ConditioningModel


class _Pipe(BaseGeneratorPipe):
    name = "test_generator"
    description = "test"

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {"quantity": 1, "seed": -1}

    @classmethod
    def inputs(cls) -> List[PipeInputSpec]:
        return []

    @classmethod
    def outputs(cls) -> List[PipeOutputSpec]:
        return [PipeOutputSpec("image", IOType.IMAGE, "", is_array=True)]

    @classmethod
    def configuration(cls) -> List[PipeConfigSpec]:
        return []

    def build_context(self, pipe_input: PipeInput) -> GeneratorContext:
        return GeneratorContext(quantity=int(self.config["quantity"]), input_seeds=None)

    def generate_one(self, ctx, index, seed, progress):
        return "item"


def _states(config):
    emitted = []
    _Pipe({**_Pipe.get_default_config(), **config}).process(PipeInput(input={}), emitted.append)
    return [o.state for o in emitted if isinstance(o, ProgressGenerationOutput)]


class TestGeneratorStatus:
    def test_a_batch_names_the_image_and_the_prompt(self):
        states = _states({"quantity": 6, "prompt_batch": {"per_prompt": 2, "count": 3}})

        assert states == [
            "Generating image 1 of 6 · prompt 1 of 3",
            "Generating image 2 of 6 · prompt 1 of 3",
            "Generating image 3 of 6 · prompt 2 of 3",
            "Generating image 4 of 6 · prompt 2 of 3",
            "Generating image 5 of 6 · prompt 3 of 3",
            "Generating image 6 of 6 · prompt 3 of 3",
        ]

    def test_a_plain_run_emits_no_status_text(self):
        assert _states({"quantity": 3}) == []

    def test_a_malformed_marker_is_ignored(self):
        assert _states({"quantity": 2, "prompt_batch": {"per_prompt": 0, "count": 3}}) == []
        assert _states({"quantity": 2, "prompt_batch": "yes"}) == []

    def test_layout_needs_more_than_one_prompt(self):
        assert PromptBatchLayout.from_config({"prompt_batch": {"per_prompt": 2, "count": 1}}) is None


class _Clip:
    _model_fingerprint = "fp"

    def encode_prompts(self, requests):
        return [ConditioningModel(p_prompt=r["prompt"], n_prompt=r["negative_prompt"], embeds={}, n_embeds={}) for r in requests]


def _encoder_state(**over):
    cfg = PromptEncoderPipe.get_default_config()
    cfg.update({"p_prompt": {"input": "a cat", "output": "a cat"}, "n_prompt": {"input": "", "output": ""}})
    cfg.update(over)
    emitted = []
    PromptEncoderPipe(config=cfg).process(PipeInput(input={"text_encoder": _Clip()}), emitted.append)
    return [o.state for o in emitted if isinstance(o, ProgressGenerationOutput)]


class TestEncoderStatus:
    def test_a_batch_reads_its_prompts(self):
        assert _encoder_state(quantity=6, prompt_batch={"per_prompt": 2, "count": 3}) == ["Reading 3 prompts"]

    def test_a_plain_run_keeps_the_original_text(self):
        assert _encoder_state(quantity=2) == ["Encoding prompts"]
