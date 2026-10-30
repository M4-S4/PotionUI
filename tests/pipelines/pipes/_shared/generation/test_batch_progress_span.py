from typing import Any, Dict, List

import pytest

from src.features.generation.status_tracker import GenerationStatusTracker
from src.pipelines.contracts import IOType, PipeConfigSpec, PipeInput, PipeInputSpec, PipeOutputSpec
from src.pipelines.outputs import ProgressGenerationOutput
from src.pipelines.pipes._shared.generation.generator_base import BaseGeneratorPipe, GeneratorContext
from src.pipelines.pipes.prompt_encoder.main import PromptEncoderPipe

STEPS = 4


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
        for step in range(STEPS):
            progress.step(step + 1, STEPS, state="TXT2IMG")
        return "item"


def _run(config):
    emitted = []
    _Pipe({**_Pipe.get_default_config(), **config}).process(PipeInput(input={}), emitted.append)
    return [o for o in emitted if isinstance(o, ProgressGenerationOutput) and o.progress is not None]


def _tracked(outputs):
    tracker = GenerationStatusTracker()
    tracker.create(id="g")
    return [round(tracker.update_from_output("g", o), 4) for o in outputs]


class TestBatchProgressSpansTheWholeRun:
    def test_the_bar_does_not_sit_at_full_after_the_first_prompt(self):
        shown = _tracked(_run({"quantity": 3, "prompt_batch": {"per_prompt": 1, "count": 3}}))

        first_prompt = shown[:STEPS]
        later = shown[STEPS:]
        assert first_prompt[-1] == pytest.approx(1 / 3, abs=1e-3)
        assert all(value < 1.0 for value in later[:-1])
        assert len(set(later)) == len(later)
        assert later[-1] == 1.0

    def test_progress_rises_strictly_across_every_image_of_a_plain_quantity(self):
        shown = _tracked(_run({"quantity": 3}))

        assert shown == sorted(set(shown))
        assert shown[-1] == 1.0

    def test_each_update_names_its_prompt_and_its_own_fraction(self):
        outputs = _run({"quantity": 4, "prompt_batch": {"per_prompt": 2, "count": 2}})

        assert [o.prompt_index for o in outputs] == [0] * 8 + [1] * 8
        assert {o.prompt_count for o in outputs} == {2}
        assert [o.item_index for o in outputs][::STEPS] == [0, 1, 2, 3]
        assert outputs[0].item_progress == pytest.approx(0.125)
        assert outputs[STEPS - 1].item_progress == pytest.approx(0.5)
        assert outputs[2 * STEPS - 1].item_progress == pytest.approx(1.0)
        assert outputs[2 * STEPS].item_progress == pytest.approx(0.125)

    def test_a_single_image_keeps_its_own_progress_untouched(self):
        outputs = _run({"quantity": 1})

        assert [(o.progress.current, o.progress.max) for o in outputs] == [(i, STEPS) for i in range(1, STEPS + 1)]
        assert all(o.item_index is None and o.prompt_index is None for o in outputs)

    def test_step_counters_stay_per_image(self):
        outputs = _run({"quantity": 2})

        assert [(o.current_step_num, o.total_steps) for o in outputs] == [(i, STEPS) for i in range(1, STEPS + 1)] * 2


class TestEncoderUsesEachPromptsOwnText:
    def test_every_image_gets_its_own_pair_substituted_into_the_template(self):
        pipe = PromptEncoderPipe({**PromptEncoderPipe.get_default_config()})
        pairs = [
            {"positive": "a cat", "negative": "no cat"},
            {"positive": "a dog", "negative": "no dog"},
            {"positive": "an owl", "negative": "no owl"},
        ]

        resolved = [
            pipe._final_prompt_pair("a cat, masterpiece", "a cat", "no cat, blurry", "no cat", pairs, i)
            for i in range(3)
        ]

        assert resolved == [
            ("a cat, masterpiece", "no cat, blurry"),
            ("a dog, masterpiece", "no dog, blurry"),
            ("an owl, masterpiece", "no owl, blurry"),
        ]
