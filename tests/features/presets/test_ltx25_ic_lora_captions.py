from pathlib import Path
from unittest.mock import Mock

import pytest

from src.features.presets import PresetTemplateLoader
from src.features.presets.processor import PresetProcessor
from src.platform.templating.processor import TemplateProcessor

from ._bound_form import bound_form_data

WATER_CARD = (
    "Reference shows the original scene. Edited shows the same scene with water added. "
    "ADD WATER {water}. "
    "Subject identity, clothing, framing, and background geometry are identical to the reference; "
    "only water-related elements differ between reference and edited."
)
COLORIZE_CARD = (
    "Reference shows {gray}. Edited shows the same scene with natural colors restored. "
    "COLORIZE {color}. "
    "Subject identity, framing, and background geometry are identical to the reference; "
    "only color information differs between reference and edited."
)
DEBLUR_CARD = (
    "Reference shows {scene}, heavily out of focus with soft defocused blur and no fine detail. "
    "Edited shows the same scene in sharp focus with crisp detail and clean edges. "
    "DEBLUR {scene}. "
    "Subject identity, framing, and background geometry are identical to the reference; "
    "only focus and sharpness differ between reference and edited."
)
DECOMPRESS_CARD = (
    "Reference shows {scene}, heavily compressed with visible macroblocking, chroma bleed, and ringing artifacts. "
    "Edited shows the same scene restored to high quality with sharp detail, clean edges, and no compression artifacts. "
    "ENHANCE QUALITY {scene}. "
    "Subject identity, framing, and background geometry are identical to the reference; "
    "only compression artifacts and image quality differ between reference and edited."
)

DEFAULT_WATER = "a clear shallow stream of water flowing across the ground and splashing around everything that moves"
SCENE = "A man walks a dog down a street"


def water(text):
    return WATER_CARD.format(water=text or DEFAULT_WATER)


def colorize(text):
    if text:
        return COLORIZE_CARD.format(
            gray=f"{text}, in black and white", color=f"{text}, in natural, true-to-life colors",
        )
    return COLORIZE_CARD.format(gray="a black and white scene", color="the same scene in natural, true-to-life colors")


def deblur(text):
    return DEBLUR_CARD.format(scene=text or "the scene")


def decompress(text):
    return DECOMPRESS_CARD.format(scene=text or "the scene")


WRAPPERS = [
    pytest.param("add_water", {}, water, id="add_water"),
    pytest.param("colorize", {}, colorize, id="colorize"),
    pytest.param("restore", {"restore_task": "deblur"}, deblur, id="deblur"),
    pytest.param("restore", {"restore_task": "decompress"}, decompress, id="decompress"),
]

PROMPTS = [
    pytest.param(f"{SCENE}.", SCENE, id="trailing_period"),
    pytest.param(f"  {SCENE} ...  ", SCENE, id="padded_ellipsis"),
    pytest.param(f"{SCENE}. .", SCENE, id="spaced_periods"),
    pytest.param(SCENE, SCENE, id="no_period"),
    pytest.param("", "", id="empty"),
    pytest.param("  . ", "", id="only_punctuation"),
    pytest.param("a {dog} runs {0}", "a {dog} runs {0}", id="braces_stay_literal"),
]


@pytest.fixture(scope="module")
def ltx25_template():
    loader = PresetTemplateLoader(["content/presets"])
    loader.load_presets()
    template = next((p for p in loader.presets if Path(p.path).as_posix().endswith("marketplace/LTX-2.5")), None)
    if template is None:
        pytest.fail("marketplace/LTX-2.5 preset not present")
    return template


def process(template, mode, prompt, **over):
    form_data = bound_form_data(template, mode, {
        "input_video": "/media/clip.mp4",
        "model": "/models/ltx25_dit.safetensors",
        "text_encoder": "/models/gemma4.safetensors",
        "video_vae": "/models/ltx25_video_vae.safetensors",
        "effect_lora": "/models/effect.safetensors",
        "restore_lora": "/models/restore.safetensors",
        "deblur_lora": "/models/deblur.safetensors",
        "decompress_lora": "/models/decompress.safetensors",
        "manual_sigmas": "",
        **over,
    })
    processor = PresetProcessor(
        template_processor=TemplateProcessor(settings=Mock()), settings=Mock(), preset_template_loader=Mock(),
    )
    pipes = processor.process(template, {
        "prompts": [{"positive": prompt, "negative": ""}], "mode": mode, "form_data": form_data,
    })
    return {pipe.get("id") or pipe["name"]: pipe for pipe in pipes}


@pytest.mark.parametrize("prompt, text", PROMPTS)
@pytest.mark.parametrize("mode, over, card", WRAPPERS)
def test_the_wrapper_is_the_model_card_caption_around_the_users_text(ltx25_template, mode, over, card, prompt, text):
    pipes = process(ltx25_template, mode, prompt, **over)

    assert pipes["prompt_encoder"]["config"]["p_prompt"]["output"] == card(text)


@pytest.mark.parametrize("mode, over, trigger", [
    pytest.param("add_water", {}, "ADD WATER ", id="add_water"),
    pytest.param("colorize", {}, "COLORIZE ", id="colorize"),
    pytest.param("restore", {"restore_task": "deblur"}, "DEBLUR ", id="deblur"),
    pytest.param("restore", {"restore_task": "decompress"}, "ENHANCE QUALITY ", id="decompress"),
])
def test_the_trigger_is_followed_by_the_users_text_and_the_caption_closes_after_it(
    ltx25_template, mode, over, trigger,
):
    pipes = process(ltx25_template, mode, f"{SCENE}.", **over)

    output = pipes["prompt_encoder"]["config"]["p_prompt"]["output"]
    after_trigger = output.split(trigger, 1)[1]
    assert after_trigger.startswith(SCENE)
    assert ".." not in output and ".," not in output
    assert output.endswith("between reference and edited.")
    assert output.count(trigger) == 1


def test_restore_without_a_trigger_passes_the_prompt_through_untouched(ltx25_template):
    pipes = process(ltx25_template, "restore", f"{SCENE}.", restore_task="restore")

    assert pipes["prompt_encoder"]["config"]["p_prompt"]["output"] == f"{SCENE}."


@pytest.mark.parametrize("mode, over", [
    pytest.param("add_water", {}, id="add_water"),
    pytest.param("colorize", {}, id="colorize"),
    pytest.param("restore", {"restore_task": "deblur"}, id="deblur"),
    pytest.param("restore", {"restore_task": "decompress"}, id="decompress"),
])
def test_the_provenance_parameter_keeps_the_users_prompt_not_the_caption(ltx25_template, mode, over):
    pipes = process(ltx25_template, mode, f"{SCENE}.", **over)

    parameters = pipes["param_emitter"]["config"]["parameters"]
    assert ["positive_prompt", [f"{SCENE}."]] in parameters
