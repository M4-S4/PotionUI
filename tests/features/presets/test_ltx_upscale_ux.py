"""Tests for the native LTX-2 preset's upscale-refine UX (UX follow-up: the
raw refine-sigmas schedule left the user with too much low-level detail to
set correctly).

The raw ``upscale_refine_sigmas`` / ``refine_sigmas`` textbox (a
literal, comma-separated sigma schedule) originally landed as the only way to touch the stage-2
refine pass. This is a UX fix, not a behavior change:

- The raw textbox becomes an ``audience: advanced`` expert override, empty by
  default -- it disappears from the normal (simple-audience) form entirely,
  mirroring SeedVR2's Advanced-tab idiom (``presets/marketplace/SeedVR2/modes/
  upscale/tabs/advanced.yml``).
- A friendly "Refine Strength" select (Generation tab, both the in-flow
  upscale option and the standalone upscale mode) picks a SUFFIX of
  Lightricks' own stage-2 recipe (``preset.vars.ltx23_stage2_sigma_recipe``):
  Strong = the full recipe (recommended, default, byte-identical output to
  before this change), Balanced/Light start later in the schedule (less noise
  re-injected -- more faithful to the upsampled latent).
- The explicit Advanced-tab override still wins over the select when non-empty
  (unchanged `or` precedence).

The standalone ``txt2vid`` MODE was retired (the ``video``/Director
mode already covers plain t2v -- a Director document with no keyframes) and
ported its in-flow upscale option onto ``video``, reusing ``generator/
video_ltx`` for both stages instead of ``generator/txt2vid_ltx``. The
"-- txt2vid mode (in-flow upscale option) --" section below was rewritten
against the ``video`` mode; the standalone upscale mode (unaffected by the
mode removal) still uses ``generator/txt2vid_ltx`` for its own refine pass.

Complements ``test_ltx_speed_profiles.py`` (Speed selector, a separate field).
"""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from src.features.presets import PresetTemplateLoader
from src.features.presets.processor import PresetProcessor
from src.platform.templating.processor import TemplateProcessor
from src.features.video_director.normalize import derive_ltx_media_fields

from ._bound_form import bound_form_data
from tests.fixtures.form_lookup import field_by_name

STRONG_RECIPE = "0.909375, 0.725, 0.421875, 0.0"
BALANCED_RECIPE = "0.725, 0.421875, 0.0"
LIGHT_RECIPE = "0.421875, 0.0"

PRESET_DIR = Path("content/presets/marketplace/LTX-2")
UPSCALE_ADVANCED = PRESET_DIR / "modes/upscale/tabs/advanced.yml"

_DOC_T2V = {
    "settings": {"seed": 123, "duration": 5, "fps": 25},
    "segments": [{"prompt": "a cat", "negative_prompt": "ugly"}],
    "media": [], "audio": [], "ic_lora": [],
}


@pytest.fixture(scope="module")
def ltx_template():
    loader = PresetTemplateLoader(["content/presets"])
    loader.load_presets()
    template = next((p for p in loader.presets if Path(p.path).as_posix().endswith("/LTX-2")), None)
    if template is None:
        pytest.fail("LTX-2 preset not present")
    return template


def _pipe(pipes, pid):
    return next(p for p in pipes if p.get("id") == pid or p["name"] == pid)


def _process(ltx_template, mode, form_over=None, prompts=None):
    processor = PresetProcessor(
        template_processor=TemplateProcessor(settings=Mock()),
        settings=Mock(),
        preset_template_loader=Mock(),
    )
    form_data = {
        "model": "/models/ltx.safetensors",
        "text_encoder": "/models/gemma3.safetensors",
        "resolution": "768x512",
    }
    if mode == "video":
        doc = copy.deepcopy(_DOC_T2V)
        doc.update(derive_ltx_media_fields(doc["media"], doc["ic_lora"], doc["settings"].get("fps")))
        form_data["video_director"] = doc
        form_data["upscale_refine_sigmas"] = ""
    if mode == "upscale":
        form_data.update({
            "input_video": "/media/in.mp4",
            "upscale_model": "/models/upscaler.safetensors",
            "refine_sigmas": "",
        })
    if form_over:
        form_data.update(form_over)
    form_data = bound_form_data(ltx_template, mode, form_data)
    generation_data = {
        "prompts": prompts if prompts is not None else [], "mode": mode,
        "form_data": form_data,
    }
    return processor.process(ltx_template, generation_data)


# -- video mode (in-flow upscale option) -----------------------

def test_video_default_upscale_untouched_is_byte_identical_to_legacy(ltx_template):
    # No touches beyond turning Upscale on: must render the exact same full
    # recipe that shipped -- the UX change must not alter default output.
    pipes = _process(ltx_template, "video", {"upscale": "1.5x", "upscale_model": "/models/upscaler.safetensors"})
    gen = _pipe(pipes, "generator_stage2")
    assert gen["config"]["refine_sigmas"] == STRONG_RECIPE


@pytest.mark.parametrize(
    "strength,expected",
    [("strong", STRONG_RECIPE), ("balanced", BALANCED_RECIPE), ("light", LIGHT_RECIPE)],
)
def test_video_refine_strength_maps_to_sigma_suffix(ltx_template, strength, expected):
    pipes = _process(ltx_template, "video", {
        "upscale": "1.5x", "upscale_model": "/models/upscaler.safetensors", "refine_strength": strength,
    })
    gen = _pipe(pipes, "generator_stage2")
    assert gen["config"]["refine_sigmas"] == expected


def test_video_advanced_override_still_wins_over_select(ltx_template):
    pipes = _process(ltx_template, "video", {
        "upscale": "1.5x", "upscale_model": "/models/upscaler.safetensors", "refine_strength": "light",
        "upscale_refine_sigmas": "0.5, 0.25, 0.0",
    })
    gen = _pipe(pipes, "generator_stage2")
    assert gen["config"]["refine_sigmas"] == "0.5, 0.25, 0.0"


def test_video_upscale_honored_when_generate_audio_on(ltx_template):
    # Upscale and Generate Audio now compose -- stage 1
    # skips the VIDEO decode (raw latent to latent_upscaler) exactly as
    # without audio, while its independent audio decode still runs and
    # hands off to stage 2 for mux (see video/pipeline.yml's "audio +
    # upscale interaction" header comment).
    pipes = _process(ltx_template, "video", {
        "upscale": "1.5x", "upscale_model": "/models/upscaler.safetensors", "generate_audio": True,
    })
    stage1 = _pipe(pipes, "generator_stage1")
    assert stage1["config"]["decode"] is False
    # Stage 1 always renders at the picked Resolution; Upscale scales stage 2
    # UP from here (see video/pipeline.yml's header).
    assert stage1["config"]["resolution"] == "768x512"
    assert stage1["config"]["audio"] is True
    assert stage1["config"]["audio_source"] == "generate"
    assert _pipe(pipes, "latent_upscaler")["enabled"] is True
    stage2 = _pipe(pipes, "generator_stage2")
    assert stage2["enabled"] is True
    assert stage2["config"]["resolution"] == "1152x768"  # 768*1.5, 512*1.5
    assert stage2["config"]["audio"] is True
    assert stage2["config"]["audio_source"] == "passthrough"
    audio_input = next(i for i in stage2["input"] if i["name"] == "audio")
    assert audio_input["provider"] == "generator_stage1"
    gallery = _pipe(pipes, "gallery")
    video_input = next(i for i in gallery["input"] if i["name"] == "video")
    assert video_input["provider"] == "generator_stage2"


def test_video_upscale_honored_when_director_audio_clip_present(ltx_template):
    # Same, exercised via the Director-timeline audio-clip path (user file,
    # not the Generate Audio toggle) -- both audio sources compose with
    # Upscale identically.
    doc = copy.deepcopy(_DOC_T2V)
    doc["audio"] = [{"media": {"path": "/media/user_audio.wav"}}]
    doc.update(derive_ltx_media_fields(doc["media"], doc["ic_lora"], doc["settings"].get("fps")))
    pipes = _process(ltx_template, "video", {
        "upscale": "1.5x", "upscale_model": "/models/upscaler.safetensors", "video_director": doc,
    })
    stage1 = _pipe(pipes, "generator_stage1")
    assert stage1["config"]["decode"] is False
    assert stage1["config"]["audio_source"] == "file"
    assert _pipe(pipes, "latent_upscaler")["enabled"] is True
    stage2 = _pipe(pipes, "generator_stage2")
    assert stage2["enabled"] is True
    assert stage2["config"]["audio_source"] == "passthrough"


def test_video_upscale_off_decode_and_enabled_are_native_bools(ltx_template):
    # Regression guard: `enabled:` must be an exact
    # `{{ expression }}` and `decode` must not be `{% set %}`-prefixed --
    # either mistake string-renders "True"/"False" instead of a native bool,
    # and Python's `bool("False")` is truthy, silently defeating the gate.
    pipes = _process(ltx_template, "video")
    stage1 = _pipe(pipes, "generator_stage1")
    assert stage1["config"]["decode"] is True
    assert isinstance(stage1["config"]["decode"], bool)
    assert _pipe(pipes, "latent_upscaler")["enabled"] is False
    assert isinstance(_pipe(pipes, "latent_upscaler")["enabled"], bool)
    assert _pipe(pipes, "generator_stage2")["enabled"] is False
    assert isinstance(_pipe(pipes, "generator_stage2")["enabled"], bool)


# -- standalone upscale mode ---------------------------------------------------

def test_upscale_mode_default_untouched_is_byte_identical_to_legacy(ltx_template):
    pipes = _process(ltx_template, "upscale")
    gen = _pipe(pipes, "generator_refine")
    assert gen["config"]["refine_sigmas"] == STRONG_RECIPE


@pytest.mark.parametrize(
    "strength,expected",
    [("strong", STRONG_RECIPE), ("balanced", BALANCED_RECIPE), ("light", LIGHT_RECIPE)],
)
def test_upscale_mode_refine_strength_maps_to_sigma_suffix(ltx_template, strength, expected):
    pipes = _process(ltx_template, "upscale", {"refine_strength": strength})
    gen = _pipe(pipes, "generator_refine")
    assert gen["config"]["refine_sigmas"] == expected


def test_upscale_mode_advanced_override_still_wins_over_select(ltx_template):
    pipes = _process(ltx_template, "upscale", {
        "refine_strength": "light", "refine_sigmas": "0.5, 0.25, 0.0",
    })
    gen = _pipe(pipes, "generator_refine")
    assert gen["config"]["refine_sigmas"] == "0.5, 0.25, 0.0"


# -- form-definition UX assertions (the raw field is advanced-only) -----------

@pytest.mark.parametrize(
    "tab_path,field_name",
    [
        (PRESET_DIR / "modes/video/tabs/advanced.yml", "upscale_refine_sigmas"),
        (UPSCALE_ADVANCED, "refine_sigmas"),
    ],
)
def test_raw_sigma_field_is_advanced_audience_with_empty_default(tab_path, field_name):
    data = yaml.safe_load(tab_path.read_text())
    field = field_by_name(data["fields"], field_name)
    assert field is not None, f"{field_name} not found in {tab_path}"
    assert field.get("audience") == "advanced"
    assert field.get("default") == ""


@pytest.mark.parametrize(
    "tab_path",
    [
        PRESET_DIR / "modes/video/tabs/enhance.yml",
        PRESET_DIR / "modes/upscale/tabs/generation.yml",
    ],
)
def test_refine_strength_select_exists_with_strong_default_and_three_options(tab_path):
    data = yaml.safe_load(tab_path.read_text())
    field = field_by_name(data["fields"], "refine_strength")
    assert field is not None, f"refine_strength not found in {tab_path}"
    assert field["type"] == "select"
    assert field["default"] == "strong"
    values = {opt["value"] for opt in field["configuration"]["options"]}
    assert values == {"strong", "balanced", "light"}


def test_video_refine_strength_hidden_when_upscale_off():
    tab_path = PRESET_DIR / "modes/video/tabs/enhance.yml"
    data = yaml.safe_load(tab_path.read_text())
    field = field_by_name(data["fields"], "refine_strength")
    reactions = field.get("reactions", [])
    off_reaction = next(r for r in reactions if r["when"] == {"field": "upscale", "equals": "off"})
    assert off_reaction["then"] == {"set_visibility": False}
    on_reaction = next(r for r in reactions if r["when"] == {"field": "upscale", "not_equals": "off"})
    assert on_reaction["then"] == {"set_visibility": True}


def test_video_upscale_has_no_generate_audio_interlock():
    # Upscale and Generate Audio compose (see
    # test_video_upscale_honored_when_generate_audio_on) -- the field must no
    # longer carry a `generate_audio` reaction that force-disables it.
    tab_path = PRESET_DIR / "modes/video/tabs/enhance.yml"
    data = yaml.safe_load(tab_path.read_text())
    field = field_by_name(data["fields"], "upscale")
    reactions = field.get("reactions", [])
    assert not any(r["when"].get("field") == "generate_audio" for r in reactions)


# -- restoration-intent quality prompting: guide LTX 2.3's upscale pass with a
# quality-suffix prompt, similar in spirit to a quality embedding -----------
#
# Only the standalone upscale mode's dedicated prompt_encoder gets the
# quality suffix -- the video mode's in-flow stage-2 refine reuses stage 1's
# already-encoded conditioning verbatim (see video/pipeline.yml's stage-2
# comment) and deliberately does NOT get a second encode just for this.

QUALITY_PROMPT = (
    "the video is sharp and richly detailed, with clean well-defined edges, "
    "natural skin and material texture, smooth coherent motion, and free of "
    "compression artifacts, noise, or blurring"
)


def test_upscale_mode_quality_prompt_appended_to_user_prompt(ltx_template):
    pipes = _process(
        ltx_template, "upscale",
        prompts=[{"positive": "a cat walking in the rain", "negative": ""}],
    )
    enc = _pipe(pipes, "prompt_encoder")
    assert enc["config"]["p_prompt"]["output"] == f"a cat walking in the rain, {QUALITY_PROMPT}"


def test_upscale_mode_quality_prompt_alone_when_prompt_empty(ltx_template):
    # No prompt submitted -- generation.prompts.first.positive is '' (docs/
    # presets.md), not undefined; the quality prompt is used alone rather
    # than sending an empty string to the text encoder.
    pipes = _process(ltx_template, "upscale", prompts=[])
    enc = _pipe(pipes, "prompt_encoder")
    assert enc["config"]["p_prompt"]["output"] == QUALITY_PROMPT


def test_upscale_mode_quality_prompt_disabled_keeps_user_prompt_untouched(ltx_template):
    pipes = _process(
        ltx_template, "upscale",
        form_over={"apply_quality_prompt": False},
        prompts=[{"positive": "a cat walking in the rain", "negative": ""}],
    )
    enc = _pipe(pipes, "prompt_encoder")
    assert enc["config"]["p_prompt"]["output"] == "a cat walking in the rain"


def test_upscale_mode_quality_prompt_disabled_with_empty_prompt_sends_empty_string(ltx_template):
    pipes = _process(ltx_template, "upscale", form_over={"apply_quality_prompt": False}, prompts=[])
    enc = _pipe(pipes, "prompt_encoder")
    assert enc["config"]["p_prompt"]["output"] == ""


def test_upscale_mode_apply_quality_prompt_field_is_advanced_checkbox_default_true():
    tab_path = UPSCALE_ADVANCED
    data = yaml.safe_load(tab_path.read_text())
    field = field_by_name(data["fields"], "apply_quality_prompt")
    assert field is not None
    assert field["type"] == "checkbox"
    assert field.get("audience") == "advanced"
    assert field.get("default") is True


def test_video_stage2_reuses_stage1_conditioning_not_a_second_encode(ltx_template):
    # Documents the finding: there is exactly ONE prompt_encoder pipe in the
    # video pipeline, and both generator_stage1 and generator_stage2 read its
    # `conditioning` output -- so the quality-prompt suffix cannot be applied
    # to stage 2 alone without either mutating stage 1's prompt too or paying
    # for a second Gemma3-12B encode. If this test ever breaks because a
    # second prompt_encoder was added, the stage-2 comment in
    # video/pipeline.yml (and this quality-prompt design note) need
    # revisiting too.
    pipes = _process(ltx_template, "video", {"upscale": "1.5x", "upscale_model": "/models/upscaler.safetensors"})
    encoders = [p for p in pipes if p["name"] == "prompt_encoder"]
    assert len(encoders) == 1

    stage1 = _pipe(pipes, "generator_stage1")
    stage2 = _pipe(pipes, "generator_stage2")
    stage1_conditioning_source = next(i for i in stage1["input"] if i["name"] == "conditioning")["provider"]
    stage2_conditioning_source = next(i for i in stage2["input"] if i["name"] == "conditioning")["provider"]
    assert stage1_conditioning_source == stage2_conditioning_source == "prompt_encoder"


def test_upscale_mode_refine_profile_runs_the_recipe(ltx_template):
    pipes = _process(ltx_template, "upscale", {"speed_profile": "refine"})
    gen = _pipe(pipes, "generator_refine")["config"]
    assert gen["cfg"] == 1.0
    assert gen["sampler"] == "euler"
    encoder = _pipe(pipes, "prompt_encoder")["config"]
    assert encoder["guidance_scale"] == 1.0
    assert encoder["negative_applied"] is False


def test_upscale_mode_custom_profile_reaches_refine_pass(ltx_template):
    pipes = _process(ltx_template, "upscale", {"speed_profile": "custom", "cfg": 3.0, "sampler": "euler_ancestral"})
    gen = _pipe(pipes, "generator_refine")["config"]
    assert gen["cfg"] == 3.0
    assert isinstance(gen["cfg"], float)
    assert gen["sampler"] == "euler_ancestral"
    encoder = _pipe(pipes, "prompt_encoder")["config"]
    assert encoder["guidance_scale"] == 3.0
    assert encoder["negative_applied"] is True
    params = dict(p for p in _pipe(pipes, "param_emitter")["config"]["parameters"] if isinstance(p, list) and len(p) == 2)
    assert params["cfg"] == 3.0
    assert params["sampler"] == "euler_ancestral"


def test_video_refine_cfg_and_sampler_reach_stage2_only(ltx_template):
    pipes = _process(ltx_template, "video", {
        "upscale": "2.0x", "upscale_model": "/models/upscaler.safetensors",
        "speed_profile": "distilled", "refine_cfg": 2.5, "refine_sampler": "euler_ancestral",
    })
    stage1 = _pipe(pipes, "generator_stage1")["config"]
    stage2 = _pipe(pipes, "generator_stage2")["config"]
    assert (stage1["cfg"], stage1["sampler"]) == (1.0, "euler_cfg_pp")
    assert (stage2["cfg"], stage2["sampler"]) == (2.5, "euler_ancestral")
    encoder = _pipe(pipes, "prompt_encoder")["config"]
    assert encoder["guidance_scale"] == 2.5
    assert encoder["negative_applied"] is True


def test_video_refine_cfg_is_ignored_while_upscale_is_off(ltx_template):
    pipes = _process(ltx_template, "video", {"speed_profile": "distilled", "refine_cfg": 6.0})
    encoder = _pipe(pipes, "prompt_encoder")["config"]
    assert encoder["guidance_scale"] == 1.0
    assert encoder["negative_applied"] is False
