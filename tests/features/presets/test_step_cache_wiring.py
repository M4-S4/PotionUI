from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pytest
import yaml

from tests.fixtures.form_lookup import field_by_name

from .wiring import find_template, pipe, render

KEYS = ("step_cache_threshold", "step_cache_warmup_steps", "step_cache_max_skips")

CACHE_ON = {
    "step_cache_threshold": 0.12,
    "step_cache_warmup_steps": 6,
    "step_cache_max_skips": 2,
}

CACHE_ON_OPTIONS = {"rel_threshold": 0.12, "warmup_steps": 6, "max_consecutive_skips": 2}

H3_PRESET_ID = "01KX47H3MINIMAX000000000VA"


def _flat_values(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "rel_threshold": cfg["step_cache_threshold"],
        "warmup_steps": cfg["step_cache_warmup_steps"],
        "max_consecutive_skips": cfg["step_cache_max_skips"],
    }


def _dict_values(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(cfg["step_cache"])


def _guidance_enabled(cfg: Dict[str, Any]) -> bool:
    from src.pipelines.pipes._shared.generation.guidance_options import sampler_step_cache_kwargs

    return sampler_step_cache_kwargs(cfg)["step_cache_options"] is not None


def _guidance_spec_defaults() -> Dict[str, Any]:
    from src.pipelines.pipes._shared.generation.guidance_options import sampler_step_cache_config_specs

    return {s.name: s.default for s in sampler_step_cache_config_specs()}


def _h3_enabled(cfg: Dict[str, Any]) -> bool:
    from src.pipelines.pipes.generator.video_minimax_h3.main import build_step_cache

    return build_step_cache(cfg) is not None


def _h3_spec_defaults() -> Dict[str, Any]:
    from src.pipelines.pipes.generator.video_minimax_h3.main import GeneratorMinimaxH3Pipe

    return {s.name: s.default for s in GeneratorMinimaxH3Pipe.configuration()}


def _dict_enabled(cfg: Dict[str, Any]) -> bool:
    from src.platform.runtime.native.sampling.step_cache import StepCacheSet

    return StepCacheSet(cfg["step_cache"]).enabled


@dataclass(frozen=True)
class Case:
    id: str
    find: Dict[str, str]
    mode: str
    generator: str
    make_form: Callable[[Optional[Dict[str, Any]]], Dict[str, Any]]
    values: Callable[[Dict[str, Any]], Dict[str, Any]]
    enabled: Callable[[Dict[str, Any]], bool]
    spec_defaults: Optional[Callable[[], Dict[str, Any]]] = None
    prompts: tuple = ()
    flat_guidance: bool = False

    def template(self):
        return find_template(**self.find)

    def config(self, over: Optional[Dict[str, Any]] = None, *, form: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        pipes = render(self.template(), self.mode, form or self.make_form(over), self.prompts)
        return pipe(pipes, self.generator)["config"]


def _wan_document(mode: str) -> dict:
    media = []
    if mode in ("i2v", "flf"):
        media.append({"role": "first", "media": {"path": "/media/first.png"}})
    if mode == "flf":
        media.append({"role": "last", "media": {"path": "/media/last.png"}})
    return {
        "mode": mode,
        "settings": {"seed": 123, "duration": 5, "fps": 24},
        "segments": [{"prompt": "a dragon", "negative_prompt": "blurry", "sub_type": mode}],
        "media": media,
        "needs_t2v_set": mode in ("t2v", "director"),
        "needs_i2v_set": mode in ("i2v", "flf", "director"),
    }


def _wan_form(mode: str):
    def make(over):
        form = {
            "t2v_high_noise_model": "/models/wan_t2v_high.safetensors",
            "t2v_low_noise_model": "/models/wan_t2v_low.safetensors",
            "i2v_high_noise_model": "/models/wan_i2v_high.safetensors",
            "i2v_low_noise_model": "/models/wan_i2v_low.safetensors",
            "text_encoder": "/models/umt5.safetensors",
            "vae": "/models/wan_vae.safetensors",
            "resolution": "832x480",
            "video_director": copy.deepcopy(_wan_document(mode)),
        }
        form.update(over or {})
        return form

    return make


def _ltx_form(over, *, upscale: str = "off", mode: str = "video"):
    form = {
        "model": "/models/ltx2.safetensors",
        "text_encoder": "/models/gemma3.safetensors",
        "resolution": "768x512",
    }
    if mode == "upscale":
        form.update({
            "input_video": "/media/in.mp4",
            "upscale_model": "/models/upscaler.safetensors",
            "refine_sigmas": "",
        })
    else:
        form.update({
            "upscale": upscale,
            "upscale_refine_sigmas": "",
            "video_director": {
                "mode": "t2v",
                "settings": {"seed": 123, "duration": 5, "fps": 24},
                "segments": [{"prompt": "a dragon", "negative_prompt": "blurry"}],
                "media": [],
                "media_images": [],
                "media_videos": [],
                "media_placements": [],
                "audio": [],
                "ic_lora": [],
            },
        })
    form.update(over or {})
    return form


def _flux_form(name: str):
    def make(over):
        form = {
            "diffusion_model": "/models/flux_dit.safetensors",
            "text_encoder": "/models/te.safetensors",
            "vae": "/models/flux_vae.safetensors",
            "resolution": "1024x1024",
        }
        if name == "Flux1":
            form["clip_l"] = "/models/clip_l.safetensors"
        form.update(over or {})
        return form

    return make


def _h3_form(over):
    form = {
        "model": "/models/minimax_h3.safetensors",
        "text_encoder": "/models/qwen3_vl.safetensors",
        "video_vae": "/models/h3_video_vae.safetensors",
        "audio_vae": "/models/h3_audio_vae.safetensors",
        "resolution": "1344x768",
        "prompt": "a dragon",
    }
    form.update(over or {})
    return form


WAN_GENERATORS = {
    "t2v": "generator/txt2vid_wan22",
    "i2v": "generator/img2vid_wan22",
    "flf": "generator/img2vid_wan22",
    "director": "generator/chain_video_wan22",
}

CASES = [
    *[
        Case(
            id=f"wan-{mode}",
            find={"suffix": "/Wan"},
            mode="video",
            generator=generator,
            make_form=_wan_form(mode),
            values=_flat_values,
            enabled=_guidance_enabled,
            spec_defaults=_guidance_spec_defaults,
            flat_guidance=True,
        )
        for mode, generator in WAN_GENERATORS.items()
    ],
    Case(
        id="ltx-stage1",
        find={"suffix": "marketplace/LTX-2"},
        mode="video",
        generator="generator_stage1",
        make_form=_ltx_form,
        values=_flat_values,
        enabled=_guidance_enabled,
        spec_defaults=_guidance_spec_defaults,
        flat_guidance=True,
    ),
    Case(
        id="flux1",
        find={"suffix": "marketplace/Flux1"},
        mode="txt2img",
        generator="generator/flux",
        make_form=_flux_form("Flux1"),
        values=_dict_values,
        enabled=_dict_enabled,
    ),
    Case(
        id="flux2",
        find={"suffix": "marketplace/Flux2"},
        mode="txt2img",
        generator="generator/flux",
        make_form=_flux_form("Flux2"),
        values=_dict_values,
        enabled=_dict_enabled,
    ),
    Case(
        id="h3",
        find={"preset_id": H3_PRESET_ID},
        mode="video",
        generator="generator/video_minimax_h3",
        make_form=_h3_form,
        values=_flat_values,
        enabled=_h3_enabled,
        spec_defaults=_h3_spec_defaults,
        prompts=({"positive": "a dragon", "negative": ""},),
    ),
]

FLAT_CASES = [c for c in CASES if c.spec_defaults is not None]
GUIDANCE_CASES = [c for c in CASES if c.flat_guidance]

TABS = {
    "wan": "content/presets/marketplace/Wan/modes/video/tabs/advanced.yml",
    "ltx": "content/presets/marketplace/LTX-2/modes/video/tabs/advanced.yml",
    "flux1": "content/presets/marketplace/Flux1/modes/txt2img/tabs/advanced.yml",
    "flux2": "content/presets/marketplace/Flux2/modes/txt2img/tabs/advanced.yml",
    "h3": "content/presets/marketplace/MiniMax-H3/modes/video/tabs/advanced.yml",
}


RANGES = {
    "step_cache_threshold": (0.0, 0.0, 0.3, 0.01),
    "step_cache_warmup_steps": (4, 0, 12, 1),
    "step_cache_max_skips": (3, 0, 10, 1),
}


def _tab_fields(tab):
    return yaml.safe_load(Path(TABS[tab]).read_text(encoding="utf-8"))["fields"]


def _ids(cases):
    return [c.id for c in cases]


@pytest.mark.parametrize("case", CASES, ids=_ids(CASES))
def test_step_cache_values_reach_the_generator(case):
    cfg = case.config(CACHE_ON)
    assert case.values(cfg) == CACHE_ON_OPTIONS


@pytest.mark.parametrize("case", CASES, ids=_ids(CASES))
def test_step_cache_values_arrive_as_the_declared_types(case):
    values = case.values(case.config(CACHE_ON))
    assert isinstance(values["rel_threshold"], float)
    assert isinstance(values["warmup_steps"], int)
    assert isinstance(values["max_consecutive_skips"], int)


@pytest.mark.parametrize("case", FLAT_CASES, ids=_ids(FLAT_CASES))
def test_step_cache_defaults_match_the_pipe_spec_defaults(case):
    cfg = case.config()
    spec_defaults = case.spec_defaults()
    for key in KEYS:
        assert cfg[key] == spec_defaults[key]


@pytest.mark.parametrize("case", CASES, ids=_ids(CASES))
def test_untouched_form_enables_no_cache(case):
    assert case.enabled(case.config()) is False


@pytest.mark.parametrize("case", CASES, ids=_ids(CASES))
def test_configured_values_enable_the_cache(case):
    assert case.enabled(case.config(CACHE_ON)) is True


@pytest.mark.parametrize("case", GUIDANCE_CASES, ids=_ids(GUIDANCE_CASES))
def test_step_cache_dict_key_stays_unset_so_the_flat_controls_win(case):
    assert not case.config(CACHE_ON).get("step_cache")


@pytest.mark.parametrize("case", GUIDANCE_CASES, ids=_ids(GUIDANCE_CASES))
def test_configured_values_resolve_into_the_denoise_kwargs(case):
    from src.pipelines.pipes._shared.generation.guidance_options import sampler_step_cache_kwargs

    assert sampler_step_cache_kwargs(case.config(CACHE_ON))["step_cache_options"] == CACHE_ON_OPTIONS


@pytest.mark.parametrize("name", ["Flux1", "Flux2"])
def test_flux_dict_is_truthy_when_off_so_only_the_cache_set_declines(name):
    case = next(c for c in CASES if c.id == name.lower())
    cfg = case.config()
    assert cfg["step_cache"]
    assert cfg["step_cache"]["rel_threshold"] == 0.0


def test_step_cache_declared_on_the_flux_generator_spec():
    from src.pipelines.pipes.generator.flux.main import GeneratorFluxPipe

    specs = {s.name: s for s in GeneratorFluxPipe.configuration()}
    assert specs["step_cache"].param_type is dict


LTX = next(c for c in CASES if c.id == "ltx-stage1")


def _ltx_pipes(over=None, **opts):
    mode = opts.get("mode", "video")
    return render(LTX.template(), mode, _ltx_form(over, **opts))


def test_ltx_step_cache_values_reach_stage1_with_upscale_on():
    cfg = pipe(_ltx_pipes(CACHE_ON, upscale="2.0x"), "generator_stage1")["config"]
    assert _flat_values(cfg) == CACHE_ON_OPTIONS


def test_ltx_step_cache_never_reaches_generator_stage2():
    cfg = pipe(_ltx_pipes(CACHE_ON, upscale="2.0x"), "generator_stage2")["config"]
    assert [k for k in KEYS if k in cfg] == []


def test_ltx_step_cache_never_reaches_upscale_mode_generator_refine():
    cfg = pipe(_ltx_pipes(CACHE_ON, mode="upscale"), "generator_refine")["config"]
    assert [k for k in KEYS if k in cfg] == []


@pytest.mark.parametrize("tab", TABS, ids=list(TABS))
@pytest.mark.parametrize("name", KEYS)
def test_step_cache_slider_is_on_the_advanced_tab(tab, name):
    fields = _tab_fields(tab)
    assert field_by_name(fields, name)["type"] == "slider"


@pytest.mark.parametrize("tab", TABS, ids=list(TABS))
def test_threshold_min_equals_the_off_value(tab):
    fields = _tab_fields(tab)
    threshold = field_by_name(fields, "step_cache_threshold")
    assert threshold["configuration"]["min"] == threshold["default"] == 0.0


@pytest.mark.parametrize("tab", TABS, ids=list(TABS))
@pytest.mark.parametrize("name", KEYS)
def test_step_cache_controls_carry_help_in_ai_hint_only(tab, name):
    fields = _tab_fields(tab)
    control = field_by_name(fields, name)
    assert control.get("ai_hint")
    assert "description" not in control
    assert "description" not in (control.get("configuration") or {})


@pytest.mark.parametrize("tab", TABS, ids=list(TABS))
@pytest.mark.parametrize("name", RANGES)
def test_step_cache_slider_range_is_pinned(tab, name):
    default, minimum, maximum, step = RANGES[name]
    control = field_by_name(_tab_fields(tab), name)
    assert control["default"] == default
    assert control["configuration"]["min"] == minimum
    assert control["configuration"]["max"] == maximum
    assert control["configuration"]["step"] == step
