from pathlib import Path

import pytest
from PIL import Image

from src.pipelines.contracts import IOType, PipeInput
from src.pipelines.outputs import GenerationExecutionError, ProgressGenerationOutput
from src.pipelines.pipes._shared.media import tools as pipe_tools
from src.pipelines.pipes.media_tool.main import MediaToolPipe
from src.platform.media_tools import MediaToolRegistry, ToolMedia, probe
from src.platform.media_tools.ops import register_core_tools
from tests.platform.media_tools.fake_tools import Mirror, make_spec
from tests.platform.media_tools.media_factory import ffmpeg_available, make_audio, make_video


def make(tool, params=None, **config):
    values = MediaToolPipe.get_default_config()
    values.update(tool=tool, params=params or {}, **config)
    return MediaToolPipe(values)


def run(pipe, cancelled=None, **inputs):
    emitted = []
    output = pipe.process(PipeInput(input=inputs), emitted.append, cancelled).output
    return output, emitted


def cutout(size=(40, 30), box=(10, 5, 30, 25)):
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    image.paste((200, 30, 30, 255), box)
    return image


@pytest.fixture
def need_ffmpeg():
    if not ffmpeg_available():
        pytest.skip("ffmpeg is not available")


@pytest.fixture
def scratch_root(tmp_path, monkeypatch):
    monkeypatch.setattr(pipe_tools.tempfile, "tempdir", str(tmp_path))
    return tmp_path


def leftovers(root):
    return sorted(path.name for path in root.iterdir() if path.name.startswith(pipe_tools.SCRATCH_PREFIX))


class TestContract:
    def test_it_is_a_pipe_named_media_tool(self):
        assert MediaToolPipe.name == "media_tool"

    def test_the_tool_is_required_and_the_params_default_to_nothing(self):
        specs = {spec.name: spec for spec in MediaToolPipe.configuration()}
        assert specs["tool"].required is True
        assert specs["params"].param_type is dict and specs["params"].default == {}
        assert specs["skip_unfit"].default is False

    def test_media_goes_in_and_out_as_arrays_typed_by_kind(self):
        inputs = {spec.name: spec for spec in MediaToolPipe.inputs()}
        outputs = {spec.name: spec for spec in MediaToolPipe.outputs()}
        for kind, io_type in (("image", IOType.IMAGE), ("video", IOType.VIDEO), ("audio", IOType.AUDIO)):
            assert inputs[kind].io_type is io_type and inputs[kind].is_array and not inputs[kind].required
            assert outputs[kind].io_type is io_type and outputs[kind].is_array

    def test_the_registry_comes_in_as_a_service(self):
        service = {spec.name: spec for spec in MediaToolPipe.inputs()}["MEDIA_TOOLS"]
        assert service.io_type is IOType.SERVICE and not service.required


class TestImages:
    def test_an_image_tool_runs_in_memory_and_every_image_comes_back(self, scratch_root):
        output, _ = run(
            make("crop_subject", {"margin": 0}),
            image=[cutout(), cutout(box=(0, 0, 10, 10))],
        )
        assert [image.size for image in output["image"]] == [(20, 20), (10, 10)]
        assert output["video"] == [] and output["audio"] == []
        assert leftovers(scratch_root) == []

    def test_an_image_tool_never_writes_the_image_to_disk(self, scratch_root, monkeypatch):
        def refuse(*args, **kwargs):
            raise AssertionError("the image was written to disk")

        monkeypatch.setattr(MediaToolPipe, "_media", staticmethod(refuse))
        output, _ = run(make("crop_subject"), image=[cutout()])
        assert output["image"][0].size == (20, 20)

    def test_params_are_validated_and_coerced_by_the_platform(self, scratch_root):
        output, _ = run(
            make("fit_canvas", {"width": "64", "height": "48", "scale_percent": "50", "anchor": "top_left", "fill": "#00B140"}),
            image=[Image.new("RGB", (20, 20), (255, 0, 0))],
        )
        (image,) = output["image"]
        assert image.size == (64, 48)
        assert image.getpixel((63, 47))[:3] == (0, 177, 64)

    def test_blank_params_take_the_tool_defaults(self, scratch_root):
        output, _ = run(
            make("crop_subject", {"margin": "", "alpha_threshold": ""}),
            image=[cutout()],
        )
        assert output["image"][0].size == (20, 20)

    def test_a_bare_image_is_accepted(self, scratch_root):
        output, _ = run(make("crop_subject"), image=cutout())
        assert len(output["image"]) == 1

    def test_progress_is_reported_with_the_tools_label(self, scratch_root):
        _, emitted = run(make("crop_subject"), image=[cutout()])
        progress = [item for item in emitted if isinstance(item, ProgressGenerationOutput)]
        assert progress and progress[-1].state == "Crop to subject"
        assert progress[-1].progress.current == 100 and progress[-1].progress.max == 100

    def test_the_param_default_for_an_unwired_key_color_does_not_leak_into_auto_keying(self, scratch_root):
        image = Image.new("RGB", (16, 16), (0, 177, 64))
        output, _ = run(make("color_key", {"key_mode": "auto", "key_color": "not-a-colour"}), image=[image])
        assert output["image"][0].getpixel((8, 8))[3] == 0


class TestRequirements:
    def test_a_tool_that_needs_transparency_refuses_an_opaque_image(self, scratch_root):
        with pytest.raises(GenerationExecutionError, match="transparency"):
            run(make("crop_subject"), image=[Image.new("RGB", (10, 10), (1, 2, 3))])

    def test_skip_unfit_passes_that_image_through_untouched(self, scratch_root):
        opaque = Image.new("RGB", (10, 10), (1, 2, 3))
        output, _ = run(make("crop_subject", skip_unfit=True), image=[opaque, cutout()])
        assert output["image"][0] is opaque
        assert output["image"][1].size == (20, 20)

    def test_skip_unfit_does_not_hide_an_unsupported_kind(self, scratch_root, tmp_path):
        with pytest.raises(GenerationExecutionError, match="cannot be applied to image"):
            run(make("trim", {"start": 0, "end": 1}, skip_unfit=True), image=[cutout()])


class TestErrors:
    def test_an_unknown_tool_is_a_configuration_error(self):
        with pytest.raises(ValueError, match="no media tool 'nope'"):
            run(make("nope"), image=[cutout()])

    def test_a_missing_tool_is_a_configuration_error(self):
        with pytest.raises(ValueError, match="needs a 'tool'"):
            run(make(""), image=[cutout()])

    def test_no_input_is_a_configuration_error(self):
        with pytest.raises(ValueError, match="needs image input"):
            run(make("crop_subject"))

    def test_bad_params_name_the_tool_and_the_field(self):
        with pytest.raises(GenerationExecutionError, match="Fit canvas: .*Width"):
            run(make("fit_canvas", {"width": "wide", "height": 10}), image=[cutout()])

    def test_an_unknown_param_is_refused(self):
        with pytest.raises(GenerationExecutionError, match="Unknown parameter: colour"):
            run(make("crop_subject", {"colour": "red"}), image=[cutout()])

    def test_a_kind_the_tool_does_not_accept_is_refused(self, tmp_path):
        with pytest.raises(GenerationExecutionError, match="cannot be applied to video"):
            run(make("fit_canvas", {"width": 10, "height": 10}), video=[str(tmp_path / "x.mp4")])

    def test_a_component_only_tool_has_nothing_to_run(self):
        registry = MediaToolRegistry()
        registry.register(make_spec("plug.panel", None, source="plug", component="plugin:plug:panel"))
        with pytest.raises(ValueError, match="nothing to run"):
            run(make("plug.panel"), image=[cutout()], MEDIA_TOOLS=registry)


class TestPluginTools:
    @pytest.fixture
    def registry(self):
        registry = MediaToolRegistry()
        register_core_tools(registry)
        registry.register(make_spec("plug.mirror", Mirror(), source="plug", accepts=("image", "video")))
        return registry

    def test_a_plugin_tool_runs_when_the_registry_is_injected(self, registry, scratch_root):
        image = Image.new("RGB", (4, 2), (0, 0, 0))
        image.putpixel((0, 0), (255, 255, 255))
        output, _ = run(make("plug.mirror"), image=[image], MEDIA_TOOLS=registry)
        assert output["image"][0].getpixel((3, 0)) == (255, 255, 255)

    def test_without_the_registry_only_the_built_in_tools_exist(self):
        with pytest.raises(ValueError, match="no media tool 'plug.mirror'"):
            run(make("plug.mirror"), image=[cutout()])

    def test_core_tools_still_run_with_an_injected_registry(self, registry, scratch_root):
        output, _ = run(make("crop_subject"), image=[cutout()], MEDIA_TOOLS=registry)
        assert output["image"][0].size == (20, 20)

    def test_a_plugin_tool_runs_on_video_through_the_platform_runner(self, registry, need_ffmpeg, tmp_path, scratch_root):
        clip = make_video(tmp_path / "in.mp4", seconds=1.0)
        output, _ = run(make("plug.mirror"), video=[str(clip)], MEDIA_TOOLS=registry)
        (result,) = output["video"]
        assert Path(result).is_file() and probe(result).kind == "video"
        Path(result).unlink()


@pytest.mark.usefixtures("need_ffmpeg")
class TestFiles:
    def test_a_video_tool_returns_a_file_that_outlives_the_run(self, tmp_path, scratch_root):
        clip = make_video(tmp_path / "in.mp4", seconds=2.0)
        output, _ = run(make("speed", {"speed": 2}), video=[str(clip)])
        (result,) = output["video"]
        assert Path(result).is_file()
        assert probe(result).duration == pytest.approx(1.0, abs=0.25)
        assert leftovers(scratch_root) == []
        assert output["image"] == [] and output["audio"] == []
        Path(result).unlink()

    def test_every_video_gets_its_own_result(self, tmp_path, scratch_root):
        clips = [make_video(tmp_path / f"{name}.mp4", seconds=2.0) for name in "ab"]
        output, _ = run(make("trim", {"start": 0, "end": 1}), video=[str(clip) for clip in clips])
        assert len(output["video"]) == 2
        assert all(probe(path).duration == pytest.approx(1.0, abs=0.25) for path in output["video"])
        for path in output["video"]:
            Path(path).unlink()

    def test_an_audio_tool_returns_audio(self, tmp_path, scratch_root):
        tone = make_audio(tmp_path / "tone.wav", seconds=2.0)
        output, _ = run(make("trim", {"start": 0.5, "end": 1.5}), audio=[str(tone)])
        (result,) = output["audio"]
        assert probe(result).kind == "audio" and probe(result).duration == pytest.approx(1.0, abs=0.1)
        Path(result).unlink()

    def test_a_tool_that_makes_many_outputs_returns_them_all(self, tmp_path, scratch_root):
        tone = make_audio(tmp_path / "tone.wav", seconds=2.0)
        output, _ = run(make("split", {"part_seconds": 0.5}), audio=[str(tone)])
        assert len(output["audio"]) == 4
        for path in output["audio"]:
            Path(path).unlink()

    def test_a_tool_that_turns_video_into_an_image_returns_an_image(self, tmp_path, scratch_root):
        clip = make_video(tmp_path / "in.mp4", seconds=1.0, size=(64, 48))
        output, _ = run(make("frame", {"time": 0.3}), video=[str(clip)])
        (image,) = output["image"]
        assert isinstance(image, Image.Image) and image.size == (64, 48)
        assert output["video"] == [] and leftovers(scratch_root) == []

    def test_a_tool_that_joins_many_makes_one_output(self, tmp_path, scratch_root):
        clips = [make_video(tmp_path / f"{name}.mp4", seconds=1.0) for name in "ab"]
        output, _ = run(make("concat"), video=[str(clip) for clip in clips])
        (result,) = output["video"]
        assert probe(result).duration == pytest.approx(2.0, abs=0.3)
        Path(result).unlink()

    def test_joining_audio_and_video_together_is_refused(self, tmp_path):
        clip = make_video(tmp_path / "a.mp4", seconds=1.0)
        tone = make_audio(tmp_path / "tone.wav")
        registry = MediaToolRegistry()
        register_core_tools(registry)
        registry.register(make_spec("plug.stack", Mirror(), source="plug", cardinality="many_to_one"))
        with pytest.raises(GenerationExecutionError, match="one kind at a time"):
            run(make("plug.stack"), video=[str(clip)], audio=[str(tone)], MEDIA_TOOLS=registry)

    def test_a_tool_asked_to_run_on_a_missing_file_fails_cleanly(self, tmp_path, scratch_root):
        with pytest.raises(GenerationExecutionError, match="missing"):
            run(make("speed", {"speed": 2}), video=[str(tmp_path / "gone.mp4")])
        assert leftovers(scratch_root) == []

    def test_an_invalid_window_fails_without_leaving_files(self, tmp_path, scratch_root):
        clip = make_video(tmp_path / "in.mp4", seconds=1.0)
        with pytest.raises(GenerationExecutionError, match="past the end"):
            run(make("trim", {"start": 5, "end": 6}), video=[str(clip)])
        assert leftovers(scratch_root) == []

    def test_progress_follows_the_encode_and_reaches_the_end(self, tmp_path, scratch_root):
        clip = make_video(tmp_path / "in.mp4", seconds=2.0)
        output, emitted = run(make("speed", {"speed": 2}), video=[str(clip)])
        percents = [item.progress.current for item in emitted if isinstance(item, ProgressGenerationOutput)]
        assert percents == sorted(percents) and percents[-1] == 100
        Path(output["video"][0]).unlink()

    def test_cancelling_before_the_encode_returns_nothing_and_leaves_nothing(self, tmp_path, scratch_root):
        clip = make_video(tmp_path / "in.mp4", seconds=2.0)
        output, _ = run(make("speed", {"speed": 2}), cancelled=lambda: True, video=[str(clip)])
        assert output == {"image": [], "video": [], "audio": []}
        assert leftovers(scratch_root) == []
        assert sorted(path.name for path in tmp_path.glob("potionui-speed-*")) == []

    def test_cancelling_after_the_first_clip_drops_the_clips_already_made(self, tmp_path, scratch_root):
        clips = [make_video(tmp_path / f"{name}.mp4", seconds=2.0) for name in "ab"]

        def cancel_on_second():
            return any(tmp_path.glob("potionui-speed-*"))

        output, _ = run(make("speed", {"speed": 2}), cancelled=cancel_on_second, video=[str(clip) for clip in clips])
        assert output == {"image": [], "video": [], "audio": []}
        assert sorted(path.name for path in tmp_path.glob("potionui-speed-*")) == []
        assert leftovers(scratch_root) == []


def test_the_progress_relay_reports_each_percent_once():
    from src.pipelines.pipes._shared.media.tools import ProgressRelay

    seen = []
    relay = ProgressRelay(seen.append, "Working")
    for fraction in (0.0, 0.004, 0.5, 0.504, 0.5, 1.0):
        relay.update(fraction)
        relay.flush()
    assert [item.progress.current for item in seen] == [0, 50, 100]


def test_a_tool_media_description_of_an_in_memory_image_knows_its_alpha():
    from src.pipelines.pipes.media_tool.main import describe_image

    assert describe_image(Image.new("RGBA", (3, 2))).has_alpha is True
    assert describe_image(Image.new("RGB", (3, 2))).has_alpha is False
    assert isinstance(describe_image(Image.new("RGB", (3, 2))), ToolMedia)
