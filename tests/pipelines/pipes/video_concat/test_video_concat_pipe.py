import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.features.generation.engine import validate_pipe_configuration
from src.pipelines.contracts import IOType, PipeInput
from src.pipelines.outputs import GenerationExecutionError, ProgressGenerationOutput
from src.pipelines.pipes._shared.media import tools as pipe_tools
from src.pipelines.pipes.video_concat import main as pipe_module
from src.pipelines.pipes.video_concat.main import VideoConcatPipe
from src.platform.media_tools import ToolCancelled, ToolMedia, probe
from src.platform.media_tools.ops.concat import concat_args
from src.platform.util.video_transcode import find_ffmpeg
from tests.platform.media_tools.media_factory import ffmpeg_available, make_video
from tests.platform.media_tools.ops.time_helpers import count_video_frames


def make(**config):
    values = VideoConcatPipe.get_default_config()
    values.update(config)
    return VideoConcatPipe(values)


def run(pipe, cancelled=None, **inputs):
    emitted = []
    output = pipe.process(PipeInput(input=inputs), emitted.append, cancelled).output["video"]
    return output, emitted


@pytest.fixture
def joins(monkeypatch):
    calls = []

    def fake(clips, out_path, **kwargs):
        calls.append({"clips": list(clips), **kwargs})
        return out_path

    monkeypatch.setattr(pipe_module, "join_video_files", fake)
    return calls


class TestContract:
    def test_the_pipe_keeps_its_name_inputs_and_outputs(self):
        assert VideoConcatPipe.name == "video_concat"
        inputs = {spec.name: spec for spec in VideoConcatPipe.inputs()}
        assert set(inputs) == {"video", "extension", "overlap"}
        assert inputs["video"].required and not inputs["extension"].required and not inputs["overlap"].required
        assert all(spec.io_type is IOType.VIDEO and spec.is_array for spec in inputs.values())
        (output,) = VideoConcatPipe.outputs()
        assert (output.name, output.io_type, output.is_array) == ("video", IOType.VIDEO, True)

    def test_the_configuration_keeps_its_names_and_defaults(self):
        assert VideoConcatPipe.get_default_config() == {
            "extension_position": "after",
            "overlap_frames": 0,
            "overlap": "crossfade",
            "fps": 24.0,
            "fit": "pad",
            "keep_audio": True,
        }

    def test_a_fit_other_than_pad_or_cover_is_rejected(self):
        with pytest.raises(ValueError, match="fit"):
            validate_pipe_configuration(VideoConcatPipe, {"fit": "stretch"})

        assert validate_pipe_configuration(VideoConcatPipe, {"fit": "cover"})["fit"] == "cover"
        assert validate_pipe_configuration(VideoConcatPipe, {})["fit"] == "pad"


class TestPlan:
    def test_without_extensions_every_video_is_joined_once_in_order(self, joins):
        out, _ = run(make(), video=["a.mp4", "b.mp4", "c.mp4"])

        assert len(out) == 1 and [call["clips"] for call in joins] == [["a.mp4", "b.mp4", "c.mp4"]]
        assert joins[0]["trims"] is None and joins[0]["crossfades"] is None and joins[0]["audio"] is True
        assert joins[0]["canvas"] == 0

    def test_each_extension_makes_its_own_joined_video_after_the_source(self, joins):
        out, _ = run(make(overlap_frames=12, fps=24), video="src.mp4", extension=["e1.mp4", "e2.mp4"])

        assert len(out) == 2
        assert [call["clips"] for call in joins] == [["src.mp4", "e1.mp4"], ["src.mp4", "e2.mp4"]]
        assert joins[0]["crossfades"] == [0.5] and joins[0]["trims"] == [(0.0, 0.0), (0.0, 0.0)]
        assert joins[0]["canvas"] == 0

    def test_an_extension_before_the_source_cuts_its_repeated_tail(self, joins):
        run(
            make(extension_position="before", overlap_frames=6, fps=24, overlap="cut", keep_audio="false"),
            video=["src.mp4"], extension=["e.mp4"],
        )

        (call,) = joins
        assert call["clips"] == ["e.mp4", "src.mp4"]
        assert call["trims"] == [(0.0, 0.25), (0.0, 0.0)] and call["crossfades"] == [0.0]
        assert call["audio"] is False and call["canvas"] == 1

    def test_an_extension_after_the_source_cuts_its_repeated_head(self, joins):
        run(make(overlap_frames=6, fps=24, overlap="cut"), video=["src.mp4"], extension=["e.mp4"])

        assert joins[0]["trims"] == [(0.0, 0.0), (0.25, 0.0)]

    def test_a_crossfade_before_the_source_sits_on_the_first_seam(self, joins):
        run(make(extension_position="before", overlap_frames=6, fps=24), video=["a.mp4", "b.mp4"], extension=["e.mp4"])

        assert joins[0]["clips"] == ["e.mp4", "a.mp4", "b.mp4"]
        assert joins[0]["crossfades"] == [0.25, 0.0] and joins[0]["canvas"] == 1

    def test_the_overlap_clip_sets_the_seam_length(self, joins, monkeypatch):
        monkeypatch.setattr(pipe_module, "probe_video_stream", lambda path: (64, 48, 30.0, 25))

        run(make(overlap_frames=99, fps=24), video=["src.mp4"], extension=["e.mp4"], overlap=["ctx.mp4"])

        assert joins[0]["crossfades"] == [pytest.approx(25 / 30)]

    def test_no_overlap_means_no_seam_handling(self, joins):
        run(make(), video=["src.mp4"], extension=["e.mp4"])

        assert joins[0]["crossfades"] == [0.0] and joins[0]["trims"] == [(0.0, 0.0), (0.0, 0.0)]

    def test_the_fit_defaults_to_padding_and_the_chosen_fit_reaches_the_join(self, joins):
        run(make(), video=["a.mp4", "b.mp4"])
        run(make(fit="cover"), video="src.mp4", extension="e.mp4")

        assert [call["fit"] for call in joins] == ["pad", "cover"]

    def test_no_video_is_an_error(self, joins):
        with pytest.raises(ValueError, match="at least one video"):
            run(make(), video=[])

    def test_a_cancel_stops_without_a_joined_video(self, monkeypatch):
        def cancelled(clips, out_path, **kwargs):
            raise ToolCancelled()

        monkeypatch.setattr(pipe_module, "join_video_files", cancelled)

        out, _ = run(make(), video=["src.mp4"], extension=["e1.mp4", "e2.mp4"])
        assert out == []

    def test_a_cancel_flag_is_handed_to_the_join(self, joins):
        flag = lambda: False
        run(make(), cancelled=flag, video=["a.mp4", "b.mp4"])

        assert joins[0]["is_cancelled"] is flag

    def test_a_join_that_cannot_be_made_is_a_clean_generation_error(self, monkeypatch):
        from src.platform.media_tools import InvalidToolParams

        def refused(clips, out_path, **kwargs):
            raise InvalidToolParams("overlap_frames", "A 9.000s crossfade is longer than the clips it joins")

        monkeypatch.setattr(pipe_module, "join_video_files", refused)

        with pytest.raises(GenerationExecutionError, match="crossfade is longer"):
            run(make(), video=["a.mp4", "b.mp4"])


def listed(path, width, height, duration, fps=10.0, audio=True):
    return ToolMedia(path=Path(path), kind="video", width=width, height=height, duration=duration, fps=fps, has_audio=audio)


def graph(clips, **kwargs):
    args = concat_args(clips, Path("/o/out.mp4"), audio=True, fit="cover", **kwargs)
    return args[args.index("-filter_complex") + 1]


SOURCE = listed("/c/src.mp4", 64, 48, 2.0)
EXTENSION = listed("/c/ext.mp4", 60, 48, 1.5)

OLD_SCALE = "scale=64:48:force_original_aspect_ratio=increase,crop=64:48,setsar=1,fps=10,format=yuv420p"
OLD_AUDIO = "aresample=48000,aformat=channel_layouts=stereo"


class TestTheJoinCommandIsWhatTheOldPipeBuilt:
    def test_an_extension_after_the_source(self):
        assert graph([SOURCE, EXTENSION], crossfades=[0.0]) == (
            f"[0:v]{OLD_SCALE}[v0];[0:a]{OLD_AUDIO}[a0];[1:v]{OLD_SCALE}[v1];[1:a]{OLD_AUDIO}[a1];"
            "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]"
        )

    def test_an_extension_before_the_source_still_takes_the_sources_canvas(self):
        assert graph([EXTENSION, SOURCE], crossfades=[0.0], canvas=1) == (
            f"[0:v]{OLD_SCALE}[v0];[0:a]{OLD_AUDIO}[a0];[1:v]{OLD_SCALE}[v1];[1:a]{OLD_AUDIO}[a1];"
            "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]"
        )

    def test_a_cut_after_the_source_trims_the_extensions_head_in_picture_and_sound(self):
        assert graph([SOURCE, EXTENSION], crossfades=[0.0], trims=[(0.0, 0.0), (0.3, 0.0)]) == (
            f"[0:v]{OLD_SCALE}[v0];[0:a]{OLD_AUDIO}[a0];"
            f"[1:v]trim=start=0.299000,setpts=PTS-STARTPTS,{OLD_SCALE}[v1];"
            f"[1:a]atrim=start=0.299000,asetpts=PTS-STARTPTS,{OLD_AUDIO}[a1];"
            "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]"
        )

    def test_a_crossfade_before_the_source_blends_at_the_end_of_the_extension(self):
        timebase = OLD_SCALE + ",settb=AVTB"
        assert graph([EXTENSION, SOURCE], crossfades=[0.3], canvas=1) == (
            f"[0:v]{timebase}[v0];[0:a]{OLD_AUDIO}[a0];[1:v]{timebase}[v1];[1:a]{OLD_AUDIO}[a1];"
            "[v0][v1]xfade=transition=fade:duration=0.300000:offset=1.200000[v];[a0][a1]acrossfade=d=0.300000[a]"
        )


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg is not available")
class TestExtendsOnRealClips:
    @pytest.fixture
    def clips(self, tmp_path):
        return {
            "source": make_video(tmp_path / "source.mp4", seconds=2.0, size=(64, 48), fps=10, audio=True),
            "narrow": make_video(tmp_path / "narrow.mp4", seconds=1.5, size=(60, 48), fps=10, audio=True, freq=660),
            "mute": make_video(tmp_path / "mute.mp4", seconds=1.5, size=(60, 48), fps=10, audio=False),
        }

    @pytest.fixture(autouse=True)
    def scratch_root(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipe_tools.tempfile, "tempdir", str(tmp_path))
        monkeypatch.setattr(pipe_module.tempfile, "tempdir", str(tmp_path))

    def joined(self, **config):
        return make(fit="cover", **config)

    def test_a_forward_extend_has_the_frames_size_and_sound_of_both_clips(self, clips):
        (out,), _ = run(self.joined(extension_position="after"), video=str(clips["source"]), extension=str(clips["narrow"]))

        info = probe(out)
        assert count_video_frames(Path(out)) == 20 + 15
        assert (info.width, info.height) == (64, 48)
        assert info.has_audio is True
        assert info.fps == pytest.approx(10.0)

    def test_a_backward_extend_has_the_frames_the_sources_size_and_sound(self, clips):
        (out,), _ = run(self.joined(extension_position="before"), video=str(clips["source"]), extension=str(clips["narrow"]))

        info = probe(out)
        assert count_video_frames(Path(out)) == 15 + 20
        assert (info.width, info.height) == (64, 48)
        assert info.has_audio is True

    def test_a_backward_extend_fills_the_source_frame_instead_of_cropping_the_source_to_the_extension(self, tmp_path):
        source = make_video(tmp_path / "wide-source.mp4", seconds=1.0, size=(72, 48), fps=10, audio=False)
        extension = make_video(tmp_path / "narrow-ext.mp4", seconds=1.0, size=(60, 48), fps=10, audio=False)

        (out,), _ = run(self.joined(extension_position="before"), video=str(source), extension=str(extension))

        assert (probe(out).width, probe(out).height) == (72, 48)

    def test_padding_leaves_black_bars_on_a_narrower_extension_and_cover_does_not(self, tmp_path):
        source = tmp_path / "white-source.mp4"
        extension = tmp_path / "white-ext.mp4"
        for path, width in ((source, 72), (extension, 56)):
            subprocess.run(
                [find_ffmpeg(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi",
                 "-i", f"color=c=white:size={width}x128:rate=10:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 str(path)],
                check=True,
            )

        edges = {}
        for fit in ("pad", "cover"):
            (out,), _ = run(make(fit=fit), video=str(source), extension=str(extension))
            capture = cv2.VideoCapture(str(out))
            last = None
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                last = frame
            capture.release()
            assert last.shape[:2] == (128, 72)
            edges[fit] = float(np.concatenate([last[:, :4], last[:, -4:]]).mean())
        assert edges["pad"] < 32 and edges["cover"] > 224

    def test_a_silent_source_with_a_sounding_extension_gives_a_video_with_sound(self, clips, tmp_path):
        silent = make_video(tmp_path / "silent.mp4", seconds=1.0, size=(64, 48), fps=10, audio=False)
        (out,), _ = run(self.joined(), video=str(silent), extension=str(clips["narrow"]))

        assert probe(out).has_audio is True
        assert count_video_frames(Path(out)) == 10 + 15

    def test_a_silent_extend_gives_a_silent_video(self, clips, tmp_path):
        silent = make_video(tmp_path / "silent.mp4", seconds=1.0, size=(64, 48), fps=10, audio=False)
        (out,), _ = run(self.joined(), video=str(silent), extension=str(clips["mute"]))

        assert probe(out).has_audio is False

    def test_keep_audio_off_drops_the_sound(self, clips):
        (out,), _ = run(self.joined(keep_audio=False), video=str(clips["source"]), extension=str(clips["narrow"]))

        assert probe(out).has_audio is False

    def test_a_cut_overlap_removes_the_repeated_head_of_the_extension(self, clips, tmp_path):
        overlap = make_video(tmp_path / "overlap.mp4", seconds=0.3, size=(64, 48), fps=10, audio=False)
        (out,), _ = run(
            self.joined(overlap="cut"),
            video=str(clips["source"]), extension=str(clips["narrow"]), overlap=str(overlap),
        )

        assert count_video_frames(Path(out)) == pytest.approx(20 + 15 - 3, abs=1)

    def test_a_crossfade_overlap_blends_the_repeated_frames_into_the_source(self, clips, tmp_path):
        overlap = make_video(tmp_path / "overlap.mp4", seconds=0.3, size=(64, 48), fps=10, audio=False)
        (out,), _ = run(
            self.joined(overlap="crossfade"),
            video=str(clips["source"]), extension=str(clips["narrow"]), overlap=str(overlap),
        )

        assert count_video_frames(Path(out)) == pytest.approx(20 + 15 - 3, abs=1)

    def test_every_extension_gets_its_own_joined_file_that_outlives_the_run(self, clips, tmp_path):
        out, _ = run(self.joined(), video=str(clips["source"]), extension=[str(clips["narrow"]), str(clips["mute"])])

        assert len(out) == 2 and all(Path(path).is_file() for path in out)
        for path in out:
            Path(path).unlink()

    def test_progress_ends_at_the_finish(self, clips):
        (out,), emitted = run(self.joined(), video=str(clips["source"]), extension=str(clips["narrow"]))

        progress = [item for item in emitted if isinstance(item, ProgressGenerationOutput)]
        assert progress[0].state == "Joining the clips"
        assert progress[-1].progress.current == 100

    def test_a_cancel_leaves_no_joined_file(self, clips, tmp_path):
        out, _ = run(
            self.joined(), cancelled=lambda: True,
            video=str(clips["source"]), extension=str(clips["narrow"]),
        )

        assert out == []
        assert sorted(path.name for path in tmp_path.glob("potionui-joined-*")) == []
        assert sorted(path.name for path in tmp_path.glob("potionui-media-tool-*")) == []
