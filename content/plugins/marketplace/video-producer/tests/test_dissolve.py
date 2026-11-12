import json
import math
import shutil
import subprocess
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy
import pytest

from backend import core_bridge, projects, takes
from backend.chat import edits
from backend.dissolve import MAX_DISSOLVE, dissolve_of, dissolves
from backend.exports import render
from backend.exports import video
from backend.exports.media import files_of
from backend.exports.mix import write_impulse
from backend.exports.video import build_command, filter_graph, plan_parts, timeline

from .test_exports import FixtureResolver

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
FIXTURES = Path(__file__).resolve().parent / "fixtures"
EXPORTS = FIXTURES / "exports"
CASES = json.loads((FIXTURES / "dissolve_cases.json").read_text(encoding="utf-8"))
LEGACY = json.loads((EXPORTS / "legacy-graphs.json").read_text(encoding="utf-8"))
PATHS = {"ffmpeg": "/usr/bin/ffmpeg", "output": "/tmp/out/mezzanine.mkv", "graph": "/tmp/out/graph.txt"}


def _clips(case):
    return [{"id": f"c{index}", **clip} for index, clip in enumerate(case["clips"])]


def _options(**fields):
    return {"format": "web", "resolution": "720p", "audio": {}, "stills": [], "impulse": "/tmp/out/reverb.wav", **fields}


def _doc(make_doc, clips, **fields):
    doc = make_doc(clips=clips)
    doc.update(fields)
    return doc


def _files(doc):
    return {clip["id"]: {"name": f"{clip['id']}.mp4", "path": f"/media/{clip['id']}.mp4"} for clip in doc["clips"]}


def _graph(doc, **options):
    return filter_graph(doc, _files(doc), _options(**options), {**PATHS, "audio": {}}).split(";")


def _run(make_doc, durations, joins, **extra):
    clips = [{"id": f"c{index}", "assetId": f"a{index}", "dur": dur, **({"dissolve": join} if join else {})} for index, (dur, join) in enumerate(zip(durations, joins))]
    doc = _doc(make_doc, clips, **extra)
    edits.relayout(doc)
    return doc


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_the_shared_cases_give_the_dissolve_each_cut_plays(case):
    assert dissolves(_clips(case)) == pytest.approx(case["applied"], abs=1e-9)


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_the_assistant_lays_the_shots_out_with_their_dissolves(case):
    doc = {"clips": [{**clip, "start": 99} for clip in _clips(case)], "prompts": [], "frames": []}

    edits.relayout(doc)

    assert [clip["start"] for clip in doc["clips"]] == pytest.approx(case["starts"], abs=1e-9)


def test_a_dissolve_moves_the_later_shots_and_their_prompts_but_not_the_sounds_titles_or_markers(make_doc):
    doc = _doc(
        make_doc,
        [{"id": "a", "dur": 4}, {"id": "b", "start": 4, "dur": 4}, {"id": "c", "start": 8, "dur": 4}],
        prompts=[{"id": "p", "shotId": "c", "start": 9, "dur": 1, "text": "x", "strength": 75}],
    )
    doc["audio"] = [{"id": "u", "assetId": None, "lane": "A1", "start": 9, "dur": 1, "srcIn": 0, "fadeIn": 0, "fadeOut": 0, "gain": 0}]
    doc["titles"] = [{"id": "t", "text": "T", "sub": "", "start": 9, "dur": 1}]
    doc["markers"] = [{"id": "m", "time": 9, "label": "M"}]
    doc["clips"][1]["dissolve"] = 1

    edits.relayout(doc)

    assert [clip["start"] for clip in doc["clips"]] == [0, 3, 7]
    assert doc["prompts"][0]["start"] == 8
    assert (doc["audio"][0]["start"], doc["titles"][0]["start"], doc["markers"][0]["time"]) == (9, 9, 9)


def test_the_document_takes_a_dissolve_up_to_two_seconds_on_a_shot(make_doc):
    doc = _doc(make_doc, [{"id": "a", "dur": 4}, {"id": "b", "start": 3, "dur": 4, "dissolve": MAX_DISSOLVE}])

    projects.validate_doc(doc)
    for wrong in (-0.5, MAX_DISSOLVE + 0.04, "1", True):
        doc["clips"][1]["dissolve"] = wrong
        with pytest.raises(projects.DocError):
            projects.validate_doc(doc)


def test_a_dissolve_never_needs_a_migration(make_doc):
    doc = _doc(make_doc, [{"id": "a", "dur": 4}, {"id": "b", "start": 3, "dur": 4, "dissolve": 1}])

    assert projects.migrate(doc) is doc


@pytest.mark.parametrize("case", LEGACY, ids=[case["name"] for case in LEGACY])
def test_a_project_without_dissolve_gives_the_graph_of_before_byte_for_byte(case):
    doc, files, options, paths = case["doc"], case["files"], case["options"], case["paths"]
    segments, total = timeline(doc)

    assert filter_graph(doc, files, options, paths) == case["graph"]
    assert build_command(doc, files, options, paths) == case["command"]
    assert (total, [dur for dur, _ in segments]) == (case["total"], case["segments"])
    assert [len(group) for group in plan_parts(doc, files, options, segments, 1)] == case["parts"]


def test_the_film_lasts_the_shots_less_their_dissolves(make_doc):
    doc = _run(make_doc, [2, 2, 1], [0, 1, 0])

    segments, total = timeline(doc)

    assert [dur for dur, _ in segments] == [2, 2, 1] and total == 4


def test_a_dissolve_joins_two_shots_by_xfade_and_their_sound_by_acrossfade_before_the_concat(make_doc):
    doc = _run(make_doc, [2, 2, 1], [0, 1, 0], fps=25)

    graph = _graph(doc)

    assert "[v0][v1]xfade=transition=fade:duration=1:offset=1[x0v1]" in graph
    assert "[a0][a1]acrossfade=d=1:c1=qsin:c2=qsin[x0a1]" in graph
    assert "[x0v1][x0a1][v2][a2]concat=n=2:v=1:a=1[vout][bed]" in graph


def test_a_run_of_dissolves_offsets_each_join_by_what_is_already_joined(make_doc):
    doc = _run(make_doc, [2, 3, 2, 1], [0, 0.6, 1, 0], fps=25)

    graph = _graph(doc)

    assert "[v0][v1]xfade=transition=fade:duration=0.6:offset=1.4[x0v1]" in graph
    assert "[x0v1][v2]xfade=transition=fade:duration=1:offset=3.4[x0v2]" in graph
    assert "[x0a1][a2]acrossfade=d=1:c1=qsin:c2=qsin[x0a2]" in graph
    assert "[x0v2][x0a2][v3][a3]concat=n=2:v=1:a=1[vout][bed]" in graph
    assert timeline(doc)[1] == pytest.approx(6.4)


def test_the_fades_of_a_dissolved_cut_are_left_out_and_the_others_kept(make_doc):
    doc = _run(make_doc, [2, 2, 2], [0, 1, 0], fps=25)
    for clip in doc["clips"]:
        clip.update(fadeIn=0.4, fadeOut=0.4)

    chains = {part.rsplit("[", 1)[1].rstrip("]"): part for part in _graph(doc) if ":v]trim=" in part}

    assert "fade=t=in" in chains["v0"] and "fade=t=out" not in chains["v0"]
    assert "fade=t=in" not in chains["v1"] and "fade=t=out" in chains["v1"]
    assert "fade=t=in" in chains["v2"] and "fade=t=out" in chains["v2"]


def test_a_dissolve_after_a_gap_or_on_the_first_shot_is_a_cut(make_doc):
    doc = _doc(make_doc, [{"id": "a", "assetId": "x", "dur": 2, "dissolve": 1}, {"id": "b", "assetId": "y", "start": 3, "dur": 2, "dissolve": 1}])

    graph = _graph(doc)

    assert not any("xfade" in part or "acrossfade" in part for part in graph)
    assert [dur for dur, _ in timeline(doc)[0]] == [2, 1, 2]


def test_the_parts_of_a_long_sequence_never_cut_inside_a_dissolve(make_doc):
    doc = _run(make_doc, [2, 2, 2, 2], [0, 1, 0, 1])
    segments, _ = timeline(doc)

    assert [len(group) for group in plan_parts(doc, _files(doc), _options(), segments, 1)] == [2, 2]


async def test_a_part_lasts_its_shots_less_its_dissolves(make_doc, tmp_path, monkeypatch):
    doc = _run(make_doc, [2, 2, 2, 2], [0, 1, 0, 1], fps=25)
    spans = []

    async def assemble(command, total, on_progress=None, on_process=None, cwd=None, code="ffmpeg_failed"):
        spans.append(total)
        return ""

    async def part_length(path, nominal):
        return nominal

    monkeypatch.setattr(video, "assemble", assemble)
    monkeypatch.setattr(video, "_part_length", part_length)
    monkeypatch.setattr(video, "PART_INPUTS", 2)
    paths = {**PATHS, "output": str(tmp_path / "m.mkv"), "graph": str(tmp_path / "g.txt"), "audio": {}}

    await video.assemble_sequence(doc, _files(doc), _options(), paths, tmp_path, lambda progress, step: True)

    assert spans == [3, 3, 6]


def _fixture():
    fixture = json.loads((EXPORTS / "dissolve-project.json").read_text(encoding="utf-8"))
    return fixture, files_of(fixture["doc"], fixture["takes"], FixtureResolver(fixture["uploads"]), media="name")


def test_the_dissolve_project_is_a_document_the_plugin_stores():
    fixture, _ = _fixture()

    projects.validate_doc(fixture["doc"])
    assert dissolves(fixture["doc"]["clips"]) == [0, 1, 0.52, 1.48]


@pytest.mark.parametrize("fmt", ("edl", "otio", "fcpxml"))
def test_each_exchange_format_writes_the_dissolves_as_its_golden_file(fmt):
    fixture, files = _fixture()

    assert render(fmt, fixture["doc"], files).encode("utf-8") == (EXPORTS / f"dissolve.{fmt}").read_bytes()


def test_the_edl_writes_a_dissolve_as_a_zero_length_cut_then_a_d_event_in_frames():
    fixture, files = _fixture()

    lines = render("edl", fixture["doc"], files).split("\r\n")

    assert "002  VP001    V     C        00:00:04:00 00:00:04:00 00:00:04:00 00:00:04:00" in lines
    assert "002  VP002    V     D    025 00:00:00:10 00:00:02:06 00:00:04:00 00:00:07:17" in lines
    assert "001  VP001    V     C        00:00:00:00 00:00:04:00 00:00:00:00 00:00:04:00" in lines
    assert "* TO CLIP NAME: Les mains du pêcheur" in lines


def test_the_fcpxml_puts_a_declared_cross_dissolve_between_the_clips_it_joins():
    fixture, files = _fixture()

    root = ElementTree.fromstring(render("fcpxml", fixture["doc"], files))
    spine = list(root.iter("spine"))[0]

    assert [child.tag for child in spine] == ["asset-clip", "transition", "asset-clip", "transition", "gap", "transition", "asset-clip"]
    effect = root.find("resources/effect")
    assert all(transition.find("filter-video").get("ref") == effect.get("id") for transition in spine.iter("transition"))
    clips = [child for child in spine if child.tag != "transition"]
    ends = [int(child.get("offset").split("/")[0]) + int(child.get("duration").split("/")[0]) for child in clips]
    assert ends[:-1] == [int(child.get("offset").split("/")[0]) for child in clips[1:]]
    assert ends[-1] == 318


def test_the_otio_reads_with_opentimelineio_and_places_each_clip_at_its_cut():
    otio = pytest.importorskip("opentimelineio")
    fixture, files = _fixture()

    track = otio.adapters.read_from_string(render("otio", fixture["doc"], files), "otio_json").tracks[0]

    transitions = [child for child in track if isinstance(child, otio.schema.Transition)]
    assert [(t.in_offset.value, t.out_offset.value) for t in transitions] == [(12, 13), (6, 7), (18, 19)]
    items = [child for child in track if not isinstance(child, otio.schema.Transition)]
    assert [track.range_of_child(item).start_time.value for item in items] == [0, 112, 198, 248]
    assert track.duration().value == 318


def _solid(folder: Path, name: str, color: str, tone: int, seconds: float) -> Path:
    path = folder / name
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"color=c={color}:s=320x180:r=25:d={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency={tone}:sample_rate=48000:duration={seconds}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    ]
    subprocess.run(command, check=True, timeout=60)
    return path


def _frame(path: Path, number: int):
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-vf", f"select=eq(n\\,{number}),scale=1:1", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    return tuple(subprocess.run(command, capture_output=True, check=True, timeout=60).stdout)


def _frames(path: Path) -> int:
    command = ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)]
    return int(subprocess.run(command, capture_output=True, check=True, text=True, timeout=60).stdout.strip())


def _samples(path: Path):
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1", "-ar", "48000", "-f", "f32le", "-"]
    return numpy.frombuffer(subprocess.run(command, capture_output=True, check=True, timeout=60).stdout, dtype=numpy.float32)


def _tone(samples, start: float, end: float, frequency: int) -> float:
    window = samples[int(start * 48000) : int(end * 48000)]
    phase = numpy.exp(-2j * math.pi * frequency * numpy.arange(len(window)) / 48000)
    return float(abs(numpy.dot(window, phase)) / len(window))


@pytest.fixture
def solids(tmp_path):
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg and ffprobe are not on the PATH")
    return _solid(tmp_path, "red.mp4", "red", 440, 2), _solid(tmp_path, "blue.mp4", "blue", 880, 2), _solid(tmp_path, "lime.mp4", "lime", 1320, 1)


@pytest.mark.requires_ffmpeg
@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg and ffprobe are not on the PATH")
async def test_a_real_export_mixes_both_pictures_halfway_through_the_dissolve_and_keeps_the_sound_in_step(make_doc, solids, tmp_path):
    red, blue, lime = solids
    doc = _run(make_doc, [2, 2, 1], [0, 1, 0], fps=25, resolution="720p")
    files = {clip["id"]: {"name": path.name, "path": str(path)} for clip, path in zip(doc["clips"], solids)}
    output = tmp_path / "mezzanine.mkv"
    options = {"format": "web", "resolution": "720p", **video.inspect(files), "impulse": str(write_impulse(tmp_path / "reverb.wav", 48000))}
    paths = {"ffmpeg": shutil.which("ffmpeg"), "graph_option": video.graph_option(shutil.which("ffmpeg")), "output": str(output), "graph": str(tmp_path / "graph.txt"), "audio": {}}
    Path(paths["graph"]).write_text(filter_graph(doc, files, options, paths), encoding="utf-8")

    await video.assemble(build_command(doc, files, options, paths), 4.0)

    assert _frames(output) == 100
    probe = core_bridge.probe_video(output)
    assert probe.has_audio and abs(probe.duration - 4.0) < 0.05
    before, middle, after, last = _frame(output, 20), _frame(output, 37), _frame(output, 55), _frame(output, 90)
    assert before[0] > 200 and before[2] < 40 and after[2] > 200 and after[0] < 40
    assert 90 < middle[0] < 170 and 90 < middle[2] < 170 and middle[1] < 40
    assert last[1] > 200 and last[0] < 40
    samples = _samples(output)
    assert len(samples) == pytest.approx(4 * 48000, abs=48000 * 0.03)
    assert _tone(samples, 0.2, 0.8, 440) > 10 * _tone(samples, 0.2, 0.8, 880)
    assert _tone(samples, 2.2, 2.8, 880) > 10 * _tone(samples, 2.2, 2.8, 440)
    assert _tone(samples, 1.4, 1.6, 440) > 0.02 and _tone(samples, 1.4, 1.6, 880) > 0.02
    assert _tone(samples, 3.2, 3.8, 1320) > 10 * _tone(samples, 3.2, 3.8, 880)


@pytest.mark.requires_ffmpeg
@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg and ffprobe are not on the PATH")
async def test_render_delivers_a_film_shortened_by_its_dissolves(make_doc, solids, monkeypatch, export_storage, make_user):
    doc = _run(make_doc, [2, 2, 1], [0, 1, 0], fps=25, resolution="720p")
    shots = {clip["id"]: {"name": path.name, "path": str(path), "still": False} for clip, path in zip(doc["clips"], solids)}
    monkeypatch.setattr(video.media, "local_inputs", lambda doc, found, resolver, folder: (shots, {}))
    monkeypatch.setattr(takes, "list_takes", lambda project_id: [])

    async def save_user_media(user, path):
        return {"value": "uploads/0f0e.mp4", "media_type": "video", "original_filename": Path(path).name, "origin": None, "ref": {"kind": "library", "item_id": "item-1"}}

    monkeypatch.setattr(core_bridge, "save_user_media", save_user_media)
    user = make_user()
    job = {"id": "job-1", "user_id": "user-1", "format": "web", "resolution": "720p"}

    result = await video.render(job, {"id": "p-1", "name": "Fondu", "doc": doc}, user, user, lambda progress, step: True)

    final = export_storage / "video_producer" / "exports" / "user-1" / "job-1" / result["file_name"]
    assert _frames(final) == 100
    middle = _frame(final, 37)
    assert 90 < middle[0] < 170 and 90 < middle[2] < 170
