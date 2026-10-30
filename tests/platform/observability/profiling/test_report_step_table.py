from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.platform.observability.profiling import report
from src.platform.observability.profiling.profiler import GenerationProfiler
from src.platform.runtime.native.sampling import step_profile
from src.platform.runtime.native.sampling.denoise_loop import denoise
from src.platform.runtime.native.sampling.step_profile import StepPlanHook
from src.platform.settings import runtime_flags


def _row(t, kind="event", event=None, **extra):
    row = {
        "t": t, "wall": t, "kind": kind, "rss_gb": 2.0, "avail_gb": 10.0, "swap_gb": 0.0, "cpu": 5.0,
        "vram_alloc_gb": {"0": 1.0}, "vram_reserved_gb": {}, "pinned_cum_gb": 0.0,
    }
    if event is not None:
        row["event"] = event
    row.update(extra)
    return row


def _step(t, step, steps, step_s, attn=None, **extra):
    fields = {"step": step, "steps": steps, "step_s": step_s, **extra}
    if attn is not None:
        fields["attn"] = attn
    return _row(t, event="sampler.step", **fields)


def _write(tmp_path, rows) -> Path:
    path = tmp_path / "profile.jsonl"
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


_REMIX = [
    _row(0.0, event="generation.start"),
    _row(0.5, event="streamer.apply", streamed_gb=18.4),
    _step(70.0, 1, 4, 69.5, "sol-attn", stream_wait_s=48.1, streamed_gb=18.4, cache_skip=False),
    _row(70.1, kind="sample"),
    _step(140.0, 2, 4, 69.9, "sol-attn", stream_wait_s=47.9, streamed_gb=18.4, cache_skip=False),
    _step(185.0, 3, 4, 44.5, "dense", stream_wait_s=22.0, streamed_gb=18.4, cache_skip=True),
    _step(230.0, 4, 4, 44.9, "dense", stream_wait_s=21.8, streamed_gb=18.4, cache_skip=False),
    _step(240.0, 1, 2, 5.0),
    _step(245.0, 2, 2, 5.5),
    _row(250.0, event="generation.end"),
]


def test_step_table_lists_every_step_by_loop_with_mode_and_stream_wait(tmp_path):
    text = report.render_report_from_file(_write(tmp_path, _REMIX))

    section = text.split("STEP TABLE (sampler.step events, one row per sampling step)\n", 1)[1]
    lines = section.split("\n")
    rows = [line.split() for line in lines[3:9]]
    assert rows == [
        ["1", "1/4", "sol-attn", "69.500", "48.100", "18.400", "no"],
        ["1", "2/4", "sol-attn", "69.900", "47.900", "18.400", "no"],
        ["1", "3/4", "dense", "44.500", "22.000", "18.400", "yes"],
        ["1", "4/4", "dense", "44.900", "21.800", "18.400", "no"],
        ["2", "1/2", "-", "5.000", "-", "-", "-"],
        ["2", "2/2", "-", "5.500", "-", "-", "-"],
    ]


def test_step_table_summarises_count_mean_and_total_per_mode(tmp_path):
    text = report.render_step_table(report.load_rows(_write(tmp_path, _REMIX)))

    summary = text.split("stream_wait_s\n", 1)[1].split("\n")[1:4]
    assert [line.split() for line in summary] == [
        ["sol-attn", "2", "69.700", "139.400", "96.000"],
        ["dense", "2", "44.700", "89.400", "43.800"],
        ["-", "2", "5.250", "10.500", "-"],
    ]


def test_report_is_unchanged_for_a_profile_without_step_events(tmp_path):
    rows = report.load_rows(_write(tmp_path, [r for r in _REMIX if r.get("event") != "sampler.step"]))

    assert report.render_step_table(rows) == ""
    assert report.render_report(rows) == "".join([
        report.render_stage_table(rows),
        report.render_top_jumps(rows),
        report.render_summary(rows),
        report.render_cpu_tensor_census_groups(rows),
        report.render_cuda_tensor_census_groups(rows),
        report.render_cpu_tensor_census(rows),
        report.render_cuda_tensor_census(rows),
        report.render_cpu_tensor_census_groups_now(rows),
        report.render_cuda_tensor_census_groups_now(rows),
    ])


def test_a_profiled_denoise_renders_its_steps_in_the_report(tmp_path, monkeypatch):
    monkeypatch.setitem(runtime_flags.runtime_flag_values(), "profiling_enabled", True)
    monkeypatch.setitem(runtime_flags.runtime_flag_values(), "profiling_census", False)
    monkeypatch.setattr(GenerationProfiler, "_SAMPLE_INTERVAL_S", 60.0)
    profiler = GenerationProfiler()
    monkeypatch.setattr(step_profile, "get_profiler", lambda: profiler)

    profiler.start("gen-1", tmp_path)
    try:
        denoise(
            lambda x, sigma, cond: torch.full_like(x, 0.5), torch.zeros(1, 4, 4, 4), {"context": "c"}, None,
            steps=3, sampling_settings={"shift": 2.02, "guidance": None}, guidance_scale=0.0,
            seed_noise=torch.randn(1, 4, 4, 4), hooks=(StepPlanHook(lambda i, n: {"attn": "dense"}),),
        )
    finally:
        profiler.stop("gen-1")

    text = report.render_report_from_file(tmp_path / "profile.jsonl")
    assert "STEP TABLE" in text
    summary = text.split("STEP TABLE", 1)[1]
    assert [line.split()[:3] for line in summary.split("\n")[4:7]] == [
        ["1", "1/3", "dense"], ["1", "2/3", "dense"], ["1", "3/3", "dense"],
    ]
    dense = next(line for line in summary.split("\n") if line.startswith("dense "))
    assert dense.split()[1] == "3"
    assert float(dense.split()[3]) == pytest.approx(
        sum(r["step_s"] for r in report.load_rows(tmp_path / "profile.jsonl") if r.get("event") == "sampler.step"),
        abs=0.01,
    )
