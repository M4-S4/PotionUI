from __future__ import annotations

import json
import struct
import weakref

import pytest
import torch
from safetensors import safe_open

from src.platform.runtime.native.ops.model_tools.errors import StreamMismatchError, WriteCancelledError
from src.platform.runtime.native.ops.model_tools.source import read_stream
from src.platform.runtime.native.ops.model_tools.spec import TensorSpec
from src.platform.runtime.native.ops.model_tools.writer import build_header, write_stream

PLAN = [
    TensorSpec("a", "BF16", (3, 5)),
    TensorSpec("b", "F32", ()),
    TensorSpec("c", "F8_E4M3", (7,)),
    TensorSpec("d", "U8", (3,)),
    TensorSpec("e", "F16", (2, 2)),
]


def tensors_for(plan=PLAN):
    dtypes = {"BF16": torch.bfloat16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn, "U8": torch.uint8, "F16": torch.float16}
    for i, spec in enumerate(plan):
        yield spec.key, (torch.arange(max(spec.numel, 1)).reshape(spec.shape or ()).to(torch.float32) + i).to(dtypes[spec.dtype])


def leftovers(folder):
    return [p.name for p in folder.iterdir()]


def test_roundtrip_order_dtypes_and_metadata(tmp_path):
    dest = tmp_path / "out.safetensors"
    result = write_stream(dest, PLAN, tensors_for(), {"k": "v"})
    assert result.path == dest
    assert result.tensor_count == 5
    assert result.size_bytes == dest.stat().st_size
    with safe_open(str(dest), framework="pt") as f:
        assert f.metadata() == {"k": "v"}
        for spec, (key, expected) in zip(PLAN, tensors_for()):
            got = f.get_tensor(spec.key)
            assert got.dtype == expected.dtype and got.shape == expected.shape
            assert torch.equal(got.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8))
    plan_back, _, meta = read_stream(dest)
    assert [s.key for s in plan_back] == [s.key for s in PLAN]
    assert meta == {"k": "v"}


def test_header_is_eight_byte_aligned_for_every_metadata_length():
    for n in range(0, 17):
        raw = build_header(PLAN, {"m": "x" * n})
        assert len(raw) % 8 == 0
        json.loads(raw)


def test_file_layout_matches_format(tmp_path):
    dest = tmp_path / "o.safetensors"
    write_stream(dest, PLAN, tensors_for(), {"m": "abc"})
    blob = dest.read_bytes()
    (size,) = struct.unpack("<Q", blob[:8])
    assert size % 8 == 0
    header = json.loads(blob[8 : 8 + size])
    end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
    assert len(blob) == 8 + size + end
    starts = [header[s.key]["data_offsets"][0] for s in PLAN]
    assert starts == sorted(starts)


def test_empty_metadata_omits_block(tmp_path):
    dest = tmp_path / "o.safetensors"
    write_stream(dest, PLAN, tensors_for(), {})
    with safe_open(str(dest), framework="pt") as f:
        assert f.metadata() is None


def test_noncontiguous_tensor_written_contiguous(tmp_path):
    plan = [TensorSpec("t", "F32", (3, 2))]
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    dest = tmp_path / "o.safetensors"
    write_stream(dest, plan, iter([("t", base.t())]), {})
    with safe_open(str(dest), framework="pt") as f:
        assert torch.equal(f.get_tensor("t"), base.t())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda items: [("zzz", items[0][1])] + items[1:],
        lambda items: [items[1], items[0]] + items[2:],
        lambda items: [(items[0][0], items[0][1].to(torch.float32))] + items[1:],
        lambda items: [(items[0][0], items[0][1].reshape(5, 3))] + items[1:],
        lambda items: items[:-1],
        lambda items: items + [("extra", torch.zeros(1))],
    ],
    ids=["key", "order", "dtype", "shape", "short", "long"],
)
def test_plan_mismatch_aborts_and_cleans(tmp_path, mutate):
    dest = tmp_path / "o.safetensors"
    with pytest.raises(StreamMismatchError):
        write_stream(dest, PLAN, iter(mutate(list(tensors_for()))), {})
    assert leftovers(tmp_path) == []


def test_mismatch_keeps_existing_destination(tmp_path):
    dest = tmp_path / "o.safetensors"
    dest.write_bytes(b"old")
    items = list(tensors_for())[:-1]
    with pytest.raises(StreamMismatchError):
        write_stream(dest, PLAN, iter(items), {})
    assert dest.read_bytes() == b"old"
    assert leftovers(tmp_path) == ["o.safetensors"]


def test_cancel_deletes_temp_and_raises(tmp_path):
    dest = tmp_path / "o.safetensors"
    calls = []

    def cancelled():
        calls.append(1)
        return len(calls) > 2

    with pytest.raises(WriteCancelledError):
        write_stream(dest, PLAN, tensors_for(), {}, is_cancelled=cancelled)
    assert leftovers(tmp_path) == []


def test_cancel_closes_stream(tmp_path):
    closed = []

    def stream():
        try:
            yield from tensors_for()
        finally:
            closed.append(True)

    calls = []

    def cancelled():
        calls.append(1)
        return len(calls) > 1

    with pytest.raises(WriteCancelledError):
        write_stream(tmp_path / "o.safetensors", PLAN, stream(), {}, is_cancelled=cancelled)
    assert closed == [True]


def test_source_error_midstream_cleans_temp(tmp_path):
    def stream():
        yield next(iter(tensors_for()))
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        write_stream(tmp_path / "o.safetensors", PLAN, stream(), {})
    assert leftovers(tmp_path) == []


def test_atomic_rename_replaces_existing_only_on_success(tmp_path, monkeypatch):
    dest = tmp_path / "o.safetensors"
    dest.write_bytes(b"old")
    seen = []
    real = __import__("os").replace

    def spy(src, dst):
        seen.append((src.name.endswith(".tmp"), dest.read_bytes()))
        real(src, dst)

    monkeypatch.setattr("src.platform.runtime.native.ops.model_tools.writer.os.replace", spy)
    write_stream(dest, PLAN, tensors_for(), {})
    assert seen == [(True, b"old")]
    assert dest.read_bytes() != b"old"
    assert leftovers(tmp_path) == ["o.safetensors"]


def test_progress_reports_bytes(tmp_path):
    events = []
    write_stream(tmp_path / "o.safetensors", PLAN, tensors_for(), {}, on_progress=lambda w, t: events.append((w, t)))
    total = sum(s.nbytes for s in PLAN)
    assert len(events) == len(PLAN)
    assert events[-1] == (total, total)
    assert [e[0] for e in events] == sorted(e[0] for e in events)


def test_writer_holds_at_most_one_tensor(tmp_path):
    plan = [TensorSpec(f"t{i}", "F32", (64, 64)) for i in range(12)]
    refs = []
    live_at_yield = []

    def stream():
        for spec in plan:
            live_at_yield.append(sum(1 for r in refs if r() is not None))
            tensor = torch.zeros(64, 64)
            refs.append(weakref.ref(tensor))
            yield spec.key, tensor
            del tensor

    write_stream(tmp_path / "o.safetensors", plan, stream(), {})
    assert max(live_at_yield) <= 1
