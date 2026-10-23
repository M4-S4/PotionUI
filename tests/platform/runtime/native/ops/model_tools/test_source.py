from __future__ import annotations

import json
import struct

import pytest
import torch
from safetensors.torch import save_file

from src.platform.runtime.native.ops.model_tools.errors import SourceFormatError
from src.platform.runtime.native.ops.model_tools.source import read_header, read_stream


def raw_file(tmp_path, header, payload=b"\0" * 16, name="f.safetensors"):
    blob = header if isinstance(header, bytes) else json.dumps(header).encode()
    path = tmp_path / name
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + payload)
    return path


def entry(**over):
    base = {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
    base.update(over)
    return base


def test_garbage_file(tmp_path):
    p = tmp_path / "g.safetensors"
    p.write_bytes(b"not a model at all, just text")
    with pytest.raises(SourceFormatError):
        read_header(p)


def test_empty_file(tmp_path):
    p = tmp_path / "e.safetensors"
    p.write_bytes(b"")
    with pytest.raises(SourceFormatError):
        read_header(p)


def test_truncated_header(tmp_path):
    p = tmp_path / "t.safetensors"
    p.write_bytes(struct.pack("<Q", 500) + b'{"a":')
    with pytest.raises(SourceFormatError):
        read_header(p)


def test_oversized_header(tmp_path):
    p = tmp_path / "o.safetensors"
    p.write_bytes(struct.pack("<Q", 2**40) + b"{}")
    with pytest.raises(SourceFormatError):
        read_header(p)


def test_bad_json(tmp_path):
    with pytest.raises(SourceFormatError):
        read_header(raw_file(tmp_path, b"{not json"))


def test_non_dict_header(tmp_path):
    with pytest.raises(SourceFormatError):
        read_header(raw_file(tmp_path, [1, 2]))


def test_non_dict_entry(tmp_path):
    with pytest.raises(SourceFormatError, match="'a'"):
        read_header(raw_file(tmp_path, {"a": 5}))


@pytest.mark.parametrize("missing", ["dtype", "shape", "data_offsets"])
def test_missing_entry_field(tmp_path, missing):
    e = entry()
    del e[missing]
    with pytest.raises(SourceFormatError, match="'a'"):
        read_header(raw_file(tmp_path, {"a": e}, b"\0" * 8))


@pytest.mark.parametrize(
    "bad",
    [
        {"dtype": "F4"},
        {"dtype": 7},
        {"shape": "2"},
        {"shape": [-2]},
        {"shape": [True]},
        {"shape": [1.5]},
        {"data_offsets": [0]},
        {"data_offsets": "ab"},
        {"data_offsets": [0, "8"]},
        {"data_offsets": [-8, 0]},
    ],
)
def test_malformed_entry_values(tmp_path, bad):
    with pytest.raises(SourceFormatError, match="'a'"):
        read_header(raw_file(tmp_path, {"a": entry(**bad)}, b"\0" * 8))


def test_offsets_disagree_with_shape(tmp_path):
    with pytest.raises(SourceFormatError, match="disagree"):
        read_header(raw_file(tmp_path, {"a": entry(data_offsets=[0, 4])}, b"\0" * 8))


def test_offsets_exceed_file(tmp_path):
    with pytest.raises(SourceFormatError, match="past the end"):
        read_header(raw_file(tmp_path, {"a": entry()}, b"\0" * 4))


def test_huge_shape_does_not_escape_as_bare_error(tmp_path):
    with pytest.raises(SourceFormatError):
        read_header(raw_file(tmp_path, {"a": entry(shape=[2**62, 2**62])}, b"\0" * 8))


def test_plan_is_sorted_by_offset(tmp_path):
    header = {
        "__metadata__": {"m": "1"},
        "late": entry(data_offsets=[8, 16]),
        "early": entry(data_offsets=[0, 8]),
    }
    plan, meta = read_header(raw_file(tmp_path, header, b"\0" * 16))
    assert [s.key for s in plan] == ["early", "late"]
    assert meta == {"m": "1"}


def test_valid_file_reads_back(tmp_path):
    p = tmp_path / "ok.safetensors"
    save_file({"w": torch.arange(6, dtype=torch.float32).reshape(2, 3)}, str(p), metadata={"a": "b"})
    plan, tensors, meta = read_stream(p)
    assert meta == {"a": "b"}
    assert [(s.key, s.dtype, s.shape) for s in plan] == [("w", "F32", (2, 3))]
    assert torch.equal(dict(tensors)["w"], torch.arange(6, dtype=torch.float32).reshape(2, 3))
