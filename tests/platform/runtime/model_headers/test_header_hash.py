from __future__ import annotations

from src.platform.runtime.model_headers import header_hash, invalid_header_hash, read_header
from tests.fixtures.model_header_fixtures import write_gguf, write_safetensors

TENSORS = {"a.weight": ("F16", (4, 8)), "b.bias": ("F32", (8,))}


def _hash(path):
    return header_hash(read_header(path).view)


def test_the_same_layout_hashes_the_same_whatever_the_order_or_trailing_data(tmp_path):
    first = write_safetensors(tmp_path / "a.safetensors", TENSORS)
    second = write_safetensors(tmp_path / "b.safetensors", dict(reversed(list(TENSORS.items()))))
    second.write_bytes(second.read_bytes() + b"\x00" * 32)

    assert _hash(first) == _hash(second)


def test_a_different_shape_dtype_or_name_changes_the_hash(tmp_path):
    base = _hash(write_safetensors(tmp_path / "a.safetensors", TENSORS))
    shape = _hash(write_safetensors(tmp_path / "b.safetensors", {**TENSORS, "a.weight": ("F16", (4, 9))}))
    dtype = _hash(write_safetensors(tmp_path / "c.safetensors", {**TENSORS, "a.weight": ("BF16", (4, 8))}))
    name = _hash(write_safetensors(tmp_path / "d.safetensors", {**TENSORS, "c.weight": ("F16", (4, 8))}))

    assert len({base, shape, dtype, name}) == 4


def test_metadata_is_part_of_the_hash(tmp_path):
    plain = _hash(write_safetensors(tmp_path / "a.safetensors", TENSORS))
    tagged = _hash(write_safetensors(tmp_path / "b.safetensors", TENSORS, {"tag": "x"}))

    assert plain != tagged


def test_a_gguf_hashes_its_architecture_metadata(tmp_path):
    llama = write_gguf(tmp_path / "a.gguf", TENSORS, {"general.architecture": "llama"})
    flux = write_gguf(tmp_path / "b.gguf", TENSORS, {"general.architecture": "flux"})

    assert _hash(llama) != _hash(flux)


def test_invalid_headers_hash_by_their_error():
    assert invalid_header_hash("bad") == invalid_header_hash("bad")
    assert invalid_header_hash("bad") != invalid_header_hash("worse")
    assert invalid_header_hash(None) == invalid_header_hash("")
