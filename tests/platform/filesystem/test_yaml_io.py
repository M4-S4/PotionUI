import os

import pytest
import yaml

from src.platform.filesystem import yaml_io

BAD_DOCUMENTS = [
    "a: [1, 2\nb: 3\n",
    "key: value\n  bad: indent\n",
    "a: 1\na: {\n",
    "- x\n- y\nz: 1\n",
    "\tkey: tab\n",
]


@pytest.fixture(autouse=True)
def clean_cache():
    yaml_io.clear_yaml_cache()
    yield
    yaml_io.clear_yaml_cache()


def _safe_error(text):
    with pytest.raises(yaml.YAMLError) as info:
        yaml.load(text, Loader=yaml.SafeLoader)
    return info.value


@pytest.mark.parametrize("text", BAD_DOCUMENTS)
def test_error_text_and_mark_match_pure_loader(text):
    expected = _safe_error(text)
    with pytest.raises(yaml.YAMLError) as info:
        yaml_io.load_yaml(text)
    assert str(info.value) == str(expected)
    got_mark = getattr(info.value, "problem_mark", None)
    want_mark = getattr(expected, "problem_mark", None)
    assert (got_mark.line, got_mark.column) == (want_mark.line, want_mark.column)


def test_stream_and_bytes_sources(tmp_path):
    path = tmp_path / "a.yml"
    path.write_text("a: 1\nb: [x, y]\n", encoding="utf-8")
    with open(path, encoding="utf-8") as handle:
        assert yaml_io.load_yaml(handle) == {"a": 1, "b": ["x", "y"]}
    assert yaml_io.load_yaml(path.read_bytes()) == {"a": 1, "b": ["x", "y"]}
    assert yaml_io.load_yaml_file(path) == {"a": 1, "b": ["x", "y"]}


def test_fallback_without_c_loader(monkeypatch):
    monkeypatch.setattr(yaml_io, "_FAST_LOADER", yaml.SafeLoader)
    assert yaml_io.load_yaml("a: 1") == {"a": 1}
    with pytest.raises(yaml.YAMLError) as info:
        yaml_io.load_yaml(BAD_DOCUMENTS[0])
    assert str(info.value) == str(_safe_error(BAD_DOCUMENTS[0]))


def test_cached_returns_equal_independent_copies(tmp_path):
    path = tmp_path / "o.yml"
    path.write_text("- label: A\n  value: a\n", encoding="utf-8")
    first = yaml_io.load_yaml_cached(path)
    first[0]["label"] = "mutated"
    second = yaml_io.load_yaml_cached(path)
    assert second == [{"label": "A", "value": "a"}]
    assert second is not first


def test_cached_skips_reparse_when_unchanged(tmp_path, monkeypatch):
    path = tmp_path / "o.yml"
    path.write_text("a: 1\n", encoding="utf-8")
    calls = []
    real = yaml_io.load_yaml_file
    monkeypatch.setattr(yaml_io, "load_yaml_file", lambda p: calls.append(p) or real(p))
    yaml_io.load_yaml_cached(path)
    yaml_io.load_yaml_cached(path)
    assert len(calls) == 1


def test_cached_refreshes_on_mtime_change(tmp_path):
    path = tmp_path / "o.yml"
    path.write_text("a: 1\n", encoding="utf-8")
    assert yaml_io.load_yaml_cached(path) == {"a": 1}
    path.write_text("a: 2\n", encoding="utf-8")
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    assert yaml_io.load_yaml_cached(path) == {"a": 2}


def test_cached_refreshes_on_size_change_with_same_mtime(tmp_path):
    path = tmp_path / "o.yml"
    path.write_text("a: 1\n", encoding="utf-8")
    stat = path.stat()
    assert yaml_io.load_yaml_cached(path) == {"a": 1}
    path.write_text("a: 1\nb: 2\n", encoding="utf-8")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert yaml_io.load_yaml_cached(path) == {"a": 1, "b": 2}


def test_cached_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        yaml_io.load_yaml_cached(tmp_path / "missing.yml")


def test_cached_invalid_yaml_not_cached(tmp_path):
    path = tmp_path / "bad.yml"
    path.write_text(BAD_DOCUMENTS[0], encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        yaml_io.load_yaml_cached(path)
    assert not yaml_io._cache


def test_cache_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(yaml_io, "_CACHE_LIMIT", 3)
    for index in range(6):
        path = tmp_path / f"{index}.yml"
        path.write_text(f"n: {index}\n", encoding="utf-8")
        yaml_io.load_yaml_cached(path)
    assert len(yaml_io._cache) == 3
