import sys

from backend.sources import coerce
from backend.sources.markdown import RenderMemo, convert_markdown, paragraphs_html

SHA = "A" * 64


def test_text_strips_and_rejects_blank_or_non_strings():
    assert coerce.text("  hi ") == "hi"
    assert coerce.text("   ") is None
    assert coerce.text(5) is None


def test_int_value_rejects_booleans_and_non_finite_floats():
    assert coerce.int_value(True) is None
    assert coerce.int_value(7) == 7
    assert coerce.int_value(7.9) == 7
    assert coerce.int_value(float("nan")) is None
    assert coerce.int_value("7") is None


def test_int_value_can_insist_on_whole_floats():
    assert coerce.int_value(7.0, whole_floats_only=True) == 7
    assert coerce.int_value(7.5, whole_floats_only=True) is None


def test_strings_dedupes_and_optionally_keeps_whitespace():
    values = [" a ", "a", "b", 3, "", "  "]

    assert coerce.strings(values) == ["a", "b"]
    assert coerce.strings(values, strip=False) == [" a ", "a", "b", "  "]
    assert coerce.strings("not a list") == []


def test_clean_list_sorts_and_dedupes():
    assert coerce.clean_list([" b", "a", "b ", "", "  "]) == ["a", "b"]
    assert coerce.clean_list(None) == []


def test_as_list_only_passes_lists():
    assert coerce.as_list([1]) == [1]
    assert coerce.as_list((1,)) == []


def test_sha256_hex_lowercases_valid_digests_only():
    assert coerce.sha256_hex(SHA) == SHA.lower()
    assert coerce.sha256_hex("abc") is None
    assert coerce.sha256_hex(None) is None


def test_paragraphs_html_escapes_and_splits_on_blank_lines():
    assert paragraphs_html("a <b>\n\n  c  ") == "<p>a &lt;b&gt;</p><p>c</p>"


def test_convert_markdown_returns_none_without_markdown2(monkeypatch):
    monkeypatch.setitem(sys.modules, "markdown2", None)

    assert convert_markdown("# hi", "x") is None


def test_convert_markdown_logs_and_returns_none_on_failure(monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "markdown2", type("M", (), {"markdown": staticmethod(broken)}))

    with caplog.at_level("WARNING", logger="plugins.model-browser"):
        assert convert_markdown("# hi", "Thing: could not convert") is None

    assert "Thing: could not convert; showing it as text" in caplog.text


def test_render_memo_renders_each_text_once_and_evicts_the_oldest():
    calls = []
    memo = RenderMemo(2)

    def render(text):
        calls.append(text)
        return text.upper()

    assert memo.render("a", render) == "A"
    assert memo.render("a", render) == "A"
    memo.render("b", render)
    memo.render("a", render)
    memo.render("c", render)
    memo.render("b", render)

    assert calls == ["a", "b", "c", "b"]


def test_render_memo_clear_forgets_everything():
    calls = []
    memo = RenderMemo(4)
    memo.render("a", calls.append)
    memo.clear()
    memo.render("a", calls.append)

    assert calls == ["a", "a"]


def test_each_source_keeps_its_own_coercion_rules():
    from backend.sources import civitai, huggingface, openmodeldb

    assert civitai._strings([" a ", "a"]) == [" a ", "a"]
    assert huggingface._strings([" a ", "a"]) == ["a"]
    assert openmodeldb._strings([" a ", "a"]) == ["a"]
    assert civitai._int(7.5) == 7
    assert huggingface._int(7.5) == 7
    assert openmodeldb._int(7.5) is None
    assert openmodeldb._int(7.0) == 7
