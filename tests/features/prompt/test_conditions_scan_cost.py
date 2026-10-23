import pytest

from src.features.prompt import conditions
from tests.fixtures.condition_scan_work import HEAD_READ_LIMIT, count_head_reads


def _fill(unit, size=conditions.MAX_SCAN_CHARS):
    return (unit * (size // len(unit) + 1))[:size]


FLOODS = {
    "if openers": _fill("<<if "),
    "if backslash openers": _fill("<<if \\"),
    "else if openers": _fill("<<else if "),
    "quote and escape after if": _fill('<<if  "\\"'),
    "unclosed quotes": _fill('<<if $a = "x '),
    "entity heads": _fill("<<if $a &gt; "),
    "heads without a closer": _fill("<<if $a > 1 "),
    "valid tags with bad heads": _fill("<<if $a>1>>x<<end>>"),
    "480 KB of if-end pairs": "<<if<<end>> " * 40000,
    "480 KB of backslash openers": _fill("<<if \\", 480_000),
}


@pytest.mark.parametrize("name", sorted(FLOODS))
def test_each_scan_reads_at_most_one_allowance_of_heads(name, monkeypatch):
    reads = count_head_reads(monkeypatch)
    conditions.parse_template(FLOODS[name])
    assert reads and max(reads) <= HEAD_READ_LIMIT


def test_the_head_read_count_sees_an_unbounded_scan(monkeypatch):
    reads = count_head_reads(monkeypatch)
    monkeypatch.setattr(conditions, "MAX_HEAD_SCAN_CHARS", 10**12)
    conditions.parse_template(FLOODS["if openers"])
    assert max(reads) > 100 * HEAD_READ_LIMIT


def test_head_reading_has_one_allowance_per_scan():
    assert conditions.MAX_HEAD_SCAN_CHARS == 4 * conditions.MAX_SCAN_CHARS
    spent = conditions.MAX_HEAD_CHARS + 2
    still_read = conditions.MAX_HEAD_SCAN_CHARS // spent
    tail = '"' + "y" * (spent + 100) + " <<if $a > 1>>A<<else>>B<<end>>"
    assert conditions.render_conditions("<<if \\" * still_read + tail, {"a": "0"})[0].endswith(" B")
    assert conditions.render_conditions("<<if \\" * (still_read + 1) + tail, {"a": "0"})[0].endswith(">>AB")
