import json

import pytest

from src.features.llm.harness.ndjson import NDJSONFramer
from src.features.llm.harness.sse import SSEFramer, SSERecord


def _feed_all(framer, chunks):
    records = []
    for chunk in chunks:
        records.extend(framer.feed(chunk))
    records.extend(framer.close())
    return records


def _split(raw: bytes, *cuts: int):
    pieces, previous = [], 0
    for cut in cuts:
        pieces.append(raw[previous:cut])
        previous = cut
    pieces.append(raw[previous:])
    return pieces


def _data(records):
    return [record.data for record in records]


def test_a_record_split_across_chunk_boundaries_is_reassembled():
    raw = b"data: hello\n\ndata: world\n\n"

    assert _data(_feed_all(SSEFramer(), _split(raw, 3, 9, 14, 20))) == ["hello", "world"]


def test_several_records_in_one_chunk_all_dispatch():
    assert _data(_feed_all(SSEFramer(), [b"data: a\n\ndata: b\n\ndata: c\n\n"])) == ["a", "b", "c"]


def test_exactly_one_leading_space_is_stripped_from_a_value():
    records = _feed_all(SSEFramer(), [b"data:a\n\ndata: b\n\ndata:  c\n\n"])

    assert _data(records) == ["a", "b", " c"]


def test_multiple_data_lines_join_with_a_newline_into_one_record():
    payload = json.dumps({"choices": [{"delta": {"content": "hello"}}]})
    cut = payload.index('"content"') + len('"content"')
    raw = f"data: {payload[:cut]}\ndata: {payload[cut:]}\n\n".encode()

    records = _feed_all(SSEFramer(), [raw])

    assert len(records) == 1
    assert records[0].data == f"{payload[:cut]}\n{payload[cut:]}"
    assert json.loads(records[0].data)["choices"][0]["delta"]["content"] == "hello"


def test_comment_lines_are_ignored_even_between_data_lines_of_one_record():
    raw = b": keep-alive\n\ndata: first\n: ping\ndata: second\n\n"

    assert _data(_feed_all(SSEFramer(), [raw])) == ["first\nsecond"]


def test_event_name_is_kept_and_other_fields_are_ignored():
    raw = b"event: content_block_delta\nid: 7\nretry: 100\ndata: {}\n\ndata: {}\n\n"

    records = _feed_all(SSEFramer(), [raw])

    assert records == [SSERecord("{}", "content_block_delta"), SSERecord("{}", None)]


def test_a_blank_line_with_no_data_dispatches_nothing_and_clears_the_event_name():
    records = _feed_all(SSEFramer(), [b"event: ping\n\ndata: x\n\n"])

    assert records == [SSERecord("x", None)]


def test_an_empty_data_field_dispatches_an_empty_record():
    assert _data(_feed_all(SSEFramer(), [b"data:\n\n"])) == [""]


@pytest.mark.parametrize("ending", ["\n\n", "\r\n\r\n", "\r\r"], ids=["lf", "crlf", "cr"])
def test_every_line_ending_style_frames_identically(ending):
    raw = f"data: a{ending}data: b{ending}".encode()

    assert _data(_feed_all(SSEFramer(), [raw])) == ["a", "b"]


def test_a_crlf_split_between_chunks_is_one_line_ending():
    chunks = [b"data: a\r", b"\n\r", b"\ndata: b\r\n\r\n"]

    assert _data(_feed_all(SSEFramer(), chunks)) == ["a", "b"]


def test_a_lone_cr_at_a_chunk_end_followed_by_text_is_a_line_ending():
    chunks = [b"data: a\r", b"\rdata: b\r", b"\r"]

    assert _data(_feed_all(SSEFramer(), chunks)) == ["a", "b"]


def test_a_leading_bom_is_stripped_once_and_a_later_one_is_data():
    raw = "\ufeffdata: a\n\ndata: \ufeffb\n\n".encode()

    assert _data(_feed_all(SSEFramer(), [raw])) == ["a", "\ufeffb"]


def test_a_multibyte_character_cut_between_its_bytes_survives():
    raw = "data: café\n\n".encode()
    cut = raw.index("é".encode()) + 1

    assert _data(_feed_all(SSEFramer(), _split(raw, cut))) == ["café"]


def test_unicode_line_separators_inside_data_do_not_split_the_line():
    raw = "data: a\u2028b\x85c\n\n".encode()

    assert _data(_feed_all(SSEFramer(), [raw])) == ["a\u2028b\x85c"]


def test_an_unterminated_record_at_close_is_discarded():
    assert _data(_feed_all(SSEFramer(), [b"data: kept\n\ndata: lost"])) == ["kept"]


def test_an_incomplete_done_sentinel_at_close_is_not_dispatched():
    assert _data(_feed_all(SSEFramer(), [b"data: [DON"])) == []


def test_an_oversized_record_is_dropped_and_the_next_one_dispatches(monkeypatch):
    monkeypatch.setattr(SSEFramer, "MAX_RECORD_BYTES", 100)
    framer = SSEFramer()
    raw = b"data: " + b"x" * 200 + b"\n\ndata: ok\n\n"

    assert _data(_feed_all(framer, [raw])) == ["ok"]
    assert framer.dropped == 1


def test_the_suffix_of_a_dropped_record_never_dispatches(monkeypatch):
    monkeypatch.setattr(SSEFramer, "MAX_RECORD_BYTES", 8)
    raw = b"data: " + b"x" * 20 + b"\n: comment\ndata: [DONE]\n\ndata: ok\n\n"

    assert _data(_feed_all(SSEFramer(), [raw])) == ["ok"]


def test_a_record_over_the_bound_only_cumulatively_is_dropped(monkeypatch):
    monkeypatch.setattr(SSEFramer, "MAX_RECORD_BYTES", 8)
    framer = SSEFramer()

    assert _feed_all(framer, [b"data: 1234\ndata: 56789\n\n"]) == []
    assert framer.dropped == 1


def test_many_empty_data_lines_eventually_trip_the_bound(monkeypatch):
    monkeypatch.setattr(SSEFramer, "MAX_RECORD_BYTES", 100)
    framer = SSEFramer()

    assert _feed_all(framer, [b"data:\n" * 10_000 + b"\n"]) == []
    assert framer.dropped == 1


def test_ndjson_lines_split_across_chunks_and_several_per_chunk():
    raw = b'{"a":1}\n{"b":2}\n{"c":3}\n'

    assert _feed_all(NDJSONFramer(), _split(raw, 4, 9, 12)) == [{"a": 1}, {"b": 2}, {"c": 3}]


def test_ndjson_final_line_without_a_newline_is_parsed_at_close():
    assert _feed_all(NDJSONFramer(), [b'{"a":1}\n{"done":true}']) == [{"a": 1}, {"done": True}]


def test_ndjson_blank_lines_and_crlf_are_tolerated():
    assert _feed_all(NDJSONFramer(), [b'\n{"a":1}\r\n\r\n{"b":2}\r\n']) == [{"a": 1}, {"b": 2}]


def test_ndjson_malformed_line_is_skipped_and_the_stream_continues():
    assert _feed_all(NDJSONFramer(), [b'{"a":1}\n{not json\n{"b":2}\n']) == [{"a": 1}, {"b": 2}]


def test_ndjson_multibyte_character_cut_between_its_bytes_survives():
    raw = json.dumps({"t": "café"}, ensure_ascii=False).encode() + b"\n"
    cut = raw.index("é".encode()) + 1

    assert _feed_all(NDJSONFramer(), _split(raw, cut)) == [{"t": "café"}]


def test_ndjson_unicode_line_separator_inside_a_string_does_not_split_the_object():
    raw = json.dumps({"t": "a\u2028b\x85c"}, ensure_ascii=False).encode() + b"\n"

    assert _feed_all(NDJSONFramer(), [raw]) == [{"t": "a\u2028b\x85c"}]
