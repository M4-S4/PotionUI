import pytest

from src.features.llm.harness.think_tags import CLOSED, TEXT, THINKING, Piece, ThinkTagStage


def split(chunks):
    stage = ThinkTagStage()
    pieces = []
    for chunk in chunks:
        pieces.extend(stage.feed(chunk))
    pieces.extend(stage.finish())
    return pieces


def joined(pieces, kind):
    return "".join(piece.text for piece in pieces if piece.kind == kind)


def kinds_collapsed(pieces):
    order = []
    for piece in pieces:
        if not order or order[-1] != piece.kind:
            order.append(piece.kind)
    return order


def every_split(text):
    yield [text]
    for cut in range(1, len(text)):
        yield [text[:cut], text[cut:]]
    yield list(text)


SAMPLES = [
    ("<think>reason</think>answer", "answer", "reason"),
    ("before<think>reason</think>after", "beforeafter", "reason"),
    ("<think>a</think>\n\nanswer", "answer", "a"),
    ("<thinking>deep</thinking>answer", "answer", "deep"),
    ("<thought>hm</thought>answer", "answer", "hm"),
    ("<THINK>loud</THINK>answer", "answer", "loud"),
    ('<think style="x">attr</think>answer', "answer", "attr"),
    ("<think>a</think>mid<think>b</think>end", "midend", "ab"),
    ("<think>a < b and c > d</think>x", "x", "a < b and c > d"),
    ("<think>keep</thin</think>x", "x", "keep</thin"),
    ("plain text only", "plain text only", ""),
    ("a < b and <b>bold</b>", "a < b and <b>bold</b>", ""),
    ("<thinker>no</thinker>", "<thinker>no</thinker>", ""),
    ("stray </think> close", "stray </think> close", ""),
    ("<think>İ</think>answer", "answer", "İ"),
    ("<think>aİb</think>x", "x", "aİb"),
    ("İ<think>İİ</think>İ", "İİ", "İİ"),
    ("<THINK>İ</THINK>x", "x", "İ"),
    ("<think>İİ</thin", "", "İİ</thin"),
    ("<think>a</think>İ<think>İ</think>b", "İb", "aİ"),
    ("<thİnk>x</thİnk>y", "<thİnk>x</thİnk>y", ""),
    ("<thinK>x</thinK>y", "<thinK>x</thinK>y", ""),
]


@pytest.mark.parametrize("raw,text,thinking", SAMPLES)
def test_a_stream_splits_the_same_whichever_way_it_is_chunked(raw, text, thinking):
    for chunks in every_split(raw):
        pieces = split(chunks)

        assert joined(pieces, TEXT) == text, chunks
        assert joined(pieces, THINKING) == thinking, chunks


def test_a_whole_block_yields_thinking_then_text():
    assert kinds_collapsed(split(["<think>reason</think>answer"])) == [THINKING, CLOSED, TEXT]


def test_text_without_tags_is_passed_through_untouched():
    pieces = split(["hello ", "world"])

    assert joined(pieces, TEXT) == "hello world"
    assert all(piece.kind == TEXT for piece in pieces)


def test_leading_whitespace_after_the_closing_tag_is_dropped_across_chunks():
    assert joined(split(["<think>a</think>", "\n", "\n  ", "answer"]), TEXT) == "answer"


def test_leading_whitespace_inside_the_block_is_dropped():
    assert joined(split(["<think>", "\n", "reason</think>x"]), THINKING) == "reason"


def test_an_unclosed_block_at_the_end_of_the_stream_is_all_thinking():
    pieces = split(["<think>still ", "going"])

    assert joined(pieces, THINKING) == "still going"
    assert joined(pieces, TEXT) == ""
    assert pieces[-1].kind == CLOSED


def test_a_held_partial_tag_at_the_end_of_the_stream_is_released_as_text():
    assert joined(split(["answer <thi"]), TEXT) == "answer <thi"


def test_a_partial_closing_tag_at_the_end_of_an_unclosed_block_stays_thinking():
    assert joined(split(["<think>abc</thi"]), THINKING) == "abc</thi"


def test_a_tag_split_inside_its_name_is_not_emitted_early():
    stage = ThinkTagStage()

    assert stage.feed("hi <thi") == [Piece(TEXT, "hi ")]
    assert stage.feed("nk>x") == [Piece(THINKING, "x")]


def test_a_closing_tag_must_match_the_opening_name():
    pieces = split(["<think>a</thinking>b</think>c"])

    assert joined(pieces, THINKING) == "a</thinking>b"
    assert joined(pieces, TEXT) == "c"


def test_a_non_ascii_lookalike_of_a_closing_tag_is_not_held_back():
    pieces = ThinkTagStage().feed("<think>abc</thinK")

    assert joined(pieces, THINKING) == "abc</thinK"


def test_an_overlong_unfinished_tag_is_released_as_text():
    junk = "<think " + "x" * 400

    assert joined(split([junk]), TEXT) == junk


def test_finishing_twice_emits_nothing_the_second_time():
    stage = ThinkTagStage()
    stage.feed("<think>a")

    assert stage.finish() != []
    assert stage.finish() == []
