from src.features.chat.context_builder import ChatContextBuilder, _truncate_prompt_content

CONDITION = '<<if $age > 65>>grey hair, deep wrinkles<<else if $age < 25 and $mood in "calm", "serene">>youthful, soft skin<<else>>auburn hair<<end>>'


def _block(content):
    history = [{"role": "user", "content": "hi"}]
    ChatContextBuilder.inject_prompt_state_block(
        history, {"segments": [{"index": 0, "id": "s1", "type": "content", "enabled": True, "content": content}]}
    )
    return history[0]["content"]


def test_short_condition_renders_losslessly():
    content = "woman, <<if $age > 65>>grey<<else>>auburn<<end>>"
    assert f'01 [on] id=s1: "{content}"' in _block(content)


def test_golden_prompt_state_line_for_a_long_condition():
    content = "portrait of a woman, " + CONDITION + ", soft window light"
    block = _block(content)
    assert f'01 [on] id=s1: "portrait of a woman, {CONDITION}…" ({len(content)} ch)' in block


def test_cut_never_lands_inside_a_condition_block():
    shown, total, truncated = _truncate_prompt_content("portrait of a woman, " + CONDITION + ", soft light, " + "x " * 80, 100)
    assert shown.endswith("<<end>>")
    assert shown.count("<<if") == shown.count("<<end>>") == 1
    assert truncated and total > len(shown)


def test_block_far_beyond_the_limit_is_dropped_whole_not_cut():
    long_body = "word " * 80
    text = "portrait of a woman, a lone figure standing in the rain, <<if $age > 65>>" + long_body + "<<end>>"
    shown, _, truncated = _truncate_prompt_content(text, 100)
    assert "<<if" not in shown
    assert truncated


def test_text_without_conditions_truncates_as_before():
    shown, total, truncated = _truncate_prompt_content("word " * 60, 100)
    assert truncated and total == 299 and len(shown) <= 100 and not shown.endswith("wor")


BLOCK = "<<if $age > 65>>grey hair<<else>>auburn hair<<end>>"


def test_block_at_offset_zero_far_beyond_four_times_the_limit_is_elided():
    text = "<<if $age > 65>>" + "word " * 120 + "<<end>>"
    shown, total, truncated = _truncate_prompt_content(text, 100)
    assert shown == "<<if …>>"
    assert truncated and total == len(" ".join(text.split()))


def test_unclosed_if_is_cut_before_its_tag():
    shown, _, _ = _truncate_prompt_content("portrait of a woman, standing <<if $age > 65>>grey hair " + "x " * 60, 100)
    assert shown == "portrait of a woman, standing"


def test_unclosed_if_at_offset_zero_is_elided():
    shown, _, _ = _truncate_prompt_content("<<if $age > 65>>grey hair " + "x " * 60, 100)
    assert shown == "<<if …>>"


def test_stray_end_is_kept_and_never_cut_in_half():
    text = "a " * 48 + "<<end>> tail words " + "y " * 30
    shown, _, _ = _truncate_prompt_content(text, 100)
    assert "<<en" not in shown.replace("<<end>>", "")


def test_second_block_straddling_the_limit_is_completed():
    text = "portrait, " + BLOCK + ", mid light, " + BLOCK + ", rim light, " + "x " * 40
    shown, _, truncated = _truncate_prompt_content(text, 100)
    assert shown.count("<<if") == shown.count("<<end>>") == 2
    assert shown.endswith("<<end>>") and truncated


def test_block_ending_exactly_at_the_limit_is_kept():
    prefix = "a" * (100 - len(BLOCK))
    shown, _, truncated = _truncate_prompt_content(prefix + BLOCK + " tail words", 100)
    assert shown == prefix + BLOCK
    assert truncated


def test_oversized_prompt_state_is_cut_on_a_line_boundary():
    segments = [
        {"index": i, "id": f"s{i}", "type": "content", "enabled": True, "name": "N" * 80, "content": "portrait " + BLOCK + " " + "word " * 60}
        for i in range(20)
    ]
    history = [{"role": "user", "content": "hi"}]
    ChatContextBuilder.inject_prompt_state_block(history, {"segments": segments})
    block = history[0]["content"].split("<context>\n", 1)[1].split("\n</context>")[0]
    lines = block.splitlines()
    assert lines[-1].startswith("…(truncated")
    assert all(line.count("<<if") == line.count("<<end>>") for line in lines)
    assert all(line.rstrip().endswith(")") or line.startswith(("PROMPT", "Positive", "…")) for line in lines)
