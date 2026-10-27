"""The structured reply contract: optional `## improved` / `## questions`
sections a chat model may append after its lead-line prose.

Models are instructed (see ``REPLY_CONTRACT_PROMPT_BLOCK`` below) to answer
with tool calls, at most one short lead line, then these two optional
sections — nothing else. Headers are case-insensitive with an optional
trailing colon, order-tolerant, and may repeat (a repeated header's items are
concatenated into the same section). Bullets are ``-``/``*``/``•``; questions
are numbered (``1.``/``1)``) or bulleted the same way, and may end with a
``[a | b | c]`` options hint. Anything that doesn't parse as a recognized
section is left as prose — this contract must never break a reply.

A `<tool_action>` version tag (the frontend's Apply-block convention, see
``src.features.llm.tools.builtin.form_context_tool``) is opaque payload to
this parser, not a recognized section — it must survive verbatim no matter
where in the reply it lands, including after a `## improved`/`## questions`
header. It is pulled out before line-based parsing (so multi-line proposed
text inside it is never misread as a bullet) and stitched back into
``cleaned`` at the end.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

_HEADER_RE = re.compile(r"^\s*##\s*(improved|questions)\s*:?\s*$", re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*[-*•]\s+(.*\S)\s*$")
_QUESTION_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+(.*\S)\s*$")
_OPTIONS_RE = re.compile(r"\[\s*([^\[\]]+?)\s*\]\s*$")
# Closing tag optional so a generation cut off mid-tag still round-trips
# (the truncated fragment is inert text as far as this parser is concerned).
_TOOL_ACTION_BLOCK_RE = re.compile(
    r"<tool_action\b[^>]*>(?:.*?</tool_action>)?", re.DOTALL | re.IGNORECASE
)
_TAG_PLACEHOLDER = "\x00REPLY_CONTRACT_TAG{}\x00"
_TAG_PLACEHOLDER_RE = re.compile(r"\x00REPLY_CONTRACT_TAG(\d+)\x00")

REPLY_CONTRACT_PROMPT_BLOCK = (
    "\n## Reply format\n\n"
    "1. After tool calls, write at most one short lead line, then `## improved`: terse "
    "bullets of what actually changed. No prose recap; say each thing once.\n"
    "2. A bullet only describes a change. New prompt text is delivered by the "
    "`<tool_action>` tag these instructions define, with the complete text, in this "
    "reply; any other change is delivered by calling its tool. Never invent a tag.\n"
    "3. Add `## questions` only when an answer would sharpen your next reply: up to 3 "
    "numbered questions, with [a | b] choices when natural.\n"
    "4. A quoted question (`> question`) in the user's message is their answer: "
    "continue the task immediately. Never end a turn announcing an action you have not taken.\n\n"
    "Example:\n"
    "Done.\n"
    "## improved\n"
    "- subject anchored: \"lone hiker\" is now \"a lone hiker in a red parka\"\n"
    "## questions\n"
    "1. keep the rain or push golden hour? [rain | golden hour]\n"
)

# Per-send reminder injected right before the last user message (see
# context_builder.inject_reply_contract_reminder_block) — a rule buried near
# the top of a long system prompt loses to recency, so this restates the
# no-prose-recap rule next to the turn it governs. Owned here so one file
# defines the contract's full wording.
REPLY_CONTRACT_REMINDER = (
    "Reply format reminder: tool calls, then `## improved` bullets, no prose recap. "
    "New prompt text goes in its `<tool_action>` tag with the full text; other changes "
    "go through their tool. `## questions` only if needed."
)

TOOL_LOOP_CONTINUATION_NUDGE = (
    "If the task is not finished, call the next tool NOW, in this same "
    "response. Never end a reply announcing an action you have not "
    "performed. When the task IS finished, reply in the reply format "
    "rules — and if that means proposing new prompt/segment/shot text, the "
    "`<tool_action>` tag with the full text belongs in this same reply, not "
    "just a bullet describing it."
)


APPROVAL_CONTINUATION_TEMPLATE = (
    "The user just resolved the pending `{tool_name}` tool call (id {tool_call_id}); "
    "its outcome is that call's tool result in the conversation above, even when "
    "other tool results follow it. Reply with one short sentence saying what was "
    "done, naming the item or change exactly as that result states it (never a "
    "name from an earlier search or another tool result), or that it was declined "
    "or failed. Never answer with only an acknowledgment such as \"OK\"."
)


def approval_continuation_message(tool_name: str, tool_call_id: str) -> Dict[str, str]:
    note = APPROVAL_CONTINUATION_TEMPLATE.format(tool_name=tool_name, tool_call_id=tool_call_id)
    return {"role": "user", "content": f"<context>\n{note}\n</context>"}


def parse_reply_contract(content: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Split ``## improved`` / ``## questions`` sections out of a reply.

    Returns ``(cleaned, reply_contract)``. ``cleaned`` is the text before the
    first recognized header plus any ``<tool_action>`` tag(s) found anywhere
    in the reply (see module docstring), stripped; if nothing parses,
    ``cleaned`` is the original ``content`` untouched. ``reply_contract`` is
    ``{"improved": [...], "questions": [...]}`` with only the keys that
    actually parsed at least one item, or ``None`` when neither did.
    """
    if not content:
        return content, None

    # Tag blocks are stashed behind placeholders before line-based parsing so
    # neither their attribute quoting nor multi-line proposed text is ever
    # read as a header/bullet/question -- and so a tag landing after a header
    # isn't silently discarded along with the rest of that section's
    # unrecognized lines (see "leftover" below).
    tag_blocks: List[str] = []

    def _stash(match: "re.Match[str]") -> str:
        tag_blocks.append(match.group(0))
        return _TAG_PLACEHOLDER.format(len(tag_blocks) - 1)

    working = _TOOL_ACTION_BLOCK_RE.sub(_stash, content)

    # A bullet/question can carry its tag inline on the same line
    # (`- warm palette <tool_action ...>...</tool_action>`) -- _BULLET_RE/
    # _QUESTION_RE capture the whole remainder of the line, placeholder
    # included, so it must be pulled back out of the captured text before
    # that text lands in the structured reply_contract (a raw null-byte
    # token must never reach the frontend), with the real tag it stood for
    # recovered here rather than lost.
    recovered_tag_blocks: List[str] = []

    def _strip_inline_tags(text: str) -> str:
        def _pull(match: "re.Match[str]") -> str:
            idx = int(match.group(1))
            if 0 <= idx < len(tag_blocks):
                recovered_tag_blocks.append(tag_blocks[idx])
            return " "
        return re.sub(r"\s+", " ", _TAG_PLACEHOLDER_RE.sub(_pull, text)).strip()

    lines = working.splitlines()
    first_header_idx: Optional[int] = None
    current_section: Optional[str] = None
    improved_items: List[str] = []
    questions: List[Dict[str, Any]] = []
    leftover_lines: List[str] = []

    for i, line in enumerate(lines):
        header_match = _HEADER_RE.match(line)
        if header_match:
            if first_header_idx is None:
                first_header_idx = i
            current_section = header_match.group(1).lower()
            continue

        if current_section == "improved":
            bullet_match = _BULLET_RE.match(line)
            if bullet_match:
                item_text = _strip_inline_tags(bullet_match.group(1).strip())
                if item_text:
                    improved_items.append(item_text)
            elif line.strip():
                leftover_lines.append(line)
        elif current_section == "questions":
            question_match = _QUESTION_RE.match(line)
            if question_match:
                text = question_match.group(1).strip()
                options: List[str] = []
                options_match = _OPTIONS_RE.search(text)
                if options_match:
                    options = [o.strip() for o in options_match.group(1).split("|") if o.strip()]
                    text = text[:options_match.start()].strip()
                text = _strip_inline_tags(text)
                if text:
                    questions.append({"text": text, "options": options})
            elif line.strip():
                leftover_lines.append(line)

    if not improved_items and not questions:
        return content, None

    cleaned = "\n".join(lines[:first_header_idx]).strip() if first_header_idx is not None else working
    if leftover_lines:
        extra = "\n".join(leftover_lines).strip()
        cleaned = f"{cleaned}\n\n{extra}".strip() if cleaned else extra
    if recovered_tag_blocks:
        extra = "\n".join(recovered_tag_blocks)
        cleaned = f"{cleaned}\n\n{extra}".strip() if cleaned else extra

    for idx, block in enumerate(tag_blocks):
        cleaned = cleaned.replace(_TAG_PLACEHOLDER.format(idx), block)

    reply_contract: Dict[str, Any] = {}
    if improved_items:
        reply_contract["improved"] = improved_items
    if questions:
        reply_contract["questions"] = questions

    return cleaned, reply_contract
