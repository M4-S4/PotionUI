from typing import Iterable, List

from src.features.prompt import conditions

_KIND_HINTS = {
    "malformed_tag": "a tag that is not a valid <<if ...>>, <<else if ...>>, <<else>> or <<end>>",
    "nested": "a condition inside another condition (nesting is not supported; use <<else if ...>>)",
    "stray": "an <<else>>, <<else if ...>> or <<end>> with no open <<if ...>>",
    "unclosed": "an <<if ...>> that never reaches <<end>>",
    "limit": (
        f"a block over the limits ({conditions.MAX_TESTS} tests per head, "
        f"{conditions.MAX_BRANCHES} branches, {conditions.MAX_BLOCKS} blocks)"
    ),
    "bad_head": (
        "a condition head the parser cannot read (spaces around every operator, quoted "
        "text values, one of and/or per head, operators that fit the variable type)"
    ),
}

_GRAMMAR = (
    '<<if $age > 65>>grey hair<<else if $age < 25>>youthful<<else>>auburn hair<<end>>'
)


def condition_tag_error(texts: Iterable[str]) -> str:
    problems: List[str] = []
    for text in texts:
        parsed = conditions.parse_template(text or "")
        if parsed is None:
            continue
        for error in parsed.errors:
            hint = _KIND_HINTS.get(error.kind, error.kind)
            problems.append(f"{hint} at character {error.offset}")
    if not problems:
        return ""
    shown = "; ".join(problems[:3])
    return (
        f"The text has malformed condition tags: {shown}. Nothing was proposed. "
        f"Write each condition as {_GRAMMAR} and call the tool again."
    )


def ensure_condition_tags(texts: Iterable[str]) -> None:
    error = condition_tag_error(texts)
    if error:
        raise ValueError(error)
