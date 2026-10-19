import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, List, Mapping, Optional, Sequence, Set, Tuple

SENTINEL_OPEN = ""
SENTINEL_CLOSE = ""

MAX_SCAN_CHARS = 20_000
MAX_BLOCKS = 32
MAX_BRANCHES = 8
MAX_HEAD_CHARS = 256
MAX_TESTS = 6
MAX_VALUES = 12
MAX_NUMBER_CHARS = 20
MAX_STRING_CHARS = 80
MAX_NAME_CHARS = 60
MAX_VALUE_NUMBER_CHARS = 40
MAX_SCRUB_PASSES = 8

_WS = r"[ \t\r\n]"
_OPEN = re.compile(rf"<<(if|else{_WS}+if){_WS}+")
_JUNK = re.compile(rf"<<(?:if|else|end)(?={_WS}|>)(?:[^>\r\n]{{0,60}}>>)?")
_HEAD = re.compile(r'(?:[^">\\]|\\.|"(?:[^"\\]|\\.)*"|>(?!>))*', re.DOTALL)
_WS_RUN = re.compile(rf"{_WS}+")
_VAR = re.compile(rf"\$([A-Za-z_][A-Za-z0-9_]{{0,{MAX_NAME_CHARS - 1}}})(?![A-Za-z0-9_])")
_NUM = re.compile(r"-?[0-9]+(?:\.[0-9]+)?")
_STR = re.compile(r'"((?:[^"\\]|\\["\\])*)"')
_IS_EMPTY = re.compile(rf"is{_WS}+(not{_WS}+)?empty(?![A-Za-z0-9_])")
_OP = re.compile(
    rf"(?:not{_WS}+(?:in|contains)|in|contains|<=|>=|!=|=|<|>)(?![A-Za-z0-9_])"
)
_TAIL_TAG = re.compile(rf"<<(?:end>>|else>>|(?:if|else|end)(?={_WS}|>))")
_JOINER = re.compile(r"(and|or)(?![A-Za-z0-9_])")
_UNESCAPE = re.compile(r"\\(.)", re.DOTALL)
_NUMERIC_VALUE = re.compile(r"-?[0-9]+(?:\.[0-9]+)?")
_SENTINEL = re.compile(SENTINEL_OPEN + r"([0-9]+)" + SENTINEL_CLOSE)
_SENTINEL_CHARS = str.maketrans("", "", SENTINEL_OPEN + SENTINEL_CLOSE)


@dataclass(frozen=True)
class Predicate:
    var: str
    op: str
    kind: str
    number: Optional[Decimal] = None
    low: Optional[Decimal] = None
    high: Optional[Decimal] = None
    strings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Cond:
    joiner: str
    tests: Tuple[Predicate, ...]


@dataclass(frozen=True)
class Branch:
    kind: str
    cond: Optional[Cond]
    body: str


@dataclass(frozen=True)
class Block:
    index: int
    branches: Tuple[Branch, ...]


@dataclass(frozen=True)
class CondError:
    kind: str
    offset: int


@dataclass(frozen=True)
class ParsedTemplate:
    skeleton: str
    blocks: Tuple[Block, ...]
    errors: Tuple[CondError, ...]

    def reads(self) -> Set[str]:
        names: Set[str] = set()
        for block in self.blocks:
            for branch in block.branches:
                if branch.cond is not None:
                    names.update(test.var for test in branch.cond.tests)
        return names


@dataclass(frozen=True)
class _Tok:
    kind: str
    start: int
    end: int
    head: str = ""


class _HeadError(Exception):
    pass


def _skip_ws(head: str, pos: int) -> int:
    m = _WS_RUN.match(head, pos)
    return m.end() if m else pos


def _need_ws(head: str, pos: int) -> int:
    m = _WS_RUN.match(head, pos)
    if not m:
        raise _HeadError
    return m.end()


def _number(head: str, pos: int) -> Tuple[Decimal, int]:
    m = _NUM.match(head, pos)
    if not m or len(m.group(0)) > MAX_NUMBER_CHARS:
        raise _HeadError
    return Decimal(m.group(0)), m.end()


def _string(head: str, pos: int) -> Tuple[str, int]:
    m = _STR.match(head, pos)
    if not m:
        raise _HeadError
    value = _UNESCAPE.sub(lambda e: e.group(1), m.group(1))
    if not 1 <= len(value) <= MAX_STRING_CHARS:
        raise _HeadError
    return value, m.end()


def _string_list(head: str, pos: int) -> Tuple[Tuple[str, ...], int]:
    values: List[str] = []
    while True:
        value, pos = _string(head, pos)
        values.append(value)
        if len(values) > MAX_VALUES:
            raise _HeadError
        probe = _skip_ws(head, pos)
        if probe < len(head) and head[probe] == ",":
            pos = _skip_ws(head, probe + 1)
            continue
        return tuple(values), pos


def _parse_test(head: str, pos: int) -> Tuple[Predicate, int]:
    m = _VAR.match(head, pos)
    if not m:
        raise _HeadError
    var = m.group(1)
    pos = _need_ws(head, m.end())

    m = _IS_EMPTY.match(head, pos)
    if m:
        return Predicate(var, "is not empty" if m.group(1) else "is empty", "none"), m.end()

    m = _OP.match(head, pos)
    if not m:
        raise _HeadError
    op = " ".join(m.group(0).split())
    pos = _need_ws(head, m.end())

    if op in ("<", "<=", ">", ">="):
        number, pos = _number(head, pos)
        return Predicate(var, op, "number", number=number), pos

    if op in ("=", "!="):
        if head.startswith('"', pos):
            value, pos = _string(head, pos)
            return Predicate(var, op, "string", strings=(value,)), pos
        number, pos = _number(head, pos)
        return Predicate(var, op, "number", number=number), pos

    if op in ("contains", "not contains"):
        value, pos = _string(head, pos)
        return Predicate(var, op, "string", strings=(value,)), pos

    if head.startswith('"', pos):
        values, pos = _string_list(head, pos)
        return Predicate(var, op, "strings", strings=values), pos

    low, pos = _number(head, pos)
    if not head.startswith("..", pos):
        raise _HeadError
    high, pos = _number(head, pos + 2)
    if low > high:
        raise _HeadError
    return Predicate(var, op, "range", low=low, high=high), pos


def parse_condition(head: str) -> Optional[Cond]:
    try:
        tests: List[Predicate] = []
        joiner: Optional[str] = None
        pos = 0
        while True:
            test, pos = _parse_test(head, pos)
            tests.append(test)
            if len(tests) > MAX_TESTS:
                return None
            if pos >= len(head):
                break
            pos = _need_ws(head, pos)
            m = _JOINER.match(head, pos)
            if not m or (joiner is not None and m.group(1) != joiner):
                return None
            joiner = m.group(1)
            pos = _need_ws(head, m.end())
        return Cond(joiner or "and", tuple(tests))
    except _HeadError:
        return None


def _as_number(value: str) -> Optional[Decimal]:
    text = value.strip()
    if len(text) > MAX_VALUE_NUMBER_CHARS or not _NUMERIC_VALUE.fullmatch(text):
        return None
    return Decimal(text)


def _eval_test(test: Predicate, values: Mapping[str, str]) -> bool:
    value = values.get(test.var)
    if value is None:
        return False
    op = test.op
    if op == "is empty":
        return value.strip() == ""
    if op == "is not empty":
        return value.strip() != ""

    if test.kind == "number":
        n = _as_number(value)
        if n is None:
            return False
        target = test.number
        if op == "=":
            return n == target
        if op == "!=":
            return n != target
        if op == "<":
            return n < target
        if op == "<=":
            return n <= target
        if op == ">":
            return n > target
        return n >= target

    if test.kind == "range":
        n = _as_number(value)
        if n is None:
            return False
        inside = test.low <= n <= test.high
        return inside if op == "in" else not inside

    if test.kind == "string":
        needle = test.strings[0]
        if op == "=":
            return value.strip() == needle
        if op == "!=":
            return value.strip() != needle
        found = needle.casefold() in value.casefold()
        return found if op == "contains" else not found

    trimmed = value.strip()
    hit = trimmed in test.strings
    return hit if op == "in" else not hit


def evaluate_condition(cond: Cond, values: Mapping[str, str]) -> bool:
    if cond.joiner == "or":
        return any(_eval_test(t, values) for t in cond.tests)
    return all(_eval_test(t, values) for t in cond.tests)


def choose_branch(block: Block, values: Mapping[str, str]) -> Tuple[Optional[int], str]:
    for position, branch in enumerate(block.branches):
        if branch.kind == "else":
            return position, branch.body
        if branch.cond is not None and evaluate_condition(branch.cond, values):
            return position, branch.body
    return None, ""


def _read_tag(text: str, i: int) -> Optional[_Tok]:
    if text.startswith("<<end>>", i):
        return _Tok("end", i, i + 7)
    if text.startswith("<<else>>", i):
        return _Tok("else", i, i + 8)
    m = _OPEN.match(text, i)
    if m:
        limit = min(len(text), m.end() + MAX_HEAD_CHARS + 2)
        hm = _HEAD.match(text, m.end(), limit)
        if text.startswith(">>", hm.end()) and hm.end() - m.end() <= MAX_HEAD_CHARS:
            kind = "if" if m.group(1) == "if" else "elseif"
            return _Tok(kind, i, hm.end() + 2, text[m.end():hm.end()].strip())
        return _Tok("junk", i, m.end())
    j = _JUNK.match(text, i)
    if j:
        return _Tok("junk", i, j.end())
    return None


def _tokenize(text: str) -> Tuple[List[_Tok], int]:
    toks: List[_Tok] = []
    pos = 0
    while True:
        i = text.find("<<", pos)
        if i < 0 or i >= MAX_SCAN_CHARS:
            return toks, pos
        tag = _read_tag(text, i)
        if tag is None:
            pos = i + 1
        else:
            toks.append(tag)
            pos = tag.end


def _scrub(text: str) -> str:
    for _ in range(MAX_SCRUB_PASSES):
        toks, _ = _tokenize(text)
        if not toks:
            break
        pieces: List[str] = []
        cursor = 0
        for tok in toks:
            pieces.append(text[cursor:tok.start])
            cursor = tok.end
        pieces.append(text[cursor:])
        text = "".join(pieces)
    return text


def parse_template(text: str) -> Optional[ParsedTemplate]:
    if "<<" not in text:
        return None
    text = text.translate(_SENTINEL_CHARS)
    toks, tail_start = _tokenize(text)
    tail = _TAIL_TAG.search(text, tail_start)
    if not toks and tail is None:
        return None

    errors: List[CondError] = []
    valid: List[Tuple[int, int]] = []
    cur: Optional[List] = None

    for idx, tok in enumerate(toks):
        kind = tok.kind
        if kind == "junk":
            errors.append(CondError("malformed_tag", tok.start))
            continue
        if kind == "if":
            if cur is not None:
                errors.append(CondError("nested", tok.start))
            cur = [idx, False, False, 1]
        elif kind == "elseif":
            if cur is None:
                errors.append(CondError("stray", tok.start))
                continue
            if cur[1]:
                cur[2] = True
                errors.append(CondError("stray", tok.start))
            cur[3] += 1
        elif kind == "else":
            if cur is None:
                errors.append(CondError("stray", tok.start))
                continue
            if cur[1]:
                cur[2] = True
                errors.append(CondError("stray", tok.start))
            cur[1] = True
            cur[3] += 1
        else:
            if cur is None:
                errors.append(CondError("stray", tok.start))
                continue
            if cur[3] > MAX_BRANCHES or len(valid) >= MAX_BLOCKS:
                cur[2] = True
                errors.append(CondError("limit", toks[cur[0]].start))
            if not cur[2]:
                valid.append((cur[0], idx))
            cur = None
    if cur is not None:
        errors.append(CondError("unclosed", toks[cur[0]].start))

    out: List[str] = []
    blocks: List[Block] = []
    cursor = 0
    idx = 0
    vi = 0
    while idx < len(toks):
        tok = toks[idx]
        if vi < len(valid) and valid[vi][0] == idx:
            lo, hi = valid[vi]
            vi += 1
            out.append(text[cursor:tok.start])
            block_index = len(blocks)
            blocks.append(_build_block(text, toks, lo, hi, block_index, errors))
            out.append(f"{SENTINEL_OPEN}{block_index}{SENTINEL_CLOSE}")
            cursor = toks[hi].end
            idx = hi + 1
            continue
        out.append(text[cursor:tok.start])
        cursor = tok.end
        idx += 1
    out.append(text[cursor:] if tail is None else _TAIL_TAG.sub("", text[cursor:]))
    if tail is not None:
        errors.append(CondError("limit", tail.start()))
    errors.sort(key=lambda e: e.offset)
    return ParsedTemplate(_scrub("".join(out)), tuple(blocks), tuple(errors))


def _build_block(
    text: str, toks: Sequence[_Tok], lo: int, hi: int, index: int, errors: List[CondError]
) -> Block:
    branches: List[Branch] = []
    head_tok: Optional[_Tok] = None
    parts: List[str] = []
    cursor = 0

    def flush() -> None:
        if head_tok is None:
            return
        cond = None
        if head_tok.kind != "else":
            cond = parse_condition(head_tok.head)
            if cond is None:
                errors.append(CondError("bad_head", head_tok.start))
        branches.append(Branch(head_tok.kind, cond, _scrub("".join(parts))))

    for idx in range(lo, hi + 1):
        tok = toks[idx]
        if tok.kind == "junk":
            parts.append(text[cursor:tok.start])
            cursor = tok.end
            continue
        if tok.kind == "end":
            parts.append(text[cursor:tok.start])
            flush()
            break
        if head_tok is not None:
            parts.append(text[cursor:tok.start])
            flush()
        head_tok = tok
        parts = []
        cursor = tok.end
    return Block(index, tuple(branches))


def _join_hole(left: str, right: str) -> str:
    if left.endswith(",") and (right == "" or right.startswith(",") or right[0] in "\r\n"):
        if right.startswith(","):
            right = right[1:].lstrip(" \t")
        else:
            left = left[:-1].rstrip(" \t")
    elif left == "" and right.startswith(","):
        right = right[1:].lstrip(" \t")
    if not left or not right or left[-1] in "\r\n" or right[0] in "\r\n,":
        return left + right
    return f"{left} {right}"


def splice(text: str, bodies: Sequence[str]) -> str:
    def fill(m: "re.Match[str]") -> str:
        k = int(m.group(1))
        body = bodies[k] if k < len(bodies) else ""
        return body.translate(_SENTINEL_CHARS) if body.strip(" \t") else m.group(0)

    text = _SENTINEL.sub(fill, text)
    while True:
        m = _SENTINEL.search(text)
        if m is None:
            return text
        left = text[:m.start()].rstrip(" \t")
        right = text[m.end():].lstrip(" \t")
        text = _join_hole(left, right)


def render_parsed(
    parsed: ParsedTemplate,
    values: Mapping[str, str],
    sample_skeleton: Callable[[str], str] = lambda text: text,
    sample_body: Callable[[int, str], str] = lambda index, body: body,
) -> str:
    bodies: List[str] = []
    for block in parsed.blocks:
        _, body = choose_branch(block, values)
        bodies.append(sample_body(block.index, body) if body.strip(" \t") else "")
    return splice(sample_skeleton(parsed.skeleton), bodies)


def render_conditions(text: str, values: Mapping[str, str]) -> Tuple[str, List[CondError]]:
    parsed = parse_template(text)
    if parsed is None:
        return text, []
    return render_parsed(parsed, values), list(parsed.errors)
