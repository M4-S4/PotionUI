import difflib
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from src.features.prompt import conditions
from src.platform.resources.prompt_variables import number_grid, option_texts, valid_options

CONDITION_MARKERS = ("<<", "{%", "{{", "&lt;&lt;")

_SCAN = conditions.MAX_SCAN_CHARS
_HEAD = conditions.MAX_HEAD_CHARS
MAX_CHECKED_CHARS = _SCAN + _HEAD
_OPS = tuple(sorted(conditions.OPERATORS, key=len, reverse=True))
_OP_LIST = ", ".join(conditions.OPERATORS)
_KEYWORDS = {"and", "or", "not", "in", "contains", "is", "empty"}
_SENTINELS = str.maketrans("", "", conditions.SENTINEL_OPEN + conditions.SENTINEL_CLOSE)
_SNIPPET_CHARS = 120
_MAX_PROBLEMS = 12
_LIST_NAMES = 12
_SUGGEST_NAMES = 200
_SUGGEST_LOOKUPS = 5
_SUGGEST_LENGTH_WINDOW = 3
_OPTION_CHARS = 60
_OPTIONS_CHARS = 400
_FIX_CHARS = 600
_STATE_MAX_VARIABLES = 16
_STATE_OPTION_CHARS = 120
_STATE_MAX_LINE_CHARS = 400
_STATE_MAX_CHARS = 1200

_ENTITY_HINT = "write <, > and \" as plain characters, not HTML entities like &gt;"
_ENTITY = re.compile(r"&(?:gt|lt|quot|amp);")
_HEAD_OPEN = re.compile(r"<<(if|else if)[ \t]")

_FOREIGN_FORMS = (
    (
        re.compile(r"\{%-?\s*(?:if|elif|elseif|else|endif)\b"),
        "{% if %} is not condition syntax and {% breaks the whole prompt; write <<if $name ...>>...<<end>>",
    ),
    (
        re.compile(r"\{\{\s*(?:[#/]\s*if|else)\b"),
        "{{#if}} is not condition syntax; write <<if $name ...>>...<<end>>",
    ),
    (
        re.compile(r"<<\s*(?:elseif|elif|elsif|else-if)\b", re.IGNORECASE),
        "write <<else if ...>> as two words",
    ),
    (
        re.compile(r"<<\s*(?:endif|end\s+if|end-if|/if)\s*>>", re.IGNORECASE),
        "close a condition with <<end>>",
    ),
    (
        re.compile(r"<<(?:(?:IF|If)[ \t]+\$|(?:ELSE|Else)(?:>>|[ \t]+(?:if|IF|If)[ \t])|(?:END|End)(?:IF|If)?>>)"),
        "condition tags are lowercase: <<if>>, <<else if>>, <<else>>, <<end>>",
    ),
    (
        re.compile(r"<<[ \t]+(?:if[ \t]+\$|else(?:[ \t]*>>|[ \t]+if[ \t])|end[ \t]*>>)"),
        "write the tag without a space after <<, like <<if $age > 65>>",
    ),
    (
        re.compile(r"&lt;&lt;\s*(?:if|else|end)"),
        _ENTITY_HINT,
    ),
)

_FOREIGN_CLOSERS = ("%}", "}}", ">>", "&gt;&gt;")

_KIND_HINTS = {
    "malformed_tag": "not a valid tag; write <<if TEST>>, <<else if TEST>>, <<else>> or <<end>>",
    "nested": "a condition inside another condition; conditions can't nest, add an <<else if ...>> branch instead",
    "stray": "has no open <<if ...>> before it, or comes after the block's <<else>> (<<else>> must be the last branch)",
    "unclosed": "never closed; end the condition with <<end>>",
    "limit": (
        f"over the limits ({conditions.MAX_TESTS} tests per head, "
        f"{conditions.MAX_BRANCHES} branches, {conditions.MAX_BLOCKS} blocks)"
    ),
}
_TOO_LONG = (
    "the text is {length:,} characters; conditions are read only in the first {scan:,} characters and "
    "texts over {checked:,} are not checked, so split it into shorter segments"
)
_TOO_FAR = (
    f"starts after character {_SCAN:,}, too far into the text to check; the generator removes it, "
    "so move the condition earlier or shorten the text before it"
)

_QUOTE = re.compile(r'"(?:[^"\\]|\\.)*(?:(")|\\?\Z)', re.DOTALL)
_JOINER = re.compile(r"(?<![A-Za-z0-9_$])(and|or)(?![A-Za-z0-9_])")
_VAR_START = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*")
_NUMBER = re.compile(conditions.NUMBER_PATTERN)
_RANGE = re.compile(rf"({conditions.NUMBER_PATTERN})\.\.({conditions.NUMBER_PATTERN})")
_SMART_QUOTES = re.compile("[“”„‘’]")
_UPPER_KEYWORD = re.compile(r"(?<![$\w])(?:AND|OR|NOT|IN|CONTAINS|IS|EMPTY|And|Or|Not|In|Contains|Is|Empty)(?!\w)")
_DASH_RANGE = re.compile(r"\bin\s+-?[0-9]+(?:\.[0-9]+)?\s*(?:-|to)\s*-?[0-9]")


def _quote_spans(text: str) -> List[Tuple[int, int, bool]]:
    return [(m.start(), m.end(), m.group(1) is not None) for m in _QUOTE.finditer(text)]


def _masked(head: str) -> str:
    out: List[str] = []
    cursor = 0
    for start, end, closed in _quote_spans(head):
        out.append(head[cursor:start])
        inner = end - start - (2 if closed else 1)
        out.append('"' + "x" * inner + ('"' if closed else ""))
        cursor = end
    out.append(head[cursor:])
    return "".join(out)


def _split_tests(head: str) -> Tuple[List[str], List[str]]:
    masked = _masked(head)
    tests: List[str] = []
    joiners: List[str] = []
    start = 0
    for m in _JOINER.finditer(masked):
        tests.append(head[start:m.start()])
        joiners.append(m.group(1))
        start = m.end()
    tests.append(head[start:])
    return tests, joiners


def _clip(text: str, limit: int = _SNIPPET_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _match_op(rest: str) -> Optional[str]:
    for op in _OPS:
        if rest.startswith(op):
            after = rest[len(op):]
            if op[-1].isalpha() and after[:1].isalnum():
                continue
            return op
    return None


def _value_hint(var: str, op: str, value: str) -> Optional[str]:
    if op in ("is empty", "is not empty"):
        return "nothing may follow is empty / is not empty" if value else None
    if not value:
        return f"{op} needs a value after it"
    if op in ("<", "<=", ">", ">="):
        if value.startswith('"'):
            return f"compare with a bare number, not quoted text: ${var} {op} 65"
        if not _NUMBER.fullmatch(value):
            return f"{op} needs a number, like ${var} {op} 65"
        if len(value) > conditions.MAX_NUMBER_CHARS:
            return f"a number holds at most {conditions.MAX_NUMBER_CHARS} characters"
        return None
    if op in ("in", "not in"):
        span = _RANGE.fullmatch(value)
        if span:
            low, high = span.groups()
            return "write the smaller number first, like 18..30" if Decimal(low) > Decimal(high) else None
        if re.fullmatch(r"-?[0-9.]+\s*(?:-|to)\s*-?[0-9.]+", value):
            return f"write a number range with two dots: ${var} {op} 18..30"
        if value.startswith("["):
            return f'list values without brackets: ${var} {op} "a", "b"'
        if not value.startswith('"'):
            return f'quote each value: ${var} {op} "a", "b" (or use a range like 18..30 for a number)'
    elif op in ("contains", "not contains"):
        if not value.startswith('"'):
            return f'quote the text: ${var} {op} "word"'
    elif not value.startswith('"') and not _NUMBER.fullmatch(value):
        return f'quote text values: ${var} {op} "{_clip(value, 40)}"'
    spans = _quote_spans(value)
    if any(not closed for _, _, closed in spans):
        return 'a value has an unbalanced quote; write \\" for a quote inside a value'
    for start, end, _ in spans:
        if end - start == 2:
            return f"a quoted value can't be empty; use ${var} is empty"
        if end - start - 2 > conditions.MAX_STRING_CHARS:
            return f"a quoted value holds at most {conditions.MAX_STRING_CHARS} characters"
    if value.startswith('"') and len(spans) > conditions.MAX_VALUES:
        return f"a test lists at most {conditions.MAX_VALUES} values"
    return None


def _test_hint(test: str) -> Optional[str]:
    text = test.strip()
    if conditions.parse_condition(text) is not None:
        return None
    if not text:
        return "a test is missing between and/or"
    m = _VAR_START.match(text)
    if not m:
        word = re.match(r"[A-Za-z_][A-Za-z0-9_]*", text)
        if word and word.group(0) not in _KEYWORDS:
            return f"start each test with the variable written ${_clip(word.group(0), 60)}"
        return "start each test with a variable written $name, like $age > 65"
    var = _clip(m.group(0)[1:], 60)
    rest = text[m.end():]
    if not rest:
        return f"${var} needs an operator and a value, like ${var} > 65"
    if not rest[0].isspace():
        if _match_op(rest) is None:
            return "variable names use only letters, digits and _"
        return f"put one space on each side of the operator: ${var} {_match_op(rest)} ..."
    rest = rest.strip()
    op = _match_op(rest)
    if op is None:
        return f"unknown operator; use one of {_OP_LIST}"
    value = rest[len(op):]
    if value and not value[0].isspace():
        return f"put one space on each side of the operator: ${var} {op} ..."
    hint = _value_hint(var, op, value.strip())
    return hint or f"not a valid test; write it like $age > 65 or $mood = \"noir\""


def head_hint(head: str) -> str:
    if not head.strip():
        return "the tag has no test; write a test like $age > 65"
    masked = _masked(head)
    if _SMART_QUOTES.search(head) and '"' not in head:
        return 'use straight double quotes "..." around text, not curly quotes'
    if "'" in masked:
        return 'quote text with double quotes "...", not single quotes'
    if "${" in masked:
        return "inside a tag write $name, not ${name}"
    if "==" in masked:
        return "use = for equals, not =="
    if re.search(r"&(?:gt|lt|quot|amp);", masked):
        return "write <, > and \" as plain characters, not HTML entities like &gt;"
    if "&&" in masked or "||" in masked:
        return "join tests with and / or, not && / ||"
    if "(" in masked or ")" in masked:
        return "parentheses are not supported; put mixed logic in separate <<else if ...>> branches"
    if _UPPER_KEYWORD.search(masked):
        return "keywords are lowercase: and, or, in, not in, contains, is empty"
    if re.search(r"\bbetween\b", masked) or _DASH_RANGE.search(masked):
        return "write a number range with two dots: $age in 18..30"
    tests, joiners = _split_tests(head)
    if len(set(joiners)) > 1:
        return (
            "one branch joins its tests with and or with or, never both; "
            "move the other case to its own <<else if ...>> branch"
        )
    if len(tests) > conditions.MAX_TESTS:
        return f"a branch takes at most {conditions.MAX_TESTS} tests"
    for test in tests:
        hint = _test_hint(test)
        if hint:
            return f"{hint} (in `{_clip(test.strip(), 60)}`)" if len(tests) > 1 and test.strip() else hint
    return f"not a valid test list; write it like $age > 65 or $mood = \"noir\""


def _snippet(text: str, offset: int, closers: Tuple[str, ...] = (">>",)) -> str:
    ends = [
        found + len(closer)
        for closer in closers
        for found in (text.find(closer, offset + 2, offset + _HEAD + 16),)
        if found >= 0
    ]
    piece = text[offset:min(ends)] if ends else text[offset:offset + 40]
    return f"`{_clip(_one_line(piece))}`"


def _syntax_problem(text: str, error: conditions.CondError) -> str:
    snippet = _snippet(text, error.offset)
    if error.kind == "bad_head":
        head = conditions.tag_head(text, error.offset)
        return f"{snippet}: {head_hint(head or '')}"
    if error.kind == "limit" and error.offset >= _SCAN:
        return f"{snippet}: {_TOO_FAR}"
    if error.kind == "malformed_tag" and snippet.startswith(("`<<if", "`<<else if")) and snippet.count('"') % 2:
        return f'{snippet}: a value has an unbalanced quote; write \\" for a quote inside a value, like $record = "12\\" vinyl"'
    return f"{snippet}: {_KIND_HINTS.get(error.kind, error.kind)}"


def _quoted(text: str, limit: int = _OPTION_CHARS) -> str:
    return conditions.format_string(_clip(_one_line(text), limit))


def _options(texts: List[str]) -> str:
    shown = ""
    for index, text in enumerate(texts):
        piece = _quoted(text)
        if index and len(shown) + len(piece) + 2 > _OPTIONS_CHARS:
            return f"{shown} and {len(texts) - index} more"
        shown = f"{shown}, {piece}" if index else piece
    return shown


def variable_kind(definition: Any) -> str:
    kind = definition.get("type") if isinstance(definition, dict) else None
    return kind if kind in ("choice", "number") else "text"


def _never_matches(test: conditions.Predicate, low: Decimal, high: Decimal) -> bool:
    if test.kind == "range":
        return test.op == "in" and (test.high < low or test.low > high)
    n = test.number
    return {
        ">": n >= high,
        ">=": n > high,
        "<": n <= low,
        "<=": n < low,
        "=": n < low or n > high,
    }.get(test.op, False)


@dataclass
class _Variable:
    kind: str
    grid: Any = None
    texts: List[str] = field(default_factory=list)
    known: Set[str] = field(default_factory=set)
    folded: Dict[str, str] = field(default_factory=dict)


class _VariableChecks:
    def __init__(self, variables: Mapping[str, Any]):
        self.variables = variables
        self.cache: Dict[str, Optional[_Variable]] = {}
        self.reported: Set[str] = set()
        self.lookups = 0

    def _info(self, name: str) -> Optional[_Variable]:
        if name not in self.cache:
            definition = self.variables.get(name)
            if definition is None:
                self.cache[name] = None
            else:
                kind = variable_kind(definition)
                info = _Variable(kind)
                if kind == "number":
                    info.grid = number_grid(definition)
                elif kind == "choice":
                    info.texts = option_texts(valid_options(definition.get("options")))
                    info.known = set(info.texts)
                    info.folded = {text.casefold(): text for text in info.texts}
                self.cache[name] = info
        return self.cache[name]

    def _unknown(self, name: str, shown: str) -> Optional[str]:
        if name in self.reported:
            return None
        self.reported.add(name)
        close: List[str] = []
        if self.lookups < _SUGGEST_LOOKUPS:
            self.lookups += 1
            names = [
                other for other in self.variables
                if isinstance(other, str) and abs(len(other) - len(name)) <= _SUGGEST_LENGTH_WINDOW
            ][:_SUGGEST_NAMES]
            close = difflib.get_close_matches(name, names, n=1, cutoff=0.6)
        guess = f"; did you mean ${close[0]}?" if close else "."
        listed = [f"${other}" for other in list(self.variables)[:_LIST_NAMES]]
        more = len(self.variables) - len(listed)
        names_text = ", ".join(listed) + (f" and {more} more" if more > 0 else "") if listed else "none yet"
        return f"`{shown}`: ${name} is not a variable on this tab, so the test is always false{guess} Variables: {names_text}."

    def issue(self, test: conditions.Predicate) -> Optional[str]:
        name = test.var
        info = self._info(name)
        shown = _clip(_one_line(conditions.format_test(test)))
        if info is None:
            return self._unknown(name, shown)
        numeric = test.kind in ("number", "range")
        if info.kind == "number":
            grid = info.grid
            if test.kind in ("string", "strings"):
                sample = f"${name} > {grid.format_index(0)}" if grid else f"${name} > 65"
                return f"`{shown}`: ${name} is a Number variable; compare it with a bare number, like {sample} or ${name} in 18..30."
            if grid is not None and numeric and _never_matches(test, grid.minimum, grid.maximum):
                low, high = grid.format_index(0), grid.format_index(grid.count - 1)
                return f"`{shown}`: ${name} only goes from {low} to {high}, so this test never matches."
            return None
        if numeric:
            if info.kind == "choice":
                return f"`{shown}`: ${name} is a Choice variable; compare it with its options in quotes: {_options(info.texts)}."
            return f'`{shown}`: ${name} is a text variable; compare it with quoted text, like ${name} = "value".'
        if info.kind != "choice" or test.op not in ("=", "!=", "in", "not in"):
            return None
        missing = next((value for value in test.strings if value not in info.known), None)
        if missing is None:
            return None
        exact = info.folded.get(missing.casefold())
        guess = f"; did you mean {_quoted(exact)}?" if exact else "."
        return (
            f"`{shown}`: {_quoted(missing)} is not an option of ${name}{guess} "
            f"Options (exact text, case-sensitive): {_options(info.texts)}."
        )


@dataclass
class Analysis:
    text: str
    parsed: Optional[conditions.ParsedTemplate]
    syntax: List[str]
    issues: List[str]

    @property
    def problems(self) -> List[str]:
        return self.syntax + self.issues


def _entity_tag(scan: str) -> Optional[int]:
    if "&" not in scan:
        return None
    for m in _HEAD_OPEN.finditer(scan):
        stop = min(len(scan), m.end() + _HEAD)
        for closer in ("<<", "\n", ">>"):
            found = scan.find(closer, m.end(), stop)
            if found >= 0:
                stop = found
        if _ENTITY.search(_masked(scan[m.end():stop])):
            return m.start()
    return None


def analyze(text: Optional[str], variables: Optional[Mapping[str, Any]] = None) -> Analysis:
    text = (text or "").translate(_SENTINELS)
    if len(text) > MAX_CHECKED_CHARS:
        problem = _TOO_LONG.format(length=len(text), scan=_SCAN, checked=MAX_CHECKED_CHARS)
        return Analysis(text, None, [problem], [])
    scan = text[:_SCAN]
    found: List[Tuple[int, str]] = []
    for pattern, hint in _FOREIGN_FORMS:
        m = pattern.search(scan)
        if m:
            found.append((m.start(), f"{_snippet(scan, m.start(), _FOREIGN_CLOSERS)}: {hint}"))
    entity = _entity_tag(scan)
    if entity is not None:
        found.append((entity, f"{_snippet(scan, entity, _FOREIGN_CLOSERS)}: {_ENTITY_HINT}"))
    foreign = {offset for offset, _ in found}
    parsed = conditions.parse_template(text) if "<<" in text else None
    issues: List[str] = []
    if parsed is not None:
        errors = [error for error in parsed.errors if error.offset not in foreign][:_MAX_PROBLEMS]
        found.extend((error.offset, _syntax_problem(text, error)) for error in errors)
        if variables is not None:
            checks = _VariableChecks(variables)
            tests = (
                test
                for block in parsed.blocks
                for branch in block.branches
                for test in (branch.cond.tests if branch.cond is not None else ())
            )
            for test in tests:
                issue = checks.issue(test)
                if issue and issue not in issues:
                    issues.append(issue)
                    if len(issues) >= _MAX_PROBLEMS:
                        break
    found.sort(key=lambda item: item[0])
    syntax = [_one_line(problem) for problem in dict.fromkeys(problem for _, problem in found)]
    return Analysis(text, parsed, syntax, issues)


def condition_problems(text: Optional[str], variables: Optional[Mapping[str, Any]] = None) -> List[str]:
    return analyze(text, variables).problems


def describe(parsed: Optional[conditions.ParsedTemplate]) -> List[List[Dict[str, str]]]:
    if parsed is None:
        return []
    return [
        [
            {
                "when": "otherwise" if branch.cond is None else conditions.format_condition(branch.cond),
                "text": branch.body,
            }
            for branch in block.branches
        ]
        for block in parsed.blocks
    ]


_JINJA_TAG = re.compile(
    rf"\{{%-?[ \t]*(if|elif|elseif|else[ \t]+if|else|endif|end)\b((?:(?!-?%\}}|\{{%).){{0,{_HEAD}}})-?%\}}",
    re.DOTALL,
)
_HBS_OPEN = re.compile(rf"\{{\{{[ \t]*#if[ \t]((?:(?!\}}\}}|\{{\{{).){{0,{_HEAD}}})\}}\}}", re.DOTALL)
_HBS_ELSE_IF = re.compile(rf"\{{\{{[ \t]*else[ \t]+if[ \t]((?:(?!\}}\}}|\{{\{{).){{0,{_HEAD}}})\}}\}}", re.DOTALL)
_HBS_ELSE = re.compile(r"\{\{\s*else\s*\}\}")
_HBS_CLOSE = re.compile(r"\{\{\s*/if\s*\}\}")
_ELSE_IF_VARIANT = re.compile(r"<<\s*(?:elseif|elif|elsif|else-if|else\s+if)(?=[ \t])", re.IGNORECASE)
_END_VARIANT = re.compile(r"<<\s*(?:endif|end\s+if|end-if|/if|end)\s*>>", re.IGNORECASE)
_ELSE_VARIANT = re.compile(r"<<\s*else\s*>>", re.IGNORECASE)
_IF_VARIANT = re.compile(r"<<\s*if(?=[ \t])", re.IGNORECASE)
_SPACED_OP = re.compile(r" ?(<=|>=|!=|=|<|>) ?")
_BARE_VAR = re.compile(
    r"(^\s*|(?<![A-Za-z0-9_$])(?:and|or)\s+)(?!(?:and|or|not|in|contains|is|empty)\b)([A-Za-z_][A-Za-z0-9_]*)"
    r"(?=\s*(?:<=|>=|!=|=|<|>)|\s+(?:not\s+)?(?:in|contains)\b|\s+is\b)"
)
_WORD_RANGE = re.compile(
    rf"\b(?:between|in)\s+({conditions.NUMBER_PATTERN})\s*(?:\band\b|\bto\b|-)\s*({conditions.NUMBER_PATTERN})"
)
_BARE_WORD = r"(?!(?:and|or|not|in|contains|is|empty)\b)([A-Za-z_][A-Za-z0-9_\-]*)"
_VALUE_END = r"(?=\s*(?:,|$)|\s+(?:and|or)\b)"
_BARE_VALUE = re.compile(r"((?:^|\s)(?:!=|=|contains|in)\s+)" + _BARE_WORD + _VALUE_END)
_BARE_LIST_ITEM = re.compile(r"(,\s*)" + _BARE_WORD + _VALUE_END)


def _jinja(m: "re.Match[str]") -> str:
    word = " ".join(m.group(1).split())
    head = m.group(2).strip()
    if word == "if":
        return f"<<if {head}>>"
    if word in ("elif", "elseif", "else if"):
        return f"<<else if {head}>>"
    return "<<else>>" if word == "else" else "<<end>>"


def _repair_unquoted(part: str) -> str:
    part = re.sub(r"\s+", " ", part).replace("&gt;", ">").replace("&lt;", "<")
    part = re.sub(r"[\[\]()]", " ", part)
    part = _WORD_RANGE.sub(r"in \1..\2", part)
    part = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", r"$\1", part)
    part = re.sub(r"(?<![$\w])(?:variables|vars|var)\.([A-Za-z_][A-Za-z0-9_]*)", r"$\1", part)
    part = part.replace("!==", "!=").replace("===", "=").replace("==", "=")
    part = part.replace("≠", "!=").replace("≥", ">=").replace("≤", "<=")
    part = part.replace("&&", " and ").replace("||", " or ")
    part = re.sub(r"\s+", " ", _UPPER_KEYWORD.sub(lambda m: m.group(0).lower(), part))
    part = _BARE_VAR.sub(r"\1$\2", part)
    part = _SPACED_OP.sub(r" \1 ", part)
    part = _BARE_VALUE.sub(lambda m: f'{m.group(1)}"{m.group(2)}"', part)
    part = _BARE_LIST_ITEM.sub(lambda m: f'{m.group(1)}"{m.group(2)}"', part)
    return re.sub(r"[ \t]+", " ", part)


def _repair_head(head: str) -> Optional[str]:
    head = head.replace("&quot;", '"')
    if '"' not in head:
        head = re.sub("[“”„]", '"', head)
        head = re.sub("[‘’']([^‘’'\"]*)[‘’']", r'"\1"', head)
    spans = _quote_spans(head)
    if any(not closed for _, _, closed in spans):
        return None
    out: List[str] = []
    cursor = 0
    for start, end, _ in spans:
        out.append(_repair_unquoted(head[cursor:start]))
        out.append(head[start:end])
        cursor = end
    out.append(_repair_unquoted(head[cursor:]))
    return "".join(out).strip()


def _head_end(text: str, start: int) -> Optional[int]:
    limit = min(len(text), start + _HEAD)
    quoted = False
    index = start
    while index < limit:
        char = text[index]
        if char == "\n" or text.startswith("<<", index):
            return None
        if quoted:
            if char == "\\":
                index += 2
                continue
            if char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif text.startswith(">>", index):
            return index
        index += 1
    return None


def _repair_heads(text: str) -> str:
    if ">>" not in text:
        return text
    out: List[str] = []
    cursor = 0
    for m in _HEAD_OPEN.finditer(text):
        if m.start() < cursor:
            continue
        end = _head_end(text, m.end())
        head = _repair_head(text[m.end():end]) if end is not None else None
        if head is None:
            continue
        out.append(text[cursor:m.start()])
        out.append(f"<<{m.group(1)} {head}>>")
        cursor = end + 2
    out.append(text[cursor:])
    return "".join(out)


def _rewrite(text: str) -> str:
    repaired = text.replace("&lt;&lt;", "<<").replace("&gt;&gt;", ">>")
    if "{%" in repaired and "%}" in repaired:
        repaired = _JINJA_TAG.sub(_jinja, repaired)
    if "{{" in repaired and "}}" in repaired:
        repaired = _HBS_OPEN.sub(r"<<if \1>>", repaired)
        repaired = _HBS_ELSE_IF.sub(r"<<else if \1>>", repaired)
        repaired = _HBS_ELSE.sub("<<else>>", repaired)
        repaired = _HBS_CLOSE.sub("<<end>>", repaired)
    repaired = _ELSE_IF_VARIANT.sub("<<else if", repaired)
    repaired = _IF_VARIANT.sub("<<if", repaired)
    repaired = _ELSE_VARIANT.sub("<<else>>", repaired)
    repaired = _END_VARIANT.sub("<<end>>", repaired)
    return _repair_heads(repaired)


class Budget:
    def __init__(self, chars: int = _SCAN):
        self.left = chars

    def take(self, text: str) -> bool:
        cost = min(len(text), _SCAN)
        if cost > self.left:
            return False
        self.left -= cost
        return True


def repair(analysis: Analysis, budget: Budget, variables: Optional[Mapping[str, Any]] = None) -> Optional[Analysis]:
    text = analysis.text
    if not analysis.syntax or len(text) > _SCAN or not any(marker in text for marker in CONDITION_MARKERS):
        return None
    candidate = _rewrite(text)
    if candidate == text or not budget.take(candidate):
        return None
    checked = analyze(candidate, variables)
    return None if checked.syntax else checked


_SEGMENT_ACTION = re.compile(
    r"(<tool_action\b[^>]{0,400}\btype=[\"'](?:update_segment|add_segment|update_director_segment)[\"'][^>]{0,400}>)"
    rf"((?:(?!</tool_action>|<tool_action\b)[\s\S]){{0,{_SCAN}}})(</tool_action>)"
)


def repair_condition_actions(content: str) -> str:
    if not content or "<tool_action" not in content:
        return content
    budget = Budget()

    def fix(m: "re.Match[str]") -> str:
        body = m.group(2)
        if not any(marker in body for marker in CONDITION_MARKERS) or not budget.take(body):
            return m.group(0)
        repaired = repair(analyze(body), budget)
        return m.group(0) if repaired is None else m.group(1) + repaired.text + m.group(3)

    return _SEGMENT_ACTION.sub(fix, content)


def condition_tag_error(texts: Mapping[str, Optional[str]]) -> str:
    checked = [
        (label, text) for label, text in texts.items()
        if text and any(marker in text for marker in CONDITION_MARKERS)
    ]
    total = sum(min(len(text), _SCAN) for _, text in checked)
    if total > _SCAN:
        return (
            f"The condition text is too long to check in one call: {len(checked)} fields with condition tags "
            f"hold {total:,} characters, and at most {_SCAN:,} are checked. Nothing was proposed. "
            "Save fewer or shorter segments at a time."
        )
    budget = Budget(_SCAN - total)
    problems: List[str] = []
    fix: Optional[Tuple[str, str]] = None
    for label, text in checked:
        analysis = analyze(text)
        problems.extend(f"{label}: {problem}" for problem in analysis.syntax)
        if analysis.syntax and fix is None:
            repaired = repair(analysis, budget)
            if repaired is not None:
                fix = (label, repaired.text)
    if not problems:
        return ""
    shown = "; ".join(problem.rstrip(".") for problem in problems[:3])
    more = f" (and {len(problems) - 3} more)" if len(problems) > 3 else ""
    if fix is None:
        corrected = ""
    elif len(fix[1]) <= _FIX_CHARS:
        corrected = f" Corrected {fix[0]}: {fix[1]}"
    else:
        corrected = f" The corrected {fix[0]} is longer than {_FIX_CHARS} characters, so it is not shown; fix the tags listed above."
    return (
        f"The text has malformed condition tags: {shown}{more}. Nothing was proposed.{corrected} "
        "Write each condition like <<if $age > 65>>grey hair<<else if $age < 25>>youthful<<else>>auburn hair<<end>> "
        "and call the tool again."
    )


def ensure_condition_tags(texts: Mapping[str, Optional[str]]) -> None:
    error = condition_tag_error(texts)
    if error:
        raise ValueError(error)


def _variable_line(name: str, definition: Any) -> str:
    kind = variable_kind(definition)
    if kind == "number":
        grid = number_grid(definition)
        if grid is None:
            return f"  ${name} (number): invalid range"
        span = f"{grid.format_index(0)}..{grid.format_index(grid.count - 1)}"
        return f"  ${name} (number): {span}" + (f", step {format(grid.step, 'f')}" if grid.count > 1 else "")
    if kind == "text":
        return f"  ${name} (text)"
    texts = option_texts(valid_options(definition.get("options")))
    line = f"  ${name} (choice):"
    for index, text in enumerate(texts):
        piece = (" " if index == 0 else ", ") + _quoted(text, _STATE_OPTION_CHARS)
        if len(line) + len(piece) > _STATE_MAX_LINE_CHARS:
            return f"{line} and {len(texts) - index} more"
        line += piece
    return line


def condition_variable_lines(variables: Optional[Mapping[str, Any]]) -> List[str]:
    if not isinstance(variables, Mapping) or not variables:
        return []
    lines = ["Variables (exact values for conditions):"]
    used = len(lines[0])
    names = [name for name in variables if isinstance(name, str)]
    for index, name in enumerate(names):
        line = _variable_line(name, variables[name])
        if index >= _STATE_MAX_VARIABLES or used + len(line) + 1 > _STATE_MAX_CHARS:
            lines.append(f"  …and {len(names) - index} more")
            break
        lines.append(line)
        used += len(line) + 1
    return lines
