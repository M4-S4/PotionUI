LITERAL_DOLLAR = "\ue000"
LITERAL_PERCENT = "\ue001"
LITERAL_HASH = "\ue002"

_TO_PLACEHOLDER = {"$": LITERAL_DOLLAR, "%": LITERAL_PERCENT, "#": LITERAL_HASH}
_FROM_PLACEHOLDER = str.maketrans({LITERAL_DOLLAR: "$", LITERAL_PERCENT: "%", LITERAL_HASH: "#"})


def _wrap_commands(text: str) -> set:
    stack = []
    wraps = set()
    i = 0
    n = len(text)
    while i < n:
        pair = text[i:i + 2]
        if pair in ("${", "%{"):
            stack.append([i, pair == "%{", False])
            i += 2
            continue
        c = text[i]
        if c == "{":
            stack.append([i, False, False])
        elif c == "}" and stack:
            start, is_wrap, separated = stack.pop()
            if is_wrap and separated:
                wraps.add(start)
        elif pair == "$$" and stack:
            stack[-1][2] = True
            i += 2
            continue
        i += 1
    return wraps


def protect_literals(text: str) -> str:
    wraps = _wrap_commands(text)
    out = []
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        pair = text[i:i + 2]
        if pair == "${" or (pair == "%{" and i in wraps):
            out.append(pair)
            depth += 1
            i += 2
            continue
        if pair == "$$" and depth > 0:
            out.append(pair)
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth = max(0, depth - 1)
        out.append(_TO_PLACEHOLDER.get(c, c))
        i += 1
    return "".join(out)


def restore_literals(text: str) -> str:
    return text.translate(_FROM_PLACEHOLDER)
