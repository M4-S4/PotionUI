"""Plain-language rendering of the generate tab's typed prompt variables.

variable is a named ``${...}`` placeholder the user can reuse across their
prompt; it is either free ``text`` or a ``choice`` of options resolved per a
mode (shuffle once per generation / pin one option / re-roll per image).

Both the ``@form`` resource dump and the ``get_form_state`` tool render this
list into short human sentences the chat model reads — e.g.::

    mood: one of noir, sunlit — shuffles each generation; last roll: sunlit

"""

import json
import random
import re
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Callable, Dict, List, Optional, Tuple

# Caps mirror FormResourceProvider's discipline (small-model payload budget).
_MAX_NAME_CHARS = 60

NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
VALID_MODES = {"shuffle", "pin", "per-image"}
MAX_NUMBER_GRID = 200
MAX_ROLL_ENTRIES = 200
MAX_ROLL_CHARS = 300
MAX_CHOICE_OPTIONS = 200
MAX_OPTION_CHARS = 2000
MAX_TEXT_VALUE_CHARS = 2000
MAX_STEP_PLACES = 6
MAX_NUMBER_EXPONENT = 12
MAX_NUMBER_DIGITS = 12
MAX_NAME_CHARS = _MAX_NAME_CHARS
SUMMARY_OPTION_PREVIEW = 3
SUMMARY_OPTION_CHARS = 24
SUMMARY_VALUE_CHARS = 40
SUMMARY_READ_HINT = (
    "Every option, condition and full value: call get_prompt_variables."
)


def _clip(text: str, limit: int) -> str:
    return text[:limit] + "…" if len(text) > limit else text


def _normalize_when(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    var = raw.get("var")
    values = raw.get("values")
    if not isinstance(var, str) or not var.strip():
        return None
    if not isinstance(values, (list, tuple)):
        return None
    out_values = [
        str(v).strip() for v in values if isinstance(v, (str, int, float)) and str(v).strip()
    ]
    return {"var": var.strip(), "values": out_values}


def normalize_option(raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw, (str, int, float)):
        text = str(raw).strip()
        return {"text": text, "when": None} if text else None
    if isinstance(raw, dict):
        text = raw.get("text")
        if not isinstance(text, (str, int, float)):
            return None
        text = str(text).strip()
        if not text:
            return None
        return {"text": text, "when": _normalize_when(raw.get("when"))}
    return None


def valid_options(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, (list, tuple)):
        return []
    out: List[Dict[str, Any]] = []
    for opt in raw:
        normalized = normalize_option(opt)
        if normalized is not None:
            out.append(normalized)
    return out


def option_texts(options: List[Dict[str, Any]]) -> List[str]:
    return [o["text"] for o in options]


def variable_dependencies(var: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(var, dict) or var.get("type") != "choice":
        return []
    deps: List[str] = []
    for opt in valid_options(var.get("options")):
        when = opt.get("when")
        if when and when["var"] not in deps:
            deps.append(when["var"])
    return deps


def is_downstream(start: str, target: str, variables_by_name: Dict[str, Any]) -> bool:
    seen = set()
    stack = [start]
    while stack:
        current = stack.pop()
        if current == target:
            return True
        if current in seen:
            continue
        seen.add(current)
        stack.extend(variable_dependencies(variables_by_name.get(current)))
    return False


def format_option(option: Dict[str, Any]) -> str:
    when = option.get("when")
    if not when:
        return option["text"]
    return f"{option['text']} (when ${when['var']} = {', '.join(when['values'])})"


def dependency_prefix(deps: List[str]) -> str:
    if not deps:
        return ""
    return f"resolves after {', '.join('$' + d for d in deps)} — "


def _choice_mode_phrase(mode: Any, options: List[str], pinned_index: Any) -> str:
    if mode == "pin":
        if isinstance(pinned_index, int) and 0 <= pinned_index < len(options):
            return f"pinned to {_clip(options[pinned_index], SUMMARY_VALUE_CHARS)}"
        return "pinned"
    if mode == "per-image":
        return "re-rolls independently per image"
    return "shuffles each generation"


def _render_choice(name: str, var: dict) -> Optional[str]:
    options = valid_options(var.get("options"))
    if not options:
        return None
    texts = option_texts(options)
    count = len(texts)
    parts = ["choice"]
    deps = variable_dependencies(var)
    if deps:
        parts.append("resolves after " + ", ".join(f"${d}" for d in deps))
    parts.append(_choice_mode_phrase(var.get("mode"), texts, var.get("pinnedIndex")))
    preview = ", ".join(_clip(t, SUMMARY_OPTION_CHARS) for t in texts[:SUMMARY_OPTION_PREVIEW])
    if count > SUMMARY_OPTION_PREVIEW:
        preview = f"first {SUMMARY_OPTION_PREVIEW}: {preview}"
    conditional = sum(1 for o in options if o.get("when"))
    suffix = f", {conditional} conditional" if conditional else ""
    parts.append(f"{count} {'option' if count == 1 else 'options'}{suffix} ({preview})")
    line = f"{name}: " + ", ".join(parts)
    roll = var.get("lastRoll")
    if isinstance(roll, (str, int, float)) and str(roll).strip():
        line += f"; last roll: {_clip(str(roll).strip(), SUMMARY_VALUE_CHARS)}"
    return line


def _render_number(name: str, var: dict) -> Optional[str]:
    grid = number_grid(var)
    if grid is None:
        return None
    line = f"{name}: number from {grid.format_index(0)} to {grid.format_index(grid.count - 1)}"
    if grid.count > 1:
        line += f" in steps of {format(grid.step, 'f')}"
    mode = var.get("mode")
    if mode == "pin":
        line += f" — pinned to {number_pinned_text(var, grid)}"
    elif mode == "per-image":
        line += " — re-rolls independently per image"
    else:
        line += " — shuffles each generation"
    roll = var.get("lastRoll")
    if isinstance(roll, (str, int, float)) and not isinstance(roll, bool) and str(roll).strip():
        line += f"; last roll: {_clip(str(roll).strip(), SUMMARY_VALUE_CHARS)}"
    return line


def _render_text(name: str, var: dict) -> str:
    value = var.get("value")
    text = str(value).strip() if isinstance(value, (str, int, float)) else ""
    if not text:
        return f"{name}: text, empty"
    return f'{name}: text, {len(text)} chars ("{_clip(text, SUMMARY_VALUE_CHARS)}")'


def render_prompt_variable_lines(variables: Any) -> List[str]:
    """One plain-language sentence per prompt variable, capped and defensive.

    Returns an empty list when ``variables`` is absent, malformed, or contains
    no renderable entries — callers treat that as "no prompt variables".
    """
    if not isinstance(variables, (list, tuple)):
        return []
    lines: List[str] = []
    for var in variables:
        if not isinstance(var, dict):
            continue
        raw_name = var.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            continue
        name = _clip(raw_name.strip(), _MAX_NAME_CHARS)
        if var.get("type") == "choice":
            line = _render_choice(name, var)
            if line is None:
                continue
        elif var.get("type") == "number":
            line = _render_number(name, var)
            if line is None:
                continue
        else:
            line = _render_text(name, var)
        lines.append(line)
    return lines


def variables_as_list(variables: Any, rolls: Any = None) -> List[Dict[str, Any]]:
    if not isinstance(variables, dict):
        return []
    last_rolls = rolls if isinstance(rolls, dict) else {}
    kept_rolls = 0
    out: List[Dict[str, Any]] = []
    for name, definition in variables.items():
        if not isinstance(name, str) or not isinstance(definition, dict):
            continue
        entry = {**definition, "name": name}
        roll = last_rolls.get(name)
        if isinstance(roll, str) and kept_rolls < MAX_ROLL_ENTRIES:
            kept_rolls += 1
            entry["lastRoll"] = roll if len(roll) <= MAX_ROLL_CHARS else roll[: MAX_ROLL_CHARS - 1] + "…"
        out.append(entry)
    return out


def render_prompt_variable_summary(variables: Any, rolls: Any = None) -> List[str]:
    lines = render_prompt_variable_lines(variables_as_list(variables, rolls))
    return [*lines, SUMMARY_READ_HINT] if lines else []


@dataclass(frozen=True)
class NumberGrid:
    minimum: Decimal
    maximum: Decimal
    step: Decimal
    places: int
    count: int

    def format_index(self, index: int) -> str:
        value = self.minimum + self.step * index
        return format(value.quantize(Decimal(1).scaleb(-self.places)), "f")

    def values(self) -> List[str]:
        return [self.format_index(k) for k in range(self.count)]

    def clamp_format(self, value: Decimal) -> str:
        clamped = min(max(value, self.minimum), self.maximum)
        quantum = Decimal(1).scaleb(-self.places)
        return format(clamped.quantize(quantum, rounding=ROUND_HALF_UP), "f")


def _to_decimal(raw: Any) -> Optional[Decimal]:
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return None
    try:
        value = Decimal(str(raw).strip())
    except (InvalidOperation, ValueError):
        return None
    if not value.is_finite() or value.adjusted() >= MAX_NUMBER_EXPONENT:
        return None
    if len(value.as_tuple().digits) > MAX_NUMBER_DIGITS:
        return None
    return value


def _step_places(step: Decimal) -> int:
    exponent = step.as_tuple().exponent
    return max(0, -exponent) if isinstance(exponent, int) else 0


def _parse_number_grid(name: str, definition: Dict[str, Any]) -> Tuple[Optional[NumberGrid], Optional[str]]:
    limits = f"|value| below 10^{MAX_NUMBER_EXPONENT} with at most {MAX_NUMBER_DIGITS} significant digits"
    for field in ("min", "max"):
        if _to_decimal(definition.get(field)) is None:
            return None, (
                f"'{name}': a number variable needs a numeric {field} ({limits}), "
                f"e.g. {{\"type\": \"number\", \"min\": 18, \"max\": 80, \"step\": 1}}."
            )
    step_raw = definition.get("step")
    step = Decimal(1) if step_raw is None else _to_decimal(step_raw)
    if step is None or step <= 0:
        return None, f"'{name}': step must be a number greater than 0 ({limits})."
    low = _to_decimal(definition.get("min"))
    high = _to_decimal(definition.get("max"))
    if low > high:
        return None, f"'{name}': min {low} is greater than max {high}."
    places = _step_places(step)
    if places > MAX_STEP_PLACES:
        return None, f"'{name}': step {step} has more than {MAX_STEP_PLACES} decimal places."
    if _step_places(low) > places:
        return None, f"'{name}': min {low} has more decimal places than step {step}; use a finer step or a rounder min."
    value = definition.get("value")
    if value is not None and _to_decimal(value) is None:
        return None, f"'{name}': a number variable's value must be a number ({limits})."
    count = int((high - low) // step) + 1
    if count > MAX_NUMBER_GRID:
        needed = ((high - low) / (MAX_NUMBER_GRID - 1)).quantize(
            Decimal(1).scaleb(-places), rounding=ROUND_CEILING
        )
        return None, (
            f"'{name}': {low} to {high} in steps of {step} is {count} values, "
            f"more than the limit of {MAX_NUMBER_GRID}. Raise the step to at least {format(needed, 'f')} or narrow the range."
        )
    return NumberGrid(low, high, step, places, count), None


def number_grid_error(name: str, definition: Dict[str, Any]) -> Optional[str]:
    return _parse_number_grid(name, definition)[1]


def number_grid(definition: Dict[str, Any]) -> Optional[NumberGrid]:
    return _parse_number_grid("", definition)[0]


def number_pinned_text(definition: Dict[str, Any], grid: NumberGrid) -> str:
    value = _to_decimal(definition.get("value"))
    return grid.clamp_format(grid.minimum if value is None else value)


def name_error(name: str) -> Optional[str]:
    if not NAME_RE.match(name) or len(name) > MAX_NAME_CHARS:
        return (
            f"'{name}' is not a valid variable name. Use letters, digits, and "
            f"underscores, starting with a letter or underscore, up to {MAX_NAME_CHARS} characters."
        )
    return None


def validate_condition(
    name: str,
    when: Dict[str, Any],
    variables_by_name: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    var = when["var"]
    values = when["values"]

    if not values:
        return None, f"'{name}': when.values must not be empty."

    if var == name:
        return None, f"'{name}': when.var cannot reference '{name}' itself."

    ref = variables_by_name.get(var)
    if not isinstance(ref, dict) or ref.get("type") != "choice":
        choices = [
            n for n, v in variables_by_name.items() if isinstance(v, dict) and v.get("type") == "choice"
        ]
        listing = ", ".join(choices) if choices else "none defined yet"
        return None, (
            f"'{name}': when.var '{var}' is not a choice variable on this tab; "
            f"choice variables: {listing}."
        )

    if is_downstream(var, name, variables_by_name):
        return None, (
            f"'{name}': when.var '{var}' would make a cycle ({var} already resolves after {name})."
        )

    ref_texts = option_texts(valid_options(ref.get("options")))
    bad = [v for v in values if v not in ref_texts]
    if bad:
        return None, (
            f"'{name}': when.values {json.dumps(bad)} are not options of ${var}; "
            f"its options are {', '.join(ref_texts)}."
        )

    return {"var": var, "values": values}, None


def validate_variables_map(variables: Any) -> List[str]:
    errors: List[str] = []
    if variables is None:
        return errors
    if not isinstance(variables, dict):
        return ["variables must be an object mapping name to definition."]

    for raw_name, definition in variables.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            errors.append("Empty variable name.")
            continue
        name = raw_name.strip()
        err = name_error(name)
        if err:
            errors.append(err)
            continue

        if not isinstance(definition, dict):
            errors.append(f"'{name}': definition must be an object.")
            continue

        var_type = definition.get("type")
        if var_type == "text":
            value = definition.get("value")
            if value is not None and not isinstance(value, str):
                errors.append(f"'{name}': a text variable's value must be a string.")
            continue

        if var_type == "number":
            grid_error = number_grid_error(name, definition)
            if grid_error:
                errors.append(grid_error)
                continue
            mode = definition.get("mode") or "shuffle"
            if mode not in VALID_MODES:
                errors.append(f"'{name}': invalid mode '{mode}'. Use shuffle, pin, or per-image.")
            continue

        if var_type != "choice":
            errors.append(f"'{name}': invalid type '{var_type}'. Use text, choice or number.")
            continue

        raw_options = definition.get("options")
        if not isinstance(raw_options, (list, tuple)) or not raw_options:
            errors.append(f"'{name}': a choice variable needs at least one non-empty option.")
            continue
        texts: List[str] = []
        condition_failed = False
        for raw_opt in raw_options:
            normalized = normalize_option(raw_opt)
            if normalized is None:
                continue
            texts.append(normalized["text"])
            when = normalized["when"]
            if when is not None:
                _, condition_error = validate_condition(name, when, variables)
                if condition_error:
                    errors.append(condition_error)
                    condition_failed = True

        if not texts:
            errors.append(f"'{name}': a choice variable needs at least one non-empty option.")
            continue
        if condition_failed:
            continue

        mode = definition.get("mode") or "shuffle"
        if mode not in VALID_MODES:
            errors.append(f"'{name}': invalid mode '{mode}'. Use shuffle, pin, or per-image.")
            continue

        pinned_index = definition.get("pinnedIndex")
        if pinned_index is not None and (
            not isinstance(pinned_index, int) or not (0 <= pinned_index < len(texts))
        ):
            errors.append(f"'{name}': pinnedIndex must be a valid index into options.")

    return errors


def _submit_dependencies(var: Optional[Dict[str, Any]], variables_by_name: Dict[str, Any]) -> List[str]:
    return [dep for dep in variable_dependencies(var) if dep in variables_by_name]


def resolve_variable_order(variables_by_name: Dict[str, Any]) -> List[str]:
    order: List[str] = []
    done: set = set()
    active: set = set()

    def visit(name: str) -> None:
        if name in done or name in active:
            return
        active.add(name)
        for dep in _submit_dependencies(variables_by_name.get(name), variables_by_name):
            visit(dep)
        active.discard(name)
        done.add(name)
        order.append(name)

    for name in variables_by_name:
        visit(name)
    return order


def build_variable_wire_value(var: Dict[str, Any]) -> str:
    if var.get("type") == "number":
        grid = number_grid(var)
        if grid is None:
            return ""
        mode = var.get("mode") or "shuffle"
        if mode == "pin":
            return number_pinned_text(var, grid)
        if grid.count == 1:
            return grid.format_index(0)
        return "{" + "|".join(grid.values()) + "}"

    if var.get("type") != "choice":
        value = var.get("value")
        return value if isinstance(value, str) else ""

    texts = option_texts(valid_options(var.get("options")))
    if not texts:
        return ""

    mode = var.get("mode") or "shuffle"
    pinned_index = var.get("pinnedIndex")
    if mode == "pin" and isinstance(pinned_index, int) and 0 <= pinned_index < len(texts):
        pinned_text = texts[pinned_index]
        if pinned_text:
            return pinned_text

    if len(texts) == 1:
        return texts[0]
    return "{" + "|".join(texts) + "}"


def _option_eligible(option: Dict[str, Any], resolved: Dict[str, str]) -> bool:
    when = option.get("when")
    if not when:
        return True
    actual = resolved.get(when["var"])
    return actual is not None and actual in when["values"]


def build_variables_for_submit(
    variables: Any,
    random_fn: Optional[Callable[[], float]] = None,
) -> Dict[str, str]:
    by_name: Dict[str, Dict[str, Any]] = {}
    if isinstance(variables, (list, tuple)):
        for var in variables:
            if not isinstance(var, dict):
                continue
            name = var.get("name")
            if isinstance(name, str) and name.strip():
                by_name[name.strip()] = var

    rand = random_fn or random.random
    resolved: Dict[str, str] = {}
    wire_map: Dict[str, str] = {}

    for name in resolve_variable_order(by_name):
        var = by_name[name]

        if var.get("type") == "number":
            grid = number_grid(var)
            if grid is None:
                continue
            mode = var.get("mode") or "shuffle"
            if mode == "shuffle":
                text = grid.format_index(min(int(rand() * grid.count), grid.count - 1))
                wire_map[name] = text
                resolved[name] = text
            else:
                text = build_variable_wire_value(var)
                wire_map[name] = text
                if mode == "pin":
                    resolved[name] = text
            continue

        if var.get("type") != "choice":
            value = var.get("value")
            value = value if isinstance(value, str) else ""
            if value:
                wire_map[name] = value
            resolved[name] = value.strip()
            continue

        mode = var.get("mode") or "shuffle"

        if mode == "shuffle":
            eligible = [o for o in valid_options(var.get("options")) if _option_eligible(o, resolved)]
            if not eligible:
                continue
            pick = eligible[int(rand() * len(eligible))]
            wire_map[name] = pick["text"]
            resolved[name] = pick["text"]
            continue

        value = build_variable_wire_value(var)
        if value:
            wire_map[name] = value
        if mode == "pin":
            resolved[name] = value

    return wire_map
