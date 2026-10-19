import json
from pathlib import Path

import pytest

from src.features.prompt.conditions import render_conditions
from src.features.prompt.expander import expand_prompts

CORPUS_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "prompt_conditions"


def _load(name):
    return json.loads((CORPUS_DIR / name).read_text(encoding="utf-8"))


EVALUATE = _load("evaluate.json")["cases"]
EXPAND = _load("expand.json")["cases"]


def test_corpus_case_names_are_unique():
    names = [c["name"] for c in EVALUATE] + [c["name"] for c in EXPAND]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("case", EVALUATE, ids=[c["name"] for c in EVALUATE])
def test_evaluate_case(case):
    text, errors = render_conditions(case["template"], case["values"])
    assert text == case["expected"]
    assert [e.kind for e in errors] == case["errors"]


@pytest.mark.parametrize("case", EXPAND, ids=[c["name"] for c in EXPAND])
def test_expand_case(case):
    warnings = []
    expanded = expand_prompts(
        case["template"],
        case.get("negative", ""),
        count=case["image"] + 1,
        base_seed=case["seed"],
        variables=case.get("variables") or None,
        on_warning=warnings.append,
    )[case["image"]]
    assert expanded.positive == case["expected"]
    assert expanded.negative == case.get("expected_negative", "")
    assert bool(warnings) == case.get("warns", False)
