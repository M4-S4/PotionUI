import json
import sqlite3

import pytest

from src.features.generation.prompt_text import prompt_contains_params, prompt_contains_sql, prompt_texts

ROWS = [
    ("plain", {"prompt": "A Lighthouse at dusk"}, None),
    ("state_only", {"steps": 20}, {"prompt": "lighthouse keeper", "promptSegments": [{"text": "lighthouse"}]}),
    ("negative_only", {"prompt": "a dog", "negative_prompt": "lighthouse"}, {"negativePrompt": "lighthouse"}),
    ("nested", {"nested": {"positive_prompt": "tall lighthouse"}}, None),
    ("segments_only", {"steps": 1}, {"promptSegments": [{"text": "lighthouse"}]}),
    ("none", {"steps": 1}, None),
    ("bad_json", None, None),
]


def python_match(form, state, needle):
    texts = prompt_texts(form) + prompt_texts(state)
    return any(needle.lower() in t.lower() for t in texts)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE generations (id TEXT, form_data TEXT, prompt_state TEXT)")
    for gid, form, state in ROWS:
        c.execute(
            "INSERT INTO generations VALUES (?, ?, ?)",
            (gid, "not json" if form is None else json.dumps(form), json.dumps(state) if state else None),
        )
    return c


@pytest.mark.parametrize("needle", ["lighthouse", "LIGHTHOUSE", "dusk", "dog", "absent"])
def test_sql_and_python_agree(conn, needle):
    sql = f"SELECT id FROM generations g WHERE {prompt_contains_sql('g')}"
    found = {r[0] for r in conn.execute(sql, prompt_contains_params(needle))}
    expected = {
        gid for gid, form, state in ROWS if form is not None and python_match(form, state, needle)
    }
    assert found == expected


def test_prompt_only_in_prompt_state_is_found_and_negative_is_not(conn):
    sql = f"SELECT id FROM generations g WHERE {prompt_contains_sql('g')}"
    found = {r[0] for r in conn.execute(sql, prompt_contains_params("lighthouse"))}
    assert "state_only" in found
    assert "negative_only" not in found
