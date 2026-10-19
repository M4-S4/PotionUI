import importlib.util
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest

from src.features.generation.prompt_text import prompt_contains_params, prompt_contains_sql, prompt_search_sql, prompt_texts

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


SEARCH_ROWS = ROWS + [
    ("accented", {"prompt": "Café by the Lighthouse"}, None),
    ("kelvin", {"prompt": "Kelvin lamp"}, None),
    ("quoted", {"prompt": 'say "hi" twice'}, None),
    ("split", {"prompt": "endword", "positive_prompt": "startword"}, None),
    ("separator", {"prompt": "left\x1fright"}, None),
    ("percent", {"prompt": "50% off_sale"}, None),
]

NEEDLES = [
    "lighthouse", "LIGHTHOUSE", "dusk", "dog", "absent", "li", "LI", "e",
    "café", "CAFÉ", "É", "kelvin", "K", 'say "hi"', '"hi"', 'a"b', '"',
    "dword", "wordstart", "endwordstartword", "endword\x1fstart", "left\x1fright", "\x1f", "t\x1fr", "%", "_", "50% off",
]


def _load_prompt_search_migration(conn):
    class _Database:
        @contextmanager
        def get_cursor(self):
            cursor = conn.cursor()
            yield cursor
            conn.commit()

    path = Path(__file__).resolve().parents[3] / "src/platform/database/migrations/059_generation_prompt_search.py"
    spec = importlib.util.spec_from_file_location("migration_059_for_prompt_text", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.db = _Database()
    return module


@pytest.fixture
def indexed():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE generations (id TEXT PRIMARY KEY, form_data TEXT, prompt_state TEXT)")
    _load_prompt_search_migration(c).up()
    for gid, form, state in SEARCH_ROWS:
        c.execute(
            "INSERT INTO generations VALUES (?, ?, ?)",
            (gid, "not json" if form is None else json.dumps(form), json.dumps(state) if state else None),
        )
    c.commit()
    return c


def _ids(conn, sql, params):
    return {r[0] for r in conn.execute(f"SELECT g.id FROM generations g WHERE {sql}", params)}


@pytest.mark.parametrize("needle", NEEDLES)
def test_side_table_search_equals_the_json_tree_predicate(indexed, needle):
    expected = _ids(indexed, f"({prompt_contains_sql('g')})", prompt_contains_params(needle))

    assert _ids(indexed, *prompt_search_sql(needle, "g")) == expected


@pytest.mark.parametrize("needle,expected", [
    ("lighthouse", {"plain", "state_only", "nested", "accented"}),
    ("li", {"plain", "state_only", "nested", "accented"}),
    ("café", {"accented"}),
    ("CAFÉ", set()),
    ("kelvin", set()),
    ('say "hi"', {"quoted"}),
    ("wordstart", set()),
    ("left\x1fright", {"separator"}),
])
def test_side_table_search_keeps_ascii_case_insensitive_substring_semantics(indexed, needle, expected):
    assert _ids(indexed, *prompt_search_sql(needle, "g")) == expected


def test_terms_holding_the_separator_fall_back_to_the_json_tree_predicate():
    sql, params = prompt_search_sql("left\x1fright", "g")

    assert "json_tree" in sql and "generation_prompt_text" not in sql
    assert params == prompt_contains_params("left\x1fright")
