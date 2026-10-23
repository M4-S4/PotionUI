import json
from decimal import Decimal
from pathlib import Path

import pytest

from src.features.prompt import conditions

_CORPUS = Path(__file__).resolve().parents[2] / "fixtures" / "prompt_conditions"


def _corpus_texts():
    texts = []

    def walk(node):
        if isinstance(node, str) and "<<" in node:
            texts.append(node)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for name in ("evaluate.json", "expand.json"):
        walk(json.loads((_CORPUS / name).read_text(encoding="utf-8")))
    return texts


def _readable_blocks():
    for text in _corpus_texts():
        parsed = conditions.parse_template(text)
        for block in parsed.blocks if parsed else ():
            if all(branch.kind == "else" or branch.cond is not None for branch in block.branches):
                yield block


def test_every_corpus_block_round_trips_through_the_serializer():
    blocks = list(_readable_blocks())
    assert len(blocks) > 100
    for block in blocks:
        text = conditions.format_block(block.branches)
        reparsed = conditions.parse_template(text)
        assert not reparsed.errors, text
        assert reparsed.blocks[0].branches == block.branches, text


@pytest.mark.parametrize(
    "value,written",
    [
        ('12" vinyl', '"12\\" vinyl"'),
        ("C:\\music", '"C:\\\\music"'),
        ("tape >> CD", '"tape >> CD"'),
        ("<<if>>", '"<<if>>"'),
    ],
)
def test_strings_are_escaped_so_they_read_back_exactly(value, written):
    assert conditions.format_string(value) == written
    cond = conditions.Cond("and", (conditions.Predicate("v", "=", "string", strings=(value,)),))
    text = conditions.format_block([conditions.Branch("if", cond, "x")])
    assert conditions.render_conditions(text, {"v": value}) == ("x", [])


def test_a_whole_block_is_written_in_the_stored_grammar():
    branches = [
        conditions.Branch("if", conditions.Cond("or", (
            conditions.Predicate("age", ">", "number", number=Decimal("65")),
            conditions.Predicate("mood", "in", "strings", strings=("calm", "serene")),
        )), "grey"),
        conditions.Branch("elseif", conditions.Cond("and", (
            conditions.Predicate("age", "not in", "range", low=Decimal("-1.50"), high=Decimal("30")),
            conditions.Predicate("name", "is not empty", "none"),
        )), "young"),
        conditions.Branch("else", None, "auburn"),
    ]
    assert conditions.format_block(branches) == (
        '<<if $age > 65 or $mood in "calm", "serene">>grey'
        "<<else if $age not in -1.50..30 and $name is not empty>>young<<else>>auburn<<end>>"
    )


def test_tag_head_reads_the_head_of_an_if_or_else_if_tag_only():
    text = 'a <<if $v = "x >> y">>b<<else if $n>2>>c<<else>>d<<end>>'
    assert conditions.tag_head(text, text.index("<<if")) == '$v = "x >> y"'
    assert conditions.tag_head(text, text.index("<<else if")) == "$n>2"
    assert conditions.tag_head(text, text.index("<<else>>")) is None
    assert conditions.tag_head(text, 0) is None
