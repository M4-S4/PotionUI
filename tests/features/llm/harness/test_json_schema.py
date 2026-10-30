import pytest

from src.features.llm.harness.json_schema import prune_nulls, strict_schema, validate

NUMBERS = {"type": "array", "items": {"type": "integer"}}


@pytest.mark.parametrize(
    "value, schema, expected",
    [
        ("x", {"type": "string"}, []),
        (1, {"type": "string"}, ["$: expected string, got integer"]),
        (True, {"type": "integer"}, ["$: expected integer, got boolean"]),
        (1.5, {"type": ["integer", "null"]}, ["$: expected integer or null, got number"]),
        (None, {"type": ["integer", "null"]}, []),
        ("c", {"enum": ["a", "b"]}, ['$: must be one of ["a", "b"]']),
        (3, {"const": 4}, ["$: must equal 4"]),
        ("", {"type": "string", "minLength": 1}, ["$: must have at least 1 characters"]),
        ("abc", {"type": "string", "maxLength": 2}, ["$: must have at most 2 characters"]),
        ("abc", {"type": "string", "pattern": "^x"}, ["$: must match the pattern ^x"]),
        (0, {"type": "number", "minimum": 1}, ["$: must be at least 1"]),
        (9, {"type": "number", "maximum": 5}, ["$: must be at most 5"]),
        ([], {"type": "array", "minItems": 1}, ["$: must have at least 1 items"]),
        ([1, 2], {"type": "array", "maxItems": 1}, ["$: must have at most 1 items"]),
        ([1, "a"], NUMBERS, ["$[1]: expected integer, got string"]),
        ({"a": 1}, {"type": "object", "required": ["b"]}, ["$: missing required property 'b'"]),
        ({"a": 1}, {"type": "object", "properties": {}, "additionalProperties": False}, ["$: unexpected property 'a'"]),
        ({"a": "x"}, {"type": "object", "additionalProperties": {"type": "integer"}}, ["$.a: expected integer, got string"]),
        ({"a": {"b": 1}}, {"properties": {"a": {"properties": {"b": {"type": "string"}}}}}, ["$.a.b: expected string, got integer"]),
        (5, {"anyOf": [{"type": "string"}, {"type": "integer"}]}, []),
        (5.5, {"anyOf": [{"type": "string"}, {"type": "integer"}]}, ["$: matches none of the allowed shapes"]),
        (5, {"oneOf": [{"type": "integer"}, {"type": "number"}]}, ["$: must match exactly one of the allowed shapes"]),
        (5, {"allOf": [{"type": "integer"}, {"minimum": 9}]}, ["$: must be at least 9"]),
    ],
)
def test_the_validator_reports_what_is_wrong(value, schema, expected):
    assert validate(value, schema) == expected


def test_references_resolve_against_the_document_root():
    schema = {"type": "array", "items": {"$ref": "#/$defs/n"}, "$defs": {"n": {"type": "integer"}}}
    assert validate([1, 2], schema) == []
    assert validate([1, "x"], schema) == ["$[1]: expected integer, got string"]


def test_recursive_references_follow_the_data_and_loops_without_data_are_refused():
    tree = {"type": "object", "properties": {"kids": {"type": "array", "items": {"$ref": "#"}}}, "required": ["kids"]}
    assert validate({"kids": [{"kids": []}]}, tree) == []
    assert validate({"kids": [{}]}, tree) == ["$.kids[0]: missing required property 'kids'"]
    with pytest.raises(ValueError):
        validate(1, {"$ref": "#/a", "a": {"$ref": "#/a"}})
    with pytest.raises(ValueError):
        validate(1, {"$ref": "https://example.com/schema"})


OPTIONAL = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "subtitle": {"type": "string"},
        "count": {"type": "integer", "minimum": 1},
        "kind": {"type": "string", "enum": ["a", "b"]},
        "meta": {"type": "object", "properties": {"x": {"type": "string"}}},
        "tags": {"type": "array", "items": {"type": "string"}},
        "ref": {"$ref": "#/$defs/Inner"},
        "note": {"type": ["string", "null"]},
    },
    "required": ["title"],
    "$defs": {"Inner": {"type": "object", "properties": {"a": {"type": "string"}}}},
}

NULL = {"type": "null"}


def test_strict_schema_closes_every_object_and_requires_every_property():
    strict = strict_schema(OPTIONAL)
    assert strict["additionalProperties"] is False
    assert strict["required"] == list(OPTIONAL["properties"])
    inner = strict["$defs"]["Inner"]
    assert inner["additionalProperties"] is False and inner["required"] == ["a"]
    meta = strict["properties"]["meta"]["anyOf"][0]
    assert meta["additionalProperties"] is False and meta["required"] == ["x"]


def test_strict_schema_makes_optional_scalars_nullable_in_place():
    properties = strict_schema(OPTIONAL)["properties"]
    assert properties["title"] == {"type": "string"}
    assert properties["subtitle"] == {"type": ["string", "null"]}
    assert properties["count"] == {"type": ["integer", "null"], "minimum": 1}
    assert properties["kind"] == {"type": ["string", "null"], "enum": ["a", "b", None]}


def test_strict_schema_wraps_optional_structures_and_references_with_a_null_branch():
    properties = strict_schema(OPTIONAL)["properties"]
    assert properties["ref"] == {"anyOf": [{"$ref": "#/$defs/Inner"}, NULL]}
    assert properties["tags"] == {"anyOf": [{"type": "array", "items": {"type": "string"}}, NULL]}
    assert properties["meta"]["anyOf"][1] == NULL


def test_strict_schema_leaves_what_is_already_nullable_alone():
    assert strict_schema(OPTIONAL)["properties"]["note"] == {"type": ["string", "null"]}
    already = {"type": "object", "properties": {"a": {"anyOf": [{"type": "string"}, NULL]}}}
    assert strict_schema(already)["properties"]["a"] == {"anyOf": [{"type": "string"}, NULL]}


def test_strict_schema_reaches_objects_inside_arrays_and_combinators():
    schema = {
        "type": "array",
        "items": {"type": "object", "properties": {"a": {"type": "string"}}},
    }
    item = strict_schema(schema)["items"]
    assert item["additionalProperties"] is False and item["properties"]["a"] == {"type": ["string", "null"]}
    union = strict_schema({"anyOf": [{"type": "object", "properties": {"b": {"type": "integer"}}}, {"type": "string"}]})
    assert union["anyOf"][0]["additionalProperties"] is False
    assert union["anyOf"][0]["required"] == ["b"]
    assert union["anyOf"][1] == {"type": "string"}


def test_strict_schema_keeps_free_form_objects_and_does_not_touch_its_input():
    free = {"type": "object", "properties": {"m": {"type": "object"}, "n": {"type": "object", "additionalProperties": {"type": "integer"}}}}
    before = repr(free)
    strict = strict_schema(free)
    assert repr(free) == before
    assert strict["properties"]["m"]["anyOf"][0] == {"type": "object"}
    assert strict["properties"]["n"]["anyOf"][0]["additionalProperties"] == {"type": "integer"}
    snapshot = repr(OPTIONAL)
    strict_schema(OPTIONAL)
    assert repr(OPTIONAL) == snapshot


def test_strict_schema_is_idempotent():
    once = strict_schema(OPTIONAL)
    assert strict_schema(once) == once


def test_a_strict_reply_with_nulls_for_the_missing_fields_is_restored_to_the_original_shape():
    reply = {
        "title": "T", "subtitle": None, "count": None, "kind": None, "meta": None, "tags": None, "ref": None, "note": None,
    }
    assert validate(reply, OPTIONAL) != []
    pruned = prune_nulls(reply, OPTIONAL)
    assert pruned == {"title": "T", "note": None}
    assert validate(pruned, OPTIONAL) == []


def test_pruning_reaches_nested_objects_references_arrays_and_branches():
    nested = {"title": "T", "meta": {"x": None}, "ref": {"a": None}}
    assert prune_nulls(nested, OPTIONAL) == {"title": "T", "meta": {}, "ref": {}}
    listed = {"type": "array", "items": OPTIONAL}
    assert prune_nulls([{"title": "A", "subtitle": None}], listed) == [{"title": "A"}]
    union = {"anyOf": [{"type": "object", "properties": {"b": {"type": "integer"}}}, {"type": "string"}]}
    assert prune_nulls({"b": None}, union) == {}


def test_pruning_never_hides_a_null_in_a_required_property():
    pruned = prune_nulls({"title": None, "subtitle": None}, OPTIONAL)
    assert pruned == {"title": None}
    assert validate(pruned, OPTIONAL) == ["$.title: expected string, got null"]
