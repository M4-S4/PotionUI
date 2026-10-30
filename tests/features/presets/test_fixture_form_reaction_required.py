from src.features.presets.fixture_form import inject_unresolvable_defaults, requires_by_reaction
from src.features.presets.templates import FieldTemplate


def _field(name, type_, reactions=None, required=False):
    return FieldTemplate(name=name, type=type_, required=required, reactions=reactions)


REQUIRE = [{"when": {"field": "task", "equals": "style"}, "then": {"update_validation": {"required": True}}}]


def test_a_reaction_that_requires_the_field_is_detected():
    assert requires_by_reaction(_field("a", "string", REQUIRE)) is True
    assert requires_by_reaction(_field("a", "string", [{"when": {}, "then": {"set_visibility": True}}])) is False
    assert requires_by_reaction(_field("a", "string")) is False


def test_fixture_fills_text_fields_that_a_reaction_requires():
    data = {}
    inject_unresolvable_defaults(
        [_field("needed", "string", REQUIRE), _field("optional", "string"), _field("static", "string", required=True)],
        data,
        "tag",
    )
    assert data == {"needed": "tag placeholder text", "static": "tag placeholder text"}
