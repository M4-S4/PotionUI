import pytest

from src.features.forms.binding import FormBindingError, bind_form
from src.features.presets.templates import FieldTemplate, FormTemplate, ModeTemplate, PresetTemplate


def _field(name, type_, default=None, configuration=None):
    return FieldTemplate(type=type_, name=name, default=default, required=False, configuration=configuration, children=None)


def _preset(fields):
    return PresetTemplate(
        id="preset_1",
        name="Preset One",
        version="1.0.0",
        path="/presets/preset_1",
        modes={"txt2img": ModeTemplate(forms=[FormTemplate(name="custom", fields=fields, default=True, order=0)], pipes=[])},
    )


FIELDS = [
    _field("fit", "segmented", "pad", {"options": [{"label": "Pad", "value": "pad"}, {"label": "Cover", "value": "cover"}]}),
    _field("angle", "segmented", 90, {"options": [90, 180, 270]}),
    _field("anchor", "anchor", "center"),
    _field("fill", "color", "transparent", {"allow_transparent": True}),
    _field("key", "color", "#00b140"),
]


def _bind(form_data):
    return bind_form(_preset(FIELDS), "txt2img", None, form_data, "user_1")


def test_defaults_and_valid_values_bind():
    bound = _bind({"fit": "cover", "angle": "180", "anchor": "Top-Left", "fill": "#ABC", "key": "FF8800"})

    assert bound.values["fit"] == "cover"
    assert bound.values["angle"] == 180
    assert bound.values["anchor"] == "top_left"
    assert bound.values["fill"] == "#aabbcc"
    assert bound.values["key"] == "#ff8800"


def test_untouched_form_keeps_its_defaults():
    bound = _bind({})

    assert bound.values["fit"] == "pad"
    assert bound.values["anchor"] == "center"
    assert bound.values["fill"] == "transparent"


@pytest.mark.parametrize(
    "field,value",
    [
        ("fit", "stretch"),
        ("angle", 45),
        ("anchor", "middle"),
        ("fill", "red"),
        ("key", "transparent"),
        ("key", "#12"),
    ],
)
def test_bad_submitted_values_are_field_errors(field, value):
    with pytest.raises(FormBindingError) as raised:
        _bind({field: value})

    assert field in raised.value.field_errors
