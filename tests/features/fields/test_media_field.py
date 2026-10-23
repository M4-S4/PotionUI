import pytest

from src.features.fields.audio import Audio
from src.features.fields.image import Image
from src.features.fields.media import Media
from src.features.fields.slider import Slider
from src.features.fields.stepper import Stepper
from src.features.fields.video import Video

MEDIA_FIELDS = [Image, Video, Audio, Media]


@pytest.mark.parametrize("field_cls", MEDIA_FIELDS)
def test_default_accept_is_sorted_by_extension_and_has_no_duplicates(field_cls):
    field = field_cls(None)
    accept = field._get_accept_string({}).split(",")

    assert len(accept) == len(set(accept))
    expected = list(dict.fromkeys(field.MIME_TYPES[ext] for ext in sorted(field.SUPPORTED_FORMATS)))
    assert accept == expected


def test_configured_formats_keep_author_order_and_collapse_aliases():
    assert Image(None)._get_accept_string({"formats": [".png", ".jpg", ".jpeg"]}) == "image/png,image/jpeg"


@pytest.mark.parametrize("field_cls,fallback", [(Image, "image/*"), (Video, "video/*"), (Audio, "audio/*"), (Media, "*/*")])
def test_unknown_formats_fall_back_to_the_kind_wildcard(field_cls, fallback):
    assert field_cls(None)._get_accept_string({"formats": [".xyz"]}) == fallback


@pytest.mark.parametrize("field_cls", [Image, Video, Audio, Media])
def test_legacy_errors_name_the_kind(field_cls):
    field = field_cls(None)
    kind = field.KIND

    assert field._validate_legacy("x", {}) == [f"Invalid {kind} data format"]
    errors = field._validate_legacy({"size": 10, "name": "a.zzz"}, {"max_size": 1})
    assert f"Missing {kind} data" in errors
    assert f"Missing {kind} type" in errors
    assert any(error.startswith(f"{kind.capitalize()} size exceeds") for error in errors)
    assert any(error.startswith(f"Unsupported {kind} format") for error in errors)


@pytest.mark.parametrize("field_cls,mime", [(Image, "image/png"), (Video, "video/mp4"), (Audio, "audio/mpeg"), (Media, "application/octet-stream")])
def test_legacy_processing_defaults_per_kind(field_cls, mime):
    field = field_cls(None)
    processed = field._process_legacy({"data": "aGk="})

    assert processed["data"] == b"hi"
    assert processed["type"] == mime
    assert processed["name"] == field.KIND


def test_only_video_and_image_check_dimensions_and_only_video_and_audio_check_duration():
    item = {"data": "x", "type": "t", "width": 99, "height": 99, "duration": 99}
    rules = {"max_width": 1, "max_height": 1, "max_duration": 1}

    assert len(Image(None)._validate_legacy(item, rules)) == 2
    assert len(Video(None)._validate_legacy(item, rules)) == 3
    assert len(Audio(None)._validate_legacy(item, rules)) == 1
    assert Media(None)._validate_legacy(item, rules) == []


def test_stepper_schema_matches_slider_for_the_same_field():
    field = {"type": "slider", "name": "n", "label": "N", "configuration": {"min": 1, "max": 9, "step": 2, "tooltip": "t"}}

    assert Stepper(None).output(field) == Slider(None).output(field)
    assert Stepper(None).configuration()[0].example == 1
    assert Slider(None).configuration()[0].example == 0


def _field(field_cls, configuration):
    return {"type": field_cls.KIND, "name": "f", "label": "F", "configuration": configuration}


@pytest.mark.parametrize("field_cls", MEDIA_FIELDS)
def test_configuration_validation_reaches_the_schema_with_only_the_keys_the_kind_has(field_cls):
    rules = {"max_size": 123, "max_width": 640, "max_height": 480, "max_duration": 9, "formats": [".x"]}

    schema = field_cls(None).output(_field(field_cls, {"validation": rules}))["validation"]

    expected = {"maxSize": 123, "formats": [".x"]}
    if field_cls.HAS_DIMENSIONS:
        expected.update(maxWidth=640, maxHeight=480)
    if field_cls.HAS_DURATION:
        expected.update(maxDuration=9)
    assert schema == expected


@pytest.mark.parametrize("field_cls", MEDIA_FIELDS)
def test_validation_defaults_use_the_kind_limits(field_cls):
    schema = field_cls(None).output(_field(field_cls, {"validation": {"max_width": 1}}))["validation"]

    assert schema["maxSize"] == field_cls.DEFAULT_MAX_SIZE
    assert schema["formats"] == sorted(field_cls.SUPPORTED_FORMATS)


@pytest.mark.parametrize("field_cls", MEDIA_FIELDS)
def test_allow_inpaint_is_passed_through_only_when_configured(field_cls):
    field = field_cls(None)

    assert field.output(_field(field_cls, {"allow_inpaint": True}))["configuration"]["allow_inpaint"] is True
    assert "allow_inpaint" not in field.output(_field(field_cls, {})).get("configuration", {})
