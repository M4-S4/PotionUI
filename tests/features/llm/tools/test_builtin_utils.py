"""Tests for builtin/utils.py's build_model_field_metadata.

`form_schema.properties` nests fields under a `tabs` root's `children` tree
arbitrarily deep (see test_model_values.py's NESTED_SCHEMA_PROPERTIES) - a
flat `.items()` scan over the top-level dict finds nothing on a schema shaped
like a real preset's.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.features.llm.tools.builtin.utils import build_model_field_metadata, parse_page

NESTED_SCHEMA_PROPERTIES = {
    "tabs": {
        "type": "tabs",
        "children": [
            {
                "type": "tab",
                "title": "Generation",
                "children": [
                    {"type": "select", "name": "speed_profile"},
                    {
                        "type": "section",
                        "children": [
                            {
                                "type": "model",
                                "name": "diffusion_model",
                                "title": "Diffusion Model",
                                "configuration": {"model_type": "diffusion_model"},
                            },
                            {
                                "type": "model",
                                "name": "text_encoder",
                                "configuration": {"model_type": "text_encoder"},
                                "ai_hint": "Pick the matching text encoder.",
                            },
                        ],
                    },
                ],
            },
        ],
    },
}


def _preset_collaborators():
    return MagicMock()


class TestBuildModelFieldMetadata:
    def test_walks_nested_tabs_sections_and_rows(self):
        with patch(
            "src.features.presets.operations.get_form_schema",
            return_value={"form_schema": {"properties": NESTED_SCHEMA_PROPERTIES}},
        ):
            result = build_model_field_metadata(_preset_collaborators(), {"preset": "krea2", "mode": "txt2img"})

        assert set(result) == {"diffusion_model", "text_encoder"}

    def test_label_falls_back_to_name_and_carries_model_type(self):
        with patch(
            "src.features.presets.operations.get_form_schema",
            return_value={"form_schema": {"properties": NESTED_SCHEMA_PROPERTIES}},
        ):
            result = build_model_field_metadata(_preset_collaborators(), {"preset": "krea2", "mode": "txt2img"})

        assert result["diffusion_model"]["label"] == "Diffusion Model"
        assert result["diffusion_model"]["model_type"] == "diffusion_model"

    def test_ai_hint_is_included_only_when_present(self):
        with patch(
            "src.features.presets.operations.get_form_schema",
            return_value={"form_schema": {"properties": NESTED_SCHEMA_PROPERTIES}},
        ):
            result = build_model_field_metadata(_preset_collaborators(), {"preset": "krea2", "mode": "txt2img"})

        assert result["text_encoder"]["ai_hint"] == "Pick the matching text encoder."
        assert "ai_hint" not in result["diffusion_model"]

    def test_ignores_non_model_fields_at_depth(self):
        with patch(
            "src.features.presets.operations.get_form_schema",
            return_value={"form_schema": {"properties": NESTED_SCHEMA_PROPERTIES}},
        ):
            result = build_model_field_metadata(_preset_collaborators(), {"preset": "krea2", "mode": "txt2img"})

        assert "speed_profile" not in result

    def test_no_preset_collaborators_returns_empty(self):
        assert build_model_field_metadata(None, {"preset": "krea2"}) == {}

    def test_no_preset_id_returns_empty(self):
        assert build_model_field_metadata(_preset_collaborators(), {}) == {}


class TestParsePage:
    def test_defaults_when_unset(self):
        assert parse_page({}, 20, 100) == (0, 20)
        assert parse_page({"limit": "", "offset": None}, 20, 100) == (0, 20)

    def test_clamps_limit_and_offset(self):
        assert parse_page({"limit": 1000, "offset": -5}, 20, 100) == (0, 100)
        assert parse_page({"limit": 0}, 20, 100) == (0, 1)

    def test_accepts_numeric_strings(self):
        assert parse_page({"limit": "7", "offset": "3"}, 20, 100) == (3, 7)

    @pytest.mark.parametrize("kwargs", [{"limit": "many"}, {"offset": "x"}, {"limit": [1]}])
    def test_rejects_non_integers(self, kwargs):
        with pytest.raises(ValueError, match="must be integers"):
            parse_page(kwargs, 20, 100)
