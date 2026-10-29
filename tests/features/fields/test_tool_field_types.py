import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.features.fields.anchor import Anchor
from src.features.fields.builtin import register_builtin_fields
from src.features.fields.color import Color
from src.features.fields.routes import FieldController, build_router
from src.features.fields.segmented import Segmented
from src.platform.plugins.field_types import FieldTypeRegistry
from src.platform.security.current_user import get_current_active_user

ANCHORS = [
    "center", "top", "bottom", "left", "right",
    "top_left", "top_right", "bottom_left", "bottom_right",
]
NEW_TYPES = {"segmented": Segmented, "anchor": Anchor, "color": Color}


def registry():
    built = FieldTypeRegistry()
    register_builtin_fields(built)
    return built


class TestRegistration(unittest.TestCase):

    def test_the_new_types_are_registered_with_their_classes_and_components(self):
        built = registry()

        for type_name, cls in NEW_TYPES.items():
            definition = built.get(type_name)
            self.assertIs(definition.schema_cls, cls, type_name)
            self.assertEqual(definition.frontend_component, f"core:{cls.__name__}Field", type_name)
            self.assertTrue(definition.shareable, type_name)
            self.assertFalse(definition.container, type_name)
            self.assertIsNone(definition.options_provider, type_name)

    def test_the_manifest_lists_the_types_with_their_configuration_keys(self):
        manifest = {entry["type"]: entry for entry in registry().frontend_manifest()}

        self.assertEqual(
            [spec["name"] for spec in manifest["segmented"]["configuration_schema"]],
            ["options", "fit", "tooltip"],
        )
        self.assertEqual(
            [spec["name"] for spec in manifest["color"]["configuration_schema"]],
            ["allow_transparent", "tooltip"],
        )
        self.assertEqual([spec["name"] for spec in manifest["anchor"]["configuration_schema"]], ["tooltip"])
        self.assertTrue(
            next(spec for spec in manifest["segmented"]["configuration_schema"] if spec["name"] == "options")["required"]
        )

    def test_get_api_fields_types_serves_them(self):
        container = SimpleNamespace(field_controller=FieldController(registry()))
        app = FastAPI()
        app.include_router(build_router(container))
        app.dependency_overrides[get_current_active_user] = lambda: object()

        response = TestClient(app).get("/api/fields/types")

        self.assertEqual(response.status_code, 200)
        by_type = {entry["type"]: entry for entry in response.json()["data"]}
        for type_name, cls in NEW_TYPES.items():
            self.assertEqual(by_type[type_name]["component"], f"core:{cls.__name__}Field")
            self.assertEqual(by_type[type_name]["source"], "core")

    def test_each_type_documents_itself(self):
        for cls in NEW_TYPES.values():
            self.assertTrue(cls.description().strip())
            self.assertTrue(cls.examples())
            for example in cls.examples():
                self.assertIn("type", example.rendered_output)
                self.assertTrue(example.yaml_config)


class TestSegmented(unittest.TestCase):

    def setUp(self):
        self.field = Segmented(Mock())

    def test_output_normalizes_options_and_keeps_the_default(self):
        schema = self.field.output({
            "type": "segmented",
            "name": "fit",
            "label": "Fit",
            "default": "pad",
            "configuration": {
                "options": [
                    {"label": "Pad", "value": "pad"},
                    {"value": "cover", "disabled": True},
                    "stretch",
                    90,
                ],
                "fit": True,
                "tooltip": "How it meets the frame",
            },
        })

        self.assertEqual(schema["type"], "segmented")
        self.assertEqual(schema["title"], "Fit")
        self.assertEqual(schema["default"], "pad")
        self.assertEqual(schema["tooltip"], "How it meets the frame")
        self.assertTrue(schema["fit"])
        self.assertEqual(schema["options"], [
            {"label": "Pad", "value": "pad"},
            {"label": "cover", "value": "cover", "disabled": True},
            {"label": "stretch", "value": "stretch"},
            {"label": "90", "value": 90},
        ])

    def test_output_without_fit_omits_it_and_survives_missing_options(self):
        schema = self.field.output({"type": "segmented", "name": "fit", "configuration": {}})

        self.assertNotIn("fit", schema)
        self.assertEqual(schema["options"], [])

    def test_output_drops_malformed_options(self):
        schema = self.field.output({
            "type": "segmented",
            "name": "x",
            "configuration": {"options": [{"label": "no value"}, {"value": [1]}, True, None, "ok"]},
        })

        self.assertEqual(schema["options"], [{"label": "ok", "value": "ok"}])

    def test_input_accepts_a_declared_value_and_returns_the_declared_type(self):
        config = {"options": [{"label": "90", "value": 90}, {"label": "180", "value": 180}, "x"]}

        self.assertEqual(self.field.input("angle", 90, config), 90)
        self.assertEqual(self.field.input("angle", "180", config), 180)
        self.assertEqual(self.field.input("angle", "x", config), "x")

    def test_input_rejects_a_value_outside_the_options_and_lists_them(self):
        config = {"options": ["pad", "cover"]}

        with self.assertRaises(ValueError) as raised:
            self.field.input("fit", "stretch", config)

        self.assertIn("pad, cover", str(raised.exception))

    def test_input_rejects_a_boolean_for_numeric_options(self):
        with self.assertRaises(ValueError):
            self.field.input("n", True, {"options": [0, 1]})

    def test_input_passes_empty_values_through(self):
        config = {"options": ["a"]}

        self.assertIsNone(self.field.input("fit", None, config))
        self.assertEqual(self.field.input("fit", "  ", config), "  ")


class TestAnchor(unittest.TestCase):

    def setUp(self):
        self.field = Anchor(Mock())

    def test_output_defaults_to_center_and_keeps_a_declared_default(self):
        bare = self.field.output({"type": "anchor", "name": "anchor", "label": "Anchor"})
        declared = self.field.output({"type": "anchor", "name": "anchor", "default": "top_left"})

        self.assertEqual(bare["type"], "anchor")
        self.assertEqual(bare["default"], "center")
        self.assertEqual(declared["default"], "top_left")

    def test_input_accepts_each_of_the_nine_names(self):
        for anchor in ANCHORS:
            self.assertEqual(self.field.input("anchor", anchor, {}), anchor)

    def test_input_normalizes_hyphens_and_case(self):
        self.assertEqual(self.field.input("anchor", "Top-Left", {}), "top_left")

    def test_input_rejects_anything_else(self):
        for bad in ("middle", "top_top", 3, ["center"], True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.field.input("anchor", bad, {})

    def test_input_passes_empty_values_through(self):
        self.assertIsNone(self.field.input("anchor", None, {}))
        self.assertEqual(self.field.input("anchor", "", {}), "")


class TestColor(unittest.TestCase):

    def setUp(self):
        self.field = Color(Mock())

    def test_output_carries_allow_transparent_explicitly(self):
        plain = self.field.output({"type": "color", "name": "fill", "default": "#00b140"})
        clear = self.field.output({
            "type": "color", "name": "fill", "default": "transparent",
            "configuration": {"allow_transparent": True},
        })

        self.assertEqual(plain["type"], "color")
        self.assertIs(plain["allow_transparent"], False)
        self.assertEqual(plain["default"], "#00b140")
        self.assertIs(clear["allow_transparent"], True)
        self.assertEqual(clear["default"], "transparent")

    def test_output_ignores_a_non_boolean_allow_transparent(self):
        schema = self.field.output({"type": "color", "name": "fill", "configuration": {"allow_transparent": "yes"}})

        self.assertIs(schema["allow_transparent"], False)

    def test_input_normalizes_hex_to_lowercase_six_digits(self):
        for raw, expected in (("#FF8800", "#ff8800"), ("ff8800", "#ff8800"), ("#abc", "#aabbcc")):
            self.assertEqual(self.field.input("fill", raw, {}), expected)

    def test_input_rejects_non_hex(self):
        for bad in ("red", "#12", "#gggggg", 5, ["#fff"]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.field.input("fill", bad, {})

    def test_input_accepts_transparent_only_when_the_field_allows_it(self):
        self.assertEqual(self.field.input("fill", "transparent", {"allow_transparent": True}), "transparent")

        with self.assertRaises(ValueError):
            self.field.input("fill", "transparent", {})
        with self.assertRaises(ValueError):
            self.field.input("fill", "transparent", {"allow_transparent": "true"})

    def test_input_passes_empty_values_through(self):
        self.assertIsNone(self.field.input("fill", None, {}))
        self.assertEqual(self.field.input("fill", "", {}), "")


if __name__ == "__main__":
    unittest.main()
