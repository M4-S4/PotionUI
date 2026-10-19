from collections import namedtuple

import pytest

from src.features.chat.modes import ChatMode, ChatModeRegistry, DuplicateChatModeError
from src.platform.plugins.automation_nodes import DuplicateNodeTypeError, NodeTypeRegistry, NodeTypeSpec
from src.platform.plugins.field_types import DuplicateFieldTypeError, FieldTypeDefinition, FieldTypeRegistry
from src.platform.plugins.limit_kinds import DuplicateLimitKindError, LimitKind, LimitKindRegistry
from src.platform.plugins.login_providers import (
    DuplicateLoginProviderError,
    LoginProviderDefinition,
    LoginProviderRegistry,
)
from src.platform.plugins.phrasebook_ops import (
    DuplicatePhrasebookOperationError,
    PhrasebookBatchOperationDefinition,
    PhrasebookOperationRegistry,
)
from src.platform.plugins.prompt_importers import (
    DuplicatePromptImporterError,
    PromptImporterDefinition,
    PromptImporterRegistry,
)
from src.platform.plugins.recipe_steps import (
    DuplicateRecipeStepKindError,
    RecipeStepKindRegistration,
    RecipeStepKindRegistry,
)
from src.platform.plugins.requirement_checkers import (
    DuplicateRequirementCheckerError,
    RequirementCheckerRegistration,
    RequirementCheckerRegistry,
)
from src.platform.plugins.sourced_registry import DuplicateRegistrationError, SourcedRegistry

Case = namedtuple("Case", "registry_cls make error_cls message")


def _plain(label):
    return lambda key, existing: f"{label} already registered: '{key}'"


def _limit_kind(name, source):
    key = name if source == "core" else f"{source}.{name}"
    return LimitKind(key=key, source=source, label="L", value_type="count", measure=lambda context: 0)


CASES = {
    "field_types": Case(
        FieldTypeRegistry,
        lambda name, source: FieldTypeDefinition(type_name=name, source=source),
        DuplicateFieldTypeError,
        _plain("Field type"),
    ),
    "prompt_importers": Case(
        PromptImporterRegistry,
        lambda name, source: PromptImporterDefinition(
            importer_id=name, label="L", frontend_component="", backend=object(), source=source
        ),
        DuplicatePromptImporterError,
        _plain("Prompt importer"),
    ),
    "phrasebook_ops": Case(
        PhrasebookOperationRegistry,
        lambda name, source: PhrasebookBatchOperationDefinition(
            op_id=name, label="L", backend=object(), source=source
        ),
        DuplicatePhrasebookOperationError,
        _plain("Phrasebook batch operation"),
    ),
    "requirement_checkers": Case(
        RequirementCheckerRegistry,
        lambda name, source: RequirementCheckerRegistration(type_name=name, checker=object(), source=source),
        DuplicateRequirementCheckerError,
        _plain("Requirement checker"),
    ),
    "recipe_steps": Case(
        RecipeStepKindRegistry,
        lambda name, source: RecipeStepKindRegistration(kind=name, executor=object(), source=source),
        DuplicateRecipeStepKindError,
        _plain("Recipe step kind"),
    ),
    "node_types": Case(
        NodeTypeRegistry,
        lambda name, source: NodeTypeSpec(key=name, kind="action", title="T", source=source),
        DuplicateNodeTypeError,
        _plain("Node type"),
    ),
    "login_providers": Case(
        LoginProviderRegistry,
        lambda name, source: LoginProviderDefinition(id=name, label="L", start_path="/start", source=source),
        DuplicateLoginProviderError,
        _plain("Login provider"),
    ),
    "limit_kinds": Case(LimitKindRegistry, _limit_kind, DuplicateLimitKindError, _plain("Limit kind")),
    "chat_modes": Case(
        ChatModeRegistry,
        lambda name, source: ChatMode(id=name, name=name, source=source),
        DuplicateChatModeError,
        lambda key, existing: f"Chat mode '{key}' is already registered (by '{existing}')",
    ),
}


@pytest.fixture(params=list(CASES.values()), ids=list(CASES))
def case(request):
    return request.param


def _key(case, entry):
    return getattr(entry, case.registry_cls.key_attr)


def _keys(case, registry):
    return {_key(case, entry) for entry in registry.all()}


def test_every_case_is_a_sourced_registry(case):
    assert issubclass(case.registry_cls, SourcedRegistry)
    assert issubclass(case.error_cls, DuplicateRegistrationError)
    assert issubclass(case.error_cls, ValueError)


def test_register_get_all(case):
    registry = case.registry_cls()
    first, second = case.make("alpha", "core"), case.make("beta", "plugin-a")
    registry.register(first)
    registry.register(second)

    assert registry.get(_key(case, first)) is first
    assert registry.get(_key(case, second)) is second
    assert _keys(case, registry) == {_key(case, first), _key(case, second)}


def test_get_unknown_key(case):
    registry = case.registry_cls()
    if case.registry_cls is FieldTypeRegistry:
        assert registry.get("nope").type_name == "__default__"
    else:
        assert registry.get("nope") is None


def test_duplicate_key_is_rejected_with_the_registries_message(case):
    registry = case.registry_cls()
    original = case.make("alpha", "plugin-a")
    registry.register(original)
    key = _key(case, original)

    with pytest.raises(case.error_cls) as raised:
        registry.register(case.make("alpha", "plugin-a"))

    assert str(raised.value) == case.message(key, "plugin-a")
    assert registry.get(key) is original


def test_unregister_source_removes_only_that_source(case):
    registry = case.registry_cls()
    core, mine_a, mine_b, other = (
        case.make("coreone", "core"),
        case.make("one", "plugin-a"),
        case.make("two", "plugin-a"),
        case.make("three", "plugin-b"),
    )
    for entry in (core, mine_a, mine_b, other):
        registry.register(entry)

    removed = registry.unregister_source("plugin-a")

    assert removed == 2
    assert _keys(case, registry) == {_key(case, core), _key(case, other)}


def test_unregister_unknown_source_is_a_noop(case):
    registry = case.registry_cls()
    registry.register(case.make("alpha", "core"))

    assert registry.unregister_source("nonexistent-plugin") == 0
    assert len(registry.all()) == 1
