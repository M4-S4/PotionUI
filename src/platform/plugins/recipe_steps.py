"""Recipe step-kind registry.

A recipe is an ordered list of steps, each with a `kind:` that names the code
which runs it (see docs/plugin-api.md "Contributing a recipe"). Core ships the
built-in kinds (`plugins.ensure`, `artifacts.fetch`, `generation.smoke`, ...);
a plugin adds its own - say `collections.ensure` - by declaring
`recipe_steps:` in its `manifest.yml` without touching core.

This module holds only the registration bookkeeping. The executor contract
itself (`StepExecutor`, `StepContext`, `StepResult`) lives in
`src.features.recipes.executors.base` (re-exported by
`src.plugin_api.recipes`), because it carries feature-shaped context (the
parsed recipe, the run record) that `src.platform` must not import. Executor
objects are stored here `Any`-typed for the same reason - mirrors
`RequirementCheckerRegistry`.

A kind registered here is both runnable (the recipes executor registry falls
back to this registry when a kind is not one of core's) and *lintable*: the
recipe catalog feeds these kind names to `validate_recipe_dict` so a recipe
using a plugin kind validates, while an unregistered kind stays an error.
"""

from dataclasses import dataclass
from typing import Any

from src.platform.plugins.sourced_registry import DuplicateRegistrationError, SourcedRegistry


class DuplicateRecipeStepKindError(DuplicateRegistrationError):
    """Raised when registering a step kind that is already registered."""


@dataclass(frozen=True)
class RecipeStepKindRegistration:
    """One registered recipe step kind."""

    kind: str
    # A `src.plugin_api.recipes.StepExecutor` instance - typed `Any` here
    # because `src.platform` must not import `src.features`/`src.plugin_api`.
    executor: Any
    source: str = "core"


class RecipeStepKindRegistry(SourcedRegistry[RecipeStepKindRegistration]):
    """Registry mapping a recipe step `kind:` -> `RecipeStepKindRegistration`."""

    key_attr = "kind"
    label = "Recipe step kind"
    duplicate_error = DuplicateRecipeStepKindError



# Module-level singleton shared by the plugin enable/disable path and the
# recipes executor registry - mirrors `requirement_checker_registry`.
recipe_step_kind_registry = RecipeStepKindRegistry()
