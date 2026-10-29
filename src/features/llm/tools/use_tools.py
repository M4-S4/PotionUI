import json
from typing import Any, Dict, Iterable, List, Optional

from src.features.llm import context_budget
from src.features.llm.tools.base import BaseTool, ToolResult

USE_TOOLS_NAME = "use_tools"

STANDARD_TIER_DEFERRED_GROUPS = frozenset({"library", "media"})

_GROUP_SUMMARIES = {
    "phrasebook_write": "browse a category's values by id and create, edit or remove categories and values",
    "memory_maintenance": "read, edit or delete saved memory notes",
    "director": "lay out a timed Prompt Relay video timeline",
    "media": "edit the user's images, videos and audio: trim, crop, resize, rotate, flip, speed, split, join and more",
    "library": (
        "saved segments, templates and prompts, uploaded files, collections, the model and "
        "preset catalogs, and starting a generation for an explicit preset"
    ),
}


def deferred_tool_groups(
    tools: Iterable[BaseTool],
    tool_tier: str,
    core_groups: Iterable[str],
    always_core: Iterable[str] = (),
    loaded_groups: Iterable[str] = (),
) -> Dict[str, List[str]]:
    if tool_tier == context_budget.TOOL_TIER_LARGE:
        return {}
    kept = set(core_groups) | set(loaded_groups)
    pinned = set(always_core)
    deferred: Dict[str, List[str]] = {}
    for tool in tools:
        group = tool.context_group
        if group in kept or tool.name in pinned:
            continue
        if tool_tier == context_budget.TOOL_TIER_STANDARD and group not in STANDARD_TIER_DEFERRED_GROUPS:
            continue
        deferred.setdefault(group, []).append(tool.name)
    return deferred


def use_tools_schema(deferred: Dict[str, List[str]]) -> Dict[str, Any]:
    groups = sorted(deferred)
    summaries = "; ".join(
        f"'{group}': {_GROUP_SUMMARIES.get(group) or ', '.join(deferred[group])}" for group in groups
    )
    return {
        "type": "function",
        "function": {
            "name": USE_TOOLS_NAME,
            "description": (
                "Load more tools for the rest of this turn. Call it when the task needs a tool "
                f"you were not given. Groups: {summaries}."
            ),
            "parameters": {
                "type": "object",
                "properties": {"group": {"type": "string", "enum": groups}},
                "required": ["group"],
            },
        },
    }


def deferred_group_of(tool_name: str, deferred: Dict[str, List[str]]) -> Optional[str]:
    return next((group for group, names in deferred.items() if tool_name in names), None)


def load_tool_group(group: Any, allowed_tools: List[str], deferred: Dict[str, List[str]]) -> ToolResult:
    names = deferred.pop(group, None) if isinstance(group, str) else None
    if names is None:
        available = ", ".join(sorted(deferred)) or "none"
        return ToolResult(
            success=False,
            data="",
            error=f"No tool group {group!r} left to load. Groups you can load: {available}.",
        )
    allowed_tools.extend(name for name in names if name not in allowed_tools)
    return ToolResult(success=True, data=json.dumps({"loaded": names}))
