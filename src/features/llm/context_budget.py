"""Model-aware request budgeting shared by every ``LLMGateway`` send path.

The chat history token budget (``conversation._apply_history_budget``) trims
*displayed* history against an operator-configured setting before any of the
per-turn context blocks (contributor, resource, memory, workspace, prompt
state, reply-contract reminder) or tool schemas are added, and it never looks
at what the target model can actually hold. That leaves the request the model
receives unbounded: a tool loop resends the whole growing message list every
round with no re-check, so a long tool conversation can silently exceed the
model's real context window.

This module is the one place request size is measured and enforced against
that real capacity. ``LLMGateway`` calls ``enforce_budget`` immediately before
every wire call (the first turn AND every tool-loop round AND the final
wrap-up — see ``LLMGateway._budgeted``), so no provider path can bypass it.
``ChatContextBuilder``/``ConversationRunner`` also call it once, before the
first call, purely to report an honest pre-flight number into the persisted
context ledger (see ``ConversationRunner._build_context_ledger``) — same
function, same numbers, not a second heuristic.

Capacity comes from explicit configuration (``provider_options`` — either the
generic ``context_window`` key or, for an Ollama config, the existing
``num_ctx`` knob already wired into the request), else from what the provider
itself reported (``capacity_source="discovered"``). Nothing here ever
infers a model's window from its name or parameter count: an unknown
capacity is reported honestly as ``capacity_source="unknown"`` with a
conservative, clearly-labelled default (see ``UNKNOWN_CAPACITY_TOKENS``)
rather than a guess dressed up as a fact.

Token counting has three tiers, reported per turn as ``accounting`` in the
ledger:

- ``"chat_template"`` / ``"chat_template_shrunk"`` — a whole-request count via
  a warm native checkpoint's own tokenizer, applying its chat template (and,
  when tools are given, the SAME prompt-injected tool text the real send
  folds into the system message — see ``MessagesCounter``) to the
  actually-kept system + history exactly as ``NativeLLMClient`` would before
  generating (see ``NativeLLMClient.messages_token_counter``). The only tiers
  that report ``measured=True`` (and only when no image is attached — a
  successful exact count already includes the tools, so nothing else is
  estimated) — everything else is honestly an estimate, never presented as
  exact. The fragment-based trim below decides an initial candidate; when a
  whole-request counter is available, ``enforce_budget`` then RECOUNTS that
  candidate exactly and, if it's over budget, drops the next-oldest eligible
  unit and recounts again (bounded by how many eligible units remain) —
  ``"chat_template"`` means the first recount already fit,
  ``"chat_template_shrunk"`` means one or more extra units had to go (see
  ``chat_template_extra_dropped`` in the ledger). This can go either way
  relative to the fragment estimate: the exact count may fit a candidate the
  fragment/framing estimate thought didn't (framing is deliberately
  conservative), or it may need to drop more than the fragment estimate
  alone would have.
- ``"fragments+framing"`` — a real per-fragment tokenizer (``counter``, e.g.
  ``NativeLLMClient.token_counter``) summed message-by-message, plus
  ``FRAMING_TOKENS_PER_MESSAGE`` per message. A per-fragment sum never sees
  the chat template's role/special-token wrapping, so even with a real
  tokenizer behind it this tier is always ``measured=False`` — it is what
  drives the incremental per-unit trimming decision (a whole-request
  chat-template count can't tell you what ONE candidate message costs). Also
  the tier a whole-request counter falls back to if it raises partway
  through the recount loop (``"chat_template_fallback"``, in the ledger's
  ``accounting``) — continuing whole-unit trimming from wherever the exact
  recount had already shrunk to, never resurrecting a larger candidate an
  earlier exact recount already proved over capacity.
- ``"estimate"`` — no tokenizer at all (Ollama/OpenAI never have one
  in-process): the labelled chars-per-token heuristic throughout.

An attached image always forces ``measured=False`` (its cost is inherently an
estimate — see ``multimodal_allowance``) regardless of which tier text
accounting used.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, TYPE_CHECKING

from src.features.llm.harness.context import user_text
from src.features.llm.harness.messages import ContextPart, ImagePart, Message, Role

from src.features.llm.harness.adapter import Settings
from src.features.llm.harness.profile import CapabilityProfile, ProfileSource, ReasoningKind
from src.features.llm.harness.tool_choice import REASONING_OFF

if TYPE_CHECKING:
    from src.features.llm.repository import LLMConfig

logger = logging.getLogger(__name__)

# Conservative labelled fallback for `count_text` when no real tokenizer is
# available: real chat text tokenizes closer to ~4 chars/token, but a smaller
# divisor means the estimate never UNDER-counts by much, which matters since
# this number gates whether a request is even submitted.
DEFAULT_CHARS_PER_TOKEN = 3.5

# Conservative per-message allowance for the chat-template role/special-token
# wrapping a per-fragment count never sees (e.g. a `<|im_start|>role\n ...
# <|im_end|>\n`-style wrapper) — applied once per message whenever accounting
# falls back to summing fragments (with or without a real per-fragment
# tokenizer), so that tier never quietly under-reports what the wire actually
# carries. This is also why "fragments+framing" is never `measured=True`: the
# allowance is itself an estimate layered on top of whatever the fragments
# measured.
FRAMING_TOKENS_PER_MESSAGE = 4

# Capacity used when a config carries no explicit context-window
# configuration and no provider metadata is available — deliberately close to
# the long-standing `chat_history_token_budget` default (8000) so an
# unconfigured deployment's behaviour doesn't change much, while still being
# clearly labelled `capacity_source="unknown"` rather than passed off as read
# from the model.
UNKNOWN_CAPACITY_TOKENS = 8192

# Per-image allowance used when the provider has no known per-image token
# cost of its own — deliberately conservative (near OpenAI's documented
# low-detail image cost) since undercounting here is what lets a request
# through that the provider then rejects.
DEFAULT_IMAGE_TOKEN_ESTIMATE = 1100

# Reserved-output fallback when no config/override is available (only reached
# from a caller — e.g. a direct context_budget unit test — that has no
# LLMConfig at hand; every real call site passes the config's own max_tokens).
DEFAULT_RESERVE_TOKENS = 1024

RESERVE_WINDOW_FRACTION = 0.15
MIN_RESERVE_TOKENS = 1024

THINKING_ALLOWANCE_FRACTION = 0.10
THINKING_ALLOWANCE_CAP_TOKENS = 4096

TOOL_TIER_COMPACT = "compact"
TOOL_TIER_STANDARD = "standard"
TOOL_TIER_LARGE = "large"
COMPACT_TOOL_TIER_BELOW_TOKENS = 24576
LARGE_TOOL_TIER_ABOVE_TOKENS = 65536

TokenCounter = Callable[[str], int]
# A whole-request counter: (system_message, kept_messages, tool_schemas) ->
# exact token count of the ACTUALLY PREPARED request via a real chat template
# (see `NativeLLMClient.messages_token_counter`) — when tool_schemas is given
# and the provider is prompt-injected-tools, the count already includes the
# tool text folded into the system message exactly as the real send builds
# it, so `enforce_budget` never adds a separate tool-schema estimate on top
# of a count this counter produced. Unlike `TokenCounter` this can't price a
# single candidate message in isolation, so it is only used on the final kept
# set `fit_messages` already decided on (and its shrink-loop candidates —
# see `enforce_budget`), never inside the incremental per-unit trimming walk
# itself. A provider that can't reproduce its own prepared tool-request shape
# should raise rather than approximate — `enforce_budget` then falls back to
# the fragment/framing tier exactly as it does for any other counter failure.
MessagesCounter = Callable[[Optional[str], List[Message], Optional[List[Dict[str, Any]]]], int]


@dataclass(frozen=True)
class CapacityInfo:
    """The effective context-window size for one call, and where it came from."""

    capacity_tokens: int
    source: str


@dataclass(frozen=True)
class TokenCount:
    tokens: int
    measured: bool  # True only when a real tokenizer produced this number


@dataclass(frozen=True)
class TrimResult:
    messages: List[Message]
    dropped_messages: int
    dropped_groups: int
    fits: bool
    used_tokens: int
    measured: bool
    # The same selection as `messages`, grouped back into atomic units
    # (oldest-first) instead of flattened — what `enforce_budget`'s exact
    # whole-request recount loop shrinks from, one unit at a time, without
    # having to re-derive grouping from the flat list.
    kept_units: List[List[Message]]
    # How many of the trailing `kept_units` are protected (the current-turn
    # stack — see `_protected_unit_count`) and therefore never eligible for
    # the recount loop to drop either.
    protected_count: int


@dataclass(frozen=True)
class BudgetOutcome:
    """What a caller needs after a successful ``enforce_budget`` call."""

    messages: List[Message]
    ledger: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextProfile:
    capacity: CapacityInfo
    reserve_tokens: int
    counter: Optional[TokenCounter] = None
    messages_counter: Optional[MessagesCounter] = None
    image_tokens_override: Optional[int] = None
    max_tokens: Optional[int] = 0
    thinking_enabled: bool = False
    thinking_allowance_tokens: int = 0


class ContextBudgetExceededError(Exception):
    """Even after trimming every eligible older message, the request still
    doesn't fit the model's context window.

    Carries the numbers that produced the refusal so a caller can surface an
    actionable message instead of letting an oversized request reach the
    provider and fail opaquely (or, worse, silently truncate something that
    matters).
    """

    def __init__(
        self,
        *,
        capacity_tokens: int,
        capacity_source: str,
        reserve_tokens: int,
        estimated_tokens: int,
        breakdown: Dict[str, Any],
    ) -> None:
        self.capacity_tokens = capacity_tokens
        self.capacity_source = capacity_source
        self.reserve_tokens = reserve_tokens
        self.estimated_tokens = estimated_tokens
        self.available_tokens = max(0, capacity_tokens - reserve_tokens)
        self.over_by_tokens = estimated_tokens - self.available_tokens
        self.breakdown = breakdown
        super().__init__(
            f"Context budget exceeded: this turn needs an estimated "
            f"{estimated_tokens} tokens but only {self.available_tokens} are "
            f"available ({capacity_tokens}-token capacity [{capacity_source}] "
            f"minus {reserve_tokens} reserved for output), over by "
            f"{self.over_by_tokens} tokens even after trimming every eligible "
            f"older message."
        )


def resolve_capacity(profile: CapabilityProfile) -> CapacityInfo:
    if profile.context_window:
        source = profile.sources.get("context_window", ProfileSource.SEED)
        return CapacityInfo(profile.context_window, ProfileSource(source).value)
    return CapacityInfo(UNKNOWN_CAPACITY_TOKENS, "unknown")


def tool_tier(capacity: CapacityInfo) -> str:
    if capacity.source == "unknown":
        return TOOL_TIER_STANDARD
    if capacity.capacity_tokens < COMPACT_TOOL_TIER_BELOW_TOKENS:
        return TOOL_TIER_COMPACT
    if capacity.capacity_tokens > LARGE_TOOL_TIER_ABOVE_TOKENS:
        return TOOL_TIER_LARGE
    return TOOL_TIER_STANDARD


def resolve_image_token_override(config: "LLMConfig") -> Optional[int]:
    """An operator-configured override for the per-image token cost —
    ``provider_options.image_token_estimate``, documented alongside
    ``context_window``/``num_ctx``. ``None`` when unset or not a positive
    number; ``multimodal_allowance`` then falls back to
    ``DEFAULT_IMAGE_TOKEN_ESTIMATE``.
    """
    provider_opts = getattr(config, "provider_options", None) or {}
    value = provider_opts.get("image_token_estimate")
    return int(value) if _is_positive_number(value) else None


def resolve_thinking_enabled(settings: Settings, profile: CapabilityProfile) -> bool:
    value = settings.reasoning
    if value is None:
        return profile.reasoning is ReasoningKind.ALWAYS_ON
    if isinstance(value, str):
        return value.strip().lower() not in REASONING_OFF
    return value is not False


@dataclass(frozen=True)
class ReserveInfo:
    reserve_tokens: int
    base_reserve_tokens: int
    thinking_enabled: bool
    thinking_allowance_tokens: int


def resolve_reserve(*, max_tokens: Any, window_tokens: int, thinking_enabled: bool) -> ReserveInfo:
    standard = max(MIN_RESERVE_TOKENS, math.ceil(RESERVE_WINDOW_FRACTION * window_tokens))
    base = standard if max_tokens is None else min(max(0, int(max_tokens)), standard)
    allowance = (
        min(THINKING_ALLOWANCE_CAP_TOKENS, math.ceil(THINKING_ALLOWANCE_FRACTION * window_tokens))
        if thinking_enabled else 0
    )
    return ReserveInfo(
        reserve_tokens=base + allowance,
        base_reserve_tokens=base,
        thinking_enabled=thinking_enabled,
        thinking_allowance_tokens=allowance,
    )


def resolve_sent_max_tokens(
    *,
    max_tokens: Optional[int],
    reserve_tokens: int,
    capacity_tokens: int,
    capacity_source: str,
    estimated_prompt_tokens: int,
) -> Optional[int]:
    if max_tokens is None or capacity_source == "unknown":
        return max_tokens
    return min(max_tokens, max(reserve_tokens, capacity_tokens - estimated_prompt_tokens))


def _is_positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def count_text(text: Optional[str], counter: Optional[TokenCounter]) -> TokenCount:
    """Token count for *text*: the real tokenizer when *counter* is given and
    doesn't raise, else the labelled chars-per-token estimate."""
    text = text or ""
    if counter is not None:
        try:
            return TokenCount(max(0, int(counter(text))), True)
        except Exception:
            logger.debug("[ContextBudget] token counter failed; falling back to estimate", exc_info=True)
    return TokenCount(math.ceil(len(text) / DEFAULT_CHARS_PER_TOKEN), False)


def _tool_calls_json(message: Message) -> Optional[str]:
    calls = message.tool_calls()
    if not calls:
        return None
    payload = [
        {
            "id": call.wire_id or call.call_id,
            "type": "function",
            "function": {"name": call.name, "arguments": call.args if call.args is not None else (call.args_raw or "")},
        }
        for call in calls
    ]
    try:
        return json.dumps(payload)
    except (TypeError, ValueError):
        return str(payload)


def wire_fragments(message: Message) -> List[List[str]]:
    if message.role is Role.TOOL:
        return [[result.text()] for result in message.tool_results()]
    if message.role is Role.USER:
        return [[user_text(message)]]
    units = [message.text()]
    reasoning = "".join(part.text or part.summary for part in message.reasoning())
    if reasoning:
        units.append(reasoning)
    calls = _tool_calls_json(message)
    if calls:
        units.append(calls)
    return [units]


def count_messages(messages: Sequence[Message], counter: Optional[TokenCounter]) -> TokenCount:
    total = 0
    wire_messages = 0
    for message in messages:
        for fragments in wire_fragments(message):
            wire_messages += 1
            for fragment in fragments:
                total += count_text(fragment, counter).tokens
    total += FRAMING_TOKENS_PER_MESSAGE * wire_messages
    return TokenCount(total, False)


def has_image(messages: Sequence[Message]) -> bool:
    for message in messages:
        if any(isinstance(part, ImagePart) for part in message.parts):
            return True
        if any(isinstance(item, ImagePart) for result in message.tool_results() for item in result.content):
            return True
    return False


def count_tool_schemas(
    tool_schemas: Optional[List[Dict[str, Any]]], counter: Optional[TokenCounter]
) -> TokenCount:
    if not tool_schemas:
        return TokenCount(0, True)
    try:
        text = json.dumps(tool_schemas)
    except (TypeError, ValueError):
        text = str(tool_schemas)
    return count_text(text, counter)


def multimodal_allowance(image_attached: bool, *, per_image_tokens: Optional[int] = None) -> int:
    if not image_attached:
        return 0
    return per_image_tokens if per_image_tokens is not None else DEFAULT_IMAGE_TOKEN_ESTIMATE


def atomic_units(messages: Sequence[Message]) -> List[List[Message]]:
    units: List[List[Message]] = []
    index, total = 0, len(messages)
    while index < total:
        message = messages[index]
        if message.role is Role.ASSISTANT and message.tool_calls():
            group = [message]
            follow = index + 1
            while follow < total and messages[follow].role is Role.TOOL:
                group.append(messages[follow])
                follow += 1
            units.append(group)
            index = follow
        else:
            units.append([message])
            index += 1
    return units


def _is_turn_anchor(unit: List[Message]) -> bool:
    if len(unit) != 1 or unit[0].role is not Role.USER:
        return False
    return any(not isinstance(part, ContextPart) for part in unit[0].parts) or not unit[0].parts


def current_turn_anchor(units: List[List[Message]]) -> Optional[int]:
    for index in range(len(units) - 1, -1, -1):
        if _is_turn_anchor(units[index]):
            return index
    return None


def _protected_unit_count(units: List[List[Message]]) -> int:
    if not units:
        return 0
    anchor = current_turn_anchor(units)
    if anchor is None:
        return 1
    protected_from = anchor
    if anchor == 1 and len(units[0]) == 1 and units[0][0].role is Role.SYSTEM:
        protected_from = 0
    return len(units) - protected_from


def fit_messages(
    messages: Sequence[Message],
    *,
    available_tokens: int,
    counter: Optional[TokenCounter],
) -> TrimResult:
    """Keep whole tool-call/result groups from the tail while they fit
    *available_tokens*, dropping the oldest first. Never mutates *messages*.
    """
    units = atomic_units(messages)
    if not units:
        return TrimResult([], 0, 0, True, 0, True, kept_units=[], protected_count=0)

    protected_count = _protected_unit_count(units)
    protected_units = units[len(units) - protected_count:]
    older_units = units[: len(units) - protected_count]

    protected_counts = [count_messages(u, counter) for u in protected_units]
    kept_units_reversed = list(reversed(protected_units))
    used = sum(c.tokens for c in protected_counts)
    measured = all(c.measured for c in protected_counts) if protected_counts else True

    for unit in reversed(older_units):
        c = count_messages(unit, counter)
        if used + c.tokens > available_tokens:
            break
        kept_units_reversed.append(unit)
        used += c.tokens
        measured = measured and c.measured

    kept_units = list(reversed(kept_units_reversed))
    kept_messages = [m for unit in kept_units for m in unit]
    total_messages = sum(len(u) for u in units)
    kept_message_count = sum(len(u) for u in kept_units)

    return TrimResult(
        messages=kept_messages,
        dropped_messages=total_messages - kept_message_count,
        dropped_groups=len(units) - len(kept_units),
        fits=used <= available_tokens,
        used_tokens=used,
        measured=measured,
        kept_units=kept_units,
        protected_count=protected_count,
    )


def enforce_budget(
    *,
    capacity_tokens: int,
    capacity_source: str,
    reserve_tokens: int,
    system_message: Optional[str],
    messages: Sequence[Message],
    tool_schemas: Optional[List[Dict[str, Any]]] = None,
    image_attached: bool = False,
    image_tokens: Optional[int] = None,
    counter: Optional[TokenCounter] = None,
    messages_counter: Optional[MessagesCounter] = None,
) -> BudgetOutcome:
    """The one shared budgeting step every send path runs through.

    Accounts system text, tool schemas, a multimodal allowance and the
    reserved output tokens as fixed costs, then trims *messages* (oldest
    eligible whole group first, current-turn units always protected — see
    ``fit_messages``) to an initial candidate. When *messages_counter* is
    given, that candidate is then RECOUNTED exactly (system + kept history +
    tool schemas, via the real chat template — see ``MessagesCounter``)
    regardless of what the fragment estimate concluded: if the exact count
    already fits, it wins even over a fragment estimate that thought
    otherwise (framing is deliberately conservative); if it's over, the
    next-oldest eligible unit is dropped and it's recounted again, bounded by
    how many eligible (non-protected) units remain — see the module
    docstring's ``"chat_template"``/``"chat_template_shrunk"`` tiers. Because
    a successful exact count already includes the tool text (when
    *tool_schemas* is given and the provider is prompt-injected-tools), it is
    used as the WHOLE total for that tier — the separate ``tools_count``
    estimate (JSON-size based) is reported in the ledger for visibility but
    never added a second time on top. A *messages_counter* that raises
    mid-recount abandons the exact path entirely for whatever candidate
    triggered the failure, but NEVER un-drops units the loop already
    shrank past: it continues whole-unit trimming from exactly that reduced
    candidate under the fragment/framing estimate instead
    (``"chat_template_fallback"``, which DOES add the separate tool-schema
    estimate, same as ``"fragments+framing"``/``"estimate"``) — this can only
    shrink the candidate further, never grow it back toward one an earlier
    exact recount already proved exceeds capacity. Raises
    ``ContextBudgetExceededError`` when even the protected tail alone doesn't
    fit, by whichever accounting produced the final decision — the caller
    must not submit that request.
    """
    reserve = max(0, int(reserve_tokens))
    system_count = count_text(system_message, counter)
    tools_count = count_tool_schemas(tool_schemas, counter)
    image_count = multimodal_allowance(image_attached, per_image_tokens=image_tokens)

    fixed_tokens = system_count.tokens + tools_count.tokens + image_count
    available_for_messages = capacity_tokens - reserve - fixed_tokens
    available_total = capacity_tokens - reserve

    trim = fit_messages(messages, available_tokens=max(0, available_for_messages), counter=counter)

    accounting = "fragments+framing" if counter is not None else "estimate"
    final_units = trim.kept_units
    final_fits = trim.fits
    final_history_and_system_tokens = system_count.tokens + trim.used_tokens
    history_tokens: Optional[int] = trim.used_tokens
    chat_template_extra_dropped = 0
    exact_includes_tools = False

    if messages_counter is not None:
        kept_units = list(trim.kept_units)
        counter_ok = True
        exact_combined = final_history_and_system_tokens  # overwritten on the first successful recount
        while True:
            flat = [m for unit in kept_units for m in unit]
            try:
                exact_combined = max(0, int(messages_counter(system_message, flat, tool_schemas)))
            except Exception:
                logger.debug(
                    "[ContextBudget] messages_counter failed mid-recount; falling back to the "
                    "fragment-based decision", exc_info=True,
                )
                counter_ok = False
                break
            if exact_combined + image_count <= available_total:
                break
            if len(kept_units) <= trim.protected_count:
                break
            kept_units.pop(0)
            chat_template_extra_dropped += 1

        if counter_ok:
            accounting = "chat_template_shrunk" if chat_template_extra_dropped else "chat_template"
            final_units = kept_units
            final_history_and_system_tokens = exact_combined
            if chat_template_extra_dropped:
                history_tokens = None
            final_fits = exact_combined + image_count <= available_total
            exact_includes_tools = True
        else:
            # The counter failed while measuring `kept_units` — whichever
            # units the exact-recount loop had already dropped before the
            # failure are real progress and must never be un-dropped: falling
            # back to the ORIGINAL (pre-loop) candidate here would silently
            # resubmit a set the loop may have already proved exceeds
            # capacity by an exact count. Continue whole-unit trimming from
            # exactly where the loop left off, using the fragment/framing
            # estimate — `fit_messages` can only shrink `kept_units` further
            # (never grow it back), so the result is always a subset of the
            # last exact-measured candidate, never a superset of one already
            # proved over.
            accounting = "chat_template_fallback"
            chat_template_extra_dropped = 0
            fallback_trim = fit_messages(
                [m for unit in kept_units for m in unit],
                available_tokens=max(0, available_for_messages),
                counter=counter,
            )
            final_units = fallback_trim.kept_units
            final_history_and_system_tokens = system_count.tokens + fallback_trim.used_tokens
            history_tokens = fallback_trim.used_tokens
            final_fits = fallback_trim.fits

    final_messages = [m for unit in final_units for m in unit]
    all_units = atomic_units(messages)
    total_messages = sum(len(u) for u in all_units)
    final_message_count = sum(len(u) for u in final_units)
    total_groups = len(all_units)

    tools_added_to_total = 0 if exact_includes_tools else tools_count.tokens
    estimated_tokens = final_history_and_system_tokens + tools_added_to_total + image_count
    measured = exact_includes_tools and image_count == 0

    ledger: Dict[str, Any] = {
        "capacity_tokens": capacity_tokens,
        "capacity_source": capacity_source,
        "reserve_tokens": reserve,
        "estimated_tokens": estimated_tokens,
        "measured": measured,
        "accounting": accounting,
        "chat_template_extra_dropped": chat_template_extra_dropped,
        "system_tokens": system_count.tokens,
        # A JSON-size estimate, reported for visibility even in a
        # "chat_template*" tier where it is NOT part of `estimated_tokens` —
        # the exact recount already folded the real tool text into the
        # system message it counted (see `exact_includes_tools` above).
        "tool_schema_tokens": tools_count.tokens,
        "image_tokens": image_count,
        # Always the fragment/framing figure of the FINAL kept set, even in
        # a "chat_template*" tier — a whole-request count can't be cleanly
        # split back into a history-only share without a second tokenize
        # pass; `estimated_tokens` is the authoritative total, this is
        # diagnostic granularity only.
        "history_tokens": (
            history_tokens if history_tokens is not None else count_messages(final_messages, counter).tokens
        ),
        "messages_total": total_messages,
        "messages_sent": final_message_count,
        "messages_dropped": total_messages - final_message_count,
        "groups_dropped": total_groups - len(final_units),
    }

    # An unknown capacity is a labelled guess, never a fact: trimming history
    # to it is prudent, refusing the turn on it is not - a tool-rich system
    # prompt alone exceeds the default and would lock the assistant out of
    # every unconfigured deployment. Send, and let the ledger say it overran.
    ledger["over_budget"] = not final_fits
    if not final_fits and capacity_source != "unknown":
        raise ContextBudgetExceededError(
            capacity_tokens=capacity_tokens,
            capacity_source=capacity_source,
            reserve_tokens=reserve,
            estimated_tokens=estimated_tokens,
            breakdown=ledger,
        )

    return BudgetOutcome(messages=final_messages, ledger=ledger)


def budget_with_profile(
    profile: Optional[ContextProfile],
    *,
    system_message: Optional[str],
    messages: Sequence[Message],
    tool_schemas: Optional[List[Dict[str, Any]]] = None,
) -> BudgetOutcome:
    p = profile or ContextProfile(
        capacity=CapacityInfo(UNKNOWN_CAPACITY_TOKENS, "unknown"),
        reserve_tokens=DEFAULT_RESERVE_TOKENS,
    )
    outcome = enforce_budget(
        capacity_tokens=p.capacity.capacity_tokens,
        capacity_source=p.capacity.source,
        reserve_tokens=p.reserve_tokens,
        system_message=system_message,
        messages=messages,
        tool_schemas=tool_schemas,
        image_attached=has_image(messages),
        image_tokens=p.image_tokens_override,
        counter=p.counter,
        messages_counter=p.messages_counter,
    )
    max_tokens_sent = resolve_sent_max_tokens(
        max_tokens=p.max_tokens,
        reserve_tokens=p.reserve_tokens,
        capacity_tokens=p.capacity.capacity_tokens,
        capacity_source=p.capacity.source,
        estimated_prompt_tokens=outcome.ledger["estimated_tokens"],
    )
    ledger = {
        **outcome.ledger,
        "max_tokens": p.max_tokens,
        "max_tokens_sent": max_tokens_sent,
        "thinking_enabled": p.thinking_enabled,
        "thinking_allowance_tokens": p.thinking_allowance_tokens,
    }
    return BudgetOutcome(messages=outcome.messages, ledger=ledger)
