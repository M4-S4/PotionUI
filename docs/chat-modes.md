# Chat Modes, Tools, Styles & @Resources

The built-in AI chat is organized around **modes**. A chat session is created in exactly
one mode and keeps it for its whole life — the mode selector locks as soon as the
conversation has messages. Each mode owns:

- a **base system prompt** (may contain the `{{TOOL_HINTS}}` placeholder, substituted with
  the hints the session's allowed tools declare; builtin tools declare none),
- a **tool set** (its own tools plus all *global* tools),
- optional **LLM option overrides** and a **context contributor** (see below).

The selection hierarchy in the chat header is **Provider + Model (LLM config) → Mode → Style**.

## Built-in mode

`generation` — the default. Its system prompt is code-owned
(`DEFAULT_TOOLS_SYSTEM_PROMPT_TEMPLATE` in `src/features/chat/modes/builtin.py`; not
an editable setting), and it includes all generation tools (form state, segments,
prompt search, run generation, etc.). Memory tools are global and available in every mode.
Saving, editing and deleting prompts belongs to the `prompts` mode, and tagging, rating
and collections to the `history` mode; the generation chat does not load them.

## What the model is sent before the user types

Every turn starts with the mode's system prompt and the allowed tools' schemas. The system
prompt carries only rules that span several tools: the order of work, the `<tool_action>`
version tags, phrasebook marker grammar, the `${name}` variable rule, the memory policy and
the reply format. What a single tool does and when to call it lives in that tool's schema
`description`, which native tool calling, prompt-injected tools and MCP clients all see, so
it is never restated in the prompt.

`tests/features/chat/test_fixed_overhead.py` pins the size of system prompt plus tool schemas
for each builtin mode and window tier (±3%) and checks that both are byte-identical across
turns with the same tool set. A change that is meant to grow or shrink them re-pins that number;
`tests/features/llm/tools/test_tool_prompt_budget.py` caps each builtin schema at 2,000
characters unless it is on the allow-list there.

## Route resolution

The frontend resolves the mode from the current route when the chat panel opens and when
a new conversation starts: the longest matching `default_route_prefixes` entry across all
registered modes wins; the fallback is `generation`. A plugin page like
`/plugins/dataset-generator/...` therefore auto-selects that plugin's mode. Restored
sessions always keep their persisted mode regardless of the current route.

Relevant endpoints:

- `GET /api/chat/modes` — all registered modes (`id`, `name`, `description`, `icon`,
  `default_route_prefixes`, `tools`, `resource_namespaces`, `source`).
- `GET /api/chat/tools?mode=<id>` — the tools visible in a mode (each item carries
  `mode`, `modes`, `icon`, `label` for the UI).
- `POST /api/chat/sessions` — takes `mode` (default `generation`); unknown modes are
  rejected with an `unknown_mode` error.
- `GET /api/chat/sessions?mode=&search=&limit=&offset=` — history listing
  (`{sessions, total, limit, offset}`).

## Tool scoping

A tool belongs to one or more modes (`BaseTool.modes`) or is global (`modes = None`).
A session sees its mode's tools plus the global ones, all enabled by default. The
session stores two **subtractive** fields: `tools_enabled` (the header's global toggle;
`false` withholds every tool) and `disabled_tools` (the names unticked in the mode's
Tools panel). A tool the mode gains later — a restart with a new builtin, a plugin
enabled — is offered automatically unless its name is in `disabled_tools`. Tools from
other modes are never visible.

### Tools by context window

Each tool has a `context_group` (`core` by default). The session's context window decides
which groups are sent up front:

| Window | Sent up front |
|---|---|
| over 65,536 tokens | every group |
| 24,576 to 65,536 tokens, or unknown | everything except `library` |
| under 24,576 tokens | the mode's `core_groups` only |

The builtin groups are `core`, `phrasebook_write` (phrasebook edits and value browsing by
id), `memory_maintenance` (`read_memory`, `update_memory`, `delete_memory`), `director`
(`set_prompt_relay_timeline`) and `library` (saved segments and prompts, uploads,
collections, the model and preset catalogs, `start_generation`). A mode's `core_groups`
are always sent: the Phrasebook mode keeps `phrasebook_write`, and the History, Models and
Prompts modes keep `library`. Tools a mode names in `tool_names` are always sent too.

Groups held back this way stay reachable. The model gets one extra tool, `use_tools`,
whose `group` argument loads a group's schemas for the rest of the turn, including the
continuation after an approval. Calling a held-back tool by name loads its group the same
way. Governance, availability and the session's own toggles apply first, so a held-back
tool is never one the session could not call. The next user message starts from the
window's set again. The behavior trace records `tool_tier` and `tools_deferred`, and the
per-turn reply-format reminder is left out on windows under 24,576 tokens.

## Tool approval

A tool with `requires_approval = True` returns a preview from `execute()`. The turn
pauses on the first such call: its execution record gets `approval_status: "pending"`,
and every later call in the same batch is recorded as `skipped` with a "Not run:
waiting for approval" result, so each tool call the model made has an answer.

1. **Resolve.** `POST /api/chat/sessions/{id}/tools/resolve` with
   `{message_id, tool_index, approved, reason?, context_metadata?, force?}`. Approving
   runs `execute_confirmed()` once and records `approved`, or `failed` with the real
   error (a failed call can be approved again to retry). Rejecting records `rejected`
   with the reason. A second resolve of the same call returns `already_resolved` and
   runs nothing. When `context_metadata` is sent and the preview's before-values no
   longer match the form, it returns `stale` and leaves the call pending, unless
   `force` is set. Resolve never calls the LLM.
2. **Continue.** `POST .../messages/stream` with `continue_from: {message_id,
   tool_index}` and empty `content`. This is an ordinary streamed send: the same
   context build, history budget, pre-chat actions and tracing, except that no user
   message is stored. It returns 409 while the call is still pending, or when that
   message has already been continued.

A turn is stored as append-only steps (`chat_steps`) with its state in `chat_turns`; an
assistant row records its `turn_id` and the `step_range` it covers, and the API builds its
`reasoning`, `usage` and `notices` from those steps. History for a new turn is replayed from
the steps, so a continuation sees its own calls and the resolved result. A paused turn keeps
the model's reasoning, the results it already gathered and its failed-call memory in its
state; continuing runs the decided calls, appends one tool step in call order and carries on
in the same turn. Reasoning is stored whatever the configuration's Show thinking switch says,
sent back only within the turn being continued, and served only while the switch is on.
Admin LLM call traces (`chat_llm_call_traces`, in the admin session detail) are the one
exception: they stay wire-exact, so they include replayed reasoning, which is meant for
admin debugging.
The round budget (20) counts across the user turn and its continuations, and a call that
repeats an already-approved call (same tool, same arguments) in that chain answers
"Already applied earlier in this turn." instead of pausing again.

Sending a normal message while a call is pending rejects it, with the message as the
reason, and answers it inside the same turn. The `done` event then carries
`resolved_message: {id, tool_executions}`.

MCP clients (`src/features/mcp/protocol.py`) run `execute_confirmed()` directly; they
have no approval step.

## Conversation titles

After the first user/assistant exchange the backend asks the session's own LLM for a
3–6 word title (small, non-thinking call). On the streaming endpoint the title arrives as
a final SSE event **after** `done`:

```json
{"event": "title", "data": {"session_id": "…", "name": "Fox Forest Prompt Ideas"}}
```

Clients must not treat `done` as end-of-stream. Sessions expose `title_generated`; a
failed title attempt is retried after the next exchange (up to 6 messages).

## @Resources

Typing `@` in the chat input opens a resource autocomplete backed by
`GET /api/chat/resources/suggest?query=&mode=&limit=`. Selected resources render as
chips; the message text keeps a readable token (`@models.lora.detailer` or
`@[path with spaces]`) and the request body carries `resources: [{uri}]`.

Resolution is a **snapshot at send time**: the backend resolves each URI, stores the
resolved content in the user message's `metadata.resources`
(`[{uri, kind, title, metadata, content}]`), and injects one system-role context block
immediately before the last user message. Unknown/stale URIs become error notes and never
fail the send. Because history is rebuilt per turn, older snapshots naturally age out of
context.

Built-in namespaces:

| URI | Resolves to |
|---|---|
| `models.<type>.<name>` | model metadata incl. CivitAI trigger words and description (`models.loras.x` plural alias accepted; dots in filenames handled) |
| `phrasebook.<category.path>` | a phrasebook category's description and values |
| `presets.<preset_id>[.<form_field>]` | a preset summary or one form field's option list |
| `generations.<id>` / `generations.recent` | a generation's prompt/parameters/models, or the last 5 |

A mode can restrict the visible namespaces via `resource_namespaces` (null = all).

## Per-mode LLM options & context contributors

- `ChatMode.llm_options` (e.g. `{"think": false}`, temperature, etc.) is merged over the
  LLM config for **every** call in the mode — plain chat, the tool loop, and streaming.
- `ChatMode.context_contributor` — a callable `(context_metadata, session, user_id) ->
  Optional[str]` (sync or async). A non-empty result is injected as a system context
  block right before the last user message (before the @resource block). Failures are
  logged and never break a send. This is how a plugin mode turns its page's
  `context_metadata` into model-visible context without any core changes.

## Learning loop (mode-aware)

Prompt feedback (`POST .../prompt-feedback`) tags verdicts and approved exemplars with the
session's mode (`enhancement_feedback.mode`, exemplar `metadata.mode`). The `enhance_prompt`
tool retrieves `chat_approved` exemplars only from the current mode; rows created
before this feature count as `generation`.

## Plugin registration

See `.claude/skills/plugin/SKILL.md` ("LLM chat extensions") for the authoritative
manifest reference. Summary:

```yaml
chat_modes:
  - id: "dataset-generator"
    name: "Dataset Generator"
    icon: "database"
    system_prompt_file: "prompts/mode.md"   # or inline `system_prompt:` (mutually exclusive)
    tools: ["list_models"]                  # borrow builtin tools by name
    default_route_prefixes: ["/plugins/dataset-generator"]
    context_contributor: "hooks.chat.contribute_context"
    llm_options: {}

tools:
  - class: "tools.dataset:CreateDatasetTool"  # subclass of BaseTool, no-arg constructor
    modes: ["dataset-generator"]              # omit for a global tool

resources:
  - namespace: "datasets"                     # must equal provider.namespace
    provider: "resources.datasets:DatasetResourceProvider"
    modes: ["dataset-generator"]              # omit for all modes
```

Registrations are removed on plugin disable; sessions left in a vanished mode stay
listable but return `unknown_mode` on send. Name collisions fail the plugin enable.

## Breaking change (for plugin authors)

Chat hook payloads carry `mode` instead of `session_type`
(`chat.session.before_create/after_create`, `chat.response.transform`). The session
create API takes `mode` (not `session_type`) and no longer accepts `enable_tools`.
