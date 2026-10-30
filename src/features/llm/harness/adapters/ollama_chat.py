import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

import httpx

from src.features.llm.harness import finish as finish_reasons
from src.features.llm.harness import prompt_tools, tool_ids
from src.features.llm.harness.adapter import RenderRequest, WireRequest
from src.features.llm.harness.blocks import FIRST_TOOL_BLOCK, REASONING_BLOCK, TEXT_BLOCK
from src.features.llm.harness.context import user_text
from src.features.llm.harness.errors import HarnessError, WireFailure
from src.features.llm.harness.events import (
    Finish,
    ReasoningDelta,
    StreamEvent,
    TextDelta,
    ToolArgsDelta,
    ToolCallEnd,
    ToolCallStart,
    Usage,
    UsageReported,
)
from src.features.llm.harness.messages import Message, Role
from src.features.llm.harness.ndjson import NDJSONFramer
from src.features.llm.harness.profile import CapabilityProfile, ProfileLayer, ProfileSource
from src.features.llm.harness.rendering import bearer_headers, image_data, last_user_index, reasoning_text, selected_image
from src.features.llm.harness.think_tags import TEXT, THINKING, ThinkTagStage
from src.features.llm.harness.tool_args import arguments_object, object_from_wire
from src.features.llm.harness.transport import HttpAdapter
from src.features.llm.model_listing import (
    CONTEXT_WINDOW_PROBE_TIMEOUT_SECONDS,
    DiscoveredModel,
    DiscoveredWindow,
    ModelListingError,
    fetch_json,
    positive_int,
    sorted_models,
)
from src.features.llm.repository import LLMConfig
from src.platform.http.tls import client_ssl_context

OPTION_KEYS = (
    "num_gpu",
    "num_thread",
    "num_batch",
    "num_keep",
    "seed",
    "top_k",
    "top_p",
    "min_p",
    "tfs_z",
    "typical_p",
    "repeat_penalty",
    "repeat_last_n",
    "presence_penalty",
    "frequency_penalty",
    "mirostat",
    "mirostat_tau",
    "mirostat_eta",
    "stop",
)

SAMPLED_OPTION_KEYS = frozenset({"top_k", "top_p", "min_p"})
KEEP_ALIVE_LOADED = "5m"
TRAINED_WINDOW_CAP_TOKENS = 32768
_MODELFILE_NUM_CTX = re.compile(r"^\s*num_ctx\s+(\d+)\s*$", re.MULTILINE)


def parse_show_window(payload: Any) -> Optional[DiscoveredWindow]:
    if not isinstance(payload, dict):
        return None
    parameters = payload.get("parameters")
    if isinstance(parameters, str):
        match = _MODELFILE_NUM_CTX.search(parameters)
        runtime = positive_int(int(match.group(1))) if match else None
        if runtime:
            return DiscoveredWindow(tokens=runtime, kind="runtime")
    model_info = payload.get("model_info")
    if isinstance(model_info, dict):
        for key, value in model_info.items():
            if isinstance(key, str) and key.endswith(".context_length"):
                trained = positive_int(value)
                if trained:
                    return DiscoveredWindow(tokens=trained, kind="trained")
    return None


def render_options(request: RenderRequest, profile: CapabilityProfile) -> Dict[str, Any]:
    settings = request.settings
    configured = request.config.provider_options or {}
    base = (("temperature", settings.temperature), ("num_predict", settings.max_tokens), ("num_ctx", profile.context_request))
    options: Dict[str, Any] = {key: value for key, value in base if value is not None}
    for key in OPTION_KEYS:
        value = getattr(settings, key) if key in SAMPLED_OPTION_KEYS else configured.get(key)
        if value is not None:
            options[key] = value
    return options


def render_think(request: RenderRequest) -> Any:
    return request.settings.reasoning if request.settings.reasoning is not None else False


def render_keep_alive(request: RenderRequest) -> Any:
    configured = (request.config.provider_options or {}).get("keep_alive")
    if request.settings.keep_model_loaded:
        return configured if configured not in (None, 0, "0", "0s") else KEEP_ALIVE_LOADED
    return configured if configured is not None else 0


def render_messages(request: RenderRequest, profile: CapabilityProfile, system: str, inline_tools: bool) -> List[Dict[str, Any]]:
    ids = tool_ids.wire_ids(request.messages, profile.tool_id_rule)
    rendered: List[Dict[str, Any]] = []
    if system:
        rendered.append({"role": "system", "content": system})
    for message in request.messages:
        if message.role is Role.SYSTEM:
            rendered.append({"role": "system", "content": message.text()})
        elif message.role is Role.USER:
            rendered.append({"role": "user", "content": user_text(message)})
        elif message.role is Role.ASSISTANT:
            rendered.append(_assistant(message, ids, inline_tools))
        else:
            for result in message.tool_results():
                if inline_tools:
                    rendered.append({
                        "role": "user",
                        "content": prompt_tools.result_text(result.name, result.text()),
                    })
                else:
                    rendered.append({
                        "role": "tool",
                        "content": result.text(),
                        "tool_call_id": ids.get(result.call_id, result.call_id),
                    })
    index = last_user_index(rendered)
    image = selected_image(request.messages)
    if image is not None and index is not None:
        rendered[index]["images"] = [image_data(image)]
    return rendered


def _assistant(message: Message, ids: Dict[str, str], inline_tools: bool) -> Dict[str, Any]:
    calls = message.tool_calls()
    if calls and inline_tools:
        entry: Dict[str, Any] = {"role": "assistant", "content": prompt_tools.assistant_text(message.text(), calls)}
    else:
        entry = {"role": "assistant", "content": message.text()}
        if calls:
            entry["tool_calls"] = [
                {
                    "id": ids[call.call_id],
                    "type": "function",
                    "function": {"name": call.name, "arguments": arguments_object(call)},
                }
                for call in calls
            ]
    if calls:
        thinking = reasoning_text(message)
        if thinking:
            entry["thinking"] = thinking
    return entry


def usage_from(data: Dict[str, Any]) -> Usage:
    def count(key: str) -> Optional[int]:
        value = data.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    prompt, completion = count("prompt_eval_count"), count("eval_count")
    total = None if prompt is None and completion is None else (prompt or 0) + (completion or 0)
    raw = {key: value for key, value in data.items() if key not in ("message", "done")}
    return Usage(
        input_tokens=prompt,
        output_tokens=completion,
        total_tokens=total,
        cached_input_tokens=count("prompt_eval_cached_count"),
        raw=raw,
    )


class OllamaChatDecoder:
    def __init__(
        self,
        seed: str,
        classify: Callable[[WireFailure], HarnessError],
        label: str = "ollama-chat",
    ) -> None:
        self._seed = seed
        self._classify = classify
        self._framer = NDJSONFramer(label)
        self._think = ThinkTagStage()
        self._calls = 0
        self._finished = False

    def feed(self, chunk: bytes) -> List[StreamEvent]:
        return self._objects(self._framer.feed(chunk))

    def close(self) -> List[StreamEvent]:
        events = self._objects(self._framer.close())
        if not self._finished:
            events.extend(self._finish(None))
        return events

    def _objects(self, values: List[Any]) -> List[StreamEvent]:
        events: List[StreamEvent] = []
        for data in values:
            if self._finished:
                break
            if isinstance(data, dict):
                events.extend(self._object(data))
        return events

    def _object(self, data: Dict[str, Any]) -> List[StreamEvent]:
        error = data.get("error")
        if error and not data.get("message"):
            text = error if isinstance(error, str) else json.dumps(error)
            status = 404 if "not found" in text.lower() else 500
            raise self._classify(WireFailure(status=status, body=json.dumps(data).encode("utf-8"), in_stream=True))
        events: List[StreamEvent] = []
        message = data.get("message")
        if isinstance(message, dict):
            thinking = message.get("thinking")
            if isinstance(thinking, str) and thinking:
                events.append(ReasoningDelta(block=REASONING_BLOCK, text=thinking))
            content = message.get("content")
            if isinstance(content, str) and content:
                events.extend(self._text(self._think.feed(content)))
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                for call in calls:
                    events.extend(self._native_call(call))
        if data.get("done"):
            events.extend(self._finish(data.get("done_reason"), usage_from(data)))
        return events

    def _text(self, pieces) -> List[StreamEvent]:
        events: List[StreamEvent] = []
        for piece in pieces:
            if piece.kind == THINKING:
                events.append(ReasoningDelta(block=REASONING_BLOCK, text=piece.text))
            elif piece.kind == TEXT:
                events.append(TextDelta(block=TEXT_BLOCK, text=piece.text))
        return events

    def _native_call(self, call: Any) -> List[StreamEvent]:
        if not isinstance(call, dict):
            return []
        function = call.get("function")
        function = function if isinstance(function, dict) else {}
        arguments = function.get("arguments")
        args = object_from_wire(arguments)
        block = FIRST_TOOL_BLOCK + self._calls
        call_id = tool_ids.mint_call_id(self._seed, self._calls)
        self._calls += 1
        events: List[StreamEvent] = [ToolCallStart(block=block, call_id=call_id, name=str(function.get("name") or ""))]
        if isinstance(arguments, str) and args is None:
            events.append(ToolArgsDelta(block=block, text=arguments))
            events.append(ToolCallEnd(block=block))
        else:
            events.append(ToolCallEnd(block=block, args=args))
        return events

    def _finish(self, raw_reason: Optional[str], usage: Optional[Usage] = None) -> List[StreamEvent]:
        self._finished = True
        events = self._text(self._think.finish())
        if usage is not None:
            events.append(UsageReported(usage))
        reason = finish_reasons.with_tool_calls(finish_reasons.from_ollama(raw_reason), self._calls > 0)
        events.append(Finish(reason=reason, raw_reason=raw_reason))
        return events


class OllamaChatAdapter(HttpAdapter):
    name = "ollama-chat"
    display_name = "Ollama"

    def build_body(self, request: RenderRequest, profile: CapabilityProfile) -> Dict[str, Any]:
        inline_tools = bool(request.tools) and profile.prompt_tools
        system = request.system
        if inline_tools:
            system = (system or "") + prompt_tools.build_prompt_tools_text(request.tools)
        body: Dict[str, Any] = {
            "model": request.config.model,
            "messages": render_messages(request, profile, system, inline_tools),
            "stream": True,
            "keep_alive": render_keep_alive(request),
            "think": render_think(request),
            "options": render_options(request, profile),
        }
        if request.tools and profile.tools and not inline_tools:
            body["tools"] = [tool.function_schema() for tool in request.tools]
        return body

    def render(self, request: RenderRequest, profile: CapabilityProfile) -> WireRequest:
        return WireRequest(
            method="POST",
            url=f"{request.config.base_url.rstrip('/')}/api/chat",
            headers={"Content-Type": "application/json", **bearer_headers(request.config.api_key)},
            body=self.build_body(request, profile),
            stream=True,
        )

    def decoder(self, request: RenderRequest, profile: CapabilityProfile) -> OllamaChatDecoder:
        return OllamaChatDecoder(
            seed=tool_ids.seed_for(request.messages),
            classify=self.classify_error,
            label=self.name,
        )

    async def list_models(self, base_url: str, api_key: Optional[str] = None) -> List[DiscoveredModel]:
        payload = await fetch_json(self.display_name, base_url, "/api/tags", bearer_headers(api_key))
        entries = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            raise ModelListingError(f"{self.display_name} returned no model list.")
        models = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name") or entry.get("model")
            if not name:
                continue
            raw_details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
            details = {
                key: raw_details[source]
                for key, source in (
                    ("parameter_size", "parameter_size"),
                    ("quantization", "quantization_level"),
                    ("family", "family"),
                )
                if raw_details.get(source)
            }
            size = entry.get("size")
            models.append(DiscoveredModel(
                id=str(name),
                size=size if isinstance(size, int) else None,
                modified_at=entry.get("modified_at") if isinstance(entry.get("modified_at"), str) else None,
                details=details or None,
            ))
        return sorted_models(models)

    async def release(self, config: LLMConfig) -> bool:
        try:
            async with httpx.AsyncClient(timeout=config.timeout, verify=client_ssl_context()) as client:
                response = await client.post(
                    f"{config.base_url.rstrip('/')}/api/chat",
                    headers=bearer_headers(config.api_key),
                    json={"model": config.model, "keep_alive": 0},
                )
            return response.is_success
        except Exception:
            logging.debug("%s release failed for %s", self.display_name, config.model, exc_info=True)
            return False

    async def show(self, base_url: str, api_key: Optional[str], model: str) -> Any:
        return await fetch_json(
            self.display_name,
            base_url,
            "/api/show",
            bearer_headers(api_key),
            body={"model": model},
            timeout=CONTEXT_WINDOW_PROBE_TIMEOUT_SECONDS,
        )

    async def discover(self, config: LLMConfig) -> ProfileLayer:
        fields: Dict[str, Any] = {}
        try:
            payload = await self.show(config.base_url, config.api_key, config.model)
        except Exception:
            logging.debug("%s discovery failed for %s", self.display_name, config.model, exc_info=True)
            return ProfileLayer(ProfileSource.DISCOVERED, fields)
        window = parse_show_window(payload)
        if window:
            loaded = min(window.tokens, TRAINED_WINDOW_CAP_TOKENS) if window.kind == "trained" else window.tokens
            fields["context_window"] = loaded
            fields["context_request"] = loaded
        capabilities = payload.get("capabilities") if isinstance(payload, dict) else None
        if isinstance(capabilities, list):
            names = {item for item in capabilities if isinstance(item, str)}
            fields["tools"] = "tools" in names
            fields["vision"] = "vision" in names
            fields["reasoning"] = "toggle" if "thinking" in names else "none"
        return ProfileLayer(ProfileSource.DISCOVERED, fields)
