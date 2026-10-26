#!/usr/bin/env python3
import argparse
import asyncio
import importlib
import inspect
import json
import os
import pkgutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.llm.harness import adapters as adapter_package
from src.features.llm.harness.assembler import StepAssembler
from src.features.llm.harness.errors import HarnessError
from src.features.llm.harness.messages import Message, Role, TextPart, ToolResultPart
from src.features.llm.harness.transport import HttpAdapter
from tests.features.llm.harness import cassettes
from tests.fixtures.llm_harness import decode_chunks

DEFAULT_TOOL_RESULT = '{"ok": true}'
DEFAULT_MAX_STEPS = 6


def find_adapter_class(name: str, dotted: Optional[str]) -> type:
    if dotted:
        module_name, _, class_name = dotted.partition(":")
        return getattr(importlib.import_module(module_name), class_name)
    for module_info in pkgutil.iter_modules(adapter_package.__path__):
        module = importlib.import_module(f"{adapter_package.__name__}.{module_info.name}")
        for _, member in inspect.getmembers(module, inspect.isclass):
            if member.__module__ == module.__name__ and issubclass(member, HttpAdapter) and member.name == name:
                return member
    raise SystemExit(f"no adapter named {name!r} under src/features/llm/harness/adapters; pass --adapter-class module:Class")


def build_config_data(spec: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    data = dict(spec.get("config", {}))
    env_name = data.pop("api_key_env", None)
    api_key = None
    if env_name:
        api_key = os.environ.get(env_name)
        if not api_key:
            raise SystemExit(f"{env_name} is not set; export it to record against the real API")
        data["api_key"] = api_key
    return data, api_key


def tool_message(index: int, message: Message, results: Dict[str, str]) -> Message:
    parts = tuple(
        ToolResultPart(
            call_id=call.call_id,
            name=call.name,
            content=(TextPart(results.get(call.name, DEFAULT_TOOL_RESULT)),),
        )
        for call in message.tool_calls()
    )
    return Message(id=f"tool_{index}", role=Role.TOOL, parts=parts)


def assert_no_secret(directory: Path, secret: Optional[str]) -> None:
    if not secret:
        return
    for path in directory.iterdir():
        if secret in path.read_text(encoding="utf-8", errors="replace"):
            raise SystemExit(f"{path} contains the API key; the recording was not kept clean, delete {directory}")


async def record(spec: Dict[str, Any], out: Path, adapter_class: type, force: bool) -> List[Path]:
    scenario_dir = out / spec["adapter"] / spec["scenario"]
    if scenario_dir.exists() and not force:
        raise SystemExit(f"{scenario_dir} exists; pass --force to replace it")
    config_data, api_key = build_config_data(spec)
    adapter = adapter_class()
    meta = {
        "adapter": spec["adapter"],
        "provider": spec.get("provider", spec["adapter"]),
        "model": spec["model"],
        "source": cassettes.RECORDED,
        "doc_url": spec.get("doc_url"),
        "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "profile_overrides": spec.get("profile_overrides", {}),
    }
    profile = cassettes.profile_for(meta)
    carried = {
        "config": config_data,
        "system": spec.get("system", ""),
        "tools": spec.get("tools", []),
        "settings": spec.get("settings", {}),
    }
    results = spec.get("tool_results", {})
    max_steps = spec.get("max_steps", DEFAULT_MAX_STEPS)
    history: List[Message] = []
    written: List[Path] = []
    step_number = 0
    for turn_index, text in enumerate(spec["turns"], start=1):
        pending: List[Message] = [Message(id=f"user_{turn_index}", role=Role.USER, parts=(TextPart(text),))]
        while pending and step_number < max_steps:
            step_number += 1
            history = history + pending
            request = cassettes.request_from(meta, carried, history)
            wire = adapter.render(request, profile)
            chunks: List[bytes] = []
            async for chunk in adapter.send(wire, request):
                chunks.append(chunk)
            try:
                events: Any = decode_chunks(adapter.decoder(request, profile), chunks)
            except HarnessError as error:
                events = error
            directory = scenario_dir / f"{step_number:02d}"
            assembler = None
            if not isinstance(events, HarnessError):
                origin = cassettes.origin_for(meta, cassettes.replayed_config(request.config))
                assembler = StepAssembler(f"assistant_{step_number}", origin)
                assembler.feed_all(events)
            cassettes.write_step(
                directory,
                meta=meta,
                input=cassettes.input_from(request, pending, full=step_number == 1),
                wire=wire,
                chunks=chunks,
                events=events,
                step=cassettes.step_payload(assembler) if assembler else None,
            )
            assert_no_secret(directory, api_key)
            written.append(directory)
            if assembler is None:
                return written
            assistant = assembler.message()
            history = history + [assistant]
            pending = [tool_message(step_number, assistant, results)] if assistant.tool_calls() else []
    return written


def list_hand_built() -> None:
    entries = cassettes.hand_built_scenarios()
    for adapter, scenario in entries:
        print(f"hand-built  {adapter}/{scenario}")
    print(f"{len(entries)} hand-built scenario(s) still to be replaced by a recording")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Record a real provider conversation as an LLM cassette.")
    parser.add_argument("spec", nargs="?", help="scenario spec JSON")
    parser.add_argument("--out", default=str(cassettes.CASSETTE_ROOT), help="cassette root directory")
    parser.add_argument("--adapter-class", help="module:Class when the adapter is not under harness/adapters")
    parser.add_argument("--force", action="store_true", help="replace an existing scenario")
    parser.add_argument("--list", action="store_true", help="list hand-built scenarios and exit")
    args = parser.parse_args(argv)
    if args.list:
        list_hand_built()
        return 0
    if not args.spec:
        parser.error("a spec file is required unless --list is given")
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    adapter_class = find_adapter_class(spec["adapter"], args.adapter_class)
    written = asyncio.run(record(spec, Path(args.out), adapter_class, args.force))
    for directory in written:
        print(f"recorded {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
