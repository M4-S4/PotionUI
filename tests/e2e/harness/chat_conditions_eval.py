#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
_HARNESS_DIR = Path(__file__).resolve().parent
for _path in (str(REPO_ROOT), str(_HARNESS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

TAB_ID = "conditions-eval-tab"
_PROPOSAL = re.compile(
    r"<tool_action\b([^>]*)\btype=[\"'](update_segment|add_segment)[\"']([^>]*)>([\s\S]*?)</tool_action>"
)

MOOD = {"type": "choice", "options": ["noir", "sunlit", "foggy dawn"], "mode": "shuffle", "pinnedIndex": None}
AGE = {"type": "number", "min": 18, "max": 90, "step": 1, "mode": "shuffle", "value": 18}
RECORD = {"type": "choice", "options": ['12" vinyl', "tape > CD", "cassette"], "mode": "shuffle", "pinnedIndex": None}


@dataclass
class Case:
    id: str
    request: str
    variables: Dict[str, Any]
    segment: str
    renders: Sequence[Tuple[Dict[str, str], Sequence[str], Sequence[str]]] = ()
    explains: Sequence[str] = ()
    explains_any: Sequence[str] = ()
    good: str = ""
    bad: str = ""


def _tag(body: str) -> str:
    return f'<tool_action type="update_segment" segment_index="0" segment_id="s1">{body}</tool_action>'


CASES: Tuple[Case, ...] = (
    Case(
        id="choice_if_else_if_else",
        request=(
            "Add a condition to my prompt: when mood is noir add 'rain-soaked neon', when it is sunlit add "
            "'warm golden backlight', otherwise add 'soft overcast light'."
        ),
        variables={"mood": MOOD},
        segment="portrait of a woman on a city street",
        renders=(
            ({"mood": "noir"}, ["portrait of a woman", "rain-soaked neon"], ["golden", "overcast"]),
            ({"mood": "sunlit"}, ["warm golden backlight"], ["neon", "overcast"]),
            ({"mood": "foggy dawn"}, ["soft overcast light"], ["neon", "golden"]),
        ),
        good=_tag(
            'portrait of a woman on a city street, <<if $mood = "noir">>rain-soaked neon<<else if $mood = "sunlit">>'
            "warm golden backlight<<else>>soft overcast light<<end>>"
        ),
        bad=_tag('portrait of a woman on a city street, <<if $mood = "Noir">>rain-soaked neon<<end>>'),
    ),
    Case(
        id="number_comparison",
        request="Make her hair grey when age is over 65; otherwise keep the auburn hair.",
        variables={"age": AGE},
        segment="portrait of a woman, auburn hair",
        renders=(
            ({"age": "70"}, ["portrait of a woman", "grey"], ["auburn"]),
            ({"age": "65"}, ["auburn"], ["grey"]),
            ({"age": "30"}, ["auburn"], ["grey"]),
        ),
        good=_tag("portrait of a woman, <<if $age > 65>>grey hair<<else>>auburn hair<<end>>"),
        bad=_tag("portrait of a woman, auburn hair, <<if $age > 65>>grey hair<<end>>"),
    ),
    Case(
        id="edit_existing_condition",
        request=(
            "Change the condition so grey hair starts above 70 instead of 65, and add a branch that gives "
            "freckles when age is under 25."
        ),
        variables={"age": AGE},
        segment="portrait of a woman, <<if $age > 65>>grey hair<<else>>auburn hair<<end>>",
        renders=(
            ({"age": "72"}, ["portrait of a woman", "grey"], ["auburn", "freckles"]),
            ({"age": "68"}, ["auburn"], ["grey"]),
            ({"age": "20"}, ["freckles"], ["grey"]),
            ({"age": "40"}, ["auburn"], ["grey", "freckles"]),
        ),
        good=_tag(
            "portrait of a woman, <<if $age > 70>>grey hair<<else if $age < 25>>freckles<<else>>auburn hair<<end>>"
        ),
        bad=_tag("portrait of a woman, <<if $age > 70>>grey hair<<elif $age < 25>>freckles<<else>>auburn hair<<end>>"),
    ),
    Case(
        id="explain_existing_condition",
        request="Explain in plain words what the condition in my prompt does: when do I get each text? Don't change anything.",
        variables={"mood": MOOD, "age": AGE},
        segment=(
            'cinematic street photo, <<if $mood = "noir" and $age >= 60>>rain-soaked neon, trench coat'
            '<<else if $mood in "sunlit", "foggy dawn">>warm backlight<<else>>overcast sky<<end>>'
        ),
        explains=("noir", "60", "sunlit", "foggy dawn"),
        explains_any=("rain-soaked neon", "trench coat", "warm backlight", "overcast sky"),
        good=(
            "When mood is noir and age is 60 or more you get rain-soaked neon, trench coat. When mood is sunlit "
            "or foggy dawn you get warm backlight. Otherwise you get overcast sky."
        ),
        bad=_tag("cinematic street photo, overcast sky"),
    ),
    Case(
        id="value_with_special_characters",
        request=(
            "When record is the 12\" vinyl option add 'spinning turntable'; when it is the tape > CD option "
            "add 'retro boombox'; for any other record add nothing."
        ),
        variables={"record": RECORD},
        segment="still life of a music shelf",
        renders=(
            ({"record": '12" vinyl'}, ["still life", "spinning turntable"], ["boombox"]),
            ({"record": "tape > CD"}, ["retro boombox"], ["turntable"]),
            ({"record": "cassette"}, ["still life"], ["turntable", "boombox"]),
        ),
        good=_tag(
            'still life of a music shelf, <<if $record = "12\\" vinyl">>spinning turntable'
            '<<else if $record = "tape > CD">>retro boombox<<end>>'
        ),
        bad=_tag('still life of a music shelf, <<if $record = "12" vinyl">>spinning turntable<<end>>'),
    ),
)


def proposals(content: str) -> List[str]:
    return [m.group(4).strip() for m in _PROPOSAL.finditer(content or "")]


def score(case: Case, content: str) -> Tuple[bool, str]:
    from src.features.llm.tools.builtin.condition_text import condition_problems
    from src.features.prompt import conditions

    found = proposals(content)
    if case.explains:
        if found:
            return False, f"proposed {len(found)} segment change(s) instead of only explaining"
        text = (content or "").casefold()
        missing = [word for word in case.explains if word.casefold() not in text]
        if missing:
            return False, f"explanation never mentions: {', '.join(missing)}"
        hits = [phrase for phrase in case.explains_any if phrase.casefold() in text]
        if len(hits) < 2:
            return False, f"explanation names {len(hits)} of the branch texts; expected at least 2"
        return True, "explained every branch"
    if not found:
        return False, "no update_segment/add_segment proposal in the reply"
    reasons: List[str] = []
    for body in found:
        problems = condition_problems(body, case.variables)
        if problems:
            reasons.append("; ".join(problems[:2]))
            continue
        parsed = conditions.parse_template(body)
        if parsed is None or len(parsed.blocks) != 1:
            reasons.append(f"expected one condition block, found {0 if parsed is None else len(parsed.blocks)}")
            continue
        failure = _render_failure(case, body)
        if failure is None:
            return True, f"proposal renders right: {body}"
        reasons.append(failure)
    return False, " | ".join(reasons)


def _render_failure(case: Case, body: str) -> Optional[str]:
    from src.features.prompt import conditions

    for values, present, absent in case.renders:
        rendered, _ = conditions.render_conditions(body, values)
        lowered = rendered.casefold()
        missing = [word for word in present if word.casefold() not in lowered]
        extra = [word for word in absent if word.casefold() in lowered]
        if missing or extra:
            return f"with {values} it renders {rendered!r} (missing {missing}, unexpected {extra})"
    return None


def self_check() -> int:
    failures = 0
    for case in CASES:
        good, good_detail = score(case, case.good)
        bad, bad_detail = score(case, case.bad)
        ok = good and not bad
        failures += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {case.id}: good -> {good_detail}; bad -> {bad_detail}")
    return 1 if failures else 0


def context_metadata(case: Case) -> Dict[str, Any]:
    return {
        "form_state": {"preset": "sdxl/base", "mode": "txt2img", "tab_id": TAB_ID, "form_data": {"prompt": case.segment}},
        "segments": [{"index": 0, "id": "s1", "type": "content", "enabled": True, "content": case.segment}],
    }


def scratch_paths(scratch: Path) -> Dict[str, Path]:
    return {
        "POTIONUI_DB_PATH": scratch / "storage" / "db.sqlite",
        "POTIONUI_STORAGE_PATH": scratch / "storage",
        "POTIONUI_MODELS_DIR": scratch / "models",
        "POTIONUI_RECIPES_DIR": scratch / "recipes",
        "POTIONUI_LOG_DIR": scratch / "logs",
    }


def assert_scratch(paths: Dict[str, Path], scratch: Path) -> None:
    root = scratch.resolve()
    repo = REPO_ROOT.resolve()
    if root == repo or repo in root.parents or root in repo.parents:
        raise RuntimeError(f"scratch directory {root} overlaps the repository")
    for name, path in paths.items():
        resolved = path.resolve()
        if root not in resolved.parents:
            raise RuntimeError(f"{name}={resolved} is outside the scratch directory {root}")
        if os.environ.get(name) != str(path):
            raise RuntimeError(f"{name} is not pinned to the scratch path")


def plan(args: argparse.Namespace) -> None:
    print("Would boot a throwaway backend on a temp directory and drive these cases:")
    for case in CASES:
        print(f"  {case.id}: {case.request}")
        print(f"    variables: {json.dumps(case.variables)}")
        print(f"    segment:   {case.segment}")
    print(f"Model: type={args.type} base_url={args.base_url} model={args.model}")


def _llm_config(args: argparse.Namespace) -> Dict[str, Any]:
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        raise SystemExit(f"{args.api_key_env} is not set")
    return {
        "name": f"conditions eval ({args.model})",
        "type": args.type,
        "enabled": True,
        "base_url": args.base_url,
        "api_key": api_key,
        "model": args.model,
        "system_message": "",
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "memory_reflection": False,
        "provider_options": {"context_window": args.context_window} if args.context_window else None,
    }


def _ok(stage: str, response) -> Dict[str, Any]:
    from e2e_harness import raise_for_status

    body = raise_for_status(stage, response, stage)
    if body.get("success") is False:
        raise RuntimeError(f"{stage}: {body.get('error')}: {body.get('message')}")
    return body


def run_case(client, config_id: str, case: Case) -> Dict[str, Any]:
    from chat_eval import parse_sse_events

    _ok("tab draft", client.put(f"/api/tab-drafts/{TAB_ID}", json={"doc": {"variables": case.variables}}))
    session = _ok("session", client.post("/api/chat/sessions", json={
        "llm_config_id": config_id, "mode": "generation", "name": f"conditions-eval:{case.id}",
    }))["data"]
    response = client.session.post(
        f"{client.base_url}/api/chat/sessions/{session['id']}/messages/stream",
        timeout=900.0,
        json={"content": case.request, "context_metadata": context_metadata(case)},
        headers={"Accept": "text/event-stream", "Authorization": f"Bearer {client.token}"},
    )
    events = parse_sse_events(response.content.decode("utf-8"))
    done = next((e for e in events if e["event"] == "done"), None)
    if done is None:
        error = next((e for e in events if e["event"] == "error"), None)
        return {"case": case.id, "passed": False, "detail": f"no reply: {error['data'] if error else response.status_code}"}
    message = done["data"].get("assistant_message") or {}
    content = message.get("content") or ""
    passed, detail = score(case, content)
    calls = [execution.get("tool_name") for execution in message.get("tool_executions") or []]
    return {"case": case.id, "passed": passed, "detail": detail, "tools": calls, "reply": content}


def live(args: argparse.Namespace) -> int:
    scratch = Path(tempfile.mkdtemp(prefix="potionui-conditions-eval-"))
    instance = None
    results: List[Dict[str, Any]] = []
    try:
        paths = scratch_paths(scratch)
        os.environ.update({name: str(path) for name, path in paths.items()})
        assert_scratch(paths, scratch)

        from e2e_harness import (
            EphemeralInstance,
            SetupClient,
            claim_owner,
            pick_free_port,
            spawn_backend,
            stage_wait_for_health,
            teardown_backend,
        )

        for name in ("POTIONUI_STORAGE_PATH", "POTIONUI_MODELS_DIR", "POTIONUI_RECIPES_DIR", "POTIONUI_LOG_DIR"):
            paths[name].mkdir(parents=True, exist_ok=True)
        instance = EphemeralInstance(
            instance_dir=scratch,
            storage_dir=paths["POTIONUI_STORAGE_PATH"],
            db_path=paths["POTIONUI_DB_PATH"],
            models_dir=paths["POTIONUI_MODELS_DIR"],
            recipes_dir=paths["POTIONUI_RECIPES_DIR"],
            port=pick_free_port(),
        )
        spawn_backend(instance, REPO_ROOT)
        client = SetupClient(f"http://127.0.0.1:{instance.port}")
        stage_wait_for_health(client)
        claim_owner(client, username="conditions-eval", email="conditions-eval@example.com", password=os.urandom(12).hex())
        config_id = _ok("llm config", client.post("/api/llm/configurations", json=_llm_config(args)))["data"]["id"]
        for case in CASES:
            if args.case and case.id not in args.case:
                continue
            try:
                result = run_case(client, config_id, case)
            except Exception as exc:
                result = {"case": case.id, "passed": False, "detail": f"{type(exc).__name__}: {exc}"}
            results.append(result)
            print(f"{'PASS' if result['passed'] else 'FAIL'}  {case.id}: {result['detail']}", flush=True)
            if result.get("tools"):
                print(f"      tools: {', '.join(result['tools'])}")
    finally:
        if instance is not None:
            teardown_backend(instance, keep=args.keep)
        if not args.keep:
            shutil.rmtree(scratch, ignore_errors=True)
    passed = sum(1 for result in results if result["passed"])
    print(f"{passed}/{len(results)} cases passed")
    if args.out:
        Path(args.out).write_text(
            json.dumps({"model": args.model, "results": results}, indent=2) + "\n", encoding="utf-8"
        )
    return 0 if results and passed == len(results) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Drive PotionAI through five prompt-condition requests against one LLM, on a throwaway backend "
            "whose database and storage live in a fresh temp directory, and print pass/fail per case."
        ),
        epilog=(
            "Examples: --self-check (offline, scores canned replies); --dry-run (prints the plan); "
            "--type ollama --base-url http://127.0.0.1:11434 --model qwen3:14b"
        ),
    )
    parser.add_argument("--self-check", action="store_true", help="score canned good and bad replies offline")
    parser.add_argument("--dry-run", action="store_true", help="print the cases and the model, start nothing")
    parser.add_argument("--type", default="ollama", help="LLM provider type (ollama, openai, native, ...)")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", help="model name on the provider")
    parser.add_argument("--api-key-env", help="environment variable holding the provider API key")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--context-window", type=int)
    parser.add_argument("--case", action="append", choices=[case.id for case in CASES], help="run only this case")
    parser.add_argument("--out", help="write the replies and verdicts as JSON")
    parser.add_argument("--keep", action="store_true", help="keep the temp directory and its backend log")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_check:
        return self_check()
    if args.dry_run:
        plan(args)
        return 0
    if not args.model:
        build_parser().error("--model is required unless --self-check or --dry-run")
    return live(args)


if __name__ == "__main__":
    sys.exit(main())
