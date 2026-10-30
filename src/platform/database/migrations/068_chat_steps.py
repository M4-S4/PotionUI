import json
import math
import mimetypes
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.platform.database.database import db

_NAME = "068_chat_steps"

_BATCH_SESSIONS = 25

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS chat_turns (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
        status TEXT NOT NULL,
        state_json TEXT NOT NULL,
        llm_config_id TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chat_turns_session ON chat_turns(session_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_chat_turns_status ON chat_turns(status)",
    """
    CREATE TABLE IF NOT EXISTS chat_steps (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
        turn_id TEXT NOT NULL REFERENCES chat_turns(id) ON DELETE CASCADE,
        seq INTEGER NOT NULL,
        message_id TEXT NOT NULL,
        role TEXT NOT NULL,
        parts TEXT NOT NULL,
        raw TEXT,
        origin TEXT,
        usage TEXT,
        finish TEXT,
        finish_raw TEXT,
        status TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (session_id, seq)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chat_steps_turn ON chat_steps(turn_id, seq)",
    """
    CREATE TRIGGER IF NOT EXISTS chat_steps_append_only
    BEFORE UPDATE ON chat_steps
    BEGIN
        SELECT RAISE(ABORT, 'chat_steps is append-only');
    END
    """,
)

_INSERT_TURN = (
    "INSERT INTO chat_turns (id, session_id, status, state_json, llm_config_id, created_at, updated_at) "
    "VALUES (?, ?, ?, ?, ?, ?, ?)"
)

_INSERT_STEP = (
    "INSERT INTO chat_steps "
    "(id, session_id, turn_id, seq, message_id, role, parts, raw, origin, usage, finish, finish_raw, status, created_at) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

_UPDATE_MESSAGE = "UPDATE chat_messages SET turn_id = ?, metadata = COALESCE(?, metadata) WHERE id = ?"

_ORIGIN = {"adapter": "legacy", "provider": "legacy", "model": "legacy", "credential": None}

_SETTINGS = {
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "min_p": None,
    "max_tokens": None,
    "reasoning": None,
    "tool_choice": None,
    "response_schema": None,
    "cache_key": None,
    "cache_ttl": None,
    "keep_model_loaded": False,
}

_PROFILE = {
    "adapter": "legacy",
    "model": "legacy",
    "context_window": None,
    "context_request": None,
    "max_output_tokens": None,
    "tools": True,
    "prompt_tools": False,
    "parallel_tools": False,
    "tool_choice": ["auto"],
    "tool_choice_with_reasoning": ["auto"],
    "strict_tools": False,
    "tool_id_rule": "passthrough",
    "structured_output": "prompted",
    "reasoning": "none",
    "reasoning_levels": [],
    "reasoning_toggle": None,
    "reasoning_field": None,
    "server": None,
    "replay": "send_all",
    "replay_required_with_tools": False,
    "reasoning_prefix_bound": False,
    "interleaved_reasoning": False,
    "vision": False,
    "max_images": 1,
    "tool_result_media": False,
    "sampling_rejected": [],
    "sampling_ignored_with_reasoning": [],
    "recommended_sampling": {},
    "cache": "none",
    "cache_min_tokens": None,
    "prices": None,
    "verified": None,
    "sources": {},
}

_MISSING_RESULT = "No result was recorded for this call."
_EXPIRED_RESULT = "The approval request expired before it was decided."
_EXPIRED_NOTICE = "The approval request expired before it was decided."


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def _finite(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def _text(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text, "phase": None}


def _reasoning(text: str, duration_ms: Optional[int] = None) -> Dict[str, Any]:
    return {
        "type": "reasoning",
        "text": text,
        "summary": "",
        "redacted": False,
        "opaque": None,
        "duration_ms": duration_ms,
    }


def _image(ref: str) -> Dict[str, Any]:
    guessed = mimetypes.guess_type(ref)[0]
    media_type = guessed if guessed and guessed.startswith("image/") else "image/png"
    return {"type": "image", "media_type": media_type, "data": None, "ref": ref}


def _call(call_id: str, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "tool_call",
        "call_id": call_id,
        "name": name,
        "args": args,
        "args_raw": None,
        "wire_id": call_id,
        "opaque": None,
    }


def _result(call_id: str, name: str, text: str, is_error: bool) -> Dict[str, Any]:
    return {"type": "tool_result", "call_id": call_id, "name": name, "content": [_text(text)], "is_error": is_error}


def _int_or_none(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if math.isfinite(value) else None


def _usage(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prompt = _int_or_none(metadata.get("prompt_tokens"))
    completion = _int_or_none(metadata.get("completion_tokens"))
    total = _int_or_none(metadata.get("tokens_used"))
    if prompt is None and completion is None and total is None:
        return None
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": total,
        "reasoning_tokens": None,
        "cached_input_tokens": None,
        "cache_write_tokens": None,
        "cost": None,
        "raw": None,
    }


def _listed(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _arguments(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        value = _loads(value)
    return _finite(value) if isinstance(value, dict) else {}


def _thinking_segments(metadata: Dict[str, Any]) -> List[tuple]:
    thinking = metadata.get("thinking")
    segments = _listed(thinking.get("segments")) if isinstance(thinking, dict) else []
    return [
        (segment["text"], _int_or_none(segment.get("duration_ms")))
        for segment in segments
        if isinstance(segment, dict) and isinstance(segment.get("text"), str)
    ]


def _resolved_text(record: Dict[str, Any]) -> Optional[str]:
    status = record.get("approval_status")
    result = record.get("result") if isinstance(record.get("result"), dict) else {}
    if status == "approved":
        return str(result.get("data") or "")
    if status == "failed":
        return f"Error: {result.get('error') or 'unknown error'}"
    if status == "rejected":
        reason = record.get("approval_reason")
        return f"User rejected this. Reason: {reason}" if reason else "User rejected this."
    return None


def _outcome(text: Optional[str], record: Optional[Dict[str, Any]]) -> tuple:
    status = record.get("approval_status") if record else None
    if status == "pending":
        return (_EXPIRED_RESULT if text is None else text), True
    if text is None:
        resolved = _resolved_text(record) if record else None
        text = resolved if resolved is not None else _MISSING_RESULT
    result = record.get("result") if record and isinstance(record.get("result"), dict) else {}
    failed = record is not None and (status in ("failed", "rejected") or result.get("success") is False)
    return text, failed


def _state_json(
    turn_id: str,
    session: Any,
    status: str,
    first_seq: int,
    rounds: int,
    notices: List[Dict[str, Any]],
) -> str:
    return _dumps({
        "turn_id": turn_id,
        "session_id": session["id"],
        "status": status,
        "snapshot": {
            "llm_config_id": session["llm_config_id"] or "",
            "system": "",
            "tools": [],
            "settings": _SETTINGS,
            "profile": _PROFILE,
            "mode": session["mode"],
            "deferred_tools": {},
        },
        "first_seq": first_seq,
        "rounds_used": rounds,
        "max_rounds": 20,
        "pending": [],
        "gathered": [],
        "failed_calls": {},
        "tool_failures": {},
        "applied_calls": [],
        "loaded_tool_groups": [],
        "rescue_retries": 0,
        "notices": notices,
        "error": None,
        "prefix_changed_at": None,
    })


def _stamp(value: Any) -> str:
    return value if isinstance(value, str) and value else datetime.now(timezone.utc).isoformat()


class _SessionPlan:
    def __init__(self, session: Any) -> None:
        self.session = session
        self.session_id = session["id"]
        self.seq = 0
        self.steps: List[tuple] = []
        self.turns: List[tuple] = []
        self.updates: List[tuple] = []
        self.turn: Optional[Dict[str, Any]] = None

    def open_turn(self, turn_id: str, created_at: Any) -> None:
        self.close_turn()
        self.turn = {
            "id": turn_id,
            "first_seq": self.seq,
            "created_at": _stamp(created_at),
            "updated_at": _stamp(created_at),
            "rounds": 0,
            "assistant_rows": 0,
            "expired_seq": None,
        }

    def close_turn(self) -> None:
        turn = self.turn
        if turn is None:
            return
        notices: List[Dict[str, Any]] = []
        if turn["expired_seq"] is not None:
            status = "cancelled"
            notices.append({
                "kind": "approval_expired",
                "message": _EXPIRED_NOTICE,
                "step_seq": turn["expired_seq"],
                "detail": None,
            })
        elif turn["assistant_rows"] == 0:
            status = "cancelled"
        else:
            status = "completed"
        self.turns.append((
            turn["id"],
            self.session_id,
            status,
            _state_json(turn["id"], self.session, status, turn["first_seq"], turn["rounds"], notices),
            self.session["llm_config_id"] or None,
            turn["created_at"],
            turn["updated_at"],
        ))
        self.turn = None

    def add_step(
        self,
        step_id: str,
        role: str,
        parts: List[Dict[str, Any]],
        created_at: Any,
        finish: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
    ) -> int:
        assert self.turn is not None
        seq = self.seq
        self.seq += 1
        self.steps.append((
            step_id,
            self.session_id,
            self.turn["id"],
            seq,
            step_id,
            role,
            _dumps(parts),
            None,
            _dumps(_ORIGIN) if role == "assistant" else None,
            _dumps(usage) if usage is not None else None,
            finish,
            None,
            "completed",
            _stamp(created_at),
        ))
        return seq

    def user_row(self, row: Any, metadata: Dict[str, Any]) -> None:
        self.open_turn(row["id"], row["created_at"])
        parts = [_text(row["content"] or "")]
        image = metadata.get("image_path")
        if isinstance(image, str) and image:
            parts.append(_image(image))
        self.add_step(row["id"], "user", parts, row["created_at"])
        self.updates.append((row["id"], row["id"], None))

    def assistant_row(self, row: Any, metadata: Dict[str, Any], raw_metadata: Optional[Dict[str, Any]]) -> None:
        if self.turn is None:
            self.open_turn(row["id"], row["created_at"])
        turn = self.turn
        assert turn is not None
        drafts, expired_index = self._drafts(row, metadata)
        first_seq = self.seq
        seqs = []
        for position, draft in enumerate(drafts):
            seqs.append(self.add_step(
                f"{row['id']}:{position}",
                draft["role"],
                draft["parts"],
                row["created_at"],
                finish=draft.get("finish"),
                usage=draft.get("usage"),
            ))
        turn["assistant_rows"] += 1
        turn["updated_at"] = _stamp(row["created_at"])
        rounds = _int_or_none(metadata.get("turn_rounds")) or 0
        turn["rounds"] = max(turn["rounds"], rounds)
        if expired_index is not None:
            turn["expired_seq"] = seqs[expired_index]
        patch = None
        if raw_metadata is not None:
            patched = dict(raw_metadata)
            patched["step_range"] = {"first": first_seq, "last": self.seq - 1}
            for record in _listed(patched.get("tool_executions")):
                if isinstance(record, dict) and record.get("approval_status") == "pending":
                    record["approval_status"] = "expired"
            patch = json.dumps(patched)
        self.updates.append((row["id"], turn["id"], patch))

    def _drafts(self, row: Any, metadata: Dict[str, Any]) -> tuple:
        row_id = row["id"]
        content = row["content"] or ""
        has_text = bool(content.strip())
        transcript = [item for item in _listed(metadata.get("tool_transcript")) if isinstance(item, dict)]
        executions = [item for item in _listed(metadata.get("tool_executions")) if isinstance(item, dict)]
        records = {item["tool_call_id"]: item for item in executions if item.get("tool_call_id")}
        paused = any(item.get("approval_status") for item in executions)
        round_texts = {
            item["reasoning"].strip()
            for item in transcript
            if isinstance(item.get("reasoning"), str) and item["reasoning"].strip()
        }
        segments = _thinking_segments(metadata)
        durations = {text.strip(): duration for text, duration in segments}
        leftover = [
            _reasoning(text, duration)
            for text, duration in segments
            if text.strip() and text.strip() not in round_texts
        ]

        drafts: List[Dict[str, Any]] = []
        expired_index: Optional[int] = None
        ordinal = 0
        for item in transcript:
            calls = [call for call in _listed(item.get("calls")) if isinstance(call, dict)]
            if not calls:
                continue
            results = {
                entry.get("tool_call_id"): entry.get("content", "")
                for entry in _listed(item.get("results"))
                if isinstance(entry, dict)
            }
            lead = (
                [_reasoning(item["reasoning"], durations.get(item["reasoning"].strip()))]
                if isinstance(item.get("reasoning"), str) and item["reasoning"]
                else []
            )
            call_parts = []
            result_parts = []
            for call in calls:
                ordinal += 1
                wire_id = call.get("id")
                call_id = wire_id or f"{row_id}.c{ordinal}"
                name = call.get("name") or ""
                record = records.get(wire_id) if wire_id else None
                text, failed = _outcome(results.get(wire_id), record)
                call_parts.append(_call(call_id, name, _arguments(call.get("arguments"))))
                result_parts.append(_result(call_id, name, text, failed))
                if record is not None and record.get("approval_status") == "pending" and expired_index is None:
                    expired_index = len(drafts)
            drafts.append({"role": "assistant", "lead": lead, "text": [], "calls": call_parts, "finish": "tool_calls"})
            drafts.append({"role": "tool", "parts": result_parts})

        usage = _usage(metadata)
        if not drafts:
            drafts.append({
                "role": "assistant",
                "lead": leftover,
                "text": [_text(content)],
                "calls": [],
                "finish": "stop",
            })
        elif has_text and paused:
            first = drafts[0]
            first["lead"] = first["lead"] + leftover
            first["text"] = [_text(content)]
        elif has_text:
            drafts.append({
                "role": "assistant",
                "lead": leftover,
                "text": [_text(content)],
                "calls": [],
                "finish": "stop",
            })
        else:
            drafts[0]["lead"] = drafts[0]["lead"] + leftover

        assistants = [draft for draft in drafts if draft["role"] == "assistant"]
        for draft in assistants:
            draft["parts"] = draft["lead"] + draft["text"] + draft["calls"]
        if usage is not None:
            assistants[-1]["usage"] = usage
        return drafts, expired_index


def _plan(session: Any, rows: List[Any]) -> _SessionPlan:
    plan = _SessionPlan(session)
    for row in rows:
        parsed = _loads(row["metadata"]) if row["metadata"] else {}
        stored = parsed if isinstance(parsed, dict) else None
        if row["role"] == "user":
            plan.user_row(row, stored or {})
        elif row["role"] == "assistant":
            plan.assistant_row(row, stored or {}, stored)
    plan.close_turn()
    return plan


def _loads(text: str) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _convert() -> int:
    after = ""
    converted = 0
    while True:
        with db.get_cursor() as cursor:
            cursor.execute(
                "SELECT s.id, s.llm_config_id, s.mode FROM chat_sessions s WHERE s.id > ? "
                "AND EXISTS (SELECT 1 FROM chat_messages m WHERE m.session_id = s.id) "
                "AND NOT EXISTS (SELECT 1 FROM chat_steps t WHERE t.session_id = s.id) "
                "ORDER BY s.id LIMIT ?",
                (after, _BATCH_SESSIONS),
            )
            sessions = cursor.fetchall()
            if not sessions:
                return converted
            for session in sessions:
                cursor.execute(
                    "SELECT id, role, content, metadata, created_at FROM chat_messages "
                    "WHERE session_id = ? ORDER BY created_at ASC, rowid ASC",
                    (session["id"],),
                )
                try:
                    plan = _plan(session, cursor.fetchall())
                except Exception as error:
                    print(f"Migration {_NAME}: session {session['id']} left unconverted ({error})")
                    continue
                cursor.executemany(_INSERT_TURN, plan.turns)
                cursor.executemany(_INSERT_STEP, plan.steps)
                cursor.executemany(
                    _UPDATE_MESSAGE, [(turn_id, patch, message_id) for message_id, turn_id, patch in plan.updates]
                )
                converted += 1
            after = sessions[-1]["id"]


def up():
    with db.get_cursor() as cursor:
        for statement in _SCHEMA:
            cursor.execute(statement)
        cursor.execute("PRAGMA table_info(chat_messages)")
        if "turn_id" not in {row[1] for row in cursor.fetchall()}:
            cursor.execute("ALTER TABLE chat_messages ADD COLUMN turn_id TEXT")

    converted = _convert()
    print(f"Migration {_NAME}: applied ({converted} sessions converted)")
