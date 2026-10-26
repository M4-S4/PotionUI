import sqlite3
from typing import Dict, List, Optional, Sequence, Tuple

from src.platform.database import get_cursor
from src.platform.database.rows import now_iso

_LOOKUP_CHUNK = 400

_INSERT_STEP = (
    "INSERT INTO chat_steps "
    "(id, session_id, turn_id, seq, message_id, role, parts, raw, origin, usage, finish, finish_raw, status, created_at) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

_UPSERT_STATE = (
    "INSERT INTO chat_turns (id, session_id, status, state_json, llm_config_id, created_at, updated_at) "
    "VALUES (?, ?, ?, ?, ?, ?, ?) "
    "ON CONFLICT (id) DO UPDATE SET status = excluded.status, state_json = excluded.state_json, "
    "llm_config_id = excluded.llm_config_id, updated_at = excluded.updated_at "
    "WHERE chat_turns.session_id = excluded.session_id"
)


class TurnStoreError(ValueError):
    pass


def _chunks(values: Sequence[str]):
    for start in range(0, len(values), _LOOKUP_CHUNK):
        yield values[start:start + _LOOKUP_CHUNK]


def _next_seq(cursor: sqlite3.Cursor, session_id: str) -> int:
    cursor.execute("SELECT COALESCE(MAX(seq), -1) + 1 FROM chat_steps WHERE session_id = ?", (session_id,))
    return cursor.fetchone()[0]


class TurnRepository:
    def step_rows(self, session_id: str) -> List[sqlite3.Row]:
        with get_cursor() as cursor:
            cursor.execute("SELECT * FROM chat_steps WHERE session_id = ? ORDER BY seq", (session_id,))
            return cursor.fetchall()

    def turn_step_rows(self, turn_id: str) -> List[sqlite3.Row]:
        with get_cursor() as cursor:
            cursor.execute("SELECT * FROM chat_steps WHERE turn_id = ? ORDER BY seq", (turn_id,))
            return cursor.fetchall()

    def step_rows_by_turn(self, turn_ids: Sequence[str]) -> Dict[str, List[sqlite3.Row]]:
        grouped: Dict[str, List[sqlite3.Row]] = {}
        with get_cursor() as cursor:
            for chunk in _chunks(list(dict.fromkeys(turn_ids))):
                marks = ", ".join("?" * len(chunk))
                cursor.execute(f"SELECT * FROM chat_steps WHERE turn_id IN ({marks}) ORDER BY seq", tuple(chunk))
                for row in cursor.fetchall():
                    grouped.setdefault(row["turn_id"], []).append(row)
        return grouped

    def state_jsons(self, turn_ids: Sequence[str]) -> Dict[str, str]:
        found: Dict[str, str] = {}
        with get_cursor() as cursor:
            for chunk in _chunks(list(dict.fromkeys(turn_ids))):
                marks = ", ".join("?" * len(chunk))
                cursor.execute(f"SELECT id, state_json FROM chat_turns WHERE id IN ({marks})", tuple(chunk))
                for row in cursor.fetchall():
                    found[row["id"]] = row["state_json"]
        return found

    def next_seq(self, session_id: str) -> int:
        with get_cursor() as cursor:
            return _next_seq(cursor, session_id)

    def insert_step(self, values: Tuple) -> None:
        step_id, session_id, turn_id, seq = values[0], values[1], values[2], values[3]
        with get_cursor() as cursor:
            cursor.execute("SELECT session_id FROM chat_turns WHERE id = ?", (turn_id,))
            turn = cursor.fetchone()
            if turn is None or turn["session_id"] != session_id:
                raise TurnStoreError(f"turn {turn_id} does not belong to session {session_id}")
            expected = _next_seq(cursor, session_id)
            if seq != expected:
                raise TurnStoreError(f"steps are append-only: expected seq {expected}, got {seq}")
            try:
                cursor.execute(_INSERT_STEP, values[:13] + (values[13] or now_iso(),))
            except sqlite3.IntegrityError as error:
                raise TurnStoreError(f"step {step_id} was refused: {error}") from error

    def upsert_state(self, turn_id: str, session_id: str, status: str, state_json: str, llm_config_id: Optional[str]) -> None:
        now = now_iso()
        with get_cursor() as cursor:
            try:
                cursor.execute(_UPSERT_STATE, (turn_id, session_id, status, state_json, llm_config_id, now, now))
            except sqlite3.IntegrityError as error:
                raise TurnStoreError(str(error)) from error
            if cursor.rowcount == 0:
                raise TurnStoreError(f"turn {turn_id} belongs to another session")

    def state_json(self, turn_id: str) -> Optional[str]:
        with get_cursor() as cursor:
            cursor.execute("SELECT state_json FROM chat_turns WHERE id = ?", (turn_id,))
            row = cursor.fetchone()
        return row["state_json"] if row else None

    def open_state_json(self, session_id: str, statuses: Sequence[str]) -> Optional[str]:
        marks = ", ".join("?" * len(statuses))
        with get_cursor() as cursor:
            cursor.execute(
                f"SELECT state_json FROM chat_turns WHERE session_id = ? AND status IN ({marks}) "
                "ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (session_id, *statuses),
            )
            row = cursor.fetchone()
        return row["state_json"] if row else None

    def state_jsons_with_status(self, status: str) -> List[Tuple[str, str]]:
        with get_cursor() as cursor:
            cursor.execute("SELECT id, state_json FROM chat_turns WHERE status = ?", (status,))
            return [(row["id"], row["state_json"]) for row in cursor.fetchall()]

    def set_status(self, turn_id: str, status: str) -> None:
        with get_cursor() as cursor:
            cursor.execute("UPDATE chat_turns SET status = ?, updated_at = ? WHERE id = ?", (status, now_iso(), turn_id))
