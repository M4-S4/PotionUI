"""Tests for src.features.llm_memory.operations (formerly LLMMemoryManager)."""

import pytest
from unittest.mock import MagicMock

from src.features.llm_memory import operations
from src.features.llm_memory.records import LLMMemoryNote


def make_note(**overrides):
    defaults = {
        "id": "note-1",
        "user_id": "user-1",
        "key": "test_key",
        "content": "test content",
        "scope": "global",
        "scope_ref": None,
    }
    defaults.update(overrides)
    return LLMMemoryNote(**defaults)


class TestWriteNote:
    @pytest.mark.parametrize("key,content,scope,scope_ref", [
        pytest.param("test_key", "test content", None, None, id="writes_global_note"),
        pytest.param("model_quirk", "Needs low CFG", "model", "model-1", id="writes_model_scoped_note"),
        pytest.param("preset_quirk", "Prefers 30 steps", "preset", "preset-1", id="writes_preset_scoped_note"),
        pytest.param(
            "captioning_habit", "Captions one image at a time", "mode", "lora-dataset",
            id="writes_mode_scoped_note",
        ),
    ])
    def test_writes_note_with_its_scope(self, key, content, scope, scope_ref):
        repo = MagicMock()
        note = make_note(scope=scope or "global", scope_ref=scope_ref)
        repo.upsert.return_value = note
        kwargs = {} if scope is None else {"scope": scope, "scope_ref": scope_ref}

        result = operations.write_note(repo, user_id="user-1", key=key, content=content, **kwargs)

        assert result == note
        repo.upsert.assert_called_once()
        call_arg = repo.upsert.call_args[0][0]
        assert call_arg.user_id == "user-1"
        assert call_arg.key == key
        assert call_arg.content == content
        assert call_arg.scope == (scope or "global")
        assert call_arg.scope_ref == scope_ref

    @pytest.mark.parametrize("scope,match", [
        pytest.param("mode", "scope_ref is required", id="mode_scope_requires_scope_ref"),
        pytest.param("model", "scope_ref is required", id="model_scope_requires_scope_ref"),
        pytest.param("preset", "scope_ref is required", id="preset_scope_requires_scope_ref"),
        pytest.param("invalid", "Invalid scope", id="invalid_scope_raises_error"),
    ])
    def test_rejects_scope_without_scope_ref_or_unknown_scope(self, scope, match):
        repo = MagicMock()

        with pytest.raises(ValueError, match=match):
            operations.write_note(repo, user_id="user-1", key="test", content="test", scope=scope)

        repo.upsert.assert_not_called()

    def test_content_over_limit_raises_teaching_error(self):
        repo = MagicMock()

        with pytest.raises(ValueError, match="500 characters"):
            operations.write_note(
            repo,
                user_id="user-1",
                key="test",
                content="x" * 501,
            )

        repo.upsert.assert_not_called()

    def test_content_at_limit_passes(self):
        repo = MagicMock()
        note = make_note(content="x" * 500)
        repo.upsert.return_value = note

        result = operations.write_note(repo, user_id="user-1", key="test", content="x" * 500)

        assert result == note
        repo.upsert.assert_called_once()

    def test_global_scope_clears_scope_ref(self):
        repo = MagicMock()
        note = make_note()
        repo.upsert.return_value = note

        operations.write_note(
            repo,
            user_id="user-1",
            key="test",
            content="test",
            scope="global",
            scope_ref="should-be-cleared",
        )

        call_arg = repo.upsert.call_args[0][0]
        assert call_arg.scope_ref is None


class TestReadNotes:
    def test_reads_all_notes(self):
        repo = MagicMock()
        notes = [make_note(id="n1"), make_note(id="n2")]
        repo.list_notes.return_value = notes

        result = operations.read_notes(repo, user_id="user-1")

        assert result == notes
        repo.list_notes.assert_called_once_with(
            user_id="user-1", scope=None, scope_ref=None
        )

    def test_reads_with_scope_filter(self):
        repo = MagicMock()
        repo.list_notes.return_value = []

        operations.read_notes(repo, user_id="user-1", scope="global")

        repo.list_notes.assert_called_once_with(
            user_id="user-1", scope="global", scope_ref=None
        )

    def test_reads_with_model_filter(self):
        repo = MagicMock()
        repo.list_notes.return_value = []

        operations.read_notes(repo, user_id="user-1", scope="model", scope_ref="m-1")

        repo.list_notes.assert_called_once_with(
            user_id="user-1", scope="model", scope_ref="m-1"
        )

    def test_reads_with_preset_filter(self):
        repo = MagicMock()
        repo.list_notes.return_value = []

        operations.read_notes(repo, user_id="user-1", scope="preset", scope_ref="p-1")

        repo.list_notes.assert_called_once_with(
            user_id="user-1", scope="preset", scope_ref="p-1"
        )


class TestGetNote:
    def test_returns_note(self):
        repo = MagicMock()
        note = make_note()
        repo.get_by_id.return_value = note

        result = operations.get_note(repo, user_id="user-1", note_id="note-1")

        assert result == note
        repo.get_by_id.assert_called_once_with("note-1", "user-1")

    def test_returns_none_when_not_found(self):
        repo = MagicMock()
        repo.get_by_id.return_value = None

        result = operations.get_note(repo, user_id="user-1", note_id="missing")

        assert result is None


class TestUpdateNote:
    def test_updates_note(self):
        repo = MagicMock()
        note = make_note(key="new_key", content="new content")
        repo.update.return_value = note

        result = operations.update_note(repo, user_id="user-1", note_id="note-1", key="new_key", content="new content")

        assert result == note
        repo.update.assert_called_once_with("note-1", "user-1", "new_key", "new content")

    def test_returns_none_when_not_found(self):
        repo = MagicMock()
        repo.update.return_value = None

        result = operations.update_note(repo, user_id="user-1", note_id="missing", key="k", content="c")

        assert result is None

    def test_content_over_limit_raises_teaching_error(self):
        repo = MagicMock()

        with pytest.raises(ValueError, match="500 characters"):
            operations.update_note(repo, user_id="user-1", note_id="note-1", key="k", content="x" * 501)

        repo.update.assert_not_called()


class TestGetNoteByKey:
    def test_resolves_global_note(self):
        repo = MagicMock()
        note = make_note()
        repo.get_by_key.return_value = note

        result = operations.get_note_by_key(repo, user_id="user-1", key="test_key", scope="global")

        assert result == note
        repo.get_by_key.assert_called_once_with("user-1", "test_key", "global", None)

    def test_resolves_scoped_note_with_scope_ref(self):
        repo = MagicMock()
        note = make_note(scope="model", scope_ref="model-1")
        repo.get_by_key.return_value = note

        result = operations.get_note_by_key(
            repo,
            user_id="user-1", key="quirk", scope="model", scope_ref="model-1",
        )

        assert result == note
        repo.get_by_key.assert_called_once_with("user-1", "quirk", "model", "model-1")

    def test_resolves_mode_scoped_note_with_scope_ref(self):
        repo = MagicMock()
        note = make_note(scope="mode", scope_ref="lora-dataset")
        repo.get_by_key.return_value = note

        result = operations.get_note_by_key(
            repo,
            user_id="user-1", key="habit", scope="mode", scope_ref="lora-dataset",
        )

        assert result == note
        repo.get_by_key.assert_called_once_with("user-1", "habit", "mode", "lora-dataset")

    def test_returns_none_on_miss(self):
        repo = MagicMock()
        repo.get_by_key.return_value = None

        result = operations.get_note_by_key(repo, user_id="user-1", key="missing", scope="global")

        assert result is None

    def test_invalid_scope_raises_error(self):
        repo = MagicMock()

        with pytest.raises(ValueError, match="Invalid scope"):
            operations.get_note_by_key(repo, user_id="user-1", key="k", scope="bogus")

        repo.get_by_key.assert_not_called()

    def test_preset_scope_requires_scope_ref(self):
        repo = MagicMock()

        with pytest.raises(ValueError, match="scope_ref is required"):
            operations.get_note_by_key(repo, user_id="user-1", key="k", scope="preset")

        repo.get_by_key.assert_not_called()

    def test_model_scope_rejects_empty_string_scope_ref(self):
        repo = MagicMock()

        with pytest.raises(ValueError, match="scope_ref is required"):
            operations.get_note_by_key(repo, user_id="user-1", key="k", scope="model", scope_ref="")

        repo.get_by_key.assert_not_called()


class TestDeleteNote:
    def test_deletes_note(self):
        repo = MagicMock()
        repo.delete.return_value = True

        result = operations.delete_note(repo, user_id="user-1", note_id="note-1")

        assert result is True
        repo.delete.assert_called_once_with("note-1", "user-1")

    def test_returns_false_when_not_found(self):
        repo = MagicMock()
        repo.delete.return_value = False

        result = operations.delete_note(repo, user_id="user-1", note_id="missing")

        assert result is False
