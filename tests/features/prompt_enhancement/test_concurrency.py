import asyncio
from types import SimpleNamespace

import pytest

from src.features.prompt_enhancement import operations
from src.features.prompt_enhancement.guidelines import PROMPT_ENHANCEMENT_GUIDELINES
from tests.features.prompt_enhancement.test_operations import LONG_PROMPT, make_collaborators, make_prompt_db, run_enhance


class _TimedLLM:
    def __init__(self):
        self.in_flight = 0
        self.max_writes_in_flight = 0
        self.write_calls = 0
        self.events = []

    async def generate_with_history(self, messages, llm_id, custom_system_message=None, options_override=None, **kwargs):
        if custom_system_message == operations.CONCEPT_SYSTEM_MESSAGE:
            self.events.append("concepts_start")
            await asyncio.sleep(0.02)
            self.events.append("concepts_end")
            return SimpleNamespace(content='["fox"]')
        if custom_system_message == operations.IDEATE_SYSTEM_MESSAGE:
            return SimpleNamespace(content="1. misty dawn\n2. neon night")
        assert custom_system_message == PROMPT_ENHANCEMENT_GUIDELINES
        self.write_calls += 1
        index = self.write_calls
        self.in_flight += 1
        self.max_writes_in_flight = max(self.max_writes_in_flight, self.in_flight)
        await asyncio.sleep(0.03 if index == 1 else 0.005)
        self.in_flight -= 1
        return SimpleNamespace(content=f"{LONG_PROMPT} variant {index}")


class TestWriteStageConcurrency:
    @pytest.mark.asyncio
    async def test_candidates_are_written_concurrently_and_keep_direction_order(self):
        llm = _TimedLLM()
        collaborators = make_collaborators(llm)

        events = await run_enhance(collaborators, n_candidates=2)

        assert llm.max_writes_in_flight == 2
        candidates = events[-1]["data"]["candidates"]
        assert [c["direction"] for c in candidates] == ["misty dawn", "neon night"]
        assert candidates[0]["text"].endswith("variant 1")
        assert candidates[1]["text"].endswith("variant 2")

    @pytest.mark.asyncio
    async def test_stage_events_still_bracket_the_write_stage(self):
        llm = _TimedLLM()
        collaborators = make_collaborators(llm)

        events = await run_enhance(collaborators, n_candidates=2)

        stages = [(e["type"], e.get("stage")) for e in events if e["type"] != "result"]
        assert stages[-2:] == [("stage_start", "enhance:write"), ("stage_end", "enhance:write")]

    @pytest.mark.asyncio
    async def test_flat_candidate_is_still_retried_hotter(self):
        calls = []

        class _FlatThenGood:
            async def generate_with_history(self, messages, llm_id, custom_system_message=None, options_override=None, **kw):
                if custom_system_message == operations.CONCEPT_SYSTEM_MESSAGE:
                    return SimpleNamespace(content='["fox"]')
                if custom_system_message == operations.IDEATE_SYSTEM_MESSAGE:
                    return SimpleNamespace(content="1. misty dawn")
                calls.append(options_override["temperature"])
                return SimpleNamespace(content="a fox in a forest" if len(calls) == 1 else LONG_PROMPT)

        events = await run_enhance(make_collaborators(_FlatThenGood()), n_candidates=1)

        assert calls == [0.85, 1.0]
        assert events[-1]["data"]["candidates"][0]["text"] == LONG_PROMPT


class TestGatherConcurrency:
    @pytest.mark.asyncio
    async def test_approved_exemplar_search_overlaps_concept_decomposition(self):
        llm = _TimedLLM()
        prompt_db = make_prompt_db()
        original_search = prompt_db.search

        async def search(**kwargs):
            llm.events.append("approved_search" if kwargs.get("source_provider") else "community_search")
            return await original_search(**kwargs)

        prompt_db.search = search
        collaborators = make_collaborators(llm, prompt_db=prompt_db)

        await run_enhance(collaborators, n_candidates=1)

        assert llm.events.index("approved_search") < llm.events.index("concepts_end")
        assert llm.events.index("community_search") > llm.events.index("concepts_end")


class TestSiblingCancellation:
    @pytest.mark.asyncio
    async def test_a_failing_write_cancels_the_other_writes(self):
        cancelled = []

        class _OneFails:
            def __init__(self):
                self.writes = 0

            async def generate_with_history(self, messages, llm_id, custom_system_message=None, options_override=None, **kw):
                if custom_system_message == operations.CONCEPT_SYSTEM_MESSAGE:
                    return SimpleNamespace(content='["fox"]')
                if custom_system_message == operations.IDEATE_SYSTEM_MESSAGE:
                    return SimpleNamespace(content="1. misty dawn\n2. neon night")
                self.writes += 1
                if self.writes == 1:
                    raise RuntimeError("provider down")
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    cancelled.append(True)
                    raise

        with pytest.raises(RuntimeError, match="provider down"):
            await run_enhance(make_collaborators(_OneFails()), n_candidates=2)

        assert cancelled == [True]

    @pytest.mark.asyncio
    async def test_a_provider_error_reaches_the_caller_as_its_own_type(self):
        class _ProviderError(Exception):
            pass

        class _Failing:
            async def generate_with_history(self, messages, llm_id, custom_system_message=None, options_override=None, **kw):
                if custom_system_message == operations.CONCEPT_SYSTEM_MESSAGE:
                    return SimpleNamespace(content='["fox"]')
                if custom_system_message == operations.IDEATE_SYSTEM_MESSAGE:
                    return SimpleNamespace(content="1. misty dawn")
                raise _ProviderError("quota")

        with pytest.raises(_ProviderError, match="quota"):
            await operations.enhance(make_collaborators(_Failing()), "u-1", "llm-1", "a fox", n_candidates=1)
