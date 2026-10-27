from __future__ import annotations

import weakref

import pytest
import torch

from src.features.llm.native.runtime import RunStats, _LoadedCheckpoint
from src.features.llm.harness.adapter import RenderRequest, Settings
from src.features.llm.harness.adapters.native import NativeAdapter
from src.features.llm.harness.messages import Message, Role, TextPart
from src.features.llm.harness.profile import CapabilityProfile
from tests.fixtures.native_runtime import NativeDriver
from src.features.llm.repository import LLMConfig
from src.platform.runtime.model_lifecycle.lifecycle import ModelLifecycle
from tests.fixtures.model_roots import make_roots


@pytest.fixture(autouse=True)
def _no_real_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None, raising=False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None, raising=False)


class _RecordingTokenizer:
    chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}"

    def __init__(self):
        self.template_calls: list = []

    def apply_chat_template(self, chat, add_generation_prompt=True, tokenize=False, **kwargs):
        self.template_calls.append(kwargs)
        return "PROMPT_TEXT"

    def __call__(self, text, return_tensors=None):
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([[1, 2, 3]])}
        return {"input_ids": [1, 2, 3]}

    def decode(self, ids, skip_special_tokens=True):
        return "the answer"


class _RecordingGenModel:

    def __init__(self):
        self.calls: list = []

    def generate(self, input_ids, **gen_kwargs):
        self.calls.append(dict(gen_kwargs))
        return torch.cat([input_ids, torch.tensor([[9, 9]])], dim=-1)


def _checkpoint():
    return _LoadedCheckpoint(
        model=_RecordingGenModel(), tokenizer=_RecordingTokenizer(),
        vision=False, model_type="qwen3", quantized=False,
    )


@pytest.fixture
def fake_model_name(tmp_path, client):
    (tmp_path / "llm" / "sampling-tiny").mkdir(parents=True)
    client._model_roots = make_roots(tmp_path, types=["llm"], home_dir=tmp_path)
    return "sampling-tiny"


@pytest.fixture
def client():
    return NativeDriver(ModelLifecycle(gpu_monitor=None, settings=None))


def _wire_fake_checkpoint(client, model_name):
    path, is_te = client._resolve_model(model_name)
    checkpoint = _checkpoint()
    client._checkpoint_refs[client._cache_key(path, is_te)] = weakref.ref(checkpoint)
    return checkpoint


def _config(model_name: str, **overrides) -> LLMConfig:
    defaults = dict(
        id="native-sampling-1",
        name="Native Sampling Test",
        type="native",
        enabled=True,
        base_url="",
        model=model_name,
        system_message="You are a test assistant.",
        temperature=0.7,
        max_tokens=4,
        timeout=30,
        supports_vision=False,
    )
    defaults.update(overrides)
    return LLMConfig(**defaults)


@pytest.mark.asyncio
async def test_min_p_unset_reaches_no_kwarg(client, fake_model_name, monkeypatch):
    checkpoint = _wire_fake_checkpoint(client, fake_model_name)
    monkeypatch.setattr(NativeDriver, "_acquire", lambda self, p, cfg, is_te=False: checkpoint)
    config = _config(fake_model_name)

    await client.generate_with_history([{"role": "user", "content": "hi"}], config, config.system_message)

    assert "min_p" not in checkpoint.model.calls[0]
    assert "repetition_penalty" not in checkpoint.model.calls[0]


@pytest.mark.asyncio
async def test_saved_min_p_and_repetition_penalty_reach_generate(client, fake_model_name, monkeypatch):
    checkpoint = _wire_fake_checkpoint(client, fake_model_name)
    monkeypatch.setattr(NativeDriver, "_acquire", lambda self, p, cfg, is_te=False: checkpoint)
    config = _config(fake_model_name, provider_options={"min_p": 0.05, "repetition_penalty": 1.15})

    await client.generate_with_history([{"role": "user", "content": "hi"}], config, config.system_message)

    assert checkpoint.model.calls[0]["min_p"] == 0.05
    assert checkpoint.model.calls[0]["repetition_penalty"] == 1.15


@pytest.mark.asyncio
async def test_per_call_override_wins_over_saved_provider_option(client, fake_model_name, monkeypatch):
    checkpoint = _wire_fake_checkpoint(client, fake_model_name)
    monkeypatch.setattr(NativeDriver, "_acquire", lambda self, p, cfg, is_te=False: checkpoint)
    config = _config(fake_model_name, provider_options={"min_p": 0.05, "repetition_penalty": 1.15})

    await client.generate_with_history(
        [{"role": "user", "content": "hi"}], config, config.system_message,
        options_override={"min_p": 0.2, "repetition_penalty": 1.3},
    )

    assert checkpoint.model.calls[0]["min_p"] == 0.2
    assert checkpoint.model.calls[0]["repetition_penalty"] == 1.3


@pytest.mark.asyncio
async def test_neutral_values_are_preserved_not_dropped(client, fake_model_name, monkeypatch):
    """min_p=0 and repetition_penalty=1 are explicit, meaningful values (no
    effect on sampling), not "unset" — they must still reach generate()."""
    checkpoint = _wire_fake_checkpoint(client, fake_model_name)
    monkeypatch.setattr(NativeDriver, "_acquire", lambda self, p, cfg, is_te=False: checkpoint)
    config = _config(fake_model_name, provider_options={"min_p": 0.0, "repetition_penalty": 1.0})

    await client.generate_with_history([{"role": "user", "content": "hi"}], config, config.system_message)

    assert checkpoint.model.calls[0]["min_p"] == 0.0
    assert checkpoint.model.calls[0]["repetition_penalty"] == 1.0


@pytest.mark.asyncio
async def test_greedy_path_never_receives_min_p(client, fake_model_name, monkeypatch):
    """temperature=0 is the greedy path (do_sample=False): min_p is a
    sampling-only knob and must not be forced in alongside it."""
    checkpoint = _wire_fake_checkpoint(client, fake_model_name)
    monkeypatch.setattr(NativeDriver, "_acquire", lambda self, p, cfg, is_te=False: checkpoint)
    config = _config(fake_model_name, temperature=0.0, provider_options={"min_p": 0.05})

    await client.generate_with_history([{"role": "user", "content": "hi"}], config, config.system_message)

    call = checkpoint.model.calls[0]
    assert call["do_sample"] is False
    assert "min_p" not in call
    assert "temperature" not in call


@pytest.mark.asyncio
async def test_greedy_path_still_applies_an_explicitly_requested_repetition_penalty(
    client, fake_model_name, monkeypatch
):
    checkpoint = _wire_fake_checkpoint(client, fake_model_name)
    monkeypatch.setattr(NativeDriver, "_acquire", lambda self, p, cfg, is_te=False: checkpoint)
    config = _config(fake_model_name, temperature=0.0, provider_options={"repetition_penalty": 1.2})

    await client.generate_with_history([{"role": "user", "content": "hi"}], config, config.system_message)

    call = checkpoint.model.calls[0]
    assert call["do_sample"] is False
    assert call["repetition_penalty"] == 1.2


@pytest.mark.asyncio
async def test_buffered_and_streamed_calls_receive_identical_sampling_kwargs(
    client, fake_model_name, monkeypatch
):
    """Both call sites build gen_kwargs through the same _generation_kwargs —
    proven here by driving both real entry points against the same fake
    model and comparing what each handed to generate()."""
    buffered_checkpoint = _wire_fake_checkpoint(client, fake_model_name)
    monkeypatch.setattr(NativeDriver, "_acquire", lambda self, p, cfg, is_te=False: buffered_checkpoint)
    config = _config(fake_model_name, provider_options={"min_p": 0.1, "repetition_penalty": 1.1})

    await client.generate_with_history([{"role": "user", "content": "hi"}], config, config.system_message)
    buffered_kwargs = buffered_checkpoint.model.calls[0]

    stream_checkpoint = _wire_fake_checkpoint(client, fake_model_name)
    monkeypatch.setattr(NativeDriver, "_acquire", lambda self, p, cfg, is_te=False: stream_checkpoint)

    async for _ in client.stream_with_history([{"role": "user", "content": "hi"}], config, config.system_message):
        pass

    # Streaming drives generation on a worker thread via TextIteratorStreamer,
    # so the recorded call is the same generate(**gen_kwargs) shape as buffered.
    assert stream_checkpoint.model.calls, "stream path never called generate()"
    stream_kwargs = stream_checkpoint.model.calls[0]
    assert stream_kwargs.get("min_p") == buffered_kwargs.get("min_p") == 0.1
    assert stream_kwargs.get("repetition_penalty") == buffered_kwargs.get("repetition_penalty") == 1.1


@pytest.mark.asyncio
async def test_adapter_settings_override_saved_options(client, fake_model_name, monkeypatch):
    checkpoint = _wire_fake_checkpoint(client, fake_model_name)
    monkeypatch.setattr(NativeDriver, "_acquire", lambda self, p, cfg, is_te=False: checkpoint)
    config = _config(fake_model_name, provider_options={"min_p": 0.05, "repetition_penalty": 1.15, "top_k": 40})
    adapter = NativeAdapter()
    adapter.runtime = client
    request = RenderRequest(
        config=config,
        system=config.system_message,
        messages=(Message(id="u1", role=Role.USER, parts=(TextPart("hi"),)),),
        settings=Settings(min_p=0.2, top_k=7, max_tokens=3),
    )
    wire = adapter.render(request, CapabilityProfile(adapter="native", model=fake_model_name))
    async for _chunk in adapter.send(wire, request):
        pass

    call = checkpoint.model.calls[0]
    assert call["min_p"] == 0.2
    assert call["top_k"] == 7
    assert call["repetition_penalty"] == 1.15
    assert call["max_new_tokens"] == 3


@pytest.mark.asyncio
async def test_adapter_with_unset_temperature_and_max_tokens_leaves_the_model_defaults(client, fake_model_name, monkeypatch):
    checkpoint = _wire_fake_checkpoint(client, fake_model_name)
    monkeypatch.setattr(NativeDriver, "_acquire", lambda self, p, cfg, is_te=False: checkpoint)
    config = _config(fake_model_name).model_copy(update={"temperature": None, "max_tokens": None})
    adapter = NativeAdapter()
    adapter.runtime = client
    request = RenderRequest(
        config=config,
        system=config.system_message,
        messages=(Message(id="u1", role=Role.USER, parts=(TextPart("hi"),)),),
    )
    wire = adapter.render(request, CapabilityProfile(adapter="native", model=fake_model_name, max_output_tokens=11))
    async for _chunk in adapter.send(wire, request):
        pass

    call = checkpoint.model.calls[0]
    assert "do_sample" not in call and "temperature" not in call
    assert call["max_new_tokens"] == 11


@pytest.mark.asyncio
async def test_stream_without_max_new_tokens_reaches_generate_with_the_default(client, fake_model_name, monkeypatch):
    checkpoint = _wire_fake_checkpoint(client, fake_model_name)
    monkeypatch.setattr(NativeDriver, "_acquire", lambda self, p, cfg, is_te=False: checkpoint)
    config = _config(fake_model_name)

    async for _text in client.stream(config, [{"role": "user", "content": "hi"}], {}, None, RunStats()):
        pass

    assert checkpoint.model.calls[0]["max_new_tokens"] == 4096
