from src.features.llm import context_budget as cb
from src.features.llm.harness.legacy import messages_from_dicts


def _messages(count):
    return messages_from_dicts([{"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 400} for i in range(count)])[0]


def _enforce(messages, *, counter, messages_counter=None, capacity=100000):
    return cb.enforce_budget(
        capacity_tokens=capacity,
        capacity_source="discovered",
        reserve_tokens=100,
        system_message="sys",
        messages=messages,
        tool_schemas=[{"a": 1}],
        counter=counter,
        messages_counter=messages_counter,
    )


class _CountingTokenizer:
    def __init__(self):
        self.calls = 0

    def __call__(self, text):
        self.calls += 1
        return len(text) // 4


def test_history_tokens_reuses_the_trim_count_instead_of_tokenizing_again():
    tokenizer = _CountingTokenizer()
    messages = _messages(11)

    outcome = _enforce(messages, counter=tokenizer)

    assert tokenizer.calls == 2 + len(messages)
    assert outcome.ledger["history_tokens"] == cb.count_messages(outcome.messages, lambda t: len(t) // 4).tokens


def test_history_tokens_is_recounted_after_an_exact_recount_dropped_units():
    tokenizer = _CountingTokenizer()
    messages = _messages(11)

    outcome = _enforce(
        messages, counter=tokenizer, messages_counter=lambda system, kept, tools: 500 + 100 * len(kept),
        capacity=1300,
    )

    assert outcome.ledger["accounting"] == "chat_template_shrunk"
    assert outcome.ledger["history_tokens"] == cb.count_messages(outcome.messages, lambda t: len(t) // 4).tokens
    assert outcome.ledger["messages_sent"] < len(messages)


def test_history_tokens_matches_the_fragment_count_when_the_exact_recount_keeps_everything():
    tokenizer = _CountingTokenizer()
    messages = _messages(11)

    outcome = _enforce(messages, counter=tokenizer, messages_counter=lambda system, kept, tools: 500)

    assert outcome.ledger["accounting"] == "chat_template"
    assert tokenizer.calls == 2 + len(messages)
    assert outcome.ledger["history_tokens"] == cb.count_messages(outcome.messages, lambda t: len(t) // 4).tokens
