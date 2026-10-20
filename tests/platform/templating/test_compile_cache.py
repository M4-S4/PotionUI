import pytest

from src.platform.templating import TemplateEvaluationError, TemplateProcessor
from src.platform.templating import processor as processor_module


@pytest.fixture
def processor(mock_settings):
    return TemplateProcessor(mock_settings)


def test_instances_share_one_environment(mock_settings):
    assert TemplateProcessor(mock_settings).env is TemplateProcessor(mock_settings).env


def test_repeated_expression_hits_cache_with_identical_result(processor):
    processor_module._compile_expression.cache_clear()
    context = {"form": {"steps": 8}}
    first = processor.process_template("{{ form.steps * 2 }}", context)
    second = processor.process_template("{{ form.steps * 2 }}", context)
    info = processor_module._compile_expression.cache_info()
    assert first == second == 16
    assert (info.hits, info.misses) == (1, 1)


def test_repeated_template_hits_cache_with_identical_result(processor):
    processor_module._compile_template.cache_clear()
    context = {"name": "x"}
    first = processor.process_template("hello {{ name }}\n", context)
    second = processor.process_template("hello {{ name }}\n", context)
    info = processor_module._compile_template.cache_info()
    assert first == second == "hello x\n"
    assert (info.hits, info.misses) == (1, 1)


def test_context_values_do_not_leak_between_cached_calls(processor):
    assert processor.process_template("{{ a }}", {"a": 1}) == 1
    assert processor.process_template("{{ a }}", {"a": 2}) == 2
    with pytest.raises(TemplateEvaluationError):
        processor.process_template("{{ a }}", {})


def test_speed_profile_binds_to_each_calls_context(processor):
    source = "{{ get_speed_profile('fast') }}"
    one = {"_speed_profiles": {"fast": {"steps": 4}}}
    two = {"_speed_profiles": {"fast": {"steps": 9}}}
    assert processor.process_template(source, one) == {"steps": 4}
    assert processor.process_template(source, two) == {"steps": 9}
    assert processor.process_template(source, one) == {"steps": 4}


def test_undefined_error_text_is_unchanged_through_cache(processor):
    messages = []
    for _ in range(2):
        with pytest.raises(TemplateEvaluationError) as exc:
            processor.process_template("{{ missing.value }}", {})
        messages.append(str(exc.value))
    assert messages[0] == messages[1]
    assert "{{ missing.value }}".strip("{} ") in messages[0]
    assert "missing" in messages[0]


def test_syntax_error_is_not_cached_and_keeps_expression(processor):
    for _ in range(2):
        with pytest.raises(TemplateEvaluationError) as exc:
            processor.process_template("{{ 1 + }}", {})
        assert "1 +" in str(exc.value)
