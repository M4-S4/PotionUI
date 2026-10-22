from src.features.tab_drafts import validation
from src.platform.resources import prompt_variables


def test_the_limits_the_frontend_mirrors_keep_their_values():
    assert prompt_variables.MAX_CHOICE_OPTIONS == 200
    assert prompt_variables.MAX_OPTION_CHARS == 2000
    assert prompt_variables.MAX_TEXT_VALUE_CHARS == 2000
    assert validation.MAX_OPERATIONS == 500
    assert validation.MAX_DOC_BYTES == 256 * 1024


def test_the_draft_validation_uses_the_shared_constants():
    assert validation.MAX_OPTIONS is prompt_variables.MAX_CHOICE_OPTIONS
    assert validation.MAX_OPTION_CHARS is prompt_variables.MAX_OPTION_CHARS
    assert validation.MAX_TEXT_CHARS is prompt_variables.MAX_TEXT_VALUE_CHARS
