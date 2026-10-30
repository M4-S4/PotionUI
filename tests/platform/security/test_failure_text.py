from src.platform.security.failure_text import GENERIC_FAILURE, basename_paths, failure_detail, scrub_paths


def test_scrub_paths_hides_absolute_paths_only():
    text = scrub_paths("missing /home/u/models/x.bin and C:\\Users\\me\\m.bin, see generations/2026/x.png")
    assert "/home/" not in text and "C:\\" not in text
    assert "generations/2026/x.png" in text


def test_regular_users_get_the_generic_message():
    assert failure_detail(RuntimeError("secret /srv/x"), False) == GENERIC_FAILURE


def test_a_caller_supplied_plain_message_replaces_the_generic_one():
    assert failure_detail(RuntimeError("secret"), False, "Could not save.") == "Could not save."


def test_admins_get_the_scrubbed_detail():
    assert failure_detail(RuntimeError("cannot open /srv/app/db"), True) == "cannot open <server path>"


def test_basename_paths_keeps_only_the_file_name_of_an_absolute_path():
    text = basename_paths(
        "cannot identify image file '/srv/potion/storage/uploads/cat.png' and C:\\Users\\me\\m.bin"
    )
    assert text == "cannot identify image file 'cat.png' and m.bin"


def test_basename_paths_leaves_relative_paths_urls_and_fractions_alone():
    text = "step 1/20 from https://example.com/org/repo and checkpoints/sdxl/x.safetensors"
    assert basename_paths(text) == text
