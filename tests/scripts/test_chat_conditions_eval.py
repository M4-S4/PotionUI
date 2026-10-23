import sys
from pathlib import Path

import pytest

_HARNESS_DIR = Path(__file__).resolve().parents[2] / "tests" / "e2e" / "harness"
if str(_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(_HARNESS_DIR))

import chat_conditions_eval as evaluation

CASES = {case.id: case for case in evaluation.CASES}


def test_the_five_cases_cover_the_card():
    assert set(CASES) == {
        "choice_if_else_if_else",
        "number_comparison",
        "edit_existing_condition",
        "explain_existing_condition",
        "value_with_special_characters",
    }


def test_self_check_passes_every_good_reply_and_fails_every_bad_one(capsys):
    assert evaluation.self_check() == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 5 and all(line.startswith("PASS") for line in lines)


def test_a_reply_without_a_proposal_fails_a_writing_case():
    passed, detail = evaluation.score(CASES["number_comparison"], "I added the condition.")
    assert not passed and "no update_segment/add_segment proposal" in detail


def test_branch_order_matters_and_any_one_correct_proposal_passes():
    case = CASES["edit_existing_condition"]
    branches = ("<<if $age > 70>>grey hair", "<<else if $age > 25>>auburn hair", "<<else if $age < 25>>freckles<<end>>")
    right = evaluation._tag("portrait of a woman, " + "".join(branches))
    wrong = evaluation._tag(
        "portrait of a woman, " + branches[1].replace("<<else if", "<<if") + branches[0].replace("<<if", "<<else if") + branches[2]
    )
    assert evaluation.score(case, right)[0] is True
    passed, detail = evaluation.score(case, wrong)
    assert passed is False and "with {'age': '72'}" in detail
    assert evaluation.score(case, wrong + "\n" + case.good)[0] is True


def test_explaining_needs_the_values_and_at_least_two_branch_texts():
    case = CASES["explain_existing_condition"]
    assert evaluation.score(case, "It depends on noir, 60, sunlit and foggy dawn.")[0] is False
    assert evaluation.score(case, case.good)[0] is True


def test_scratch_paths_live_under_the_scratch_dir_and_are_pinned(tmp_path, monkeypatch):
    paths = evaluation.scratch_paths(tmp_path)
    assert set(paths) == {
        "POTIONUI_DB_PATH", "POTIONUI_STORAGE_PATH", "POTIONUI_MODELS_DIR", "POTIONUI_RECIPES_DIR", "POTIONUI_LOG_DIR",
    }
    for name, path in paths.items():
        monkeypatch.setenv(name, str(path))
    evaluation.assert_scratch(paths, tmp_path)

    monkeypatch.setenv("POTIONUI_DB_PATH", str(evaluation.REPO_ROOT / "storage" / "db.sqlite"))
    with pytest.raises(RuntimeError, match="not pinned"):
        evaluation.assert_scratch(paths, tmp_path)


@pytest.mark.parametrize("scratch", [evaluation.REPO_ROOT / "storage", evaluation.REPO_ROOT, evaluation.REPO_ROOT.parent])
def test_a_scratch_dir_overlapping_the_repository_is_refused(monkeypatch, scratch):
    paths = evaluation.scratch_paths(scratch)
    for name, path in paths.items():
        monkeypatch.setenv(name, str(path))
    with pytest.raises(RuntimeError, match="overlaps the repository"):
        evaluation.assert_scratch(paths, scratch)


def test_a_path_outside_the_scratch_dir_is_refused(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    paths = {**evaluation.scratch_paths(scratch), "POTIONUI_LOG_DIR": tmp_path / "elsewhere" / "logs"}
    for name, path in paths.items():
        monkeypatch.setenv(name, str(path))
    with pytest.raises(RuntimeError, match="POTIONUI_LOG_DIR=.* is outside the scratch directory"):
        evaluation.assert_scratch(paths, scratch)


def test_a_refused_run_removes_its_temp_dir_and_boots_nothing(tmp_path, monkeypatch):
    for name in evaluation.scratch_paths(tmp_path):
        monkeypatch.setenv(name, "unchanged")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(evaluation.tempfile, "mkdtemp", lambda prefix: str(scratch))
    monkeypatch.setattr(evaluation, "scratch_paths", lambda root: {"POTIONUI_DB_PATH": tmp_path / "elsewhere" / "db.sqlite"})
    monkeypatch.setitem(sys.modules, "e2e_harness", object())
    args = evaluation.build_parser().parse_args(["--model", "m"])
    with pytest.raises(RuntimeError, match="is outside the scratch directory"):
        evaluation.live(args)
    assert not scratch.exists()


def test_dry_run_and_self_check_never_boot_a_backend(monkeypatch, capsys):
    monkeypatch.setattr(evaluation, "live", lambda args: pytest.fail("booted a backend"))
    assert evaluation.main(["--dry-run", "--model", "m"]) == 0
    assert "Would boot a throwaway backend" in capsys.readouterr().out
    assert evaluation.main(["--self-check"]) == 0
    with pytest.raises(SystemExit):
        evaluation.main([])


def test_the_eval_config_never_reflects_into_memory():
    args = evaluation.build_parser().parse_args(["--model", "m"])
    config = evaluation._llm_config(args)
    assert config["memory_reflection"] is False
    assert config["api_key"] is None and config["model"] == "m"
