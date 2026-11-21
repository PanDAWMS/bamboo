"""scripts/mypy_ratchet.py: each file held to its baseline count of mypy errors, with mypy itself faked."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "mypy_ratchet",
    Path(__file__).resolve().parent.parent / "scripts" / "mypy_ratchet.py",
)
assert SPEC and SPEC.loader
ratchet = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ratchet)


def errors(*files: str) -> list[str]:
    return [f"{f}:{i + 1}: error: something  [misc]" for i, f in enumerate(files)]


@pytest.fixture
def baseline(tmp_path, monkeypatch):
    path = tmp_path / "mypy_baseline.json"
    monkeypatch.setattr(ratchet, "BASELINE", path)
    monkeypatch.setattr(ratchet, "ROOT", tmp_path)
    path.write_text(
        json.dumps({"mypy": "2.4.1", "target": "bamboo/", "counts": {"bamboo/a.py": 2}})
    )
    return path


def fake_mypy(monkeypatch, found: list[str], version: str = "2.4.1") -> None:
    monkeypatch.setattr(ratchet, "run_mypy", lambda extra: (version, found))


def test_the_same_counts_pass(baseline, monkeypatch, capsys):
    fake_mypy(monkeypatch, errors("bamboo/a.py", "bamboo/a.py"))
    assert ratchet.main(["check"]) == 0
    assert "2 errors in 1 files; the baseline holds 2" in capsys.readouterr().out


def test_a_rise_fails_and_prints_the_files_errors(baseline, monkeypatch, capsys):
    fake_mypy(monkeypatch, errors("bamboo/a.py", "bamboo/a.py", "bamboo/a.py"))
    assert ratchet.main(["check"]) == 1
    out = capsys.readouterr().out
    assert "bamboo/a.py: 2 -> 3 errors" in out
    assert "    bamboo/a.py:3: error: something  [misc]" in out


def test_a_file_not_in_the_baseline_fails_on_one_error(baseline, monkeypatch, capsys):
    fake_mypy(monkeypatch, errors("bamboo/a.py", "bamboo/a.py", "bamboo/b.py"))
    assert ratchet.main(["check"]) == 1
    assert "bamboo/b.py: 0 -> 1 errors" in capsys.readouterr().out


def test_a_fall_fails_until_the_baseline_is_lowered(baseline, monkeypatch, capsys):
    fake_mypy(monkeypatch, errors("bamboo/a.py"))
    assert ratchet.main(["check"]) == 1
    assert "lower the baseline" in capsys.readouterr().out
    assert ratchet.main(["update"]) == 0
    assert json.loads(baseline.read_text())["counts"] == {"bamboo/a.py": 1}
    assert ratchet.main(["check"]) == 0


def test_update_refuses_a_rise_unless_allowed(baseline, monkeypatch):
    fake_mypy(monkeypatch, errors("bamboo/a.py", "bamboo/a.py", "bamboo/b.py"))
    assert ratchet.main(["update"]) == 1
    assert json.loads(baseline.read_text())["counts"] == {"bamboo/a.py": 2}
    assert ratchet.main(["update", "--allow-rise"]) == 0
    assert json.loads(baseline.read_text())["counts"] == {
        "bamboo/a.py": 2,
        "bamboo/b.py": 1,
    }


def test_another_mypy_version_is_said(baseline, monkeypatch, capsys):
    fake_mypy(monkeypatch, errors("bamboo/a.py", "bamboo/a.py"), version="2.5.0")
    assert ratchet.main(["check"]) == 0
    assert (
        "mypy 2.5.0 here, the baseline was taken with 2.4.1" in capsys.readouterr().out
    )


def test_mypy_args_after_the_separator_reach_mypy(baseline, monkeypatch):
    seen = []
    monkeypatch.setattr(
        ratchet,
        "run_mypy",
        lambda extra: (
            seen.append(extra),
            ("2.4.1", errors("bamboo/a.py", "bamboo/a.py")),
        )[1],
    )
    assert ratchet.main(["check", "--", "--python-executable", "/x/python"]) == 0
    assert seen == [["--python-executable", "/x/python"]]
