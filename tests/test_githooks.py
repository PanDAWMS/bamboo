"""The English-only check -- ``.githooks/english-only`` and the hooks that call it.

Each test makes a small repository of its own.  Japanese is written here as
escapes, so this file holds none.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

HOOKS = Path(__file__).resolve().parents[1] / ".githooks"
CHECK = HOOKS / "english-only"
#: "Japanese", and a katakana file name
JA, JA_NAME = "\u65e5\u672c\u8a9e", "\u30e1\u30e2.txt"

pytestmark = pytest.mark.skipif(
    not (shutil.which("perl") and shutil.which("git")), reason="needs perl and git"
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True
    )


def _check(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["perl", str(CHECK), *args], cwd=repo, capture_output=True, text=True
    )


def _commit(repo: Path, message: str) -> None:
    _git(repo, "add", "-A")
    assert _git(repo, "commit", "-q", "--no-verify", "-m", message).returncode == 0


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.org")
    _git(tmp_path, "config", "user.name", "test")
    (tmp_path / "a.txt").write_text("base\n")
    _commit(tmp_path, "base")
    return tmp_path


def test_a_staged_japanese_line_is_refused_with_its_file_and_line(repo):
    (repo / "a.txt").write_text(f"base\nok\n{JA}\n")
    _git(repo, "add", "a.txt")
    r = _check(repo, "staged")
    assert r.returncode == 1 and f"a.txt:3: {JA}" in r.stderr


def test_other_non_ascii_passes(repo):
    (repo / "a.txt").write_text("base\n\u2500 a \u2014 b \u2192 c \u2026 \u2713\n")
    _git(repo, "add", "a.txt")
    assert _check(repo, "staged").returncode == 0


def test_only_added_lines_count(repo):
    (repo / "a.txt").write_text(f"base\n{JA}\n")
    _commit(repo, "old")
    (repo / "a.txt").write_text("base\n")
    _git(repo, "add", "a.txt")
    assert _check(repo, "staged").returncode == 0


def test_a_japanese_file_name_is_refused(repo):
    (repo / JA_NAME).write_text("x\n")
    _git(repo, "add", JA_NAME)
    r = _check(repo, "staged")
    assert r.returncode == 1 and f"file name: {JA_NAME}" in r.stderr


def test_a_message_is_read_without_gits_comments(repo, tmp_path):
    msg = tmp_path / "msg"
    msg.write_text(
        f"Subject\n\n# {JA}\n# ------------------------ >8 ------------------------\n{JA}\n"
    )
    assert _check(repo, "message", str(msg)).returncode == 0
    msg.write_text(f"Subject\n\nBody {JA}\n")
    r = _check(repo, "message", str(msg))
    assert r.returncode == 1 and "the commit message, line 3" in r.stderr


def test_a_range_reads_each_commits_message_and_what_it_adds(repo):
    base = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "b.txt").write_text(f"{JA}\n")
    _commit(repo, f"Add b {JA}")
    r = _check(repo, "range", base, "HEAD")
    assert r.returncode == 1
    assert ", line 1: Add b" in r.stderr and f"b.txt:1: {JA}" in r.stderr
    clean = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "c.txt").write_text("c\n")
    _commit(repo, "Add c")
    assert _check(repo, "range", clean, "HEAD").returncode == 0


def test_the_hooks_refuse_a_commit_through_core_hookspath(repo):
    _git(repo, "config", "core.hooksPath", str(HOOKS))
    (repo / "a.txt").write_text("base\nmore\n")
    _git(repo, "add", "a.txt")
    r = _git(repo, "commit", "-q", "-m", f"More {JA}")
    assert r.returncode != 0 and "English only" in r.stderr
    (repo / "a.txt").write_text(f"base\n{JA}\n")
    _git(repo, "add", "a.txt")
    r = _git(repo, "commit", "-q", "-m", "More")
    assert r.returncode != 0 and f"a.txt:2: {JA}" in r.stderr
    (repo / "a.txt").write_text("base\nmore\n")
    _git(repo, "add", "a.txt")
    assert _git(repo, "commit", "-q", "-m", "More").returncode == 0
