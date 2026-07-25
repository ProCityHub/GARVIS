"""Tests for `.github/scripts/guard_check.py`.

The guard reads real git history, so these tests build real repositories
rather than mocking git. Each case creates a `main` branch, branches off it,
commits a change, and runs the guard exactly as CI does.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD = REPO_ROOT / ".github" / "scripts" / "guard_check.py"

pytestmark = pytest.mark.skipif(
    not GUARD.is_file() or shutil.which("git") is None,
    reason="guard script or git unavailable",
)

BASE_GOVERNED = """# governed
src/garvis/thanos_mode.py
src/garvis/stage_gate.py
.github/
"""

BASE_FROZEN = """# frozen
results/run_001.json
"""


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def _write(repo: Path, path: str, text: str) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "adrien@example.invalid")
    _git(repo, "config", "user.name", "Adrien D. Thomas")

    _write(repo, "GOVERNED_FILES.txt", BASE_GOVERNED)
    _write(repo, "FROZEN_FILES.txt", BASE_FROZEN)
    _write(repo, "src/garvis/thanos_mode.py", "OWNER = 'Adrien D. Thomas'\n")
    _write(repo, "src/garvis/assistant.py", "VALUE = 1\n")
    _write(repo, "results/run_001.json", '{"outcome": "NOT_SUPPORTED"}\n')
    _write(repo, "docs/notes.md", "# Notes\n")
    _write(repo, "claims/CLAIMS.json", "{}\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def commit_change(repo: Path, files: dict, message: str = "change") -> None:
    if _git(repo, "branch", "--show-current").strip() == "main":
        _git(repo, "checkout", "-q", "-b", "feature")
    for path, text in files.items():
        _write(repo, path, text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)


def run_guard(repo: Path):
    # Inherit the real environment. Hardcoding PATH breaks on Termux, where
    # git lives under /data/data/com.termux/files/usr/bin rather than /usr/bin.
    env = dict(os.environ)
    env["GITHUB_BASE_REF"] = "main"
    return subprocess.run(
        [sys.executable, str(GUARD)],
        cwd=str(repo),
        text=True,
        capture_output=True,
        env=env,
    )


# --------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------


def test_ordinary_change_passes(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    commit_change(repo, {"src/garvis/assistant.py": "VALUE = 2\n"})
    result = run_guard(repo)
    assert "GUARD PASS" in result.stdout
    assert result.returncode == 0


def test_new_ordinary_file_passes(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    commit_change(repo, {"src/garvis/repair_engine.py": "def repair():\n    pass\n"})
    assert run_guard(repo).returncode == 0


# --------------------------------------------------------------------------
# Governed source
# --------------------------------------------------------------------------


def test_governance_change_fails_without_exception(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    commit_change(repo, {"src/garvis/thanos_mode.py": "OWNER = 'someone else'\n"})
    result = run_guard(repo)
    assert result.returncode == 1
    assert "GUARD FAILED" in result.stdout
    assert "thanos_mode.py" in result.stdout
    assert "AUDIT-EXCEPTION" in result.stdout


def test_governance_change_passes_with_declared_exception(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    commit_change(
        repo,
        {"src/garvis/thanos_mode.py": "OWNER = 'Adrien D. Thomas'  # tuned\n"},
        message="chore: adjust\n\nAUDIT-EXCEPTION: reviewed governance tweak",
    )
    result = run_guard(repo)
    assert result.returncode == 0
    assert "GUARD NOTICE" in result.stdout
    assert "code-owner review" in result.stdout.lower()


def test_directory_ledger_entry_covers_subtree(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    commit_change(repo, {".github/workflows/tests.yml": "name: Tests\n"})
    result = run_guard(repo)
    assert result.returncode == 1
    assert ".github/workflows/tests.yml" in result.stdout


# --------------------------------------------------------------------------
# The closure property: a PR cannot unfreeze itself
# --------------------------------------------------------------------------


def test_pr_cannot_ungovern_a_file_it_also_edits(tmp_path: Path) -> None:
    """The load-bearing test.

    A cycle removes thanos_mode.py from the governed ledger AND edits it in
    the same pull request. The ledger is read from the base branch, so the
    removal has no effect on this pull request.
    """

    repo = make_repo(tmp_path)
    commit_change(
        repo,
        {
            "GOVERNED_FILES.txt": "# governed\nsrc/garvis/stage_gate.py\n.github/\n",
            "src/garvis/thanos_mode.py": "OWNER = 'self-approved'\n",
        },
    )
    result = run_guard(repo)
    assert result.returncode == 1
    assert "thanos_mode.py" in result.stdout
    assert "does not unfreeze" in result.stdout or "audit exception" in result.stdout


def test_pr_cannot_empty_the_governed_ledger(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    commit_change(
        repo,
        {
            "GOVERNED_FILES.txt": "# emptied\n",
            "src/garvis/stage_gate.py": "BYPASS = True\n",
        },
    )
    assert run_guard(repo).returncode == 1


# --------------------------------------------------------------------------
# Frozen artifacts
# --------------------------------------------------------------------------


def test_frozen_artifact_change_fails(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    commit_change(repo, {"results/run_001.json": '{"outcome": "SUPPORTED"}\n'})
    result = run_guard(repo)
    assert result.returncode == 1
    assert "Frozen evidence artifacts changed" in result.stdout


def test_frozen_artifact_has_no_audit_override(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    commit_change(
        repo,
        {"results/run_001.json": '{"outcome": "SUPPORTED"}\n'},
        message="fix\n\nAUDIT-EXCEPTION: I would like this result to be different",
    )
    assert run_guard(repo).returncode == 1


# --------------------------------------------------------------------------
# Claim boundary: markers must come from a run
# --------------------------------------------------------------------------


def test_completion_marker_asserted_in_document_fails(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    commit_change(
        repo,
        {"docs/status.md": "# Status\n\nFULL_TEST_SUITE=PASS\nROLLBACK=PASS\n"},
    )
    result = run_guard(repo)
    assert result.returncode == 1
    assert "asserted inside a" in result.stdout
    assert "FULL_TEST_SUITE" in result.stdout


def test_marker_in_bullet_list_is_still_caught(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    commit_change(repo, {"docs/status.md": "# Status\n\n- HEARTBEAT=PASS\n"})
    assert run_guard(repo).returncode == 1


def test_honest_not_implemented_marker_passes(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    commit_change(
        repo,
        {"docs/status.md": "# Status\n\nROLLBACK=NOT_IMPLEMENTED\nHEARTBEAT=FAIL\n"},
    )
    assert run_guard(repo).returncode == 0


def test_marker_in_source_code_is_not_a_document_assertion(tmp_path: Path) -> None:
    """render_status builds these strings; that is code, not a claim."""

    repo = make_repo(tmp_path)
    commit_change(
        repo,
        {"src/garvis/render.py": 'LINE = "FULL_TEST_SUITE=PASS"\n'},
    )
    assert run_guard(repo).returncode == 0


def test_allowlisted_documentation_may_describe_the_format(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    commit_change(
        repo,
        {"RETRACTIONS.md": "# Retractions\n\nFULL_TEST_SUITE=PASS is a run marker.\n"},
    )
    assert run_guard(repo).returncode == 0


# --------------------------------------------------------------------------
# Vocabulary discipline
# --------------------------------------------------------------------------


def test_empirical_verdict_in_claims_document_fails(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    commit_change(repo, {"claims/CLAIMS.json": '{"lattice": "SUPPORTED"}\n'})
    result = run_guard(repo)
    assert result.returncode == 1
    assert "reserved empirical verdict" in result.stdout


def test_empirical_verdict_in_ordinary_document_passes(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    commit_change(
        repo,
        {"docs/notes.md": "# Notes\n\nWe use SUPPORTED only for preregistered runs.\n"},
    )
    assert run_guard(repo).returncode == 0


# --------------------------------------------------------------------------
# Retracted formula
# --------------------------------------------------------------------------


# The retracted scalar formula is assembled from fragments at runtime.
# No source line here contains the pattern, so the guard's own test data
# cannot trip the guard -- and no allowlist exception is needed, which
# means nobody can smuggle the retracted formula in via a test file.
_PHI = "phi"
_MUL = " * "

SCALAR_PHI_CASES = (
    "score = " + _MUL.join((_PHI, "o", "a", "b")),
    "C = " + _MUL.join((_PHI.upper(), "observer", "actor", "bridge")),
    "value = " + _MUL.join(("observer", "actor", "environment", _PHI)),
)


@pytest.mark.parametrize("line", SCALAR_PHI_CASES)
def test_retracted_scalar_phi_formula_is_blocked(tmp_path: Path, line: str) -> None:
    repo = make_repo(tmp_path)
    commit_change(repo, {"docs/notes.md": f"# Notes\n\n{line}\n"})
    result = run_guard(repo)
    assert result.returncode == 1
    assert "scalar-phi" in result.stdout


def test_canonical_exponent_formula_is_allowed(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    commit_change(
        repo,
        {"docs/notes.md": "# Notes\n\nC = O^1 . A^(1/phi) . B^(1/phi^2)\n"},
    )
    assert run_guard(repo).returncode == 0
