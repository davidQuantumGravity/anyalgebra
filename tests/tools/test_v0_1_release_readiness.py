from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest


pytestmark = pytest.mark.internal_records

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "check_v0_1_release_readiness.py"
SPEC = importlib.util.spec_from_file_location("v0_1_release_readiness", TOOL)
assert SPEC and SPEC.loader
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def _candidate_copy(tmp_path: Path) -> Path:
    destination = tmp_path / "candidate"
    for relative in checker.REQUIRED_FILES:
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return destination


def test_current_complete_milestone_passes_static_readiness() -> None:
    result = checker.audit(ROOT, execute=False)

    assert result.ready
    assert result.state == "complete"
    assert result.findings == ()
    assert result.commands == ()


def test_missing_gate_and_coverage_tamper_fail_closed(tmp_path: Path) -> None:
    candidate = _candidate_copy(tmp_path)
    progress_path = candidate / checker.PROGRESS
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["accepted"] = progress["accepted"][:-2]
    progress_path.write_text(json.dumps(progress), encoding="utf-8")
    (candidate / checker.COVERAGE).write_text("{}", encoding="utf-8")

    findings = checker.audit(candidate, execute=False).findings

    assert (
        "accepted tasks must be the ordered prefix V01-001 through V01-059" in findings
    )
    assert "coverage artifact SHA-256 does not match the recorded v0.1 gate" in findings


def test_overclaim_and_extra_unchecked_item_fail_closed(tmp_path: Path) -> None:
    candidate = _candidate_copy(tmp_path)
    status_path = candidate / "docs/status.md"
    status = status_path.read_text(encoding="utf-8")
    status_path.write_text(
        status.replace("No Git tag; not tagged", "Git tag: released")
        .replace("Not published; publication is not claimed", "Published to PyPI")
        .replace("Research completion | Not claimed", "Research completion | Complete"),
        encoding="utf-8",
    )
    checklist_path = candidate / checker.CHECKLIST
    checklist_path.write_text(
        checklist_path.read_text(encoding="utf-8")
        + "\n- [ ] undocumented extra gate\n",
        encoding="utf-8",
    )

    findings = checker.audit(candidate, execute=False).findings

    assert "status document does not deny a Git tag" in findings
    assert "status document does not deny publication" in findings
    assert "status document does not preserve the research boundary" in findings
    assert "complete checklist has unchecked items" in findings


def test_command_failure_is_sanitized_and_deterministic(tmp_path: Path) -> None:
    candidate = _candidate_copy(tmp_path)

    def runner(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 7, "secret output", "secret error")

    first = checker.audit(candidate, execute=True, runner=runner)
    second = checker.audit(candidate, execute=True, runner=runner)

    assert first == second
    assert not first.ready
    assert len(first.commands) == len(checker.GATES)
    assert all(record.returncode == 7 for record in first.commands)
    assert all("secret" not in finding for finding in first.findings)
    assert first.findings[0].startswith("gate failed: ")


def test_candidate_state_requires_the_exact_pending_gate(
    tmp_path: Path,
) -> None:
    candidate = _candidate_copy(tmp_path)
    progress_path = candidate / checker.PROGRESS
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["accepted"] = progress["accepted"][:-1]
    progress["activeTask"] = "V01-060"
    progress["nextTask"] = "V01-060"
    progress["status"] = "in-progress"
    progress_path.write_text(json.dumps(progress), encoding="utf-8")
    scope_path = candidate / checker.SCOPE
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    scope["status"] = "in-progress"
    scope_path.write_text(json.dumps(scope), encoding="utf-8")
    checklist_path = candidate / checker.CHECKLIST
    checklist_path.write_text(
        checklist_path.read_text(encoding="utf-8").replace(
            "- [x] All 60 sequential implementation tasks are accepted with evidence.",
            "- [ ] All 60 sequential implementation tasks are accepted with evidence.",
        ),
        encoding="utf-8",
    )
    registry_path = candidate / checker.TEST_REGISTRY
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    release_gate = next(
        item for item in registry if item["id"] == "test.v01.release.gates"
    )
    release_gate["status"] = "in-progress"
    release_gate["evidencePaths"] = []
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    notes_path = candidate / checker.RELEASE_NOTES
    notes_path.write_text(
        notes_path.read_text(encoding="utf-8").replace(
            "locally complete 0.1.0 source milestone",
            "0.1.0 local release candidate",
        ),
        encoding="utf-8",
    )

    result = checker.audit(candidate, execute=False)

    assert result.ready
    assert result.state == "candidate"
