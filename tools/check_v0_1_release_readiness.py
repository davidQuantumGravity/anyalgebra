"""Fail-closed aggregate readiness gate for the bounded AnyAlgebra v0.1 scope."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys
from typing import NamedTuple, cast


PROCESS = Path(".agents/project-process")
VERSION = PROCESS / "versions/v0.1"
PROGRESS = VERSION / "implementation/implementation-progress.json"
SCOPE = VERSION / "scope.json"
CHECKLIST = VERSION / "acceptance-checklist.md"
TEST_RESULTS = VERSION / "test-results.md"
RELEASE_NOTES = VERSION / "release-notes.md"
CARRYOVER = VERSION / "carryover.md"
TEST_REGISTRY = PROCESS / "test-matrix.json"
COVERAGE = Path("build/coverage-v01-final.json")
EXPECTED_COVERAGE_SHA256 = (
    "FDBDD25FC998294B25B8DAEA685ACC5EE7215D8843F1B1367562E7BCC621E3C6"
)
RELEASE_GATE_ID = "test.v01.release.gates"
RELEASE_GATE_EVIDENCE = "tests/tools/test_v0_1_release_readiness.py"

REQUIRED_FILES = (
    Path("src/anyalgebra/_version.py"),
    Path("README.md"),
    Path("CHANGELOG.md"),
    Path("docs/README.md"),
    Path("docs/api/api-v0.1.md"),
    Path("docs/capabilities.md"),
    Path("docs/status.md"),
    Path("docs/version.md"),
    PROCESS / "process-config.json",
    TEST_REGISTRY,
    SCOPE,
    VERSION / "milestone-brief.md",
    VERSION / "planned-contracts.json",
    CHECKLIST,
    TEST_RESULTS,
    RELEASE_NOTES,
    CARRYOVER,
    PROGRESS,
    Path("docs/projects/project-registry.json"),
    COVERAGE,
)


class CommandRecord(NamedTuple):
    label: str
    returncode: int


class ReadinessResult(NamedTuple):
    state: str
    findings: tuple[str, ...]
    commands: tuple[CommandRecord, ...]

    @property
    def ready(self) -> bool:
        return not self.findings

    def record(self) -> dict[str, object]:
        return {
            "commands": [
                {"label": item.label, "returncode": item.returncode}
                for item in self.commands
            ],
            "findingCount": len(self.findings),
            "findings": list(self.findings),
            "ready": self.ready,
            "schemaType": "anyalgebra.v0.1.release-readiness",
            "schemaVersion": 1,
            "state": self.state,
        }


GATES = (
    (
        "project-contract",
        ("{python}", "tools/validate_project_contract.py", "."),
    ),
    (
        "traceability",
        ("{python}", "tools/audit_v0_1_traceability.py", "--no-execute"),
    ),
    (
        "neutral-tests",
        (
            "{python}",
            "-m",
            "pytest",
            "-q",
            "-m",
            "not legacy and not optional_backend",
        ),
    ),
    (
        "ruff-lint",
        ("{python}", "-m", "ruff", "check", "src", "tests", "examples", "tools"),
    ),
    (
        "ruff-format",
        (
            "{python}",
            "-m",
            "ruff",
            "format",
            "--check",
            "src",
            "tests",
            "examples",
            "tools",
        ),
    ),
    ("strict-mypy", ("{python}", "-m", "mypy")),
    ("distribution-artifacts", ("{python}", "tools/check_v0_1_artifacts.py")),
    ("atlas-reproduction", ("{python}", "tools/reproduce_v0_1_atlas.py")),
)


def _load_json(root: Path, relative: Path, findings: list[str]) -> object | None:
    try:
        return cast(object, json.loads((root / relative).read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        findings.append(f"missing or malformed JSON: {relative.as_posix()}")
        return None


def _read(root: Path, relative: Path, findings: list[str]) -> str:
    try:
        text = (root / relative).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        findings.append(f"missing or unreadable document: {relative.as_posix()}")
        return ""
    if not text.strip():
        findings.append(f"empty required document: {relative.as_posix()}")
    return text


def _task_ids(progress: Mapping[str, object]) -> tuple[str, ...]:
    accepted = progress.get("accepted")
    if not isinstance(accepted, list):
        return ()
    return tuple(
        item["taskId"]
        for item in accepted
        if isinstance(item, dict) and type(item.get("taskId")) is str
    )


def _state_checks(root: Path, findings: list[str]) -> str:
    progress_raw = _load_json(root, PROGRESS, findings)
    scope_raw = _load_json(root, SCOPE, findings)
    registry_raw = _load_json(root, TEST_REGISTRY, findings)
    config_raw = _load_json(root, PROCESS / "process-config.json", findings)
    if not isinstance(progress_raw, Mapping) or not isinstance(scope_raw, Mapping):
        return "invalid"
    progress = progress_raw
    scope = scope_raw
    identifiers = _task_ids(progress)
    candidate_ids = tuple(f"V01-{index:03d}" for index in range(1, 60))
    complete_ids = (*candidate_ids, "V01-060")
    if identifiers == candidate_ids:
        state = "candidate"
        if (
            progress.get("status") != "in-progress"
            or progress.get("activeTask") not in {None, "V01-060"}
            or progress.get("nextTask") != "V01-060"
            or scope.get("status") != "in-progress"
        ):
            findings.append("candidate progress and scope state do not agree")
    elif identifiers == complete_ids:
        state = "complete"
        if (
            progress.get("status") != "complete"
            or progress.get("activeTask") is not None
            or progress.get("nextTask") is not None
            or scope.get("status") != "complete"
        ):
            findings.append("complete progress and scope state do not agree")
    else:
        state = "invalid"
        findings.append(
            "accepted tasks must be the ordered prefix V01-001 through V01-059"
        )
    if scope.get("version") != "v0.1":
        findings.append("scope does not identify v0.1")
    versioning = (
        config_raw.get("versioning") if isinstance(config_raw, Mapping) else None
    )
    recorded_release = (
        versioning.get("releasedVersion", versioning.get("activeVersion"))
        if isinstance(versioning, Mapping)
        else None
    )
    if recorded_release != "v0.1":
        findings.append("project process does not identify v0.1 as released")

    release_entry: Mapping[str, object] | None = None
    if isinstance(registry_raw, list):
        release_entry = next(
            (
                item
                for item in registry_raw
                if isinstance(item, Mapping) and item.get("id") == RELEASE_GATE_ID
            ),
            None,
        )
    if release_entry is None:
        findings.append("release-gate test contract is missing")
    elif state == "candidate" and (
        release_entry.get("status") != "in-progress"
        or release_entry.get("evidencePaths") != []
    ):
        findings.append("candidate release-gate test status is inconsistent")
    elif state == "complete":
        evidence_paths = release_entry.get("evidencePaths")
        if (
            release_entry.get("status") != "verified"
            or not isinstance(evidence_paths, list)
            or RELEASE_GATE_EVIDENCE not in evidence_paths
        ):
            findings.append("complete release-gate test lacks verified evidence")
    return state


def _document_checks(root: Path, state: str, findings: list[str]) -> None:
    texts = {
        relative: _read(root, relative, findings)
        for relative in REQUIRED_FILES
        if relative.suffix == ".md"
    }
    source = _read(root, Path("src/anyalgebra/_version.py"), findings)
    # The milestone was 0.1.0; later patch releases keep the 0.1 series.
    if re.search(r'__version__ = "0\.1\.\d+"', source) is None:
        findings.append("package source version is not in the 0.1 series")
    status = texts[Path("docs/status.md")].lower()
    if "no git tag; not tagged" not in status:
        findings.append("status document does not deny a Git tag")
    if "not published; publication is not claimed" not in status:
        findings.append("status document does not deny publication")
    if "research completion | not claimed" not in status:
        findings.append("status document does not preserve the research boundary")
    release = texts[RELEASE_NOTES].lower()
    status_token = (
        "local release candidate"
        if state == "candidate"
        else "locally complete 0.1.0 source milestone"
    )
    for token in (status_token, "not tagged", "not published", "research"):
        if token not in release:
            findings.append(f"release notes omit boundary token: {token}")
    evidence_documents = (
        texts[Path("docs/status.md")],
        texts[RELEASE_NOTES],
        texts[TEST_RESULTS],
    )
    for token in (
        "2,421 passed",
        "2,392 passed",
        "20,320/21,093",
        "6,529/7,052",
        "95.3953",
        "245384be017e38895d684456272eddb5e81a2974220debf348f41e17561eb843",
        "862",
        "129",
    ):
        if any(token not in text for text in evidence_documents):
            findings.append(f"release evidence documents disagree or omit: {token}")
    checklist_lines = texts[CHECKLIST].splitlines()
    unchecked = tuple(line for line in checklist_lines if line.startswith("- [ ]"))
    expected = (
        ("- [ ] All 60 sequential implementation tasks are accepted with evidence.",)
        if state == "candidate"
        else ()
    )
    if unchecked != expected:
        if state == "candidate":
            findings.append(
                "candidate checklist must reserve exactly the V01-060 task-count gate"
            )
        else:
            findings.append("complete checklist has unchecked items")


def _coverage_check(root: Path, findings: list[str]) -> None:
    try:
        digest = hashlib.sha256((root / COVERAGE).read_bytes()).hexdigest().upper()
    except OSError:
        findings.append("coverage artifact is missing")
        return
    if digest != EXPECTED_COVERAGE_SHA256:
        findings.append(
            "coverage artifact SHA-256 does not match the recorded v0.1 gate"
        )


def audit(
    repository_root: Path | str,
    *,
    execute: bool = True,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> ReadinessResult:
    """Inspect candidate evidence and optionally execute every aggregate gate."""
    root = Path(repository_root).resolve()
    findings: list[str] = []
    for relative in REQUIRED_FILES:
        if not (root / relative).is_file():
            findings.append(f"missing required file: {relative.as_posix()}")
    state = _state_checks(root, findings)
    _document_checks(root, state, findings)
    _coverage_check(root, findings)
    records: list[CommandRecord] = []
    if execute and not findings:
        for label, template in GATES:
            command = [
                sys.executable if item == "{python}" else item for item in template
            ]
            try:
                completed = runner(
                    command,
                    cwd=root,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=1800,
                )
                returncode = completed.returncode
            except (OSError, subprocess.TimeoutExpired):
                returncode = 124
            records.append(CommandRecord(label, returncode))
            if returncode != 0:
                findings.append(f"gate failed: {label} (exit {returncode})")
    return ReadinessResult(
        state,
        tuple(sorted(set(findings))),
        tuple(records),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument("--no-execute", action="store_true")
    arguments = parser.parse_args(argv)
    result = audit(arguments.repository, execute=not arguments.no_execute)
    print(json.dumps(result.record(), sort_keys=True, separators=(",", ":")))
    return 0 if result.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
