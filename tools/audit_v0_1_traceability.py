"""Deterministically audit promoted v0.1 registries and optional test evidence."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROCESS = Path(".agents") / "project-process"
VERSION = PROCESS / "versions" / "v0.1"
REGISTRIES = {
    "features": "features.json",
    "actions": "actions.json",
    "stories": "user-stories.json",
    "tests": "test-matrix.json",
    "components": "components.json",
    "apis": "api-contracts.json",
}
PLURAL_KEYS = {
    "features": "features",
    "actions": "actions",
    "stories": "stories",
    "tests": "tests",
    "components": "components",
    "apis": "apis",
}
SCOPE_KEYS = {
    "features": "featureIds",
    "actions": "actionIds",
    "stories": "storyIds",
    "tests": "testIds",
    "components": "componentIds",
    "apis": "apiIds",
}
REFERENCE_KINDS = {value: key for key, value in SCOPE_KEYS.items()}
SYMMETRIC_LINKS = (
    ("features", "actionIds", "actions", "featureIds"),
    ("features", "storyIds", "stories", "featureIds"),
    ("features", "testIds", "tests", "featureIds"),
    ("features", "componentIds", "components", "featureIds"),
    ("features", "apiIds", "apis", "featureIds"),
    ("actions", "storyIds", "stories", "actionIds"),
    ("actions", "testIds", "tests", "actionIds"),
    ("stories", "testIds", "tests", "storyIds"),
    ("components", "testIds", "tests", "componentIds"),
    ("apis", "testIds", "tests", "apiIds"),
)


@dataclass(frozen=True, slots=True)
class AuditResult:
    findings: tuple[str, ...]
    promoted_counts: tuple[tuple[str, int], ...]
    executed_test_files: tuple[str, ...]
    pytest_returncode: int | None

    def record(self) -> dict[str, object]:
        return {
            "executedTestFiles": list(self.executed_test_files),
            "findingCount": len(self.findings),
            "findings": list(self.findings),
            "mode": "executing" if self.pytest_returncode is not None else "no-execute",
            "promotedCounts": dict(self.promoted_counts),
            "pytestReturncode": self.pytest_returncode,
            "schemaType": "anyalgebra.v0.1.traceability-audit",
            "schemaVersion": 1,
        }


def _load(path: Path, findings: list[str], label: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        findings.append(f"malformed or missing JSON: {label}")
        return None


def _entries(value: object, findings: list[str], label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        findings.append(f"registry is not a list: {label}")
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or type(item.get("id")) is not str:
            findings.append(f"invalid registry entry: {label}")
            continue
        result.append(item)
    identifiers = [item["id"] for item in result]
    if len(identifiers) != len(set(identifiers)):
        findings.append(f"duplicate registry ID: {label}")
    return result


def _mapping(entries: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(item["id"]): item for item in entries}


def _ids(value: object) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if type(item) is str}


def _check_symmetric(
    promoted: Mapping[str, Mapping[str, Mapping[str, Any]]], findings: list[str]
) -> None:
    for left_kind, left_field, right_kind, right_field in SYMMETRIC_LINKS:
        for left_id, left in promoted[left_kind].items():
            declared = _ids(left.get(left_field))
            reverse = {
                right_id
                for right_id, right in promoted[right_kind].items()
                if left_id in _ids(right.get(right_field))
            }
            if declared != reverse:
                findings.append(f"asymmetric {left_kind}.{left_field}: {left_id}")


def audit(
    repository_root: Path,
    *,
    execute: bool,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> AuditResult:
    """Return all structural findings and optionally execute verified evidence."""
    root = repository_root.resolve()
    findings: list[str] = []
    planned_raw = _load(root / VERSION / "planned-contracts.json", findings, "planned")
    scope_raw = _load(root / VERSION / "scope.json", findings, "scope")
    queue_raw = _load(
        root / VERSION / "implementation" / "implementation-tasks.json",
        findings,
        "queue",
    )
    progress_raw = _load(
        root / VERSION / "implementation" / "implementation-progress.json",
        findings,
        "progress",
    )
    if (
        not isinstance(planned_raw, dict)
        or not isinstance(scope_raw, dict)
        or not isinstance(queue_raw, dict)
        or not isinstance(progress_raw, dict)
    ):
        return AuditResult(tuple(sorted(findings)), (), (), None)
    planned = planned_raw
    scope = scope_raw
    queue = queue_raw
    progress = progress_raw

    registries: dict[str, dict[str, Mapping[str, Any]]] = {}
    planned_ids: dict[str, set[str]] = {}
    promoted: dict[str, dict[str, Mapping[str, Any]]] = {}
    for kind, filename in REGISTRIES.items():
        entries = _entries(
            _load(root / PROCESS / filename, findings, filename), findings, filename
        )
        registries[kind] = _mapping(entries)
        planned_items = planned.get(PLURAL_KEYS[kind])
        if not isinstance(planned_items, list):
            findings.append(f"planned {kind} is not a list")
            planned_items = []
        expected = {
            str(item.get("id"))
            for item in planned_items
            if isinstance(item, dict) and type(item.get("id")) is str
        }
        planned_ids[kind] = expected
        if _ids(scope.get(SCOPE_KEYS[kind])) != expected:
            findings.append(f"scope/planned mismatch: {kind}")
        missing = expected - set(registries[kind])
        if missing:
            findings.extend(f"missing promoted {kind} ID: {item}" for item in missing)
        promoted[kind] = {
            identifier: registries[kind][identifier]
            for identifier in sorted(expected & set(registries[kind]))
        }
        for identifier, entry in promoted[kind].items():
            if entry.get("version") != "v0.1":
                findings.append(f"missing v0.1 ownership: {identifier}")

    all_ids = {kind: set(values) for kind, values in registries.items()}
    for _kind, values in promoted.items():
        for identifier, entry in values.items():
            for field, target_kind in REFERENCE_KINDS.items():
                references = entry.get(field)
                if references is None:
                    continue
                if not isinstance(references, list) or any(
                    type(item) is not str for item in references
                ):
                    findings.append(f"invalid {field}: {identifier}")
                    continue
                for reference in references:
                    if reference not in all_ids[target_kind]:
                        findings.append(f"unknown {field} on {identifier}: {reference}")

    _check_symmetric(promoted, findings)

    tasks = queue.get("tasks")
    accepted = progress.get("accepted")
    task_map = (
        {str(item.get("id")): item for item in tasks if isinstance(item, dict)}
        if isinstance(tasks, list)
        else {}
    )
    accepted_ids = (
        {
            str(item.get("taskId"))
            for item in accepted
            if isinstance(item, dict) and type(item.get("taskId")) is str
        }
        if isinstance(accepted, list)
        else set()
    )
    queue_coverage: dict[str, set[str]] = {
        "features": set(),
        "apis": set(),
        "tests": set(),
    }
    for task_id, task in task_map.items():
        traceability = task.get("traceability")
        if not isinstance(traceability, dict):
            continue
        for kind, field in (
            ("features", "featureIds"),
            ("apis", "apiIds"),
            ("tests", "testIds"),
        ):
            for identifier in _ids(traceability.get(field)):
                queue_coverage[kind].add(identifier)
        if task_id in accepted_ids:
            continue
    for kind, covered in queue_coverage.items():
        if not planned_ids[kind] <= covered:
            for identifier in sorted(planned_ids[kind] - covered):
                findings.append(f"queue omits {kind} ID: {identifier}")

    evidence_paths: set[str] = set()
    for identifier, entry in promoted["tests"].items():
        status = entry.get("status")
        completion = entry.get("completionTask")
        if type(completion) is not str or completion not in task_map:
            findings.append(f"invalid completion task: {identifier}")
            continue
        expected_status = "verified" if completion in accepted_ids else "in-progress"
        if status != expected_status:
            findings.append(f"status/evidence mismatch: {identifier}")
        paths = entry.get("evidencePaths")
        if not isinstance(paths, list) or any(type(path) is not str for path in paths):
            findings.append(f"invalid evidence paths: {identifier}")
            continue
        if status == "verified" and not paths:
            findings.append(f"verified test has no evidence path: {identifier}")
        for path in paths:
            candidate = root / path
            if not candidate.is_file() or not candidate.resolve().is_relative_to(root):
                findings.append(
                    f"missing or escaping evidence path: {identifier}:{path}"
                )
            elif status == "verified":
                evidence_paths.add(path)
        task_trace = task_map[completion].get("traceability")
        if not isinstance(task_trace, dict) or identifier not in _ids(
            task_trace.get("testIds")
        ):
            findings.append(f"completion task omits test ID: {identifier}")

    for kind in ("features", "actions", "stories", "components", "apis"):
        for identifier, entry in promoted[kind].items():
            if entry.get("status") != "verified":
                findings.append(
                    f"promoted implemented ID is not verified: {identifier}"
                )
            linked_tests = _ids(entry.get("testIds"))
            if not linked_tests:
                findings.append(f"promoted ID has no test links: {identifier}")
            if any(
                promoted["tests"].get(test_id, {}).get("status") != "verified"
                for test_id in linked_tests
            ):
                findings.append(f"promoted ID relies on unverified test: {identifier}")

    executed = tuple(sorted(evidence_paths)) if execute and not findings else ()
    returncode: int | None = None
    if executed:
        completed = runner(
            [sys.executable, "-m", "pytest", "-q", *executed],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        returncode = completed.returncode
        if returncode != 0:
            findings.append(f"evidence pytest failed with return code {returncode}")
    return AuditResult(
        tuple(sorted(findings)),
        tuple((kind, len(promoted[kind])) for kind in sorted(promoted)),
        executed,
        returncode,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument("--no-execute", action="store_true")
    arguments = parser.parse_args(argv)
    result = audit(arguments.repository, execute=not arguments.no_execute)
    print(json.dumps(result.record(), sort_keys=True, separators=(",", ":")))
    return 1 if result.findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
