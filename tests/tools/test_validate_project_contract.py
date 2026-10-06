from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


HERE = Path(__file__).resolve().parents[2]
TOOL_PATH = HERE / "tools" / "validate_project_contract.py"
SPEC = importlib.util.spec_from_file_location("validate_project_contract", TOOL_PATH)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


REGISTRIES = {
    "features.json": "fe.sample",
    "actions.json": "action.sample",
    "api-contracts.json": "api.sample",
    "components.json": "component.sample",
    "user-stories.json": "us.sample",
    "test-matrix.json": "test.sample",
    "states.json": "state.sample",
    "routes.json": "route.sample",
    "events.json": "event.sample",
    "permissions.json": "permission.sample",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_valid_repo(tmp_path: Path) -> Path:
    process = tmp_path / ".agents" / "project-process"
    write_json(
        process / "process-config.json",
        {"versioning": {"activeVersion": "v0.0", "versionsPath": "versions"}},
    )
    for filename, identifier in REGISTRIES.items():
        write_json(process / filename, [{"id": identifier}])

    (process / "decisions.md").write_text("# ADR-0001\n", encoding="utf-8")
    scope = {
        "version": "v0.0",
        "featureIds": ["fe.sample"],
        "actionIds": ["action.sample"],
        "storyIds": ["us.sample"],
        "testIds": ["test.sample"],
        "componentIds": ["component.sample"],
        "routeIds": ["route.sample"],
        "apiIds": ["api.sample"],
        "stateIds": ["state.sample"],
        "eventIds": ["event.sample"],
        "decisionIds": ["ADR-0001"],
    }
    version_dir = process / "versions" / "v0.0"
    write_json(version_dir / "scope.json", scope)
    write_json(
        version_dir / "implementation" / "implementation-tasks.json",
        {
            "version": "v0.0",
            "queueId": "sample-queue",
            "tasks": [
                {
                    "id": "V00-001",
                    "work": {
                        "deliverables": ["tools/check.py"],
                        "allowedPaths": ["tools/check.py"],
                    },
                }
            ],
        },
    )
    write_json(
        version_dir / "implementation" / "implementation-progress.json",
        {"version": "v0.0", "queueId": "sample-queue"},
    )
    return tmp_path


@pytest.mark.internal_records
def test_validate_accepts_current_repository_and_does_not_write() -> None:
    tracked = [
        path
        for path in (HERE / ".agents" / "project-process").rglob("*")
        if path.is_file()
    ]
    before = {path: path.read_bytes() for path in tracked}

    assert validator.validate(HERE) == []

    assert {path: path.read_bytes() for path in tracked} == before


def test_validate_reports_missing_and_malformed_required_registries(
    tmp_path: Path,
) -> None:
    repo = write_valid_repo(tmp_path)
    (repo / ".agents" / "project-process" / "features.json").unlink()
    (repo / ".agents" / "project-process" / "actions.json").write_text(
        "{", encoding="utf-8"
    )

    assert validator.validate(repo) == [
        "missing required registry: .agents/project-process/features.json",
        "malformed JSON: .agents/project-process/actions.json",
    ]


def test_validate_reports_duplicate_and_unknown_stable_ids(tmp_path: Path) -> None:
    repo = write_valid_repo(tmp_path)
    features = repo / ".agents" / "project-process" / "features.json"
    write_json(features, [{"id": "fe.sample"}, {"id": "fe.sample"}])
    scope_path = (
        repo / ".agents" / "project-process" / "versions" / "v0.0" / "scope.json"
    )
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    scope["featureIds"] = ["fe.unknown"]
    write_json(scope_path, scope)

    assert validator.validate(repo) == [
        "duplicate id in features.json: fe.sample",
        "unknown featureIds reference in scope.json: fe.unknown",
    ]


def test_validate_rejects_unsafe_scoped_paths_and_version_mismatches(
    tmp_path: Path,
) -> None:
    repo = write_valid_repo(tmp_path)
    process = repo / ".agents" / "project-process"
    tasks_path = (
        process / "versions" / "v0.0" / "implementation" / "implementation-tasks.json"
    )
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    tasks["version"] = "v0.1"
    tasks["tasks"][0]["work"]["allowedPaths"] = [
        "../outside.py",
        r"C:\outside.py",
    ]
    write_json(tasks_path, tasks)
    progress_path = (
        process
        / "versions"
        / "v0.0"
        / "implementation"
        / "implementation-progress.json"
    )
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["queueId"] = "other-queue"
    write_json(progress_path, progress)

    assert validator.validate(repo) == [
        "implementation queue version does not match scope version: v0.1 != v0.0",
        "implementation task scoped paths differ from deliverables: V00-001",
        "unsafe scoped path in V00-001: ../outside.py",
        r"unsafe scoped path in V00-001: C:\outside.py",
        "implementation progress queueId does not match implementation queue",
    ]
