from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


HERE = Path(__file__).resolve().parents[2]
TOOL_PATH = HERE / "tools" / "check_contract_snapshot.py"
SPEC = importlib.util.spec_from_file_location("check_contract_snapshot", TOOL_PATH)
assert SPEC and SPEC.loader
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


API_IDS = [f"api.sample.{number:02d}" for number in range(1, 24)]
TEST_IDS = [f"test.sample.{number:02d}" for number in range(1, 23)]
SOURCE_PATHS = (
    ".agents/project-process/process-config.json",
    ".agents/project-process/versions/v0.0/scope.json",
    ".agents/project-process/api-contracts.json",
    ".agents/project-process/test-matrix.json",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_snapshot(repository_root: Path) -> Path:
    snapshot_path = (
        repository_root
        / ".agents"
        / "project-process"
        / "versions"
        / "v0.0"
        / "contract-snapshot.json"
    )
    write_json(
        snapshot_path,
        {
            "schemaVersion": "1.0",
            "scopeVersion": "v0.0",
            "ordering": "scope-list-order",
            "counts": {"apiIds": 23, "testIds": 22},
            "apiIds": API_IDS,
            "testIds": TEST_IDS,
            "sources": [
                {"path": path, "sha256": sha256(repository_root / path)}
                for path in SOURCE_PATHS
            ],
        },
    )
    return snapshot_path


def write_valid_repo(tmp_path: Path) -> tuple[Path, Path]:
    process = tmp_path / ".agents" / "project-process"
    write_json(
        process / "process-config.json",
        {"versioning": {"activeVersion": "v0.0"}},
    )
    write_json(
        process / "versions" / "v0.0" / "scope.json",
        {"version": "v0.0", "apiIds": API_IDS, "testIds": TEST_IDS},
    )
    write_json(
        process / "api-contracts.json", [{"id": identifier} for identifier in API_IDS]
    )
    write_json(
        process / "test-matrix.json", [{"id": identifier} for identifier in TEST_IDS]
    )
    return tmp_path, write_snapshot(tmp_path)


@pytest.mark.internal_records
def test_current_snapshot_is_valid_and_checker_is_read_only() -> None:
    tracked = [
        HERE / path
        for path in (
            *SOURCE_PATHS,
            ".agents/project-process/versions/v0.0/contract-snapshot.json",
        )
    ]
    before = {path: path.read_bytes() for path in tracked}

    assert checker.validate(HERE) == []

    assert {path: path.read_bytes() for path in tracked} == before


def test_historical_snapshot_allows_active_version_to_advance(tmp_path: Path) -> None:
    repository_root, _ = write_valid_repo(tmp_path)
    config_path = (
        repository_root / ".agents" / "project-process" / "process-config.json"
    )
    write_json(config_path, {"versioning": {"activeVersion": "v0.0.1"}})

    assert checker.validate(repository_root) == []


def test_rejects_snapshot_counts_duplicates_and_reordering(tmp_path: Path) -> None:
    repository_root, snapshot_path = write_valid_repo(tmp_path)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["apiIds"][1] = snapshot["apiIds"][0]
    snapshot["testIds"] = list(reversed(snapshot["testIds"]))
    snapshot["counts"]["apiIds"] = 24
    write_json(snapshot_path, snapshot)

    errors = checker.validate(repository_root)

    assert "snapshot apiIds has duplicate IDs: ['api.sample.01']" in errors
    assert "snapshot apiIds count must be 23: 24" in errors
    assert "scope.json apiIds do not match snapshot order" in errors
    assert "scope.json testIds do not match snapshot order" in errors


def test_rejects_source_missing_extra_reordered_and_duplicate_ids(
    tmp_path: Path,
) -> None:
    repository_root, _ = write_valid_repo(tmp_path)
    scope_path = (
        repository_root
        / ".agents"
        / "project-process"
        / "versions"
        / "v0.0"
        / "scope.json"
    )
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    scope["apiIds"] = [API_IDS[1], API_IDS[0], *API_IDS[2:], "api.sample.extra"]
    scope["testIds"] = [TEST_IDS[0], TEST_IDS[0], *TEST_IDS[2:]]
    write_json(scope_path, scope)

    errors = checker.validate(repository_root)

    assert (
        "source hash mismatch: .agents/project-process/versions/v0.0/scope.json"
        in errors
    )
    assert "scope.json apiIds has extra IDs: ['api.sample.extra']" in errors
    assert "scope.json apiIds do not match snapshot order" in errors
    assert "scope.json testIds has duplicate IDs: ['test.sample.01']" in errors
    assert "scope.json testIds is missing IDs: ['test.sample.02']" in errors


def test_rejects_malformed_files_and_version_or_hash_drift(tmp_path: Path) -> None:
    repository_root, snapshot_path = write_valid_repo(tmp_path)
    snapshot_path.write_text("{", encoding="utf-8")
    assert checker.validate(repository_root) == ["malformed snapshot JSON"]

    _, snapshot_path = write_valid_repo(tmp_path)
    scope_path = (
        repository_root
        / ".agents"
        / "project-process"
        / "versions"
        / "v0.0"
        / "scope.json"
    )
    scope_path.write_text("{", encoding="utf-8")
    errors = checker.validate(repository_root)
    assert (
        "malformed source JSON: .agents/project-process/versions/v0.0/scope.json"
        in errors
    )

    _, snapshot_path = write_valid_repo(tmp_path)
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    scope["version"] = "v1.0"
    write_json(scope_path, scope)
    errors = checker.validate(repository_root)
    assert (
        "source hash mismatch: .agents/project-process/versions/v0.0/scope.json"
        in errors
    )
    assert "scope version does not match snapshot: v1.0 != v0.0" in errors
