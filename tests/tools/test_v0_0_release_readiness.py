from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[2]
TOOL_PATH = HERE / "tools" / "check_v0_0_release_readiness.py"
SPEC = importlib.util.spec_from_file_location("v0_0_release_readiness", TOOL_PATH)
assert SPEC and SPEC.loader
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


@pytest.mark.internal_records
def test_current_release_has_no_static_readiness_drift() -> None:
    assert checker.validate(HERE).errors == ()


@pytest.mark.internal_records
def test_current_release_executes_the_semantic_exit_demo() -> None:
    assert checker.validate(HERE, execute_exit_demo=True).errors == ()


@pytest.mark.internal_records
def test_strict_release_accepts_the_recorded_final_gate_evidence() -> None:
    assert (
        checker.validate(HERE, execute_exit_demo=True, strict_release=True).errors == ()
    )


def test_readiness_rejects_stale_documentation_and_unverified_registry(
    tmp_path: Path,
) -> None:
    for path in ("README.md", "docs/legacy/algmul-parity.md"):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("pre-v0.0", encoding="utf-8")
    source = tmp_path / "src/anyalgebra/_version.py"
    source.parent.mkdir(parents=True)
    source.write_text('__version__ = "0.0.0.dev0"\n', encoding="utf-8")
    scope = tmp_path / ".agents/project-process/versions/v0.0/scope.json"
    scope.parent.mkdir(parents=True)
    scope.write_text(
        json.dumps({"version": "v0.0", "status": "in-progress"}), encoding="utf-8"
    )
    for name in checker.REGISTRY_PATHS:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps([{"id": "x", "status": "planned"}]), encoding="utf-8"
        )
    for name in checker.REQUIRED_DOCUMENTS:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text("release boundary", encoding="utf-8")

    errors = checker.validate(tmp_path).errors

    assert "package version is not a compatible v0.0 patch: 0.0.0.dev0" in errors
    assert "release documentation retains stale phrase: pre-v0.0" in errors
    assert "scope is not marked release-candidate or released" in errors
    assert "unverified scoped registry entry: x" in errors
    assert "missing v0.0 semantic exit demo" in errors


def test_historical_readiness_ignores_explicit_later_version_records(
    tmp_path: Path,
) -> None:
    scope = tmp_path / checker.VERSION_DIR / "scope.json"
    scope.parent.mkdir(parents=True, exist_ok=True)
    scope.write_text(
        json.dumps({"version": "v0.0", "status": "released"}),
        encoding="utf-8",
    )
    payload = [
        {"id": "old", "status": "verified"},
        {"id": "later", "version": "v0.1", "status": "in-progress"},
    ]
    for relative in checker.REGISTRY_PATHS:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload), encoding="utf-8")
    errors: list[str] = []

    checker._check_scope_and_registries(tmp_path, errors, strict_release=False)

    assert "unverified scoped registry entry: later" not in errors
