"""Contract and tamper controls for the v0.1 traceability audit."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "audit_v0_1_traceability.py"


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("audit_v01_traceability", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.internal_records
def test_no_execute_audit_has_zero_findings_and_exact_promoted_counts() -> None:
    result = _module().audit(ROOT, execute=False)

    assert result.findings == ()
    assert dict(result.promoted_counts) == {
        "actions": 7,
        "apis": 8,
        "components": 5,
        "features": 7,
        "stories": 6,
        "tests": 12,
    }
    assert result.executed_test_files == ()
    assert result.pytest_returncode is None


@pytest.mark.internal_records
def test_cli_no_execute_bytes_are_deterministic_and_json_shaped() -> None:
    command = [sys.executable, str(TOOL), str(ROOT), "--no-execute"]
    first = subprocess.run(command, check=False, capture_output=True, text=True)
    second = subprocess.run(command, check=False, capture_output=True, text=True)

    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    record = json.loads(first.stdout)
    assert record["findingCount"] == 0
    assert record["mode"] == "no-execute"
    assert record["schemaVersion"] == 1


@pytest.mark.internal_records
def test_executing_mode_invokes_each_verified_evidence_file_once() -> None:
    calls: list[tuple[list[str], Path]] = []

    def runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs["cwd"]))  # type: ignore[arg-type]
        return subprocess.CompletedProcess(command, 0, "", "")

    result = _module().audit(ROOT, execute=True, runner=runner)

    assert result.findings == ()
    assert result.pytest_returncode == 0
    assert len(calls) == 1
    command, cwd = calls[0]
    assert command[:4] == [sys.executable, "-m", "pytest", "-q"]
    assert tuple(command[4:]) == result.executed_test_files
    assert len(command[4:]) == len(set(command[4:]))
    assert cwd == ROOT


def test_structural_findings_prevent_evidence_execution(tmp_path: Path) -> None:
    calls: list[object] = []

    def runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess([], 0, "", "")

    result = _module().audit(tmp_path, execute=True, runner=runner)

    assert result.findings
    assert all("malformed or missing JSON" in item for item in result.findings)
    assert calls == []
    assert result.pytest_returncode is None


@pytest.mark.internal_records
def test_failed_evidence_execution_is_a_stable_finding() -> None:
    def runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        del kwargs
        return subprocess.CompletedProcess(
            command, 7, "private stdout", "private stderr"
        )

    result = _module().audit(ROOT, execute=True, runner=runner)

    assert result.findings == ("evidence pytest failed with return code 7",)
    assert result.pytest_returncode == 7
    assert "private" not in json.dumps(result.record())
