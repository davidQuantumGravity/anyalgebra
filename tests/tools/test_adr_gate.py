from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest


pytestmark = pytest.mark.internal_records

HERE = Path(__file__).resolve().parents[2]
GATE_PATH = HERE / ".agents" / "project-process" / "versions" / "v0.0" / "adr-gate.json"
ADR_DIRECTORY = HERE / "docs" / "architecture" / "architecture-decisions"
ADR_0002_PATH = (
    "docs/architecture/architecture-decisions/ADR-0002-equality-and-hashing.md"
)
REQUIRED_IDS = {f"ADR-{number:04d}" for number in range(1, 9)}
VALID_DECISIONS = {"accepted", "blocking"}


def load_gate() -> dict[str, Any]:
    gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
    assert isinstance(gate, dict)
    return cast(dict[str, Any], gate)


def validate_gate(gate: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    entries = gate.get("entries")
    if not isinstance(entries, list):
        return ["entries must be a list"]

    entry_ids = [
        adr_id
        for entry in entries
        if isinstance(entry, dict) and isinstance((adr_id := entry.get("adrId")), str)
    ]
    duplicate_ids = {adr_id for adr_id in entry_ids if entry_ids.count(adr_id) > 1}
    missing_ids = REQUIRED_IDS - set(entry_ids)
    extra_ids = set(entry_ids) - REQUIRED_IDS
    if missing_ids:
        errors.append(f"missing ADR IDs: {sorted(missing_ids)}")
    if duplicate_ids:
        errors.append(f"duplicate ADR IDs: {sorted(duplicate_ids)}")
    if extra_ids:
        errors.append(f"extra ADR IDs: {sorted(extra_ids)}")

    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("ADR entry must be an object")
            continue
        adr_id = entry.get("adrId")
        if not isinstance(adr_id, str):
            errors.append("ADR entry has a non-string adrId")
            continue
        decision = entry.get("gateDecision")
        if decision not in VALID_DECISIONS:
            errors.append(f"invalid gate decision for {adr_id}: {decision!r}")
        for field in (
            "path",
            "sourceStatus",
            "rationale",
            "reviewer",
            "authorityBasis",
        ):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                errors.append(f"missing {field} for {adr_id}")
        path_value = entry.get("path")
        if isinstance(path_value, str):
            source_path = HERE / path_value
            expected_name = f"{adr_id.lower()}-"
            if (
                source_path.parent != ADR_DIRECTORY
                or not source_path.name.lower().startswith(expected_name)
            ):
                errors.append(f"path mismatch for {adr_id}: {path_value}")
            elif not source_path.is_file():
                errors.append(f"missing ADR source for {adr_id}: {path_value}")
            elif (
                entry.get("sourceSha256")
                != hashlib.sha256(source_path.read_bytes()).hexdigest()
            ):
                errors.append(f"stale source hash for {adr_id}")

    decisions = [
        entry.get("gateDecision") for entry in entries if isinstance(entry, dict)
    ]
    mechanically_expected_status = (
        "blocking" if errors or "blocking" in decisions else "accepted"
    )
    aggregate = gate.get("aggregate")
    expected_rule = {
        "coverage": "exact",
        "requiredAdrIds": sorted(REQUIRED_IDS),
        "requiredGateDecision": "accepted",
        "blockingGateDecision": "blocking",
        "sourceIntegrity": "sha256-path-bound",
        "sourceStatusDeterminesDecision": False,
    }
    if not isinstance(aggregate, dict) or aggregate.get("rule") != expected_rule:
        errors.append("aggregate rule is not the required machine-readable rule")
    if (
        not isinstance(aggregate, dict)
        or aggregate.get("status") != mechanically_expected_status
    ):
        errors.append(
            "aggregate status does not match the mechanical decision and integrity rule"
        )
    return errors


def test_current_adr_gate_is_explicit_and_integrity_pinned() -> None:
    gate = load_gate()

    assert gate["schemaVersion"] == "1.0"
    assert gate["aggregate"]["status"] == "accepted"
    assert gate["aggregate"]["rule"]["sourceStatusDeterminesDecision"] is False
    assert {entry["adrId"] for entry in gate["entries"]} == REQUIRED_IDS
    assert {entry["sourceStatus"] for entry in gate["entries"]} == {"Proposed"}
    assert {entry["gateDecision"] for entry in gate["entries"]} == {"accepted"}
    assert validate_gate(gate) == []


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (lambda gate: gate["entries"].pop(), "missing ADR IDs"),
        (
            lambda gate: gate["entries"].append(copy.deepcopy(gate["entries"][0])),
            "duplicate ADR IDs",
        ),
        (
            lambda gate: gate["entries"].append(
                {
                    **copy.deepcopy(gate["entries"][0]),
                    "adrId": "ADR-9999",
                }
            ),
            "extra ADR IDs",
        ),
        (
            lambda gate: gate["entries"][0].pop("gateDecision"),
            "invalid gate decision",
        ),
        (
            lambda gate: gate["entries"][0].update({"gateDecision": "proposed"}),
            "invalid gate decision",
        ),
        (
            lambda gate: gate["entries"][0].update({"sourceSha256": "0" * 64}),
            "stale source hash",
        ),
        (
            lambda gate: gate["entries"][0].update({"path": ADR_0002_PATH}),
            "path mismatch",
        ),
    ],
)
def test_gate_rejects_invalid_coverage_decisions_and_source_integrity(
    mutation: Any, expected_error: str
) -> None:
    gate = copy.deepcopy(load_gate())

    mutation(gate)

    assert expected_error in "\n".join(validate_gate(gate))


def test_proposed_source_status_does_not_autoaccept_and_blocking_aggregates() -> None:
    gate = copy.deepcopy(load_gate())
    entry = gate["entries"][0]
    assert entry["sourceStatus"] == "Proposed"

    entry["gateDecision"] = "blocking"
    gate["aggregate"]["status"] = "blocking"

    assert validate_gate(gate) == []

    entry.pop("gateDecision")
    gate["aggregate"]["status"] = "accepted"
    errors = validate_gate(gate)
    assert "invalid gate decision for ADR-0001: None" in errors
    assert (
        "aggregate status does not match the mechanical decision and integrity rule"
        in errors
    )
