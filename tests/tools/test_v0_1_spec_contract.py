from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

HERE = Path(__file__).resolve().parents[2]
SPEC = HERE / "docs" / "specifications" / "v0.1-census-spec.md"
AUDIT = HERE / "tests" / "fixtures" / "legacy" / "v0.1-baseline-audit.json"
PINNED_HASH = "3A13C9F47092B5B9814AA08873FF4F1DE61D4635DF68FC7F3A039A7DA5B0D00D"
ACTIVE_HASH = "2752CF4A6146E97CAE0EB093725641F9A03D9AF2410D30CBBA7721594C60F98E"


def _read_audit() -> dict[str, Any]:
    value = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def test_v01_001_baseline_audit_is_canonical_and_fail_closed() -> None:
    payload = _read_audit()

    assert (
        AUDIT.read_bytes()
        == (
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode()
    )
    assert set(payload) == {
        "algmul",
        "fullSuite",
        "observedOn",
        "resolution",
        "schemaVersion",
        "taskId",
    }
    assert payload["schemaVersion"] == 1
    assert payload["taskId"] == "V01-001"
    assert payload["resolution"]["status"] == "known-drift-preserved"
    assert payload["resolution"]["sourceOverwritten"] is False
    assert payload["resolution"]["receiptRepinned"] is False

    algmul = payload["algmul"]
    assert algmul["active"]["bytes"] == 194003
    assert algmul["active"]["sha256"] == ACTIVE_HASH
    assert algmul["pinnedReceipt"]["bytes"] == 194012
    assert algmul["pinnedReceipt"]["sha256"] == PINNED_HASH
    assert set(algmul) == {"active", "differences", "pinnedReceipt"}
    assert [item["kind"] for item in algmul["differences"]] == [
        "expression-edit",
        "notebook-metadata-edit",
    ]
    assert payload["fullSuite"] == {
        "failed": 36,
        "passed": 2025,
        "seconds": 281.34,
        "skipped": 1,
    }


def test_v01_001_spec_has_complete_bounded_contract_and_traceability() -> None:
    text = SPEC.read_text(encoding="utf-8")

    for heading in (
        "## Status and authority",
        "## Scope",
        "## Definitions",
        "## Finite-carrier versus finite-basis structures",
        "## Census specification axes",
        "## Outcomes and evidence",
        "## Determinism and resource bounds",
        "## Control examples",
        "## Roadmap acceptance mapping",
        "## Explicit non-goals",
    ):
        assert heading in text

    for axis in (
        "carrier size",
        "operation arity",
        "constraints",
        "equivalence policy",
        "ordering",
        "resource bounds",
        "convention references",
    ):
        assert axis in text.lower()

    for outcome in (
        "Complete",
        "BoundedIncomplete",
        "Unsatisfiable",
        "InvalidSpec",
        "Failed",
    ):
        assert f"`{outcome}`" in text

    for identifier in (
        "fe.v01.finite_corpus",
        "api.census.spec",
        "test.v01.corpus.contract",
        "V01-A01",
        "V01-A08",
    ):
        assert identifier in text

    assert "fingerprint equality is not proof of isomorphism" in text.lower()
    assert (
        "finite-basis controls are not members of a finite-carrier census"
        in text.lower()
    )
    assert "unbounded" in text.lower()


def test_v01_001_spec_maps_all_milestone_acceptance_signals_once() -> None:
    text = SPEC.read_text(encoding="utf-8")

    rows = [line for line in text.splitlines() if line.startswith("| `V01-A")]
    assert len(rows) == 8
    assert [row.split("|")[1].strip() for row in rows] == [
        f"`V01-A{number:02d}`" for number in range(1, 9)
    ]
    assert all("`test.v01." in row for row in rows)
