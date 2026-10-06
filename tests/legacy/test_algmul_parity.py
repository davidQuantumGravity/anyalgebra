"""V00-090 integration closure for the bounded AlgMul supersession slice.

Corrected typed behavior is asserted from definitions first.  The pinned
Mathematica receipt is checked only as secondary evidence of the legacy
surface.  A disposition closes inventory classification; ``pending`` still
means that an independent oracle is missing.
"""

from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import cast

import pytest

from anyalgebra.legacy.models import (
    DispositionRecord,
    LegacyManifestError,
    LegacyPartition,
)


_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE = _ROOT / "examples/algmul_property_mapping.py"
_GENERATED = _ROOT / "docs/legacy/generated"
_DISPOSITIONS = _GENERATED / "algmul-dispositions.json"
_SURFACE = _GENERATED / "algmul-surface-manifest.json"
_ORACLES = _GENERATED / "algmul-oracle-evidence.json"
_RUNTIME = _GENERATED / "algmul-evaluated-surface.json"
_API_REGISTRY = _ROOT / ".agents/project-process/api-contracts.json"
_TEST_REGISTRY = _ROOT / ".agents/project-process/test-matrix.json"

_PINNED_HASHES = {
    "algmul-dispositions.json": (
        "887B2FF9CF20F4D40928CCE137019E5BD3D31C5C6BE81D7F40B79F7F21B815AD"
    ),
    "algmul-oracle-evidence.json": (
        "4DE645E746FE9DB1D46E943286D50EE6533C46CEDBAFA6A3EC3A872246FCC11D"
    ),
}
_ACTIVE_PENDING = {
    "arbitrary_arrays",
    "display_report_tables",
    "generic_elements",
    "global_aliases_operators",
    "matrices",
    "product_operators",
    "registry_bases",
    "scalar_composition_products",
    "tensor_products",
}
_DEFERRED = {
    "clifford_geometric_grassmann": "v0.3",
    "conjugations_norms": "v0.2",
    "incomplete_stubs_comments": "v0.1",
    "jordan_matrices": "v0.2",
    "lie_group_experiments": "v0.4",
}


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def _records(value: object) -> list[dict[str, object]]:
    assert type(value) is list and all(type(item) is dict for item in value)
    return cast(list[dict[str, object]], value)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_corrected_example_is_definition_first_deterministic_and_executable() -> None:
    namespace = runpy.run_path(str(_EXAMPLE))
    run_example = namespace["run_example"]
    assert callable(run_example)
    payload = run_example()
    assert type(payload) is dict
    assert payload["inventory"] == {
        "factory": "MakeProperty",
        "factory_legacy_status": "absent_in_pinned_runtime_capture",
        "law_count": 8,
        "surface_count": 16,
        "intended_name_count": 128,
        "unique_name_count": 128,
        "localized_defect_names": [
            "AMTAlternative",
            "AMTCommutative",
            "AMTFlexible",
            "AMTJIdentity",
            "NMAssociative",
            "NMJacobi",
        ],
    }
    assert payload["positive"] == {
        "law": "Commutative",
        "outcome": "Proved",
        "expected_assignments": 4,
        "evaluated_assignments": 4,
        "enumeration_policy": "finite_substitution_domain_lexicographic",
    }
    assert payload["negative"] == {
        "law": "Commutative",
        "outcome": "Disproved",
        "expected_assignments": 4,
        "evaluated_assignments": 2,
        "witness_indices": [0, 1],
        "enumeration_policy": "finite_substitution_domain_lexicographic",
    }
    assert payload["boundary"] == {
        "field": "Jacobi",
        "reason": "requires explicit addition and zero operations",
    }
    assert payload["scope"] == (
        "typed finite replacement behavior only; legacy captures are secondary "
        "reproduction evidence and no scientific claim is verified"
    )
    assert run_example() == payload

    execution = subprocess.run(
        [sys.executable, str(_EXAMPLE)],
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert execution.returncode == 0, execution.stderr
    assert execution.stderr == ""
    assert json.loads(execution.stdout) == payload
    assert execution.stdout == (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )


def test_frozen_parity_ledgers_are_hash_bound_and_traceable() -> None:
    for name, expected in _PINNED_HASHES.items():
        assert _digest(_GENERATED / name) == expected

    dispositions = _object(_DISPOSITIONS)
    surface = _object(_SURFACE)
    oracles = _object(_ORACLES)
    surface_binding = cast(dict[str, object], dispositions["surfaceManifest"])
    oracle_binding = cast(dict[str, object], dispositions["oracleEvidenceRegistry"])
    assert surface_binding == {
        "path": "docs/legacy/generated/algmul-surface-manifest.json",
        "sha256": _digest(_SURFACE),
    }
    assert oracle_binding == {
        "path": "docs/legacy/generated/algmul-oracle-evidence.json",
        "sha256": _PINNED_HASHES["algmul-oracle-evidence.json"],
    }

    rules = _records(dispositions["rules"])
    assignments = _records(dispositions["assignments"])
    evidence = _records(oracles["entries"])
    inventory = cast(dict[str, object], surface["targetInventory"])
    layers = _records(inventory["layers"])
    assert len(rules) == 20
    assert len(assignments) == sum(cast(int, layer["targetCount"]) for layer in layers)
    assert len(assignments) == 3000
    assert len(evidence) == 10

    api_records = _records(json.loads(_API_REGISTRY.read_text(encoding="utf-8")))
    test_records = _records(json.loads(_TEST_REGISTRY.read_text(encoding="utf-8")))
    api_ids = {cast(str, item["id"]) for item in api_records}
    test_ids = {cast(str, item["id"]) for item in test_records}
    # The v0.0.1 registry floor is historical evidence, while later versions
    # append records to the shared registries.  Preserve uniqueness and the
    # floor without freezing all future registry growth at the old count.
    assert len(api_records) == len(api_ids)
    assert len(api_records) >= 23
    assert len(test_records) == len(test_ids)
    assert len(test_records) >= 22
    assert {
        cast(str, identifier)
        for rule in rules
        for identifier in cast(list[object], rule["apiIds"])
    } <= api_ids
    assert {
        cast(str, identifier)
        for rule in rules
        for identifier in cast(list[object], rule["testIds"])
    } <= test_ids


def test_parity_status_is_six_available_nine_active_gaps_and_five_defers() -> None:
    rules = _records(_object(_DISPOSITIONS)["rules"])
    status = Counter(
        (
            cast(str, rule["oracleStatus"]),
            cast(str, rule["disposition"]),
        )
        for rule in rules
    )
    assert (
        sum(count for (oracle, _), count in status.items() if oracle == "available")
        == 6
    )
    active_pending = {
        cast(str, rule["partition"])
        for rule in rules
        if rule["oracleStatus"] == "pending" and rule["disposition"] != "defer"
    }
    deferred = {
        cast(str, rule["partition"]): cast(str, rule["owner"])
        for rule in rules
        if rule["disposition"] == "defer"
    }
    assert active_pending == _ACTIVE_PENDING
    assert deferred == _DEFERRED
    assert len(active_pending) == 9
    assert len(deferred) == 5
    assert all(
        rule["oracleIds"] == []
        for rule in rules
        if cast(str, rule["partition"]) in active_pending | set(deferred)
    )
    assert all(
        rule["owner"] == "v0.0"
        for rule in rules
        if cast(str, rule["partition"]) in active_pending
    )


def test_legacy_receipt_is_secondary_absence_and_reproduction_evidence() -> None:
    receipt = _object(_RUNTIME)
    assert receipt["sourceSHA256"] == (
        "3A13C9F47092B5B9814AA08873FF4F1DE61D4635DF68FC7F3A039A7DA5B0D00D"
    )
    assert receipt["kernelVersion"] == (
        "12.0.0 for Microsoft Windows (64-bit) (April 6, 2019)"
    )
    assert receipt["loadStatus"] == "completed-with-missing-required-symbols"
    assert receipt["expectedPropertySymbolCount"] == 128
    assert receipt["propertySymbolsPresentCount"] == 0
    assert receipt["propertySymbolsDefinedCount"] == 0
    assert set(cast(dict[str, object], receipt["compositionTables"])) == {
        "C",
        "Cs",
        "H",
        "Hs",
        "O",
        "Os",
    }
    assert receipt["missingRequiredSymbols"] == ["AlgMul`MakeProperty"]


def test_status_boundary_rejects_false_parity_promotion() -> None:
    rules = _records(_object(_DISPOSITIONS)["rules"])
    target = next(
        rule for rule in rules if rule["partition"] == "scalar_composition_products"
    )
    assert "scalar_composition_products" in _ACTIVE_PENDING
    with pytest.raises(
        LegacyManifestError,
        match="available oracle status requires at least one evidence link",
    ):
        DispositionRecord.create(
            "symbol:CMul",
            LegacyPartition.SCALAR_COMPOSITION_PRODUCTS,
            target["disposition"],
            owner=target["owner"],
            note=target["note"],
            api_ids=tuple(cast(list[object], target["apiIds"])),
            test_ids=tuple(cast(list[object], target["testIds"])),
            oracle_ids=(),
            oracle_status="available",
        )
