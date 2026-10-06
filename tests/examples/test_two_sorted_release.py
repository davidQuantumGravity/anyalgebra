"""V00-091 executable release slice for arbitrary two-sorted structures."""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from anyalgebra.evidence.contracts import (
    CalculationContract,
    contract_record_registry,
)
from anyalgebra.evidence.models import (
    EVIDENCE_RECORD_REGISTRY,
    ConventionManifest,
    EvidenceModelError,
    SourceAnchor,
)
from anyalgebra.evidence.replay import ReplayReport
from anyalgebra.evidence.run import ResultReceipt, result_receipt_registry
from anyalgebra.structures.operations import PartialOperation
from anyalgebra.structures.outcomes import Defined, Undefined
from anyalgebra.structures.relations import Relation, RelationResult
from anyalgebra.structures.structure import Structure
from anyalgebra.structures.terms import Term, TermDefinitionError
from examples import two_sorted_release


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_release_fixture_uses_generic_two_sorted_structure_apis() -> None:
    fixture = two_sorted_release.build_example()

    assert type(fixture["structure"]) is Structure
    assert type(fixture["operation"]) is PartialOperation
    assert type(fixture["relation"]) is Relation
    assert fixture["operation"].symbol.arity == 3
    assert fixture["operation"].totality == "partial"
    assert tuple(sort.name for sort in fixture["structure"].signature.sorts) == (
        "point",
        "color",
    )
    assert type(fixture["defined_evaluation"].outcome) is Defined
    assert fixture["defined_evaluation"].outcome.value == "left"
    assert fixture["term_text"] == (
        "(select(first:point, shade:color, "
        "(select(second:point, shade:color, first:point):point)):point)"
    )


def test_release_fixture_preserves_negative_and_partial_results() -> None:
    fixture = two_sorted_release.build_example()
    operation = fixture["operation"]
    undefined = fixture["undefined_evaluation"].outcome

    assert type(undefined) is Undefined
    assert undefined.reason == "table cell explicitly marked undefined"
    assert undefined.witness == (0, 0, 0)
    assert len(operation.output_carrier) == 2
    assert "bottom" not in operation.output_carrier
    assert fixture["relation"].apply("right", "blue") == RelationResult(True)
    assert fixture["relation"].apply("left", "blue") == RelationResult(False)


def test_release_fixture_rejects_a_sort_mismatch_before_evaluation() -> None:
    fixture = two_sorted_release.build_example()
    point_term = Term.variable(fixture["variables"][0])
    color_term = Term.variable(fixture["variables"][1])

    with pytest.raises(TermDefinitionError) as caught:
        Term.apply(
            fixture["operation"].symbol,
            color_term,
            color_term,
            point_term,
        )

    assert caught.value.index == 0
    assert caught.value.expected_sort == fixture["sorts"][0]
    assert caught.value.actual_sort == fixture["sorts"][1]
    with pytest.raises(ValueError, match="variant"):
        two_sorted_release.build_example(variant="unknown")  # type: ignore[arg-type]


def test_release_pipeline_serializes_receipt_replays_and_detects_staleness() -> None:
    release = two_sorted_release.execute_release_example()
    contract = release["contract"]
    receipt = release["receipt"]
    source = release["source_anchor"]
    manifest = release["convention_manifest"]

    assert type(contract) is CalculationContract
    assert type(receipt) is ResultReceipt
    assert type(source) is SourceAnchor
    assert type(manifest) is ConventionManifest
    assert EVIDENCE_RECORD_REGISTRY.from_record(source.to_record()) == source
    assert EVIDENCE_RECORD_REGISTRY.from_record(manifest.to_record()) == manifest
    assert contract_record_registry().from_record(contract.to_record()) == contract
    assert result_receipt_registry().from_record(receipt.to_record()) == receipt
    assert release["source_round_trip"] is True
    assert release["convention_round_trip"] is True
    assert release["contract_round_trip"] is True
    assert release["receipt_round_trip"] is True
    assert contract.source_anchors == (source.semantic_hash,)
    assert contract.convention_manifests == (manifest.semantic_hash,)
    assert receipt.source_anchors == contract.source_anchors
    assert receipt.convention_manifests == contract.convention_manifests
    assert manifest.sources == (source.semantic_hash,)
    assert type(release["exact_replay"]) is ReplayReport
    assert release["exact_replay"].status == "exact_match"
    assert release["stale_replay"].status == "stale"
    assert release["stale_replay"].stale_edges == ("contract.inputs",)
    assert receipt.execution_status == "completed"
    assert receipt.mathematical_outcome == "constructed"
    assert receipt.evidence_tier == "E1-executed"


def test_release_provenance_is_actual_bounded_and_claim_neutral() -> None:
    release = two_sorted_release.execute_release_example()
    source = release["source_anchor"]
    manifest = release["convention_manifest"]
    source_path = REPOSITORY_ROOT / source.location
    expected_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()

    assert source.location == "examples/two_sorted_release.py"
    assert source.content_hash.digest == expected_digest
    assert source.locator == "build_example; _calculation"
    assert manifest.basis_order == (
        "point:left",
        "point:right",
        "color:red",
        "color:blue",
    )
    assert dict(manifest.normalization) == {
        "evaluation_order": "children-left-to-right-before-parent",
        "partiality": "marked-or-missing-cell-is-Undefined",
        "relation_truth": "declared-tuples-true-otherwise-false",
        "sort_order": "point-before-color",
    }
    assert manifest.indexing == "zero-based-carrier-index-order"
    assert manifest.parenthesization == "explicit-operation-tree"
    assert manifest.status == "accepted"
    assert all(
        key not in manifest.to_record()
        for key in ("evidenceTier", "mathematicalOutcome", "verified")
    )
    with pytest.raises(EvidenceModelError, match="must not be empty"):
        ConventionManifest.create(
            "example.invalid",
            ("point:left",),
            coefficient_domain="finite-carriers",
            multiplication_fixture=manifest.multiplication_fixture,
            sources=(),
            status="accepted",
            signs=(),
            involutions=(),
            normalization=(),
            indexing="zero-based",
            parenthesization="explicit",
            conversions=(),
        )


def test_changed_fixture_contract_and_result_are_semantically_consistent() -> None:
    release = two_sorted_release.execute_release_example()
    base = release["fixture"]
    changed = release["changed_fixture"]
    base_outcome = base["defined_evaluation"].outcome
    changed_outcome = changed["defined_evaluation"].outcome

    assert type(base_outcome) is Defined
    assert type(changed_outcome) is Defined
    assert base_outcome.value == "left"
    assert changed_outcome.value == "right"
    assert base["operation"].table != changed["operation"].table
    assert dict(base["operation"].table)[(1, 1, 0)] == "left"
    assert dict(changed["operation"].table)[(1, 1, 0)] == "right"
    assert release["fixture_hash"] != release["changed_fixture_hash"]
    assert release["receipt"].input_hashes == (("structure", release["fixture_hash"]),)
    assert release["changed_receipt"].input_hashes == (
        ("structure", release["changed_fixture_hash"]),
    )
    assert release["receipt"].artifacts != release["changed_receipt"].artifacts
    assert (
        release["receipt"].primary_result_hash
        != release["changed_receipt"].primary_result_hash
    )
    assert (
        release["changed_receipt"].convention_manifests
        == release["receipt"].convention_manifests
    )


def test_release_summary_and_cli_are_deterministic_and_claim_bounded(
    tmp_path: Path,
) -> None:
    expected = {
        "canonical_convention_round_trip": True,
        "canonical_contract_round_trip": True,
        "canonical_receipt_round_trip": True,
        "canonical_source_round_trip": True,
        "changed_input_detected": True,
        "changed_result_detected": True,
        "convention_manifest_count": 1,
        "defined_value": "left",
        "evidence_tier": "E1-executed",
        "exact_replay": "exact_match",
        "execution_status": "completed",
        "mathematical_outcome": "constructed",
        "operation_totality": "partial",
        "relation_false": False,
        "relation_true": True,
        "source_anchor_count": 1,
        "stale_edges": ["contract.inputs"],
        "stale_replay": "stale",
        "term": (
            "(select(first:point, shade:color, "
            "(select(second:point, shade:color, first:point):point)):point)"
        ),
        "undefined_reason": "table cell explicitly marked undefined",
    }

    assert two_sorted_release.run_example() == expected
    assert two_sorted_release.run_example() == expected
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "examples" / "two_sorted_release.py"),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,  # a hang guard, not a performance budget
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == expected
    assert all(
        phrase not in completed.stdout.lower()
        for phrase in ("proved", "confirmed", "ruled out")
    )
