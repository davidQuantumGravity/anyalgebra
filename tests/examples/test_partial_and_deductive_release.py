"""V00-092 release integration for partiality and bounded deduction."""

from __future__ import annotations

import hashlib
import json
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
    SourceAnchor,
)
from anyalgebra.evidence.run import ResultReceipt, result_receipt_registry
from anyalgebra.persistence.registry import SchemaError
from anyalgebra.structures.deductive import (
    BoundExhausted,
    CompleteClosure,
    DeductiveDefinitionError,
    closure,
)
from anyalgebra.structures.operations import PartialOperation
from anyalgebra.structures.outcomes import Defined, Undefined
from anyalgebra.structures.partiality import (
    Totalization,
    TotalizationDefinitionError,
    totalize,
)
from anyalgebra.structures.terms import Term, TermDefinitionError, Variable
from examples import partial_and_deductive_release


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_partial_magma_remains_partial_after_explicit_totalization() -> None:
    fixture = partial_and_deductive_release.build_example()
    source = fixture["partial_operation"]
    totalized = fixture["totalization"]

    assert type(source) is PartialOperation
    assert type(totalized) is Totalization
    assert source.totality == "partial"
    assert source.output_carrier.items == ("a", "b", "c")
    assert source.apply("a", "b") == Defined("c")
    assert source.apply("b", "c") == Undefined(
        "table cell explicitly marked undefined", witness=(1, 2)
    )
    assert source.apply("c", "c") == Undefined(
        "table cell is not declared", witness=(2, 2)
    )


def test_totalization_has_a_new_carrier_embedding_and_strict_bottom() -> None:
    fixture = partial_and_deductive_release.build_example()
    source = fixture["partial_operation"]
    totalized = fixture["totalization"]
    bottom = fixture["bottom"]

    assert totalized.source is source
    assert totalized.totalized_carrier is not source.output_carrier
    assert totalized.totalized_carrier.items == ("a", "b", "c", bottom)
    assert totalized.embedding.source_carrier is source.output_carrier
    assert totalized.embedding.target_carrier is totalized.totalized_carrier
    assert totalized.embedding.embed("c") == "c"
    assert totalized.operation.apply("b", "c") == Defined(bottom)
    assert totalized.operation.apply(bottom, "a") == Defined(bottom)
    assert totalized.operation.apply("a", bottom) == Defined(bottom)
    assert source.output_carrier.items == ("a", "b", "c")
    assert type(source.apply("b", "c")) is Undefined


def test_bounded_closure_distinguishes_exhaustion_and_records_replayable_traces() -> (
    None
):
    fixture = partial_and_deductive_release.build_example()
    exhausted = fixture["bounded_closure"]
    completed = fixture["complete_closure"]

    assert type(exhausted) is BoundExhausted
    assert exhausted.max_rounds == 1
    assert [
        partial_and_deductive_release.term_name(item) for item in exhausted.conclusions
    ] == ["seed", "middle"]
    assert type(completed) is CompleteClosure
    assert [
        partial_and_deductive_release.term_name(item) for item in completed.conclusions
    ] == ["seed", "middle", "goal"]
    assert [trace.rule_index for trace in completed.derivations] == [0, 1]
    assert [trace.premise_indices for trace in completed.derivations] == [(0,), (1,)]
    for trace in completed.derivations:
        assert (
            tuple(completed.conclusions[index] for index in trace.premise_indices)
            == trace.premises
        )


def test_terms_preserve_parentheses_and_reject_sort_mismatches() -> None:
    fixture = partial_and_deductive_release.build_example()

    assert fixture["left_term_text"] == (
        "(product((product(x:partial_magma_element, "
        "y:partial_magma_element):partial_magma_element), "
        "z:partial_magma_element):partial_magma_element)"
    )
    assert fixture["right_term_text"] == (
        "(product(x:partial_magma_element, "
        "(product(y:partial_magma_element, "
        "z:partial_magma_element):partial_magma_element)"
        "):partial_magma_element)"
    )
    assert fixture["left_term"] != fixture["right_term"]

    wrong = Term.variable(Variable("judgment", fixture["judgment_sort"]))
    with pytest.raises(TermDefinitionError) as caught:
        Term.apply(
            fixture["partial_operation"].symbol,
            wrong,
            Term.variable(fixture["variables"][0]),
        )
    assert caught.value.index == 0
    assert caught.value.expected_sort == fixture["element_sort"]
    assert caught.value.actual_sort == fixture["judgment_sort"]


def test_unsupported_totalization_and_malformed_bounds_fail_typed() -> None:
    fixture = partial_and_deductive_release.build_example()
    source = fixture["partial_operation"]
    callable_partial = PartialOperation.from_callable(
        source.symbol,
        source.carrier_bindings,
        lambda _left, _right: "a",
    )

    with pytest.raises(TotalizationDefinitionError) as unsupported:
        totalize(callable_partial, bottom=fixture["bottom"])
    assert unsupported.value.reason == "callable partial operations cannot be totalized"
    with pytest.raises(DeductiveDefinitionError) as malformed:
        closure(
            fixture["premises"],
            fixture["deductive_system"],
            max_rounds=-1,
        )
    assert malformed.value.field == "max_rounds"


def test_public_stable_records_round_trip_and_unknown_tags_are_rejected() -> None:
    release = partial_and_deductive_release.execute_release_example()
    source = release["source_anchor"]
    manifest = release["convention_manifest"]
    contract = release["contract"]
    receipt = release["receipt"]

    assert type(source) is SourceAnchor
    assert type(manifest) is ConventionManifest
    assert type(contract) is CalculationContract
    assert type(receipt) is ResultReceipt
    assert EVIDENCE_RECORD_REGISTRY.from_record(source.to_record()) == source
    assert EVIDENCE_RECORD_REGISTRY.from_record(manifest.to_record()) == manifest
    assert contract_record_registry().from_record(contract.to_record()) == contract
    assert result_receipt_registry().from_record(receipt.to_record()) == receipt
    assert release["source_round_trip"] is True
    assert release["convention_round_trip"] is True
    assert release["contract_round_trip"] is True
    assert release["receipt_round_trip"] is True
    assert receipt.execution_status == "completed"
    assert receipt.mathematical_outcome == "constructed"
    assert receipt.evidence_tier == "E1-executed"

    malformed = source.to_record()
    malformed["schemaType"] = "example.unknown"
    with pytest.raises(SchemaError, match="unknown_schema_type"):
        EVIDENCE_RECORD_REGISTRY.from_record(malformed)


def test_release_provenance_and_bounds_are_actual_and_claim_neutral() -> None:
    release = partial_and_deductive_release.execute_release_example()
    source = release["source_anchor"]
    manifest = release["convention_manifest"]
    contract = release["contract"]
    receipt = release["receipt"]
    source_path = REPOSITORY_ROOT / source.location

    assert source.location == "examples/partial_and_deductive_release.py"
    assert (
        source.content_hash.digest
        == hashlib.sha256(source_path.read_bytes()).hexdigest()
    )
    assert source.locator == "build_example; _calculation"
    assert manifest.parenthesization == "explicit-operation-tree"
    assert dict(manifest.normalization) == {
        "deduction": "synchronous-ground-rule-rounds",
        "partiality": "marked-or-missing-cell-is-Undefined",
        "term_rendering": "fully-parenthesized-with-result-sorts",
        "totalization": "explicit-strict-bottom",
    }
    assert dict(contract.bounds) == {
        "closureMaxRounds": 2,
        "deductionRules": 2,
        "sourceCells": 9,
        "totalizedCells": 16,
    }
    assert dict(receipt.case_counts) == {
        "casesExpected": 28,
        "casesRun": 28,
        "deductionDerivations": 2,
        "sourceCells": 9,
        "totalizedCells": 16,
    }
    assert all(
        key not in manifest.to_record()
        for key in ("evidenceTier", "mathematicalOutcome", "verified")
    )


def test_release_summary_and_cli_are_deterministic_and_claim_bounded(
    tmp_path: Path,
) -> None:
    expected = {
        "bounded_conclusions": ["seed", "middle"],
        "bounded_status": "bound_exhausted",
        "canonical_convention_round_trip": True,
        "canonical_contract_round_trip": True,
        "canonical_receipt_round_trip": True,
        "canonical_source_round_trip": True,
        "complete_conclusions": ["seed", "middle", "goal"],
        "derivation_premise_indices": [[0], [1]],
        "evidence_tier": "E1-executed",
        "execution_status": "completed",
        "left_term": (
            "(product((product(x:partial_magma_element, "
            "y:partial_magma_element):partial_magma_element), "
            "z:partial_magma_element):partial_magma_element)"
        ),
        "mathematical_outcome": "constructed",
        "original_carrier_size": 3,
        "original_undefined": "table cell explicitly marked undefined",
        "strict_bottom": True,
        "totalized_carrier_size": 4,
        "totalized_value": "bottom",
    }

    assert partial_and_deductive_release.run_example() == expected
    assert partial_and_deductive_release.run_example() == expected
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "examples" / "partial_and_deductive_release.py"),
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
