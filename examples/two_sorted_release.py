"""One bounded two-sorted infrastructure calculation with durable evidence.

The example uses only the generic public structure APIs: no user-defined
``Structure`` subclass or specialized algebra family is needed.  Its receipt
records an E1 execution of the fixture.  It does not promote a mathematical or
scientific claim.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Literal, TypedDict, cast

from anyalgebra.core.parents import FiniteCarrier, SemanticHash, Sort
from anyalgebra.evidence.contracts import (
    CalculationContract,
    contract_record_registry,
)
from anyalgebra.evidence.models import (
    EVIDENCE_RECORD_REGISTRY,
    ConventionManifest,
    SourceAnchor,
)
from anyalgebra.evidence.replay import ReplayReport, ReplayResolver, verify_replay
from anyalgebra.evidence.run import (
    CalculationResult,
    ExecutionEnvironment,
    ResultReceipt,
    result_receipt_registry,
    run_calculation,
)
from anyalgebra.structures.evaluate import EvaluationResult, evaluate_term
from anyalgebra.structures.operations import PartialOperation
from anyalgebra.structures.outcomes import Defined, Undefined
from anyalgebra.structures.relations import Relation, RelationResult
from anyalgebra.structures.signatures import (
    OperationSymbol,
    RelationSymbol,
    Signature,
)
from anyalgebra.structures.structure import Structure, StructureBuilder
from anyalgebra.structures.terms import Term, Variable


class TwoSortedFixture(TypedDict):
    """Generic structure objects and the checked terms used by the run."""

    defined_evaluation: EvaluationResult
    defined_environment: Mapping[Variable, object]
    defined_term: Term
    operation: PartialOperation
    relation: Relation
    sorts: tuple[Sort, Sort]
    structure: Structure
    term_text: str
    undefined_evaluation: EvaluationResult
    undefined_environment: Mapping[Variable, object]
    undefined_term: Term
    variables: tuple[Variable, Variable, Variable]
    variant: Literal["base", "changed"]


class ReleaseExecution(TypedDict):
    """Evidence artifacts retained by the executable release slice."""

    changed_fixture: TwoSortedFixture
    changed_fixture_hash: SemanticHash
    changed_receipt: ResultReceipt
    convention_manifest: ConventionManifest
    convention_round_trip: bool
    contract: CalculationContract
    contract_round_trip: bool
    exact_replay: ReplayReport
    fixture: TwoSortedFixture
    fixture_hash: SemanticHash
    receipt: ResultReceipt
    receipt_round_trip: bool
    source_anchor: SourceAnchor
    source_round_trip: bool
    stale_replay: ReplayReport


def _hash_bytes(value: bytes) -> SemanticHash:
    """Return one exact content hash for deterministic fixture material."""
    return SemanticHash("sha256", hashlib.sha256(value).hexdigest())


def _render_term(term: Term) -> str:
    """Render every application with its result sort and explicit parentheses."""
    if term.variable_value is not None:
        return f"{term.variable_value.name}:{term.sort.name}"
    if term.symbol is None:
        raise RuntimeError("term node has neither a variable nor an operation")
    arguments = ", ".join(_render_term(argument) for argument in term.arguments)
    return f"({term.symbol.name}({arguments}):{term.sort.name})"


def build_example(*, variant: Literal["base", "changed"] = "base") -> TwoSortedFixture:
    """Construct a finite two-sorted partial structure through generic APIs."""
    if variant not in ("base", "changed"):
        raise ValueError("variant must be 'base' or 'changed'")
    point = Sort("point")
    color = Sort("color")
    points = FiniteCarrier(("left", "right"), sort=point)
    colors = FiniteCarrier(("red", "blue"), sort=color)
    select = OperationSymbol("select", (point, color, point), point)
    permits = RelationSymbol("permits", (point, color))
    signature = Signature((point, color), (select,), (permits,))

    undefined_marker = object()

    def table_value(first: object, shade: object, second: object) -> object:
        """Define one cell, including the changed fixture's consumed cell."""
        if first == second and shade == "red":
            return undefined_marker
        if variant == "changed" and (first, shade, second) == (
            "right",
            "blue",
            "left",
        ):
            return "right"
        return second if shade == "blue" else first

    table = tuple(
        ((first, shade, second), table_value(first, shade, second))
        for first in points
        for shade in colors
        for second in points
    )
    operation = PartialOperation.from_table(
        select,
        ((point, points), (color, colors)),
        table,
        undefined_marker=undefined_marker,
    )
    relation = Relation.from_tuples(
        permits,
        ((point, points), (color, colors)),
        (("left", "red"), ("right", "blue")),
    )
    structure = (
        StructureBuilder(signature)
        .with_carrier(point, points)
        .with_carrier(color, colors)
        .with_operation(select, operation)
        .with_relation(permits, relation)
        .freeze()
    )

    first = Variable("first", point)
    shade = Variable("shade", color)
    second = Variable("second", point)
    first_term = Term.variable(first)
    shade_term = Term.variable(shade)
    second_term = Term.variable(second)
    inner = Term.apply(select, second_term, shade_term, first_term)
    defined_term = Term.apply(select, first_term, shade_term, inner)
    undefined_term = Term.apply(select, first_term, shade_term, first_term)
    defined_environment: Mapping[Variable, object] = MappingProxyType(
        {first: "left", shade: "blue", second: "right"}
    )
    undefined_environment: Mapping[Variable, object] = MappingProxyType(
        {first: "left", shade: "red", second: "right"}
    )
    defined_evaluation = evaluate_term(
        structure,
        defined_term,
        defined_environment,
    )
    undefined_evaluation = evaluate_term(
        structure,
        undefined_term,
        undefined_environment,
    )
    return {
        "defined_evaluation": defined_evaluation,
        "defined_environment": defined_environment,
        "defined_term": defined_term,
        "operation": operation,
        "relation": relation,
        "sorts": (point, color),
        "structure": structure,
        "term_text": _render_term(defined_term),
        "undefined_evaluation": undefined_evaluation,
        "undefined_environment": undefined_environment,
        "undefined_term": undefined_term,
        "variables": (first, shade, second),
        "variant": variant,
    }


def _fixture_hash(fixture: TwoSortedFixture) -> SemanticHash:
    """Hash the bounded neutral definition represented by the public objects."""
    operation = fixture["operation"]
    relation = fixture["relation"]
    record = {
        "operation": {
            "defined": [[list(indices), output] for indices, output in operation.table],
            "name": operation.symbol.name,
            "undefined": [list(indices) for indices in operation.undefined_indices],
        },
        "relation": {
            "name": relation.symbol.name,
            "tuples": [list(indices) for indices in relation.tuples],
        },
        "sorts": [
            {
                "items": list(carrier),
                "name": sort.name,
            }
            for sort, carrier in zip(
                fixture["structure"].signature.sorts,
                fixture["structure"].carriers,
                strict=True,
            )
        ],
        "term": fixture["term_text"],
    }
    return _hash_bytes(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _source_anchor() -> SourceAnchor:
    """Anchor the calculation to the exact executable example source bytes."""
    source_path = Path(__file__).resolve()
    return SourceAnchor.create(
        location="examples/two_sorted_release.py",
        content_hash=_hash_bytes(source_path.read_bytes()),
        edition_or_commit="v0.0-working-tree",
        locator="build_example; _calculation",
    )


def _convention_manifest(source: SourceAnchor) -> ConventionManifest:
    """Select explicit sort, carrier, partiality, and evaluation conventions."""
    fixture_identity = _hash_bytes(
        b"sorts=point,color|point=left,right|color=red,blue|"
        b"partial=Undefined|relation=finite-extension|terms=explicit-tree"
    )
    return ConventionManifest.create(
        "example.two_sorted_release.v1",
        ("point:left", "point:right", "color:red", "color:blue"),
        coefficient_domain="finite-carriers",
        multiplication_fixture=fixture_identity,
        sources=(source,),
        status="accepted",
        signs=(),
        involutions=(),
        normalization=(
            ("sort_order", "point-before-color"),
            ("partiality", "marked-or-missing-cell-is-Undefined"),
            ("evaluation_order", "children-left-to-right-before-parent"),
            ("relation_truth", "declared-tuples-true-otherwise-false"),
        ),
        indexing="zero-based-carrier-index-order",
        parenthesization="explicit-operation-tree",
        conversions=(),
    )


def _contract(
    input_hash: SemanticHash,
    source: SourceAnchor,
    manifest: ConventionManifest,
) -> CalculationContract:
    """Declare the exact bounded infrastructure question before execution."""
    return CalculationContract.create(
        contract_name="two-sorted-release-example",
        project_id="example.two_sorted_release",
        question="exercise one bounded generic two-sorted structure fixture",
        acceptable_outcomes=("constructed",),
        input_hashes=(("structure", input_hash),),
        source_anchors=(source,),
        convention_manifests=(manifest,),
        algorithms=(("finite-table-evaluation", "1"),),
        backend_request="exact",
        backend_name="reference",
        backend_version="1",
        assumptions=(),
        theorem_hypotheses=(),
        bounds=(("cases", 4),),
        required_cross_checks=(),
        expected_artifacts=("two-sorted-result-summary",),
        acceptance_predicates=(
            "defined and undefined outcomes remain distinct",
            "relation returns exact true and false results",
        ),
    )


def _calculation(fixture: TwoSortedFixture) -> CalculationResult:
    """Evaluate positive and negative fixture branches into a neutral packet."""
    defined = evaluate_term(
        fixture["structure"],
        fixture["defined_term"],
        fixture["defined_environment"],
    ).outcome
    undefined = evaluate_term(
        fixture["structure"],
        fixture["undefined_term"],
        fixture["undefined_environment"],
    ).outcome
    relation_true = fixture["relation"].apply("right", "blue")
    relation_false = fixture["relation"].apply("left", "blue")
    expected_value = "left" if fixture["variant"] == "base" else "right"
    if type(defined) is not Defined or defined.value != expected_value:
        raise RuntimeError("defined term did not produce the declared fixture value")
    if type(undefined) is not Undefined:
        raise RuntimeError("partial table cell was silently totalized")
    if (
        type(relation_true) is not RelationResult
        or type(relation_false) is not RelationResult
        or relation_true != RelationResult(True)
        or relation_false != RelationResult(False)
    ):
        raise RuntimeError("relation extension did not return exact truth values")
    result_material = (
        f"defined={defined.value}|undefined={undefined.reason}|"
        f"relation={relation_true.holds},{relation_false.holds}"
    ).encode()
    return CalculationResult.create(
        outcome="constructed",
        summary="bounded two-sorted infrastructure fixture executed",
        case_counts=(("casesExpected", 4), ("casesRun", 4)),
        artifacts=(_hash_bytes(result_material),),
        checks_performed=(
            "nested sorted term evaluation",
            "explicit partial-operation outcome",
            "positive relation query",
            "negative relation query",
        ),
    )


def execute_release_example() -> ReleaseExecution:
    """Run, safely serialize, replay, and invalidate the bounded calculation."""
    fixture = build_example()
    changed_fixture = build_example(variant="changed")
    fixture_hash = _fixture_hash(fixture)
    changed_fixture_hash = _fixture_hash(changed_fixture)
    source_anchor = _source_anchor()
    convention_manifest = _convention_manifest(source_anchor)
    reloaded_source = cast(
        SourceAnchor,
        EVIDENCE_RECORD_REGISTRY.from_record(source_anchor.to_record()),
    )
    reloaded_manifest = cast(
        ConventionManifest,
        EVIDENCE_RECORD_REGISTRY.from_record(convention_manifest.to_record()),
    )
    contract = _contract(fixture_hash, reloaded_source, reloaded_manifest)
    reloaded_contract = cast(
        CalculationContract,
        contract_record_registry().from_record(contract.to_record()),
    )
    receipt = run_calculation(
        reloaded_contract,
        lambda _: _calculation(fixture),
        environment=ExecutionEnvironment.create(
            platform="example",
            python_version="reference",
            software_revision="v0.0",
        ),
    )
    reloaded_receipt = cast(
        ResultReceipt,
        result_receipt_registry().from_record(receipt.to_record()),
    )
    rerun = run_calculation(
        reloaded_contract,
        lambda _: _calculation(fixture),
        environment=ExecutionEnvironment.create(
            platform="example-replay",
            python_version="reference",
            software_revision="v0.0",
        ),
    )
    exact_replay = verify_replay(
        reloaded_receipt,
        resolver=ReplayResolver.create(
            receipts=(reloaded_receipt, rerun),
            reruns=((reloaded_receipt.semantic_hash, rerun.semantic_hash),),
        ),
    )

    changed_contract = _contract(
        changed_fixture_hash,
        reloaded_source,
        reloaded_manifest,
    )
    changed = run_calculation(
        changed_contract,
        lambda _: _calculation(changed_fixture),
        environment=ExecutionEnvironment.create(
            platform="example-changed-input",
            python_version="reference",
            software_revision="v0.0",
        ),
    )
    stale_replay = verify_replay(
        reloaded_receipt,
        resolver=ReplayResolver.create(
            receipts=(reloaded_receipt, changed),
            reruns=((reloaded_receipt.semantic_hash, changed.semantic_hash),),
        ),
    )
    return {
        "changed_fixture": changed_fixture,
        "changed_fixture_hash": changed_fixture_hash,
        "changed_receipt": changed,
        "convention_manifest": reloaded_manifest,
        "convention_round_trip": (
            reloaded_manifest.semantic_hash == convention_manifest.semantic_hash
            and reloaded_manifest.canonical_bytes()
            == convention_manifest.canonical_bytes()
        ),
        "contract": reloaded_contract,
        "contract_round_trip": (
            reloaded_contract.canonical_bytes() == contract.canonical_bytes()
        ),
        "exact_replay": exact_replay,
        "fixture": fixture,
        "fixture_hash": fixture_hash,
        "receipt": reloaded_receipt,
        "receipt_round_trip": (
            reloaded_receipt.canonical_bytes() == receipt.canonical_bytes()
        ),
        "source_anchor": reloaded_source,
        "source_round_trip": (
            reloaded_source.semantic_hash == source_anchor.semantic_hash
            and reloaded_source.canonical_bytes() == source_anchor.canonical_bytes()
        ),
        "stale_replay": stale_replay,
    }


def run_example() -> dict[str, bool | int | str | list[str]]:
    """Return a deterministic, claim-bounded summary suitable for the CLI."""
    release = execute_release_example()
    fixture = release["fixture"]
    defined = fixture["defined_evaluation"].outcome
    undefined = fixture["undefined_evaluation"].outcome
    relation_true = fixture["relation"].apply("right", "blue")
    relation_false = fixture["relation"].apply("left", "blue")
    if (
        type(defined) is not Defined
        or type(undefined) is not Undefined
        or type(relation_true) is not RelationResult
        or type(relation_false) is not RelationResult
    ):
        raise RuntimeError("release summary received an unexpected result branch")
    receipt = release["receipt"]
    return {
        "canonical_convention_round_trip": release["convention_round_trip"],
        "canonical_contract_round_trip": release["contract_round_trip"],
        "canonical_receipt_round_trip": release["receipt_round_trip"],
        "canonical_source_round_trip": release["source_round_trip"],
        "changed_input_detected": (
            release["fixture_hash"] != release["changed_fixture_hash"]
        ),
        "changed_result_detected": (
            release["receipt"].artifacts != release["changed_receipt"].artifacts
        ),
        "convention_manifest_count": len(receipt.convention_manifests),
        "defined_value": cast(str, defined.value),
        "evidence_tier": receipt.evidence_tier,
        "exact_replay": release["exact_replay"].status,
        "execution_status": receipt.execution_status,
        "mathematical_outcome": receipt.mathematical_outcome,
        "operation_totality": fixture["operation"].totality,
        "relation_false": relation_false.holds,
        "relation_true": relation_true.holds,
        "source_anchor_count": len(receipt.source_anchors),
        "stale_edges": list(release["stale_replay"].stale_edges),
        "stale_replay": release["stale_replay"].status,
        "term": fixture["term_text"],
        "undefined_reason": undefined.reason,
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), sort_keys=True, separators=(",", ":")))
