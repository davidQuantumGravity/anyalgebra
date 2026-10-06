"""One claim-neutral release calculation for partiality and bounded deduction.

The example uses the generic finite-operation, typed-term, totalization, and
ground-deduction APIs.  It defines no ``Structure`` subclass.  Its E1 receipt
records execution of the declared finite checks only; it does not promote a
mathematical or scientific claim.
"""

from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path
from typing import TypedDict, cast

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
from anyalgebra.evidence.run import (
    CalculationResult,
    ExecutionEnvironment,
    ResultReceipt,
    result_receipt_registry,
    run_calculation,
)
from anyalgebra.structures.deductive import (
    BoundExhausted,
    CompleteClosure,
    DeductiveSystem,
    InferenceRule,
    closure,
)
from anyalgebra.structures.operations import PartialOperation
from anyalgebra.structures.outcomes import Defined, Undefined
from anyalgebra.structures.partiality import Totalization, totalize
from anyalgebra.structures.signatures import OperationSymbol
from anyalgebra.structures.terms import Term, Variable


class PartialDeductiveFixture(TypedDict):
    """The neutral objects and bounded outcomes consumed by the calculation."""

    bottom: str
    bounded_closure: BoundExhausted
    complete_closure: CompleteClosure
    deductive_system: DeductiveSystem
    element_sort: Sort
    judgment_sort: Sort
    left_term: Term
    left_term_text: str
    partial_operation: PartialOperation
    premises: tuple[Term, ...]
    right_term: Term
    right_term_text: str
    totalization: Totalization
    variables: tuple[Variable, Variable, Variable]


class ReleaseExecution(TypedDict):
    """The stable public records retained by this bounded release example."""

    convention_manifest: ConventionManifest
    convention_round_trip: bool
    contract: CalculationContract
    contract_round_trip: bool
    fixture: PartialDeductiveFixture
    fixture_hash: SemanticHash
    receipt: ResultReceipt
    receipt_round_trip: bool
    source_anchor: SourceAnchor
    source_round_trip: bool


def _hash_bytes(value: bytes) -> SemanticHash:
    """Return a deterministic SHA-256 semantic hash."""
    return SemanticHash("sha256", hashlib.sha256(value).hexdigest())


def _atom(name: str, sort: Sort) -> Term:
    """Construct one ground judgment as a nullary typed term."""
    return Term.apply(OperationSymbol(name, (), sort))


def term_name(term: Term) -> str:
    """Return the operation name of one exact ground atom."""
    if type(term) is not Term or term.symbol is None or term.arguments:
        raise ValueError("term must be an exact ground nullary operation term")
    return term.symbol.name


def _render_term(term: Term) -> str:
    """Render applications with explicit parentheses and result sorts."""
    if term.variable_value is not None:
        return f"{term.variable_value.name}:{term.sort.name}"
    if term.symbol is None:
        raise RuntimeError("term node has neither a variable nor an operation")
    arguments = ", ".join(_render_term(argument) for argument in term.arguments)
    return f"({term.symbol.name}({arguments}):{term.sort.name})"


def build_example() -> PartialDeductiveFixture:
    """Build one partial magma, its explicit totalization, and a finite closure."""
    element = Sort("partial_magma_element")
    carrier = FiniteCarrier(("a", "b", "c"), sort=element)
    product_symbol = OperationSymbol("product", (element, element), element)
    undefined_marker = object()
    partial_operation = PartialOperation.from_table(
        product_symbol,
        ((element, carrier),),
        (
            (("a", "a"), "a"),
            (("a", "b"), "c"),
            (("b", "a"), "b"),
            (("b", "c"), undefined_marker),
        ),
        undefined_marker=undefined_marker,
    )
    bottom = "bottom"
    totalization = totalize(partial_operation, bottom=bottom)

    x = Variable("x", element)
    y = Variable("y", element)
    z = Variable("z", element)
    x_term = Term.variable(x)
    y_term = Term.variable(y)
    z_term = Term.variable(z)
    left_term = Term.apply(
        product_symbol,
        Term.apply(product_symbol, x_term, y_term),
        z_term,
    )
    right_term = Term.apply(
        product_symbol,
        x_term,
        Term.apply(product_symbol, y_term, z_term),
    )

    judgment = Sort("judgment")
    seed = _atom("seed", judgment)
    middle = _atom("middle", judgment)
    goal = _atom("goal", judgment)
    system = DeductiveSystem(
        (
            InferenceRule((seed,), middle),
            InferenceRule((middle,), goal),
        )
    )
    bounded = closure((seed,), system, max_rounds=1)
    completed = closure((seed,), system, max_rounds=2)
    if type(bounded) is not BoundExhausted:
        raise RuntimeError("one-round deduction did not retain its exhausted bound")
    if type(completed) is not CompleteClosure:
        raise RuntimeError("two-round deduction did not reach finite closure")
    return {
        "bottom": bottom,
        "bounded_closure": bounded,
        "complete_closure": completed,
        "deductive_system": system,
        "element_sort": element,
        "judgment_sort": judgment,
        "left_term": left_term,
        "left_term_text": _render_term(left_term),
        "partial_operation": partial_operation,
        "premises": (seed,),
        "right_term": right_term,
        "right_term_text": _render_term(right_term),
        "totalization": totalization,
        "variables": (x, y, z),
    }


def _fixture_hash(fixture: PartialDeductiveFixture) -> SemanticHash:
    """Hash the complete finite fixture without using object representations."""
    source = fixture["partial_operation"]
    totalized = fixture["totalization"].operation
    completed = fixture["complete_closure"]
    bounded = fixture["bounded_closure"]
    record = {
        "deduction": {
            "bounded": [term_name(item) for item in bounded.conclusions],
            "boundedMaxRounds": bounded.max_rounds,
            "complete": [term_name(item) for item in completed.conclusions],
            "traces": [
                {
                    "conclusion": term_name(trace.conclusion),
                    "premiseIndices": list(trace.premise_indices),
                    "ruleIndex": trace.rule_index,
                }
                for trace in completed.derivations
            ],
        },
        "partial": {
            "carrier": list(source.output_carrier),
            "defined": [
                [list(index_tuple), output] for index_tuple, output in source.table
            ],
            "undefined": [
                list(index_tuple) for index_tuple in source.undefined_indices
            ],
        },
        "terms": {
            "left": fixture["left_term_text"],
            "right": fixture["right_term_text"],
        },
        "totalization": {
            "bottom": fixture["bottom"],
            "carrier": list(fixture["totalization"].totalized_carrier),
            "table": [
                [list(index_tuple), output] for index_tuple, output in totalized.table
            ],
        },
    }
    return _hash_bytes(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _source_anchor() -> SourceAnchor:
    """Anchor the calculation to the exact executable source bytes."""
    source_path = Path(__file__).resolve()
    return SourceAnchor.create(
        location="examples/partial_and_deductive_release.py",
        content_hash=_hash_bytes(source_path.read_bytes()),
        edition_or_commit="v0.0-working-tree",
        locator="build_example; _calculation",
    )


def _convention_manifest(
    source: SourceAnchor, fixture_hash: SemanticHash
) -> ConventionManifest:
    """Record the partiality, totalization, term, and deduction conventions."""
    return ConventionManifest.create(
        "example.partial_and_deductive_release.v1",
        (
            "partial_magma:a",
            "partial_magma:b",
            "partial_magma:c",
            "partial_magma:bottom",
            "judgment:seed",
            "judgment:middle",
            "judgment:goal",
        ),
        coefficient_domain="finite-carriers",
        multiplication_fixture=fixture_hash,
        sources=(source,),
        status="accepted",
        signs=(),
        involutions=(),
        normalization=(
            ("partiality", "marked-or-missing-cell-is-Undefined"),
            ("totalization", "explicit-strict-bottom"),
            ("deduction", "synchronous-ground-rule-rounds"),
            ("term_rendering", "fully-parenthesized-with-result-sorts"),
        ),
        indexing="zero-based-carrier-and-conclusion-order",
        parenthesization="explicit-operation-tree",
        conversions=(("partial-operation-to-total-operation", "strict-bottom"),),
    )


def _contract(
    fixture_hash: SemanticHash,
    source: SourceAnchor,
    manifest: ConventionManifest,
) -> CalculationContract:
    """Declare the exact finite checks before running them."""
    return CalculationContract.create(
        contract_name="partial-and-deductive-release-example",
        project_id="example.partial_and_deductive_release",
        question="exercise bounded partiality, totalization, and ground deduction",
        acceptable_outcomes=("constructed",),
        input_hashes=(("fixture", fixture_hash),),
        source_anchors=(source,),
        convention_manifests=(manifest,),
        algorithms=(
            ("finite-table-enumeration", "1"),
            ("strict-bottom-totalization", "1"),
            ("synchronous-ground-closure", "1"),
        ),
        backend_request="exact",
        backend_name="reference",
        backend_version="1",
        assumptions=(),
        theorem_hypotheses=(),
        bounds=(
            ("sourceCells", 9),
            ("totalizedCells", 16),
            ("deductionRules", 2),
            ("closureMaxRounds", 2),
        ),
        required_cross_checks=(),
        expected_artifacts=("partial-and-deductive-result-summary",),
        acceptance_predicates=(
            "undefined source cells remain Undefined",
            "totalization is explicit and strict at bottom",
            "deduction traces replay within their recorded round bound",
            "term trees preserve parentheses and sorts",
        ),
    )


def _calculation(fixture: PartialDeductiveFixture) -> CalculationResult:
    """Execute 28 declared checks over the finite fixture."""
    source = fixture["partial_operation"]
    totalized = fixture["totalization"]
    source_cases = tuple(product(source.output_carrier, repeat=2))
    totalized_cases = tuple(product(totalized.totalized_carrier, repeat=2))
    source_outcomes = tuple(source.apply(*arguments) for arguments in source_cases)
    totalized_outcomes = tuple(
        totalized.operation.apply(*arguments) for arguments in totalized_cases
    )
    if len(source_outcomes) != 9 or any(
        type(outcome) not in (Defined, Undefined) for outcome in source_outcomes
    ):
        raise RuntimeError("partial source enumeration did not retain exact outcomes")
    if len(totalized_outcomes) != 16 or any(
        type(outcome) is not Defined for outcome in totalized_outcomes
    ):
        raise RuntimeError("explicit totalization did not define every finite cell")
    original_undefined = source.apply("b", "c")
    totalized_cell = totalized.operation.apply("b", "c")
    strict_left = totalized.operation.apply(fixture["bottom"], "a")
    strict_right = totalized.operation.apply("a", fixture["bottom"])
    if type(original_undefined) is not Undefined:
        raise RuntimeError("source undefinedness changed during totalization")
    if any(
        type(outcome) is not Defined or outcome.value != fixture["bottom"]
        for outcome in (totalized_cell, strict_left, strict_right)
    ):
        raise RuntimeError("total operation did not apply the strict-bottom policy")
    completed = fixture["complete_closure"]
    bounded = fixture["bounded_closure"]
    if [term_name(item) for item in bounded.conclusions] != ["seed", "middle"]:
        raise RuntimeError("bounded closure did not retain its finite frontier")
    for trace in completed.derivations:
        if (
            tuple(completed.conclusions[index] for index in trace.premise_indices)
            != trace.premises
        ):
            raise RuntimeError("deduction derivation trace did not replay")
    if fixture["left_term"] == fixture["right_term"]:
        raise RuntimeError("distinct term parentheses were silently reassociated")
    material = {
        "bounded": [term_name(item) for item in bounded.conclusions],
        "complete": [term_name(item) for item in completed.conclusions],
        "sourceUndefined": original_undefined.reason,
        "strictBottom": fixture["bottom"],
        "traces": [list(item.premise_indices) for item in completed.derivations],
    }
    return CalculationResult.create(
        outcome="constructed",
        summary="bounded partiality and ground-deduction fixture executed",
        case_counts=(
            ("casesExpected", 28),
            ("casesRun", 28),
            ("sourceCells", len(source_outcomes)),
            ("totalizedCells", len(totalized_outcomes)),
            ("deductionDerivations", len(completed.derivations)),
        ),
        artifacts=(
            _hash_bytes(
                json.dumps(material, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ),
        ),
        checks_performed=(
            "nine partial-operation cells",
            "sixteen strict-bottom total-operation cells",
            "one bound-exhaustion status",
            "two replayable derivation traces",
            "typed parenthesized term distinction",
        ),
    )


def execute_release_example() -> ReleaseExecution:
    """Run the bounded calculation through all public stable record codecs."""
    fixture = build_example()
    fixture_hash = _fixture_hash(fixture)
    source = _source_anchor()
    manifest = _convention_manifest(source, fixture_hash)
    reloaded_source = cast(
        SourceAnchor,
        EVIDENCE_RECORD_REGISTRY.from_record(source.to_record()),
    )
    reloaded_manifest = cast(
        ConventionManifest,
        EVIDENCE_RECORD_REGISTRY.from_record(manifest.to_record()),
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
    return {
        "convention_manifest": reloaded_manifest,
        "convention_round_trip": (
            reloaded_manifest.canonical_bytes() == manifest.canonical_bytes()
        ),
        "contract": reloaded_contract,
        "contract_round_trip": (
            reloaded_contract.canonical_bytes() == contract.canonical_bytes()
        ),
        "fixture": fixture,
        "fixture_hash": fixture_hash,
        "receipt": reloaded_receipt,
        "receipt_round_trip": (
            reloaded_receipt.canonical_bytes() == receipt.canonical_bytes()
        ),
        "source_anchor": reloaded_source,
        "source_round_trip": (
            reloaded_source.canonical_bytes() == source.canonical_bytes()
        ),
    }


def run_example() -> dict[str, object]:
    """Return one deterministic CLI summary with bounded evidence language."""
    release = execute_release_example()
    fixture = release["fixture"]
    original = fixture["partial_operation"].apply("b", "c")
    totalized = fixture["totalization"].operation.apply("b", "c")
    strict = fixture["totalization"].operation.apply(fixture["bottom"], "a")
    if (
        type(original) is not Undefined
        or type(totalized) is not Defined
        or type(strict) is not Defined
    ):
        raise RuntimeError("release summary received an unexpected outcome branch")
    bounded = fixture["bounded_closure"]
    completed = fixture["complete_closure"]
    receipt = release["receipt"]
    return {
        "bounded_conclusions": [term_name(item) for item in bounded.conclusions],
        "bounded_status": "bound_exhausted",
        "canonical_convention_round_trip": release["convention_round_trip"],
        "canonical_contract_round_trip": release["contract_round_trip"],
        "canonical_receipt_round_trip": release["receipt_round_trip"],
        "canonical_source_round_trip": release["source_round_trip"],
        "complete_conclusions": [term_name(item) for item in completed.conclusions],
        "derivation_premise_indices": [
            list(item.premise_indices) for item in completed.derivations
        ],
        "evidence_tier": receipt.evidence_tier,
        "execution_status": receipt.execution_status,
        "left_term": fixture["left_term_text"],
        "mathematical_outcome": receipt.mathematical_outcome,
        "original_carrier_size": len(fixture["partial_operation"].output_carrier),
        "original_undefined": original.reason,
        "strict_bottom": strict.value == fixture["bottom"],
        "totalized_carrier_size": len(fixture["totalization"].totalized_carrier),
        "totalized_value": totalized.value,
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), sort_keys=True, separators=(",", ":")))
