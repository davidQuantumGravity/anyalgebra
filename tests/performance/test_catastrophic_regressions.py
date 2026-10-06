"""Bounded catastrophic-regression guardrails for v0.0 release verification.

These are deliberately *not* benchmarks and make no speed claim.  Each gate
uses a fixed, exact work count or a semantic result ceiling.  Wall-clock timing
is intentionally excluded because this suite must remain meaningful across
developer and CI environments.
"""

from __future__ import annotations

import json

import pytest

from anyalgebra.algebra.multilinear import (
    FiniteMultilinearStructure,
    StructureConstants,
)
from anyalgebra.analysis.fingerprint import algebra_fingerprint
from anyalgebra.core.coercions import CoercionGraph, CoercionMap, coerce
from anyalgebra.core.domains import DomainElement, QQ, ZZ
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.core.parents import FiniteCarrier, SemanticHash, Sort
from anyalgebra.core.rational import Rational
from anyalgebra.evidence.contracts import CalculationContract
from anyalgebra.evidence.replay import (
    ReplayResolver,
    replay_report_record,
    replay_report_registry,
    verify_replay,
)
from anyalgebra.evidence.run import (
    CalculationResult,
    ExecutionEnvironment,
    result_receipt_record,
    result_receipt_registry,
    run_calculation,
)
from anyalgebra.fixtures.composition import composition_fixture
from anyalgebra.structures.laws import Equation, Law
from anyalgebra.structures.operations import Operation
from anyalgebra.structures.signatures import OperationSymbol, Signature
from anyalgebra.structures.structure import Structure, StructureBuilder
from anyalgebra.structures.terms import Term, Variable
from anyalgebra.validation.validate import Proved, validate_law
import anyalgebra.core.coercions as coercions_module
import anyalgebra.evidence.replay as replay_module
import anyalgebra.evidence.run as run_module


# These are workload contracts, not calibrated performance baselines.  A
# changed ceiling must be justified by a deliberate fixture or algorithm change.
EXACT_COERCION_CASES = 32
FINITE_LAW_ASSIGNMENTS = 8
COMPOSITION_TABLE_CELLS = 160
COMPOSITION_FINGERPRINT_TRIPLES = 64
REPLAY_CASES = 8


def _exact_graph(calls: list[int]) -> CoercionGraph:
    """Return one literal, lossless ZZ-to-QQ edge with an observable count."""

    def lift(value: DomainElement[int]) -> DomainElement[Rational]:
        calls.append(value.value)
        return QQ().element(value.value)

    edge = CoercionMap(
        id="performance.zz_to_qq.v1",
        source=ZZ(),
        target=QQ(),
        forward=lift,
        injective=True,
        exact=True,
        lossless=True,
    )
    return CoercionGraph().with_map(edge)


def _assert_exact_coercion_workload() -> None:
    """Exercise exactly the declared coercion workload and semantic output."""

    calls: list[int] = []
    graph = _exact_graph(calls)
    for value in range(EXACT_COERCION_CASES):
        lifted = coerce(ZZ().element(value), QQ(), graph=graph)
        assert lifted.parent is QQ()
        assert lifted == QQ().element(value)
    assert calls == list(range(EXACT_COERCION_CASES))
    assert len(calls) <= EXACT_COERCION_CASES


def test_exact_domain_coercion_has_a_fixed_work_ceiling() -> None:
    """A one-edge exact embedding performs one declared map call per input."""

    _assert_exact_coercion_workload()


def _binary_structure() -> tuple[Structure, Law]:
    """Construct the exact two-element associative finite-law fixture."""

    bit = Sort("performance-bit")
    product = OperationSymbol("and", (bit, bit), bit)
    carrier = FiniteCarrier((0, 1), sort=bit)
    operation = Operation.from_table(
        product,
        ((bit, carrier),),
        tuple(((left, right), left & right) for left in (0, 1) for right in (0, 1)),
    )
    structure = (
        StructureBuilder(Signature((bit,), (product,)))
        .with_carrier(bit, carrier)
        .with_operation(product, operation)
        .freeze()
    )
    x, y, z = (Variable(name, bit) for name in ("x", "y", "z"))
    x_term, y_term, z_term = (Term.variable(item) for item in (x, y, z))
    law = Law(
        "associativity",
        (x, y, z),
        Equation(
            Term.apply(product, Term.apply(product, x_term, y_term), z_term),
            Term.apply(product, x_term, Term.apply(product, y_term, z_term)),
        ),
    )
    return structure, law


def test_finite_law_validation_has_an_exact_assignment_ceiling() -> None:
    """The declared binary three-variable grid is exactly 2**3 assignments."""

    structure, law = _binary_structure()
    report = validate_law(structure, law)
    assert type(report) is Proved
    assert (report.expected_assignments, report.evaluated_assignments) == (
        FINITE_LAW_ASSIGNMENTS,
        FINITE_LAW_ASSIGNMENTS,
    )
    assert report.conclusion_evaluations == FINITE_LAW_ASSIGNMENTS


def _qq_lift(value: FiniteMultilinearStructure) -> FiniteMultilinearStructure:
    """Lift one finite ZZ table to QQ through the generic sparse API."""

    domain = QQ()
    module = FreeModule(
        domain, Basis(value.module.basis.labels, coefficient_domain=domain)
    )
    constants = StructureConstants.from_sparse(
        (module.basis, module.basis),
        module.basis,
        tuple(
            (key, domain.element(coefficient.value))
            for key, coefficient in value.operation.constants.entries
        ),
    )
    return FiniteMultilinearStructure.from_structure_constants(module, constants)


def test_composition_fixture_and_fingerprint_work_remain_bounded() -> None:
    """All named tables and a QQ quaternion fingerprint retain fixed counts."""

    identifiers = ("algmul.H.v1", "algmul.Hs.v1", "algmul.O.v1", "algmul.Os.v1")
    cells = 0
    for identifier in identifiers:
        fixture = composition_fixture(identifier)
        rank = fixture.module.rank
        for left in range(rank):
            for right in range(rank):
                assert fixture.evaluate_basis(left, right).parent is fixture.module
                cells += 1
    assert cells == COMPOSITION_TABLE_CELLS

    first = algebra_fingerprint(_qq_lift(composition_fixture("algmul.H.v1")))
    second = algebra_fingerprint(_qq_lift(composition_fixture("algmul.H.v1")))
    assert first == second
    assert first.isomorphism_rejection_only is True
    assert first.associativity_triples_checked == COMPOSITION_FINGERPRINT_TRIPLES
    assert first.associativity_triples_checked <= COMPOSITION_FINGERPRINT_TRIPLES


def _contract() -> CalculationContract:
    """Return a small exact, non-promoting replay contract."""

    return CalculationContract.create(
        contract_name="performance-replay",
        project_id="test.performance",
        question="replay only",
        acceptable_outcomes=("constructed",),
        input_hashes=(("fixture", SemanticHash("sha256", "a" * 64)),),
        source_anchors=(),
        convention_manifests=(),
        algorithms=(("reference", "1"),),
        backend_request="exact",
        backend_name="reference",
        backend_version="1",
        assumptions=(),
        theorem_hypotheses=(),
        bounds=(("cases", REPLAY_CASES),),
        required_cross_checks=(),
        expected_artifacts=(),
        acceptance_predicates=("stored",),
    )


def _result(_: CalculationContract) -> CalculationResult:
    """Return fixed semantic output with one declared bounded case count."""

    return CalculationResult.create(
        outcome="constructed",
        summary="bounded reproducibility fixture",
        case_counts=(("casesRun", REPLAY_CASES),),
        checks_performed=("canonical replay",),
    )


def _semantic_receipt_projection(record: dict[str, object]) -> bytes:
    """Canonicalize semantic receipt fields, excluding documented run metadata."""

    projection = dict(record)
    for field in ("environment", "startedAt", "finishedAt"):
        projection.pop(field)
    return json.dumps(projection, sort_keys=True, separators=(",", ":")).encode()


def _assert_replay_workload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compare canonical semantic output while allowing environment/time changes."""

    timestamps = iter(
        (
            "2026-07-25T00:00:00Z",
            "2026-07-25T00:00:01Z",
            "2026-07-25T00:01:00Z",
            "2026-07-25T00:01:01Z",
        )
    )
    monkeypatch.setattr(run_module, "_utc_now", lambda: next(timestamps))
    contract = _contract()
    original = run_calculation(
        contract,
        _result,
        environment=ExecutionEnvironment.create(platform="first"),
    )
    rerun = run_calculation(
        contract,
        _result,
        environment=ExecutionEnvironment.create(platform="second"),
    )
    original_record = result_receipt_record(original)
    rerun_record = result_receipt_record(rerun)
    assert original.semantic_hash != rerun.semantic_hash
    original_semantic = _semantic_receipt_projection(original_record)
    rerun_semantic = _semantic_receipt_projection(rerun_record)
    assert original_semantic == rerun_semantic
    assert (
        result_receipt_record(result_receipt_registry().from_record(original_record))
        == original_record
    )

    report = verify_replay(
        original,
        resolver=ReplayResolver.create(
            receipts=(original, rerun),
            reruns=((original.semantic_hash, rerun.semantic_hash),),
        ),
    )
    assert report.status == "exact_match"
    assert report.matched_fields == (
        "contract",
        "direct_dependencies",
        "mathematical_outcome",
        "primary_result_hash",
    )
    assert replay_report_registry().from_record(replay_report_record(report)) == report


def test_evidence_serialization_and_replay_exclude_run_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run metadata changes do not change the compared semantic replay output."""

    _assert_replay_workload(monkeypatch)


def test_selected_mutants_are_killed_by_bounded_oracles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Perturb two actual decision seams and prove these workload oracles fail.

    Killed mutants: ``core.coercions._unique_lossless_route -> ()`` and
    ``evidence.replay.ReplayResolver._rerun_for -> None``.  No survivor is
    accepted by this sample; this is a small mutation check, not a score.
    """

    monkeypatch.setattr(coercions_module, "_unique_lossless_route", lambda *_: ())
    with pytest.raises(ValueError, match="route does not end"):
        _assert_exact_coercion_workload()
    monkeypatch.undo()

    monkeypatch.setattr(replay_module.ReplayResolver, "_rerun_for", lambda *_: None)
    with pytest.raises(AssertionError):
        _assert_replay_workload(monkeypatch)
