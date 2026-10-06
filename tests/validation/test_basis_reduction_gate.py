"""Contract tests for conservative theorem-reduction eligibility gates."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import FrozenInstanceError, dataclass
from itertools import repeat
from typing import Any, cast

import pytest

from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.structures.laws import Equation, Law
from anyalgebra.structures.operations import Operation
from anyalgebra.structures.signatures import OperationSymbol, Signature
from anyalgebra.structures.structure import Structure, StructureBuilder
from anyalgebra.structures.terms import Term, Variable
from anyalgebra.validation.reduction import (
    BasisReductionDefinitionError,
    BasisReductionDenied,
    BasisReductionReady,
    ClosureObligation,
    MultilinearityObligation,
    Obligation,
    PremiseObligation,
    ScalarDomainObligation,
    basis_reduction_gate,
)
from anyalgebra.validation.validate import Disproved, Proved, validate_law


@dataclass
class _Context:
    """One deliberately non-model target with independently proved evidence."""

    structure: Structure
    scalar: Sort
    vector: Sort
    product: OperationSymbol
    transport: OperationSymbol
    x: Variable
    y: Variable
    target: Law
    premise: Equation
    obligations: tuple[
        MultilinearityObligation
        | ClosureObligation
        | ScalarDomainObligation
        | PremiseObligation,
        ...,
    ]
    calls: list[int]


def _context() -> _Context:
    scalar = Sort("scalar")
    vector = Sort("vector")
    scalar_carrier = FiniteCarrier((0, 1), sort=scalar)
    vector_carrier = FiniteCarrier((0, 1), sort=vector)
    product = OperationSymbol("product", (scalar, scalar), scalar)
    transport = OperationSymbol("transport", (scalar, vector), vector)
    calls = [0]

    def product_function(left: object, right: object) -> int:
        del left, right
        calls[0] += 1
        return 0

    def transport_function(left: object, right: object) -> int:
        del left, right
        calls[0] += 1
        return 0

    structure = (
        StructureBuilder(Signature((scalar, vector), (product, transport)))
        .with_carrier(scalar, scalar_carrier)
        .with_carrier(vector, vector_carrier)
        .with_operation(
            product,
            Operation.from_callable(
                product,
                ((scalar, scalar_carrier), (vector, vector_carrier)),
                product_function,
            ),
        )
        .with_operation(
            transport,
            Operation.from_callable(
                transport,
                ((scalar, scalar_carrier), (vector, vector_carrier)),
                transport_function,
            ),
        )
        .freeze()
    )
    x, y = Variable("x", scalar), Variable("y", vector)
    x_term, y_term = Term.variable(x), Term.variable(y)
    square = Term.apply(product, x_term, x_term)
    premise = Equation(square, square)
    target = Law(
        "unproved_target",
        (x, y),
        Equation(Term.apply(transport, square, y_term), y_term),
        hypotheses=(premise,),
    )

    def proved(name: str) -> Proved:
        report = validate_law(structure, Law(name, (x,), Equation(x_term, x_term)))
        assert type(report) is Proved
        return report

    premise_report = validate_law(structure, Law("premise", (x,), premise))
    assert type(premise_report) is Proved
    reports = tuple(proved(f"evidence_{index}") for index in range(7))
    obligations = (
        MultilinearityObligation(product, (0,), reports[0]),
        MultilinearityObligation(product, (1,), reports[1]),
        MultilinearityObligation(transport, (0,), reports[2]),
        MultilinearityObligation(transport, (1,), reports[3]),
        ClosureObligation(product, scalar, reports[4]),
        ClosureObligation(transport, vector, reports[5]),
        ScalarDomainObligation("QQ", "characteristic_zero", "polarization", reports[6]),
        PremiseObligation(0, premise_report),
    )
    calls[0] = 0
    return _Context(
        structure,
        scalar,
        vector,
        product,
        transport,
        x,
        y,
        target,
        premise,
        obligations,
        calls,
    )


def _proved(context: _Context, name: str) -> Proved:
    term = Term.variable(context.x)
    report = validate_law(
        context.structure, Law(name, (context.x,), Equation(term, term))
    )
    assert type(report) is Proved
    context.calls[0] = 0
    return report


def _replace(
    context: _Context, index: int, replacement: Obligation
) -> tuple[Obligation, ...]:
    values = list(context.obligations)
    values[index] = replacement
    return tuple(values)


def test_complete_explicit_obligations_only_make_a_conditional_certificate() -> None:
    context = _context()
    target_report = validate_law(context.structure, context.target)
    assert type(target_report) is Disproved
    context.calls[0] = 0

    result = basis_reduction_gate(
        context.structure, context.target, context.obligations
    )

    assert type(result) is BasisReductionReady
    assert result.structure is context.structure and result.law is context.target
    assert result.obligations == context.obligations
    assert "Proved" not in type(result).__name__
    assert result.is_conditional is True
    assert context.calls == [0]


@pytest.mark.parametrize(
    ("index", "issue"),
    (
        (0, "missing_multilinearity:product:0"),
        (4, "missing_closure:product:scalar"),
        (6, "missing_scalar_domain"),
        (7, "missing_premise:0"),
    ),
)
def test_each_required_hypothesis_kind_is_independently_denied(
    index: int, issue: str
) -> None:
    context = _context()
    supplied = tuple(
        item
        for item_index, item in enumerate(context.obligations)
        if item_index != index
    )

    result = basis_reduction_gate(context.structure, context.target, supplied)

    assert type(result) is BasisReductionDenied
    assert issue in result.issues
    assert context.calls == [0]


def test_zero_premise_law_needs_no_premise_report() -> None:
    context = _context()
    target = Law("no_premise", (context.x, context.y), context.target.conclusion)
    supplied = tuple(
        item for item in context.obligations if type(item) is not PremiseObligation
    )

    result = basis_reduction_gate(context.structure, target, supplied)

    assert type(result) is BasisReductionReady
    assert not hasattr(result, "issues")


def test_incomplete_slots_and_extras_deny_even_when_complete_set_exists() -> None:
    context = _context()
    incomplete = _replace(
        context,
        1,
        MultilinearityObligation(context.product, (), _proved(context, "incomplete")),
    )
    foreign = OperationSymbol("foreign", (context.scalar,), context.scalar)
    cases = (
        (
            incomplete,
            "missing_multilinearity:product:1",
        ),
        (
            (
                *context.obligations,
                MultilinearityObligation(foreign, (0,), _proved(context, "foreign")),
            ),
            "irrelevant_multilinearity",
        ),
        (
            (
                *context.obligations,
                ClosureObligation(
                    context.product, context.vector, _proved(context, "wrong_sort")
                ),
            ),
            "invalid_closure_sort",
        ),
        (
            (
                *context.obligations,
                ClosureObligation(
                    foreign, context.scalar, _proved(context, "foreign_closure")
                ),
            ),
            "irrelevant_closure",
        ),
        (
            (
                *context.obligations,
                PremiseObligation(1, _proved(context, "wrong_premise")),
            ),
            "invalid_premise_index",
        ),
        (
            _replace(
                context,
                7,
                PremiseObligation(0, _proved(context, "wrong_premise_evidence")),
            ),
            "invalid_premise_evidence",
        ),
    )

    for supplied, issue in cases:
        result = basis_reduction_gate(context.structure, context.target, supplied)
        assert type(result) is BasisReductionDenied
        assert issue in result.issues
        assert context.calls == [0]


def test_duplicate_keys_and_reused_reports_are_rejected() -> None:
    context = _context()
    report = context.obligations[0].evidence
    duplicate_key = ScalarDomainObligation(
        "RR", "characteristic_zero", "polarization", _proved(context, "duplicate_key")
    )
    reused = ClosureObligation(context.product, context.scalar, report)

    result = basis_reduction_gate(
        context.structure,
        context.target,
        (*context.obligations, duplicate_key, reused),
    )

    assert type(result) is BasisReductionDenied
    assert result.issues.count("duplicate_obligation") == 2
    assert "duplicate_evidence_report" in result.issues


def test_invalid_or_circular_evidence_never_counts_toward_readiness() -> None:
    context = _context()
    circular = _proved(context, "circular")
    object.__setattr__(circular, "law", context.target)
    stale = _proved(context, "stale")
    object.__setattr__(stale, "algorithm_version", 2)
    foreign_context = _context()
    foreign = _proved(foreign_context, "foreign_structure")
    replacements = (
        (
            ScalarDomainObligation(
                "QQ", "characteristic_zero", "polarization", circular
            ),
            "invalid_provenance",
        ),
        (
            ScalarDomainObligation("QQ", "characteristic_zero", "polarization", stale),
            "invalid_provenance",
        ),
        (
            ScalarDomainObligation(
                "QQ", "characteristic_zero", "polarization", foreign
            ),
            "invalid_provenance",
        ),
    )

    for replacement, issue in replacements:
        result = basis_reduction_gate(
            context.structure, context.target, _replace(context, 6, replacement)
        )
        assert type(result) is BasisReductionDenied
        assert issue in result.issues
        assert "missing_scalar_domain" in result.issues


def test_obligation_constructors_reject_malformed_exact_types_and_reports() -> None:
    context = _context()
    report = context.obligations[0].evidence
    with pytest.raises(BasisReductionDefinitionError, match="operation"):
        MultilinearityObligation(cast(OperationSymbol, object()), (), report)
    with pytest.raises(BasisReductionDefinitionError, match="slots"):
        MultilinearityObligation(context.product, cast(Iterable[int], 4), report)
    with pytest.raises(BasisReductionDefinitionError, match="duplicate slot"):
        MultilinearityObligation(context.product, (0, 0), report)
    with pytest.raises(BasisReductionDefinitionError, match="valid built-in slot"):
        MultilinearityObligation(context.product, (-1,), report)
    with pytest.raises(BasisReductionDefinitionError, match="too many"):
        MultilinearityObligation(context.product, (0, 1, 2), report)
    with pytest.raises(BasisReductionDefinitionError, match="sort"):
        ClosureObligation(context.product, cast(Sort, object()), report)
    with pytest.raises(BasisReductionDefinitionError, match="operation"):
        ClosureObligation(cast(OperationSymbol, object()), context.scalar, report)
    with pytest.raises(BasisReductionDefinitionError, match="premise_index"):
        PremiseObligation(-1, report)
    with pytest.raises(BasisReductionDefinitionError, match="scope"):
        ScalarDomainObligation(" QQ", "zero", "polarization", report)
    with pytest.raises(BasisReductionDefinitionError, match="Proved"):
        MultilinearityObligation(
            context.product,
            (0,),
            cast(Proved, validate_law(context.structure, context.target)),
        )
    malformed = _proved(context, "incomplete_report")
    object.__setattr__(malformed, "evaluated_assignments", 0)
    with pytest.raises(BasisReductionDefinitionError, match="exhaust"):
        ScalarDomainObligation("QQ", "zero", "polarization", malformed)


def test_gate_rejects_exact_public_input_types_and_hostile_iterables() -> None:
    context = _context()

    class Hostile:
        def __iter__(self) -> object:
            raise RuntimeError("do not consume")

    class FailsOnNext:
        def __iter__(self) -> FailsOnNext:
            return self

        def __next__(self) -> object:
            raise RuntimeError("do not consume")

    with pytest.raises(BasisReductionDefinitionError, match="structure"):
        basis_reduction_gate(cast(Structure, object()), context.target, ())
    with pytest.raises(BasisReductionDefinitionError, match="law"):
        basis_reduction_gate(context.structure, cast(Law, object()), ())
    with pytest.raises(BasisReductionDefinitionError, match="snapshotted"):
        basis_reduction_gate(
            context.structure, context.target, cast(Iterable[Obligation], Hostile())
        )
    with pytest.raises(BasisReductionDefinitionError, match="snapshotted"):
        basis_reduction_gate(
            context.structure,
            context.target,
            cast(Iterable[Obligation], FailsOnNext()),
        )
    with pytest.raises(BasisReductionDefinitionError, match="exact obligation"):
        basis_reduction_gate(
            context.structure,
            context.target,
            (cast(Obligation, report) for report in (object(),)),
        )


def test_snapshots_are_one_pass_bounded_and_reject_truly_infinite_inputs() -> None:
    context = _context()
    seen = [0]

    def one_pass() -> Iterator[Obligation]:
        for obligation in context.obligations:
            seen[0] += 1
            yield obligation

    accepted = basis_reduction_gate(context.structure, context.target, one_pass())
    assert type(accepted) is BasisReductionReady
    assert seen == [len(context.obligations)]
    repeated = context.obligations[0]
    snapshot_512 = basis_reduction_gate(
        context.structure, context.target, repeat(repeated, 512)
    )
    assert type(snapshot_512) is BasisReductionDenied
    with pytest.raises(BasisReductionDefinitionError, match="obligations\\[513\\]"):
        basis_reduction_gate(context.structure, context.target, repeat(repeated))
    with pytest.raises(BasisReductionDefinitionError, match="obligations\\[513\\]"):
        basis_reduction_gate(context.structure, context.target, repeat(repeated, 513))


def test_target_syntax_must_belong_to_the_structure_before_evidence() -> None:
    context = _context()
    foreign_sort = Sort("foreign")
    foreign_variable = Variable("foreign_variable", foreign_sort)
    foreign_operation = OperationSymbol(
        "foreign_operation", (context.scalar,), context.scalar
    )
    scalar_term = Term.variable(context.x)
    cases = (
        (
            Law(
                "foreign_variable",
                (foreign_variable,),
                Equation(
                    Term.variable(foreign_variable), Term.variable(foreign_variable)
                ),
            ),
            "variable sort",
        ),
        (
            Law(
                "foreign_result",
                (),
                Equation(
                    Term.apply(OperationSymbol("foreign_constant", (), foreign_sort)),
                    Term.apply(OperationSymbol("foreign_constant", (), foreign_sort)),
                ),
            ),
            "equation result sort",
        ),
        (
            Law(
                "foreign_operation",
                (context.x,),
                Equation(
                    Term.apply(foreign_operation, scalar_term),
                    Term.apply(foreign_operation, scalar_term),
                ),
            ),
            "operation symbol",
        ),
    )

    for target, reason in cases:
        with pytest.raises(BasisReductionDefinitionError, match=reason):
            basis_reduction_gate(context.structure, target, context.obligations)
        assert context.calls == [0]


def test_term_node_bounds_are_exact_and_do_not_execute_operations() -> None:
    scalar = Sort("scalar")
    carrier = FiniteCarrier((0,), sort=scalar)
    unary = OperationSymbol("unary", (scalar,), scalar)
    calls = [0]

    def unary_function(value: object) -> int:
        del value
        calls[0] += 1
        return 0

    structure = (
        StructureBuilder(Signature((scalar,), (unary,)))
        .with_carrier(scalar, carrier)
        .with_operation(
            unary,
            Operation.from_callable(unary, ((scalar, carrier),), unary_function),
        )
        .freeze()
    )
    variable = Variable("x", scalar)
    term = Term.variable(variable)
    for _ in range(511):
        term = Term.apply(unary, term)
    accepted = Law("node_512", (variable,), Equation(term, term))
    result = basis_reduction_gate(structure, accepted, ())
    assert type(result) is BasisReductionDenied
    assert calls == [0]
    term = Term.apply(unary, term)
    rejected = Law("node_513", (variable,), Equation(term, term))
    with pytest.raises(BasisReductionDefinitionError, match="node count"):
        basis_reduction_gate(structure, rejected, ())
    assert calls == [0]


def test_issue_order_and_records_are_deterministic_immutable_and_inert() -> None:
    context = _context()
    first = basis_reduction_gate(context.structure, context.target, ())
    second = basis_reduction_gate(context.structure, context.target, ())
    assert type(first) is type(second) is BasisReductionDenied
    assert (
        first.issues
        == second.issues
        == (
            "missing_multilinearity:transport:0",
            "missing_multilinearity:transport:1",
            "missing_closure:transport:vector",
            "missing_multilinearity:product:0",
            "missing_multilinearity:product:1",
            "missing_closure:product:scalar",
            "missing_result_closure:vector",
            "missing_scalar_domain",
            "missing_premise:0",
        )
    )
    obligation = context.obligations[0]
    ready = basis_reduction_gate(context.structure, context.target, context.obligations)
    same_fields = MultilinearityObligation(context.product, (0,), obligation.evidence)
    assert obligation is not same_fields and obligation != same_fields
    assert ready is not basis_reduction_gate(
        context.structure, context.target, context.obligations
    )
    for value in (*context.obligations, ready, first):
        assert not hasattr(value, "__dict__")
        with pytest.raises(TypeError):
            hash(value)
        with pytest.raises((FrozenInstanceError, TypeError)):
            cast(Any, value).is_conditional = False
        assert type(repr(value)) is str
    with pytest.raises(BasisReductionDefinitionError, match="gate-owned"):
        BasisReductionReady()
    with pytest.raises(BasisReductionDefinitionError, match="gate-owned"):
        BasisReductionDenied()
