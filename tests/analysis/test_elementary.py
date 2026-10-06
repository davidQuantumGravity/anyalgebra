"""Observable contracts for exact finite elementary algebra analysis."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from anyalgebra.algebra.multilinear import BasisProductTable, FiniteMultilinearStructure
from anyalgebra.analysis import elementary
from anyalgebra.analysis.elementary import (
    AnalysisBounds,
    ElementaryAnalysisError,
    SubspaceReport,
    associator,
    center,
    commutator,
    left_nucleus,
    middle_nucleus,
    nucleus,
    right_nucleus,
)
from anyalgebra.core.domains import QQ, ZZ
from anyalgebra.core.elements import SparseElement
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.core.rational import Rational
from anyalgebra.structures.operations import PartialOperation
from anyalgebra.structures.signatures import OperationSymbol, Signature
from anyalgebra.structures.structure import StructureBuilder


def _algebra(cells: list[dict[int, int]]) -> FiniteMultilinearStructure:
    domain = QQ()
    module = FreeModule(domain, Basis(("e0", "e1"), coefficient_domain=domain))
    table = BasisProductTable.from_cells(
        module,
        2,
        [
            {index: domain.element(value) for index, value in cell.items()}
            for cell in cells
        ],
    )
    return FiniteMultilinearStructure.from_tables(module, table, name="fixture")


def _nonassociative() -> FiniteMultilinearStructure:
    # e0*e0=e1, e1*e0=e1; the remaining basis products vanish.
    return _algebra([{1: 1}, {}, {1: 1}, {}])


def _associative_noncommutative() -> FiniteMultilinearStructure:
    # e0 is a left identity and e1*e0=0.
    return _algebra([{0: 1}, {1: 1}, {}, {}])


def _commutative_associative() -> FiniteMultilinearStructure:
    # Dual numbers in the ordered basis (1, epsilon).
    return _algebra([{0: 1}, {1: 1}, {1: 1}, {}])


def test_commutator_and_associator_extend_exactly_to_sparse_elements() -> None:
    algebra = _nonassociative()
    e0 = algebra.module.element({0: 1})
    e1 = algebra.module.element({1: 1})

    assert commutator(algebra, e0, e1) == algebra.module.element({1: -1})
    assert associator(algebra, e0, e0, e0) == e1
    assert associator(algebra, e0.add(e1), e0, e0) == algebra.module.element({1: 2})


def test_element_products_also_preserve_exact_zz_fixture_controls() -> None:
    domain = ZZ()
    module = FreeModule(domain, Basis(("e0", "e1"), coefficient_domain=domain))
    algebra = FiniteMultilinearStructure.from_tables(
        module,
        BasisProductTable.from_cells(
            module,
            2,
            [
                {1: domain.element(1)},
                {},
                {1: domain.element(1)},
                {},
            ],
        ),
    )
    e0, e1 = module.element({0: 1}), module.element({1: 1})

    assert commutator(algebra, e0, e1) == module.element({1: -1})
    assert associator(algebra, e0, e0, e0) == e1


def test_sparse_point_operations_do_not_inherit_subspace_dimension_bounds() -> None:
    domain = ZZ()
    module = FreeModule(
        domain,
        Basis(tuple(f"e{index}" for index in range(33)), coefficient_domain=domain),
    )
    algebra = FiniteMultilinearStructure.from_tables(
        module, BasisProductTable.from_cells(module, 2, [{} for _ in range(33**2)])
    )
    assert commutator(algebra, module.zero(), module.zero()) == module.zero()


def test_nuclei_are_full_linear_kernels_and_center_intersects_commutants() -> None:
    nonassociative = _nonassociative()
    left = left_nucleus(nonassociative)
    middle = middle_nucleus(nonassociative)
    right = right_nucleus(nonassociative)
    full = nucleus(nonassociative)

    assert (left.dimension, middle.dimension, right.dimension, full.dimension) == (
        1,
        1,
        1,
        0,
    )
    assert left.basis[0] == nonassociative.module.element({0: -1, 1: 1})
    assert middle.basis == (nonassociative.module.element({1: 1}),)
    assert right.basis == (nonassociative.module.element({1: 1}),)

    noncommutative = _associative_noncommutative()
    assert nucleus(noncommutative).dimension == 2
    assert center(noncommutative).dimension == 0

    commutative = _commutative_associative()
    assert center(commutative).basis == (
        commutative.module.element({0: 1}),
        commutative.module.element({1: 1}),
    )


def test_reports_are_sealed_deterministic_and_retain_complete_work_counts() -> None:
    report = center(_commutative_associative())

    # Counts are scalar QQ equations: every vector condition contributes rank rows.
    assert report.checked_constraints == report.expected_constraints == 28
    assert (report.kind, report.ambient_dimension, report.elimination_work) == (
        "center",
        2,
        112,
    )
    assert report.bounds is not AnalysisBounds()
    assert (
        report.bounds.max_dimension,
        report.bounds.max_constraints,
        report.bounds.max_elimination_work,
    ) == (32, 65_536, 1_000_000)
    assert (report.algorithm, report.algorithm_version) == (
        "anyalgebra.elementary_linear_kernel",
        1,
    )
    assert report.basis_indices == ((0,), (1,))
    with pytest.raises(FrozenInstanceError):
        report.dimension = 0  # type: ignore[misc]
    with pytest.raises(TypeError):
        hash(report)
    with pytest.raises(TypeError):
        hash(AnalysisBounds())
    assert "algorithm_version=1" in repr(report)
    with pytest.raises(ElementaryAnalysisError, match="analysis-owned"):
        SubspaceReport()


def test_declared_bounds_are_checked_before_any_kernel_claim() -> None:
    algebra = _nonassociative()
    with pytest.raises(ElementaryAnalysisError) as caught:
        nucleus(algebra, options=AnalysisBounds(max_constraints=7))

    assert caught.value.reason == "declared constraint bound exceeded"


@pytest.mark.parametrize(
    ("field", "tampered"),
    [
        ("max_dimension", 0),
        ("max_constraints", True),
        ("max_elimination_work", "bad"),
    ],
)
def test_tampered_frozen_bounds_are_revalidated_at_public_use(
    field: str, tampered: object
) -> None:
    bounds = AnalysisBounds()
    object.__setattr__(bounds, field, tampered)

    with pytest.raises(ElementaryAnalysisError) as caught:
        nucleus(_nonassociative(), options=bounds)

    assert (caught.value.field, caught.value.reason) == (
        field,
        "must be a positive built-in int",
    )


def test_bounds_are_snapshotted_before_public_kernel_work() -> None:
    supplied = AnalysisBounds(max_constraints=24, max_elimination_work=96)
    checked = elementary._bounds(supplied)

    assert checked is not supplied
    object.__setattr__(supplied, "max_constraints", 1)
    assert nucleus(_nonassociative(), options=checked).checked_constraints == 24


def test_exact_constraint_and_elimination_limits_preflight_before_products(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    algebra = _nonassociative()
    exact = AnalysisBounds(max_constraints=24, max_elimination_work=96)
    assert nucleus(algebra, options=exact).checked_constraints == 24

    monkeypatch.setattr(
        elementary,
        "_associator_columns",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not multiply")),
    )
    with pytest.raises(ElementaryAnalysisError, match="constraint"):
        nucleus(algebra, options=AnalysisBounds(max_constraints=23))
    with pytest.raises(ElementaryAnalysisError, match="elimination-work"):
        nucleus(
            algebra,
            options=AnalysisBounds(max_constraints=24, max_elimination_work=95),
        )


def test_bounds_and_zero_dimensional_algebra_are_explicit() -> None:
    for kwargs in (
        {"max_dimension": 0},
        {"max_constraints": True},
        {"max_elimination_work": "1"},
    ):
        with pytest.raises(ElementaryAnalysisError, match="positive"):
            AnalysisBounds(**kwargs)

    domain = QQ()
    module = FreeModule(domain, Basis((), coefficient_domain=domain))
    algebra = FiniteMultilinearStructure.from_tables(
        module, BasisProductTable.from_cells(module, 2, ())
    )
    report = center(algebra, options=AnalysisBounds(max_dimension=1))
    assert report.basis == ()
    assert report.expected_constraints == report.checked_constraints == 0
    with pytest.raises(ElementaryAnalysisError, match="dimension bound"):
        left_nucleus(_nonassociative(), options=AnalysisBounds(max_dimension=1))
    with pytest.raises(ElementaryAnalysisError, match="exact AnalysisBounds"):
        elementary._bounds(object())


def test_foreign_parent_unsupported_domain_and_nonbinary_arity_are_typed() -> None:
    algebra = _nonassociative()
    foreign_module = FreeModule(QQ(), Basis(("x", "y"), coefficient_domain=QQ()))
    with pytest.raises(ElementaryAnalysisError) as mismatch:
        commutator(algebra, algebra.module.element({}), foreign_module.element({}))
    assert (
        mismatch.value.reason == "element must have the literal algebra module parent"
    )

    integer_domain = ZZ()
    integer_module = FreeModule(
        integer_domain, Basis(("x",), coefficient_domain=integer_domain)
    )
    integer_table = BasisProductTable.from_cells(
        integer_module, 2, [{0: integer_domain.element(1)}]
    )
    integer_algebra = FiniteMultilinearStructure.from_tables(
        integer_module, integer_table
    )
    with pytest.raises(ElementaryAnalysisError) as domain:
        left_nucleus(integer_algebra)
    assert domain.value.reason == "coefficient domain must be the literal QQ parent"

    unary_table = BasisProductTable.from_cells(
        algebra.module,
        1,
        [{0: QQ().element(1)}, {1: QQ().element(1)}],
    )
    unary = FiniteMultilinearStructure.from_tables(algebra.module, unary_table)
    with pytest.raises(ElementaryAnalysisError) as arity:
        left_nucleus(unary)
    assert arity.value.reason == "operation arity must be exactly two"


def test_partial_and_many_sorted_structures_remain_opaque_and_not_totalized() -> None:
    value = Sort("value")
    flag = Sort("flag")
    values = FiniteCarrier((object(),), sort=value)
    flags = FiniteCarrier(("yes",), sort=flag)
    product = OperationSymbol("partial", (value, value), value)
    partial = PartialOperation.from_table(
        product, ((value, values), (flag, flags)), (), undefined_marker=object()
    )
    structure = (
        StructureBuilder(Signature((value, flag), (product,)))
        .with_carrier(value, values)
        .with_carrier(flag, flags)
        .with_operation(product, partial)
        .freeze()
    )

    with pytest.raises(ElementaryAnalysisError) as caught:
        left_nucleus(structure)  # type: ignore[arg-type]
    assert caught.value.reason == "algebra must be an exact FiniteMultilinearStructure"
    assert "0x" not in str(caught.value)


def test_malformed_q_payload_and_subclass_inputs_are_typed_without_repr_leaks() -> None:
    algebra = _nonassociative()
    value = algebra.module.element({0: 1})
    coefficient = next(iter(value.coordinates().values()))
    object.__setattr__(coefficient, "value", "not-a-rational")
    with pytest.raises(ElementaryAnalysisError) as malformed:
        commutator(algebra, value, value)
    assert malformed.value.reason == "element has malformed exact coefficient payload"

    class BoundsSubclass(AnalysisBounds):
        pass

    with pytest.raises(ElementaryAnalysisError) as bad_bounds:
        left_nucleus(algebra, options=cast(AnalysisBounds, BoundsSubclass()))
    assert bad_bounds.value.reason == "must be an exact AnalysisBounds or None"
    with pytest.raises(ElementaryAnalysisError) as bad_algebra:
        left_nucleus(cast(FiniteMultilinearStructure, object()))
    assert (
        bad_algebra.value.reason
        == "algebra must be an exact FiniteMultilinearStructure"
    )
    assert "0x" not in str(bad_algebra.value)


def test_internal_failure_translation_and_kernel_edges_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    algebra = _nonassociative()
    value = algebra.module.element({0: 1})
    coefficient = next(iter(value.coordinates().values()))
    object.__setattr__(coefficient, "value", "bad")
    with pytest.raises(ElementaryAnalysisError, match="multiplication extension"):
        elementary._multiply_elements(algebra, value, value)
    with pytest.raises(ElementaryAnalysisError, match="QQ coordinate"):
        elementary._element_vector(value, algebra.module.rank)
    assert elementary._element_vector(_nonassociative().module.element({0: 1}), 2) == (
        Rational(1),
        Rational(0),
    )

    clean = _nonassociative()
    bad_constant = QQ().element(1)
    object.__setattr__(bad_constant, "value", "bad")
    monkeypatch.setattr(
        type(clean.operation),
        "coefficients_for_basis",
        lambda *_args: ((0, bad_constant),),
    )
    with pytest.raises(ElementaryAnalysisError, match="structure-constant"):
        elementary._product(
            clean, elementary._basis_vector(2, 0), elementary._basis_vector(2, 0)
        )

    monkeypatch.setattr(
        elementary,
        "_multiply_elements",
        lambda *_args: cast(SparseElement, object()),
    )
    with pytest.raises(ElementaryAnalysisError, match="subtraction"):
        commutator(clean, clean.module.element({}), clean.module.element({}))
    with pytest.raises(ElementaryAnalysisError, match="subtraction"):
        associator(
            clean,
            clean.module.element({}),
            clean.module.element({}),
            clean.module.element({}),
        )

    monkeypatch.setattr(
        elementary,
        "_multiply_elements",
        lambda *_args: (_ for _ in ()).throw(
            ElementaryAnalysisError(field="algebra", reason="injected")
        ),
    )
    with pytest.raises(ElementaryAnalysisError, match="injected"):
        commutator(clean, clean.module.element({}), clean.module.element({}))
    with pytest.raises(ElementaryAnalysisError, match="injected"):
        associator(
            clean,
            clean.module.element({}),
            clean.module.element({}),
            clean.module.element({}),
        )

    samples = ((elementary._basis_vector(2, 0), elementary._zero(2)),)
    assert elementary._nullspace(clean, samples) == (clean.module.element({1: 1}),)

    domain = QQ()
    rank_one_module = FreeModule(domain, Basis(("x",), coefficient_domain=domain))
    rank_one = FiniteMultilinearStructure.from_tables(
        rank_one_module,
        BasisProductTable.from_cells(rank_one_module, 2, [{0: domain.element(1)}]),
    )
    assert elementary._nullspace(rank_one, ((elementary._basis_vector(1, 0),),)) == ()

    generic = _nonassociative()
    generic_value = generic.module.element({})
    object.__setattr__(generic.module, "domain", object())
    generic_result = elementary._element(generic, generic_value, field="x")
    assert generic_result.parent is generic.module

    monkeypatch.setattr(
        elementary,
        "_associator_columns",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("hidden")),
    )
    with pytest.raises(ElementaryAnalysisError, match="exact kernel"):
        nucleus(_nonassociative())
    monkeypatch.setattr(
        elementary,
        "_associator_columns",
        lambda *_args: (_ for _ in ()).throw(
            ElementaryAnalysisError(field="algebra", reason="injected")
        ),
    )
    with pytest.raises(ElementaryAnalysisError, match="injected"):
        nucleus(_nonassociative())


@pytest.mark.parametrize(
    ("entries", "reason"),
    [
        (((2, QQ().element(1)),), "exact in-range"),
        (((True, QQ().element(1)),), "exact in-range"),
        (
            ((0, QQ().element(1)), (0, QQ().element(1))),
            "is duplicated",
        ),
    ],
)
def test_tampered_structure_constant_output_coordinates_are_typed(
    monkeypatch: pytest.MonkeyPatch,
    entries: tuple[tuple[object, object], ...],
    reason: str,
) -> None:
    algebra = _nonassociative()
    monkeypatch.setattr(
        type(algebra.operation), "coefficients_for_basis", lambda *_args: entries
    )

    with pytest.raises(ElementaryAnalysisError, match=reason) as caught:
        center(algebra)

    assert caught.value.field == "algebra"


@pytest.mark.parametrize(
    ("returned", "reason"),
    [
        (object(), "entries are malformed"),
        (((0, object()),), "payload is malformed"),
        (((0, QQ().element(1), "extra"),), "entries are malformed"),
    ],
)
def test_malformed_structure_constant_entries_are_typed_and_inert(
    monkeypatch: pytest.MonkeyPatch, returned: object, reason: str
) -> None:
    algebra = _nonassociative()
    monkeypatch.setattr(
        type(algebra.operation), "coefficients_for_basis", lambda *_args: returned
    )

    with pytest.raises(ElementaryAnalysisError, match=reason):
        center(algebra)

    monkeypatch.setattr(
        type(algebra.operation),
        "coefficients_for_basis",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("hidden")),
    )
    with pytest.raises(ElementaryAnalysisError, match="entries are malformed"):
        center(algebra)
