from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from anyalgebra.core.domains import QQ, ZZ
from anyalgebra.linear.matrix import Matrix, MatrixSpace
from anyalgebra.linear.solve import (
    DependentSolution,
    ExactSolveError,
    InconsistentSolution,
    RowReduction,
    UniqueSolution,
    rref,
    solve,
)


def _matrix(rows: tuple[tuple[object, ...], ...]) -> Matrix:
    return MatrixSpace(len(rows), len(rows[0]) if rows else 0, QQ()).element(rows)


def _column(values: tuple[object, ...]) -> Matrix:
    return MatrixSpace(len(values), 1, QQ()).element(
        tuple((value,) for value in values)
    )


def _assert_entries(matrix: Matrix, expected: tuple[tuple[object, ...], ...]) -> None:
    assert matrix.parent.rows == len(expected)
    assert matrix.parent.columns == (
        len(expected[0]) if expected else matrix.parent.columns
    )
    assert tuple(
        tuple(matrix.entry(row, column) for column in range(matrix.parent.columns))
        for row in range(matrix.parent.rows)
    ) == tuple(tuple(QQ().element(value) for value in row) for row in expected)


def test_rref_row_swap_fractions_rank_and_pivot_order() -> None:
    matrix = _matrix(((0, 2, 4), (1, 1, 3)))
    report = rref(matrix)
    assert report.rref.parent is matrix.parent
    _assert_entries(report.rref, ((1, 0, 1), (0, 1, 2)))
    assert report.rank == 2
    assert report.pivot_columns == (0, 1)
    assert report.free_columns == (2,)
    _assert_entries(report.kernel_basis[0], ((-1,), (-2,), (1,)))
    _assert_entries(matrix, ((0, 2, 4), (1, 1, 3)))
    fractional = _matrix(((2, 1),))
    fractional_report = rref(fractional)
    assert fractional_report.rref.parent is fractional.parent
    _assert_entries(fractional_report.rref, ((1, (1, 2)),))


@pytest.mark.parametrize(
    "matrix, expected_rref, rank, pivots, free",
    [
        (_matrix(((0, 0), (0, 0))), ((0, 0), (0, 0)), 0, (), (0, 1)),
        (_matrix(((1, 0), (0, 1))), ((1, 0), (0, 1)), 2, (0, 1), ()),
        (_matrix(((1, 2), (2, 4))), ((1, 2), (0, 0)), 1, (0,), (1,)),
        (MatrixSpace(0, 3, QQ()).element(()), (), 0, (), (0, 1, 2)),
        (MatrixSpace(0, 0, QQ()).element(()), (), 0, (), ()),
        (MatrixSpace(2, 0, QQ()).element(((), ())), ((), ()), 0, (), ()),
    ],
)
def test_rref_rank_and_zero_shape_classification(
    matrix: Matrix,
    expected_rref: tuple[tuple[object, ...], ...],
    rank: int,
    pivots: tuple[int, ...],
    free: tuple[int, ...],
) -> None:
    report = rref(matrix)
    assert (report.rank, report.pivot_columns, report.free_columns) == (
        rank,
        pivots,
        free,
    )
    _assert_entries(report.rref, expected_rref)
    assert all(vector.parent.entry_parent is QQ() for vector in report.kernel_basis)


def test_kernel_vectors_are_independently_expected_and_annihilated() -> None:
    coefficients = _matrix(((1, 2, 3), (0, 1, 1)))
    report = rref(coefficients)
    expected = report.kernel_basis[0]
    _assert_entries(expected, ((-1,), (-1,), (1,)))
    assert expected.parent is report.solution_space
    assert len(report.kernel_basis) == len(report.free_columns)
    _assert_entries(coefficients.matmul(expected), ((0,), (0,)))


def test_solve_unique_over_and_underdetermined_outcomes_with_oracles() -> None:
    unique_a, unique_b = _matrix(((2, 1), (1, -1))), _column((5, 1))
    unique = solve(unique_a, unique_b)
    assert type(unique) is UniqueSolution
    assert unique.status == "unique"
    assert unique.reduction.rref.parent is unique_a.parent
    _assert_entries(unique.solution, ((2,), (1,)))
    _assert_entries(unique_a.matmul(unique.solution), ((5,), (1,)))
    _assert_entries(unique_a, ((2, 1), (1, -1)))
    _assert_entries(unique_b, ((5,), (1,)))

    over_a, over_b = _matrix(((1, 1), (2, 2), (1, -1))), _column((3, 6, 1))
    over = solve(over_a, over_b)
    assert type(over) is UniqueSolution
    _assert_entries(over.solution, ((2,), (1,)))
    _assert_entries(over_a.matmul(over.solution), ((3,), (6,), (1,)))

    under_a, under_b = _matrix(((1, 1, 1),)), _column((3,))
    under = solve(under_a, under_b)
    assert type(under) is DependentSolution
    assert under.status == "dependent"
    _assert_entries(under.particular_solution, ((3,), (0,), (0,)))
    _assert_entries(under_a.matmul(under.particular_solution), ((3,),))
    for vector in under.kernel_basis:
        assert vector.parent is under.reduction.solution_space
        assert under.particular_solution.parent is vector.parent
        _assert_entries(under_a.matmul(vector), ((0,),))
    _assert_entries(
        under.particular_solution.add(under.kernel_basis[0]), ((2,), (1,), (0,))
    )

    singular_a, singular_b = _matrix(((1, 1), (2, 2))), _column((1, 2))
    singular = solve(singular_a, singular_b)
    assert type(singular) is DependentSolution
    _assert_entries(singular.particular_solution, ((1,), (0,)))
    _assert_entries(singular.kernel_basis[0], ((-1,), (1,)))
    _assert_entries(singular_a.matmul(singular.particular_solution), ((1,), (2,)))


def test_solve_inconsistent_and_zero_equation_witnesses() -> None:
    inconsistent = solve(_matrix(((1,), (1,))), _column((0, 1)))
    assert type(inconsistent) is InconsistentSolution
    assert inconsistent.status == "inconsistent"
    assert inconsistent.contradictory_row == 1
    _assert_entries(inconsistent.reduced_augmented_row, ((0, 1),))

    zero_equations = solve(MatrixSpace(0, 2, QQ()).element(()), _column(()))
    assert type(zero_equations) is DependentSolution
    _assert_entries(zero_equations.particular_solution, ((0,), (0,)))
    assert len(zero_equations.kernel_basis) == 2

    contradictory_zero_unknowns = solve(
        MatrixSpace(1, 0, QQ()).element(((),)), _column((1,))
    )
    assert type(contradictory_zero_unknowns) is InconsistentSolution
    _assert_entries(contradictory_zero_unknowns.reduced_augmented_row, ((1,),))

    consistent_zero_unknowns = solve(
        MatrixSpace(1, 0, QQ()).element(((),)), _column((0,))
    )
    assert type(consistent_zero_unknowns) is UniqueSolution
    assert (
        consistent_zero_unknowns.solution.parent
        is consistent_zero_unknowns.reduction.solution_space
    )
    _assert_entries(consistent_zero_unknowns.solution, ())
    zero_by_zero = solve(MatrixSpace(0, 0, QQ()).element(()), _column(()))
    assert type(zero_by_zero) is UniqueSolution
    assert zero_by_zero.solution.parent.rows == 0


def test_input_boundaries_limits_and_determinism() -> None:
    class ForeignValue:
        def __init__(self, parent: object) -> None:
            self.parent = parent

    class CountingNonField:
        def __init__(self) -> None:
            self.element_calls = 0

        def element(self, value: object) -> ForeignValue:
            del value
            self.element_calls += 1
            return ForeignValue(self)

    qq_matrix = _matrix(((1,),))
    with pytest.raises(ExactSolveError, match="exact Matrix"):
        rref(object())
    with pytest.raises(ExactSolveError, match="literal QQ"):
        rref(MatrixSpace(1, 1, ZZ()).element(((1,),)))
    nonfield_parent = CountingNonField()
    nonfield = MatrixSpace(1, 1, nonfield_parent).element((("x",),))
    nonfield_parent.element_calls = 0
    with pytest.raises(ExactSolveError, match="literal QQ"):
        rref(nonfield)
    assert nonfield_parent.element_calls == 0
    malformed = Matrix._create(MatrixSpace(1, 1, QQ()), ((object(),),))
    with pytest.raises(ExactSolveError, match="non-QQ entry"):
        rref(malformed)
    with pytest.raises(ExactSolveError, match="literal QQ"):
        solve(qq_matrix, MatrixSpace(1, 1, ZZ()).element(((1,),)))
    with pytest.raises(ExactSolveError, match="same number"):
        solve(qq_matrix, _column((1, 2)))
    with pytest.raises(ExactSolveError, match="exactly one column"):
        solve(qq_matrix, _matrix(((1, 2),)))
    exact = MatrixSpace(256, 1, QQ()).zero()
    assert rref(exact).rank == 0
    with pytest.raises(ExactSolveError) as overflow:
        rref(MatrixSpace(257, 1, QQ()).zero())
    assert (overflow.value.kind, overflow.value.observed, overflow.value.maximum) == (
        "work",
        66049,
        65536,
    )
    first, second = rref(_matrix(((1, 2),))).rref, rref(_matrix(((1, 2),))).rref
    _assert_entries(first, ((1, 2),))
    _assert_entries(second, ((1, 2),))
    with pytest.raises(ExactSolveError, match="augmented"):
        solve(MatrixSpace(0, 512, QQ()).element(()), _column(()))
    exact_solve = solve(
        MatrixSpace(256, 0, QQ()).element(tuple(() for _ in range(256))),
        _column(tuple(0 for _ in range(256))),
    )
    assert type(exact_solve) is UniqueSolution
    with pytest.raises(ExactSolveError) as solve_overflow:
        solve(
            MatrixSpace(257, 0, QQ()).element(tuple(() for _ in range(257))),
            _column(tuple(0 for _ in range(257))),
        )
    assert (
        solve_overflow.value.kind,
        solve_overflow.value.observed,
        solve_overflow.value.maximum,
    ) == (
        "work",
        66049,
        65536,
    )


def test_reports_are_sealed_frozen_unhashable_and_safe() -> None:
    report = rref(_matrix(((1,),)))
    unique = solve(_matrix(((1,),)), _column((2,)))
    dependent = solve(_matrix(((1, 1),)), _column((2,)))
    inconsistent = solve(_matrix(((1,), (1,))), _column((0, 1)))
    records = (report, unique, dependent, inconsistent)
    for record in records:
        assert not hasattr(record, "__dict__")
        with pytest.raises(TypeError):
            hash(record)
        assert "0x" not in repr(record)
    with pytest.raises(FrozenInstanceError):
        report.rank = 2  # type: ignore[misc]

    class ReductionSubclass(RowReduction):
        pass

    class UniqueSubclass(UniqueSolution):
        pass

    class DependentSubclass(DependentSolution):
        pass

    class InconsistentSubclass(InconsistentSolution):
        pass

    with pytest.raises(ExactSolveError):
        ReductionSubclass._create(report.rref, 1, (0,), (), report.solution_space, ())
    with pytest.raises(ExactSolveError):
        UniqueSubclass._create(report, _column((1,)))
    for record_type in (
        RowReduction,
        UniqueSolution,
        DependentSolution,
        InconsistentSolution,
    ):
        with pytest.raises(ExactSolveError):
            record_type()
    with pytest.raises(ExactSolveError):
        DependentSubclass._create(report, _column((1,)), ())
    with pytest.raises(ExactSolveError):
        InconsistentSubclass._create(report, 0, _matrix(((0, 1),)))
