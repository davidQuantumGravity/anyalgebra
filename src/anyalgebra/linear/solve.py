"""Deterministic exact row reduction over literal ``QQ`` matrices only.

This v0.0 reference implementation deliberately does *not* infer that an
arbitrary entry parent is a field merely because its elements have arithmetic
methods.  It accepts only matrices whose literal entry parent is :func:`QQ`.
Pivot selection is leftmost-column, then topmost-row; every pivot is normalized
and eliminated above and below, yielding exact RREF evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, cast

from anyalgebra.core.domains import QQ, _RationalElement
from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.linear.matrix import Matrix, MatrixSpace

_MAX_ELIMINATION_WORK = 65_536


class ExactSolveError(AnyAlgebraError, ValueError):
    """A QQ-only exact solve request is malformed or exceeds its finite bound."""

    def __init__(
        self,
        reason: str,
        *,
        kind: str | None = None,
        observed: int | None = None,
        maximum: int | None = None,
    ) -> None:
        self.reason, self.kind, self.observed, self.maximum = (
            reason,
            kind,
            observed,
            maximum,
        )
        detail = (
            ""
            if kind is None
            else f"; kind={kind}; observed={observed}; maximum={maximum}"
        )
        super().__init__(f"exact solve rejected: {reason}{detail}")


@dataclass(frozen=True, slots=True, init=False)
class RowReduction:
    """RREF evidence, its rank data, and an ordered exact kernel basis."""

    rref: Matrix
    rank: int
    pivot_columns: tuple[int, ...]
    free_columns: tuple[int, ...]
    solution_space: MatrixSpace
    kernel_basis: tuple[Matrix, ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ExactSolveError("use rref(matrix)")

    @classmethod
    def _create(
        cls,
        rref: Matrix,
        rank: int,
        pivot_columns: tuple[int, ...],
        free_columns: tuple[int, ...],
        solution_space: MatrixSpace,
        kernel_basis: tuple[Matrix, ...],
    ) -> RowReduction:
        if cls is not RowReduction:
            raise ExactSolveError("factory requires exact RowReduction")
        result = object.__new__(cls)
        object.__setattr__(result, "rref", rref)
        object.__setattr__(result, "rank", rank)
        object.__setattr__(result, "pivot_columns", pivot_columns)
        object.__setattr__(result, "free_columns", free_columns)
        object.__setattr__(result, "solution_space", solution_space)
        object.__setattr__(result, "kernel_basis", kernel_basis)
        return result


@dataclass(frozen=True, slots=True, init=False)
class UniqueSolution:
    """A solve report with one exact solution column."""

    reduction: RowReduction
    solution: Matrix
    status: str
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ExactSolveError("use solve(a, b)")

    @classmethod
    def _create(cls, reduction: RowReduction, solution: Matrix) -> UniqueSolution:
        if cls is not UniqueSolution:
            raise ExactSolveError("factory requires exact UniqueSolution")
        result = object.__new__(cls)
        object.__setattr__(result, "reduction", reduction)
        object.__setattr__(result, "solution", solution)
        object.__setattr__(result, "status", "unique")
        return result


@dataclass(frozen=True, slots=True, init=False)
class DependentSolution:
    """A consistent nonunique report retaining a particular solution and kernel."""

    reduction: RowReduction
    particular_solution: Matrix
    kernel_basis: tuple[Matrix, ...]
    status: str
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ExactSolveError("use solve(a, b)")

    @classmethod
    def _create(
        cls,
        reduction: RowReduction,
        particular_solution: Matrix,
        kernel_basis: tuple[Matrix, ...],
    ) -> DependentSolution:
        if cls is not DependentSolution:
            raise ExactSolveError("factory requires exact DependentSolution")
        result = object.__new__(cls)
        object.__setattr__(result, "reduction", reduction)
        object.__setattr__(result, "particular_solution", particular_solution)
        object.__setattr__(result, "kernel_basis", kernel_basis)
        object.__setattr__(result, "status", "dependent")
        return result


@dataclass(frozen=True, slots=True, init=False)
class InconsistentSolution:
    """An inconsistent report retaining the first canonical contradictory row."""

    reduction: RowReduction
    contradictory_row: int
    reduced_augmented_row: Matrix
    status: str
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ExactSolveError("use solve(a, b)")

    @classmethod
    def _create(
        cls,
        reduction: RowReduction,
        contradictory_row: int,
        reduced_augmented_row: Matrix,
    ) -> InconsistentSolution:
        if cls is not InconsistentSolution:
            raise ExactSolveError("factory requires exact InconsistentSolution")
        result = object.__new__(cls)
        object.__setattr__(result, "reduction", reduction)
        object.__setattr__(result, "contradictory_row", contradictory_row)
        object.__setattr__(result, "reduced_augmented_row", reduced_augmented_row)
        object.__setattr__(result, "status", "inconsistent")
        return result


SolveOutcome: TypeAlias = UniqueSolution | DependentSolution | InconsistentSolution


def _qq_matrix(value: object, name: str) -> Matrix:
    if type(value) is not Matrix:
        raise ExactSolveError(f"{name} must be an exact Matrix")
    if value.parent.entry_parent is not QQ():
        raise ExactSolveError(f"{name} must have literal QQ entry parent")
    for row in value.entries:
        for entry in row:
            if type(entry) is not _RationalElement or entry.parent is not QQ():
                raise ExactSolveError(f"{name} has a non-QQ entry")
    return value


def _work(rows: int, columns: int) -> None:
    observed = rows * columns * max(rows, columns)
    if observed > _MAX_ELIMINATION_WORK:
        raise ExactSolveError(
            "elimination exceeds work limit",
            kind="work",
            observed=observed,
            maximum=_MAX_ELIMINATION_WORK,
        )


def _zero(value: _RationalElement) -> bool:
    return value.value.numerator == 0


def _divide(left: _RationalElement, right: _RationalElement) -> _RationalElement:
    return left.divide(right)


def _subtract(left: _RationalElement, right: _RationalElement) -> _RationalElement:
    return left.subtract(right)


def _multiply(left: _RationalElement, right: _RationalElement) -> _RationalElement:
    return left.multiply(right)


def _negate(value: _RationalElement) -> _RationalElement:
    return value.negate()


def _reduce(
    entries: tuple[tuple[_RationalElement, ...], ...], columns: int
) -> tuple[tuple[tuple[_RationalElement, ...], ...], tuple[int, ...]]:
    """Return deterministic RREF rows and pivot columns after prior preflight."""
    rows = [list(row) for row in entries]
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(columns):
        found = next(
            (
                index
                for index in range(pivot_row, len(rows))
                if not _zero(rows[index][column])
            ),
            None,
        )
        if found is None:
            continue
        if found != pivot_row:
            rows[pivot_row], rows[found] = rows[found], rows[pivot_row]
        pivot = rows[pivot_row][column]
        rows[pivot_row] = [_divide(entry, pivot) for entry in rows[pivot_row]]
        for index, row in enumerate(rows):
            if index == pivot_row:
                continue
            factor = row[column]
            if _zero(factor):
                continue
            rows[index] = [
                _subtract(entry, _multiply(factor, pivot_entry))
                for entry, pivot_entry in zip(row, rows[pivot_row], strict=True)
            ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return tuple(tuple(row) for row in rows), tuple(pivot_columns)


def _kernel(
    rows: tuple[tuple[_RationalElement, ...], ...],
    columns: int,
    pivot_columns: tuple[int, ...],
    solution_space: MatrixSpace,
) -> tuple[tuple[int, ...], tuple[Matrix, ...]]:
    pivots = set(pivot_columns)
    free = tuple(column for column in range(columns) if column not in pivots)
    one, zero = QQ().element(1), QQ().element(0)
    basis: list[Matrix] = []
    for free_column in free:
        values = [zero for _ in range(columns)]
        values[free_column] = one
        for row_index, pivot_column in enumerate(pivot_columns):
            values[pivot_column] = _negate(rows[row_index][free_column])
        basis.append(solution_space.element(tuple((value,) for value in values)))
    return free, tuple(basis)


def _report(
    rows: tuple[tuple[_RationalElement, ...], ...],
    rref_space: MatrixSpace,
    pivot_columns: tuple[int, ...],
) -> RowReduction:
    solution_space = MatrixSpace(rref_space.columns, 1, QQ())
    reduced = Matrix._create(rref_space, rows)
    free_columns, kernel_basis = _kernel(
        rows, rref_space.columns, pivot_columns, solution_space
    )
    return RowReduction._create(
        reduced,
        len(pivot_columns),
        pivot_columns,
        free_columns,
        solution_space,
        kernel_basis,
    )


def rref(matrix: object) -> RowReduction:
    """Return exact deterministic RREF evidence for one literal-``QQ`` matrix."""
    value = _qq_matrix(matrix, "matrix")
    _work(value.parent.rows, value.parent.columns)
    entries = cast(
        tuple[tuple[_RationalElement, ...], ...],
        value.entries,
    )
    reduced_rows, pivots = _reduce(entries, value.parent.columns)
    return _report(reduced_rows, value.parent, pivots)


def solve(a: object, b: object) -> SolveOutcome:
    """Classify and witness the literal-``QQ`` system ``A x = b`` for one RHS."""
    coefficients, right_side = _qq_matrix(a, "a"), _qq_matrix(b, "b")
    if coefficients.parent.rows != right_side.parent.rows:
        raise ExactSolveError("a and b must have the same number of rows")
    if right_side.parent.columns != 1:
        raise ExactSolveError("b must have exactly one column")
    augmented_columns = coefficients.parent.columns + 1
    if augmented_columns > 512:
        raise ExactSolveError("augmented system exceeds matrix column bound")
    _work(coefficients.parent.rows, augmented_columns)
    augmented = tuple(
        (*coefficient_row, right_row[0])
        for coefficient_row, right_row in zip(
            coefficients.entries, right_side.entries, strict=True
        )
    )
    exact_augmented = cast(
        tuple[tuple[_RationalElement, ...], ...],
        augmented,
    )
    reduced_rows, all_pivots = _reduce(exact_augmented, augmented_columns)
    coefficient_columns = coefficients.parent.columns
    pivots = tuple(column for column in all_pivots if column < coefficient_columns)
    coefficient_rows = tuple(tuple(row[:coefficient_columns]) for row in reduced_rows)
    report = _report(coefficient_rows, coefficients.parent, pivots)
    contradictory = next(
        (
            index
            for index, row in enumerate(reduced_rows)
            if all(_zero(entry) for entry in row[:coefficient_columns])
            and not _zero(row[coefficient_columns])
        ),
        None,
    )
    if contradictory is not None:
        witness = MatrixSpace(1, augmented_columns, QQ()).element(
            (reduced_rows[contradictory],)
        )
        return InconsistentSolution._create(report, contradictory, witness)
    zero = QQ().element(0)
    values = [zero for _ in range(coefficient_columns)]
    for row_index, pivot_column in enumerate(pivots):
        values[pivot_column] = reduced_rows[row_index][coefficient_columns]
    particular = report.solution_space.element(tuple((value,) for value in values))
    if report.free_columns:
        return DependentSolution._create(report, particular, report.kernel_basis)
    return UniqueSolution._create(report, particular)
