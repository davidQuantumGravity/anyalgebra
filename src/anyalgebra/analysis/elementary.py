"""Exact elementary analysis for one bounded ``QQ`` multilinear algebra.

The universal :mod:`anyalgebra.structures` layer intentionally permits partial,
many-sorted, and opaque carriers.  This module does not reinterpret any of
those structures as a vector-space algebra.  Point commutator and associator
operations accept a total binary :class:`FiniteMultilinearStructure` whose
existing sparse arithmetic supports the requested exact calculation.  The
separate nucleus and center boundary is the literal canonical ``QQ`` parent,
where multiplication is extended bilinearly from sparse structure constants
and the requested spaces are computed as linear kernels, rather than by
testing which basis vectors happen to pass.
"""

from __future__ import annotations

from dataclasses import dataclass

from anyalgebra.algebra.multilinear import FiniteMultilinearStructure
from anyalgebra.core.domains import DomainElement, QQ, ZZ
from anyalgebra.core.elements import SparseElement
from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.rational import Rational


class ElementaryAnalysisError(AnyAlgebraError, ValueError):
    """An elementary-analysis request lies outside this exact bounded slice."""

    def __init__(self, *, field: str, reason: str) -> None:
        """Keep a deterministic diagnostic free of arbitrary caller rendering."""
        self.field = field
        self.reason = reason
        super().__init__(f"invalid elementary analysis {field}: {reason}")


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class AnalysisBounds:
    """Declared resource bounds for one complete finite kernel calculation."""

    max_dimension: int
    max_constraints: int
    max_elimination_work: int
    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self,
        *,
        max_dimension: int = 32,
        max_constraints: int = 65_536,
        max_elimination_work: int = 1_000_000,
    ) -> None:
        """Require positive built-in limits before any multiplication is read."""
        for field, value in (
            ("max_dimension", max_dimension),
            ("max_constraints", max_constraints),
            ("max_elimination_work", max_elimination_work),
        ):
            if type(value) is not int or value <= 0:
                raise ElementaryAnalysisError(
                    field=field, reason="must be a positive built-in int"
                )
        object.__setattr__(self, "max_dimension", max_dimension)
        object.__setattr__(self, "max_constraints", max_constraints)
        object.__setattr__(self, "max_elimination_work", max_elimination_work)


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class SubspaceReport:
    """A complete exact kernel with its deterministic scalar-equation evidence."""

    basis: tuple[SparseElement, ...]
    basis_indices: tuple[tuple[int, ...], ...]
    dimension: int
    ambient_dimension: int
    expected_constraints: int
    checked_constraints: int
    elimination_work: int
    bounds: AnalysisBounds
    kind: str
    algorithm: str
    algorithm_version: int
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Keep report construction owned by a completed kernel calculation."""
        del args, kwargs
        raise ElementaryAnalysisError(field="report", reason="is analysis-owned")

    @classmethod
    def _create(
        cls,
        basis: tuple[SparseElement, ...],
        expected_constraints: int,
        elimination_work: int,
        bounds: AnalysisBounds,
        kind: str,
        ambient_dimension: int,
    ) -> SubspaceReport:
        """Seal one canonical nullspace basis and its complete work count."""
        value = object.__new__(cls)
        object.__setattr__(value, "basis", basis)
        object.__setattr__(
            value,
            "basis_indices",
            tuple(tuple(element.coordinates()) for element in basis),
        )
        object.__setattr__(value, "dimension", len(basis))
        object.__setattr__(value, "ambient_dimension", ambient_dimension)
        object.__setattr__(value, "expected_constraints", expected_constraints)
        object.__setattr__(value, "checked_constraints", expected_constraints)
        object.__setattr__(value, "elimination_work", elimination_work)
        object.__setattr__(
            value,
            "bounds",
            AnalysisBounds(
                max_dimension=bounds.max_dimension,
                max_constraints=bounds.max_constraints,
                max_elimination_work=bounds.max_elimination_work,
            ),
        )
        object.__setattr__(value, "kind", kind)
        object.__setattr__(value, "algorithm", "anyalgebra.elementary_linear_kernel")
        object.__setattr__(value, "algorithm_version", 1)
        return value

    def __repr__(self) -> str:
        """Expose only stable dimensions and checked-work metadata."""
        return (
            "SubspaceReport("
            f"dimension={self.dimension}, "
            f"ambient_dimension={self.ambient_dimension}, "
            f"expected_constraints={self.expected_constraints}, "
            f"checked_constraints={self.checked_constraints}, "
            f"elimination_work={self.elimination_work}, "
            f"kind={self.kind!r}, "
            f"algorithm={self.algorithm!r}, "
            f"algorithm_version={self.algorithm_version})"
        )


_Vector = tuple[Rational, ...]
_Columns = tuple[_Vector, ...]


def _bounds(value: object) -> AnalysisBounds:
    """Use the default only for ``None`` and reject duck-typed bound records."""
    if value is None:
        return AnalysisBounds()
    if type(value) is not AnalysisBounds:
        raise ElementaryAnalysisError(
            field="options", reason="must be an exact AnalysisBounds or None"
        )
    for field, limit in (
        ("max_dimension", value.max_dimension),
        ("max_constraints", value.max_constraints),
        ("max_elimination_work", value.max_elimination_work),
    ):
        if type(limit) is not int or limit <= 0:
            raise ElementaryAnalysisError(
                field=field, reason="must be a positive built-in int"
            )
    return AnalysisBounds(
        max_dimension=value.max_dimension,
        max_constraints=value.max_constraints,
        max_elimination_work=value.max_elimination_work,
    )


def _algebra(
    value: object, bounds: AnalysisBounds | None, *, require_qq: bool = True
) -> FiniteMultilinearStructure:
    """Preflight the literal total exact multilinear boundary before analysis."""
    if type(value) is not FiniteMultilinearStructure:
        raise ElementaryAnalysisError(
            field="algebra",
            reason="algebra must be an exact FiniteMultilinearStructure",
        )
    if require_qq and value.module.domain is not QQ():
        raise ElementaryAnalysisError(
            field="algebra", reason="coefficient domain must be the literal QQ parent"
        )
    if value.arity != 2:
        raise ElementaryAnalysisError(
            field="algebra", reason="operation arity must be exactly two"
        )
    if bounds is not None and value.module.rank > bounds.max_dimension:
        raise ElementaryAnalysisError(
            field="bounds", reason="declared dimension bound exceeded"
        )
    return value


def _element(
    algebra: FiniteMultilinearStructure, value: object, *, field: str
) -> SparseElement:
    """Require an exact sparse value owned by the literal algebra module."""
    if type(value) is not SparseElement or value.parent is not algebra.module:
        raise ElementaryAnalysisError(
            field=field,
            reason="element must have the literal algebra module parent",
        )
    domain = algebra.module.domain
    if domain is QQ():
        expected_payload_type: type[object] = Rational
    elif domain is ZZ():
        expected_payload_type = int
    else:
        return value
    for coefficient in value.coordinates().values():
        if (
            coefficient.parent is not domain
            or type(coefficient.value) is not expected_payload_type
        ):
            raise ElementaryAnalysisError(
                field=field, reason="element has malformed exact coefficient payload"
            )
    return value


def _multiply_elements(
    algebra: FiniteMultilinearStructure, left: SparseElement, right: SparseElement
) -> SparseElement:
    """Extend a total binary basis table bilinearly using existing module arithmetic."""
    result = algebra.module.zero()
    try:
        for left_index, left_coefficient in left.coordinates().items():
            for right_index, right_coefficient in right.coordinates().items():
                result = result.add(
                    algebra.evaluate_basis(left_index, right_index)
                    .scale(left_coefficient)
                    .scale(right_coefficient)
                )
    except Exception as error:
        raise ElementaryAnalysisError(
            field="algebra", reason="sparse element multiplication extension failed"
        ) from error
    return result


def _zero(rank: int) -> _Vector:
    """Return a fresh exact zero coefficient vector."""
    return tuple(Rational(0) for _ in range(rank))


def _add(left: _Vector, right: _Vector) -> _Vector:
    """Add two already dimension-matched exact rational coordinate vectors."""
    return tuple(a.add(b) for a, b in zip(left, right, strict=True))


def _subtract(left: _Vector, right: _Vector) -> _Vector:
    """Subtract two already dimension-matched exact rational coordinate vectors."""
    return tuple(a.subtract(b) for a, b in zip(left, right, strict=True))


def _scale(vector: _Vector, scalar: Rational) -> _Vector:
    """Scale an exact coordinate vector by one rational coefficient."""
    return tuple(value.multiply(scalar) for value in vector)


def _element_vector(element: SparseElement, rank: int) -> _Vector:
    """Expand canonical sparse QQ coordinates without using arbitrary equality."""
    coordinates = element.coordinates()
    values = [Rational(0) for _ in range(rank)]
    for index, coefficient in coordinates.items():
        if type(coefficient.value) is not Rational:
            raise ElementaryAnalysisError(
                field="algebra", reason="QQ coordinate payload is malformed"
            )
        values[index] = coefficient.value
    return tuple(values)


def _as_element(algebra: FiniteMultilinearStructure, vector: _Vector) -> SparseElement:
    """Reconstruct one canonical module element from exact rational coordinates."""
    return algebra.module.element(
        {index: value for index, value in enumerate(vector) if value.numerator != 0}
    )


def _basis_vector(rank: int, index: int) -> _Vector:
    """Return one canonical coordinate basis vector in declared basis order."""
    return tuple(Rational(1 if position == index else 0) for position in range(rank))


def _product(
    algebra: FiniteMultilinearStructure, left: _Vector, right: _Vector
) -> _Vector:
    """Extend the stored binary basis product bilinearly over exact QQ vectors."""
    result = _zero(algebra.module.rank)
    for left_index, left_coefficient in enumerate(left):
        if left_coefficient.numerator == 0:
            continue
        for right_index, right_coefficient in enumerate(right):
            if right_coefficient.numerator == 0:
                continue
            scalar = left_coefficient.multiply(right_coefficient)
            cell = _zero(algebra.module.rank)
            output_indices: set[int] = set()
            try:
                entries = tuple(
                    algebra.operation.coefficients_for_basis(left_index, right_index)
                )
            except Exception as error:
                raise ElementaryAnalysisError(
                    field="algebra", reason="structure-constant entries are malformed"
                ) from error
            for entry in entries:
                if type(entry) is not tuple or len(entry) != 2:
                    raise ElementaryAnalysisError(
                        field="algebra",
                        reason="structure-constant entries are malformed",
                    )
                output_index, coefficient = entry
                if type(output_index) is not int or not (
                    0 <= output_index < algebra.module.rank
                ):
                    raise ElementaryAnalysisError(
                        field="algebra",
                        reason=(
                            "structure-constant output index must be an exact "
                            "in-range built-in int"
                        ),
                    )
                if output_index in output_indices:
                    raise ElementaryAnalysisError(
                        field="algebra",
                        reason="structure-constant output index is duplicated",
                    )
                output_indices.add(output_index)
                if (
                    not isinstance(coefficient, DomainElement)
                    or coefficient.parent is not QQ()
                    or type(coefficient.value) is not Rational
                ):
                    raise ElementaryAnalysisError(
                        field="algebra",
                        reason="QQ structure-constant payload is malformed",
                    )
                cell = tuple(
                    coefficient.value if position == output_index else value
                    for position, value in enumerate(cell)
                )
            result = _add(result, _scale(cell, scalar))
    return result


def _associator_vector(
    algebra: FiniteMultilinearStructure, left: _Vector, middle: _Vector, right: _Vector
) -> _Vector:
    """Compute ``(left*middle)*right - left*(middle*right)`` exactly."""
    return _subtract(
        _product(algebra, _product(algebra, left, middle), right),
        _product(algebra, left, _product(algebra, middle, right)),
    )


def commutator(
    algebra: FiniteMultilinearStructure, x: SparseElement, y: SparseElement
) -> SparseElement:
    """Return ``x*y - y*x`` for two exact sparse elements of one algebra."""
    checked = _algebra(algebra, None, require_qq=False)
    left = _element(checked, x, field="x")
    right = _element(checked, y, field="y")
    try:
        return _multiply_elements(checked, left, right).subtract(
            _multiply_elements(checked, right, left)
        )
    except ElementaryAnalysisError:
        raise
    except Exception as error:
        raise ElementaryAnalysisError(
            field="algebra", reason="sparse element subtraction failed"
        ) from error


def associator(
    algebra: FiniteMultilinearStructure,
    x: SparseElement,
    y: SparseElement,
    z: SparseElement,
) -> SparseElement:
    """Return ``(x*y)*z - x*(y*z)`` preserving the documented parentheses."""
    checked = _algebra(algebra, None, require_qq=False)
    left = _element(checked, x, field="x")
    middle = _element(checked, y, field="y")
    right = _element(checked, z, field="z")
    try:
        return _multiply_elements(
            checked, _multiply_elements(checked, left, middle), right
        ).subtract(
            _multiply_elements(
                checked, left, _multiply_elements(checked, middle, right)
            )
        )
    except ElementaryAnalysisError:
        raise
    except Exception as error:
        raise ElementaryAnalysisError(
            field="algebra", reason="sparse element subtraction failed"
        ) from error


def _associator_columns(
    algebra: FiniteMultilinearStructure, slot: int
) -> tuple[_Columns, ...]:
    """Build all linear associator constraints for one declared nucleus slot."""
    rank = algebra.module.rank
    basis = tuple(_basis_vector(rank, index) for index in range(rank))
    samples: list[_Columns] = []
    for first in basis:
        for second in basis:
            columns: list[_Vector] = []
            for candidate in basis:
                arguments = (candidate, first, second)
                if slot == 1:
                    arguments = (first, candidate, second)
                elif slot == 2:
                    arguments = (first, second, candidate)
                columns.append(_associator_vector(algebra, *arguments))
            samples.append(tuple(columns))
    return tuple(samples)


def _commutator_columns(algebra: FiniteMultilinearStructure) -> tuple[_Columns, ...]:
    """Build all linear ``[candidate, basis]`` constraints in declared order."""
    rank = algebra.module.rank
    basis = tuple(_basis_vector(rank, index) for index in range(rank))
    samples: list[_Columns] = []
    for fixed in basis:
        samples.append(
            tuple(
                _subtract(
                    _product(algebra, candidate, fixed),
                    _product(algebra, fixed, candidate),
                )
                for candidate in basis
            )
        )
    return tuple(samples)


def _nullspace(
    algebra: FiniteMultilinearStructure, samples: tuple[_Columns, ...]
) -> tuple[SparseElement, ...]:
    """Return a canonical free-column basis of the stacked exact QQ kernel."""
    rank = algebra.module.rank
    rows = [
        [columns[column][coordinate] for column in range(rank)]
        for columns in samples
        for coordinate in range(rank)
    ]
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(rank):
        candidate = next(
            (row for row in range(pivot_row, len(rows)) if rows[row][column].numerator),
            None,
        )
        if candidate is None:
            continue
        rows[pivot_row], rows[candidate] = rows[candidate], rows[pivot_row]
        divisor = rows[pivot_row][column]
        rows[pivot_row] = [entry.divide(divisor) for entry in rows[pivot_row]]
        for row in range(len(rows)):
            if row == pivot_row or rows[row][column].numerator == 0:
                continue
            factor = rows[row][column]
            rows[row] = [
                entry.subtract(factor.multiply(pivot))
                for entry, pivot in zip(rows[row], rows[pivot_row], strict=True)
            ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(rows):
            break
    pivot_for = {column: row for row, column in enumerate(pivot_columns)}
    basis: list[SparseElement] = []
    for free_column in range(rank):
        if free_column in pivot_for:
            continue
        vector = [Rational(0) for _ in range(rank)]
        vector[free_column] = Rational(1)
        for pivot_column in reversed(pivot_columns):
            vector[pivot_column] = rows[pivot_for[pivot_column]][free_column].negate()
        basis.append(_as_element(algebra, tuple(vector)))
    return tuple(basis)


def _report(
    algebra: object, options: object, *, slots: tuple[int, ...], commute: bool
) -> SubspaceReport:
    """Preflight, bound, exhaust, and seal one requested exact linear kernel."""
    bounds = _bounds(options)
    checked = _algebra(algebra, bounds)
    condition_count = len(slots) * checked.module.rank**2
    if commute:
        condition_count += checked.module.rank
    # Each tested vector condition is expanded into one scalar QQ equation per
    # output coordinate before exact row reduction.
    expected = condition_count * checked.module.rank
    if expected > bounds.max_constraints:
        raise ElementaryAnalysisError(
            field="bounds", reason="declared constraint bound exceeded"
        )
    elimination_work = expected * checked.module.rank * checked.module.rank
    if elimination_work > bounds.max_elimination_work:
        raise ElementaryAnalysisError(
            field="bounds", reason="declared elimination-work bound exceeded"
        )
    kind = (
        "left_nucleus"
        if slots == (0,)
        else "middle_nucleus"
        if slots == (1,)
        else "right_nucleus"
        if slots == (2,)
        else "center"
        if commute
        else "nucleus"
    )
    try:
        samples = tuple(
            sample for slot in slots for sample in _associator_columns(checked, slot)
        )
        if commute:
            samples = (*samples, *_commutator_columns(checked))
        basis = _nullspace(checked, samples)
    except ElementaryAnalysisError:
        raise
    except Exception as error:
        raise ElementaryAnalysisError(
            field="algebra", reason="exact kernel evaluation failed"
        ) from error
    return SubspaceReport._create(
        basis, expected, elimination_work, bounds, kind, checked.module.rank
    )


def left_nucleus(
    algebra: FiniteMultilinearStructure, *, options: AnalysisBounds | None = None
) -> SubspaceReport:
    """Compute the exact left-nucleus kernel ``(x,a,b)=0`` on all basis pairs."""
    return _report(algebra, options, slots=(0,), commute=False)


def middle_nucleus(
    algebra: FiniteMultilinearStructure, *, options: AnalysisBounds | None = None
) -> SubspaceReport:
    """Compute the exact middle-nucleus kernel ``(a,x,b)=0`` on all basis pairs."""
    return _report(algebra, options, slots=(1,), commute=False)


def right_nucleus(
    algebra: FiniteMultilinearStructure, *, options: AnalysisBounds | None = None
) -> SubspaceReport:
    """Compute the exact right-nucleus kernel ``(a,b,x)=0`` on all basis pairs."""
    return _report(algebra, options, slots=(2,), commute=False)


def nucleus(
    algebra: FiniteMultilinearStructure, *, options: AnalysisBounds | None = None
) -> SubspaceReport:
    """Compute the intersection of the three exact nucleus kernels."""
    return _report(algebra, options, slots=(0, 1, 2), commute=False)


def center(
    algebra: FiniteMultilinearStructure, *, options: AnalysisBounds | None = None
) -> SubspaceReport:
    """Compute the nucleus intersected with the exact commutant kernel."""
    return _report(algebra, options, slots=(0, 1, 2), commute=True)
