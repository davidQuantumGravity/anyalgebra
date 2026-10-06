"""Exact QQ span evidence and conditional finite closure for typed matrices.

Matrices are treated as vectors by their deterministic row-major flattening.
This intentionally supports rectangular operator matrices as well as columns;
all input matrices must nevertheless have one literal ``MatrixSpace``.
Closure evidence is conditional on a caller-supplied bilinearity declaration.
The implementation never attempts to infer bilinearity from opaque Python code.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TypeAlias, cast

from anyalgebra.core.domains import QQ, _RationalElement
from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.linear.matrix import Matrix, MatrixSpace
from anyalgebra.linear.solve import (
    DependentSolution,
    InconsistentSolution,
    UniqueSolution,
    rref,
    solve,
)

_MAX_GENERATORS = 512
_MAX_CLOSURE_PAIRS = 64
_MAX_CLOSURE_WORK = 65_536
_MAX_FLATTENED_DIMENSION = 512


class SpanError(AnyAlgebraError, ValueError):
    """A span or closure request violates exact typed preconditions or bounds."""

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
        super().__init__(f"exact span rejected: {reason}{detail}")


def _error_name(error: Exception) -> str:
    name = type(error).__name__
    return name if type(name) is str and name.isidentifier() else "exception"


@dataclass(frozen=True, slots=True, init=False)
class UniqueSpan:
    """One coordinate column for an independent declared generator family."""

    coordinates: Matrix
    status: str
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise SpanError("use span_decompose(target, generators)")

    @classmethod
    def _create(cls, coordinates: Matrix) -> UniqueSpan:
        if cls is not UniqueSpan:
            raise SpanError("factory requires exact UniqueSpan")
        result = object.__new__(cls)
        object.__setattr__(result, "coordinates", coordinates)
        object.__setattr__(result, "status", "unique")
        return result


@dataclass(frozen=True, slots=True, init=False)
class DependentSpan:
    """A particular coordinate column plus complete ordered dependency basis."""

    particular_coordinates: Matrix
    dependency_basis: tuple[Matrix, ...]
    status: str
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise SpanError("use span_decompose(target, generators)")

    @classmethod
    def _create(
        cls, particular_coordinates: Matrix, dependency_basis: tuple[Matrix, ...]
    ) -> DependentSpan:
        if cls is not DependentSpan:
            raise SpanError("factory requires exact DependentSpan")
        result = object.__new__(cls)
        object.__setattr__(result, "particular_coordinates", particular_coordinates)
        object.__setattr__(result, "dependency_basis", dependency_basis)
        object.__setattr__(result, "status", "dependent")
        return result


@dataclass(frozen=True, slots=True, init=False)
class OutsideSpan:
    """A rigorous left-nullspace functional separating target from generators."""

    functional: Matrix
    target_pairing: _RationalElement
    status: str
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise SpanError("use span_decompose(target, generators)")

    @classmethod
    def _create(
        cls, functional: Matrix, target_pairing: _RationalElement
    ) -> OutsideSpan:
        if cls is not OutsideSpan:
            raise SpanError("factory requires exact OutsideSpan")
        result = object.__new__(cls)
        object.__setattr__(result, "functional", functional)
        object.__setattr__(result, "target_pairing", target_pairing)
        object.__setattr__(result, "status", "outside")
        return result


SpanResult: TypeAlias = UniqueSpan | DependentSpan | OutsideSpan


@dataclass(frozen=True, slots=True, init=False, eq=False)
class DeclaredBilinearOperation:
    """A caller declaration that an operation is bilinear on one MatrixSpace.

    ``bilinear`` is intentionally retained as supplied.  Closure accepts only
    the literal boolean ``True`` and otherwise produces ``ClosureFailure``.
    This records a conditional assumption; it is never a proof about callable.
    """

    space: MatrixSpace
    operation: Callable[[Matrix, Matrix], object]
    bilinear: object
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise SpanError("use DeclaredBilinearOperation.create")

    def __repr__(self) -> str:
        declaration = self.bilinear if type(self.bilinear) is bool else "<non-bool>"
        return (
            "DeclaredBilinearOperation("
            f"space={self.space!r}, operation=<callable>, bilinear={declaration})"
        )

    @classmethod
    def create(
        cls,
        space: object,
        operation: object,
        *,
        bilinear: object,
    ) -> DeclaredBilinearOperation:
        if cls is not DeclaredBilinearOperation:
            raise SpanError("factory requires exact DeclaredBilinearOperation")
        if type(space) is not MatrixSpace or space.entry_parent is not QQ():
            raise SpanError("declared operation requires a literal QQ MatrixSpace")
        if not callable(operation):
            raise SpanError("declared operation requires a callable")
        result = object.__new__(cls)
        object.__setattr__(result, "space", space)
        object.__setattr__(result, "operation", operation)
        object.__setattr__(result, "bilinear", bilinear)
        return result


@dataclass(frozen=True, slots=True, init=False)
class PairCoordinates:
    """Coordinates for one ordered basis product in a closed declaration."""

    left_index: int
    right_index: int
    coordinates: Matrix
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise SpanError("closure reports construct pair evidence")

    @classmethod
    def _create(
        cls, left_index: int, right_index: int, coordinates: Matrix
    ) -> PairCoordinates:
        if cls is not PairCoordinates:
            raise SpanError("factory requires exact PairCoordinates")
        result = object.__new__(cls)
        object.__setattr__(result, "left_index", left_index)
        object.__setattr__(result, "right_index", right_index)
        object.__setattr__(result, "coordinates", coordinates)
        return result


@dataclass(frozen=True, slots=True, init=False)
class Closed:
    """Conditional closure evidence for all lexicographically ordered basis pairs."""

    basis: tuple[Matrix, ...]
    declaration: DeclaredBilinearOperation
    coordinate_space: MatrixSpace
    decompositions: tuple[PairCoordinates, ...]
    pair_count: int
    status: str
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise SpanError("use closure(generators, declared_operation)")

    @classmethod
    def _create(
        cls,
        basis: tuple[Matrix, ...],
        declaration: DeclaredBilinearOperation,
        coordinate_space: MatrixSpace,
        decompositions: tuple[PairCoordinates, ...],
        pair_count: int,
    ) -> Closed:
        if cls is not Closed:
            raise SpanError("factory requires exact Closed")
        result = object.__new__(cls)
        object.__setattr__(result, "basis", basis)
        object.__setattr__(result, "declaration", declaration)
        object.__setattr__(result, "coordinate_space", coordinate_space)
        object.__setattr__(result, "decompositions", decompositions)
        object.__setattr__(result, "pair_count", pair_count)
        object.__setattr__(result, "status", "closed")
        return result


@dataclass(frozen=True, slots=True, init=False)
class NonClosed:
    """First ordered product whose rigorous witness proves nonmembership."""

    left_index: int
    right_index: int
    basis: tuple[Matrix, ...]
    declaration: DeclaredBilinearOperation
    pair_count: int
    product: Matrix
    witness: OutsideSpan
    status: str
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise SpanError("use closure(generators, declared_operation)")

    @classmethod
    def _create(
        cls,
        left_index: int,
        right_index: int,
        basis: tuple[Matrix, ...],
        declaration: DeclaredBilinearOperation,
        pair_count: int,
        product: Matrix,
        witness: OutsideSpan,
    ) -> NonClosed:
        if cls is not NonClosed:
            raise SpanError("factory requires exact NonClosed")
        result = object.__new__(cls)
        object.__setattr__(result, "left_index", left_index)
        object.__setattr__(result, "right_index", right_index)
        object.__setattr__(result, "basis", basis)
        object.__setattr__(result, "declaration", declaration)
        object.__setattr__(result, "pair_count", pair_count)
        object.__setattr__(result, "product", product)
        object.__setattr__(result, "witness", witness)
        object.__setattr__(result, "status", "nonclosed")
        return result


@dataclass(frozen=True, slots=True, init=False)
class ClosureFailure:
    """A declared operation could not support a closure claim."""

    reason: str
    declaration: DeclaredBilinearOperation | None
    left_index: int | None
    right_index: int | None
    pair_count: int
    kind: str | None
    observed: int | None
    maximum: int | None
    status: str
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise SpanError("use closure(generators, declared_operation)")

    @classmethod
    def _create(
        cls,
        reason: str,
        *,
        declaration: DeclaredBilinearOperation | None = None,
        left_index: int | None = None,
        right_index: int | None = None,
        pair_count: int = 0,
        kind: str | None = None,
        observed: int | None = None,
        maximum: int | None = None,
    ) -> ClosureFailure:
        if cls is not ClosureFailure:
            raise SpanError("factory requires exact ClosureFailure")
        result = object.__new__(cls)
        object.__setattr__(result, "reason", reason)
        object.__setattr__(result, "declaration", declaration)
        object.__setattr__(result, "left_index", left_index)
        object.__setattr__(result, "right_index", right_index)
        object.__setattr__(result, "pair_count", pair_count)
        object.__setattr__(result, "kind", kind)
        object.__setattr__(result, "observed", observed)
        object.__setattr__(result, "maximum", maximum)
        object.__setattr__(result, "status", "failure")
        return result


ClosureReport: TypeAlias = Closed | NonClosed | ClosureFailure


def _snapshot(value: object) -> tuple[object, ...]:
    if isinstance(value, str | bytes | Mapping):
        raise SpanError("generators must be an iterable")
    failure: str | None = None
    try:
        iterator = iter(cast(Iterable[object], value))
    except Exception as error:
        iterator = None
        failure = _error_name(error)
    if failure is not None:
        raise SpanError(f"generators could not be iterated ({failure})")
    assert iterator is not None
    result: list[object] = []
    for _ in range(_MAX_GENERATORS + 1):
        try:
            item = next(iterator)
        except StopIteration:
            break
        except Exception as error:
            failure = _error_name(error)
            break
        result.append(item)
    if failure is not None:
        raise SpanError(f"generators could not be iterated ({failure})")
    if len(result) > _MAX_GENERATORS:
        raise SpanError(
            "generator count exceeds bound",
            kind="generators",
            observed=len(result),
            maximum=_MAX_GENERATORS,
        )
    return tuple(result)


def _matrix(value: object, space: MatrixSpace, name: str) -> Matrix:
    if type(value) is not Matrix:
        raise SpanError(f"{name} must be an exact Matrix")
    if value.parent is not space:
        raise SpanError(f"{name} must have the literal same MatrixSpace")
    if space.entry_parent is not QQ():
        raise SpanError(f"{name} must have literal QQ entry parent")
    for row in value.entries:
        for entry in row:
            if type(entry) is not _RationalElement or entry.parent is not QQ():
                raise SpanError(f"{name} has a non-QQ entry")
    return value


def _flatten(matrix: Matrix) -> tuple[_RationalElement, ...]:
    values = cast(
        tuple[_RationalElement, ...],
        tuple(entry for row in matrix.entries for entry in row),
    )
    if len(values) > _MAX_FLATTENED_DIMENSION:
        raise SpanError(
            "flattened matrix dimension exceeds axis bound",
            kind="dimension",
            observed=len(values),
            maximum=_MAX_FLATTENED_DIMENSION,
        )
    return values


def _system(target: Matrix, generators: tuple[Matrix, ...]) -> tuple[Matrix, Matrix]:
    values = _flatten(target)
    columns = tuple(_flatten(generator) for generator in generators)
    coefficient_space = MatrixSpace(len(values), len(generators), QQ())
    coefficients = coefficient_space.element(
        tuple(tuple(column[row] for column in columns) for row in range(len(values)))
    )
    target_column = MatrixSpace(len(values), 1, QQ()).element(
        tuple((value,) for value in values)
    )
    return coefficients, target_column


def _pair(left: Matrix, right: Matrix) -> _RationalElement:
    total = QQ().element(0)
    for left_entry, right_entry in zip(_flatten(left), _flatten(right), strict=True):
        total = total.add(left_entry.multiply(right_entry))
    return total


def _outside_witness(
    target: Matrix, generators: tuple[Matrix, ...], coefficients: Matrix
) -> OutsideSpan:
    left_kernel = rref(coefficients.transpose()).kernel_basis
    for vector in left_kernel:
        values = tuple(vector.entry(index, 0) for index in range(vector.parent.rows))
        functional = target.parent.element(
            tuple(
                tuple(
                    values[row * target.parent.columns + column]
                    for column in range(target.parent.columns)
                )
                for row in range(target.parent.rows)
            )
        )
        pairing = _pair(functional, target)
        if pairing.value.numerator != 0:
            if any(
                _pair(functional, generator).value.numerator != 0
                for generator in generators
            ):
                raise SpanError("left-nullspace witness failed generator pairing")
            return OutsideSpan._create(functional, pairing)
    raise SpanError("outside-span solve lacked a separating left-nullspace witness")


def _empty_family_result(target: Matrix) -> UniqueSpan | OutsideSpan:
    values = _flatten(target)
    coordinates = MatrixSpace(0, 1, QQ()).zero()
    first_nonzero = next(
        (index for index, value in enumerate(values) if value.value.numerator != 0),
        None,
    )
    if first_nonzero is None:
        return UniqueSpan._create(coordinates)
    one, zero = QQ().element(1), QQ().element(0)
    functional = target.parent.element(
        tuple(
            tuple(
                one if row * target.parent.columns + column == first_nonzero else zero
                for column in range(target.parent.columns)
            )
            for row in range(target.parent.rows)
        )
    )
    return OutsideSpan._create(functional, _pair(functional, target))


def span_decompose(target: object, generators: object) -> SpanResult:
    """Return exact coordinates, dependencies, or a separating Frobenius witness."""
    if type(target) is not Matrix:
        raise SpanError("target must be an exact Matrix")
    space = target.parent
    target_matrix = _matrix(target, space, "target")
    raw_generators = _snapshot(generators)
    family = tuple(
        _matrix(generator, space, f"generator {index}")
        for index, generator in enumerate(raw_generators)
    )
    if not family:
        return _empty_family_result(target_matrix)
    coefficients, target_column = _system(target_matrix, family)
    outcome = solve(coefficients, target_column)
    if type(outcome) is UniqueSolution:
        return UniqueSpan._create(outcome.solution)
    if type(outcome) is DependentSolution:
        return DependentSpan._create(outcome.particular_solution, outcome.kernel_basis)
    assert type(outcome) is InconsistentSolution
    return _outside_witness(target_matrix, family, coefficients)


def span_basis(generators: object) -> tuple[Matrix, ...]:
    """Return original left-to-right pivot generators without reconstruction."""
    raw_generators = _snapshot(generators)
    if not raw_generators:
        return ()
    if type(raw_generators[0]) is not Matrix:
        raise SpanError("generator 0 must be an exact Matrix")
    space = raw_generators[0].parent
    family = tuple(
        _matrix(generator, space, f"generator {index}")
        for index, generator in enumerate(raw_generators)
    )
    _flatten(family[0])
    zero_target = space.zero()
    coefficients, _ = _system(zero_target, family)
    pivots = rref(coefficients).pivot_columns
    return tuple(family[index] for index in pivots)


def closure(generators: object, declared_operation: object) -> ClosureReport:
    """Check declared-bilinear closure, conditional on the declaration only."""
    if type(declared_operation) is not DeclaredBilinearOperation:
        return ClosureFailure._create("operation declaration is malformed")
    if declared_operation.bilinear is not True:
        return ClosureFailure._create(
            "operation declaration is not exact bilinear=True",
            declaration=declared_operation,
        )
    try:
        basis = span_basis(generators)
    except SpanError as error:
        return ClosureFailure._create(
            f"generator preflight failed ({_error_name(error)})",
            declaration=declared_operation,
        )
    space = declared_operation.space
    if basis and basis[0].parent is not space:
        return ClosureFailure._create(
            "operation declaration owns a different MatrixSpace",
            declaration=declared_operation,
        )
    pair_count = len(basis) * len(basis)
    if pair_count > _MAX_CLOSURE_PAIRS:
        return ClosureFailure._create(
            "closure pair bound exceeded; "
            f"observed={pair_count}; maximum={_MAX_CLOSURE_PAIRS}",
            declaration=declared_operation,
            pair_count=pair_count,
            kind="pairs",
            observed=pair_count,
            maximum=_MAX_CLOSURE_PAIRS,
        )
    flattened_dimension = len(_flatten(basis[0])) if basis else 0
    observed_work = (
        pair_count
        * flattened_dimension
        * (len(basis) + 1)
        * max(flattened_dimension, len(basis) + 1)
    )
    if observed_work > _MAX_CLOSURE_WORK:
        return ClosureFailure._create(
            "closure decomposition work exceeds bound",
            declaration=declared_operation,
            pair_count=pair_count,
            kind="work",
            observed=observed_work,
            maximum=_MAX_CLOSURE_WORK,
        )
    coordinate_space = MatrixSpace(len(basis), 1, QQ())
    decompositions: list[PairCoordinates] = []
    for left_index, left in enumerate(basis):
        for right_index, right in enumerate(basis):
            try:
                product = declared_operation.operation(left, right)
            except Exception as error:
                return ClosureFailure._create(
                    f"operation failed ({_error_name(error)})",
                    declaration=declared_operation,
                    left_index=left_index,
                    right_index=right_index,
                    pair_count=pair_count,
                )
            if type(product) is not Matrix or product.parent is not space:
                return ClosureFailure._create(
                    "operation returned a foreign or malformed Matrix",
                    declaration=declared_operation,
                    left_index=left_index,
                    right_index=right_index,
                    pair_count=pair_count,
                )
            try:
                result = span_decompose(product, basis)
            except SpanError as error:
                return ClosureFailure._create(
                    f"product decomposition failed ({_error_name(error)})",
                    declaration=declared_operation,
                    left_index=left_index,
                    right_index=right_index,
                    pair_count=pair_count,
                )
            if type(result) is OutsideSpan:
                return NonClosed._create(
                    left_index,
                    right_index,
                    basis,
                    declared_operation,
                    pair_count,
                    product,
                    result,
                )
            if type(result) is not UniqueSpan:
                return ClosureFailure._create(
                    "independent basis produced a nonunique decomposition",
                    declaration=declared_operation,
                    left_index=left_index,
                    right_index=right_index,
                    pair_count=pair_count,
                    kind="invariant",
                )
            coordinates = result.coordinates
            coordinates = coordinate_space.element(coordinates.entries)
            decompositions.append(
                PairCoordinates._create(left_index, right_index, coordinates)
            )
    return Closed._create(
        basis,
        declared_operation,
        coordinate_space,
        tuple(decompositions),
        pair_count,
    )
