"""Exact bounded ideal, derivation, and rejection-fingerprint calculations."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd, isqrt

from anyalgebra.algebra.multilinear import (
    FiniteMultilinearStructure,
    MultilinearOperation,
    StructureConstants,
)
from anyalgebra.analysis import elementary
from anyalgebra.analysis.elementary import AnalysisBounds, ElementaryAnalysisError
from anyalgebra.core.domains import QQ, _RationalElement
from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.rational import Rational
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.linear.matrix import Matrix, MatrixSpace


class FingerprintAnalysisError(AnyAlgebraError, ValueError):
    """The bounded exact fingerprint boundary rejected one request."""

    def __init__(self, *, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid fingerprint analysis {field}: {reason}")


_Vector = tuple[Rational, ...]


def _zero(width: int) -> _Vector:
    return tuple(Rational(0) for _ in range(width))


def _basis_vector(width: int, index: int) -> _Vector:
    return tuple(Rational(1 if position == index else 0) for position in range(width))


def _check_bounds(value: object) -> AnalysisBounds:
    """Use Elementary's defensive snapshot without accepting lookalikes."""
    try:
        return elementary._bounds(value)
    except ElementaryAnalysisError as error:
        raise FingerprintAnalysisError(
            field=error.field, reason=error.reason
        ) from error


def _check_algebra(value: object, bounds: AnalysisBounds) -> FiniteMultilinearStructure:
    """Require the same literal total exact algebra boundary as V00-067."""
    try:
        algebra = elementary._algebra(value, bounds)
    except ElementaryAnalysisError as error:
        raise FingerprintAnalysisError(
            field=error.field, reason=error.reason
        ) from error
    except Exception as error:
        raise FingerprintAnalysisError(
            field="algebra", reason="algebra metadata is malformed"
        ) from error
    if (
        type(algebra.module) is not FreeModule
        or type(algebra.module.basis) is not Basis
    ):
        raise FingerprintAnalysisError(field="algebra", reason="module is malformed")
    labels = algebra.module.basis.labels
    if type(labels) is not tuple or any(type(label) is not str for label in labels):
        raise FingerprintAnalysisError(
            field="algebra", reason="basis labels are malformed"
        )
    snapshot = _check_constants(algebra)
    try:
        module = FreeModule(QQ(), Basis(labels, coefficient_domain=QQ()))
        constants = StructureConstants.from_sparse(
            (module.basis, module.basis),
            module.basis,
            tuple((key, QQ().element(value)) for key, value in snapshot),
        )
        return FiniteMultilinearStructure.from_structure_constants(module, constants)
    except Exception as error:
        raise FingerprintAnalysisError(
            field="algebra", reason="validated constants could not be snapshotted"
        ) from error


def _check_constants(
    algebra: FiniteMultilinearStructure,
) -> tuple[tuple[tuple[int, ...], Rational], ...]:
    """Validate all table cells, including cells no calculation happens to query."""
    if type(algebra.operation) is not MultilinearOperation:
        raise FingerprintAnalysisError(field="algebra", reason="operation is malformed")
    constants = algebra.operation.constants
    if (
        type(constants) is not StructureConstants
        or constants.coefficient_domain is not QQ()
        or type(constants.input_bases) is not tuple
        or len(constants.input_bases) != 2
        or any(basis is not algebra.module.basis for basis in constants.input_bases)
        or constants.output_basis is not algebra.module.basis
        or type(constants.entries) is not tuple
    ):
        raise FingerprintAnalysisError(
            field="algebra", reason="constants are malformed"
        )
    previous: tuple[int, ...] | None = None
    snapshot: list[tuple[tuple[int, ...], Rational]] = []
    for entry in constants.entries:
        if type(entry) is not tuple or len(entry) != 2:
            raise FingerprintAnalysisError(
                field="algebra", reason="entries are malformed"
            )
        key, coefficient = entry
        if type(key) is not tuple or len(key) != 3:
            raise FingerprintAnalysisError(field="algebra", reason="key is malformed")
        if any(
            type(index) is not int or not 0 <= index < algebra.module.rank
            for index in key
        ):
            raise FingerprintAnalysisError(
                field="algebra", reason="key is outside range"
            )
        if previous is not None and key <= previous:
            raise FingerprintAnalysisError(
                field="algebra", reason="keys are not strictly sorted and unique"
            )
        previous = key
        if (
            type(coefficient) is not _RationalElement
            or coefficient.parent is not QQ()
            or type(coefficient.value) is not Rational
            or coefficient.value.numerator == 0
        ):
            raise FingerprintAnalysisError(
                field="algebra", reason="payload is malformed"
            )
        payload = coefficient.value
        try:
            numerator, denominator = payload.numerator, payload.denominator
            if (
                type(numerator) is not int
                or type(denominator) is not int
                or denominator <= 0
                or gcd(numerator, denominator) != 1
            ):
                raise ValueError
            snapshot.append((key, Rational(numerator, denominator)))
        except Exception as error:
            raise FingerprintAnalysisError(
                field="algebra", reason="rational payload is malformed"
            ) from error
    return tuple(snapshot)


def _product(
    algebra: FiniteMultilinearStructure, left: _Vector, right: _Vector
) -> _Vector:
    try:
        return elementary._product(algebra, left, right)
    except ElementaryAnalysisError as error:
        raise FingerprintAnalysisError(
            field=error.field, reason=error.reason
        ) from error


def _preflight(*, bounds: AnalysisBounds, constraints: int, work: int) -> None:
    """Reject declared finite work before reading any structure constants."""
    if constraints > bounds.max_constraints:
        raise FingerprintAnalysisError(
            field="bounds", reason="declared constraint bound exceeded"
        )
    if work > bounds.max_elimination_work:
        raise FingerprintAnalysisError(
            field="bounds", reason="declared elimination-work bound exceeded"
        )


def _rref(
    rows: tuple[_Vector, ...], width: int
) -> tuple[tuple[_Vector, ...], tuple[int, ...]]:
    """Compute exact reduced-echelon rows and their pivot columns."""
    reduced = [list(row) for row in rows]
    pivots: list[int] = []
    pivot_row = 0
    for column in range(width):
        candidate = next(
            (
                index
                for index in range(pivot_row, len(reduced))
                if reduced[index][column].numerator
            ),
            None,
        )
        if candidate is None:
            continue
        reduced[pivot_row], reduced[candidate] = (
            reduced[candidate],
            reduced[pivot_row],
        )
        divisor = reduced[pivot_row][column]
        reduced[pivot_row] = [entry.divide(divisor) for entry in reduced[pivot_row]]
        for index, row in enumerate(reduced):
            if index == pivot_row or row[column].numerator == 0:
                continue
            factor = row[column]
            reduced[index] = [
                entry.subtract(factor.multiply(pivot))
                for entry, pivot in zip(row, reduced[pivot_row], strict=True)
            ]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == len(reduced):
            break
    return tuple(tuple(row) for row in reduced), tuple(pivots)


def _kernel(rows: tuple[_Vector, ...], width: int) -> tuple[_Vector, ...]:
    """Return the canonical free-coordinate basis of an exact homogeneous kernel."""
    reduced, pivots = _rref(rows, width)
    locations = {column: row for row, column in enumerate(pivots)}
    answer: list[_Vector] = []
    for free in range(width):
        if free in locations:
            continue
        vector = [Rational(0) for _ in range(width)]
        vector[free] = Rational(1)
        for pivot in reversed(pivots):
            vector[pivot] = reduced[locations[pivot]][free].negate()
        answer.append(tuple(vector))
    return tuple(answer)


def _span(vectors: tuple[_Vector, ...]) -> tuple[_Vector, ...]:
    """Return the canonical RREF basis of a coordinate-vector span."""
    if not vectors:
        return ()
    reduced, _ = _rref(vectors, len(vectors[0]))
    return tuple(row for row in reduced if any(entry.numerator != 0 for entry in row))


def _ambient(rank: int) -> tuple[_Vector, ...]:
    """Return the declared coordinate basis once."""
    return tuple(_basis_vector(rank, index) for index in range(rank))


def _annihilator_basis(
    algebra: FiniteMultilinearStructure,
) -> tuple[tuple[_Vector, ...], int]:
    """Return the two-sided annihilator and exact scalar-equation count."""
    rank = algebra.module.rank
    ambient = _ambient(rank)
    columns = tuple(
        tuple(_product(algebra, candidate, fixed) for candidate in ambient)
        for fixed in ambient
    ) + tuple(
        tuple(_product(algebra, fixed, candidate) for candidate in ambient)
        for fixed in ambient
    )
    rows = tuple(
        tuple(column[candidate][coordinate] for candidate in range(rank))
        for column in columns
        for coordinate in range(rank)
    )
    return _kernel(rows, rank), 2 * rank * rank


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class IdealData:
    """One explicitly constructed two-sided ideal, represented by a QQ basis."""

    origins: tuple[str, ...]
    basis: tuple[_Vector, ...]
    dimension: int
    ambient_dimension: int
    closed: bool
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise FingerprintAnalysisError(field="ideal", reason="is analysis-owned")

    @classmethod
    def _create(
        cls,
        origins: tuple[str, ...],
        basis: tuple[_Vector, ...],
        ambient_dimension: int,
    ) -> IdealData:
        if cls is not IdealData:
            raise FingerprintAnalysisError(
                field="ideal", reason="factory requires exact IdealData"
            )
        value = object.__new__(cls)
        object.__setattr__(value, "origins", origins)
        object.__setattr__(value, "basis", basis)
        object.__setattr__(value, "dimension", len(basis))
        object.__setattr__(value, "ambient_dimension", ambient_dimension)
        object.__setattr__(value, "closed", True)
        return value

    def __repr__(self) -> str:
        return (
            "IdealData("
            f"origins={self.origins!r}, dimension={self.dimension}, "
            f"ambient_dimension={self.ambient_dimension}, closed=True)"
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class IdealSearchResult:
    """Finite canonical discovery; this does not classify all QQ ideals."""

    ideals: tuple[IdealData, ...]
    complete: bool
    status: str
    bounds: AnalysisBounds
    evaluations: int
    algorithm: str
    algorithm_version: int
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise FingerprintAnalysisError(field="ideals", reason="is analysis-owned")

    @classmethod
    def _create(
        cls, ideals: tuple[IdealData, ...], bounds: AnalysisBounds, evaluations: int
    ) -> IdealSearchResult:
        if cls is not IdealSearchResult:
            raise FingerprintAnalysisError(
                field="ideals", reason="factory requires exact IdealSearchResult"
            )
        value = object.__new__(cls)
        for field, item in (
            ("ideals", ideals),
            ("complete", False),
            ("status", "incomplete_canonical_discovery"),
            (
                "bounds",
                AnalysisBounds(
                    max_dimension=bounds.max_dimension,
                    max_constraints=bounds.max_constraints,
                    max_elimination_work=bounds.max_elimination_work,
                ),
            ),
            # This is an exact count of calls to the binary product table,
            # not a count of QQ subspaces.
            ("evaluations", evaluations),
            ("algorithm", "anyalgebra.canonical_ideal_discovery"),
            ("algorithm_version", 1),
        ):
            object.__setattr__(value, field, item)
        return value

    def __repr__(self) -> str:
        return (
            "IdealSearchResult("
            f"ideal_count={len(self.ideals)}, complete={self.complete}, "
            f"status={self.status!r}, evaluations={self.evaluations}, "
            f"algorithm_version={self.algorithm_version})"
        )


def ideals(
    algebra: FiniteMultilinearStructure, *, options: AnalysisBounds | None = None
) -> IdealSearchResult:
    """Discover canonical actual ideals without QQ-subspace-enumeration claims."""
    bounds = _check_bounds(options)
    checked = _check_algebra(algebra, bounds)
    rank = checked.module.rank
    evaluations = 3 * rank * rank
    _preflight(
        bounds=bounds,
        constraints=evaluations * max(rank, 1),
        work=evaluations * max(rank * rank, 1),
    )
    try:
        ambient = _ambient(rank)
        product_span = _span(
            tuple(
                _product(checked, left, right) for left in ambient for right in ambient
            )
        )
        actual_evaluations = rank * rank
        annihilator, annihilator_evaluations = _annihilator_basis(checked)
        actual_evaluations += annihilator_evaluations
    except FingerprintAnalysisError:
        raise
    except Exception as error:
        raise FingerprintAnalysisError(
            field="algebra", reason="canonical ideal evaluation failed"
        ) from error
    # De-duplicate actual subspaces but preserve every canonical construction
    # origin, including the zero-algebra coincidences.
    candidates = (
        ("zero", ()),
        ("product_span_two_sided", product_span),
        ("two_sided_annihilator", annihilator),
        ("whole", ambient),
    )
    grouped: dict[tuple[_Vector, ...], list[str]] = {}
    for origin, basis in candidates:
        grouped.setdefault(_span(basis), []).append(origin)
    return IdealSearchResult._create(
        tuple(
            IdealData._create(tuple(origins), basis, rank)
            for basis, origins in grouped.items()
        ),
        bounds,
        actual_evaluations,
    )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class DerivationReport:
    """The exact QQ solution space of the Leibniz derivation equation."""

    basis: tuple[Matrix, ...]
    matrix_space: MatrixSpace
    dimension: int
    constraints: int
    elimination_work: int
    bounds: AnalysisBounds
    convention: str
    algorithm: str
    algorithm_version: int
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise FingerprintAnalysisError(field="derivations", reason="is analysis-owned")

    @classmethod
    def _create(
        cls,
        basis: tuple[Matrix, ...],
        matrix_space: MatrixSpace,
        constraints: int,
        work: int,
        bounds: AnalysisBounds,
    ) -> DerivationReport:
        if cls is not DerivationReport:
            raise FingerprintAnalysisError(
                field="derivations", reason="factory requires exact DerivationReport"
            )
        value = object.__new__(cls)
        for field, item in (
            ("basis", basis),
            ("matrix_space", matrix_space),
            ("dimension", len(basis)),
            ("constraints", constraints),
            ("elimination_work", work),
            (
                "bounds",
                AnalysisBounds(
                    max_dimension=bounds.max_dimension,
                    max_constraints=bounds.max_constraints,
                    max_elimination_work=bounds.max_elimination_work,
                ),
            ),
            (
                "convention",
                "D(e_j)=sum_a D[a,j] e_a; variables flatten row-major a*n+j; "
                "matrix columns are inputs",
            ),
            ("algorithm", "anyalgebra.derivation_linear_kernel"),
            ("algorithm_version", 1),
        ):
            object.__setattr__(value, field, item)
        return value

    def __repr__(self) -> str:
        return (
            "DerivationReport("
            f"dimension={self.dimension}, constraints={self.constraints}, "
            f"elimination_work={self.elimination_work}, "
            f"algorithm_version={self.algorithm_version})"
        )


def _derivation_rows(algebra: FiniteMultilinearStructure) -> tuple[_Vector, ...]:
    """Build coefficient rows for the declared column-major derivation convention."""
    rank = algebra.module.rank
    ambient = _ambient(rank)
    rows: list[_Vector] = []
    for left_index, left in enumerate(ambient):
        for right_index, right in enumerate(ambient):
            product = _product(algebra, left, right)
            columns: list[_Vector] = []
            for output in range(rank):
                image = _basis_vector(rank, output)
                for input_index in range(rank):
                    # D(e_j)=sum_a D[a,j]e_a.  In D(e_i*e_r), the variable
                    # D[a,j] contributes c[i,r,j] in output coordinate a.
                    first = tuple(
                        product[input_index] if coordinate == output else Rational(0)
                        for coordinate in range(rank)
                    )
                    second = (
                        _product(algebra, image, right)
                        if input_index == left_index
                        else _zero(rank)
                    )
                    third = (
                        _product(algebra, left, image)
                        if input_index == right_index
                        else _zero(rank)
                    )
                    columns.append(
                        tuple(
                            first[coordinate]
                            .subtract(second[coordinate])
                            .subtract(third[coordinate])
                            for coordinate in range(rank)
                        )
                    )
            rows.extend(
                tuple(column[coordinate] for column in columns)
                for coordinate in range(rank)
            )
    return tuple(rows)


def _matrix_from_vector(vector: _Vector, rank: int, space: MatrixSpace) -> Matrix:
    """Construct one explicit QQ matrix from one column-major solution vector."""
    return space.element(
        tuple(
            tuple(vector[output * rank + input_index] for input_index in range(rank))
            for output in range(rank)
        )
    )


def derivation_algebra(
    algebra: FiniteMultilinearStructure, *, options: AnalysisBounds | None = None
) -> DerivationReport:
    """Compute all derivations by exact solution of finite Leibniz equations."""
    bounds = _check_bounds(options)
    checked = _check_algebra(algebra, bounds)
    rank = checked.module.rank
    constraints = rank**3
    work = constraints * (rank * rank) * (rank * rank)
    _preflight(bounds=bounds, constraints=constraints, work=work)
    try:
        vectors = _kernel(_derivation_rows(checked), rank * rank)
        space = MatrixSpace.from_shape(rank, rank, checked.module.domain)
        matrices = tuple(_matrix_from_vector(vector, rank, space) for vector in vectors)
    except FingerprintAnalysisError:
        raise
    except Exception as error:
        raise FingerprintAnalysisError(
            field="derivations", reason="exact derivation kernel evaluation failed"
        ) from error
    return DerivationReport._create(matrices, space, constraints, work, bounds)


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class LawReport:
    """Complete basis reduction evidence or the first deterministic counterexample."""

    kind: str
    status: str
    expected_assignments: int
    checked_assignments: int
    witness_indices: tuple[int, ...] | None
    witness_value: _Vector | None
    reduction_hypothesis: str
    reduction_complete: bool
    algorithm: str
    algorithm_version: int
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise FingerprintAnalysisError(field="law", reason="is analysis-owned")

    @classmethod
    def _create(
        cls,
        kind: str,
        expected: int,
        checked: int,
        witness_indices: tuple[int, ...] | None,
        witness_value: _Vector | None,
    ) -> LawReport:
        if cls is not LawReport:
            raise FingerprintAnalysisError(
                field="law", reason="factory requires exact LawReport"
            )
        arity = 2 if kind == "commutativity" else 3 if kind == "associativity" else 0
        if (
            arity == 0
            or type(expected) is not int
            or type(checked) is not int
            or expected < 0
            or not 0 <= checked <= expected
        ):
            raise FingerprintAnalysisError(
                field="law", reason="evidence metadata is malformed"
            )
        rank = (
            isqrt(expected)
            if arity == 2
            else next(
                (value for value in range(expected + 1) if value**3 == expected), -1
            )
        )
        if rank < 0 or rank**arity != expected:
            raise FingerprintAnalysisError(
                field="law", reason="expected count is malformed"
            )
        if witness_indices is None or witness_value is None:
            if (
                witness_indices is not None
                or witness_value is not None
                or checked != expected
            ):
                raise FingerprintAnalysisError(
                    field="law", reason="proved evidence is malformed"
                )
        else:
            if (
                type(witness_indices) is not tuple
                or len(witness_indices) != arity
                or any(
                    type(index) is not int or not 0 <= index < rank
                    for index in witness_indices
                )
                or type(witness_value) is not tuple
                or len(witness_value) != rank
                or any(type(value) is not Rational for value in witness_value)
                or not any(value.numerator for value in witness_value)
            ):
                raise FingerprintAnalysisError(
                    field="law", reason="counterexample is malformed"
                )
            flat = 0
            for index in witness_indices:
                flat = flat * rank + index
            if checked != flat + 1:
                raise FingerprintAnalysisError(
                    field="law", reason="counterexample order is malformed"
                )
        value = object.__new__(cls)
        for field, item in (
            ("kind", kind),
            ("status", "Disproved" if witness_indices is not None else "Proved"),
            ("expected_assignments", expected),
            ("checked_assignments", checked),
            ("witness_indices", witness_indices),
            ("witness_value", witness_value),
            (
                "reduction_hypothesis",
                "operation is the exact bilinear "
                "FiniteMultilinearStructure table over QQ",
            ),
            ("reduction_complete", True),
            ("algorithm", "anyalgebra.finite_multilinearity_basis_reduction"),
            ("algorithm_version", 1),
        ):
            object.__setattr__(value, field, item)
        return value

    def __repr__(self) -> str:
        return (
            "LawReport("
            f"kind={self.kind!r}, status={self.status!r}, "
            f"checked_assignments={self.checked_assignments})"
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class UnitReport:
    """Exact two-sided-unit solve evidence with a reduced contradictory row."""

    status: str
    scalar_equations: int
    unit_vector: _Vector | None
    witness_row: tuple[Rational, ...] | None
    algorithm: str
    algorithm_version: int
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise FingerprintAnalysisError(field="unit", reason="is analysis-owned")

    @classmethod
    def _create(
        cls,
        status: str,
        scalar_equations: int,
        unit_vector: _Vector | None,
        witness_row: tuple[Rational, ...] | None,
    ) -> UnitReport:
        if cls is not UnitReport:
            raise FingerprintAnalysisError(
                field="unit", reason="factory requires exact UnitReport"
            )
        if type(scalar_equations) is not int or scalar_equations < 0:
            raise FingerprintAnalysisError(
                field="unit", reason="evidence metadata is malformed"
            )
        half = scalar_equations // 2
        rank = isqrt(half)
        if scalar_equations != 2 * rank * rank or status not in ("present", "absent"):
            raise FingerprintAnalysisError(
                field="unit", reason="evidence metadata is malformed"
            )
        if status == "present":
            if (
                type(unit_vector) is not tuple
                or len(unit_vector) != rank
                or any(type(value) is not Rational for value in unit_vector)
                or witness_row is not None
            ):
                raise FingerprintAnalysisError(
                    field="unit", reason="present evidence is malformed"
                )
        elif (
            rank == 0
            or unit_vector is not None
            or type(witness_row) is not tuple
            or len(witness_row) != rank + 1
            or any(type(value) is not Rational for value in witness_row)
            or any(value.numerator for value in witness_row[:rank])
            or witness_row[rank].numerator == 0
        ):
            raise FingerprintAnalysisError(
                field="unit", reason="absent evidence is malformed"
            )
        value = object.__new__(cls)
        for field, item in (
            ("status", status),
            ("scalar_equations", scalar_equations),
            ("unit_vector", unit_vector),
            ("witness_row", witness_row),
            ("algorithm", "anyalgebra.exact_affine_unit_solve"),
            ("algorithm_version", 1),
        ):
            object.__setattr__(value, field, item)
        return value

    def __repr__(self) -> str:
        return (
            "UnitReport("
            f"status={self.status!r}, scalar_equations={self.scalar_equations})"
        )


def _law_data(
    algebra: FiniteMultilinearStructure,
) -> tuple[LawReport, LawReport, int, int, int]:
    """Exhaustively test laws and compute the three annihilator dimensions."""
    rank = algebra.module.rank
    ambient = _ambient(rank)
    left_rows: list[_Vector] = []
    right_rows: list[_Vector] = []
    for fixed in ambient:
        left_columns = tuple(
            _product(algebra, candidate, fixed) for candidate in ambient
        )
        right_columns = tuple(
            _product(algebra, fixed, candidate) for candidate in ambient
        )
        left_rows.extend(
            tuple(column[index] for column in left_columns) for index in range(rank)
        )
        right_rows.extend(
            tuple(column[index] for column in right_columns) for index in range(rank)
        )
    commutation: LawReport | None = None
    for first_index, first in enumerate(ambient):
        for second_index, second in enumerate(ambient):
            difference = tuple(
                left.subtract(right)
                for left, right in zip(
                    _product(algebra, first, second),
                    _product(algebra, second, first),
                    strict=True,
                )
            )
            if any(value.numerator for value in difference):
                commutation = LawReport._create(
                    "commutativity",
                    rank**2,
                    first_index * rank + second_index + 1,
                    (first_index, second_index),
                    difference,
                )
                break
        if commutation is not None:
            break
    associativity: LawReport | None = None
    for first_index, first in enumerate(ambient):
        for second_index, second in enumerate(ambient):
            for third_index, third in enumerate(ambient):
                difference = tuple(
                    left.subtract(right)
                    for left, right in zip(
                        _product(algebra, _product(algebra, first, second), third),
                        _product(algebra, first, _product(algebra, second, third)),
                        strict=True,
                    )
                )
                if any(value.numerator for value in difference):
                    associativity = LawReport._create(
                        "associativity",
                        rank**3,
                        first_index * rank * rank
                        + second_index * rank
                        + third_index
                        + 1,
                        (first_index, second_index, third_index),
                        difference,
                    )
                    break
            if associativity is not None:
                break
        if associativity is not None:
            break
    two_sided, _ = _annihilator_basis(algebra)
    return (
        commutation
        if commutation is not None
        else LawReport._create("commutativity", rank**2, rank**2, None, None),
        associativity
        if associativity is not None
        else LawReport._create("associativity", rank**3, rank**3, None, None),
        len(_kernel(tuple(left_rows), rank)),
        len(_kernel(tuple(right_rows), rank)),
        len(two_sided),
    )


def _unit_report(algebra: FiniteMultilinearStructure) -> UnitReport:
    """Solve both finite identity systems; rank zero has the vacuous zero unit."""
    rank = algebra.module.rank
    ambient = _ambient(rank)
    rows: list[_Vector] = []
    right: list[Rational] = []
    for fixed_index, fixed in enumerate(ambient):
        left_columns = tuple(
            _product(algebra, candidate, fixed) for candidate in ambient
        )
        right_columns = tuple(
            _product(algebra, fixed, candidate) for candidate in ambient
        )
        for coordinate in range(rank):
            rows.append(tuple(column[coordinate] for column in left_columns))
            right.append(Rational(1 if coordinate == fixed_index else 0))
            rows.append(tuple(column[coordinate] for column in right_columns))
            right.append(Rational(1 if coordinate == fixed_index else 0))
    augmented = tuple(
        tuple((*row, value)) for row, value in zip(rows, right, strict=True)
    )
    reduced, pivots = _rref(augmented, rank)
    witness = next(
        (
            row
            for row in reduced
            if all(value.numerator == 0 for value in row[:rank])
            and row[rank].numerator != 0
        ),
        None,
    )
    if witness is not None:
        return UnitReport._create("absent", 2 * rank * rank, None, witness)
    if rank and len(pivots) != rank:
        raise FingerprintAnalysisError(
            field="unit", reason="consistent unit equations were unexpectedly nonunique"
        )
    locations = {column: row for row, column in enumerate(pivots)}
    vector = tuple(
        reduced[locations[index]][rank] if index in locations else Rational(0)
        for index in range(rank)
    )
    return UnitReport._create("present", 2 * rank * rank, vector, None)


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class AlgebraFingerprint:
    """Exact rejection data.  Equal data never establishes an isomorphism."""

    dimension: int
    unit_status: str
    commutative: bool
    associative: bool
    commutativity_report: LawReport
    associativity_report: LawReport
    unit_report: UnitReport
    commutativity_pairs_checked: int
    associativity_triples_checked: int
    unit_scalar_equations_checked: int
    left_nucleus_dimension: int
    middle_nucleus_dimension: int
    right_nucleus_dimension: int
    nucleus_dimension: int
    center_dimension: int
    left_annihilator_dimension: int
    right_annihilator_dimension: int
    two_sided_annihilator_dimension: int
    product_span_dimension: int
    product_span_ideal_dimension: int
    derivation_dimension: int
    ideal_classification_status: str
    conventions: tuple[str, ...]
    bounds: AnalysisBounds
    isomorphism_rejection_only: bool
    algorithm: str
    algorithm_version: int
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise FingerprintAnalysisError(field="fingerprint", reason="is analysis-owned")

    @classmethod
    def _create(
        cls,
        *,
        algebra: FiniteMultilinearStructure,
        bounds: AnalysisBounds,
        unit: UnitReport,
        commutative: LawReport,
        associative: LawReport,
        annihilators: tuple[int, int, int],
        ideal_data: IdealSearchResult,
        derivations: DerivationReport,
    ) -> AlgebraFingerprint:
        if cls is not AlgebraFingerprint:
            raise FingerprintAnalysisError(
                field="fingerprint", reason="factory requires exact AlgebraFingerprint"
            )
        rank = algebra.module.rank
        product_span = _span(
            tuple(
                _product(algebra, _basis_vector(rank, left), _basis_vector(rank, right))
                for left in range(rank)
                for right in range(rank)
            )
        )
        product_ideal = next(
            record
            for record in ideal_data.ideals
            if "product_span_two_sided" in record.origins
        )
        value = object.__new__(cls)
        for field, item in (
            ("dimension", rank),
            ("unit_status", unit.status),
            ("commutative", commutative.status == "Proved"),
            ("associative", associative.status == "Proved"),
            ("commutativity_report", commutative),
            ("associativity_report", associative),
            ("unit_report", unit),
            ("commutativity_pairs_checked", commutative.checked_assignments),
            ("associativity_triples_checked", associative.checked_assignments),
            ("unit_scalar_equations_checked", unit.scalar_equations),
            (
                "left_nucleus_dimension",
                elementary.left_nucleus(algebra, options=bounds).dimension,
            ),
            (
                "middle_nucleus_dimension",
                elementary.middle_nucleus(algebra, options=bounds).dimension,
            ),
            (
                "right_nucleus_dimension",
                elementary.right_nucleus(algebra, options=bounds).dimension,
            ),
            (
                "nucleus_dimension",
                elementary.nucleus(algebra, options=bounds).dimension,
            ),
            ("center_dimension", elementary.center(algebra, options=bounds).dimension),
            ("left_annihilator_dimension", annihilators[0]),
            ("right_annihilator_dimension", annihilators[1]),
            ("two_sided_annihilator_dimension", annihilators[2]),
            ("product_span_dimension", len(product_span)),
            ("product_span_ideal_dimension", product_ideal.dimension),
            ("derivation_dimension", derivations.dimension),
            ("ideal_classification_status", ideal_data.status),
            (
                "conventions",
                (
                    "QQ exact finite-dimensional total binary algebra",
                    "commutator [x,y]=xy-yx",
                    "associator (x,y,z)=(xy)z-x(yz)",
                    "derivation matrices use columns D(declared basis input)",
                    "canonical ideal discovery is incomplete over QQ",
                ),
            ),
            (
                "bounds",
                AnalysisBounds(
                    max_dimension=bounds.max_dimension,
                    max_constraints=bounds.max_constraints,
                    max_elimination_work=bounds.max_elimination_work,
                ),
            ),
            ("isomorphism_rejection_only", True),
            ("algorithm", "anyalgebra.algebra_rejection_fingerprint"),
            ("algorithm_version", 1),
        ):
            object.__setattr__(value, field, item)
        return value

    def __eq__(self, other: object) -> bool:
        """Compare only scalar conventions and transport-invariant rejection data."""
        if type(other) is not AlgebraFingerprint:
            return False
        fields = (
            "dimension",
            "unit_status",
            "commutative",
            "associative",
            "left_nucleus_dimension",
            "middle_nucleus_dimension",
            "right_nucleus_dimension",
            "nucleus_dimension",
            "center_dimension",
            "left_annihilator_dimension",
            "right_annihilator_dimension",
            "two_sided_annihilator_dimension",
            "product_span_dimension",
            "product_span_ideal_dimension",
            "derivation_dimension",
        )
        # Bounds are evidence metadata, not intrinsic algebra data: two
        # complete computations with different sufficient bounds compare by
        # their invariant payload rather than their resource declarations.
        return all(getattr(self, field) == getattr(other, field) for field in fields)

    def __repr__(self) -> str:
        return (
            "AlgebraFingerprint("
            f"dimension={self.dimension}, unit_status={self.unit_status!r}, "
            f"derivation_dimension={self.derivation_dimension}, "
            f"algorithm_version={self.algorithm_version})"
        )


def algebra_fingerprint(
    algebra: FiniteMultilinearStructure, *, options: AnalysisBounds | None = None
) -> AlgebraFingerprint:
    """Compute exact bounded rejection data, not a complete invariant."""
    bounds = _check_bounds(options)
    checked = _check_algebra(algebra, bounds)
    rank = checked.module.rank
    # Product calls: law data costs 4n^3+6n^2, unit solving 2n^2, and the
    # product-span invariant another n^2.  Nested analyses preflight their
    # own complete kernel/closure work before they read their tables.
    direct = 4 * rank**3 + 9 * rank**2
    _preflight(
        bounds=bounds,
        constraints=direct * max(rank, 1),
        work=direct * max(rank * rank, 1),
    )
    # Every nested exact analysis below must also fit before this function reads
    # a single product.  These are per-stage limits (not a fabricated summed
    # work claim): ideals, derivations, and the largest elementary kernel,
    # namely the center.
    ideal_evaluations = 3 * rank * rank
    _preflight(
        bounds=bounds,
        constraints=ideal_evaluations * max(rank, 1),
        work=ideal_evaluations * max(rank * rank, 1),
    )
    _preflight(
        bounds=bounds,
        constraints=rank**3,
        work=rank**3 * (rank * rank) * (rank * rank),
    )
    center_constraints = 3 * rank**3 + rank**2
    _preflight(
        bounds=bounds,
        constraints=center_constraints,
        work=center_constraints * rank * rank,
    )
    try:
        commutative, associative, left, right, two_sided = _law_data(checked)
        return AlgebraFingerprint._create(
            algebra=checked,
            bounds=bounds,
            unit=_unit_report(checked),
            commutative=commutative,
            associative=associative,
            annihilators=(left, right, two_sided),
            ideal_data=ideals(checked, options=bounds),
            derivations=derivation_algebra(checked, options=bounds),
        )
    except FingerprintAnalysisError:
        raise
    except ElementaryAnalysisError as error:
        raise FingerprintAnalysisError(
            field=error.field, reason=error.reason
        ) from error
    except Exception as error:
        raise FingerprintAnalysisError(
            field="fingerprint", reason="exact fingerprint evaluation failed"
        ) from error
