"""QQ operator-bracket recovery guarded by span closure and unique coordinates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TypeAlias, cast

from anyalgebra.algebra.multilinear import StructureConstants
from anyalgebra.core.domains import QQ, _RationalElement
from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.modules import Basis
from anyalgebra.linear.matrix import Matrix
from anyalgebra.linear.solve import ExactSolveError
from anyalgebra.linear.span import (
    Closed,
    ClosureFailure,
    DeclaredBilinearOperation,
    DependentSpan,
    NonClosed,
    SpanError,
    closure,
    span_basis,
    span_decompose,
)


class RecoveryError(AnyAlgebraError, ValueError):
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
        super().__init__(f"operator recovery rejected: {reason}")


@dataclass(frozen=True, slots=True, init=False, eq=False)
class ReconstructionCheck:
    left_index: int
    right_index: int
    captured_product: Matrix
    reconstructed_product: Matrix
    exact: bool
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *a: object, **k: object) -> None:
        del a, k
        raise RecoveryError("recovery constructs reconstruction checks")

    @classmethod
    def _create(
        cls, left: int, right: int, captured: Matrix, reconstructed: Matrix, exact: bool
    ) -> ReconstructionCheck:
        if cls is not ReconstructionCheck:
            raise RecoveryError("factory requires exact ReconstructionCheck")
        value = object.__new__(cls)
        for name, item in (
            ("left_index", left),
            ("right_index", right),
            ("captured_product", captured),
            ("reconstructed_product", reconstructed),
            ("exact", exact),
        ):
            object.__setattr__(value, name, item)
        return value


@dataclass(frozen=True, slots=True, init=False, eq=False)
class RecoverySuccess:
    original_family: tuple[Matrix, ...]
    basis_operators: tuple[Matrix, ...]
    rank: int
    dependencies: tuple[Matrix, ...]
    declaration: DeclaredBilinearOperation
    closure: Closed
    constants: StructureConstants
    reconstruction_checks: tuple[ReconstructionCheck, ...]
    status: str
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *a: object, **k: object) -> None:
        del a, k
        raise RecoveryError("use recover_structure_constants")

    @classmethod
    def _create(
        cls,
        family: tuple[Matrix, ...],
        basis: tuple[Matrix, ...],
        declaration: DeclaredBilinearOperation,
        closed: Closed,
        constants: StructureConstants,
        products: tuple[ReconstructionCheck, ...],
    ) -> RecoverySuccess:
        if cls is not RecoverySuccess:
            raise RecoveryError("factory requires exact RecoverySuccess")
        value = object.__new__(cls)
        for name, item in (
            ("original_family", family),
            ("basis_operators", basis),
            ("rank", len(basis)),
            ("dependencies", ()),
            ("declaration", declaration),
            ("closure", closed),
            ("constants", constants),
            ("reconstruction_checks", products),
            ("status", "success"),
        ):
            object.__setattr__(value, name, item)
        return value


@dataclass(frozen=True, slots=True, init=False, eq=False)
class RecoveryFailure:
    reason: str
    original_family: tuple[Matrix, ...]
    basis_operators: tuple[Matrix, ...]
    rank: int
    dependencies: tuple[Matrix, ...]
    closure_evidence: Closed | NonClosed | ClosureFailure | None
    declaration: DeclaredBilinearOperation | None
    kind: str | None
    left_index: int | None
    right_index: int | None
    observed: int | None
    maximum: int | None
    status: str
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *a: object, **k: object) -> None:
        del a, k
        raise RecoveryError("use recover_structure_constants")

    @classmethod
    def _create(
        cls,
        reason: str,
        family: tuple[Matrix, ...],
        basis: tuple[Matrix, ...] = (),
        dependencies: tuple[Matrix, ...] = (),
        closure_evidence: Closed | NonClosed | ClosureFailure | None = None,
        declaration: DeclaredBilinearOperation | None = None,
        *,
        kind: str | None = None,
        left_index: int | None = None,
        right_index: int | None = None,
        observed: int | None = None,
        maximum: int | None = None,
    ) -> RecoveryFailure:
        if cls is not RecoveryFailure:
            raise RecoveryError("factory requires exact RecoveryFailure")
        value = object.__new__(cls)
        for name, item in (
            ("reason", reason),
            ("original_family", family),
            ("basis_operators", basis),
            ("rank", len(basis)),
            ("dependencies", dependencies),
            ("closure_evidence", closure_evidence),
            ("declaration", declaration),
            ("kind", kind),
            ("left_index", left_index),
            ("right_index", right_index),
            ("observed", observed),
            ("maximum", maximum),
            ("status", "failure"),
        ):
            object.__setattr__(value, name, item)
        return value


RecoveryReport: TypeAlias = RecoverySuccess | RecoveryFailure


def _public_evidence(
    evidence: Closed | NonClosed | ClosureFailure,
    declaration: DeclaredBilinearOperation,
) -> Closed | NonClosed | ClosureFailure:
    if type(evidence) is Closed:
        return Closed._create(
            evidence.basis,
            declaration,
            evidence.coordinate_space,
            evidence.decompositions,
            evidence.pair_count,
        )
    if type(evidence) is NonClosed:
        return NonClosed._create(
            evidence.left_index,
            evidence.right_index,
            evidence.basis,
            declaration,
            evidence.pair_count,
            evidence.product,
            evidence.witness,
        )
    assert type(evidence) is ClosureFailure
    return ClosureFailure._create(
        evidence.reason,
        declaration=declaration,
        left_index=evidence.left_index,
        right_index=evidence.right_index,
        pair_count=evidence.pair_count,
        kind=evidence.kind,
        observed=evidence.observed,
        maximum=evidence.maximum,
    )


def _snapshot(value: object) -> tuple[Matrix, ...]:
    if isinstance(value, str | bytes | Mapping):
        raise RecoveryError("operators must be an iterable")
    try:
        iterator = iter(cast(Iterable[object], value))
    except Exception as error:
        raise RecoveryError(
            f"operators iteration failed ({type(error).__name__})"
        ) from None
    result = []
    try:
        for _ in range(513):
            result.append(next(iterator))
    except StopIteration:
        pass
    except Exception as error:
        raise RecoveryError(
            f"operators iteration failed ({type(error).__name__})"
        ) from None
    if len(result) > 512:
        raise RecoveryError(
            "operator count exceeds 512",
            kind="operators",
            observed=len(result),
            maximum=512,
        )
    if any(type(item) is not Matrix for item in result):
        raise RecoveryError("every operator must be an exact Matrix")
    return cast(tuple[Matrix, ...], tuple(result))


def recover_structure_constants(
    operators: object, declared_bracket: object
) -> RecoveryReport:
    """Recover constants only after independent-family closure is evidenced.

    ``basis=`` is deliberately deferred: v0.0 labels are canonically generated
    from the declaration-order pivot basis, avoiding an unvalidated alternate
    coordinate system.
    """
    try:
        family = _snapshot(operators)
    except RecoveryError:
        raise
    if type(declared_bracket) is not DeclaredBilinearOperation:
        return RecoveryFailure._create("bracket declaration is malformed", family)
    if declared_bracket.bilinear is not True:
        return RecoveryFailure._create(
            "bracket declaration is not bilinear=True",
            family,
            declaration=declared_bracket,
        )
    try:
        basis = span_basis(family)
    except (SpanError, ExactSolveError) as error:
        return RecoveryFailure._create(
            error.reason,
            family,
            kind=error.kind,
            observed=error.observed,
            maximum=error.maximum,
            declaration=declared_bracket,
        )
    if family and family[0].parent is not declared_bracket.space:
        return RecoveryFailure._create(
            "bracket owns a different MatrixSpace",
            family,
            basis,
            declaration=declared_bracket,
        )
    if family:
        try:
            dependency = span_decompose(family[0].parent.zero(), family)
        except (SpanError, ExactSolveError) as error:
            return RecoveryFailure._create(
                error.reason,
                family,
                basis,
                kind=error.kind,
                observed=error.observed,
                maximum=error.maximum,
                declaration=declared_bracket,
            )
        if type(dependency) is DependentSpan:
            return RecoveryFailure._create(
                "declared operator family is dependent; recovery is deferred "
                "to avoid changing the declaration",
                family,
                basis,
                dependency.dependency_basis,
                declaration=declared_bracket,
            )
    captured: dict[tuple[int, int], Matrix] = {}
    index = {id(value): number for number, value in enumerate(basis)}

    def capture(left: Matrix, right: Matrix) -> object:
        product = declared_bracket.operation(left, right)
        if type(product) is Matrix:
            captured[(index[id(left)], index[id(right)])] = product
        return product

    wrapper = DeclaredBilinearOperation.create(
        declared_bracket.space, capture, bilinear=True
    )
    evidence = closure(basis, wrapper)
    if type(evidence) not in (Closed, NonClosed, ClosureFailure):
        return RecoveryFailure._create(
            "operator bracket did not prove closed",
            family,
            basis,
            declaration=declared_bracket,
        )
    evidence = _public_evidence(evidence, declared_bracket)
    if type(evidence) is ClosureFailure:
        return RecoveryFailure._create(
            evidence.reason,
            family,
            basis,
            (),
            evidence,
            declaration=declared_bracket,
            kind=evidence.kind,
            left_index=evidence.left_index,
            right_index=evidence.right_index,
            observed=evidence.observed,
            maximum=evidence.maximum,
        )
    if type(evidence) is NonClosed:
        return RecoveryFailure._create(
            "operator bracket did not prove closed",
            family,
            basis,
            (),
            evidence,
            declaration=declared_bracket,
            left_index=evidence.left_index,
            right_index=evidence.right_index,
        )
    if type(evidence) is not Closed:
        return RecoveryFailure._create(
            "operator bracket did not prove closed",
            family,
            basis,
            declaration=declared_bracket,
        )
    labels = tuple(f"O{index}" for index in range(len(basis)))
    algebra_basis = Basis(labels, coefficient_domain=QQ())
    entries: list[tuple[tuple[int, int, int], _RationalElement]] = []
    rebuilt: list[ReconstructionCheck] = []
    for pair in evidence.decompositions:
        key = (pair.left_index, pair.right_index)
        if key not in captured:
            return RecoveryFailure._create(
                "capture evidence is incomplete",
                family,
                basis,
                (),
                evidence,
                declaration=declared_bracket,
                kind="invariant",
                left_index=pair.left_index,
                right_index=pair.right_index,
            )
        product = captured[key]
        flattened = [
            QQ().element(0) for _ in range(product.parent.rows * product.parent.columns)
        ]
        for output in range(len(basis)):
            coefficient = cast(_RationalElement, pair.coordinates.entry(output, 0))
            if coefficient.value.numerator != 0:
                entries.append(
                    ((pair.left_index, pair.right_index, output), coefficient)
                )
            for row, basis_row in enumerate(basis[output].entries):
                for column, scalar in enumerate(basis_row):
                    offset = row * product.parent.columns + column
                    flattened[offset] = flattened[offset].add(
                        cast(_RationalElement, scalar).multiply(coefficient)
                    )
        reconstruction = product.parent.element(
            tuple(
                tuple(
                    flattened[row * product.parent.columns + column]
                    for column in range(product.parent.columns)
                )
                for row in range(product.parent.rows)
            )
        )
        exact = reconstruction == product
        rebuilt.append(
            ReconstructionCheck._create(
                pair.left_index, pair.right_index, product, reconstruction, exact
            )
        )
        if not exact:
            return RecoveryFailure._create(
                "reconstruction mismatch",
                family,
                basis,
                (),
                evidence,
                declaration=declared_bracket,
                kind="reconstruction",
                left_index=pair.left_index,
                right_index=pair.right_index,
            )
    constants = StructureConstants.from_sparse(
        (algebra_basis, algebra_basis), algebra_basis, tuple(entries)
    )
    return RecoverySuccess._create(
        family, basis, declared_bracket, evidence, constants, tuple(rebuilt)
    )


structure_constants_from_operators = recover_structure_constants
