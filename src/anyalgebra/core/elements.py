"""Canonical immutable sparse elements of finite free modules.

Construction is owned by :class:`~anyalgebra.core.modules.FreeModule`.  This
module validates mapping coordinates, performs only explicit coefficient
coercions, freezes nonzero support in ascending basis-index order, and provides
exact same-parent module arithmetic.  Algebra multiplication and presentation
transport remain deliberately later boundaries.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType, NotImplementedType
from typing import TYPE_CHECKING

from .coercions import CoercionGraph, CoercionPlan
from .domains import ZZ, Domain, DomainElement
from .errors import AnyAlgebraError

if TYPE_CHECKING:
    from .modules import FreeModule


class SparseElementConstructionError(AnyAlgebraError, ValueError):
    """Sparse coordinate construction violated a deterministic boundary."""

    def __init__(
        self,
        reason: str,
        *,
        coordinate_index: object | None = None,
        coefficient_parent: object | None = None,
    ) -> None:
        """Retain structured rejected data without rendering arbitrary objects."""
        self.reason = reason
        self.coordinate_index = coordinate_index
        self.coefficient_parent = coefficient_parent
        detail = reason
        if type(coordinate_index) is int:
            detail = f"{reason} at coordinate index {coordinate_index}"
        super().__init__(detail)


class CoordinateIndexError(SparseElementConstructionError, IndexError):
    """A sparse coordinate key was not an exact in-range basis index."""


class ModuleParentMismatchError(AnyAlgebraError, TypeError):
    """A module operation received values from different literal parents."""

    def __init__(
        self,
        operation: str,
        left_parent: object,
        right_parent: object | None,
    ) -> None:
        """Retain both parent witnesses without using shape or display names."""
        self.operation = operation
        self.left_parent = left_parent
        self.right_parent = right_parent
        super().__init__(
            f"module {operation} requires operands with the literal same parent"
        )


class ModuleArithmeticError(AnyAlgebraError):
    """An exact coefficient capability was absent or violated its contract."""

    def __init__(
        self,
        reason: str,
        *,
        operation: str,
        coordinate_index: int | None = None,
    ) -> None:
        """Record the named operation and deterministic failing coordinate."""
        self.reason = reason
        self.operation = operation
        self.coordinate_index = coordinate_index
        detail = f"module coefficient {operation} failed: {reason}"
        if coordinate_index is not None:
            detail = (
                f"module coefficient {operation} failed at coordinate index "
                f"{coordinate_index}: {reason}"
            )
        super().__init__(detail)


class SparseElementMapDefinitionError(AnyAlgebraError, ValueError):
    """An explicit sparse-element map lacks declared valid endpoints."""

    def __init__(self, reason: str) -> None:
        """Store a stable configuration diagnostic without rendering objects."""
        self.reason = reason
        super().__init__(reason)


class SparseElementTransportError(AnyAlgebraError, TypeError):
    """An explicit sparse-element map violated its declared transport contract."""

    def __init__(self, reason: str, *, element_map: SparseElementMap) -> None:
        """Retain the declared map while keeping the error text address-free."""
        self.reason = reason
        self.element_map = element_map
        super().__init__(reason)


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class SparseElement:
    """One immutable canonical sparse value owned by a literal module parent.

    Support is stored as immutable ``(basis_index, coefficient)`` pairs in
    strictly ascending index order.  Equality requires the very same module
    object as well as equal canonical pairs.  The class is universally
    unhashable because the neutral coefficient-domain protocol does not promise
    hashable elements for every admissible exact domain.
    """

    parent: FreeModule
    _coordinate_pairs: tuple[tuple[int, DomainElement[object]], ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self,
        parent: FreeModule,
        coordinate_pairs: tuple[tuple[int, DomainElement[object]], ...],
    ) -> None:
        """Reject direct construction; the owning module validates membership."""
        raise SparseElementConstructionError(
            "SparseElement values must be constructed by FreeModule.element"
        )

    @classmethod
    def _from_canonical(
        cls,
        parent: FreeModule,
        coordinate_pairs: tuple[tuple[int, DomainElement[object]], ...],
    ) -> SparseElement:
        """Freeze pairs already validated by the module construction boundary."""
        element = object.__new__(cls)
        object.__setattr__(element, "parent", parent)
        object.__setattr__(element, "_coordinate_pairs", coordinate_pairs)
        return element

    def coordinates(self) -> Mapping[int, DomainElement[object]]:
        """Return a fresh immutable mapping in canonical basis-index order."""
        return MappingProxyType(dict(self._coordinate_pairs))

    def __eq__(self, other: object) -> bool:
        """Compare coordinates conservatively under one literal module parent.

        Arbitrary coefficient domains may return symbolic or indeterminate
        equality results.  Such a result cannot establish mathematical
        equality merely because Python considers the result truthy.
        """
        if type(other) is not SparseElement or self.parent is not other.parent:
            return False
        if len(self._coordinate_pairs) != len(other._coordinate_pairs):
            return False
        return all(
            left_index == right_index
            and _coefficients_equal(left_coefficient, right_coefficient)
            for (left_index, left_coefficient), (right_index, right_coefficient) in zip(
                self._coordinate_pairs, other._coordinate_pairs, strict=True
            )
        )

    def __repr__(self) -> str:
        """Render stable rank and support metadata without parent addresses."""
        indices = tuple(index for index, _ in self._coordinate_pairs)
        return f"SparseElement(rank={self.parent.rank}, support={indices!r})"

    def add(self, other: SparseElement) -> SparseElement:
        """Return the exact sum with an element of this literal module parent."""
        right = self._require_same_parent(other, operation="add")
        return _merge_sparse(self, right, operation="add")

    def subtract(self, other: SparseElement) -> SparseElement:
        """Return the exact difference from an element of this literal parent."""
        right = self._require_same_parent(other, operation="subtract")
        return _merge_sparse(self, right, operation="subtract")

    def negate(self) -> SparseElement:
        """Return the exact additive inverse in this literal module parent."""
        pairs = tuple(
            (
                index,
                _coefficient_unary(
                    coefficient,
                    operation="negate",
                    coordinate_index=index,
                    domain=self.parent.domain,
                ),
            )
            for index, coefficient in self._coordinate_pairs
        )
        return _from_arithmetic_pairs(self.parent, pairs)

    def scale(
        self,
        scalar: object,
        *,
        graph: CoercionGraph | None = None,
    ) -> SparseElement:
        """Apply one exact scalar from the module domain to every coefficient.

        Raw literals enter only through the owning domain constructor.  A
        foreign domain element requires an explicit graph with one unique
        lossless route.  Scalar validation and route application finish before
        any coefficient multiplication begins.
        """
        normalized = _module_scalar(self.parent.domain, scalar, graph=graph)
        pairs = tuple(
            (
                index,
                _coefficient_binary(
                    coefficient,
                    normalized,
                    operation="multiply",
                    coordinate_index=index,
                    domain=self.parent.domain,
                ),
            )
            for index, coefficient in self._coordinate_pairs
        )
        return _from_arithmetic_pairs(self.parent, pairs)

    def _require_same_parent(self, other: object, *, operation: str) -> SparseElement:
        """Validate literal parent identity without coordinate reinterpretation."""
        if type(other) is not SparseElement or other.parent is not self.parent:
            right_parent = other.parent if type(other) is SparseElement else None
            raise ModuleParentMismatchError(operation, self.parent, right_parent)
        return other

    def __add__(self, other: SparseElement) -> SparseElement:
        """Delegate unambiguous same-parent module addition to :meth:`add`."""
        return self.add(other)

    def __sub__(self, other: SparseElement) -> SparseElement:
        """Delegate unambiguous same-parent subtraction to :meth:`subtract`."""
        return self.subtract(other)

    def __neg__(self) -> SparseElement:
        """Delegate additive inversion to :meth:`negate`."""
        return self.negate()

    def __mul__(self, scalar: object) -> SparseElement | NotImplementedType:
        """Delegate only scalar action; never infer element multiplication."""
        if type(scalar) is SparseElement:
            return NotImplemented
        return self.scale(scalar)

    def __rmul__(self, scalar: object) -> SparseElement | NotImplementedType:
        """Delegate left scalar action while refusing module-element products."""
        if type(scalar) is SparseElement:
            return NotImplemented
        return self.scale(scalar)


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class SparseElementMap:
    """One explicit whole-element transport between literal module parents.

    This small v0.0 seam deliberately maps a complete ``SparseElement`` rather
    than exposing or inferring a coordinate reinterpretation.  It has declared
    literal source and target module objects, and it validates both ends at the
    application boundary.  General linear maps, basis changes, map registries,
    and route planning remain later-version responsibilities.
    """

    source: FreeModule
    target: FreeModule
    forward: Callable[[SparseElement], SparseElement]
    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self,
        source: FreeModule,
        target: FreeModule,
        forward: Callable[[SparseElement], SparseElement],
    ) -> None:
        """Freeze one declared endpoint pair and a whole-element callable."""
        if not _is_exact_free_module(source):
            raise SparseElementMapDefinitionError(
                "sparse element map source must be an exact FreeModule"
            )
        if not _is_exact_free_module(target):
            raise SparseElementMapDefinitionError(
                "sparse element map target must be an exact FreeModule"
            )
        if not callable(forward):
            raise SparseElementMapDefinitionError(
                "sparse element map forward must be callable"
            )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "forward", forward)

    def apply(self, value: SparseElement) -> SparseElement:
        """Transport one source-owned value and validate its target ownership."""
        if type(value) is not SparseElement or value.parent is not self.source:
            value_parent = value.parent if type(value) is SparseElement else None
            raise ModuleParentMismatchError("transport", self.source, value_parent)
        try:
            result = self.forward(value)
        except Exception as error:
            raise SparseElementTransportError(
                "sparse element map forward callable raised an exception",
                element_map=self,
            ) from error
        if type(result) is not SparseElement:
            raise SparseElementTransportError(
                "sparse element map forward callable did not return SparseElement",
                element_map=self,
            )
        if result.parent is not self.target:
            raise SparseElementTransportError(
                "sparse element map forward callable returned an element outside "
                "the declared target parent",
                element_map=self,
            )
        return result

    def __repr__(self) -> str:
        """Render stable endpoint rank metadata and no callable address."""
        return (
            "SparseElementMap("
            f"source_rank={self.source.rank}, target_rank={self.target.rank})"
        )


def transport_sparse_element(
    value: SparseElement, element_map: SparseElementMap
) -> SparseElement:
    """Apply one declared whole-element map without implicit parent routing."""
    if type(element_map) is not SparseElementMap:
        raise TypeError("element_map must be an exact SparseElementMap")
    return element_map.apply(value)


def _is_exact_free_module(candidate: object) -> bool:
    """Recognize the one current module-parent implementation without duck typing."""
    from .modules import FreeModule

    return type(candidate) is FreeModule


def _merge_sparse(
    left: SparseElement,
    right: SparseElement,
    *,
    operation: str,
) -> SparseElement:
    """Merge two ascending supports with exact coefficient operations."""
    domain = left.parent.domain
    left_pairs = left._coordinate_pairs
    right_pairs = right._coordinate_pairs
    merged: list[tuple[int, DomainElement[object]]] = []
    left_position = 0
    right_position = 0

    while left_position < len(left_pairs) or right_position < len(right_pairs):
        if right_position >= len(right_pairs):
            merged.extend(left_pairs[left_position:])
            break
        if left_position >= len(left_pairs):
            for index, coefficient in right_pairs[right_position:]:
                if operation == "add":
                    merged.append((index, coefficient))
                else:
                    merged.append(
                        (
                            index,
                            _coefficient_unary(
                                coefficient,
                                operation="negate",
                                coordinate_index=index,
                                domain=domain,
                            ),
                        )
                    )
            break

        left_index, left_coefficient = left_pairs[left_position]
        right_index, right_coefficient = right_pairs[right_position]
        if left_index < right_index:
            merged.append((left_index, left_coefficient))
            left_position += 1
        elif right_index < left_index:
            if operation == "add":
                merged.append((right_index, right_coefficient))
            else:
                merged.append(
                    (
                        right_index,
                        _coefficient_unary(
                            right_coefficient,
                            operation="negate",
                            coordinate_index=right_index,
                            domain=domain,
                        ),
                    )
                )
            right_position += 1
        else:
            merged.append(
                (
                    left_index,
                    _coefficient_binary(
                        left_coefficient,
                        right_coefficient,
                        operation=operation,
                        coordinate_index=left_index,
                        domain=domain,
                    ),
                )
            )
            left_position += 1
            right_position += 1

    return _from_arithmetic_pairs(left.parent, tuple(merged))


def _module_scalar(
    domain: Domain[object],
    scalar: object,
    *,
    graph: CoercionGraph | None,
) -> DomainElement[object]:
    """Normalize or explicitly transport a scalar before coefficient work."""
    if graph is not None and not isinstance(graph, CoercionGraph):
        raise TypeError("graph must be a CoercionGraph or None")

    if isinstance(scalar, DomainElement):
        if scalar.parent is domain:
            normalized = scalar
        else:
            if graph is None:
                raise ModuleArithmeticError(
                    "foreign scalar requires an explicit CoercionGraph",
                    operation="scale",
                )
            normalized = graph.plan(scalar.parent, domain).apply((scalar,))[0]
    else:
        normalized = domain.element(scalar)

    if not isinstance(normalized, DomainElement) or normalized.parent is not domain:
        raise ModuleArithmeticError(
            "scalar construction did not return an element of the module domain",
            operation="scale",
        )
    return normalized


def _coefficient_binary(
    left: DomainElement[object],
    right: DomainElement[object],
    *,
    operation: str,
    coordinate_index: int,
    domain: Domain[object],
) -> DomainElement[object]:
    """Perform one named exact binary coefficient operation."""
    if domain is ZZ():
        if type(left.value) is not int or type(right.value) is not int:
            raise ModuleArithmeticError(
                "canonical ZZ coefficient payload is malformed",
                operation=operation,
                coordinate_index=coordinate_index,
            )
        if operation == "add":
            result = domain.element(left.value + right.value)
        elif operation == "subtract":
            result = domain.element(left.value - right.value)
        elif operation == "multiply":
            result = domain.element(left.value * right.value)
        else:
            raise ModuleArithmeticError(
                "unsupported internal binary operation",
                operation=operation,
                coordinate_index=coordinate_index,
            )
        return _validate_arithmetic_result(
            result,
            domain=domain,
            operation=operation,
            coordinate_index=coordinate_index,
        )

    return _invoke_coefficient_capability(
        left,
        operation,
        right,
        domain=domain,
        coordinate_index=coordinate_index,
    )


def _coefficient_unary(
    coefficient: DomainElement[object],
    *,
    operation: str,
    coordinate_index: int,
    domain: Domain[object],
) -> DomainElement[object]:
    """Perform one named exact unary coefficient operation."""
    if domain is ZZ():
        if type(coefficient.value) is not int:
            raise ModuleArithmeticError(
                "canonical ZZ coefficient payload is malformed",
                operation=operation,
                coordinate_index=coordinate_index,
            )
        result = domain.element(-coefficient.value)
        return _validate_arithmetic_result(
            result,
            domain=domain,
            operation=operation,
            coordinate_index=coordinate_index,
        )

    return _invoke_coefficient_capability(
        coefficient,
        operation,
        domain=domain,
        coordinate_index=coordinate_index,
    )


def _invoke_coefficient_capability(
    coefficient: DomainElement[object],
    operation: str,
    *arguments: DomainElement[object],
    domain: Domain[object],
    coordinate_index: int,
) -> DomainElement[object]:
    """Invoke an explicit named domain-element capability and validate it."""
    try:
        capability = getattr(coefficient, operation)
    except Exception as error:
        raise ModuleArithmeticError(
            f"coefficient does not provide a usable {operation} capability",
            operation=operation,
            coordinate_index=coordinate_index,
        ) from error
    if not callable(capability):
        raise ModuleArithmeticError(
            f"coefficient does not provide a callable {operation} capability",
            operation=operation,
            coordinate_index=coordinate_index,
        )
    try:
        result = capability(*arguments)
    except Exception as error:
        raise ModuleArithmeticError(
            f"coefficient {operation} capability raised an exception",
            operation=operation,
            coordinate_index=coordinate_index,
        ) from error
    return _validate_arithmetic_result(
        result,
        domain=domain,
        operation=operation,
        coordinate_index=coordinate_index,
    )


def _validate_arithmetic_result(
    result: object,
    *,
    domain: Domain[object],
    operation: str,
    coordinate_index: int,
) -> DomainElement[object]:
    """Require every coefficient result to belong to the literal domain."""
    if not isinstance(result, DomainElement) or result.parent is not domain:
        raise ModuleArithmeticError(
            "coefficient capability returned a value outside the module domain",
            operation=operation,
            coordinate_index=coordinate_index,
        )
    return result


def _from_arithmetic_pairs(
    module: FreeModule,
    pairs: tuple[tuple[int, DomainElement[object]], ...],
) -> SparseElement:
    """Recanonicalize proven same-domain arithmetic results without mutation."""
    zero = _domain_zero(module.domain)
    canonical = tuple(
        (index, coefficient)
        for index, coefficient in pairs
        if not _is_proven_zero(coefficient, zero)
    )
    return SparseElement._from_canonical(module, canonical)


def construct_sparse_element(
    module: FreeModule,
    coordinates: Mapping[int, object] | SparseElement,
    *,
    graph: CoercionGraph | None = None,
) -> SparseElement:
    """Construct one canonical sparse value through ``module.element``.

    Container and all coordinate indices are validated before any coefficient
    constructor or user coercion callable runs.  Every foreign coefficient
    route is likewise planned before any route is applied, so a later
    ambiguity, lossy route, or missing route cannot leave partial effects.
    """
    if graph is not None and not isinstance(graph, CoercionGraph):
        raise TypeError("graph must be a CoercionGraph or None")

    if type(coordinates) is SparseElement:
        if coordinates.parent is module:
            return coordinates
        raise SparseElementConstructionError(
            "SparseElement input must have the literal module parent"
        )
    if not isinstance(coordinates, Mapping):
        raise SparseElementConstructionError(
            "sparse coordinates must be a Mapping from basis indices to coefficients"
        )

    keys = tuple(coordinates)
    for key in keys:
        if type(key) is not int:
            raise CoordinateIndexError(
                "coordinate index must be an exact built-in int, excluding bool",
                coordinate_index=key,
            )
        if key < 0 or key >= module.rank:
            raise CoordinateIndexError(
                "coordinate index is outside the module basis range",
                coordinate_index=key,
            )

    ordered_inputs = tuple((index, coordinates[index]) for index in sorted(keys))
    plans: dict[int, CoercionPlan] = {}
    for index, raw in ordered_inputs:
        if isinstance(raw, DomainElement) and raw.parent is not module.domain:
            if graph is None:
                raise SparseElementConstructionError(
                    "foreign coefficient requires an explicit CoercionGraph",
                    coordinate_index=index,
                    coefficient_parent=raw.parent,
                )
            plans[index] = graph.plan(raw.parent, module.domain)

    normalized: list[tuple[int, DomainElement[object]]] = []
    for index, raw in ordered_inputs:
        if isinstance(raw, DomainElement):
            if raw.parent is module.domain:
                coefficient = raw
            else:
                coefficient = plans[index].apply((raw,))[0]
        else:
            coefficient = module.domain.element(raw)
        if (
            not isinstance(coefficient, DomainElement)
            or coefficient.parent is not module.domain
        ):
            raise SparseElementConstructionError(
                "coefficient construction did not return an element of the "
                "module domain",
                coordinate_index=index,
            )
        normalized.append((index, coefficient))

    zero = _domain_zero(module.domain)
    canonical = tuple(
        (index, coefficient)
        for index, coefficient in normalized
        if not _is_proven_zero(coefficient, zero)
    )
    return SparseElement._from_canonical(module, canonical)


def _domain_zero(domain: Domain[object]) -> DomainElement[object] | None:
    """Return the domain's exact literal zero when construction establishes it."""
    try:
        zero = domain.element(0)
    except Exception:
        return None
    if isinstance(zero, DomainElement) and zero.parent is domain:
        return zero
    return None


def _is_proven_zero(
    coefficient: DomainElement[object], zero: DomainElement[object] | None
) -> bool:
    """Discard only when equality returns the literal built-in truth value."""
    if zero is None:
        return False
    try:
        equality = coefficient == zero
    except Exception:
        return False
    return type(equality) is bool and equality is True


def _coefficients_equal(
    left: DomainElement[object], right: DomainElement[object]
) -> bool:
    """Establish coefficient equality only by identity or literal ``True``."""
    if left is right:
        return True
    try:
        equality = left == right
    except Exception:
        return False
    return type(equality) is bool and equality is True
