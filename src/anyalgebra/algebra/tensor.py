"""Bounded metadata and pure values for ordered, unflattened tensor products.

This v0.0 seam records only immediate factor parent identity and pure-tensor
membership.  It deliberately provides no tensor arithmetic, tensor unit,
associator, braiding, flattening, basis construction, conversion, or semantic
fingerprint: those require additional scalar and presentation hypotheses.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from anyalgebra.core.domains import Domain, DomainElement
from anyalgebra.core.elements import SparseElement
from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.modules import FreeModule


class TensorDefinitionError(AnyAlgebraError, ValueError):
    """A bounded tensor parent or pure tensor declaration was malformed."""

    def __init__(self, reason: str, *, index: int | None = None) -> None:
        """Store stable location metadata without rendering arbitrary objects."""
        self.reason = reason
        self.index = index
        location = "" if index is None else f" at declared index {index}"
        super().__init__(f"invalid tensor declaration: {reason}{location}")


def _bounded_snapshot(
    values: object, *, maximum: int, field: str, overlong_reason: str | None = None
) -> tuple[object, ...]:
    """Consume no more than ``maximum + 1`` source values.

    Calling ``tuple(values)`` first would permit an infinite or oversized
    generator to run forever.  The one extra pull distinguishes an exact bound
    from an overlong declaration while preserving the source's declared order.
    """
    iterator_error: str | None = None
    try:
        iterator = iter(cast(Iterable[object], values))
    except Exception as error:
        iterator_error = _exception_name(error)
    if iterator_error is not None:
        raise TensorDefinitionError(f"{field} must be an iterable ({iterator_error})")
    snapshot: list[object] = []
    for _ in range(maximum + 1):
        iteration_error: str | None = None
        try:
            snapshot.append(next(iterator))
        except StopIteration:
            break
        except Exception as error:
            iteration_error = _exception_name(error)
        if iteration_error is not None:
            raise TensorDefinitionError(
                f"{field} could not be iterated ({iteration_error})"
            )
    if len(snapshot) > maximum:
        raise TensorDefinitionError(
            overlong_reason or f"{field} exceed the declared maximum"
        )
    return tuple(snapshot)


def _exception_name(error: Exception) -> str:
    """Return safe exception type metadata without retaining arbitrary payloads."""
    name = type(error).__name__
    return name if name.isidentifier() else "exception"


def _is_declared_parent(candidate: object) -> bool:
    """Recognize the narrow parent set currently supported by this seam.

    v0.0 accepts structural scalar ``Domain`` instances, exact ``FreeModule``
    parents, and exact nested ``TensorProductParent`` objects.  It intentionally
    excludes arbitrary objects merely carrying a ``parent`` attribute, classes,
    carriers, and future algebra parents until they expose an explicit common
    parent protocol and exact element-construction contract.
    """
    if type(candidate) in (FreeModule, TensorProductParent):
        return True
    if isinstance(candidate, type):
        return False
    try:
        return (
            isinstance(candidate, Domain)
            and callable(candidate.normalize)
            and callable(candidate.element)
        )
    except Exception:
        return False


def _value_matches_parent(value: object, parent: object) -> tuple[bool, bool]:
    """Return ``(is_supported_parented_value, has_literal_parent)`` safely."""
    try:
        if type(parent) is FreeModule:
            if type(value) is not SparseElement:
                return False, False
            return True, value.parent is parent
        if type(parent) is TensorProductParent:
            if type(value) is not TensorElement:
                return False, False
            return True, value.parent is parent
        if isinstance(parent, Domain):
            if not isinstance(value, DomainElement):
                return False, False
            return True, value.parent is parent
    except Exception:
        return False, False
    return False, False


def _conservatively_equal(left: object, right: object) -> bool:
    """Accept equality only when it produces the literal built-in ``True``."""
    if left is right:
        return True
    try:
        result = left == right
    except Exception:
        return False
    return type(result) is bool and result is True


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class TensorProductParent:
    """One ordered, parenthesized finite tensor-product parent definition.

    Factors are immutable immediate parents.  In particular, a nested tensor
    parent occupies one factor position: this constructor never treats either
    associativity or factor reordering as an implicit equivalence.  The parent
    is identity-only and intentionally has no fingerprint in this slice,
    because not every admitted factor has a complete semantic fingerprint.
    """

    factors: tuple[object, ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked construction; use :meth:`from_factors`."""
        del args, kwargs
        raise TensorDefinitionError(
            "use TensorProductParent.from_factors to construct a tensor parent"
        )

    @classmethod
    def from_factors(
        cls, factors: Iterable[object], *, max_factors: int
    ) -> TensorProductParent:
        """Freeze two or more immediate factor parents within an exact bound."""
        if type(max_factors) is not int:
            raise TensorDefinitionError("maximum factor count must be a built-in int")
        if max_factors <= 0:
            raise TensorDefinitionError("maximum factor count must be positive")
        snapshot = _bounded_snapshot(factors, maximum=max_factors, field="factors")
        if len(snapshot) < 2:
            raise TensorDefinitionError("tensor products require at least two factors")
        for index, factor in enumerate(snapshot):
            if not _is_declared_parent(factor):
                raise TensorDefinitionError(
                    "factor must be a supported declared parent", index=index
                )
        parent = object.__new__(cls)
        object.__setattr__(parent, "factors", snapshot)
        return parent

    def __eq__(self, other: object) -> bool:
        """Keep separately constructed tensor parent definitions distinct."""
        return self is other

    def __repr__(self) -> str:
        """Render only stable nonsemantic cardinality metadata."""
        return f"TensorProductParent(factor_count={len(self.factors)})"


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class TensorElement:
    """One metadata-only pure tensor with exactly one value per factor."""

    parent: TensorProductParent
    factors: tuple[object, ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked construction; use :meth:`from_factors`."""
        del args, kwargs
        raise TensorDefinitionError(
            "use TensorElement.from_factors to construct a pure tensor"
        )

    @classmethod
    def from_factors(
        cls, parent: TensorProductParent, factors: Iterable[object]
    ) -> TensorElement:
        """Freeze one exact literal-parent value for each immediate factor."""
        if type(parent) is not TensorProductParent:
            raise TensorDefinitionError("parent must be an exact TensorProductParent")
        snapshot = _bounded_snapshot(
            factors,
            maximum=len(parent.factors),
            field="pure-tensor factors",
            overlong_reason="pure tensor has the wrong number of factors",
        )
        if len(snapshot) != len(parent.factors):
            raise TensorDefinitionError("pure tensor has the wrong number of factors")
        for index, (value, factor_parent) in enumerate(
            zip(snapshot, parent.factors, strict=True)
        ):
            is_parented, has_literal_parent = _value_matches_parent(
                value, factor_parent
            )
            if not is_parented:
                raise TensorDefinitionError(
                    "factor must be an exact supported parented value", index=index
                )
            if not has_literal_parent:
                raise TensorDefinitionError(
                    "factor value must have the literal parent", index=index
                )
        element = object.__new__(cls)
        object.__setattr__(element, "parent", parent)
        object.__setattr__(element, "factors", snapshot)
        return element

    def __eq__(self, other: object) -> bool:
        """Compare factors only under the exact same tensor parent object."""
        return (
            type(other) is TensorElement
            and self.parent is other.parent
            and len(self.factors) == len(other.factors)
            and all(
                _conservatively_equal(left, right)
                for left, right in zip(self.factors, other.factors, strict=True)
            )
        )

    def __repr__(self) -> str:
        """Render only stable cardinality metadata, never factor addresses."""
        return f"TensorElement(factor_count={len(self.factors)})"


def tensor_product(*parents: object, max_factors: int) -> TensorProductParent:
    """Convenience factory for one bounded ordered tensor parent."""
    return TensorProductParent.from_factors(parents, max_factors=max_factors)


def pure_tensor(parent: TensorProductParent, *values: object) -> TensorElement:
    """Convenience factory for one unflattened pure tensor value."""
    return TensorElement.from_factors(parent, values)
