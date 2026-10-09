"""Neutral scalar-domain protocols plus canonical exact ``ZZ`` and ``QQ``.

The concrete parents in this module own eager normalization and their local
exact-element behavior.  Coercion maps, global promotion, common-parent
planning, and cross-parent arithmetic remain explicit later boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

from .errors import DomainConstructionError
from .rational import Rational


T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class DomainElement(Protocol[T_co]):
    """An immutable normalized scalar value owned by one exact domain.

    Implementations expose their parent and normalized payload read-only.  The
    protocol does not prescribe arithmetic operators because their meaning is
    specific to the parent and must remain explicit until a concrete domain
    defines it.
    """

    @property
    def parent(self) -> Domain[T_co]:
        """Return the exact domain that owns this element."""

    @property
    def value(self) -> T_co:
        """Return this element's canonical normalized payload."""


@runtime_checkable
class Domain(Protocol[T_co]):
    """A neutral exact-domain parent responsible for validation and normal form."""

    def normalize(self, value: object) -> T_co:
        """Validate ``value`` and return the parent-specific canonical payload."""

    def element(self, value: object) -> DomainElement[T_co]:
        """Construct an immutable parent-aware element from an exact input."""


class IntegerDomain:
    """The canonical exact parent for native Python integers.

    Construction accepts only values whose exact runtime type is ``int``.  In
    particular, ``bool`` is not admitted through Python's ``bool``-is-``int``
    subtype relation.  ``ZZ()`` and ``IntegerDomain()`` return the same
    immutable parent; this slice deliberately defines no arithmetic surface.
    """

    __slots__ = ()

    def __new__(cls) -> IntegerDomain:
        """Return the one canonical integer-domain parent."""
        if cls is not IntegerDomain:
            raise TypeError("IntegerDomain is final and has one canonical parent")
        return _ZZ

    def __repr__(self) -> str:
        """Return a concise, stable representation of the canonical parent."""
        return "ZZ()"

    def normalize(self, value: object) -> int:
        """Validate and return a native integer without coercing any other type."""
        if type(value) is not int:
            raise DomainConstructionError(
                self, "expected a built-in int, excluding bool"
            )
        return value

    def element(self, value: object) -> _IntegerElement:
        """Construct or idempotently return an immutable normalized integer."""
        if isinstance(value, _IntegerElement):
            if value.parent is self:
                return value
            raise DomainConstructionError(
                self, "cannot implicitly import an element from another parent"
            )
        if isinstance(value, DomainElement):
            raise DomainConstructionError(
                self, "cannot implicitly import an element from another parent"
            )
        return _IntegerElement(parent=self, value=self.normalize(value))


@dataclass(frozen=True, slots=True)
class _IntegerElement:
    """Private normalized integer payload owned by the canonical ``ZZ`` parent."""

    parent: IntegerDomain
    value: int

    def __str__(self) -> str:
        """Render the integer itself; ``repr`` keeps the full record."""
        return str(self.value)


_ZZ = object.__new__(IntegerDomain)


def ZZ() -> IntegerDomain:
    """Return the canonical exact integer-domain parent."""
    return IntegerDomain()


class RationalDomain:
    """The canonical exact parent for normalized :class:`Rational` payloads.

    The constructor accepts an already-normalized ``Rational``, a built-in
    ``int``, or an exact two-item ``(numerator, denominator)`` tuple of built-in
    integers.  The latter two forms are direct QQ literals, not coercion edges:
    in particular, a ``ZZ`` element is rejected until the ``ZZ -> QQ`` map is
    explicitly introduced in a later slice.
    """

    __slots__ = ()

    def __new__(cls) -> RationalDomain:
        """Return the one canonical rational-domain parent."""
        if cls is not RationalDomain:
            raise TypeError("RationalDomain is final and has one canonical parent")
        return _QQ

    def __repr__(self) -> str:
        """Return a concise, stable representation of the canonical parent."""
        return "QQ()"

    def normalize(self, value: object) -> Rational:
        """Validate one exact QQ literal and return its normalized payload."""
        if type(value) is Rational:
            return value
        if type(value) is int:
            return Rational(value)
        if type(value) is tuple:
            if len(value) != 2:
                raise DomainConstructionError(
                    self,
                    "expected a Rational, a built-in int, or a pair of built-in ints",
                )
            numerator, denominator = value
            if type(numerator) is int and type(denominator) is int:
                return Rational(numerator, denominator)
            raise DomainConstructionError(
                self,
                "expected a Rational, a built-in int, or a pair of built-in ints",
            )
        raise DomainConstructionError(
            self,
            "expected a Rational, a built-in int, or a pair of built-in ints",
        )

    def element(self, value: object) -> _RationalElement:
        """Construct or idempotently return an immutable normalized QQ element."""
        if isinstance(value, _RationalElement):
            if value.parent is self:
                return value
            raise DomainConstructionError(
                self, "cannot implicitly import an element from another parent"
            )
        if isinstance(value, DomainElement):
            raise DomainConstructionError(
                self, "cannot implicitly import an element from another parent"
            )
        return _RationalElement(parent=self, value=self.normalize(value))


@dataclass(frozen=True, slots=True)
class _RationalElement:
    """Private immutable rational payload owned by the canonical ``QQ`` parent."""

    parent: RationalDomain
    value: Rational

    def __str__(self) -> str:
        """Render the value as ``3`` or ``3/2``; ``repr`` keeps the full record."""
        if self.value.denominator == 1:
            return str(self.value.numerator)
        return f"{self.value.numerator}/{self.value.denominator}"

    def _require_compatible(self, other: object) -> _RationalElement:
        """Reject every arithmetic input except an element of this exact parent."""
        if type(other) is not _RationalElement or other.parent is not self.parent:
            raise TypeError("rational element arithmetic requires another QQ element")
        return other

    def add(self, other: object) -> _RationalElement:
        """Return the exact sum with another QQ element."""
        right = self._require_compatible(other)
        return self.parent.element(self.value.add(right.value))

    def subtract(self, other: object) -> _RationalElement:
        """Return the exact difference from another QQ element."""
        right = self._require_compatible(other)
        return self.parent.element(self.value.subtract(right.value))

    def multiply(self, other: object) -> _RationalElement:
        """Return the exact product with another QQ element."""
        right = self._require_compatible(other)
        return self.parent.element(self.value.multiply(right.value))

    def divide(self, other: object) -> _RationalElement:
        """Return the exact quotient by another nonzero QQ element."""
        right = self._require_compatible(other)
        return self.parent.element(self.value.divide(right.value))

    def negate(self) -> _RationalElement:
        """Return the additive inverse as an element of the same QQ parent."""
        return self.parent.element(self.value.negate())

    def __add__(self, other: object) -> _RationalElement:
        """Delegate unambiguous QQ addition to :meth:`add`."""
        return self.add(other)

    def __sub__(self, other: object) -> _RationalElement:
        """Delegate unambiguous QQ subtraction to :meth:`subtract`."""
        return self.subtract(other)

    def __mul__(self, other: object) -> _RationalElement:
        """Delegate unambiguous QQ multiplication to :meth:`multiply`."""
        return self.multiply(other)

    def __truediv__(self, other: object) -> _RationalElement:
        """Delegate unambiguous QQ division to :meth:`divide`."""
        return self.divide(other)

    def __neg__(self) -> _RationalElement:
        """Delegate additive inversion to :meth:`negate`."""
        return self.negate()


_QQ = object.__new__(RationalDomain)


def QQ() -> RationalDomain:
    """Return the canonical exact rational-domain parent."""
    return RationalDomain()
