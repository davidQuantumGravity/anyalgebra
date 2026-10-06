"""Internal exact rational payloads, independent of parent-domain semantics.

This module deliberately supplies only normalized scalar data and named exact
arithmetic.  ``RationalDomain`` and ``QQ`` remain responsible for accepting
user-facing values, owning elements, and declaring coercions in a later slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd


def _require_builtin_int(value: object, *, name: str) -> int:
    """Return an exact integer input without admitting Python subtype coercions."""
    if type(value) is not int:
        raise TypeError(f"rational {name} must be a built-in int, excluding bool")
    return value


@dataclass(frozen=True, slots=True)
class Rational:
    """An immutable canonical pair of exact built-in integers.

    ``numerator`` and ``denominator`` are reduced eagerly, the denominator is
    always positive, and zero has the sole representation ``0/1``.  The
    standard ``ZeroDivisionError`` deliberately distinguishes invalid
    zero-denominator construction and division by an existing zero rational.
    """

    numerator: int
    denominator: int = 1

    def __post_init__(self) -> None:
        """Validate and rewrite this pair to its unique rational normal form."""
        numerator = _require_builtin_int(self.numerator, name="numerator")
        denominator = _require_builtin_int(self.denominator, name="denominator")
        if denominator == 0:
            raise ZeroDivisionError("rational denominator must not be zero")
        if numerator == 0:
            object.__setattr__(self, "numerator", 0)
            object.__setattr__(self, "denominator", 1)
            return

        common_factor = gcd(numerator, denominator)
        numerator //= common_factor
        denominator //= common_factor
        if denominator < 0:
            numerator = -numerator
            denominator = -denominator
        object.__setattr__(self, "numerator", numerator)
        object.__setattr__(self, "denominator", denominator)

    def add(self, other: Rational) -> Rational:
        """Return the exact sum with another rational payload."""
        right = _require_rational(other)
        return Rational(
            self.numerator * right.denominator + right.numerator * self.denominator,
            self.denominator * right.denominator,
        )

    def subtract(self, other: Rational) -> Rational:
        """Return the exact difference from another rational payload."""
        right = _require_rational(other)
        return Rational(
            self.numerator * right.denominator - right.numerator * self.denominator,
            self.denominator * right.denominator,
        )

    def multiply(self, other: Rational) -> Rational:
        """Return the exact product with another rational payload."""
        right = _require_rational(other)
        return Rational(
            self.numerator * right.numerator,
            self.denominator * right.denominator,
        )

    def divide(self, other: Rational) -> Rational:
        """Return the exact quotient by a nonzero rational payload."""
        right = _require_rational(other)
        if right.numerator == 0:
            raise ZeroDivisionError("division by zero rational")
        return Rational(
            self.numerator * right.denominator,
            self.denominator * right.numerator,
        )

    def negate(self) -> Rational:
        """Return the additive inverse without changing exactness."""
        return Rational(-self.numerator, self.denominator)


def _require_rational(value: object) -> Rational:
    """Reject cross-type arithmetic before any implicit coercion can occur."""
    if type(value) is not Rational:
        raise TypeError("rational arithmetic requires another Rational")
    return value
