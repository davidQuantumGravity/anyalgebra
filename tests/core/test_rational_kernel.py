"""Behavioural contract for the internal canonical rational kernel."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from anyalgebra.core.rational import Rational


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [
        (6, 8, (3, 4)),
        (-6, 8, (-3, 4)),
        (6, -8, (-3, 4)),
        (-6, -8, (3, 4)),
        (0, -99, (0, 1)),
    ],
)
def test_rational_normalizes_gcd_sign_and_zero(
    numerator: int, denominator: int, expected: tuple[int, int]
) -> None:
    """Every rational payload has one reduced sign-normalized representation."""
    rational = Rational(numerator, denominator)

    assert (rational.numerator, rational.denominator) == expected


def test_rational_handles_arbitrarily_large_exact_integers() -> None:
    """Normalization must not use bounded machine arithmetic."""
    scale = 10**400

    rational = Rational(21 * scale, -35 * scale)

    assert (rational.numerator, rational.denominator) == (-3, 5)


@pytest.mark.parametrize(
    ("numerator", "denominator"),
    [
        (True, 1),
        (1, False),
        (1.0, 1),
        (1, 1.0),
        ("1", 1),
        (1, "1"),
    ],
)
def test_rational_rejects_bool_and_non_integer_payloads(
    numerator: object, denominator: object
) -> None:
    """The kernel admits only built-in integers, never implicit Python coercions."""
    with pytest.raises(TypeError, match="built-in int, excluding bool"):
        Rational(numerator, denominator)  # type: ignore[arg-type]


def test_rational_rejects_a_zero_denominator() -> None:
    """A malformed rational literal has the standard typed zero-division failure."""
    with pytest.raises(ZeroDivisionError, match="denominator must not be zero"):
        Rational(1, 0)


def test_rational_is_immutable_and_has_canonical_equality_and_hashing() -> None:
    """Reduced payload equality is deterministic and never coerces native numbers."""
    left = Rational(2, 4)
    same = Rational(-3, -6)
    different = Rational(3, 4)
    native_float: object = 0.5

    assert left == same
    assert hash(left) == hash(same)
    assert left != different
    assert left != native_float
    assert {left: "canonical"}[same] == "canonical"
    with pytest.raises(FrozenInstanceError):
        left.numerator = 7  # type: ignore[misc]


def test_rational_exact_named_arithmetic_primitives() -> None:
    """Arithmetic stays exact and returns normalized rational payloads."""
    left = Rational(-7, 12)
    right = Rational(5, -18)

    assert left.add(right) == Rational(-31, 36)
    assert left.subtract(right) == Rational(-11, 36)
    assert left.multiply(right) == Rational(35, 216)
    assert left.divide(right) == Rational(21, 10)
    assert left.negate() == Rational(7, 12)


def test_rational_division_by_zero_is_typed_and_deterministic() -> None:
    """Dividing by a valid zero rational is distinct from malformed construction."""
    with pytest.raises(ZeroDivisionError, match="division by zero rational"):
        Rational(7, 12).divide(Rational(0, 5))


def test_rational_arithmetic_primitives_do_not_coerce_other_types() -> None:
    """Kernel arithmetic accepts only another exact rational payload."""
    rational = Rational(1, 2)

    with pytest.raises(TypeError, match="requires another Rational"):
        rational.add(1)  # type: ignore[arg-type]
