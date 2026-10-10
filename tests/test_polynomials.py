"""Exact polynomials with rational coefficients."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

import pytest

from anyalgebra.polynomials import Polynomial, symbols


def test_arithmetic_and_printing() -> None:
    a, b = symbols("a, b")
    assert str((a + b) ** 2 - (a - b) ** 2) == "4*a*b"
    assert str((a + b) ** 2) == "a^2 + 2*a*b + b^2"
    assert str(a * a / 2 - b + 1) == "1/2*a^2 - b + 1"
    assert str(-a + Fraction(3, 2)) == "-a + 3/2" and str(Polynomial()) == "0"
    assert str(1 - a) == "-a + 1" and str(2 * a) == "2*a" and a + 0 == a
    assert repr(a * b) == "Polynomial(a*b)" and a**0 == 1
    assert (a - a) == 0 and not (a - a) and bool(a)
    assert a * b == b * a and hash(a * b) == hash(b * a) and a != b
    assert (a == "a") is False


def test_facts_and_substitution() -> None:
    a, b = symbols("a b")
    p = a**2 * b + 3 * a - 5
    assert p.variables == ("a", "b") and p.degree == 3
    assert Polynomial().degree == -1 and Polynomial.constant(7).is_constant
    assert p.subs(a=2) == 4 * b + 1 and p.subs({"a": 2}, b=1).as_fraction() == 5
    assert p.subs(a=b) == b**3 + 3 * b - 5
    assert Polynomial().as_fraction() == 0


def test_rejections() -> None:
    (a,) = symbols("a")
    with pytest.raises(ValueError, match="not a constant"):
        a.as_fraction()
    with pytest.raises(ValueError, match="valid symbol"):
        Polynomial.symbol("2a")
    with pytest.raises(TypeError, match="cannot substitute"):
        a.subs(a="x")
    wrong: tuple[Callable[[], object], ...] = (
        lambda: a + "x",
        lambda: a - 1.5,
        lambda: "x" - a,
        lambda: a * 0.5,
        lambda: a / 0,
        lambda: a / a,
        lambda: a**-1,
        lambda: a**0.5,
    )
    for call in wrong:
        with pytest.raises(TypeError):
            call()
