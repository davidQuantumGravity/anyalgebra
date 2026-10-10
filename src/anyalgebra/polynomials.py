"""Exact polynomials in named symbols with rational coefficients.

A polynomial identity that holds here holds for every value of the symbols
in any commutative ring containing the rationals, so one symbolic calculation
replaces a check over all numerical coefficients::

    from anyalgebra.polynomials import symbols

    a, b = symbols("a b")
    print((a + b) ** 2 - (a - b) ** 2)  # 4*a*b

The class is deliberately small: addition, multiplication, powers,
substitution and printing.  It does not factor or divide.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Final

Monomial = tuple[tuple[str, int], ...]
Number = int | Fraction


def _times(left: Monomial, right: Monomial) -> Monomial:
    powers = dict(left)
    for name, power in right:
        powers[name] = powers.get(name, 0) + power
    return tuple(sorted(powers.items()))


def _order(monomial: Monomial) -> tuple[int, tuple[tuple[str, int], ...]]:
    # Higher degree first, then alphabetical with higher powers first.
    return (
        -sum(power for _, power in monomial),
        tuple((name, -power) for name, power in monomial),
    )


class Polynomial:
    """An immutable polynomial with rational coefficients."""

    __slots__ = ("terms",)

    def __init__(self, terms: Mapping[Monomial, Fraction] | None = None) -> None:
        kept = {
            monomial: Fraction(value)
            for monomial, value in (terms or {}).items()
            if value != 0
        }
        self.terms: Final[dict[Monomial, Fraction]] = kept

    @classmethod
    def constant(cls, value: Number) -> Polynomial:
        """Return a number as a polynomial of degree zero."""
        return cls({(): Fraction(value)})

    @classmethod
    def symbol(cls, name: str) -> Polynomial:
        """Return one symbol as a polynomial."""
        if not name.isidentifier():
            raise ValueError(f"{name!r} is not a valid symbol name")
        return cls({((name, 1),): Fraction(1)})

    @classmethod
    def _coerce(cls, value: object) -> Polynomial | None:
        if isinstance(value, Polynomial):
            return value
        if type(value) is int or isinstance(value, Fraction):
            return cls.constant(value)
        return None

    # facts

    @property
    def variables(self) -> tuple[str, ...]:
        """Return the symbols that occur, in alphabetical order."""
        return tuple(sorted({name for monomial in self.terms for name, _ in monomial}))

    @property
    def degree(self) -> int:
        """Return the total degree; the zero polynomial has degree -1."""
        return max(
            (sum(power for _, power in monomial) for monomial in self.terms),
            default=-1,
        )

    @property
    def is_constant(self) -> bool:
        """Say whether no symbol occurs."""
        return all(monomial == () for monomial in self.terms)

    def as_fraction(self) -> Fraction:
        """Return a constant polynomial as a number."""
        if not self.is_constant:
            raise ValueError(f"{self} is not a constant")
        return self.terms.get((), Fraction(0))

    # arithmetic

    def __add__(self, other: object) -> Polynomial:
        right = self._coerce(other)
        if right is None:
            return NotImplemented
        total = dict(self.terms)
        for monomial, value in right.terms.items():
            total[monomial] = total.get(monomial, Fraction(0)) + value
        return Polynomial(total)

    __radd__ = __add__

    def __neg__(self) -> Polynomial:
        return Polynomial({monomial: -value for monomial, value in self.terms.items()})

    def __sub__(self, other: object) -> Polynomial:
        right = self._coerce(other)
        if right is None:
            return NotImplemented
        return self + (-right)

    def __rsub__(self, other: object) -> Polynomial:
        left = self._coerce(other)
        if left is None:
            return NotImplemented
        return left + (-self)

    def __mul__(self, other: object) -> Polynomial:
        right = self._coerce(other)
        if right is None:
            return NotImplemented
        total: dict[Monomial, Fraction] = {}
        for first, a in self.terms.items():
            for second, b in right.terms.items():
                monomial = _times(first, second)
                total[monomial] = total.get(monomial, Fraction(0)) + a * b
        return Polynomial(total)

    __rmul__ = __mul__

    def __truediv__(self, other: object) -> Polynomial:
        if not (type(other) is int or isinstance(other, Fraction)) or other == 0:
            return NotImplemented
        return self * (Fraction(1) / other)

    def __pow__(self, exponent: object) -> Polynomial:
        if type(exponent) is not int or exponent < 0:
            return NotImplemented
        result = Polynomial.constant(1)
        for _ in range(exponent):
            result = result * self
        return result

    def subs(
        self, values: Mapping[str, object] | None = None, **named: object
    ) -> Polynomial:
        """Replace symbols by numbers or polynomials."""
        chosen = {**(values or {}), **named}
        total = Polynomial()
        for monomial, value in self.terms.items():
            term = Polynomial.constant(value)
            for name, power in monomial:
                if name in chosen:
                    factor = self._coerce(chosen[name])
                    if factor is None:
                        raise TypeError(
                            f"cannot substitute {chosen[name]!r} for {name}"
                        )
                else:
                    factor = Polynomial.symbol(name)
                term = term * factor**power
            total = total + term
        return total

    # comparison and printing

    def __eq__(self, other: object) -> bool:
        right = self._coerce(other)
        if right is None:
            return NotImplemented
        return self.terms == right.terms

    def __hash__(self) -> int:
        return hash(frozenset(self.terms.items()))

    def __bool__(self) -> bool:
        return bool(self.terms)

    def _parts(self) -> list[tuple[bool, str]]:
        parts: list[tuple[bool, str]] = []
        for monomial in sorted(self.terms, key=_order):
            value = self.terms[monomial]
            size = abs(value)
            number = (
                str(size.numerator)
                if size.denominator == 1
                else f"{size.numerator}/{size.denominator}"
            )
            names = "*".join(
                name if power == 1 else f"{name}^{power}" for name, power in monomial
            )
            if not names:
                body = number
            elif size == 1:
                body = names
            else:
                body = f"{number}*{names}"
            parts.append((value < 0, body))
        return parts

    def __str__(self) -> str:
        parts = self._parts()
        if not parts:
            return "0"
        text = ("-" if parts[0][0] else "") + parts[0][1]
        return text + "".join(
            (" - " if negative else " + ") + body for negative, body in parts[1:]
        )

    def __repr__(self) -> str:
        return f"Polynomial({self})"


def symbols(names: str) -> tuple[Polynomial, ...]:
    """Return one polynomial per name in a space- or comma-separated list."""
    return tuple(Polynomial.symbol(name) for name in names.replace(",", " ").split())
