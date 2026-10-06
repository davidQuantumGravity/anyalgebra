"""Behavioural contract for the canonical exact rational domain."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass

import pytest

from anyalgebra.core.domains import Domain, DomainElement, QQ, RationalDomain, ZZ
from anyalgebra.core.errors import DomainConstructionError
from anyalgebra.core.rational import Rational


@dataclass(frozen=True)
class _ForeignElement:
    """A domain-shaped value which must not be silently imported into ``QQ``."""

    parent: object
    value: object


def test_qq_is_the_canonical_immutable_rational_parent() -> None:
    """Every public construction returns the one exact rational parent."""
    first = QQ()
    second = QQ()

    assert first is second
    assert RationalDomain() is first
    assert isinstance(first, Domain)
    assert repr(first) == "QQ()"
    with pytest.raises(AttributeError):
        first.label = "mutable"  # type: ignore[attr-defined]


def test_rational_domain_rejects_subclass_construction() -> None:
    """A subclass cannot create a second rational-domain parent."""

    class _AttemptedRationalDomain(RationalDomain):
        pass

    with pytest.raises(TypeError, match="final and has one canonical parent"):
        _AttemptedRationalDomain()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0, Rational(0, 1)),
        (-17, Rational(-17, 1)),
        ((6, 8), Rational(3, 4)),
        ((6, -8), Rational(-3, 4)),
        (Rational(-21, -35), Rational(3, 5)),
        ((21 * 10**250, -35 * 10**250), Rational(-3, 5)),
    ],
)
def test_qq_accepts_only_declared_exact_raw_constructor_forms(
    raw: object, expected: Rational
) -> None:
    """``QQ`` accepts declared exact raw literals without a coercion graph."""
    element = QQ().element(raw)

    assert isinstance(element, DomainElement)
    assert element.parent is QQ()
    assert element.value == expected
    assert QQ().normalize(raw) == expected


def test_qq_rewraps_its_own_element_idempotently() -> None:
    """Reconstructing an already normalized QQ element preserves its identity."""
    element = QQ().element((7, 13))

    assert QQ().element(element) is element


@pytest.mark.parametrize(
    "raw",
    [
        True,
        False,
        1.0,
        "1/2",
        [1, 2],
        (1,),
        (1, 2, 3),
        (True, 1),
        (1, False),
        (1.0, 2),
        (1, 2.0),
        ("1", 2),
        object(),
    ],
)
def test_qq_rejects_bool_floats_and_malformed_raw_forms(raw: object) -> None:
    """Only the declared exact raw forms may cross the QQ constructor boundary."""
    with pytest.raises(DomainConstructionError) as caught:
        QQ().element(raw)

    assert caught.value.domain is QQ()
    assert "expected a Rational, a built-in int, or a pair of built-in ints" in (
        caught.value.reason
    )


def test_qq_propagates_deliberate_zero_denominator_failure() -> None:
    """A malformed exact fraction remains distinct from a domain coercion error."""
    with pytest.raises(ZeroDivisionError, match="denominator must not be zero"):
        QQ().element((1, 0))


def test_qq_rejects_foreign_domain_elements_and_zz_elements_without_coercion() -> None:
    """The later ZZ-to-QQ edge is not silently inferred at this boundary."""
    for foreign in (
        _ForeignElement(parent=object(), value=Rational(1, 2)),
        ZZ().element(2),
    ):
        with pytest.raises(DomainConstructionError) as caught:
            QQ().element(foreign)

        assert caught.value.domain is QQ()
        assert caught.value.reason == (
            "cannot implicitly import an element from another parent"
        )


def test_qq_elements_are_immutable_parent_aware_and_hash_consistent() -> None:
    """QQ equality and hashing stay canonical without Python-number coercion."""
    left = QQ().element((2, 4))
    same = QQ().element(Rational(-3, -6))
    different = QQ().element((3, 4))
    native_integer: object = 1
    native_float: object = 0.5

    assert left == same
    assert hash(left) == hash(same)
    assert left != different
    assert left != native_integer
    assert left != native_float
    assert {left: "canonical"}[same] == "canonical"
    with pytest.raises(FrozenInstanceError):
        left.value = Rational(0, 1)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        left.parent = QQ()  # type: ignore[misc]


def test_qq_element_arithmetic_is_exact_and_obeys_bounded_field_identities() -> None:
    """Named and unique Python operators preserve the QQ parent and exact kernel."""
    left = QQ().element((-7, 12))
    right = QQ().element((5, -18))
    zero = QQ().element(0)
    one = QQ().element(1)

    assert left.add(right) == QQ().element((-31, 36))
    assert left.subtract(right) == QQ().element((-11, 36))
    assert left.multiply(right) == QQ().element((35, 216))
    assert left.divide(right) == QQ().element((21, 10))
    assert left.negate() == QQ().element((7, 12))
    assert left + right == left.add(right)
    assert left - right == left.subtract(right)
    assert left * right == left.multiply(right)
    assert left / right == left.divide(right)
    assert -left == left.negate()
    assert left + zero == left
    assert left * one == left
    assert left - left == zero
    assert left + (-left) == zero
    assert left.divide(left) == one
    assert all(
        value.parent is QQ()
        for value in (
            left.add(right),
            left.subtract(right),
            left.multiply(right),
            left.divide(right),
            left.negate(),
        )
    )


def test_qq_element_arithmetic_rejects_python_numbers_and_foreign_domain_elements() -> (
    None
):
    """Arithmetic does not create an undeclared implicit conversion route."""
    element = QQ().element((1, 2))
    foreign = ZZ().element(1)

    for other in (1, 0.5, Rational(1, 2), foreign):
        with pytest.raises(TypeError, match="requires another QQ element"):
            element.add(other)
        with pytest.raises(TypeError, match="requires another QQ element"):
            element + other


def test_qq_element_division_by_zero_propagates_the_kernel_failure() -> None:
    """Division by a valid QQ zero element has the kernel's typed failure."""
    with pytest.raises(ZeroDivisionError, match="division by zero rational"):
        QQ().element((7, 12)).divide(QQ().element(0))
