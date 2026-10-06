"""Behavioural contract for the canonical exact integer domain."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass

import pytest

from anyalgebra.core.domains import Domain, DomainElement, IntegerDomain, ZZ
from anyalgebra.core.errors import DomainConstructionError


@dataclass(frozen=True)
class _ForeignElement:
    """A domain-shaped value which must not be silently imported into ``ZZ``."""

    parent: object
    value: int


def test_zz_is_the_canonical_immutable_integer_parent() -> None:
    """Every public construction returns the one exact integer parent."""
    first = ZZ()
    second = ZZ()

    assert first is second
    assert IntegerDomain() is first
    assert isinstance(first, Domain)
    with pytest.raises(AttributeError):
        first.label = "mutable"  # type: ignore[attr-defined]


def test_integer_domain_rejects_subclass_construction() -> None:
    """A subclass cannot create a second integer-domain parent."""

    class _AttemptedIntegerDomain(IntegerDomain):
        pass

    with pytest.raises(TypeError, match="final and has one canonical parent"):
        _AttemptedIntegerDomain()


@pytest.mark.parametrize("value", [0, -1, 1, -(10**200), 10**1000])
def test_zz_constructs_normalized_exact_integer_elements(value: int) -> None:
    """Zero, signs, and arbitrarily large native integers remain exact."""
    element = ZZ().element(value)

    assert isinstance(element, DomainElement)
    assert element.parent is ZZ()
    assert element.value == value
    assert ZZ().normalize(value) == value


@pytest.mark.parametrize("value", [True, False, 1.0, "1", b"1", object()])
def test_zz_rejects_bool_and_non_integer_inputs_exactly(value: object) -> None:
    """Only built-in integers enter through the exact constructor boundary."""
    with pytest.raises(DomainConstructionError) as caught:
        ZZ().element(value)

    assert caught.value.domain is ZZ()
    assert caught.value.reason == "expected a built-in int, excluding bool"


def test_zz_rewraps_its_own_element_idempotently() -> None:
    """Reconstructing an already normalized element preserves its identity."""
    element = ZZ().element(-17)

    assert ZZ().element(element) is element


def test_zz_rejects_foreign_parented_values_without_coercion() -> None:
    """A domain-shaped value cannot cross a parent boundary implicitly."""
    foreign = _ForeignElement(parent=object(), value=3)

    with pytest.raises(DomainConstructionError) as caught:
        ZZ().element(foreign)

    assert caught.value.domain is ZZ()
    assert caught.value.reason == (
        "cannot implicitly import an element from another parent"
    )


def test_zz_elements_are_immutable_parent_aware_and_hash_consistent() -> None:
    """Element equality is canonical inside ``ZZ`` and never coerces Python ints."""
    left = ZZ().element(23)
    same = ZZ().element(23)
    different = ZZ().element(29)
    native_integer: object = 23

    assert left == same
    assert hash(left) == hash(same)
    assert left != different
    assert left != native_integer
    assert {left: "canonical"}[same] == "canonical"
    with pytest.raises(FrozenInstanceError):
        left.value = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        left.parent = ZZ()  # type: ignore[misc]


def test_zz_element_does_not_claim_later_arithmetic_surface() -> None:
    """Arithmetic remains owned by a later exact-domain task."""
    assert getattr(type(ZZ().element(1)), "__add__", None) is None
