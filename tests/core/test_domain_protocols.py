"""Tests for the neutral exact-domain protocol boundary."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from anyalgebra.core.domains import Domain, DomainElement
from anyalgebra.core.errors import (
    AnyAlgebraError,
    CoercionAmbiguityError,
    DomainConstructionError,
    LossyCoercionError,
)


@dataclass(frozen=True)
class _Element:
    """A minimal immutable implementation used to exercise the protocol."""

    parent: object
    value: int


class _Domain:
    """A deliberately small domain implementing only the neutral interface."""

    def normalize(self, value: object) -> int:
        if type(value) is not int:
            raise DomainConstructionError("test", "invalid exact input")
        return value

    def element(self, value: object) -> _Element:
        return _Element(parent=self, value=self.normalize(value))


def test_domain_and_element_protocols_are_runtime_checkable() -> None:
    """Minimal parent-aware immutable values satisfy the neutral contracts."""
    domain = _Domain()
    element = domain.element(3)

    assert isinstance(domain, Domain)
    assert isinstance(element, DomainElement)
    assert element.parent is domain
    assert element.value == 3


def test_domain_construction_error_is_typed_and_preserves_context() -> None:
    """Constructor rejection is distinguishable from ordinary value errors."""
    with pytest.raises(DomainConstructionError) as caught:
        _Domain().element(True)

    error = caught.value
    assert isinstance(error, AnyAlgebraError)
    assert error.domain == "test"
    assert error.reason == "invalid exact input"
    assert "test" in str(error)


@pytest.mark.parametrize(
    ("error_type", "reason"),
    [
        (CoercionAmbiguityError, "two lossless paths"),
        (LossyCoercionError, "projection would discard data"),
    ],
)
def test_coercion_diagnostics_are_typed_and_preserve_reason(
    error_type: type[CoercionAmbiguityError] | type[LossyCoercionError],
    reason: str,
) -> None:
    """Ambiguity and prohibited loss remain distinguishable failure classes."""
    error = error_type(reason)

    assert isinstance(error, AnyAlgebraError)
    assert error.reason == reason
    assert reason in str(error)


def test_protocols_reject_objects_missing_required_boundary_members() -> None:
    """Runtime checks do not mistake an incomplete object for a domain value."""

    class _MissingElementConstructor:
        def normalize(self, value: object) -> object:
            return value

    assert not isinstance(_MissingElementConstructor(), Domain)
    assert not isinstance(object(), DomainElement)
