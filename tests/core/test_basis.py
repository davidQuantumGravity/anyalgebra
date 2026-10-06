"""Tests for immutable ordered finite bases over exact coefficient parents."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from collections.abc import Iterator

import pytest

from anyalgebra.core.domains import DomainElement, QQ, ZZ
from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.modules import (
    Basis,
    BasisDefinitionError,
    BasisLabelNotFoundError,
)


class _StringSubclass(str):
    """A string subtype proving that labels require exact built-in strings."""


@dataclass(frozen=True)
class _EqualLookingDomain:
    """A structural domain fixture whose distinct instances compare equal."""

    name: str = "K"

    def normalize(self, value: object) -> object:
        """Return the fixture's already-exact payload."""
        return value

    def element(self, value: object) -> DomainElement[object]:
        """This fixture is never asked to construct elements in this slice."""
        raise NotImplementedError


class _MalformedDomain:
    """An object with non-callable protocol-shaped attributes."""

    normalize = 1
    element = 2


class _OnePassLabels:
    """An iterable that fails if construction consumes it more than once."""

    def __init__(self) -> None:
        self._iterated = False

    def __iter__(self) -> Iterator[str]:
        if self._iterated:
            raise RuntimeError("labels were consumed twice")
        self._iterated = True
        yield "x"
        yield "y"


def test_basis_preserves_order_parent_identity_and_a_tuple_snapshot() -> None:
    """A basis freezes one-pass input while retaining the literal parent."""
    labels = ["one", "i", "j", "k"]
    domain = QQ()
    basis = Basis(labels, coefficient_domain=domain)
    labels.append("extra")

    assert basis.labels == ("one", "i", "j", "k")
    assert basis.coefficient_domain is domain
    assert basis.rank == 4
    assert len(basis) == 4
    assert tuple(basis) == basis.labels
    assert basis[0] == "one"
    assert basis[-1] == "k"
    assert basis.index("j") == 2
    assert basis.index_for_label("i") == 1


def test_basis_consumes_a_general_label_iterable_exactly_once() -> None:
    """Generator-backed declarations are snapshotted in one pass."""
    basis = Basis(_OnePassLabels(), coefficient_domain=ZZ())

    assert basis.labels == ("x", "y")


def test_rank_zero_and_singleton_bases_are_valid_boundaries() -> None:
    """Unlike FiniteCarrier, Basis admits rank zero for the zero free module."""
    empty = Basis((), coefficient_domain=ZZ())
    singleton = Basis(["e"], coefficient_domain=QQ())

    assert empty.rank == 0
    assert len(empty) == 0
    assert tuple(empty) == ()
    assert singleton.rank == 1
    assert singleton[0] == "e"


@pytest.mark.parametrize(
    "labels",
    [
        "xy",
        7,
        [""],
        [" "],
        [" x"],
        ["x "],
        [_StringSubclass("x")],
        [3],
        [True],
    ],
)
def test_basis_rejects_scalar_or_malformed_label_declarations(labels: object) -> None:
    """Every declared label is an exact nonempty unpadded built-in string."""
    with pytest.raises(BasisDefinitionError) as caught:
        Basis(labels, coefficient_domain=ZZ())  # type: ignore[arg-type]

    assert isinstance(caught.value, AnyAlgebraError)
    assert caught.value.reason


def test_first_duplicate_diagnostic_is_deterministic() -> None:
    """The first repeated label reports its first conflicting declaration."""
    with pytest.raises(BasisDefinitionError) as caught:
        Basis(["a", "b", "a", "a"], coefficient_domain=ZZ())

    error = caught.value
    assert error.reason == "duplicate basis label"
    assert error.index == 2
    assert error.conflicting_index == 0
    assert "0 and 2" in str(error)


@pytest.mark.parametrize(
    "candidate", [None, object(), ZZ, _MalformedDomain(), _EqualLookingDomain]
)
def test_basis_rejects_values_that_are_not_coefficient_domain_instances(
    candidate: object,
) -> None:
    """A protocol-shaped class or malformed object is not an exact parent value."""
    with pytest.raises(BasisDefinitionError, match="coefficient domain") as caught:
        Basis(["e"], coefficient_domain=candidate)  # type: ignore[arg-type]

    assert caught.value.coefficient_domain is candidate


def test_equal_looking_domain_instances_remain_semantically_distinct() -> None:
    """Basis equality never substitutes an equal-looking coefficient parent."""
    left_domain = _EqualLookingDomain()
    right_domain = _EqualLookingDomain()
    assert left_domain == right_domain
    assert left_domain is not right_domain

    left = Basis(["e"], coefficient_domain=left_domain)
    same_parent = Basis(("e",), coefficient_domain=left_domain)
    other_parent = Basis(["e"], coefficient_domain=right_domain)

    assert left == same_parent
    assert left != other_parent
    assert left.coefficient_domain is left_domain
    assert other_parent.coefficient_domain is right_domain
    with pytest.raises(TypeError, match="unhashable"):
        hash(left)


def test_basis_lookup_errors_are_typed_and_indices_have_tuple_semantics() -> None:
    """Missing labels and invalid integer indices have deterministic diagnostics."""
    basis = Basis(["a", "b"], coefficient_domain=ZZ())

    with pytest.raises(BasisLabelNotFoundError) as absent:
        basis.index("z")
    assert absent.value.basis is basis
    assert absent.value.label == "z"
    assert absent.value.reason == "label is not present"

    with pytest.raises(BasisLabelNotFoundError, match="built-in str"):
        basis.index(_StringSubclass("a"))
    with pytest.raises(TypeError, match="built-in int"):
        basis[True]
    with pytest.raises(TypeError, match="built-in int"):
        basis[1.0]  # type: ignore[index]
    with pytest.raises(IndexError, match="out of range"):
        basis[2]
    with pytest.raises(IndexError, match="out of range"):
        basis[-3]


def test_basis_is_immutable_and_repr_is_stable_without_addresses() -> None:
    """Frozen basis metadata cannot be mutated and repr avoids object addresses."""
    domain = _EqualLookingDomain()
    basis = Basis(["e0", "e1"], coefficient_domain=domain)

    rendered = repr(basis)
    assert rendered == (
        "Basis(labels=('e0', 'e1'), coefficient_domain=test_basis._EqualLookingDomain)"
    )
    assert "0x" not in rendered
    with pytest.raises(FrozenInstanceError):
        basis.labels = ("f",)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        basis.coefficient_domain = ZZ()  # type: ignore[misc]


def test_basis_has_no_arithmetic_or_module_element_behavior() -> None:
    """V00-020 stops before FreeModule and SparseElement construction."""
    basis = Basis(["e"], coefficient_domain=ZZ())

    assert not hasattr(basis, "element")
    assert not hasattr(basis, "zero")
    with pytest.raises(TypeError):
        _ = basis + basis  # type: ignore[operator]
