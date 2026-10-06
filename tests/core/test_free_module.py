"""Tests for immutable finite free-module parents."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass

import pytest

from anyalgebra.core.domains import DomainElement, QQ, ZZ
from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.modules import Basis, FreeModule, FreeModuleDefinitionError


@dataclass(frozen=True)
class _EqualLookingDomain:
    """A valid structural domain whose instances compare equal."""

    name: str = "K"

    def normalize(self, value: object) -> object:
        """Return the fixture's exact payload."""
        return value

    def element(self, value: object) -> DomainElement[object]:
        """Element construction is outside this task's module-only slice."""
        raise NotImplementedError


class _MalformedDomain:
    """An object with non-callable protocol-shaped attributes."""

    normalize = 1
    element = 2


class _BasisSubclass(Basis):
    """A subtype proving that FreeModule requires the exact Basis class."""


class _StringSubclass(str):
    """A subtype proving that module names require exact built-in strings."""


def test_free_module_retains_exact_zz_parent_and_basis_instances() -> None:
    """Construction preserves literal parent metadata and finite rank."""
    domain = ZZ()
    basis = Basis(["e0", "e1", "e2"], coefficient_domain=domain)
    module = FreeModule(domain, basis, name="V")

    assert module.domain is domain
    assert module.coefficient_domain is domain
    assert module.basis is basis
    assert module.name == "V"
    assert module.rank == 3
    assert module.dimension == 3
    assert len(module) == 3


def test_rank_zero_singleton_and_qq_modules_are_valid_boundaries() -> None:
    """Finite free modules include zero, rank-one, and higher-rank cases."""
    domain = QQ()
    zero = FreeModule(domain, Basis((), coefficient_domain=domain))
    line = FreeModule(domain, Basis(("q",), coefficient_domain=domain), name="Q")
    plane = FreeModule(
        domain,
        Basis(("x", "y"), coefficient_domain=domain),
        name="Q2",
    )

    assert (zero.rank, zero.dimension, len(zero), zero.name) == (0, 0, 0, None)
    assert (line.rank, line.dimension, len(line)) == (1, 1, 1)
    assert (plane.rank, plane.dimension, len(plane)) == (2, 2, 2)


def test_domain_and_basis_must_name_the_same_literal_parent() -> None:
    """Equal-looking or merely different coefficient parents cannot mix."""
    left_domain = _EqualLookingDomain()
    right_domain = _EqualLookingDomain()
    assert left_domain == right_domain
    assert left_domain is not right_domain

    basis = Basis(("e",), coefficient_domain=left_domain)
    with pytest.raises(FreeModuleDefinitionError) as caught:
        FreeModule(right_domain, basis)

    assert isinstance(caught.value, AnyAlgebraError)
    assert caught.value.domain is right_domain
    assert caught.value.basis is basis
    assert caught.value.reason == "basis coefficient domain must be the module domain"


@pytest.mark.parametrize("candidate", [None, object(), ZZ, _MalformedDomain()])
def test_free_module_rejects_non_domain_instances(candidate: object) -> None:
    """A class, function, malformed object, or scalar is not a module domain."""
    basis = Basis(("e",), coefficient_domain=ZZ())

    with pytest.raises(FreeModuleDefinitionError, match="exact Domain") as caught:
        FreeModule(candidate, basis)  # type: ignore[arg-type]

    assert caught.value.domain is candidate


@pytest.mark.parametrize("candidate", [None, object(), ("e",), Basis])
def test_free_module_requires_an_exact_basis_object(candidate: object) -> None:
    """Labels, arbitrary objects, and the Basis class are not a Basis value."""
    with pytest.raises(FreeModuleDefinitionError, match="exact Basis") as caught:
        FreeModule(ZZ(), candidate)  # type: ignore[arg-type]

    assert caught.value.basis is candidate


def test_free_module_rejects_a_basis_subclass() -> None:
    """Parent semantics are not inferred from a Basis subtype."""
    domain = ZZ()
    basis = _BasisSubclass(("e",), coefficient_domain=domain)

    with pytest.raises(FreeModuleDefinitionError, match="exact Basis"):
        FreeModule(domain, basis)


@pytest.mark.parametrize(
    "name",
    ["", " ", " V", "V ", 3, True, _StringSubclass("V")],
)
def test_free_module_rejects_malformed_names(name: object) -> None:
    """Optional names are exact, nonempty, and have no outer whitespace."""
    domain = ZZ()
    basis = Basis(("e",), coefficient_domain=domain)

    with pytest.raises(FreeModuleDefinitionError) as caught:
        FreeModule(domain, basis, name=name)  # type: ignore[arg-type]

    assert caught.value.name is name
    assert caught.value.reason


def test_module_equality_is_structural_over_one_literal_domain() -> None:
    """Display names do not alter structural identity over one exact domain."""
    domain = _EqualLookingDomain()
    other_domain = _EqualLookingDomain()
    left_basis = Basis(("e0", "e1"), coefficient_domain=domain)
    right_basis = Basis(["e0", "e1"], coefficient_domain=domain)

    left = FreeModule(domain, left_basis, name="V")
    same_definition = FreeModule(domain, right_basis, name="V")
    different_name = FreeModule(domain, right_basis, name="W")
    different_domain = FreeModule(
        other_domain,
        Basis(("e0", "e1"), coefficient_domain=other_domain),
        name="V",
    )

    assert left is not same_definition
    assert left == same_definition
    assert left == different_name
    assert left != different_domain
    assert left != object()
    with pytest.raises(TypeError, match="unhashable"):
        hash(left)


def test_free_module_is_immutable_and_repr_is_address_free() -> None:
    """Frozen defining data have a stable representation without object ids."""
    domain = _EqualLookingDomain()
    basis = Basis(("e0", "e1"), coefficient_domain=domain)
    module = FreeModule(domain, basis, name="V")

    rendered = repr(module)
    assert rendered == (
        "FreeModule(domain=test_free_module._EqualLookingDomain, "
        "basis=Basis(labels=('e0', 'e1'), "
        "coefficient_domain=test_free_module._EqualLookingDomain), name='V')"
    )
    assert "0x" not in rendered
    with pytest.raises(FrozenInstanceError):
        module.name = "W"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        module.basis = Basis(("f",), coefficient_domain=domain)  # type: ignore[misc]


def test_free_module_exposes_elements_but_no_early_algebra_operations() -> None:
    """Element construction is present while algebra operations remain deferred."""
    domain = ZZ()
    module = FreeModule(domain, Basis(("e",), coefficient_domain=domain))

    assert callable(module.element)
    assert callable(module.zero)
    for attribute in ("multiply", "product", "freeze"):
        assert not hasattr(module, attribute)
    with pytest.raises(TypeError):
        _ = module + module  # type: ignore[operator]
    with pytest.raises(TypeError):
        _ = module * module  # type: ignore[operator]
