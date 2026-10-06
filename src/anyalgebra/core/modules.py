"""Immutable basis and finite free-module parent metadata.

This slice defines ordered bases, finite free-module parents, their stable
semantic fingerprints, and the parent-owned sparse construction entry points.
Module arithmetic belongs to a later task.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .domains import Domain
from .errors import AnyAlgebraError

if TYPE_CHECKING:
    from .coercions import CoercionGraph
    from .elements import SparseElement
    from .parents import SemanticHash


class BasisDefinitionError(AnyAlgebraError, ValueError):
    """A basis declaration or its coefficient parent was malformed."""

    def __init__(
        self,
        reason: str,
        *,
        index: int | None = None,
        conflicting_index: int | None = None,
        coefficient_domain: object | None = None,
    ) -> None:
        """Record deterministic declaration locations without arbitrary reprs."""
        self.reason = reason
        self.index = index
        self.conflicting_index = conflicting_index
        self.coefficient_domain = coefficient_domain
        detail = reason
        if index is not None and conflicting_index is not None:
            detail = f"{reason} at declared indices {conflicting_index} and {index}"
        elif index is not None:
            detail = f"{reason} at declared index {index}"
        super().__init__(detail)


class BasisLabelNotFoundError(AnyAlgebraError, LookupError):
    """A label query did not identify a member of an ordered basis."""

    def __init__(self, basis: Basis, label: object, reason: str) -> None:
        """Preserve the basis and query while keeping the message stable."""
        self.basis = basis
        self.label = label
        self.reason = reason
        super().__init__(reason)


class FreeModuleDefinitionError(AnyAlgebraError, ValueError):
    """A finite free-module declaration was malformed or inconsistent."""

    def __init__(
        self,
        reason: str,
        *,
        domain: object | None = None,
        basis: object | None = None,
        name: object | None = None,
    ) -> None:
        """Preserve rejected metadata while keeping the message deterministic."""
        self.reason = reason
        self.domain = domain
        self.basis = basis
        self.name = name
        super().__init__(reason)


def _label_tuple(labels: Iterable[str]) -> tuple[object, ...]:
    """Consume a label declaration exactly once and normalize iteration errors."""
    if type(labels) is str:
        raise BasisDefinitionError(
            "basis labels must be an iterable of labels, not one str"
        )
    try:
        return tuple(labels)
    except (TypeError, RuntimeError) as error:
        raise BasisDefinitionError("basis labels must be an iterable") from error


def _is_domain_instance(candidate: object) -> bool:
    """Recognize a literal structural domain value, never a domain class."""
    if isinstance(candidate, type):
        return False
    try:
        return (
            isinstance(candidate, Domain)
            and callable(candidate.normalize)
            and callable(candidate.element)
        )
    except Exception:
        return False


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class Basis:
    """An immutable ordered finite basis over one exact coefficient parent.

    Rank zero is valid: it is the basis of the zero free module.  Labels use
    declared order, must be unique exact built-in strings, and are frozen as a
    tuple.  The supplied coefficient-domain instance is retained literally.

    Basis equality therefore compares labels structurally and the coefficient
    parent by identity.  The class is deliberately unhashable because the
    neutral :class:`Domain` protocol does not promise a sound stable hash for
    every exact parent.
    """

    labels: tuple[str, ...]
    coefficient_domain: Domain[object]
    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self,
        labels: Iterable[str],
        *,
        coefficient_domain: Domain[object],
    ) -> None:
        """Validate and freeze labels while preserving the literal domain."""
        if not _is_domain_instance(coefficient_domain):
            raise BasisDefinitionError(
                "coefficient domain must be an exact Domain instance",
                coefficient_domain=coefficient_domain,
            )

        raw_labels = _label_tuple(labels)
        validated: list[str] = []
        for index, label in enumerate(raw_labels):
            if type(label) is not str:
                raise BasisDefinitionError(
                    "basis label must be an exact built-in str", index=index
                )
            if not label:
                raise BasisDefinitionError("basis label must not be empty", index=index)
            if label != label.strip():
                raise BasisDefinitionError(
                    "basis label must not have outer whitespace", index=index
                )
            for prior_index, prior in enumerate(validated):
                if prior == label:
                    raise BasisDefinitionError(
                        "duplicate basis label",
                        index=index,
                        conflicting_index=prior_index,
                    )
            validated.append(label)

        object.__setattr__(self, "labels", tuple(validated))
        object.__setattr__(self, "coefficient_domain", coefficient_domain)

    @property
    def rank(self) -> int:
        """Return the number of declared basis vectors."""
        return len(self.labels)

    def __len__(self) -> int:
        """Return the finite rank."""
        return len(self.labels)

    def __iter__(self) -> Iterator[str]:
        """Iterate labels in declared order."""
        return iter(self.labels)

    def __getitem__(self, index: int) -> str:
        """Return a label by built-in integer index, including negative indices."""
        if type(index) is not int:
            raise TypeError("basis index must be a built-in int, excluding bool")
        try:
            return self.labels[index]
        except IndexError as error:
            raise IndexError("basis index out of range") from error

    def index(self, label: str) -> int:
        """Return a label's declared index or a typed missing-label diagnostic."""
        if type(label) is not str:
            raise BasisLabelNotFoundError(
                self, label, "label lookup requires an exact built-in str"
            )
        try:
            return self.labels.index(label)
        except ValueError as error:
            raise BasisLabelNotFoundError(
                self, label, "label is not present"
            ) from error

    def index_for_label(self, label: str) -> int:
        """Return a label's declared index using the explicit lookup name."""
        return self.index(label)

    def __eq__(self, other: object) -> bool:
        """Compare labels only when both bases retain the same exact parent."""
        return (
            type(other) is Basis
            and self.coefficient_domain is other.coefficient_domain
            and self.labels == other.labels
        )

    def __repr__(self) -> str:
        """Render stable defining metadata without a parent object address."""
        domain_type = type(self.coefficient_domain)
        domain_name = f"{domain_type.__module__}.{domain_type.__qualname__}"
        return f"Basis(labels={self.labels!r}, coefficient_domain={domain_name})"


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class FreeModule:
    """An immutable finite free-module parent over one exact scalar domain.

    The supplied domain and basis instances are retained literally.  Separate
    ``FreeModule`` instances compare structurally when they have equal basis
    declarations over the same domain object.  The optional name is display
    metadata and is deliberately nonsemantic.
    Equal-looking but distinct domain instances therefore never merge module
    parents.  Modules remain universally unhashable; callers use the explicit
    stable :meth:`fingerprint` value for content-addressed identity.

    Element construction is parent-owned through :meth:`element` and
    :meth:`zero`.  Algebra operations remain deliberately deferred.
    """

    domain: Domain[object]
    basis: Basis
    name: str | None
    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self,
        domain: Domain[object],
        basis: Basis,
        *,
        name: str | None = None,
    ) -> None:
        """Validate defining metadata and retain the exact supplied instances."""
        if not _is_domain_instance(domain):
            raise FreeModuleDefinitionError(
                "module domain must be an exact Domain instance",
                domain=domain,
                basis=basis,
                name=name,
            )
        if type(basis) is not Basis:
            raise FreeModuleDefinitionError(
                "module basis must be an exact Basis object",
                domain=domain,
                basis=basis,
                name=name,
            )
        if basis.coefficient_domain is not domain:
            raise FreeModuleDefinitionError(
                "basis coefficient domain must be the module domain",
                domain=domain,
                basis=basis,
                name=name,
            )
        if name is not None:
            if type(name) is not str:
                raise FreeModuleDefinitionError(
                    "module name must be an exact built-in str or None",
                    domain=domain,
                    basis=basis,
                    name=name,
                )
            if not name:
                raise FreeModuleDefinitionError(
                    "module name must not be empty",
                    domain=domain,
                    basis=basis,
                    name=name,
                )
            if name != name.strip():
                raise FreeModuleDefinitionError(
                    "module name must not have outer whitespace",
                    domain=domain,
                    basis=basis,
                    name=name,
                )

        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "name", name)

    @property
    def coefficient_domain(self) -> Domain[object]:
        """Return the exact scalar parent under its explicit metadata name."""
        return self.domain

    @property
    def rank(self) -> int:
        """Return the number of basis vectors."""
        return self.basis.rank

    @property
    def dimension(self) -> int:
        """Return the finite rank under the common dimension name."""
        return self.basis.rank

    def __len__(self) -> int:
        """Return the finite rank."""
        return self.basis.rank

    def __eq__(self, other: object) -> bool:
        """Compare semantic data while preserving literal domain identity."""
        return (
            type(other) is FreeModule
            and self.domain is other.domain
            and self.basis == other.basis
        )

    def fingerprint(self) -> SemanticHash:
        """Return a fresh stable hash of canonical semantic defining data.

        The local import keeps the parent-construction module boundary acyclic:
        ``parents`` owns hash policy while this value only supplies its already
        validated immutable definition.
        """
        from .parents import _free_module_fingerprint

        return _free_module_fingerprint(self, self.domain, self.basis.labels)

    def element(
        self,
        coordinates: Mapping[int, object] | SparseElement,
        *,
        graph: CoercionGraph | None = None,
    ) -> SparseElement:
        """Construct or idempotently retain one canonical sparse element."""
        from .elements import construct_sparse_element

        return construct_sparse_element(self, coordinates, graph=graph)

    def zero(self) -> SparseElement:
        """Return the canonical empty-support element of this literal parent."""
        return self.element({})

    def __repr__(self) -> str:
        """Render defining metadata without using a domain object's repr."""
        domain_type = type(self.domain)
        domain_name = f"{domain_type.__module__}.{domain_type.__qualname__}"
        return (
            f"FreeModule(domain={domain_name}, basis={self.basis!r}, "
            f"name={self.name!r})"
        )
