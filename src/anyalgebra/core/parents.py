"""Immutable named sorts and ordered finite carriers.

Sorts are small structural values, not interned identities: equal names compare
equal, while a carrier retains the literal sort instance supplied to it.  Later
signatures and parents provide the surrounding semantic identity that keeps
equal-looking structures distinct.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
import hashlib
import json
from typing import NoReturn, Protocol, cast

from .domains import Domain
from .errors import AnyAlgebraError
from .modules import Basis, FreeModule


class ParentFingerprintError(AnyAlgebraError, ValueError):
    """A parent definition cannot produce one canonical semantic hash."""

    def __init__(self, reason: str, *, parent: object | None = None) -> None:
        """Retain the rejected parent without rendering its arbitrary repr."""
        self.reason = reason
        self.parent = parent
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class SemanticHash:
    """An immutable, algorithm-tagged canonical content digest.

    v0.0 admits exactly lowercase SHA-256 hexadecimal digests.  Restricting the
    representation prevents equivalent hashes from acquiring multiple spellings
    and ensures nested parent references have one canonical JSON encoding.
    """

    algorithm: str
    digest: str

    def __post_init__(self) -> None:
        """Validate the exact v0.0 algorithm and lowercase digest encoding."""
        if type(self.algorithm) is not str or self.algorithm != "sha256":
            raise ParentFingerprintError(
                "semantic hash algorithm must be the exact string 'sha256'"
            )
        if type(self.digest) is not str:
            raise ParentFingerprintError(
                "semantic hash digest must be an exact built-in str"
            )
        if len(self.digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.digest
        ):
            raise ParentFingerprintError(
                "semantic hash digest must be 64 lowercase hexadecimal characters"
            )

    def __str__(self) -> str:
        """Render the stable algorithm-qualified reference form."""
        return f"{self.algorithm}:{self.digest}"


class _FingerprintDomain(Protocol):
    """The opt-in semantic identity surface for noncanonical domains."""

    def fingerprint(self) -> object:
        """Return a validated semantic hash."""


def _domain_fingerprint_record(
    domain: Domain[object], *, parent: object
) -> dict[str, object]:
    """Return an allow-listed or explicitly fingerprinted domain record."""
    from .domains import QQ, ZZ

    if domain is ZZ():
        return {
            "schemaType": "anyalgebra.domain.integer",
            "schemaVersion": 1,
        }
    if domain is QQ():
        return {
            "schemaType": "anyalgebra.domain.rational",
            "schemaVersion": 1,
        }

    try:
        fingerprint_method = cast(_FingerprintDomain, domain).fingerprint
    except AttributeError as error:
        raise ParentFingerprintError(
            "coefficient domain has no semantic fingerprint", parent=parent
        ) from error
    except Exception as error:
        raise ParentFingerprintError(
            "coefficient domain fingerprint lookup failed", parent=parent
        ) from error
    if not callable(fingerprint_method):
        raise ParentFingerprintError(
            "coefficient domain fingerprint must be callable", parent=parent
        )
    try:
        domain_hash = fingerprint_method()
    except Exception as error:
        raise ParentFingerprintError(
            "coefficient domain fingerprint failed", parent=parent
        ) from error
    if type(domain_hash) is not SemanticHash:
        raise ParentFingerprintError(
            "coefficient domain fingerprint must be a SemanticHash", parent=parent
        )
    return {
        "fingerprint": {
            "algorithm": domain_hash.algorithm,
            "digest": domain_hash.digest,
        },
        "schemaType": "anyalgebra.domain.reference",
        "schemaVersion": 1,
    }


def _free_module_fingerprint(
    parent: object,
    domain: Domain[object],
    labels: tuple[str, ...],
) -> SemanticHash:
    """Hash the complete v0.0 semantic definition of a finite free module."""
    record = {
        "basis": {"labels": list(labels)},
        "coefficientDomain": _domain_fingerprint_record(domain, parent=parent),
        "schemaType": "anyalgebra.parent.free_module",
        "schemaVersion": 1,
    }
    canonical_bytes = json.dumps(
        record,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return SemanticHash("sha256", hashlib.sha256(canonical_bytes).hexdigest())


@dataclass(frozen=True, slots=True)
class ParentBuildIssue:
    """One deterministic, immutable problem in a staged parent definition."""

    code: str
    field: str
    message: str


class ParentBuildError(AnyAlgebraError, ValueError):
    """All independent issues found while validating one staged parent."""

    def __init__(self, issues: tuple[ParentBuildIssue, ...]) -> None:
        """Retain issues in the builder's documented validation order."""
        self.issues = issues
        codes = ", ".join(issue.code for issue in issues)
        super().__init__(f"parent build failed: {codes}")


class FrozenBuilderError(AnyAlgebraError, RuntimeError):
    """An operation attempted to reuse a successfully consumed builder."""

    def __init__(self, operation: str) -> None:
        """Retain the rejected lifecycle operation without rendering values."""
        self.operation = operation
        super().__init__(f"parent builder is frozen; cannot {operation}")


_UNSET = object()


def _is_domain_instance(candidate: object) -> bool:
    """Recognize the same literal structural domain values as ``FreeModule``."""
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


class ParentBuilder:
    """Mutable staged builder for the current v0.0 ``FreeModule`` parent.

    Domain, basis, and optional name declarations may be supplied in any order.
    :meth:`freeze` reports every independent validation issue in the fixed order
    domain, basis, name, then domain/basis compatibility.  A failed freeze does
    not consume the builder.  A successful freeze constructs one immutable
    module before consuming the builder, after which mutation and repeated
    freeze are rejected.

    This bounded builder does not yet model arbitrary parent operations,
    fingerprints, references, or cyclic build sessions.
    """

    __slots__ = ("_basis", "_domain", "_frozen", "_name")

    def __init__(self) -> None:
        """Create an empty, mutable finite free-module definition."""
        self._domain: object = _UNSET
        self._basis: object = _UNSET
        self._name: object = None
        self._frozen = False

    def _require_mutable(self, operation: str) -> None:
        """Reject every operation after one parent was built successfully."""
        if self._frozen:
            raise FrozenBuilderError(operation)

    def with_domain(self, domain: object) -> ParentBuilder:
        """Stage or replace the proposed coefficient-domain declaration."""
        self._require_mutable("with_domain")
        self._domain = domain
        return self

    def with_basis(self, basis: object) -> ParentBuilder:
        """Stage or replace the proposed ordered-basis declaration."""
        self._require_mutable("with_basis")
        self._basis = basis
        return self

    def with_name(self, name: object | None) -> ParentBuilder:
        """Stage, replace, or clear the optional module display name."""
        self._require_mutable("with_name")
        self._name = name
        return self

    def _issues(self) -> tuple[ParentBuildIssue, ...]:
        """Collect definition issues in a stable order without short-circuiting."""
        issues: list[ParentBuildIssue] = []
        valid_domain = _is_domain_instance(self._domain)
        valid_basis = type(self._basis) is Basis

        if self._domain is _UNSET:
            issues.append(
                ParentBuildIssue(
                    "missing_domain", "domain", "module domain is required"
                )
            )
        elif not valid_domain:
            issues.append(
                ParentBuildIssue(
                    "invalid_domain",
                    "domain",
                    "module domain must be an exact Domain instance",
                )
            )

        if self._basis is _UNSET:
            issues.append(
                ParentBuildIssue("missing_basis", "basis", "module basis is required")
            )
        elif not valid_basis:
            issues.append(
                ParentBuildIssue(
                    "invalid_basis",
                    "basis",
                    "module basis must be an exact Basis object",
                )
            )

        if self._name is not None and (
            type(self._name) is not str
            or not self._name
            or self._name != self._name.strip()
        ):
            issues.append(
                ParentBuildIssue(
                    "invalid_name",
                    "name",
                    (
                        "module name must be an exact nonempty unpadded "
                        "built-in str or None"
                    ),
                )
            )

        if (
            valid_domain
            and valid_basis
            and cast(Basis, self._basis).coefficient_domain is not self._domain
        ):
            issues.append(
                ParentBuildIssue(
                    "domain_basis_mismatch",
                    "basis",
                    "basis coefficient domain must be the module domain",
                )
            )
        return tuple(issues)

    def freeze(self) -> FreeModule:
        """Validate atomically, build one module, then consume this builder."""
        self._require_mutable("freeze")
        issues = self._issues()
        if issues:
            raise ParentBuildError(issues)

        domain = self._domain
        basis = self._basis
        name = self._name
        assert isinstance(domain, Domain)
        assert type(basis) is Basis
        assert name is None or type(name) is str
        parent = FreeModule(domain, basis, name=name)
        self._frozen = True
        return parent


class SortDefinitionError(AnyAlgebraError, ValueError):
    """A sort name failed the exact v0.0 naming contract."""

    def __init__(self, name: object, reason: str) -> None:
        """Preserve the rejected value and a deterministic reason."""
        self.name = name
        self.reason = reason
        super().__init__(f"invalid sort name: {reason}")


class CarrierDefinitionError(AnyAlgebraError, ValueError):
    """A finite carrier definition was malformed or non-unique."""

    def __init__(
        self,
        reason: str,
        *,
        index: int | None = None,
        conflicting_index: int | None = None,
    ) -> None:
        """Record optional declared-order locations without rendering values."""
        self.reason = reason
        self.index = index
        self.conflicting_index = conflicting_index
        detail = reason
        if index is not None and conflicting_index is not None:
            detail = f"{reason} at declared indices {conflicting_index} and {index}"
        elif index is not None:
            detail = f"{reason} at declared index {index}"
        super().__init__(detail)


class CarrierMemberNotFoundError(AnyAlgebraError, LookupError):
    """An item query did not identify a member of the finite carrier."""

    def __init__(
        self,
        sort: Sort,
        member: object,
        reason: str = "item is not a member",
    ) -> None:
        """Preserve the query without placing arbitrary item reprs in messages."""
        self.sort = sort
        self.member = member
        self.reason = reason
        super().__init__(f"{reason} of sort {sort.name!r}")


class CarrierLabelNotFoundError(AnyAlgebraError, LookupError):
    """A label query was unavailable or absent from a finite carrier."""

    def __init__(self, sort: Sort, label: object, reason: str) -> None:
        """Preserve the label query and deterministic diagnostic reason."""
        self.sort = sort
        self.label = label
        self.reason = reason
        super().__init__(f"{reason} for sort {sort.name!r}")


@dataclass(frozen=True, slots=True)
class Sort:
    """An immutable structural name for one sort in a many-sorted signature.

    A sort name is not a global registry key.  Separately constructed sorts
    with the same name are equal structural values, but no constructor interns
    them or substitutes one instance for another.
    """

    name: str

    def __post_init__(self) -> None:
        """Require an exact, nonempty built-in string with no outer whitespace."""
        if type(self.name) is not str:
            raise SortDefinitionError(
                self.name, "expected an exact built-in str, not a subtype"
            )
        if not self.name:
            raise SortDefinitionError(self.name, "name must not be empty")
        if self.name != self.name.strip():
            raise SortDefinitionError(
                self.name, "name must not have leading or trailing whitespace"
            )


class _EqualityFailure(Exception):
    """Private signal that an item's equality result was not truth-valued."""


def _equal(left: object, right: object) -> bool:
    """Compare arbitrary, possibly unhashable items without using a set."""
    if left is right:
        return True
    try:
        result = left == right
        return bool(result)
    except Exception as error:
        raise _EqualityFailure from error


def _as_tuple(values: Iterable[object], *, kind: str) -> tuple[object, ...]:
    """Consume an iterable once and normalize iteration failure."""
    try:
        return tuple(values)
    except (TypeError, RuntimeError) as error:
        raise CarrierDefinitionError(f"carrier {kind} must be an iterable") from error


@dataclass(frozen=True, slots=True, init=False)
class FiniteCarrier:
    """A nonempty ordered finite collection belonging to one named sort.

    Items need only support equality; they need not be hashable.  Consequently
    carriers use structural equality but are deliberately unhashable, even when
    one particular carrier happens to contain only hashable items.  This keeps
    the equality/hash contract uniform across the class.
    """

    items: tuple[object, ...]
    sort: Sort
    labels: tuple[str, ...] | None
    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self,
        items: Iterable[object],
        *,
        sort: Sort,
        labels: Iterable[str] | None = None,
    ) -> None:
        """Validate and freeze one carrier in its declared iteration order."""
        if type(sort) is not Sort:
            raise CarrierDefinitionError("carrier sort must be an exact Sort")

        item_tuple = _as_tuple(items, kind="items")
        if not item_tuple:
            raise CarrierDefinitionError("carrier must contain at least one item")
        self._reject_item_duplicates(item_tuple)

        label_tuple: tuple[str, ...] | None = None
        if labels is not None:
            if type(labels) is str:
                raise CarrierDefinitionError(
                    "carrier labels must be an iterable of labels, not one str"
                )
            raw_labels = _as_tuple(labels, kind="labels")
            if len(raw_labels) != len(item_tuple):
                raise CarrierDefinitionError(
                    "carrier labels must have the same length as carrier items"
                )
            label_tuple = self._validate_labels(raw_labels)

        object.__setattr__(self, "items", item_tuple)
        object.__setattr__(self, "sort", sort)
        object.__setattr__(self, "labels", label_tuple)

    @staticmethod
    def _reject_item_duplicates(items: tuple[object, ...]) -> None:
        """Reject the first repeated item under declared-order equality."""
        for index, item in enumerate(items):
            for prior_index in range(index):
                try:
                    duplicate = _equal(items[prior_index], item)
                except _EqualityFailure as error:
                    raise CarrierDefinitionError(
                        "carrier item equality comparison failed", index=index
                    ) from error
                if duplicate:
                    raise CarrierDefinitionError(
                        "duplicate carrier item",
                        index=index,
                        conflicting_index=prior_index,
                    )

    @staticmethod
    def _validate_labels(labels: tuple[object, ...]) -> tuple[str, ...]:
        """Validate label strings and reject their first declared duplicate."""
        validated: list[str] = []
        for index, label in enumerate(labels):
            if type(label) is not str:
                raise CarrierDefinitionError(
                    "carrier label must be an exact built-in str", index=index
                )
            if not label:
                raise CarrierDefinitionError(
                    "carrier label must not be empty", index=index
                )
            if label != label.strip():
                raise CarrierDefinitionError(
                    "carrier label must not have outer whitespace", index=index
                )
            for prior_index, prior in enumerate(validated):
                if prior == label:
                    raise CarrierDefinitionError(
                        "duplicate carrier label",
                        index=index,
                        conflicting_index=prior_index,
                    )
            validated.append(label)
        return tuple(validated)

    def __len__(self) -> int:
        """Return the finite cardinality."""
        return len(self.items)

    def __iter__(self) -> Iterator[object]:
        """Iterate items in their declared order."""
        return iter(self.items)

    def __contains__(self, member: object) -> bool:
        """Test membership by equality in declared order, without hashing."""
        try:
            self.index(member)
        except CarrierMemberNotFoundError as error:
            if error.reason == "item is not a member":
                return False
            raise
        return True

    def __getitem__(self, index: int) -> object:
        """Return one item by built-in integer index, including negative indices."""
        if type(index) is not int:
            raise TypeError("carrier index must be a built-in int, excluding bool")
        try:
            return self.items[index]
        except IndexError as error:
            raise IndexError("carrier index out of range") from error

    def index(self, member: object) -> int:
        """Return a member's first declared index or a typed diagnostic."""
        for index, item in enumerate(self.items):
            try:
                if _equal(item, member):
                    return index
            except _EqualityFailure as error:
                raise CarrierMemberNotFoundError(
                    self.sort,
                    member,
                    f"item equality comparison failed at declared index {index}",
                ) from error
        raise CarrierMemberNotFoundError(self.sort, member)

    def _raise_no_labels(self, label: object) -> NoReturn:
        """Raise the common diagnostic for an unlabeled carrier."""
        raise CarrierLabelNotFoundError(self.sort, label, "carrier has no labels")

    def index_for_label(self, label: object) -> int:
        """Return a label's declared index or a typed diagnostic."""
        if type(label) is not str:
            raise CarrierLabelNotFoundError(
                self.sort, label, "label lookup requires an exact built-in str"
            )
        if self.labels is None:
            self._raise_no_labels(label)
        for index, candidate in enumerate(self.labels):
            if candidate == label:
                return index
        raise CarrierLabelNotFoundError(self.sort, label, "label is not present")

    def item_for_label(self, label: object) -> object:
        """Return the item aligned with one declared label."""
        return self.items[self.index_for_label(label)]

    def label_for(self, member: object) -> str:
        """Return the label aligned with a carrier member."""
        if self.labels is None:
            self._raise_no_labels(None)
        return self.labels[self.index(member)]
