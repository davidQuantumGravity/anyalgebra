"""Sealed core metadata for a bounded finite-carrier census specification.

The final public ``CensusSpec`` also needs constraint, equivalence, ordering,
bound, convention, serialization, and content-address types owned by later
v0.1 tasks. This module freezes their independent carrier core without exposing
a temporary public ``CensusSpec.create`` signature.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import SemanticHash

if TYPE_CHECKING:
    from .bounds import CensusOrdering, EnumerationBounds
    from .constraints import ConstraintSet
    from .equivalence import EquivalencePolicy


_MAX_CARRIER_SIZE = 256
_MAX_ARITY = 64
_MAX_INPUT_TUPLES = 65_536
_MAX_DISTINGUISHED_ELEMENTS = 256
_MAX_NAME_CODE_POINTS = 256
_MAX_CONVENTION_REFERENCES = 256
_SPEC_SCHEMA_VERSION = 1
_ALGORITHM_CONTRACT_VERSION = "anyalgebra.census.reference.v1"


class CensusSpecError(AnyAlgebraError, ValueError):
    """A census specification field failed a deterministic preflight."""

    def __init__(
        self,
        *,
        field: str,
        reason: str,
        index: int | None = None,
        conflicting_index: int | None = None,
    ) -> None:
        self.field = field
        self.reason = reason
        self.index = index
        self.conflicting_index = conflicting_index
        location = field if index is None else f"{field}[{index}]"
        detail = f"invalid census specification {location}: {reason}"
        if conflicting_index is not None:
            detail = f"{detail} (conflicts with index {conflicting_index})"
        super().__init__(detail)


def _bounded_nonnegative_int(value: object, field: str, maximum: int) -> int:
    if type(value) is not int or value < 0:
        raise CensusSpecError(
            field=field,
            reason="must be a non-negative built-in int, excluding bool",
        )
    if value > maximum:
        raise CensusSpecError(field=field, reason="hard safety limit exceeded")
    return value


def _input_tuple_count(carrier_size: int, arity: int) -> int:
    if arity == 0:
        return 1
    if carrier_size == 0:
        return 0
    count = 1
    for _ in range(arity):
        if count > _MAX_INPUT_TUPLES // carrier_size:
            raise CensusSpecError(
                field="input_tuple_count",
                reason="hard finite-table work limit exceeded",
            )
        count *= carrier_size
    return count


def _corpus_name(value: object) -> str:
    if type(value) is not str or not value:
        raise CensusSpecError(
            field="corpus_name", reason="must be a non-empty exact built-in str"
        )
    if value != value.strip():
        raise CensusSpecError(
            field="corpus_name", reason="must not have outer whitespace"
        )
    if len(value) > _MAX_NAME_CODE_POINTS:
        raise CensusSpecError(field="corpus_name", reason="text limit exceeded")
    if any("\ud800" <= character <= "\udfff" for character in value):
        raise CensusSpecError(
            field="corpus_name", reason="contains an invalid Unicode scalar"
        )
    return value


def _snapshot_distinguished(value: object) -> tuple[object, ...]:
    source = value.items() if isinstance(value, Mapping) else value
    if isinstance(source, str | bytes):
        raise CensusSpecError(
            field="distinguished_elements",
            reason="must be an iterable of exact pairs",
        )
    try:
        iterator = iter(cast(Iterable[object], source))
    except Exception as error:
        raise CensusSpecError(
            field="distinguished_elements", reason="must be an iterable"
        ) from error
    snapshot: list[object] = []
    try:
        for _ in range(_MAX_DISTINGUISHED_ELEMENTS + 1):
            snapshot.append(next(iterator))
    except StopIteration:
        return tuple(snapshot)
    except Exception as error:
        raise CensusSpecError(
            field="distinguished_elements", reason="iterable snapshot failed"
        ) from error
    raise CensusSpecError(
        field="distinguished_elements", reason="declaration limit exceeded"
    )


def _distinguished_elements(
    value: object, carrier_size: int
) -> tuple[tuple[str, int], ...]:
    declarations = _snapshot_distinguished(value)
    checked: list[tuple[str, int]] = []
    names: dict[str, int] = {}
    for index, declaration in enumerate(declarations):
        if type(declaration) is not tuple or len(declaration) != 2:
            raise CensusSpecError(
                field="distinguished_elements",
                reason="declaration must be an exact pair",
                index=index,
            )
        raw_name, raw_element = declaration
        if type(raw_name) is not str or not raw_name:
            raise CensusSpecError(
                field="distinguished_elements",
                reason="name must be a non-empty exact built-in str",
                index=index,
            )
        if raw_name != raw_name.strip():
            raise CensusSpecError(
                field="distinguished_elements",
                reason="name must not have outer whitespace",
                index=index,
            )
        if len(raw_name) > _MAX_NAME_CODE_POINTS:
            raise CensusSpecError(
                field="distinguished_elements",
                reason="name text limit exceeded",
                index=index,
            )
        if any("\ud800" <= character <= "\udfff" for character in raw_name):
            raise CensusSpecError(
                field="distinguished_elements",
                reason="name contains an invalid Unicode scalar",
                index=index,
            )
        if type(raw_element) is not int:
            raise CensusSpecError(
                field="distinguished_elements",
                reason="element index must be a built-in int, excluding bool",
                index=index,
            )
        if raw_element < 0 or raw_element >= carrier_size:
            raise CensusSpecError(
                field="distinguished_elements",
                reason="element index is outside the carrier range",
                index=index,
            )
        prior = names.get(raw_name)
        if prior is not None:
            raise CensusSpecError(
                field="distinguished_elements",
                reason="duplicate name",
                index=index,
                conflicting_index=prior,
            )
        names[raw_name] = index
        checked.append((raw_name, raw_element))
    return tuple(sorted(checked))


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class CensusSpecCore:
    """Immutable carrier metadata shared by every final v0.1 census spec."""

    carrier_size: int
    arity: int
    distinguished_elements: tuple[tuple[str, int], ...]
    corpus_name: str
    input_tuple_count: int
    candidate_count: int
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise CensusSpecError(field="core", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("CensusSpecCore cannot be subclassed")

    @classmethod
    def create(
        cls,
        *,
        carrier_size: int,
        arity: int,
        distinguished_elements: Iterable[tuple[str, int]] | Mapping[str, int] = (),
        corpus_name: str,
    ) -> CensusSpecCore:
        """Validate fixed-cost fields before consuming the point declarations."""
        if cls is not CensusSpecCore:
            raise CensusSpecError(field="core", reason="factory requires exact type")
        checked_size = _bounded_nonnegative_int(
            carrier_size, "carrier_size", _MAX_CARRIER_SIZE
        )
        checked_arity = _bounded_nonnegative_int(arity, "arity", _MAX_ARITY)
        checked_name = _corpus_name(corpus_name)
        input_count = _input_tuple_count(checked_size, checked_arity)
        points = _distinguished_elements(distinguished_elements, checked_size)
        if checked_size == 0 and checked_arity == 0:
            candidate_count = 0
        elif checked_size == 0:
            candidate_count = 1
        else:
            candidate_count = pow(checked_size, input_count)

        core = object.__new__(CensusSpecCore)
        object.__setattr__(core, "carrier_size", checked_size)
        object.__setattr__(core, "arity", checked_arity)
        object.__setattr__(core, "distinguished_elements", points)
        object.__setattr__(core, "corpus_name", checked_name)
        object.__setattr__(core, "input_tuple_count", input_count)
        object.__setattr__(core, "candidate_count", candidate_count)
        return core

    def __eq__(self, other: object) -> bool:
        return type(other) is CensusSpecCore and (
            self.carrier_size,
            self.arity,
            self.distinguished_elements,
            self.corpus_name,
            self.input_tuple_count,
            self.candidate_count,
        ) == (
            other.carrier_size,
            other.arity,
            other.distinguished_elements,
            other.corpus_name,
            other.input_tuple_count,
            other.candidate_count,
        )

    def __repr__(self) -> str:
        return (
            "CensusSpecCore("
            f"carrier_size={self.carrier_size}, "
            f"arity={self.arity}, "
            f"distinguished_count={len(self.distinguished_elements)}, "
            f"corpus_name={self.corpus_name!r}, "
            f"input_tuple_count={self.input_tuple_count}, "
            f"candidate_count_bits={self.candidate_count.bit_length()})"
        )


def _convention_references(value: object) -> tuple[SemanticHash, ...]:
    """Snapshot and canonicalize bounded immutable provenance references."""
    if isinstance(value, str | bytes):
        raise CensusSpecError(
            field="convention_refs", reason="must be an iterable of content hashes"
        )
    try:
        iterator = iter(cast(Iterable[object], value))
    except Exception as error:
        raise CensusSpecError(
            field="convention_refs", reason="must be an iterable"
        ) from error
    references: list[SemanticHash] = []
    try:
        for _ in range(_MAX_CONVENTION_REFERENCES + 1):
            item = next(iterator)
            if type(item) is not SemanticHash:
                raise CensusSpecError(
                    field="convention_refs",
                    reason="must contain exact SemanticHash values",
                    index=len(references),
                )
            references.append(SemanticHash(item.algorithm, item.digest))
    except StopIteration:
        pass
    except CensusSpecError:
        raise
    except Exception as error:
        raise CensusSpecError(
            field="convention_refs", reason="iterable snapshot failed"
        ) from error
    else:
        raise CensusSpecError(
            field="convention_refs", reason="declaration limit exceeded"
        )
    ordered = tuple(sorted(references, key=str))
    if len(set(ordered)) != len(ordered):
        raise CensusSpecError(
            field="convention_refs", reason="contains duplicate content hashes"
        )
    return ordered


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class CensusSpec:
    """Complete immutable contract for one bounded finite-carrier census."""

    core: CensusSpecCore
    carrier_size: int
    arity: int
    constraints: ConstraintSet
    equivalence: EquivalencePolicy
    ordering: CensusOrdering
    bounds: EnumerationBounds
    convention_refs: tuple[SemanticHash, ...]
    schema_version: int
    algorithm_contract_version: str
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise CensusSpecError(field="spec", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("CensusSpec cannot be subclassed")

    @classmethod
    def create(
        cls,
        *,
        carrier_size: int,
        arity: int,
        constraints: ConstraintSet,
        equivalence: EquivalencePolicy,
        ordering: CensusOrdering,
        bounds: EnumerationBounds,
        convention_refs: Iterable[SemanticHash] = (),
    ) -> CensusSpec:
        """Validate every semantic axis and derive its canonical content address."""
        from .bounds import CensusOrdering, EnumerationBounds
        from .constraints import ConstraintSet
        from .equivalence import EquivalencePolicy

        if cls is not CensusSpec:
            raise CensusSpecError(field="spec", reason="factory requires exact type")
        checked_size = _bounded_nonnegative_int(
            carrier_size, "carrier_size", _MAX_CARRIER_SIZE
        )
        checked_arity = _bounded_nonnegative_int(arity, "arity", _MAX_ARITY)
        if type(constraints) is not ConstraintSet:
            raise CensusSpecError(
                field="constraints", reason="must be an exact ConstraintSet"
            )
        if type(equivalence) is not EquivalencePolicy:
            raise CensusSpecError(
                field="equivalence", reason="must be an exact EquivalencePolicy"
            )
        if constraints.core != equivalence.core:
            raise CensusSpecError(
                field="constraints",
                reason="constraints and equivalence must use the same core",
            )
        core = equivalence.core
        if checked_size != core.carrier_size:
            raise CensusSpecError(
                field="carrier_size", reason="does not match the validated core"
            )
        if checked_arity != core.arity:
            raise CensusSpecError(
                field="arity", reason="does not match the validated core"
            )
        if type(ordering) is not CensusOrdering:
            raise CensusSpecError(
                field="ordering", reason="must be an exact CensusOrdering"
            )
        if type(bounds) is not EnumerationBounds:
            raise CensusSpecError(
                field="bounds", reason="must be exact EnumerationBounds"
            )
        references = _convention_references(convention_refs)

        value = object.__new__(CensusSpec)
        for field, item in (
            ("core", core),
            ("carrier_size", checked_size),
            ("arity", checked_arity),
            ("constraints", constraints),
            ("equivalence", equivalence),
            ("ordering", ordering),
            ("bounds", bounds),
            ("convention_refs", references),
            ("schema_version", _SPEC_SCHEMA_VERSION),
            ("algorithm_contract_version", _ALGORITHM_CONTRACT_VERSION),
            ("semantic_hash", SemanticHash("sha256", "0" * 64)),
        ):
            object.__setattr__(value, field, item)
        from .serialization import census_spec_semantic_hash

        object.__setattr__(value, "semantic_hash", census_spec_semantic_hash(value))
        return value

    def to_record(self) -> dict[str, object]:
        """Return a fresh allow-listed content-addressed JSON record."""
        from .serialization import census_spec_record

        return census_spec_record(self)

    def canonical_bytes(self) -> bytes:
        """Return the exact canonical bytes for persistence and replay."""
        from .serialization import census_spec_canonical_bytes

        return census_spec_canonical_bytes(self)

    def __eq__(self, other: object) -> bool:
        return type(other) is CensusSpec and (
            self.core,
            self.constraints,
            self.equivalence,
            self.ordering,
            self.bounds,
            self.convention_refs,
            self.schema_version,
            self.algorithm_contract_version,
            self.semantic_hash,
        ) == (
            other.core,
            other.constraints,
            other.equivalence,
            other.ordering,
            other.bounds,
            other.convention_refs,
            other.schema_version,
            other.algorithm_contract_version,
            other.semantic_hash,
        )

    def __hash__(self) -> int:
        return int.from_bytes(
            bytes.fromhex(self.semantic_hash.digest[:16]), "big", signed=True
        )

    def __repr__(self) -> str:
        return f"CensusSpec(semantic_hash={str(self.semantic_hash)!r})"
