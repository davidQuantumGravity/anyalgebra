"""Exact canonical-orbit grouping for complete reference enumerations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import SemanticHash

from .canonical import CanonicalLabel, canonicalize_operation_table
from .certificates import (
    CanonicalCertificate,
    CanonicalCertificateVerification,
    verify_canonical_certificate,
)
from .filtering import compile_constraint_filter
from .reference import ReferenceEnumeration
from .spec import CensusSpec


_SCHEMA_VERSION = 1


class OrbitClassificationError(AnyAlgebraError, ValueError):
    """A complete-stream or retained orbit equation was invalid."""

    def __init__(self, *, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid orbit classification {field}: {reason}")


def _encoded(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _hash_record(value: SemanticHash) -> dict[str, str]:
    return {"algorithm": value.algorithm, "digest": value.digest}


def _semantic_hash(body: dict[str, object]) -> SemanticHash:
    return SemanticHash("sha256", hashlib.sha256(_encoded(body)).hexdigest())


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class OrbitMember:
    """One accepted labelled table and its independently replayed map."""

    candidate_index: int
    outputs: tuple[int, ...]
    label: CanonicalLabel
    certificate: CanonicalCertificate
    verification: CanonicalCertificateVerification
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OrbitClassificationError(field="member", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("OrbitMember cannot be subclassed")

    @classmethod
    def _create(cls, candidate_index: int, label: CanonicalLabel) -> OrbitMember:
        certificate = CanonicalCertificate.from_label(label)
        verification = verify_canonical_certificate(certificate)
        value = object.__new__(OrbitMember)
        for field, item in (
            ("candidate_index", candidate_index),
            ("outputs", label.source_outputs),
            ("label", label),
            ("certificate", certificate),
            ("verification", verification),
        ):
            object.__setattr__(value, field, item)
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is OrbitMember and (
            self.candidate_index,
            self.outputs,
            self.label,
            self.certificate,
            self.verification,
        ) == (
            other.candidate_index,
            other.outputs,
            other.label,
            other.certificate,
            other.verification,
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class CanonicalOrbit:
    """One canonical representative and all accepted members of its orbit."""

    canonical_id: SemanticHash
    canonical_bytes: bytes
    representative_outputs: tuple[int, ...]
    full_orbit_size: int
    accepted_member_count: int
    members: tuple[OrbitMember, ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OrbitClassificationError(field="orbit", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("CanonicalOrbit cannot be subclassed")

    @classmethod
    def _create(cls, members: tuple[OrbitMember, ...]) -> CanonicalOrbit:
        if not members:
            raise OrbitClassificationError(field="members", reason="must not be empty")
        ordered = tuple(sorted(members, key=lambda member: member.candidate_index))
        first = ordered[0].label
        if any(
            member.label.canonical_id != first.canonical_id
            or member.label.canonical_bytes != first.canonical_bytes
            or member.label.canonical_outputs != first.canonical_outputs
            or member.label.orbit_size != first.orbit_size
            for member in ordered
        ):
            raise OrbitClassificationError(
                field="members", reason="canonical orbit evidence disagrees"
            )
        value = object.__new__(CanonicalOrbit)
        for field, item in (
            ("canonical_id", first.canonical_id),
            ("canonical_bytes", first.canonical_bytes),
            ("representative_outputs", first.canonical_outputs),
            ("full_orbit_size", first.orbit_size),
            ("accepted_member_count", len(ordered)),
            ("members", ordered),
        ):
            object.__setattr__(value, field, item)
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is CanonicalOrbit and (
            self.canonical_id,
            self.canonical_bytes,
            self.representative_outputs,
            self.full_orbit_size,
            self.accepted_member_count,
            self.members,
        ) == (
            other.canonical_id,
            other.canonical_bytes,
            other.representative_outputs,
            other.full_orbit_size,
            other.accepted_member_count,
            other.members,
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class OrbitClassification:
    """Complete accepted-table partition with exact labelled-count equations."""

    enumeration: ReferenceEnumeration
    spec: CensusSpec
    complete: bool
    raw_labeled_count: int
    accepted_labeled_count: int
    rejected_labeled_count: int
    orbit_count: int
    orbits: tuple[CanonicalOrbit, ...]
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OrbitClassificationError(
            field="classification", reason="is factory-owned"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("OrbitClassification cannot be subclassed")

    @classmethod
    def _create(
        cls,
        enumeration: ReferenceEnumeration,
        accepted_count: int,
        orbits: tuple[CanonicalOrbit, ...],
    ) -> OrbitClassification:
        raw = enumeration.emitted_count
        rejected = raw - accepted_count
        if accepted_count != sum(orbit.accepted_member_count for orbit in orbits):
            raise OrbitClassificationError(
                field="accounting", reason="orbit members do not cover accepted tables"
            )
        value = object.__new__(OrbitClassification)
        for field, item in (
            ("enumeration", enumeration),
            ("spec", enumeration.spec),
            ("complete", True),
            ("raw_labeled_count", raw),
            ("accepted_labeled_count", accepted_count),
            ("rejected_labeled_count", rejected),
            ("orbit_count", len(orbits)),
            ("orbits", orbits),
            ("semantic_hash", SemanticHash("sha256", "0" * 64)),
        ):
            object.__setattr__(value, field, item)
        object.__setattr__(value, "semantic_hash", _semantic_hash(_body(value)))
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is OrbitClassification and (
            self.enumeration,
            self.spec,
            self.complete,
            self.raw_labeled_count,
            self.accepted_labeled_count,
            self.rejected_labeled_count,
            self.orbit_count,
            self.orbits,
            self.semantic_hash,
        ) == (
            other.enumeration,
            other.spec,
            other.complete,
            other.raw_labeled_count,
            other.accepted_labeled_count,
            other.rejected_labeled_count,
            other.orbit_count,
            other.orbits,
            other.semantic_hash,
        )

    def __hash__(self) -> int:
        return int.from_bytes(
            bytes.fromhex(self.semantic_hash.digest[:16]), "big", signed=True
        )

    def __repr__(self) -> str:
        return (
            "OrbitClassification("
            f"semantic_hash={str(self.semantic_hash)!r}, "
            f"accepted_labeled_count={self.accepted_labeled_count}, "
            f"orbit_count={self.orbit_count})"
        )


def _member_record(value: OrbitMember) -> dict[str, object]:
    return {
        "candidateIndex": value.candidate_index,
        "outputs": list(value.outputs),
        "witnessPermutation": list(value.label.certificate.permutation.images),
    }


def _orbit_record(value: CanonicalOrbit) -> dict[str, object]:
    return {
        "acceptedMemberCount": value.accepted_member_count,
        "canonicalId": _hash_record(value.canonical_id),
        "fullOrbitSize": value.full_orbit_size,
        "members": [_member_record(member) for member in value.members],
        "representativeOutputs": list(value.representative_outputs),
    }


def _body(value: OrbitClassification) -> dict[str, object]:
    return {
        "acceptedLabeledCount": value.accepted_labeled_count,
        "complete": value.complete,
        "orbitCount": value.orbit_count,
        "orbits": [_orbit_record(orbit) for orbit in value.orbits],
        "rawLabeledCount": value.raw_labeled_count,
        "rejectedLabeledCount": value.rejected_labeled_count,
        "schemaType": "anyalgebra.census.orbit_classification",
        "schemaVersion": _SCHEMA_VERSION,
        "specHash": _hash_record(value.spec.semantic_hash),
    }


def classify_reference_orbits(
    enumeration: ReferenceEnumeration,
) -> OrbitClassification:
    """Filter one complete stream and group every accepted table canonically."""
    if type(enumeration) is not ReferenceEnumeration:
        raise OrbitClassificationError(
            field="enumeration", reason="must be an exact ReferenceEnumeration"
        )
    if not enumeration.complete:
        raise OrbitClassificationError(
            field="enumeration", reason="must be complete before orbit classification"
        )
    compiled = compile_constraint_filter(enumeration.spec)
    grouped: dict[SemanticHash, list[OrbitMember]] = {}
    accepted = 0
    for candidate in enumeration.candidates:
        if not compiled.evaluate(candidate).accepted:
            continue
        accepted += 1
        label = canonicalize_operation_table(
            enumeration.core,
            candidate.outputs,
            enumeration.spec.equivalence,
        )
        grouped.setdefault(label.canonical_id, []).append(
            OrbitMember._create(candidate.candidate_index, label)
        )
    if len(grouped) > enumeration.spec.bounds.max_orbits:
        raise OrbitClassificationError(
            field="orbit_count", reason="declared maximum orbit count exceeded"
        )
    orbits = tuple(
        CanonicalOrbit._create(tuple(grouped[canonical_id]))
        for canonical_id in sorted(grouped, key=str)
    )
    return OrbitClassification._create(enumeration, accepted, orbits)


def orbit_classification_record(value: OrbitClassification) -> dict[str, object]:
    """Replay the full stream classification before returning a fresh record."""
    if type(value) is not OrbitClassification:
        raise OrbitClassificationError(
            field="classification", reason="must be exact OrbitClassification"
        )
    expected = classify_reference_orbits(value.enumeration)
    if value != expected:
        raise OrbitClassificationError(field="classification", reason="content drift")
    body = _body(value)
    semantic_hash = _semantic_hash(body)
    if semantic_hash != value.semantic_hash:
        raise OrbitClassificationError(field="semantic_hash", reason="content drift")
    return {**body, "contentHash": _hash_record(semantic_hash)}


def orbit_classification_canonical_bytes(value: OrbitClassification) -> bytes:
    """Return deterministic UTF-8 canonical JSON for one replayed result."""
    return _encoded(orbit_classification_record(value))


__all__ = (
    "CanonicalOrbit",
    "OrbitClassification",
    "OrbitClassificationError",
    "OrbitMember",
    "classify_reference_orbits",
    "orbit_classification_canonical_bytes",
    "orbit_classification_record",
)
