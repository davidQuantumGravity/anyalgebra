"""Immutable v1 records for bounded finite-algebra atlas artifacts.

The model stores content addresses, representative tables, and exact accounting
equations.  It deliberately does not load files or trust an atlas as evidence;
later builder, query, and replay layers must resolve and verify every link.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
from typing import cast

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import SemanticHash


_SCHEMA_TYPE = "anyalgebra.atlas.finite_algebra"
_SCHEMA_VERSION = 1
_MAX_CORPORA = 256
_MAX_REPRESENTATIVES = 4_096
_MAX_LINKS = 4_096
_MAX_OUTPUTS = 65_536
_MAX_TEXT = 256


class AtlasError(AnyAlgebraError, ValueError):
    """An atlas record violated the bounded v1 schema."""

    def __init__(self, *, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid finite algebra atlas {field}: {reason}")


def _text(value: object, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise AtlasError(field=field, reason="must be nonempty stable text")
    if len(value) > _MAX_TEXT or any("\ud800" <= char <= "\udfff" for char in value):
        raise AtlasError(field=field, reason="text limit or Unicode scalar violation")
    return value


def _count(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise AtlasError(field=field, reason="must be a nonnegative exact int")
    return value


def _snapshot(value: object, field: str, maximum: int) -> tuple[object, ...]:
    if isinstance(value, str | bytes):
        raise AtlasError(field=field, reason="must be an iterable")
    try:
        iterator = iter(cast(Iterable[object], value))
    except Exception as error:
        raise AtlasError(field=field, reason="must be an iterable") from error
    result: list[object] = []
    try:
        for _ in range(maximum + 1):
            result.append(next(iterator))
    except StopIteration:
        return tuple(result)
    except Exception as error:
        raise AtlasError(field=field, reason="iterable snapshot failed") from error
    raise AtlasError(field=field, reason="item limit exceeded")


def _hash(value: object, field: str) -> SemanticHash:
    if type(value) is not SemanticHash:
        raise AtlasError(field=field, reason="must be an exact SemanticHash")
    return SemanticHash(value.algorithm, value.digest)


def _hashes(value: object, field: str, *, nonempty: bool) -> tuple[SemanticHash, ...]:
    items = tuple(_hash(item, field) for item in _snapshot(value, field, _MAX_LINKS))
    if nonempty and not items:
        raise AtlasError(field=field, reason="must not be empty")
    if len(set(items)) != len(items):
        noun = "certificate" if field == "certificate_ids" else "content-address"
        raise AtlasError(field=field, reason=f"contains duplicate {noun}")
    return tuple(sorted(items, key=str))


def _outputs(value: object) -> tuple[int, ...]:
    items = _snapshot(value, "representative_outputs", _MAX_OUTPUTS)
    if any(type(item) is not int or item < 0 for item in items):
        raise AtlasError(
            field="representative_outputs",
            reason="must contain nonnegative exact ints",
        )
    return cast(tuple[int, ...], items)


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class AtlasRepresentative:
    """One canonical table and the hashes of its certificates and analyses."""

    canonical_id: SemanticHash
    representative_outputs: tuple[int, ...]
    full_orbit_size: int
    accepted_member_count: int
    certificate_ids: tuple[SemanticHash, ...]
    analysis_ids: tuple[SemanticHash, ...]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AtlasError(field="representative", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("AtlasRepresentative cannot be subclassed")

    @classmethod
    def create(
        cls,
        *,
        canonical_id: object,
        representative_outputs: object,
        full_orbit_size: object,
        accepted_member_count: object,
        certificate_ids: object,
        analysis_ids: object,
    ) -> AtlasRepresentative:
        if cls is not AtlasRepresentative:
            raise AtlasError(
                field="representative", reason="factory requires exact type"
            )
        full = _count(full_orbit_size, "full_orbit_size")
        accepted = _count(accepted_member_count, "accepted_member_count")
        if full == 0 or accepted == 0 or accepted > full:
            raise AtlasError(
                field="orbit_counts",
                reason="must satisfy 0 < accepted_member_count <= full_orbit_size",
            )
        certificates = _hashes(certificate_ids, "certificate_ids", nonempty=True)
        if len(certificates) != accepted:
            raise AtlasError(
                field="certificate_ids",
                reason="must provide one certificate per accepted member",
            )
        value = object.__new__(AtlasRepresentative)
        for field, item in (
            ("canonical_id", _hash(canonical_id, "canonical_id")),
            ("representative_outputs", _outputs(representative_outputs)),
            ("full_orbit_size", full),
            ("accepted_member_count", accepted),
            ("certificate_ids", certificates),
            ("analysis_ids", _hashes(analysis_ids, "analysis_ids", nonempty=True)),
        ):
            object.__setattr__(value, field, item)
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is AtlasRepresentative and _representative_body(
            self
        ) == _representative_body(other)


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class AtlasCorpus:
    """One census corpus with exact enumeration and orbit accounting."""

    corpus_id: str
    spec_id: SemanticHash
    enumeration_id: SemanticHash
    classification_id: SemanticHash | None
    receipt_id: SemanticHash
    total_candidate_count: int
    examined_candidate_count: int
    accepted_labeled_count: int
    rejected_labeled_count: int
    orbit_count: int
    representatives: tuple[AtlasRepresentative, ...]
    complete: bool

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AtlasError(field="corpus", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("AtlasCorpus cannot be subclassed")

    def __eq__(self, other: object) -> bool:
        return type(other) is AtlasCorpus and _corpus_body(self) == _corpus_body(other)


def create_atlas_corpus(
    *,
    corpus_id: object,
    spec_id: object,
    enumeration_id: object,
    classification_id: object,
    receipt_id: object,
    total_candidate_count: object,
    examined_candidate_count: object,
    accepted_labeled_count: object,
    rejected_labeled_count: object,
    representatives: object,
    complete: object,
) -> AtlasCorpus:
    """Validate one corpus and its complete or bounded-prefix equations."""
    total = _count(total_candidate_count, "total_candidate_count")
    examined = _count(examined_candidate_count, "examined_candidate_count")
    accepted = _count(accepted_labeled_count, "accepted_labeled_count")
    rejected = _count(rejected_labeled_count, "rejected_labeled_count")
    if type(complete) is not bool:
        raise AtlasError(field="complete", reason="must be an exact bool")
    if examined > total:
        raise AtlasError(field="accounting", reason="examined cannot exceed total")
    if examined != accepted + rejected:
        raise AtlasError(
            field="accounting", reason="accepted plus rejected must equal examined"
        )
    raw_representatives = _snapshot(
        representatives, "representatives", _MAX_REPRESENTATIVES
    )
    if any(type(item) is not AtlasRepresentative for item in raw_representatives):
        raise AtlasError(
            field="representatives", reason="must contain exact representative records"
        )
    ordered = tuple(
        sorted(
            cast(tuple[AtlasRepresentative, ...], raw_representatives),
            key=lambda item: str(item.canonical_id),
        )
    )
    if len({item.canonical_id for item in ordered}) != len(ordered):
        raise AtlasError(
            field="representatives", reason="contains duplicate canonical id"
        )
    if complete:
        if sum(item.accepted_member_count for item in ordered) != accepted:
            raise AtlasError(
                field="accounting",
                reason="representatives do not account for accepted tables",
            )
        if examined != total:
            raise AtlasError(
                field="complete", reason="complete corpus must examine every candidate"
            )
        checked_classification = _hash(classification_id, "classification_id")
    else:
        if classification_id is not None or ordered:
            raise AtlasError(
                field="complete",
                reason="incomplete corpus cannot carry classified orbits",
            )
        checked_classification = None
    value = object.__new__(AtlasCorpus)
    for field, item in (
        ("corpus_id", _text(corpus_id, "corpus_id")),
        ("spec_id", _hash(spec_id, "spec_id")),
        ("enumeration_id", _hash(enumeration_id, "enumeration_id")),
        ("classification_id", checked_classification),
        ("receipt_id", _hash(receipt_id, "receipt_id")),
        ("total_candidate_count", total),
        ("examined_candidate_count", examined),
        ("accepted_labeled_count", accepted),
        ("rejected_labeled_count", rejected),
        ("orbit_count", len(ordered)),
        ("representatives", ordered),
        ("complete", complete),
    ):
        object.__setattr__(value, field, item)
    return value


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class AtlasHeader:
    """Derived atlas-wide status and cross-record link inventory."""

    schema_version: int
    status: str
    corpus_count: int
    representative_count: int
    evidence_ids: tuple[SemanticHash, ...]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AtlasError(field="header", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("AtlasHeader cannot be subclassed")

    @classmethod
    def _create(cls, corpora: tuple[AtlasCorpus, ...]) -> AtlasHeader:
        links: set[SemanticHash] = set()
        for corpus in corpora:
            links.update((corpus.spec_id, corpus.enumeration_id, corpus.receipt_id))
            if corpus.classification_id is not None:
                links.add(corpus.classification_id)
            for representative in corpus.representatives:
                links.update(representative.certificate_ids)
                links.update(representative.analysis_ids)
        value = object.__new__(AtlasHeader)
        for field, item in (
            ("schema_version", _SCHEMA_VERSION),
            (
                "status",
                "complete" if all(item.complete for item in corpora) else "incomplete",
            ),
            ("corpus_count", len(corpora)),
            ("representative_count", sum(item.orbit_count for item in corpora)),
            ("evidence_ids", tuple(sorted(links, key=str))),
        ):
            object.__setattr__(value, field, item)
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is AtlasHeader and _header_body(self) == _header_body(other)


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class FiniteAlgebraAtlas:
    """One bounded atlas document with a derived content address."""

    header: AtlasHeader
    corpora: tuple[AtlasCorpus, ...]
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AtlasError(field="atlas", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("FiniteAlgebraAtlas cannot be subclassed")

    @classmethod
    def create(cls, corpora: object) -> FiniteAlgebraAtlas:
        if cls is not FiniteAlgebraAtlas:
            raise AtlasError(field="atlas", reason="factory requires exact type")
        raw = _snapshot(corpora, "corpora", _MAX_CORPORA)
        if not raw:
            raise AtlasError(field="corpora", reason="must contain a corpus")
        if any(type(item) is not AtlasCorpus for item in raw):
            raise AtlasError(
                field="corpora", reason="must contain exact corpus records"
            )
        ordered = tuple(
            sorted(cast(tuple[AtlasCorpus, ...], raw), key=lambda item: item.corpus_id)
        )
        if len({item.corpus_id for item in ordered}) != len(ordered):
            raise AtlasError(field="corpora", reason="contains duplicate corpus id")
        if len({item.spec_id for item in ordered}) != len(ordered):
            raise AtlasError(
                field="corpora", reason="contains duplicate specification id"
            )
        value = object.__new__(FiniteAlgebraAtlas)
        object.__setattr__(value, "header", AtlasHeader._create(ordered))
        object.__setattr__(value, "corpora", ordered)
        object.__setattr__(value, "semantic_hash", SemanticHash("sha256", "0" * 64))
        object.__setattr__(value, "semantic_hash", _semantic_hash(_atlas_body(value)))
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is FiniteAlgebraAtlas and (
            self.header,
            self.corpora,
            self.semantic_hash,
        ) == (other.header, other.corpora, other.semantic_hash)

    def __hash__(self) -> int:
        return int.from_bytes(
            bytes.fromhex(self.semantic_hash.digest[:16]), "big", signed=True
        )


def _hash_body(value: SemanticHash) -> dict[str, str]:
    return {"algorithm": value.algorithm, "digest": value.digest}


def _representative_body(value: AtlasRepresentative) -> dict[str, object]:
    return {
        "acceptedMemberCount": value.accepted_member_count,
        "analysisIds": [_hash_body(item) for item in value.analysis_ids],
        "canonicalId": _hash_body(value.canonical_id),
        "certificateIds": [_hash_body(item) for item in value.certificate_ids],
        "fullOrbitSize": value.full_orbit_size,
        "representativeOutputs": list(value.representative_outputs),
    }


def _corpus_body(value: AtlasCorpus) -> dict[str, object]:
    return {
        "acceptedLabeledCount": value.accepted_labeled_count,
        "classificationId": None
        if value.classification_id is None
        else _hash_body(value.classification_id),
        "complete": value.complete,
        "corpusId": value.corpus_id,
        "enumerationId": _hash_body(value.enumeration_id),
        "examinedCandidateCount": value.examined_candidate_count,
        "orbitCount": value.orbit_count,
        "receiptId": _hash_body(value.receipt_id),
        "rejectedLabeledCount": value.rejected_labeled_count,
        "representatives": [
            _representative_body(item) for item in value.representatives
        ],
        "specId": _hash_body(value.spec_id),
        "totalCandidateCount": value.total_candidate_count,
    }


def _header_body(value: AtlasHeader) -> dict[str, object]:
    return {
        "corpusCount": value.corpus_count,
        "evidenceIds": [_hash_body(item) for item in value.evidence_ids],
        "representativeCount": value.representative_count,
        "schemaVersion": value.schema_version,
        "status": value.status,
    }


def _atlas_body(value: FiniteAlgebraAtlas) -> dict[str, object]:
    return {
        "corpora": [_corpus_body(item) for item in value.corpora],
        "header": _header_body(value.header),
        "schemaType": _SCHEMA_TYPE,
        "schemaVersion": _SCHEMA_VERSION,
    }


def _encoded(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _semantic_hash(body: dict[str, object]) -> SemanticHash:
    return SemanticHash("sha256", hashlib.sha256(_encoded(body)).hexdigest())


def _replay(value: FiniteAlgebraAtlas) -> FiniteAlgebraAtlas:
    corpora: list[AtlasCorpus] = []
    for corpus in value.corpora:
        representatives = tuple(
            AtlasRepresentative.create(
                canonical_id=item.canonical_id,
                representative_outputs=item.representative_outputs,
                full_orbit_size=item.full_orbit_size,
                accepted_member_count=item.accepted_member_count,
                certificate_ids=item.certificate_ids,
                analysis_ids=item.analysis_ids,
            )
            for item in corpus.representatives
        )
        corpora.append(
            create_atlas_corpus(
                corpus_id=corpus.corpus_id,
                spec_id=corpus.spec_id,
                enumeration_id=corpus.enumeration_id,
                classification_id=corpus.classification_id,
                receipt_id=corpus.receipt_id,
                total_candidate_count=corpus.total_candidate_count,
                examined_candidate_count=corpus.examined_candidate_count,
                accepted_labeled_count=corpus.accepted_labeled_count,
                rejected_labeled_count=corpus.rejected_labeled_count,
                representatives=representatives,
                complete=corpus.complete,
            )
        )
    return FiniteAlgebraAtlas.create(tuple(corpora))


def atlas_record(value: object) -> dict[str, object]:
    """Replay all equations before returning the canonical v1 record."""
    if type(value) is not FiniteAlgebraAtlas:
        raise AtlasError(field="atlas", reason="must be exact FiniteAlgebraAtlas")
    expected = _replay(value)
    if value != expected:
        raise AtlasError(field="atlas", reason="content drift")
    body = _atlas_body(value)
    if _semantic_hash(body) != value.semantic_hash:
        raise AtlasError(field="semantic_hash", reason="content drift")
    return {**body, "contentHash": _hash_body(value.semantic_hash)}


def atlas_canonical_bytes(value: object) -> bytes:
    """Return deterministic UTF-8 JSON only after complete model replay."""
    return _encoded(atlas_record(value))


__all__ = (
    "AtlasCorpus",
    "AtlasError",
    "AtlasHeader",
    "AtlasRepresentative",
    "FiniteAlgebraAtlas",
    "atlas_canonical_bytes",
    "atlas_record",
    "create_atlas_corpus",
)
