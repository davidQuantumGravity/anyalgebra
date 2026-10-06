"""Immutable, content-addressed source and convention evidence records.

This module owns the v0.0 source/convention identity boundary and its strict,
allow-listed v1 record registry.  Its ``*_record`` helpers expose the same
exact record bodies used by canonical hashing and safe registry round trips.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import SemanticHash
from anyalgebra.persistence.canonical import canonical_json
from anyalgebra.persistence.registry import JSONValue, SchemaError, SerializerRegistry


_MAX_DECLARATIONS = 256
_MAX_TEXT_CODE_POINTS = 4_096
_SOURCE_TAG = "anyalgebra.evidence.source_anchor"
_MANIFEST_TAG = "anyalgebra.evidence.convention_manifest"
_RECORD_VERSION = 1
_PROHIBITED_CLAIM_KEYS = frozenset(
    (
        "claim",
        "claim_status",
        "claimStatus",
        "evidence_tier",
        "evidenceTier",
        "mathematical_outcome",
        "mathematicalOutcome",
        "verified",
        "verified_within_domain",
        "verifiedWithinDomain",
    )
)
_MANIFEST_STATUSES = frozenset(("provisional", "accepted", "deprecated", "superseded"))


class EvidenceModelError(AnyAlgebraError, ValueError):
    """One source or convention input failed its closed identity contract."""

    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid evidence model {field}: {reason}")


def _text(value: object, field: str, *, identifier: bool = False) -> str:
    """Accept one bounded exact Unicode scalar string without rendering it."""
    if type(value) is not str:
        raise EvidenceModelError(field, "must be an exact built-in str")
    if not value:
        raise EvidenceModelError(field, "must not be empty")
    if len(value) > _MAX_TEXT_CODE_POINTS:
        raise EvidenceModelError(field, "text limit exceeded")
    if any("\ud800" <= character <= "\udfff" for character in value):
        raise EvidenceModelError(field, "contains an invalid Unicode scalar")
    if identifier and value != value.strip():
        raise EvidenceModelError(field, "must not have leading or trailing whitespace")
    return value


def _hash(value: object, field: str) -> SemanticHash:
    """Require the established exact semantic-hash representation."""
    if type(value) is not SemanticHash:
        raise EvidenceModelError(field, "must be an exact SemanticHash")
    return value


def _items(value: object, field: str) -> tuple[object, ...]:
    """Snapshot one iterable once, accepting 256 and rejecting its 257th item."""
    if type(value) is str:
        raise EvidenceModelError(field, "must be an iterable of declarations")
    try:
        iterator = iter(cast(Iterable[object], value))
    except Exception as error:
        raise EvidenceModelError(field, "must be an iterable") from error
    snapshot: list[object] = []
    try:
        for _ in range(_MAX_DECLARATIONS + 1):
            snapshot.append(next(iterator))
    except StopIteration:
        return tuple(snapshot)
    except Exception as error:
        raise EvidenceModelError(field, "iterable snapshot failed") from error
    raise EvidenceModelError(field, "declaration limit exceeded")


def _basis(value: object) -> tuple[str, ...]:
    """Snapshot the semantically ordered nonempty basis labels exactly once."""
    labels = tuple(_text(item, "basis_order") for item in _items(value, "basis_order"))
    if not labels:
        raise EvidenceModelError("basis_order", "must not be empty")
    if len(set(labels)) != len(labels):
        raise EvidenceModelError("basis_order", "contains duplicate labels")
    return labels


def _declarations(value: object, field: str) -> tuple[tuple[str, str], ...]:
    """Freeze unordered named key/value clauses in canonical key order."""
    pairs: list[tuple[str, str]] = []
    keys: set[str] = set()
    for item in _items(value, field):
        if (
            not isinstance(item, tuple | list)
            or type(item) not in (tuple, list)
            or len(item) != 2
        ):
            raise EvidenceModelError(field, "each declaration must be an exact pair")
        key = _text(item[0], field, identifier=True)
        if key in _PROHIBITED_CLAIM_KEYS:
            raise EvidenceModelError(field, "must not contain a self-promotion claim")
        if key in keys:
            raise EvidenceModelError(field, "contains duplicate keys")
        keys.add(key)
        pairs.append((key, _text(item[1], field)))
    return tuple(sorted(pairs))


def _validate_source_anchor_fields(anchor: SourceAnchor) -> SemanticHash:
    """Validate every stored source field before encoding an adversarial object."""
    _text(anchor.location, "sources", identifier=True)
    _hash(anchor.content_hash, "sources")
    _text(anchor.edition_or_commit, "sources", identifier=True)
    _text(anchor.locator, "sources", identifier=True)
    if anchor.excerpt_hash is not None:
        _hash(anchor.excerpt_hash, "sources")
    return _hash(anchor.semantic_hash, "sources")


def _sources(value: object) -> tuple[SemanticHash, ...]:
    """Validate and snapshot declared source IDs before later iterables run.

    An anchor is accepted only when its strict record round trip and recomputed
    digest agree with its declared content address.  Illicit mutation is
    rejected rather than silently promoted to a different source reference.
    """
    references: list[SemanticHash] = []
    hashes: set[SemanticHash] = set()
    for item in _items(value, "sources"):
        if type(item) is not SourceAnchor:
            raise EvidenceModelError(
                "sources", "must contain exact SourceAnchor values"
            )
        try:
            declared = _validate_source_anchor_fields(item)
            record = source_anchor_record(item)
            canonical_bytes = source_anchor_canonical_bytes(item)
            parsed = EVIDENCE_RECORD_REGISTRY.from_record(record)
            if (
                type(parsed) is not SourceAnchor
                or parsed.canonical_bytes() != canonical_bytes
            ):
                raise EvidenceModelError("sources", "source round trip is not exact")
            reference = SemanticHash(
                "sha256", hashlib.sha256(canonical_bytes).hexdigest()
            )
            if reference != declared:
                raise EvidenceModelError(
                    "sources", "declared source content address does not match record"
                )
        except (
            AttributeError,
            EvidenceModelError,
            SchemaError,
            TypeError,
            ValueError,
        ) as error:
            raise EvidenceModelError(
                "sources", "source anchor integrity rejected"
            ) from error
        if reference in hashes:
            raise EvidenceModelError("sources", "contains duplicate source anchors")
        hashes.add(reference)
        references.append(reference)
    if not references:
        raise EvidenceModelError("sources", "must not be empty")
    return tuple(sorted(references, key=str))


def _record_hash(value: object, field: str) -> SemanticHash:
    """Hash accepted canonical bytes without leaking serializer-resource failures."""
    try:
        canonical_bytes = canonical_json(value, registry=EVIDENCE_RECORD_REGISTRY)
    except SchemaError as error:
        raise EvidenceModelError(field, "canonical record encoding rejected") from error
    return SemanticHash("sha256", hashlib.sha256(canonical_bytes).hexdigest())


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class SourceAnchor:
    """Exact source location plus immutable content identity.

    ``location`` may be a path or URI, but it is never sufficient provenance on
    its own: ``content_hash`` is mandatory and participates in ``semantic_hash``.
    """

    location: str
    content_hash: SemanticHash
    edition_or_commit: str
    locator: str
    excerpt_hash: SemanticHash | None
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise EvidenceModelError("source_anchor", "is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("SourceAnchor cannot be subclassed")

    @classmethod
    def create(
        cls,
        location: object,
        content_hash: object,
        edition_or_commit: object,
        locator: object,
        *,
        excerpt_hash: object | None = None,
    ) -> SourceAnchor:
        """Create a sealed snapshot after exact field validation and hashing."""
        if cls is not SourceAnchor:
            raise EvidenceModelError(
                "source_anchor", "factory requires exact SourceAnchor"
            )
        checked_location = _text(location, "location", identifier=True)
        checked_content = _hash(content_hash, "content_hash")
        checked_edition = _text(edition_or_commit, "edition_or_commit", identifier=True)
        checked_locator = _text(locator, "locator", identifier=True)
        checked_excerpt = (
            None if excerpt_hash is None else _hash(excerpt_hash, "excerpt_hash")
        )
        value = object.__new__(SourceAnchor)
        for name, item in (
            ("location", checked_location),
            ("content_hash", checked_content),
            ("edition_or_commit", checked_edition),
            ("locator", checked_locator),
            ("excerpt_hash", checked_excerpt),
            ("semantic_hash", SemanticHash("sha256", "0" * 64)),
        ):
            object.__setattr__(value, name, item)
        object.__setattr__(value, "semantic_hash", _record_hash(value, "source_anchor"))
        return value

    def to_record(self) -> dict[str, object]:
        """Return a fresh exact source record for later registry registration."""
        return source_anchor_record(self)

    def canonical_bytes(self) -> bytes:
        """Return canonical bytes used to derive this record's semantic hash."""
        return source_anchor_canonical_bytes(self)

    def __repr__(self) -> str:
        """Render only durable, non-payload identity metadata."""
        return f"SourceAnchor(semantic_hash={str(self.semantic_hash)!r})"

    def __eq__(self, other: object) -> bool:
        """Compare defining fields plus declared content address, never callbacks."""
        return type(other) is SourceAnchor and (
            self.location,
            self.content_hash,
            self.edition_or_commit,
            self.locator,
            self.excerpt_hash,
            self.semantic_hash,
        ) == (
            other.location,
            other.content_hash,
            other.edition_or_commit,
            other.locator,
            other.excerpt_hash,
            other.semantic_hash,
        )

    def __hash__(self) -> int:
        """Return a process-independent integer derived from canonical content."""
        return int.from_bytes(
            bytes.fromhex(self.semantic_hash.digest[:16]), "big", signed=True
        )


def _hash_record(value: SemanticHash) -> dict[str, str]:
    """Encode an already validated hash without relying on its display repr."""
    return {"algorithm": value.algorithm, "digest": value.digest}


def source_anchor_record(value: object) -> dict[str, object]:
    """Encode one exact source anchor in the future codec's stable record shape."""
    if type(value) is not SourceAnchor:
        raise EvidenceModelError("source_anchor", "must be an exact SourceAnchor")
    return {
        "schemaType": _SOURCE_TAG,
        "schemaVersion": _RECORD_VERSION,
        "contentHash": _hash_record(value.content_hash),
        "editionOrCommit": value.edition_or_commit,
        "excerptHash": (
            None if value.excerpt_hash is None else _hash_record(value.excerpt_hash)
        ),
        "location": value.location,
        "locator": value.locator,
    }


def source_anchor_canonical_bytes(value: object) -> bytes:
    """Encode one exact source anchor through the canonical serializer boundary."""
    if type(value) is not SourceAnchor:
        raise EvidenceModelError("source_anchor", "must be an exact SourceAnchor")
    return canonical_json(value, registry=EVIDENCE_RECORD_REGISTRY)


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ConventionManifest:
    """Immutable named convention contract with ordered bases and sorted clauses."""

    manifest_id: str
    basis_order: tuple[str, ...]
    coefficient_domain: str
    multiplication_fixture: SemanticHash
    sources: tuple[SemanticHash, ...]
    status: str
    signs: tuple[tuple[str, str], ...]
    involutions: tuple[tuple[str, str], ...]
    normalization: tuple[tuple[str, str], ...]
    indexing: str
    parenthesization: str
    conversions: tuple[tuple[str, str], ...]
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise EvidenceModelError("convention_manifest", "is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("ConventionManifest cannot be subclassed")

    @classmethod
    def create(
        cls,
        manifest_id: object,
        basis_order: object,
        *,
        coefficient_domain: object,
        multiplication_fixture: object,
        sources: object,
        status: object,
        signs: object,
        involutions: object,
        normalization: object,
        indexing: object,
        parenthesization: object,
        conversions: object,
    ) -> ConventionManifest:
        """Create a sealed convention record and hash its accepted canonical bytes."""
        if cls is not ConventionManifest:
            raise EvidenceModelError(
                "convention_manifest", "factory requires exact ConventionManifest"
            )
        checked_id = _text(manifest_id, "manifest_id", identifier=True)
        checked_basis = _basis(basis_order)
        checked_domain = _text(
            coefficient_domain, "coefficient_domain", identifier=True
        )
        checked_fixture = _hash(multiplication_fixture, "multiplication_fixture")
        checked_sources = _sources(sources)
        checked_status = _text(status, "status", identifier=True)
        if checked_status not in _MANIFEST_STATUSES:
            raise EvidenceModelError("status", "must be a manifest lifecycle status")
        checked_signs = _declarations(signs, "signs")
        checked_involutions = _declarations(involutions, "involutions")
        checked_normalization = _declarations(normalization, "normalization")
        checked_indexing = _text(indexing, "indexing", identifier=True)
        checked_parenthesization = _text(
            parenthesization, "parenthesization", identifier=True
        )
        checked_conversions = _declarations(conversions, "conversions")
        return _manifest_instance(
            checked_id,
            checked_basis,
            checked_domain,
            checked_fixture,
            checked_sources,
            checked_status,
            checked_signs,
            checked_involutions,
            checked_normalization,
            checked_indexing,
            checked_parenthesization,
            checked_conversions,
        )

    def to_record(self) -> dict[str, object]:
        """Return a fresh exact convention record for later codec registration."""
        return convention_manifest_record(self)

    def canonical_bytes(self) -> bytes:
        """Return canonical bytes used to derive this manifest's semantic hash."""
        return convention_manifest_canonical_bytes(self)

    def __repr__(self) -> str:
        """Render only durable, non-payload identity metadata."""
        return f"ConventionManifest(semantic_hash={str(self.semantic_hash)!r})"

    def __eq__(self, other: object) -> bool:
        """Compare exact canonical defining data without object identity shortcuts."""
        return type(other) is ConventionManifest and (
            self.manifest_id,
            self.basis_order,
            self.coefficient_domain,
            self.multiplication_fixture,
            self.sources,
            self.status,
            self.signs,
            self.involutions,
            self.normalization,
            self.indexing,
            self.parenthesization,
            self.conversions,
            self.semantic_hash,
        ) == (
            other.manifest_id,
            other.basis_order,
            other.coefficient_domain,
            other.multiplication_fixture,
            other.sources,
            other.status,
            other.signs,
            other.involutions,
            other.normalization,
            other.indexing,
            other.parenthesization,
            other.conversions,
            other.semantic_hash,
        )

    def __hash__(self) -> int:
        """Return a process-independent integer derived from canonical content."""
        return int.from_bytes(
            bytes.fromhex(self.semantic_hash.digest[:16]), "big", signed=True
        )


def _manifest_instance(
    manifest_id: str,
    basis_order: tuple[str, ...],
    coefficient_domain: str,
    multiplication_fixture: SemanticHash,
    sources: tuple[SemanticHash, ...],
    status: str,
    signs: tuple[tuple[str, str], ...],
    involutions: tuple[tuple[str, str], ...],
    normalization: tuple[tuple[str, str], ...],
    indexing: str,
    parenthesization: str,
    conversions: tuple[tuple[str, str], ...],
) -> ConventionManifest:
    """Construct an already validated manifest, including parser-owned source IDs."""
    value = object.__new__(ConventionManifest)
    for name, item in (
        ("manifest_id", manifest_id),
        ("basis_order", basis_order),
        ("coefficient_domain", coefficient_domain),
        ("multiplication_fixture", multiplication_fixture),
        ("sources", sources),
        ("status", status),
        ("signs", signs),
        ("involutions", involutions),
        ("normalization", normalization),
        ("indexing", indexing),
        ("parenthesization", parenthesization),
        ("conversions", conversions),
        ("semantic_hash", SemanticHash("sha256", "0" * 64)),
    ):
        object.__setattr__(value, name, item)
    object.__setattr__(
        value, "semantic_hash", _record_hash(value, "convention_manifest")
    )
    return value


def _clause_record(value: tuple[tuple[str, str], ...]) -> list[dict[str, str]]:
    """Encode one sorted declaration tuple as explicit JSON key/value objects."""
    return [{"key": key, "value": item} for key, item in value]


def _manifest_body(
    manifest_id: str,
    basis_order: tuple[str, ...],
    coefficient_domain: str,
    multiplication_fixture: SemanticHash,
    sources: tuple[SemanticHash, ...],
    status: str,
    signs: tuple[tuple[str, str], ...],
    involutions: tuple[tuple[str, str], ...],
    normalization: tuple[tuple[str, str], ...],
    indexing: str,
    parenthesization: str,
    conversions: tuple[tuple[str, str], ...],
) -> dict[str, object]:
    """Build the full semantic body with order preserved only where meaningful."""
    return {
        "basisOrder": list(basis_order),
        "coefficientDomain": coefficient_domain,
        "conversions": _clause_record(conversions),
        "id": manifest_id,
        "indexing": indexing,
        "involutions": _clause_record(involutions),
        "multiplicationFixture": _hash_record(multiplication_fixture),
        "normalization": _clause_record(normalization),
        "parenthesization": parenthesization,
        "signs": _clause_record(signs),
        "sources": [_hash_record(reference) for reference in sources],
        "status": status,
    }


def convention_manifest_record(value: object) -> dict[str, object]:
    """Encode one exact manifest in the future codec's stable record shape."""
    if type(value) is not ConventionManifest:
        raise EvidenceModelError(
            "convention_manifest", "must be an exact ConventionManifest"
        )
    return {
        "schemaType": _MANIFEST_TAG,
        "schemaVersion": _RECORD_VERSION,
        **_manifest_body(
            value.manifest_id,
            value.basis_order,
            value.coefficient_domain,
            value.multiplication_fixture,
            value.sources,
            value.status,
            value.signs,
            value.involutions,
            value.normalization,
            value.indexing,
            value.parenthesization,
            value.conversions,
        ),
    }


def convention_manifest_canonical_bytes(value: object) -> bytes:
    """Encode one exact manifest through the canonical serializer boundary."""
    if type(value) is not ConventionManifest:
        raise EvidenceModelError(
            "convention_manifest", "must be an exact ConventionManifest"
        )
    return canonical_json(value, registry=EVIDENCE_RECORD_REGISTRY)


def _body(record: dict[str, object]) -> dict[str, JSONValue]:
    """Drop codec-owned tag/version fields from one trusted direct encoder record."""
    return {
        key: cast(JSONValue, item)
        for key, item in record.items()
        if key not in {"schemaType", "schemaVersion"}
    }


def _encode_source(value: SourceAnchor) -> dict[str, JSONValue]:
    """Allow-list the source encoder without accepting a caller-provided record."""
    return _body(source_anchor_record(value))


def _encode_manifest(value: ConventionManifest) -> dict[str, JSONValue]:
    """Allow-list the manifest encoder without accepting a caller-provided record."""
    return _body(convention_manifest_record(value))


def _mapping(value: object, keys: frozenset[str], field: str) -> Mapping[str, object]:
    """Require the exact key set of a registry-snapshotted JSON object."""
    if not isinstance(value, Mapping) or set(value) != keys:
        raise EvidenceModelError(field, "has an invalid record shape")
    return value


def _record_hash_value(value: object, field: str) -> SemanticHash:
    """Parse one exact nested semantic-hash record with no display aliases."""
    record = _mapping(value, frozenset(("algorithm", "digest")), field)
    if type(record["algorithm"]) is not str or type(record["digest"]) is not str:
        raise EvidenceModelError(field, "has an invalid semantic hash")
    try:
        return SemanticHash(record["algorithm"], record["digest"])
    except Exception as error:
        raise EvidenceModelError(field, "has an invalid semantic hash") from error


def _parse_source(record: object) -> SourceAnchor:
    """Strictly parse one source record through the factory-owned boundary."""
    fields = _mapping(
        record,
        frozenset(
            (
                "schemaType",
                "schemaVersion",
                "contentHash",
                "editionOrCommit",
                "excerptHash",
                "location",
                "locator",
            )
        ),
        "source_anchor",
    )
    excerpt = fields["excerptHash"]
    if excerpt is not None and not isinstance(excerpt, Mapping):
        raise EvidenceModelError("excerpt_hash", "has an invalid record shape")
    return SourceAnchor.create(
        fields["location"],
        _record_hash_value(fields["contentHash"], "content_hash"),
        fields["editionOrCommit"],
        fields["locator"],
        excerpt_hash=(
            None if excerpt is None else _record_hash_value(excerpt, "excerpt_hash")
        ),
    )


def _parse_clauses(value: object, field: str) -> tuple[tuple[str, str], ...]:
    """Parse a bounded list of exact declaration objects into sorted pairs."""
    if type(value) is not tuple:
        raise EvidenceModelError(field, "has an invalid record shape")
    pairs = tuple(
        (
            _mapping(item, frozenset(("key", "value")), field)["key"],
            _mapping(item, frozenset(("key", "value")), field)["value"],
        )
        for item in value
    )
    return _declarations(pairs, field)


def _parse_sources(value: object) -> tuple[SemanticHash, ...]:
    """Parse nonempty canonical source-ID references, never source objects."""
    if type(value) is not tuple:
        raise EvidenceModelError("sources", "has an invalid record shape")
    hashes = tuple(_record_hash_value(item, "sources") for item in value)
    if not hashes:
        raise EvidenceModelError("sources", "must not be empty")
    if len(hashes) > _MAX_DECLARATIONS:
        raise EvidenceModelError("sources", "declaration limit exceeded")
    if len(set(hashes)) != len(hashes):
        raise EvidenceModelError("sources", "contains duplicate source anchors")
    return tuple(
        sorted((SemanticHash(item.algorithm, item.digest) for item in hashes), key=str)
    )


def _parse_manifest(record: object) -> ConventionManifest:
    """Strictly parse one manifest record without imports, callbacks, or extras."""
    fields = _mapping(
        record,
        frozenset(
            (
                "schemaType",
                "schemaVersion",
                "id",
                "basisOrder",
                "coefficientDomain",
                "multiplicationFixture",
                "sources",
                "status",
                "signs",
                "involutions",
                "normalization",
                "indexing",
                "parenthesization",
                "conversions",
            )
        ),
        "convention_manifest",
    )
    status = _text(fields["status"], "status", identifier=True)
    if status not in _MANIFEST_STATUSES:
        raise EvidenceModelError("status", "must be a manifest lifecycle status")
    return _manifest_instance(
        _text(fields["id"], "manifest_id", identifier=True),
        _basis(fields["basisOrder"]),
        _text(fields["coefficientDomain"], "coefficient_domain", identifier=True),
        _record_hash_value(fields["multiplicationFixture"], "multiplication_fixture"),
        _parse_sources(fields["sources"]),
        status,
        _parse_clauses(fields["signs"], "signs"),
        _parse_clauses(fields["involutions"], "involutions"),
        _parse_clauses(fields["normalization"], "normalization"),
        _text(fields["indexing"], "indexing", identifier=True),
        _text(fields["parenthesization"], "parenthesization", identifier=True),
        _parse_clauses(fields["conversions"], "conversions"),
    )


# Lowercase namespaced tags and integer v1 are the normative registry encoding;
# older document examples with title-cased tags are illustrative only.
EVIDENCE_RECORD_REGISTRY = (
    SerializerRegistry()
    .with_codec(
        _SOURCE_TAG, _RECORD_VERSION, SourceAnchor, _encode_source, _parse_source
    )
    .with_codec(
        _MANIFEST_TAG,
        _RECORD_VERSION,
        ConventionManifest,
        _encode_manifest,
        _parse_manifest,
    )
)


def evidence_record_registry() -> SerializerRegistry:
    """Return the immutable allow-list used by ``to_record`` and ``from_record``."""
    return EVIDENCE_RECORD_REGISTRY
