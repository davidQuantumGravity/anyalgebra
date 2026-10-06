"""Golden and boundary tests for immutable source and convention evidence."""

from __future__ import annotations

import hashlib

import pytest

import anyalgebra.evidence.models as models
from anyalgebra.core.parents import SemanticHash
from anyalgebra.evidence.models import (
    ConventionManifest,
    EVIDENCE_RECORD_REGISTRY,
    EvidenceModelError,
    SourceAnchor,
    convention_manifest_canonical_bytes,
    convention_manifest_record,
    evidence_record_registry,
    source_anchor_canonical_bytes,
    source_anchor_record,
)
from anyalgebra.persistence.registry import SchemaError


def _hash(character: str) -> SemanticHash:
    return SemanticHash("sha256", character * 64)


def _source(**changes: object) -> SourceAnchor:
    values: dict[str, object] = {
        "location": "doi:10.1000/example",
        "content_hash": _hash("a"),
        "edition_or_commit": "rev-7",
        "locator": "p. 13, eq. (4)",
        "excerpt_hash": _hash("b"),
    }
    values.update(changes)
    return SourceAnchor.create(**values)


def _manifest(**changes: object) -> ConventionManifest:
    values: dict[str, object] = {
        "manifest_id": "algmul.O.v1",
        "basis_order": ("1", "e1", "e2"),
        "coefficient_domain": "QQ",
        "multiplication_fixture": _hash("c"),
        "sources": (_source(),),
        "status": "accepted",
        "signs": (("commutator", "xy-yx"),),
        "involutions": (("conjugation", "negate-imaginary"),),
        "normalization": (("long-root-square", "2"),),
        "indexing": "zero-based-storage",
        "parenthesization": "explicit",
        "conversions": (("to.alt", "explicit-map-v1"),),
    }
    values.update(changes)
    return ConventionManifest.create(**values)


def test_source_golden_record_bytes_and_independent_digest() -> None:
    anchor = _source()

    expected = {
        "schemaType": "anyalgebra.evidence.source_anchor",
        "schemaVersion": 1,
        "contentHash": {"algorithm": "sha256", "digest": "a" * 64},
        "editionOrCommit": "rev-7",
        "excerptHash": {"algorithm": "sha256", "digest": "b" * 64},
        "location": "doi:10.1000/example",
        "locator": "p. 13, eq. (4)",
    }
    assert source_anchor_record(anchor) == expected
    assert anchor.canonical_bytes() == source_anchor_canonical_bytes(anchor)
    assert anchor.semantic_hash == SemanticHash(
        "sha256", hashlib.sha256(anchor.canonical_bytes()).hexdigest()
    )
    assert anchor.canonical_bytes() == (
        b'{"contentHash":{"algorithm":"sha256","digest":"'
        + b"a" * 64
        + b'"},"editionOrCommit":"rev-7","excerptHash":{"algorithm":"sha256","digest":"'
        + b"b" * 64
        + b'"},"location":"doi:10.1000/example","locator":"p. 13, eq. (4)",'
        b'"schemaType":"anyalgebra.evidence.source_anchor","schemaVersion":1}'
    )
    assert anchor.semantic_hash.digest == (
        "ec1f5ba6b17aa85d747a25c3ce3e995df45eb5860fb0dec7345490da0460d239"
    )


@pytest.mark.parametrize(
    "changes",
    (
        {"location": "file:///other"},
        {"content_hash": _hash("c")},
        {"edition_or_commit": "rev-8"},
        {"locator": "line 7"},
        {"excerpt_hash": None},
    ),
)
def test_every_source_semantic_field_changes_hash(changes: dict[str, object]) -> None:
    assert _source().semantic_hash != _source(**changes).semantic_hash


def test_location_content_and_excerpt_are_distinct_semantics() -> None:
    baseline = _source()
    assert baseline.semantic_hash != _source(location="doi:10.1000/other").semantic_hash
    assert baseline.semantic_hash != _source(content_hash=_hash("d")).semantic_hash
    assert baseline.semantic_hash != _source(excerpt_hash=_hash("e")).semantic_hash


def test_valid_source_without_excerpt_has_a_declared_manifest_reference() -> None:
    anchor = _source(excerpt_hash=None)
    assert _manifest(sources=(anchor,)).sources == (anchor.semantic_hash,)


def test_source_requires_content_hash_and_exact_hash_type() -> None:
    with pytest.raises(EvidenceModelError, match="content_hash"):
        _source(content_hash=None)

    class HashSubclass(SemanticHash):
        pass

    with pytest.raises(EvidenceModelError, match="content_hash"):
        _source(content_hash=HashSubclass("sha256", "f" * 64))
    with pytest.raises(EvidenceModelError, match="excerpt_hash"):
        _source(excerpt_hash=HashSubclass("sha256", "f" * 64))


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    (
        ("location", 7, "exact built-in str"),
        ("locator", "", "must not be empty"),
        ("edition_or_commit", "x" * 4_097, "text limit"),
        ("locator", "\ud800", "Unicode scalar"),
        ("location", " path", "leading or trailing"),
    ),
)
def test_source_text_boundary_is_exact_bounded_and_unicode_safe(
    field: str, value: object, reason: str
) -> None:
    with pytest.raises(EvidenceModelError, match=reason):
        _source(**{field: value})


def test_manifest_golden_bytes_and_independent_digest() -> None:
    manifest = _manifest()

    assert convention_manifest_record(manifest)["schemaType"] == (
        "anyalgebra.evidence.convention_manifest"
    )
    assert convention_manifest_record(manifest)["coefficientDomain"] == "QQ"
    assert convention_manifest_record(manifest)["multiplicationFixture"] == {
        "algorithm": "sha256",
        "digest": "c" * 64,
    }
    assert convention_manifest_record(manifest)["status"] == "accepted"
    assert manifest.canonical_bytes() == convention_manifest_canonical_bytes(manifest)
    assert manifest.semantic_hash == SemanticHash(
        "sha256", hashlib.sha256(manifest.canonical_bytes()).hexdigest()
    )
    assert manifest.canonical_bytes() == (
        b'{"basisOrder":["1","e1","e2"],"coefficientDomain":"QQ",'
        b'"conversions":[{"key":"to.alt","value":"explicit-map-v1"}],'
        b'"id":"algmul.O.v1","indexing":"zero-based-storage",'
        b'"involutions":[{"key":"conjugation","value":"negate-imaginary"}],'
        b'"multiplicationFixture":{"algorithm":"sha256","digest":"'
        + b"c"
        * 64
        + b'"},"normalization":[{"key":"long-root-square","value":"2"}],'
        b'"parenthesization":"explicit",'
        b'"schemaType":"anyalgebra.evidence.convention_manifest",'
        b'"schemaVersion":1,"signs":[{"key":"commutator","value":"xy-yx"}],'
        b'"sources":[{"algorithm":"sha256","digest":"'
        b"ec1f5ba6b17aa85d747a25c3ce3e995df45eb5860fb0dec7345490da0460d239"
        b'"}],"status":"accepted"}'
    )
    assert manifest.semantic_hash.digest == (
        "f877c45b81fcecc5ac7c19d886791aa82fdb2a21d075c51f2e4bdd1b08d7e7f4"
    )
    assert b" " not in manifest.canonical_bytes()
    assert convention_manifest_record(manifest) == {
        "schemaType": "anyalgebra.evidence.convention_manifest",
        "schemaVersion": 1,
        "id": "algmul.O.v1",
        "basisOrder": ["1", "e1", "e2"],
        "coefficientDomain": "QQ",
        "multiplicationFixture": {"algorithm": "sha256", "digest": "c" * 64},
        "sources": [
            {
                "algorithm": "sha256",
                "digest": _source().semantic_hash.digest,
            }
        ],
        "status": "accepted",
        "signs": [{"key": "commutator", "value": "xy-yx"}],
        "involutions": [{"key": "conjugation", "value": "negate-imaginary"}],
        "normalization": [{"key": "long-root-square", "value": "2"}],
        "indexing": "zero-based-storage",
        "parenthesization": "explicit",
        "conversions": [{"key": "to.alt", "value": "explicit-map-v1"}],
    }


def test_registry_round_trips_exact_tagged_records_without_imports() -> None:
    anchor = _source()
    manifest = _manifest()
    registry = evidence_record_registry()
    assert registry is EVIDENCE_RECORD_REGISTRY
    assert registry.to_record(anchor) == source_anchor_record(anchor)
    assert registry.to_record(manifest) == convention_manifest_record(manifest)
    assert registry.from_record(source_anchor_record(anchor)) == anchor
    assert registry.from_record(convention_manifest_record(manifest)) == manifest
    assert isinstance(registry.from_record(source_anchor_record(anchor)), SourceAnchor)
    assert isinstance(
        registry.from_record(convention_manifest_record(manifest)), ConventionManifest
    )


def test_unordered_clauses_canonicalize_but_ordered_basis_does_not() -> None:
    first = _manifest(
        signs=(("left", "-"), ("right", "+")),
        conversions=(("z", "last"), ("a", "first")),
    )
    reordered_clauses = _manifest(
        signs=(("right", "+"), ("left", "-")),
        conversions=(("a", "first"), ("z", "last")),
    )
    reordered_basis = _manifest(basis_order=("e1", "1", "e2"))

    assert first.semantic_hash == reordered_clauses.semantic_hash
    assert first.signs == (("left", "-"), ("right", "+"))
    assert first.semantic_hash != reordered_basis.semantic_hash


@pytest.mark.parametrize(
    "changes",
    (
        {"manifest_id": "algmul.O.v2"},
        {"coefficient_domain": "ZZ"},
        {"multiplication_fixture": _hash("d")},
        {"sources": (_source(location="file:///other"),)},
        {"status": "deprecated"},
        {"signs": (("commutator", "yx-xy"),)},
        {"involutions": (("conjugation", "identity"),)},
        {"normalization": (("long-root-square", "1"),)},
        {"indexing": "one-based-storage"},
        {"parenthesization": "left"},
        {"conversions": (("to.alt", "other-map"),)},
    ),
)
def test_every_manifest_semantic_field_changes_hash(changes: dict[str, object]) -> None:
    assert _manifest().semantic_hash != _manifest(**changes).semantic_hash


def test_sources_are_unordered_but_must_be_nonempty_exact_and_unique() -> None:
    first = _source(location="a")
    second = _source(location="b")
    assert (
        _manifest(sources=(first, second)).semantic_hash
        == _manifest(sources=(second, first)).semantic_hash
    )
    for sources, reason in (
        ((), "must not be empty"),
        ((first, first), "duplicate"),
        ((object(),), "exact SourceAnchor"),
    ):
        with pytest.raises(EvidenceModelError, match=reason):
            _manifest(sources=sources)
    with pytest.raises(EvidenceModelError, match="duplicate"):
        _manifest(sources=(_source(), _source()))


def test_source_references_snapshot_before_a_later_iterable_can_mutate_anchor() -> None:
    anchor = _source()
    expected = SemanticHash("sha256", anchor.semantic_hash.digest)

    def signs() -> object:
        object.__setattr__(anchor, "location", "file:///mutated")
        yield ("commutator", "xy-yx")

    manifest = _manifest(sources=(anchor,), signs=signs())
    assert manifest.sources == (expected,)
    assert convention_manifest_record(manifest)["sources"] == [
        {"algorithm": "sha256", "digest": expected.digest}
    ]


def test_source_reference_snapshot_translates_tampered_canonical_failure() -> None:
    anchor = _source()
    object.__setattr__(anchor, "location", "x" * 65_537)
    with pytest.raises(EvidenceModelError, match="source anchor integrity rejected"):
        _manifest(sources=(anchor,))


def test_source_integrity_rejects_tampering_before_later_iterables_execute() -> None:
    anchor = _source()
    ran = False

    def signs() -> object:
        nonlocal ran
        ran = True
        yield ("commutator", "xy-yx")

    object.__setattr__(anchor, "location", "doi:10.1000/tampered")
    with pytest.raises(EvidenceModelError, match="source anchor integrity rejected"):
        _manifest(sources=(anchor,), signs=signs())
    assert not ran


@pytest.mark.parametrize(
    ("attribute", "value"),
    (("digest", "A" * 64), ("algorithm", "sha1")),
)
def test_source_integrity_rejects_tampered_nested_hash(
    attribute: str, value: str
) -> None:
    anchor = _source()
    object.__setattr__(anchor.content_hash, attribute, value)
    with pytest.raises(EvidenceModelError, match="source anchor integrity rejected"):
        _manifest(sources=(anchor,))


@pytest.mark.parametrize("attribute", ("content_hash", "excerpt_hash", "semantic_hash"))
def test_source_integrity_sanitizes_wholesale_hash_replacement(
    attribute: str,
) -> None:
    anchor = _source()
    ran = False

    def signs() -> object:
        nonlocal ran
        ran = True
        yield ("commutator", "xy-yx")

    object.__setattr__(anchor, attribute, object())
    with pytest.raises(
        EvidenceModelError, match="source anchor integrity rejected"
    ) as caught:
        _manifest(sources=(anchor,), signs=signs())
    assert not ran
    assert (
        str(caught.value)
        == "invalid evidence model sources: source anchor integrity rejected"
    )
    assert "AttributeError" not in str(caught.value)
    assert "object at" not in str(caught.value)


def test_source_integrity_rejects_a_nonexact_strict_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor = _source()
    snapshots = iter((b"first", b"second"))
    monkeypatch.setattr(
        models, "source_anchor_canonical_bytes", lambda _: next(snapshots)
    )
    with pytest.raises(EvidenceModelError, match="source anchor integrity rejected"):
        _manifest(sources=(anchor,))


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    (
        ("coefficient_domain", "", "must not be empty"),
        ("multiplication_fixture", None, "exact SemanticHash"),
        ("status", "verified", "lifecycle"),
    ),
)
def test_manifest_extra_required_identity_fields_are_checked(
    field: str, value: object, reason: str
) -> None:
    with pytest.raises(EvidenceModelError, match=reason):
        _manifest(**{field: value})


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    (
        ("basis_order", (), "must not be empty"),
        ("basis_order", ("x", "x"), "duplicate"),
        ("signs", (("x", "1"), ("x", "1")), "duplicate"),
        ("signs", (("x", "1"), ("x", "2")), "duplicate"),
        ("normalization", (("verified", "yes"),), "self-promotion"),
        ("conversions", (("bad",),), "exact pair"),
    ),
)
def test_manifest_rejects_empty_duplicate_and_claim_declarations(
    field: str, value: object, reason: str
) -> None:
    with pytest.raises(EvidenceModelError, match=reason):
        _manifest(**{field: value})


def test_iterable_cap_accepts_256_and_rejects_257_without_reiteration() -> None:
    accepted = ((f"s{index}", "+") for index in range(256))
    assert len(_manifest(signs=accepted).signs) == 256
    rejected = ((f"s{index}", "+") for index in range(257))
    with pytest.raises(EvidenceModelError, match="limit"):
        _manifest(signs=rejected)


@pytest.mark.parametrize(
    "field",
    ("basis_order", "sources", "signs", "involutions", "normalization", "conversions"),
)
def test_all_bounded_collections_accept_256_and_reject_257(field: str) -> None:
    if field == "basis_order":
        accepted: object = tuple(f"e{index}" for index in range(256))
        rejected: object = tuple(f"e{index}" for index in range(257))
    elif field == "sources":
        accepted = tuple(_source(location=f"file:///{index}") for index in range(256))
        rejected = tuple(_source(location=f"file:///{index}") for index in range(257))
    else:
        accepted = tuple((f"k{index}", "v") for index in range(256))
        rejected = tuple((f"k{index}", "v") for index in range(257))
    assert _manifest(**{field: accepted})
    with pytest.raises(EvidenceModelError, match="limit"):
        _manifest(**{field: rejected})


def test_malformed_iterables_and_pairs_have_typed_diagnostics() -> None:
    class FailingIterator:
        def __iter__(self) -> FailingIterator:
            return self

        def __next__(self) -> object:
            raise RuntimeError("private failure")

    class PairSubclass(tuple[object, object]):
        pass

    for signs, reason in (
        ("not-declarations", "iterable of declarations"),
        (7, "must be an iterable"),
        (FailingIterator(), "snapshot failed"),
        ((object(),), "exact pair"),
        ((PairSubclass(("x", "y")),), "exact pair"),
    ):
        with pytest.raises(EvidenceModelError, match=reason):
            _manifest(signs=signs)


def test_canonical_resource_rejection_is_an_evidence_error() -> None:
    oversized = tuple((f"key{index}", "x" * 4_096) for index in range(256))
    with pytest.raises(EvidenceModelError, match="canonical record encoding rejected"):
        _manifest(signs=oversized)


def test_record_helpers_reject_wrong_type_and_registry_cannot_decode() -> None:
    for helper in (
        source_anchor_record,
        source_anchor_canonical_bytes,
        convention_manifest_record,
        convention_manifest_canonical_bytes,
    ):
        with pytest.raises(EvidenceModelError, match="exact"):
            helper(object())
    with pytest.raises(SchemaError, match="parser_failed"):
        EVIDENCE_RECORD_REGISTRY.from_record(
            {"schemaType": "anyalgebra.evidence.source_anchor", "schemaVersion": 1}
        )


def test_registry_parsers_reject_extra_bad_nested_and_lifecycle_record_shapes() -> None:
    source = source_anchor_record(_source())
    manifest = convention_manifest_record(_manifest())
    cases = []
    extra_source = {**source, "extra": 1}
    cases.append(extra_source)
    bad_hash = {**source, "contentHash": {"algorithm": "sha256", "digest": 3}}
    cases.append(bad_hash)
    invalid_hash = {
        **source,
        "contentHash": {"algorithm": "sha256", "digest": "A" * 64},
    }
    cases.append(invalid_hash)
    bad_excerpt = {**source, "excerptHash": []}
    cases.append(bad_excerpt)
    extra_manifest = {**manifest, "extra": 1}
    cases.append(extra_manifest)
    bad_clauses = {**manifest, "signs": [{"key": "x", "value": "y", "z": "z"}]}
    cases.append(bad_clauses)
    nonlist_clauses = {**manifest, "signs": "invalid"}
    cases.append(nonlist_clauses)
    bad_sources = {**manifest, "sources": []}
    cases.append(bad_sources)
    nonscalar_sources = {**manifest, "sources": {"algorithm": "sha256"}}
    cases.append(nonscalar_sources)
    assert isinstance(manifest["sources"], list)
    bad_status = {**manifest, "status": "verified"}
    cases.append(bad_status)
    for record in cases:
        with pytest.raises(SchemaError, match="parser_failed"):
            EVIDENCE_RECORD_REGISTRY.from_record(record)
    too_many_sources = tuple(
        {"algorithm": "sha256", "digest": f"{index:064x}"} for index in range(257)
    )
    with pytest.raises(EvidenceModelError, match="limit"):
        models._parse_sources(too_many_sources)
    with pytest.raises(EvidenceModelError, match="duplicate"):
        models._parse_sources(tuple(manifest["sources"] * 2))


def test_factory_exact_class_guards_and_to_record_methods() -> None:
    with pytest.raises(EvidenceModelError, match="factory requires"):
        SourceAnchor.create.__func__(  # type: ignore[attr-defined]
            object, "a", _hash("a"), "b", "c"
        )
    with pytest.raises(EvidenceModelError, match="factory requires"):
        ConventionManifest.create.__func__(  # type: ignore[attr-defined]
            object,
            "id",
            ("e",),
            coefficient_domain="QQ",
            multiplication_fixture=_hash("a"),
            sources=(_source(),),
            status="accepted",
            signs=(),
            involutions=(),
            normalization=(),
            indexing="zero",
            parenthesization="explicit",
            conversions=(),
        )
    assert _source().to_record() == source_anchor_record(_source())
    assert _manifest().to_record() == convention_manifest_record(_manifest())


def test_records_are_factory_owned_immutable_sealed_and_safe() -> None:
    anchor = _source()
    manifest = _manifest()
    with pytest.raises(EvidenceModelError, match="factory-owned"):
        SourceAnchor()
    with pytest.raises(EvidenceModelError, match="factory-owned"):
        ConventionManifest()
    with pytest.raises(AttributeError):
        anchor.location = "changed"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        manifest.basis_order = ()  # type: ignore[misc]
    with pytest.raises(TypeError):
        type("ChildSource", (SourceAnchor,), {})
    with pytest.raises(TypeError):
        type("ChildManifest", (ConventionManifest,), {})
    assert "doi:" not in repr(anchor)
    assert "algmul.O.v1" not in repr(manifest)
    assert "0x" not in repr(anchor)
    assert anchor == _source()
    assert manifest == _manifest()
    assert hash(anchor) == hash(_source())
    assert hash(manifest) == hash(_manifest())
    assert anchor.semantic_hash == _source().semantic_hash
    assert anchor.canonical_bytes() == _source().canonical_bytes()
    assert manifest.semantic_hash == _manifest().semantic_hash
    assert manifest.canonical_bytes() == _manifest().canonical_bytes()


def test_tampered_declared_addresses_do_not_compare_equal_to_valid_records() -> None:
    valid_source = _source()
    stale_source = _source()
    object.__setattr__(stale_source, "semantic_hash", _hash("f"))
    assert stale_source != valid_source
    assert hash(stale_source) != hash(valid_source)
    valid_manifest = _manifest()
    stale_manifest = _manifest()
    object.__setattr__(stale_manifest, "semantic_hash", _hash("e"))
    assert stale_manifest != valid_manifest
    assert hash(stale_manifest) != hash(valid_manifest)


def test_content_hashes_are_equal_for_equals_and_signed_for_both_digest_halves() -> (
    None
):
    high = _source()
    assert high.semantic_hash.digest[0] in "89abcdef"
    assert hash(high) == hash(_source())
    assert hash(_manifest()) == hash(_manifest())
    low = next(
        _source(locator=f"line {index}")
        for index in range(256)
        if _source(locator=f"line {index}").semantic_hash.digest[0] in "01234567"
    )
    assert low.semantic_hash.digest[0] in "01234567"
    assert low.__hash__() == int.from_bytes(
        bytes.fromhex(low.semantic_hash.digest[:16]), "big", signed=True
    )
