"""Defensive-path coverage for the v0.0.1 legacy evidence boundary."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

import anyalgebra.legacy.algmul_static as static
import anyalgebra.legacy.models as models
from anyalgebra.core.parents import SemanticHash
from anyalgebra.legacy.models import (
    AlgMulDefinitionRecord,
    AlgMulManifest,
    DefinitionKind,
    DefinitionState,
    EvidenceOrigin,
    EvidenceOutcome,
    LegacyManifestError,
    LegacyPartition,
    LoadStatus,
    RegistryRecord,
    SourceMetadata,
    SourceSpan,
    SymbolAttribute,
    SymbolObservation,
    SymbolRecord,
)


def _hash(seed: str) -> SemanticHash:
    import hashlib

    return SemanticHash("sha256", hashlib.sha256(seed.encode()).hexdigest())


def _empty_manifest(*, source: SourceMetadata | None = None) -> AlgMulManifest:
    return AlgMulManifest.create(
        manifest_id="coverage-legacy",
        source=source
        or SourceMetadata.create(
            "C:/AlgMul.wl", _hash("source"), load_status=LoadStatus.STATIC_ONLY
        ),
        symbols=(),
        definitions=(),
        factories=(),
        registries=(),
        behaviors=(),
        messages=(),
        dispositions=(),
    )


def test_small_value_record_repr_and_factory_ownership_guards() -> None:
    span = SourceSpan.create(1, 1)
    observation = SymbolObservation.create(
        EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT
    )
    definition = AlgMulDefinitionRecord.create(
        "symbol:x",
        EvidenceOrigin.STATIC,
        DefinitionState.ABSENT,
        DefinitionKind.NONE,
    )
    assert repr(span) == "SourceSpan()"
    assert repr(observation) == "SymbolObservation()"
    assert repr(definition) == "AlgMulDefinitionRecord()"
    with pytest.raises(LegacyManifestError, match="factory-owned"):
        AlgMulDefinitionRecord()
    with pytest.raises(TypeError, match="cannot be subclassed"):

        class _DefinitionSubclass(AlgMulDefinitionRecord):
            pass


def test_symbol_and_definition_reject_remaining_malformed_shapes() -> None:
    observations = (
        SymbolObservation.create(EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT),
    )
    with pytest.raises(LegacyManifestError, match="must not be empty"):
        models._observations((), "observations")
    for options, message in (
        ({"downvalue_count": True}, "bounded exact ints"),
        (
            {"attributes": (SymbolAttribute.FLAT, SymbolAttribute.FLAT)},
            "duplicate values",
        ),
        ({"option_hashes": (_hash("x"), _hash("x"))}, "duplicate values"),
        ({"source_span": object()}, "exact SourceSpan"),
    ):
        with pytest.raises(LegacyManifestError, match=message):
            SymbolRecord.create("x", LegacyPartition.MATRICES, observations, **options)
    for options, message in (
        ({"downvalue_count": True}, "bounded exact ints"),
        ({"source_span": object()}, "exact SourceSpan"),
        (
            {"attributes": (SymbolAttribute.FLAT, SymbolAttribute.FLAT)},
            "duplicate values",
        ),
    ):
        with pytest.raises(LegacyManifestError, match=message):
            AlgMulDefinitionRecord.create(
                "symbol:x",
                EvidenceOrigin.STATIC,
                DefinitionState.ABSENT,
                DefinitionKind.NONE,
                **options,
            )


def test_manifest_rejects_remaining_cross_record_inconsistencies() -> None:
    source = SourceMetadata.create(
        "C:/AlgMul.wl", _hash("source"), load_status=LoadStatus.STATIC_ONLY
    )
    observation = SymbolObservation.create(
        EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT
    )
    symbol = SymbolRecord.create("x", LegacyPartition.MATRICES, (observation,))
    unknown = AlgMulDefinitionRecord.create(
        "symbol:unknown",
        EvidenceOrigin.STATIC,
        DefinitionState.ABSENT,
        DefinitionKind.NONE,
    )
    with pytest.raises(LegacyManifestError, match="unknown symbol"):
        AlgMulManifest.create(
            manifest_id="unknown-definition",
            source=source,
            symbols=(symbol,),
            definitions=(unknown,),
            factories=(),
            registries=(),
            behaviors=(),
            messages=(),
            dispositions=(),
        )

    anchored_symbol = SymbolRecord.create(
        "anchored",
        LegacyPartition.MATRICES,
        (observation,),
        definition_state=DefinitionState.PRESENT_DEFINED,
        definition_kind=DefinitionKind.UNKNOWN,
        source_span=SourceSpan.create(1, 1),
        definition_hash=_hash("definition"),
        summary_origin=EvidenceOrigin.STATIC,
    )
    with pytest.raises(LegacyManifestError, match="summary origin"):
        AlgMulManifest.create(
            manifest_id="missing-summary",
            source=source,
            symbols=(anchored_symbol,),
            definitions=(),
            factories=(),
            registries=(),
            behaviors=(),
            messages=(),
            dispositions=(),
        )


def test_registry_and_manifest_integrity_defenses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(LegacyManifestError, match="both snapshot and hash"):
        RegistryRecord.create(
            "A", ("1",), "Mul", basis_table_snapshot=_hash("snapshot")
        )

    manifest = _empty_manifest()
    assert repr(manifest) == "AlgMulManifest()"
    forged = object.__new__(AlgMulManifest)
    object.__setattr__(forged, "semantic_hash", _hash("different"))

    def rebuild(cls: type[AlgMulManifest], /, **kwargs: object) -> AlgMulManifest:
        del cls, kwargs
        return forged

    monkeypatch.setattr(AlgMulManifest, "create", classmethod(rebuild))
    with pytest.raises(LegacyManifestError, match="canonical bytes"):
        models._verified_manifest(manifest)


def test_compact_parsers_reject_remaining_invalid_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = models._manifest_record(_empty_manifest())["source"]
    assert type(record) is dict
    bad_source = dict(cast(dict[str, object], record))
    bad_source["kernelVersion"] = 12
    with pytest.raises(LegacyManifestError, match="kernel_version"):
        models._parse_source(bad_source)

    with pytest.raises(LegacyManifestError, match="page size"):
        models._parse_pages(([],), "pages", lambda item: item)
    with pytest.raises(LegacyManifestError, match="page shape"):
        models._parse_name_pages(tuple(() for _ in range(9)), "names")
    with pytest.raises(LegacyManifestError, match="page size"):
        models._parse_name_pages(([],), "names")

    monkeypatch.setattr(models, "_MAX_ITEMS", 1)
    with pytest.raises(LegacyManifestError, match="logical item limit"):
        models._parse_pages((("1", "2"),), "pages", lambda item: item)


def test_static_parser_remaining_lexical_and_arity_paths() -> None:
    assert static._masked_source('"\\\n"') == "  \n "
    assert static._assignment_operator("]x = 1") == 3
    assert static._assignment_name("1 = 2") is None
    assert static._top_level_call_targets("anything", "anonymous") == ()
    assert static._top_level_call_targets("Clear[", "Clear") == ()

    malformed = static._Statement("MakeProperty", "MakeProperty", 1, 1)
    assert not static._makeproperty_factory_definition(
        (("MakeProperty", DefinitionKind.DOWN_VALUE, malformed),)
    )
    assert static._top_level_call_arity("g[x]", "f") is None
    assert static._top_level_call_arity("f[g[x], y]", "f") == 2
    assert static._top_level_call_arity("f[ ]", "f") == 0
    assert static._top_level_call_arity("f[x, y]", "f") == 2
    assert static._top_level_call_arity("f[x", "f") is None


def test_capture_read_failure_and_pinned_unreviewed_name_guards(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "AlgMul.wl"
    source.write_text("x = 1", encoding="utf-8")

    def unreadable(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise OSError("denied")

    monkeypatch.setattr(type(source), "open", unreadable)
    with pytest.raises(static.StaticCaptureError, match="could not be read"):
        static.capture_static(source)

    fake = SimpleNamespace(
        source=SimpleNamespace(
            content_hash=SimpleNamespace(digest=static.PINNED_ALGMUL_SHA256)
        ),
        symbols=(
            SimpleNamespace(
                name="FutureStub",
                partition=LegacyPartition.INCOMPLETE_STUBS_COMMENTS,
            ),
        ),
    )
    monkeypatch.setattr(static, "capture_static", lambda path: fake)
    with pytest.raises(static.StaticCaptureError, match="unreviewed catalogue names"):
        static.capture_pinned_static(source)
