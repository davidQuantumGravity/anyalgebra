"""Persistence-grade contract tests for the non-oracular AlgMul manifest."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

import anyalgebra.legacy.models as models
from anyalgebra.core.parents import SemanticHash
from anyalgebra.legacy.models import (
    EXPECTED_PROPERTY_NAMES,
    AlgMulDefinitionRecord,
    AlgMulManifest,
    AlgMulSurfaceManifest,
    BehaviorRecord,
    DefinitionKind,
    DefinitionState,
    Disposition,
    DispositionRecord,
    EvidenceOrigin,
    EvidenceOutcome,
    FactoryRecord,
    FactoryCallKind,
    LegacyManifestError,
    LegacyPartition,
    LoadStatus,
    MessageRecord,
    OracleStatus,
    ParityDisposition,
    RegistryRecord,
    SourceMetadata,
    SourceSpan,
    SymbolObservation,
    SymbolAttribute,
    SymbolRecord,
    canonical_manifest_bytes,
    legacy_manifest_registry,
    manifest_record,
)
from anyalgebra.persistence.registry import SchemaError


def _hash(text: str) -> SemanticHash:
    return SemanticHash("sha256", hashlib.sha256(text.encode()).hexdigest())


def _states(
    *pairs: tuple[EvidenceOrigin, EvidenceOutcome],
) -> tuple[SymbolObservation, ...]:
    return tuple(SymbolObservation.create(*pair) for pair in pairs)


def _disposition(target_id: str, partition: LegacyPartition) -> DispositionRecord:
    return DispositionRecord.create(
        target_id,
        partition,
        Disposition.REPLACE,
        owner="v0.0",
        note=f"migration rationale for {target_id}",
        api_ids=(f"api.{partition.value}",),
        test_ids=(f"test.{partition.value}",),
        oracle_ids=(f"oracle.{partition.value}",),
    )


def _runtime_source(content_hash: SemanticHash, kernel: str = "12") -> SourceMetadata:
    return SourceMetadata.create(
        "C:/AlgMul.wl",
        content_hash,
        kernel_version=kernel,
        load_status=LoadStatus.LOADED,
        initial_context="Global`",
        context_after_load="Global`",
        context_path_after_load=("System`", "Global`"),
        load_result_head="Symbol",
    )


def _property_symbols() -> tuple[SymbolRecord, ...]:
    states = _states(
        (EvidenceOrigin.EVALUATED, EvidenceOutcome.ABSENT),
        (EvidenceOrigin.INTENDED, EvidenceOutcome.PRESENT),
    )
    return tuple(
        SymbolRecord.create(
            name,
            LegacyPartition.GENERATED_PROPERTY_API,
            states,
            definition_state=DefinitionState.ABSENT,
        )
        for name in EXPECTED_PROPERTY_NAMES
    )


def _summary_definition(record: SymbolRecord) -> AlgMulDefinitionRecord:
    """Materialize the explicit per-origin definition named by a summary."""
    assert record.summary_origin is not None
    return AlgMulDefinitionRecord.create(
        record.record_id,
        record.summary_origin,
        record.definition_state,
        record.definition_kind,
        downvalue_count=record.downvalue_count,
        ownvalue_count=record.ownvalue_count,
        upvalue_count=record.upvalue_count,
        subvalue_count=record.subvalue_count,
        attributes=record.attributes,
        option_hashes=record.option_hashes,
        source_span=record.source_span,
        definition_hash=record.definition_hash,
        record_id=(
            f"summary-definition:{record.record_id}:{record.summary_origin.value}"
        ),
    )


def _manifest(
    *, symbols: tuple[SymbolRecord, ...] | None = None, kernel: str | None = "12.0.0"
) -> AlgMulManifest:
    make_property_states = _states(
        (EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT),
        (EvidenceOrigin.EVALUATED, EvidenceOutcome.ABSENT),
        (EvidenceOrigin.INTENDED, EvidenceOutcome.PRESENT),
    )
    all_symbols = (
        (
            SymbolRecord.create(
                "MakeProperty",
                LegacyPartition.GENERATED_PROPERTY_API,
                make_property_states,
                definition_state=DefinitionState.PRESENT_DEFINED,
                source_span=SourceSpan.create(1750, 1819),
                definition_hash=_hash("MakeProperty static definition"),
                summary_origin=EvidenceOrigin.STATIC,
            ),
            *_property_symbols(),
        )
        if symbols is None
        else symbols
    )
    factories = (
        FactoryRecord.create(
            "MakeProperty",
            make_property_states,
            requested_names=EXPECTED_PROPERTY_NAMES,
        ),
    )
    registries = (RegistryRecord.create("O2", ("1", "e1"), "OMul"),)
    behaviors = (BehaviorRecord.create("MakeChar", "legacy registry mutation"),)
    messages = (MessageRecord.create("load", "warning", "capture message"),)
    definitions = tuple(
        _summary_definition(record)
        if record.summary_origin is not None
        else AlgMulDefinitionRecord.create(
            record.record_id,
            EvidenceOrigin.STATIC,
            DefinitionState.PRESENT_DEFINED,
            DefinitionKind.UNKNOWN,
            source_span=SourceSpan.create(1750, 1819),
            definition_hash=_hash("MakeProperty static definition"),
        )
        if record.name == "MakeProperty"
        else AlgMulDefinitionRecord.create(
            record.record_id,
            EvidenceOrigin.EVALUATED,
            DefinitionState.ABSENT,
            DefinitionKind.NONE,
        )
        for record in all_symbols
        if record.summary_origin is not None
        or record.name == "MakeProperty"
        or record.name in EXPECTED_PROPERTY_NAMES
    )
    dispositions = tuple(
        _disposition(record.record_id, record.partition)
        for records in (all_symbols, factories, registries, behaviors)
        for record in records
    ) + tuple(
        _disposition(
            record.record_id,
            next(
                item.partition
                for item in all_symbols
                if item.record_id == record.symbol_id
            ),
        )
        for record in definitions
    )
    return AlgMulManifest.create(
        manifest_id="algmul.wl-2026-07-22",
        source=SourceMetadata.create(
            "C:/AlgMul.wl",
            _hash("AlgMul source"),
            kernel_version=kernel,
            source_bytes=180_000,
            load_status=LoadStatus.LOADED,
            initial_context="Global`",
            context_after_load="Global`",
            context_path_after_load=("System`", "Global`"),
            load_result_head="Symbol",
        ),
        symbols=all_symbols,
        definitions=definitions,
        factories=factories,
        registries=registries,
        behaviors=behaviors,
        messages=messages,
        dispositions=dispositions,
    )


def test_normative_partitions_and_dispositions_are_complete_and_stable() -> None:
    assert len(LegacyPartition) == 20
    assert {item.value for item in Disposition} == {
        "preserve-exactly",
        "preserve-concept",
        "redesign",
        "replace",
        "drop",
        "defer",
    }
    assert ParityDisposition is Disposition


def test_128_names_have_explicit_intended_and_missing_evaluated_records() -> None:
    manifest = _manifest()
    names = {symbol.name for symbol in manifest.symbols}
    assert len(EXPECTED_PROPERTY_NAMES) == 128
    assert set(EXPECTED_PROPERTY_NAMES) <= names
    for symbol in manifest.symbols:
        if symbol.name in EXPECTED_PROPERTY_NAMES:
            assert symbol.definition_state is DefinitionState.ABSENT
            assert {(item.origin, item.outcome) for item in symbol.observations} == {
                (EvidenceOrigin.EVALUATED, EvidenceOutcome.ABSENT),
                (EvidenceOrigin.INTENDED, EvidenceOutcome.PRESENT),
            }


@pytest.mark.parametrize("mode", ("missing", "duplicate", "wrong-state"))
def test_property_surface_rejects_missing_duplicate_or_wrong_state(mode: str) -> None:
    symbols = list(_manifest().symbols)
    if mode == "missing":
        del symbols[-1]
    elif mode == "duplicate":
        symbols.append(symbols[-1])
    else:
        symbols[-1] = SymbolRecord.create(
            symbols[-1].name,
            LegacyPartition.GENERATED_PROPERTY_API,
            _states((EvidenceOrigin.INTENDED, EvidenceOutcome.PRESENT)),
        )
    with pytest.raises(LegacyManifestError, match=r"symbols|properties|definitions"):
        _manifest(symbols=tuple(symbols)).validate_complete(
            owned_partitions=(LegacyPartition.GENERATED_PROPERTY_API,)
        )


def test_static_makeproperty_requires_a_pinned_definition() -> None:
    manifest = _manifest()
    missing_definition = SymbolRecord.create(
        "MakeProperty",
        LegacyPartition.GENERATED_PROPERTY_API,
        next(
            item.observations
            for item in manifest.symbols
            if item.name == "MakeProperty"
        ),
        definition_state=DefinitionState.PRESENT_UNDEFINED,
        summary_origin=EvidenceOrigin.STATIC,
    )
    with pytest.raises(LegacyManifestError, match="MakeProperty definition"):
        _manifest(
            symbols=(
                missing_definition,
                *(item for item in manifest.symbols if item.name != "MakeProperty"),
            )
        ).validate_complete(owned_partitions=(LegacyPartition.GENERATED_PROPERTY_API,))


def test_final_property_gate_requires_makeproperty_symbol_partition() -> None:
    manifest = _manifest()
    maker = next(item for item in manifest.symbols if item.name == "MakeProperty")
    moved_maker = SymbolRecord.create(
        maker.name,
        LegacyPartition.MATRICES,
        maker.observations,
        definition_state=maker.definition_state,
        source_span=maker.source_span,
        definition_hash=maker.definition_hash,
        summary_origin=EvidenceOrigin.STATIC,
        record_id=maker.record_id,
    )
    with pytest.raises(LegacyManifestError, match="properties"):
        _manifest(
            symbols=(
                moved_maker,
                *(item for item in manifest.symbols if item.name != "MakeProperty"),
            )
        ).validate_complete(owned_partitions=(LegacyPartition.GENERATED_PROPERTY_API,))


def test_final_property_gate_requires_expected_symbols_to_remain_absent() -> None:
    manifest = _manifest()
    property_symbol = next(
        item for item in manifest.symbols if item.name in EXPECTED_PROPERTY_NAMES
    )
    defined_property = SymbolRecord.create(
        property_symbol.name,
        property_symbol.partition,
        (
            *property_symbol.observations,
            SymbolObservation.create(EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT),
        ),
        definition_state=DefinitionState.PRESENT_DEFINED,
        source_span=SourceSpan.create(1, 1),
        definition_hash=_hash("unexpected property definition"),
        summary_origin=EvidenceOrigin.STATIC,
        record_id=property_symbol.record_id,
    )
    with pytest.raises(LegacyManifestError, match="property symbol states"):
        _manifest(
            symbols=(
                defined_property,
                *(item for item in manifest.symbols if item != property_symbol),
            )
        ).validate_complete(owned_partitions=(LegacyPartition.GENERATED_PROPERTY_API,))


@pytest.mark.parametrize(
    ("state", "span", "definition_hash", "works"),
    (
        (DefinitionState.PRESENT_DEFINED, SourceSpan.create(1, 1), _hash("d"), True),
        (DefinitionState.PRESENT_UNDEFINED, None, None, True),
        (DefinitionState.ABSENT, None, None, True),
        (DefinitionState.PRESENT_DEFINED, None, None, False),
        (DefinitionState.ABSENT, SourceSpan.create(1, 1), _hash("d"), False),
    ),
)
def test_definition_state_pins_or_rejects_static_identity(
    state: DefinitionState,
    span: SourceSpan | None,
    definition_hash: SemanticHash | None,
    works: bool,
) -> None:
    if works:
        value = SymbolRecord.create(
            "X",
            LegacyPartition.MATRICES,
            _states((EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT)),
            definition_state=state,
            source_span=span,
            definition_hash=definition_hash,
            summary_origin=EvidenceOrigin.STATIC,
        )
        assert value.definition_state is state
    else:
        with pytest.raises(LegacyManifestError, match="definition"):
            SymbolRecord.create(
                "X",
                LegacyPartition.MATRICES,
                _states((EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT)),
                definition_state=state,
                source_span=span,
                definition_hash=definition_hash,
                summary_origin=EvidenceOrigin.STATIC,
            )


def test_round_trip_canonical_hash_and_order_independence() -> None:
    manifest = _manifest()
    registry = legacy_manifest_registry()
    record = registry.to_record(manifest)
    assert registry.from_record(record) == manifest
    manifest.validate_complete(
        owned_partitions=(
            LegacyPartition.GENERATED_PROPERTY_API,
            LegacyPartition.REGISTRY_BASES,
        )
    )
    assert manifest_record(manifest) == record
    assert manifest.semantic_hash == SemanticHash(
        "sha256", hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()
    )
    assert (
        _manifest(symbols=tuple(reversed(manifest.symbols))).canonical_bytes()
        == manifest.canonical_bytes()
    )


def test_dispositions_assign_each_record_once_with_matching_target_and_partition() -> (
    None
):
    manifest = _manifest()
    assert {item.target_id for item in manifest.dispositions} == (
        {item.record_id for item in manifest.symbols}
        | {item.record_id for item in manifest.definitions}
        | {item.record_id for item in manifest.factories}
        | {item.record_id for item in manifest.registries}
        | {item.record_id for item in manifest.behaviors}
    )
    common = dict(
        manifest_id=manifest.manifest_id,
        source=manifest.source,
        symbols=manifest.symbols,
        definitions=manifest.definitions,
        factories=manifest.factories,
        registries=manifest.registries,
        behaviors=manifest.behaviors,
        messages=manifest.messages,
    )
    incomplete = AlgMulManifest.create(
        **common, dispositions=manifest.dispositions[:-1]
    )
    with pytest.raises(LegacyManifestError, match="incomplete"):
        incomplete.validate_complete(
            owned_partitions=(
                LegacyPartition.GENERATED_PROPERTY_API,
                LegacyPartition.REGISTRY_BASES,
            )
        )
    for dispositions, text in (
        ((*manifest.dispositions, manifest.dispositions[0]), "duplicate"),
        (
            (
                DispositionRecord.create(
                    "unknown.target",
                    manifest.dispositions[0].partition,
                    manifest.dispositions[0].disposition,
                    owner="v0.0",
                    note="unknown target regression",
                    api_ids=("api",),
                    test_ids=("test",),
                    oracle_ids=("oracle",),
                ),
                *manifest.dispositions[1:],
            ),
            "unknown target",
        ),
        (
            (
                DispositionRecord.create(
                    manifest.dispositions[0].target_id,
                    LegacyPartition.MATRICES,
                    manifest.dispositions[0].disposition,
                    owner="v0.0",
                    note="wrong partition regression",
                    api_ids=("api",),
                    test_ids=("test",),
                    oracle_ids=("oracle",),
                ),
                *manifest.dispositions[1:],
            ),
            "partition",
        ),
    ):
        with pytest.raises(LegacyManifestError, match=text):
            AlgMulManifest.create(**common, dispositions=dispositions)


def test_assign_disposition_replaces_one_record_target_without_mutation() -> None:
    manifest = _manifest()
    target_id = manifest.factories[0].record_id
    updated = manifest.assign_disposition(
        target_id,
        Disposition.PRESERVE_CONCEPT,
        note="preserve the factory migration contract",
        api_ids=("api.updated",),
        test_ids=("test.updated",),
        oracle_ids=("oracle.updated",),
    )
    assert updated is not manifest
    assert (
        next(item for item in updated.dispositions if item.target_id == target_id).note
        == "preserve the factory migration contract"
    )
    with pytest.raises(LegacyManifestError, match="does not name"):
        manifest.assign_disposition(
            "unknown.target",
            Disposition.DROP,
            note="unknown target regression",
            api_ids=("api",),
            test_ids=("test",),
            oracle_ids=("oracle",),
        )


def test_paged_production_scale_surface_round_trips_and_hashes() -> None:
    static = _states((EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT))
    definitions = tuple(
        SymbolRecord.create(
            f"Definition{index}",
            LegacyPartition.MATRICES,
            static,
            definition_state=DefinitionState.PRESENT_DEFINED,
            source_span=SourceSpan.create(index + 1, index + 1),
            definition_hash=_hash(f"definition-{index}"),
            summary_origin=EvidenceOrigin.STATIC,
            record_id=f"definition:{index}",
        )
        for index in range(480)
    )
    manifest = _manifest(symbols=(*_manifest().symbols, *definitions))
    registry = legacy_manifest_registry()
    record = registry.to_record(manifest)
    assert isinstance(record["symbolPages"], list)
    assert isinstance(record["dispositionPages"], list)
    assert len(record["symbolPages"]) >= 3
    assert len(record["dispositionPages"]) >= 3
    assert registry.from_record(record) == manifest
    assert canonical_manifest_bytes(manifest) == manifest.canonical_bytes()


def test_maximum_symbol_text_round_trips_and_compact_boundary_is_enforced() -> None:
    symbol = SymbolRecord.create(
        "x" * models._MAX_TEXT,
        LegacyPartition.MATRICES,
        _states((EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT)),
        definition_state=DefinitionState.PRESENT_DEFINED,
        source_span=SourceSpan.create(1, 1),
        definition_hash=_hash("long symbol"),
        summary_origin=EvidenceOrigin.STATIC,
        record_id="symbol:long",
    )
    manifest = AlgMulManifest.create(
        manifest_id="long-symbol",
        source=SourceMetadata.create("C:/AlgMul.wl", _hash("source"), source_bytes=1),
        symbols=(symbol,),
        definitions=(_summary_definition(symbol),),
        factories=(),
        registries=(),
        behaviors=(),
        messages=(),
        dispositions=(
            _disposition(symbol.record_id, symbol.partition),
            _disposition(
                _summary_definition(symbol).record_id,
                symbol.partition,
            ),
        ),
    )
    registry = legacy_manifest_registry()
    record = registry.to_record(manifest)
    assert registry.from_record(record) == manifest
    boundary_payload = "x" * (models._MAX_COMPACT_RECORD - len('{"x":""}'))
    assert len(models._compact({"x": boundary_payload})) == models._MAX_COMPACT_RECORD
    with pytest.raises(LegacyManifestError, match="compact record limit"):
        models._compact({"x": boundary_payload + "x"})
    with pytest.raises(LegacyManifestError, match="compact record"):
        models._parse_pages(
            (("x" * (models._MAX_COMPACT_RECORD + 1),),), "pages", lambda item: item
        )


@pytest.mark.parametrize(
    ("state", "observations"),
    (
        (
            DefinitionState.PRESENT_DEFINED,
            _states((EvidenceOrigin.STATIC, EvidenceOutcome.ABSENT)),
        ),
        (
            DefinitionState.PRESENT_UNDEFINED,
            _states((EvidenceOrigin.STATIC, EvidenceOutcome.ABSENT)),
        ),
        (
            DefinitionState.ABSENT,
            _states((EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT)),
        ),
    ),
)
def test_definition_records_cannot_contradict_static_observation(
    state: DefinitionState, observations: tuple[SymbolObservation, ...]
) -> None:
    span = SourceSpan.create(1, 1) if state is DefinitionState.PRESENT_DEFINED else None
    definition_hash = _hash("definition") if span is not None else None
    symbol = SymbolRecord.create(
        "Contradiction",
        LegacyPartition.MATRICES,
        observations,
        definition_state=state,
        source_span=span,
        definition_hash=definition_hash,
        summary_origin=(
            EvidenceOrigin.STATIC if state is not DefinitionState.ABSENT else None
        ),
    )
    definition = AlgMulDefinitionRecord.create(
        symbol.record_id,
        EvidenceOrigin.STATIC,
        state,
        (
            DefinitionKind.NONE
            if state in (DefinitionState.ABSENT, DefinitionState.PRESENT_UNDEFINED)
            else DefinitionKind.UNKNOWN
        ),
        source_span=span,
        definition_hash=definition_hash,
    )
    with pytest.raises(LegacyManifestError, match="contradicts"):
        AlgMulManifest.create(
            manifest_id="contradiction",
            source=SourceMetadata.create("C:/AlgMul.wl", _hash("source")),
            symbols=(symbol,),
            definitions=(definition,),
            factories=(),
            registries=(),
            behaviors=(),
            messages=(),
            dispositions=(
                _disposition(symbol.record_id, symbol.partition),
                _disposition(definition.record_id, symbol.partition),
            ),
        )


def test_symbol_summary_origin_requires_non_neutral_detail_and_exact_definition() -> (
    None
):
    static = _states((EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT))
    with pytest.raises(LegacyManifestError, match="summary_origin"):
        SymbolRecord.create(
            "MissingOrigin",
            LegacyPartition.MATRICES,
            static,
            definition_state=DefinitionState.PRESENT_DEFINED,
            source_span=SourceSpan.create(1, 1),
            definition_hash=_hash("missing origin"),
        )
    symbol = SymbolRecord.create(
        "PinnedSummary",
        LegacyPartition.MATRICES,
        static,
        definition_state=DefinitionState.PRESENT_DEFINED,
        source_span=SourceSpan.create(1, 1),
        definition_hash=_hash("summary"),
        summary_origin=EvidenceOrigin.STATIC,
    )
    mismatch = AlgMulDefinitionRecord.create(
        symbol.record_id,
        EvidenceOrigin.STATIC,
        DefinitionState.PRESENT_DEFINED,
        DefinitionKind.DOWN_VALUE,
        downvalue_count=1,
        source_span=SourceSpan.create(1, 1),
        definition_hash=_hash("summary"),
    )
    with pytest.raises(LegacyManifestError, match="summary does not match"):
        AlgMulManifest.create(
            manifest_id="summary-mismatch",
            source=SourceMetadata.create("C:/AlgMul.wl", _hash("source")),
            symbols=(symbol,),
            definitions=(mismatch,),
            factories=(),
            registries=(),
            behaviors=(),
            messages=(),
            dispositions=(
                _disposition(symbol.record_id, symbol.partition),
                _disposition(mismatch.record_id, symbol.partition),
            ),
        )


@pytest.mark.parametrize(
    ("state", "kind", "counts", "attributes", "span", "definition_hash"),
    (
        (
            DefinitionState.ABSENT,
            DefinitionKind.NONE,
            (0, 0, 0, 0),
            (SymbolAttribute.PROTECTED,),
            None,
            None,
        ),
        (
            DefinitionState.PRESENT_DEFINED,
            DefinitionKind.NONE,
            (0, 0, 0, 0),
            (),
            SourceSpan.create(1, 1),
            _hash("none"),
        ),
        (
            DefinitionState.PRESENT_DEFINED,
            DefinitionKind.DOWN_VALUE,
            (0, 0, 0, 0),
            (),
            SourceSpan.create(1, 1),
            _hash("down"),
        ),
        (
            DefinitionState.PRESENT_DEFINED,
            DefinitionKind.MIXED,
            (1, 0, 0, 0),
            (),
            SourceSpan.create(1, 1),
            _hash("mixed"),
        ),
    ),
)
def test_definition_record_rejects_incoherent_shape(
    state: DefinitionState,
    kind: DefinitionKind,
    counts: tuple[int, int, int, int],
    attributes: tuple[SymbolAttribute, ...],
    span: SourceSpan | None,
    definition_hash: SemanticHash | None,
) -> None:
    with pytest.raises(LegacyManifestError, match="definition"):
        AlgMulDefinitionRecord.create(
            "symbol:shape",
            EvidenceOrigin.STATIC,
            state,
            kind,
            downvalue_count=counts[0],
            ownvalue_count=counts[1],
            upvalue_count=counts[2],
            subvalue_count=counts[3],
            attributes=attributes,
            source_span=span,
            definition_hash=definition_hash,
        )


def test_definition_record_enforces_static_anchors_and_option_hash_integrity() -> None:
    for span, definition_hash in ((SourceSpan.create(1, 1), None), (None, _hash("x"))):
        with pytest.raises(LegacyManifestError, match="source"):
            AlgMulDefinitionRecord.create(
                "symbol:anchors",
                EvidenceOrigin.STATIC,
                DefinitionState.PRESENT_DEFINED,
                DefinitionKind.UNKNOWN,
                source_span=span,
                definition_hash=definition_hash,
            )
    with pytest.raises(LegacyManifestError, match="only static"):
        AlgMulDefinitionRecord.create(
            "symbol:anchors",
            EvidenceOrigin.EVALUATED,
            DefinitionState.PRESENT_DEFINED,
            DefinitionKind.UNKNOWN,
            source_span=SourceSpan.create(1, 1),
            definition_hash=_hash("x"),
        )
    option_hash = _hash("option")
    with pytest.raises(LegacyManifestError, match="option_hashes"):
        AlgMulDefinitionRecord.create(
            "symbol:options",
            EvidenceOrigin.EVALUATED,
            DefinitionState.PRESENT_DEFINED,
            DefinitionKind.UNKNOWN,
            option_hashes=(option_hash, option_hash),
        )
    symbol = SymbolRecord.create(
        "OptionCarrier",
        LegacyPartition.MATRICES,
        _states((EvidenceOrigin.EVALUATED, EvidenceOutcome.PRESENT)),
    )
    definition = AlgMulDefinitionRecord.create(
        symbol.record_id,
        EvidenceOrigin.EVALUATED,
        DefinitionState.PRESENT_DEFINED,
        DefinitionKind.UNKNOWN,
        option_hashes=(option_hash,),
    )
    manifest = AlgMulManifest.create(
        manifest_id="options",
        source=_runtime_source(_hash("source")),
        symbols=(symbol,),
        definitions=(definition,),
        factories=(),
        registries=(),
        behaviors=(),
        messages=(),
        dispositions=(
            _disposition(symbol.record_id, symbol.partition),
            _disposition(definition.record_id, symbol.partition),
        ),
    )
    assert (
        legacy_manifest_registry().from_record(
            legacy_manifest_registry().to_record(manifest)
        )
        == manifest
    )
    object.__setattr__(
        manifest.definitions[0], "option_hashes", (option_hash, option_hash)
    )
    with pytest.raises((LegacyManifestError, SchemaError), match="semantic_hash"):
        manifest_record(manifest)


def test_undefined_and_evaluated_summary_shapes_are_origin_aware() -> None:
    static = _states((EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT))
    undefined_symbol = SymbolRecord.create(
        "Undefined",
        LegacyPartition.MATRICES,
        static,
        definition_state=DefinitionState.PRESENT_UNDEFINED,
        summary_origin=EvidenceOrigin.STATIC,
    )
    undefined_definition = AlgMulDefinitionRecord.create(
        undefined_symbol.record_id,
        EvidenceOrigin.STATIC,
        DefinitionState.PRESENT_UNDEFINED,
        DefinitionKind.NONE,
    )
    manifest = AlgMulManifest.create(
        manifest_id="undefined-summary",
        source=SourceMetadata.create("C:/AlgMul.wl", _hash("source")),
        symbols=(undefined_symbol,),
        definitions=(undefined_definition,),
        factories=(),
        registries=(),
        behaviors=(),
        messages=(),
        dispositions=(
            _disposition(undefined_symbol.record_id, undefined_symbol.partition),
            _disposition(undefined_definition.record_id, undefined_symbol.partition),
        ),
    )
    assert manifest.definitions == (undefined_definition,)
    for create in (
        lambda: SymbolRecord.create(
            "InvalidUndefined",
            LegacyPartition.MATRICES,
            static,
            definition_state=DefinitionState.PRESENT_UNDEFINED,
            definition_kind=DefinitionKind.DOWN_VALUE,
            downvalue_count=1,
            summary_origin=EvidenceOrigin.STATIC,
        ),
        lambda: AlgMulDefinitionRecord.create(
            "symbol:invalid-undefined",
            EvidenceOrigin.STATIC,
            DefinitionState.PRESENT_UNDEFINED,
            DefinitionKind.DOWN_VALUE,
            downvalue_count=1,
        ),
    ):
        with pytest.raises(LegacyManifestError, match="undefined summaries"):
            create()
    evaluated_symbol = SymbolRecord.create(
        "EvaluatedDefinition",
        LegacyPartition.MATRICES,
        _states((EvidenceOrigin.EVALUATED, EvidenceOutcome.PRESENT)),
        definition_state=DefinitionState.PRESENT_DEFINED,
        definition_kind=DefinitionKind.DOWN_VALUE,
        downvalue_count=1,
        summary_origin=EvidenceOrigin.EVALUATED,
    )
    evaluated_definition = AlgMulDefinitionRecord.create(
        evaluated_symbol.record_id,
        EvidenceOrigin.EVALUATED,
        DefinitionState.PRESENT_DEFINED,
        DefinitionKind.DOWN_VALUE,
        downvalue_count=1,
    )
    evaluated_manifest = AlgMulManifest.create(
        manifest_id="evaluated-summary",
        source=_runtime_source(_hash("evaluated source")),
        symbols=(evaluated_symbol,),
        definitions=(evaluated_definition,),
        factories=(),
        registries=(),
        behaviors=(),
        messages=(),
        dispositions=(
            _disposition(evaluated_symbol.record_id, evaluated_symbol.partition),
            _disposition(evaluated_definition.record_id, evaluated_symbol.partition),
        ),
    )
    assert evaluated_manifest.definitions == (evaluated_definition,)
    with pytest.raises(LegacyManifestError, match="non-static summaries"):
        SymbolRecord.create(
            "AnchoredEvaluated",
            LegacyPartition.MATRICES,
            _states((EvidenceOrigin.EVALUATED, EvidenceOutcome.PRESENT)),
            definition_state=DefinitionState.PRESENT_DEFINED,
            source_span=SourceSpan.create(1, 1),
            definition_hash=_hash("forbidden evaluated anchor"),
            summary_origin=EvidenceOrigin.EVALUATED,
        )


def test_unknown_tags_versions_keys_and_unsafe_values_fail_without_execution() -> None:
    registry = legacy_manifest_registry()
    record = registry.to_record(_manifest())
    with pytest.raises(SchemaError, match="unknown_schema_type"):
        registry.from_record({**record, "schemaType": "anyalgebra.legacy.unknown"})
    with pytest.raises(SchemaError, match="unknown_schema_version"):
        registry.from_record({**record, "schemaVersion": 2})
    bodies: tuple[dict[str, object], ...] = (
        {**record, "extra": 1},
        {**record, "symbolPages": [['{"callable":"os.system"}']]},
        {
            **record,
            "source": {
                "location": "x",
                "contentHash": {},
                "kernelVersion": None,
                "sourceBytes": float("nan"),
            },
        },
    )
    for body in bodies:
        with pytest.raises(SchemaError):
            registry.from_record(body)


def test_registry_wire_basis_requires_a_sequence_and_origin_qualifies_names() -> None:
    registry = legacy_manifest_registry()
    wire = registry.to_record(_manifest())
    malformed = json.loads(json.dumps(wire))
    malformed["registries"][0]["basis"] = "{1, e1}"
    with pytest.raises((LegacyManifestError, SchemaError), match="parser_failed"):
        registry.from_record(malformed)
    static = RegistryRecord.create(
        "O", ("1",), "OMul", origin=EvidenceOrigin.STATIC, record_id="static:O"
    )
    evaluated = RegistryRecord.create(
        "O", ("1",), "OMul", origin=EvidenceOrigin.EVALUATED, record_id="eval:O"
    )

    def manifest_for(registries: tuple[RegistryRecord, ...]) -> AlgMulManifest:
        return AlgMulManifest.create(
            manifest_id="registry-origins",
            source=_runtime_source(_hash("registry source")),
            symbols=(),
            factories=(),
            registries=registries,
            behaviors=(),
            messages=(),
            dispositions=tuple(
                _disposition(item.record_id, item.partition) for item in registries
            ),
        )

    assert manifest_for((static, evaluated)).registries == (evaluated, static)
    conflicting = RegistryRecord.create(
        "O", ("1",), "OtherMul", origin=EvidenceOrigin.EVALUATED, record_id="eval:O2"
    )
    with pytest.raises(LegacyManifestError, match=r"registries.*duplicate"):
        manifest_for((evaluated, conflicting))


def test_disposition_links_and_deferred_owner_are_enforced() -> None:
    with pytest.raises(LegacyManifestError, match=r"v0\.0"):
        DispositionRecord.create(
            "record.matrices",
            LegacyPartition.MATRICES,
            Disposition.REPLACE,
            owner="v0.0",
            note="v0.0 links must be complete",
            oracle_ids=("oracle",),
        )
    with pytest.raises(LegacyManifestError, match="later owner"):
        DispositionRecord.create(
            "record.jordan",
            LegacyPartition.JORDAN_MATRICES,
            Disposition.DEFER,
            owner="v0.0",
            note="deferred work needs another owner",
            oracle_ids=(),
        )
    with pytest.raises(LegacyManifestError, match="note"):
        DispositionRecord.create(
            "record.empty-note",
            LegacyPartition.MATRICES,
            Disposition.REPLACE,
            owner="v0.1",
            note="",
            oracle_ids=(),
        )
    deferred = DispositionRecord.create(
        "record.jordan",
        LegacyPartition.JORDAN_MATRICES,
        Disposition.DEFER,
        owner="v0.1",
        note="deferred migration rationale",
        oracle_ids=(),
    )
    assert deferred.owner == "v0.1"
    pending = DispositionRecord.create(
        "record.pending",
        LegacyPartition.MATRICES,
        Disposition.REDESIGN,
        owner="v0.0",
        note=(
            "Independent entrywise nonassociative matrix oracle remains unimplemented."
        ),
        api_ids=("api.matrix.construct",),
        test_ids=("test.legacy.disposition_complete",),
        oracle_ids=(),
        oracle_status=OracleStatus.PENDING,
    )
    assert pending.oracle_ids == ()
    assert pending.oracle_status is OracleStatus.PENDING
    with pytest.raises(LegacyManifestError, match="precise gap"):
        DispositionRecord.create(
            "record.vague-pending",
            LegacyPartition.MATRICES,
            Disposition.REDESIGN,
            owner="v0.0",
            note="oracle pending",
            api_ids=("api.matrix.construct",),
            test_ids=("test.legacy.disposition_complete",),
            oracle_ids=(),
            oracle_status=OracleStatus.PENDING,
        )
    with pytest.raises(LegacyManifestError, match="available oracle"):
        DispositionRecord.create(
            "record.unbacked-available",
            LegacyPartition.MATRICES,
            Disposition.REDESIGN,
            owner="v0.0",
            note="claims an available oracle without a durable evidence link",
            api_ids=("api.matrix.construct",),
            test_ids=("test.legacy.disposition_complete",),
            oracle_ids=(),
            oracle_status=OracleStatus.AVAILABLE,
        )
    with pytest.raises(LegacyManifestError, match="pending oracle"):
        DispositionRecord.create(
            "record.pending-with-link",
            LegacyPartition.MATRICES,
            Disposition.REDESIGN,
            owner="v0.0",
            note="pending evidence cannot carry a fabricated evidence link",
            api_ids=("api.matrix.construct",),
            test_ids=("test.legacy.disposition_complete",),
            oracle_ids=("oracle.fabricated",),
            oracle_status=OracleStatus.PENDING,
        )


def test_final_gate_requires_api_and_test_links_for_deferred_records() -> None:
    symbol = SymbolRecord.create(
        "DeferredSymbol",
        LegacyPartition.JORDAN_MATRICES,
        _states((EvidenceOrigin.STATIC, EvidenceOutcome.ABSENT)),
    )
    deferred = DispositionRecord.create(
        symbol.record_id,
        symbol.partition,
        Disposition.DEFER,
        owner="v0.1",
        note="defer pending a later migration owner",
        oracle_ids=(),
    )
    manifest = AlgMulManifest.create(
        manifest_id="deferred-links",
        source=SourceMetadata.create("C:/AlgMul.wl", _hash("source"), source_bytes=1),
        symbols=(symbol,),
        factories=(),
        registries=(),
        behaviors=(),
        messages=(),
        dispositions=(deferred,),
    )
    with pytest.raises(LegacyManifestError, match="API and test"):
        manifest.validate_complete(owned_partitions=(LegacyPartition.JORDAN_MATRICES,))


def test_evaluated_records_need_kernel_and_messages_are_unique() -> None:
    with pytest.raises(LegacyManifestError, match="runtime source"):
        _manifest(kernel=None)
    manifest = _manifest()
    with pytest.raises(LegacyManifestError, match="messages"):
        AlgMulManifest.create(
            manifest_id="x",
            source=manifest.source,
            symbols=manifest.symbols,
            factories=manifest.factories,
            registries=manifest.registries,
            behaviors=manifest.behaviors,
            messages=(manifest.messages[0], manifest.messages[0]),
            dispositions=manifest.dispositions,
        )


@pytest.mark.parametrize("line", (0, -1, 1_000_001))
def test_source_ranges_and_bounded_source_counts_reject_invalid_values(
    line: int,
) -> None:
    with pytest.raises(LegacyManifestError, match="source_span"):
        SourceSpan.create(line, 1)
    with pytest.raises(LegacyManifestError, match="source_bytes"):
        SourceMetadata.create("x", _hash("x"), source_bytes=-1)


def test_factory_ownership_sealing_and_sanitized_repr() -> None:
    values = (
        _manifest().source,
        _manifest().symbols[0],
        _manifest().factories[0],
        _manifest().registries[0],
        _manifest().behaviors[0],
        _manifest().messages[0],
        _manifest().dispositions[0],
    )
    assert all(
        "AlgMul.wl" not in repr(value) and "capture message" not in repr(value)
        for value in values
    )
    with pytest.raises(LegacyManifestError, match="factory-owned"):
        AlgMulManifest()
    with pytest.raises(TypeError, match="cannot be subclassed"):

        class Bad(SourceMetadata):
            pass


@pytest.mark.parametrize(
    "factory",
    (
        lambda: SourceMetadata(),
        lambda: SourceSpan(),
        lambda: SymbolObservation(),
        lambda: SymbolRecord(),
        lambda: FactoryRecord(),
        lambda: RegistryRecord(),
        lambda: BehaviorRecord(),
        lambda: MessageRecord(),
        lambda: DispositionRecord(),
    ),
)
def test_all_component_records_reject_direct_construction(
    factory: Callable[[], object],
) -> None:
    with pytest.raises(LegacyManifestError, match="factory-owned"):
        factory()


@pytest.mark.parametrize(
    "record_type",
    (
        SourceSpan,
        SymbolObservation,
        SymbolRecord,
        FactoryRecord,
        RegistryRecord,
        BehaviorRecord,
        MessageRecord,
        DispositionRecord,
        AlgMulManifest,
    ),
)
def test_all_records_are_non_subclassable(record_type: type[object]) -> None:
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("IllicitSubclass", (record_type,), {})


@pytest.mark.parametrize(
    ("value", "identifier"),
    ((None, False), ("", False), ("x" * 4_097, False), ("\ud800", False), (" x", True)),
)
def test_text_and_json_boundary_controls_are_explicit(
    value: object, identifier: bool
) -> None:
    with pytest.raises(LegacyManifestError):
        models._text(value, "field", identifier=identifier)
    for value in (None, [], {"x": 1}):
        with pytest.raises(LegacyManifestError):
            models._hash(value, "hash")
    bad_hash = _hash("safe")
    object.__setattr__(bad_hash, "digest", "broken")
    with pytest.raises(LegacyManifestError):
        models._hash(bad_hash, "hash")


def test_closed_enum_tuple_name_record_and_mapping_controls() -> None:
    for enum_value in (None, "unknown"):
        with pytest.raises(LegacyManifestError):
            models._enum(enum_value, EvidenceOrigin, "origin")
    container_values: tuple[object, ...] = ([], tuple(range(2_049)))
    for container_value in container_values:
        with pytest.raises(LegacyManifestError):
            models._tuple(container_value, "items")
    with pytest.raises(LegacyManifestError, match="duplicate"):
        models._names(("x", "x"), "names")
    with pytest.raises(LegacyManifestError, match="128"):
        models._names(("x",), "names", expected=True)
    with pytest.raises(LegacyManifestError):
        models._records(("x",), SymbolRecord, "records")
    with pytest.raises(LegacyManifestError):
        models._mapping({"wrong": 1}, frozenset(("right",)), "record")
    with pytest.raises(LegacyManifestError):
        models._parse_hash({"algorithm": "sha256", "digest": "broken"}, "hash")


def test_record_level_negative_controls_and_duplicate_observations() -> None:
    observation = SymbolObservation.create(
        EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT
    )
    with pytest.raises(LegacyManifestError, match="duplicate observations"):
        SymbolRecord.create("X", LegacyPartition.MATRICES, (observation, observation))
    with pytest.raises(LegacyManifestError, match="requested_names"):
        FactoryRecord.create(
            "MakeProperty",
            _manifest().factories[0].observations,
            requested_names=("wrong",),
        )
    with pytest.raises(LegacyManifestError, match="basis"):
        RegistryRecord.create("R", (), "mul")
    for value in (" ", "x" * 4_097):
        with pytest.raises(LegacyManifestError):
            MessageRecord.create(value, "warning", "message")


def test_parser_shape_and_size_boundaries_are_rejected() -> None:
    invalid_calls = (
        lambda: models._parse_observations([], "observations"),
        lambda: models._parse_items([], "items", lambda item: item),
        lambda: models._parse_source({}),
        lambda: models._parse_symbol({}),
        lambda: models._parse_factory({}),
        lambda: models._parse_registry({}),
        lambda: models._parse_behavior({}),
        lambda: models._parse_message({}),
        lambda: models._parse_disposition({}),
    )
    for invalid_call in invalid_calls:
        with pytest.raises(LegacyManifestError):
            invalid_call()
    with pytest.raises(LegacyManifestError, match="item limit"):
        models._parse_items(tuple(range(2_049)), "items", lambda item: item)


def test_paged_wire_rejects_invalid_shapes_values_and_logical_overflow() -> None:
    def parser(item: object) -> object:
        return item

    invalid_pages: tuple[object, ...] = (
        [],
        tuple(() for _ in range(9)),
        ((1,),),
        (("not json",),),
        (('{"recordId":"a","recordId":"b"}',),),
    )
    for pages in invalid_pages:
        with pytest.raises(LegacyManifestError):
            models._parse_pages(pages, "pages", parser)
    with pytest.raises(LegacyManifestError, match="unsafe"):
        models._freeze_wire(1.0)


def test_static_capture_can_be_incomplete_then_assigned_and_validated() -> None:
    symbol = SymbolRecord.create(
        "StaticOnly",
        LegacyPartition.MATRICES,
        _states((EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT)),
        definition_state=DefinitionState.PRESENT_DEFINED,
        source_span=SourceSpan.create(1, 1),
        definition_hash=_hash("static"),
        summary_origin=EvidenceOrigin.STATIC,
    )
    capture = AlgMulManifest.create(
        manifest_id="static-capture",
        source=SourceMetadata.create("C:/AlgMul.wl", _hash("source"), source_bytes=1),
        symbols=(symbol,),
        definitions=(_summary_definition(symbol),),
        factories=(),
        registries=(),
        behaviors=(),
        messages=(),
        dispositions=(),
    )
    assert (
        legacy_manifest_registry().from_record(
            legacy_manifest_registry().to_record(capture)
        )
        == capture
    )
    with pytest.raises(LegacyManifestError, match="incomplete"):
        capture.validate_complete(owned_partitions=(LegacyPartition.MATRICES,))
    complete = capture.assign_disposition(
        symbol.record_id,
        Disposition.PRESERVE_CONCEPT,
        note="static capture migration rationale",
        api_ids=("api.static",),
        test_ids=("test.static",),
        oracle_ids=("oracle.static",),
    )
    complete = complete.assign_disposition(
        complete.definitions[0].record_id,
        Disposition.PRESERVE_CONCEPT,
        note="static definition migration rationale",
        api_ids=("api.static-definition",),
        test_ids=("test.static-definition",),
        oracle_ids=("oracle.static-definition",),
    )
    complete.validate_complete(owned_partitions=(LegacyPartition.MATRICES,))


def test_same_name_distinct_definition_ids_have_canonical_order() -> None:
    records = tuple(
        SymbolRecord.create(
            "OsMul",
            LegacyPartition.SCALAR_COMPOSITION_PRODUCTS,
            _states((EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT)),
            definition_state=DefinitionState.PRESENT_DEFINED,
            source_span=SourceSpan.create(index, index),
            definition_hash=_hash(f"OsMul-{index}"),
            summary_origin=EvidenceOrigin.STATIC,
            record_id=f"definition:osmul:{index}",
        )
        for index in (2, 1)
    )
    source = SourceMetadata.create("C:/AlgMul.wl", _hash("source"), source_bytes=1)
    dispositions = tuple(
        _disposition(record.record_id, record.partition) for record in records
    ) + tuple(
        _disposition(_summary_definition(record).record_id, record.partition)
        for record in records
    )
    with pytest.raises(LegacyManifestError, match=r"symbols.*duplicate"):
        AlgMulManifest.create(
            manifest_id="same-name",
            source=source,
            symbols=records,
            definitions=tuple(_summary_definition(record) for record in records),
            factories=(),
            registries=(),
            behaviors=(),
            messages=(),
            dispositions=dispositions,
        )


def test_manifest_wrong_types_and_global_duplicate_ids_reject() -> None:
    manifest = _manifest()
    common = dict(
        manifest_id=manifest.manifest_id,
        source=manifest.source,
        symbols=manifest.symbols,
        definitions=manifest.definitions,
        factories=manifest.factories,
        registries=manifest.registries,
        behaviors=manifest.behaviors,
        messages=manifest.messages,
        dispositions=manifest.dispositions,
    )
    with pytest.raises(LegacyManifestError, match="source"):
        AlgMulManifest.create(**{**common, "source": object()})
    with pytest.raises(LegacyManifestError, match="dispositions"):
        AlgMulManifest.create(**{**common, "factories": ()})
    duplicate_registry = RegistryRecord.create(
        "Duplicate",
        ("1",),
        "mul",
        record_id=manifest.symbols[0].record_id,
    )
    with pytest.raises(LegacyManifestError, match="record IDs"):
        AlgMulManifest.create(
            **{**common, "registries": (*manifest.registries, duplicate_registry)}
        )
    for value in (object(),):
        with pytest.raises(LegacyManifestError):
            manifest_record(value)
        with pytest.raises(LegacyManifestError):
            canonical_manifest_bytes(value)


@pytest.mark.parametrize(
    "target",
    (
        "source",
        "symbol",
        "factory",
        "registry",
        "behavior",
        "message",
        "disposition",
        "note",
        "observation",
        "hash",
    ),
)
def test_tampered_live_objects_are_rejected_by_public_record_and_bytes(
    target: str,
) -> None:
    manifest = _manifest()
    if target == "source":
        object.__setattr__(manifest.source, "location", "tampered")
    elif target == "symbol":
        object.__setattr__(manifest.symbols[0], "name", "tampered")
    elif target == "factory":
        object.__setattr__(manifest.factories[0], "requested_names", ())
    elif target == "registry":
        object.__setattr__(manifest.registries[0], "basis", ())
    elif target == "behavior":
        object.__setattr__(manifest.behaviors[0], "description", "tampered")
    elif target == "message":
        object.__setattr__(manifest.messages[0], "text", "tampered")
    elif target == "disposition":
        object.__setattr__(manifest.dispositions[0], "api_ids", ())
    elif target == "note":
        object.__setattr__(manifest.dispositions[0], "note", "")
    elif target == "observation":
        object.__setattr__(manifest.symbols[0].observations[0], "origin", "bad")
    else:
        object.__setattr__(manifest.semantic_hash, "digest", "0" * 64)
    with pytest.raises((LegacyManifestError, SchemaError)):
        manifest_record(manifest)
    with pytest.raises((LegacyManifestError, SchemaError)):
        canonical_manifest_bytes(manifest)


def test_static_only_makeproperty_factory_manifest_round_trips_without_kernel() -> None:
    static = _states((EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT))
    symbol = SymbolRecord.create(
        "MakeProperty",
        LegacyPartition.GENERATED_PROPERTY_API,
        static,
        definition_state=DefinitionState.PRESENT_DEFINED,
        source_span=SourceSpan.create(1, 1),
        definition_hash=_hash("static maker"),
        summary_origin=EvidenceOrigin.STATIC,
    )
    factory = FactoryRecord.create(
        "MakeProperty", static, requested_names=EXPECTED_PROPERTY_NAMES
    )
    manifest = AlgMulManifest.create(
        manifest_id="static-maker",
        source=SourceMetadata.create(
            "C:/AlgMul.wl",
            _hash("static source"),
            contexts=("AlgMul`",),
            load_status=LoadStatus.STATIC_ONLY,
        ),
        symbols=(symbol,),
        definitions=(_summary_definition(symbol),),
        factories=(factory,),
        registries=(),
        behaviors=(),
        messages=(),
        dispositions=(
            _disposition(symbol.record_id, symbol.partition),
            _disposition(_summary_definition(symbol).record_id, symbol.partition),
            _disposition(factory.record_id, factory.partition),
        ),
    )
    registry = legacy_manifest_registry()
    assert registry.from_record(registry.to_record(manifest)) == manifest
    with pytest.raises(LegacyManifestError, match="MakeProperty states"):
        manifest.validate_complete(
            owned_partitions=(LegacyPartition.GENERATED_PROPERTY_API,)
        )


def test_observations_reject_contradictory_origins_for_symbols_and_factories() -> None:
    contradictory = _states(
        (EvidenceOrigin.EVALUATED, EvidenceOutcome.PRESENT),
        (EvidenceOrigin.EVALUATED, EvidenceOutcome.ABSENT),
    )
    with pytest.raises(LegacyManifestError, match="contradictory"):
        SymbolRecord.create("Conflict", LegacyPartition.MATRICES, contradictory)
    with pytest.raises(LegacyManifestError, match="contradictory"):
        FactoryRecord.create("Conflict", contradictory, requested_names=())
    manifest = _manifest()
    object.__setattr__(manifest.factories[0], "observations", contradictory)
    with pytest.raises(LegacyManifestError, match="semantic_hash"):
        manifest.validate_complete(
            owned_partitions=(LegacyPartition.GENERATED_PROPERTY_API,)
        )


@pytest.mark.parametrize("record_id", ("symbol:000", "symbol:zzz"))
def test_final_property_gate_rejects_same_name_duplicates_regardless_of_order(
    record_id: str,
) -> None:
    manifest = _manifest()
    property_symbol = next(
        item for item in manifest.symbols if item.name in EXPECTED_PROPERTY_NAMES
    )
    duplicate = SymbolRecord.create(
        property_symbol.name,
        property_symbol.partition,
        property_symbol.observations,
        record_id=record_id,
    )
    with pytest.raises(LegacyManifestError, match=r"symbols.*duplicate"):
        _manifest(symbols=(*manifest.symbols, duplicate)).validate_complete(
            owned_partitions=(LegacyPartition.GENERATED_PROPERTY_API,)
        )


@pytest.mark.parametrize("target", ("symbol", "behavior", "disposition", "hash"))
def test_public_manifest_methods_reject_stale_hash_tampering(target: str) -> None:
    manifest = _manifest()
    if target == "symbol":
        object.__setattr__(manifest.symbols[0], "name", "Tampered")
    elif target == "behavior":
        object.__setattr__(manifest.behaviors[0], "description", "Tampered")
    elif target == "disposition":
        object.__setattr__(manifest.dispositions[0], "note", "Tampered rationale")
    else:
        object.__setattr__(manifest.semantic_hash, "digest", "0" * 64)
    with pytest.raises(LegacyManifestError, match="semantic_hash"):
        manifest.validate_complete(
            owned_partitions=(LegacyPartition.GENERATED_PROPERTY_API,)
        )
    with pytest.raises(LegacyManifestError, match="semantic_hash"):
        manifest.assign_disposition(
            manifest.factories[0].record_id,
            Disposition.REPLACE,
            note="replacement rationale",
            api_ids=("api",),
            test_ids=("test",),
            oracle_ids=("oracle",),
        )


@pytest.mark.parametrize(
    ("disposition", "owner", "note", "oracle_ids"),
    (
        (Disposition.REPLACE, "v0.1", "later owner evasion", ()),
        (Disposition.DEFER, "v0.1", "too short", ()),
    ),
)
def test_final_gate_enforces_owner_oracle_and_deferred_explanation(
    disposition: Disposition, owner: str, note: str, oracle_ids: tuple[str, ...]
) -> None:
    symbol = SymbolRecord.create(
        "OwnerGate",
        LegacyPartition.MATRICES,
        _states((EvidenceOrigin.STATIC, EvidenceOutcome.ABSENT)),
    )
    entry = DispositionRecord.create(
        symbol.record_id,
        symbol.partition,
        disposition,
        owner=owner,
        note=note,
        api_ids=("api.owner",),
        test_ids=("test.owner",),
        oracle_ids=oracle_ids,
    )
    manifest = AlgMulManifest.create(
        manifest_id="owner-gate",
        source=SourceMetadata.create("C:/AlgMul.wl", _hash("owner")),
        symbols=(symbol,),
        factories=(),
        registries=(),
        behaviors=(),
        messages=(),
        dispositions=(entry,),
    )
    with pytest.raises(LegacyManifestError, match="ownership or oracle"):
        manifest.validate_complete(owned_partitions=(LegacyPartition.MATRICES,))


def test_stable_definition_and_surface_api_names_are_manifest_compatible() -> None:
    manifest = _manifest()
    assert AlgMulDefinitionRecord.__name__ == "AlgMulDefinitionRecord"
    assert AlgMulSurfaceManifest is AlgMulManifest
    assert isinstance(manifest, AlgMulSurfaceManifest)
    assert callable(manifest.validate_complete)
    assert callable(manifest.assign_disposition)


def test_pinned_static_production_shape_round_trips_structured_evidence() -> None:
    static = _states((EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT))
    intended = _states((EvidenceOrigin.INTENDED, EvidenceOutcome.PRESENT))
    maker = SymbolRecord.create(
        "MakeProperty",
        LegacyPartition.GENERATED_PROPERTY_API,
        static,
        definition_state=DefinitionState.PRESENT_DEFINED,
        definition_kind=DefinitionKind.OWN_VALUE,
        ownvalue_count=1,
        attributes=(SymbolAttribute.PROTECTED,),
        option_hashes=(_hash("maker options"),),
        source_span=SourceSpan.create(1, 1),
        definition_hash=_hash("maker definition"),
        summary_origin=EvidenceOrigin.STATIC,
    )
    intended_properties = tuple(
        SymbolRecord.create(name, LegacyPartition.GENERATED_PROPERTY_API, intended)
        for name in EXPECTED_PROPERTY_NAMES
    )
    summaries = tuple(
        SymbolRecord.create(
            f"Definition{index:03}",
            LegacyPartition.MATRICES,
            static,
            definition_state=DefinitionState.PRESENT_DEFINED,
            definition_kind=DefinitionKind.MIXED,
            downvalue_count=1,
            ownvalue_count=1,
            upvalue_count=0,
            subvalue_count=0,
            attributes=(SymbolAttribute.PROTECTED,),
            option_hashes=(_hash(f"options-{index}"),),
            source_span=SourceSpan.create(index + 2, index + 2),
            definition_hash=_hash(f"definition-{index}"),
            summary_origin=EvidenceOrigin.STATIC,
            record_id=f"definition:{index:03}",
        )
        for index in range(351)
    )
    symbols = (maker, *intended_properties, *summaries)
    package_names = tuple(f"Package{index:03}" for index in range(470))
    factories = (
        FactoryRecord.create(
            "MakeProperty",
            static,
            requested_names=EXPECTED_PROPERTY_NAMES,
            call_kind=FactoryCallKind.GENERATES,
            call_relationships=(maker.record_id,),
        ),
        FactoryRecord.create(
            "PackageCatalog",
            static,
            requested_names=package_names,
            call_kind=FactoryCallKind.INVOKES,
            call_relationships=(summaries[0].record_id,),
        ),
    )
    registries = tuple(
        RegistryRecord.create(
            f"Registry{index:02}",
            ("1", f"e{index}"),
            "OMul",
            basis_table_snapshot=_hash(f"snapshot-{index}"),
            basis_table_hash=_hash(f"table-{index}"),
            origin=EvidenceOrigin.STATIC,
        )
        for index in range(20)
    )
    behaviors = (
        BehaviorRecord.create(
            "MakeChar",
            "registry mutation",
            input_snapshot=_hash("behavior-input"),
            output_snapshot=_hash("behavior-output"),
        ),
    )
    definition_records = tuple(
        _summary_definition(item) for item in symbols if item.summary_origin
    )
    partitions_by_symbol = {item.record_id: item.partition for item in symbols}
    dispositions = tuple(
        _disposition(record.record_id, record.partition)
        for records in (symbols, factories, registries, behaviors)
        for record in records
    ) + tuple(
        _disposition(record.record_id, partitions_by_symbol[record.symbol_id])
        for record in definition_records
    )
    manifest = AlgMulManifest.create(
        manifest_id="v83-static-production-shape",
        source=SourceMetadata.create(
            "C:/AlgMul.wl",
            _hash("v83 source"),
            contexts=("AlgMul`", "System`"),
            load_status=LoadStatus.STATIC_ONLY,
            source_bytes=180_000,
        ),
        symbols=symbols,
        definitions=definition_records,
        factories=factories,
        registries=registries,
        behaviors=behaviors,
        messages=(),
        dispositions=dispositions,
    )
    assert len(package_names) == 470
    assert len(manifest.symbols) == 480
    assert len(intended_properties) == 128
    assert len(manifest.registries) == 20
    registry = legacy_manifest_registry()
    assert registry.from_record(registry.to_record(manifest)) == manifest
    assert canonical_manifest_bytes(manifest) == manifest.canonical_bytes()
    object.__setattr__(manifest.registries[0], "basis_table_hash", _hash("tampered"))
    with pytest.raises(LegacyManifestError, match="semantic_hash"):
        manifest.validate_complete(owned_partitions=(LegacyPartition.REGISTRY_BASES,))


def test_evaluated_artifact_shape_and_per_origin_definitions_round_trip() -> None:
    artifact = json.loads(
        (
            Path(__file__).parents[2]
            / "docs/legacy/generated/algmul-evaluated-surface.json"
        ).read_text()
    )
    attributes = {
        attribute
        for item in artifact["symbolSummaries"]
        for attribute in item["attributes"]
    }
    assert attributes == {"Temporary"}
    assert artifact["loadStatus"] == "completed-with-missing-required-symbols"
    assert artifact["newPackageNameCount"] == 470
    assert artifact["newGlobalNameCount"] == 10
    assert len(artifact["symbolSummaries"]) == 480
    assert len(artifact["algebraRegistry"]) == 20
    assert artifact["expectedPropertySymbolCount"] == 128

    def definition_shape(
        item: dict[str, object],
    ) -> tuple[DefinitionState, DefinitionKind, tuple[int, int, int, int]]:
        counts = (
            item["downValueCount"],
            item["ownValueCount"],
            item["upValueCount"],
            item["subValueCount"],
        )
        assert all(type(count) is int for count in counts)
        typed_counts = cast(tuple[int, int, int, int], counts)
        down, own, up, sub = typed_counts
        if not any(typed_counts):
            return DefinitionState.PRESENT_UNDEFINED, DefinitionKind.NONE, typed_counts
        kind = {
            (True, False, False, False): DefinitionKind.DOWN_VALUE,
            (False, True, False, False): DefinitionKind.OWN_VALUE,
            (False, False, True, False): DefinitionKind.UP_VALUE,
            (False, False, False, True): DefinitionKind.SUB_VALUE,
        }.get((down > 0, own > 0, up > 0, sub > 0), DefinitionKind.MIXED)
        return DefinitionState.PRESENT_DEFINED, kind, typed_counts

    source = SourceMetadata.create(
        artifact["source"],
        SemanticHash("sha256", artifact["sourceSHA256"].lower()),
        kernel_version=artifact["kernelVersion"],
        load_status=LoadStatus.COMPLETED_WITH_MISSING_REQUIRED_SYMBOLS,
        initial_context=artifact["initialContext"],
        context_after_load=artifact["contextAfterLoad"],
        context_path_after_load=tuple(artifact["contextPathAfterLoad"]),
        load_result_head=artifact["loadResultHead"],
    )
    summaries = tuple(
        SymbolRecord.create(
            item["name"],
            LegacyPartition.MATRICES,
            _states((EvidenceOrigin.EVALUATED, EvidenceOutcome.PRESENT)),
            definition_state=definition_shape(item)[0],
            definition_kind=definition_shape(item)[1],
            downvalue_count=definition_shape(item)[2][0],
            ownvalue_count=definition_shape(item)[2][1],
            upvalue_count=definition_shape(item)[2][2],
            subvalue_count=definition_shape(item)[2][3],
            attributes=tuple(attribute.lower() for attribute in item["attributes"]),
            summary_origin=EvidenceOrigin.EVALUATED,
            record_id=f"summary:{index}",
        )
        for index, item in enumerate(artifact["symbolSummaries"])
    )
    definitions = tuple(
        AlgMulDefinitionRecord.create(
            symbol.record_id,
            EvidenceOrigin.EVALUATED,
            definition_shape(item)[0],
            definition_shape(item)[1],
            downvalue_count=definition_shape(item)[2][0],
            ownvalue_count=definition_shape(item)[2][1],
            upvalue_count=definition_shape(item)[2][2],
            subvalue_count=definition_shape(item)[2][3],
            attributes=tuple(attribute.lower() for attribute in item["attributes"]),
            record_id=f"evaluated:{index}",
        )
        for index, (symbol, item) in enumerate(
            zip(summaries, artifact["symbolSummaries"], strict=True)
        )
    )
    properties = _property_symbols()
    factory = FactoryRecord.create(
        "MakeProperty",
        _states(
            (EvidenceOrigin.EVALUATED, EvidenceOutcome.ABSENT),
            (EvidenceOrigin.INTENDED, EvidenceOutcome.PRESENT),
        ),
        requested_names=EXPECTED_PROPERTY_NAMES,
    )
    packages = FactoryRecord.create(
        "PackageCatalog",
        _states((EvidenceOrigin.EVALUATED, EvidenceOutcome.PRESENT)),
        requested_names=tuple(artifact["newPackageNames"]),
    )
    registries = tuple(
        RegistryRecord.create(
            item["name"],
            (item["characters"],),
            item["multiplication"],
            structure_dot_symbol=item["structureDot"],
            record_id=f"registry:{index}",
        )
        for index, item in enumerate(artifact["algebraRegistry"])
    )
    symbols = (*summaries, *properties)
    factories = (factory, packages)
    summary_partitions = {item.record_id: item.partition for item in symbols}
    dispositions = tuple(
        _disposition(record.record_id, record.partition)
        for records in (symbols, factories, registries)
        for record in records
    ) + tuple(
        _disposition(item.record_id, summary_partitions[item.symbol_id])
        for item in definitions
    )
    manifest = AlgMulManifest.create(
        manifest_id="evaluated-artifact-shape",
        source=source,
        symbols=symbols,
        definitions=definitions,
        factories=factories,
        registries=registries,
        behaviors=(),
        messages=(),
        dispositions=dispositions,
    )
    assert manifest.source.initial_context == artifact["initialContext"]
    assert manifest.source.context_after_load == artifact["contextAfterLoad"]
    assert manifest.source.context_path_after_load == tuple(
        artifact["contextPathAfterLoad"]
    )
    assert all(
        SymbolAttribute.TEMPORARY in item.attributes
        for item, raw in zip(definitions, artifact["symbolSummaries"], strict=True)
        if raw["attributes"]
    )
    assert (
        sum(
            item.definition_state is DefinitionState.PRESENT_DEFINED
            for item in definitions
        )
        == 251
    )
    assert (
        sum(
            item.definition_state is DefinitionState.PRESENT_UNDEFINED
            for item in definitions
        )
        == 229
    )
    assert tuple(
        (
            item.name,
            item.basis[0],
            item.multiplication_symbol,
            item.structure_dot_symbol,
        )
        for item in registries
    ) == tuple(
        (
            item["name"],
            item["characters"],
            item["multiplication"],
            item["structureDot"],
        )
        for item in artifact["algebraRegistry"]
    )
    registry = legacy_manifest_registry()
    assert registry.from_record(registry.to_record(manifest)) == manifest
    with pytest.raises(LegacyManifestError, match="MakeProperty"):
        manifest.validate_complete(
            owned_partitions=(LegacyPartition.GENERATED_PROPERTY_API,)
        )
    object.__setattr__(manifest.definitions[1], "downvalue_count", 999)
    with pytest.raises(LegacyManifestError, match="semantic_hash"):
        canonical_manifest_bytes(manifest)


def test_per_origin_definitions_support_runtime_generation_and_repeated_literals() -> (
    None
):
    generated = SymbolRecord.create(
        "GeneratedOnly",
        LegacyPartition.MATRICES,
        _states(
            (EvidenceOrigin.STATIC, EvidenceOutcome.ABSENT),
            (EvidenceOrigin.EVALUATED, EvidenceOutcome.PRESENT),
        ),
        definition_state=DefinitionState.ABSENT,
    )
    osmul = SymbolRecord.create(
        "OsMul",
        LegacyPartition.MATRICES,
        _states((EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT)),
    )
    definitions = (
        AlgMulDefinitionRecord.create(
            generated.record_id,
            EvidenceOrigin.EVALUATED,
            DefinitionState.PRESENT_DEFINED,
            DefinitionKind.DOWN_VALUE,
            downvalue_count=3,
        ),
        *(
            AlgMulDefinitionRecord.create(
                osmul.record_id,
                EvidenceOrigin.STATIC,
                DefinitionState.PRESENT_DEFINED,
                DefinitionKind.DOWN_VALUE,
                downvalue_count=1,
                source_span=SourceSpan.create(line, line),
                definition_hash=_hash(f"osmul literal {line}"),
                record_id=f"definition:osmul:{line}",
            )
            for line in (11, 12)
        ),
    )
    manifest = AlgMulManifest.create(
        manifest_id="per-origin-definitions",
        source=_runtime_source(_hash("evaluated source"), "12.0"),
        symbols=(generated, osmul),
        definitions=definitions,
        factories=(),
        registries=(),
        behaviors=(),
        messages=(),
        dispositions=(
            _disposition(generated.record_id, generated.partition),
            _disposition(osmul.record_id, osmul.partition),
            *(
                _disposition(item.record_id, LegacyPartition.MATRICES)
                for item in definitions
            ),
        ),
    )
    assert manifest.definitions[0].downvalue_count == 3
    assert (
        len(
            [item for item in manifest.definitions if item.symbol_id == osmul.record_id]
        )
        == 2
    )
    assert (
        legacy_manifest_registry().from_record(
            legacy_manifest_registry().to_record(manifest)
        )
        == manifest
    )


def test_static_only_source_rejects_evaluated_provenance() -> None:
    symbol = SymbolRecord.create(
        "EvaluatedInStatic",
        LegacyPartition.MATRICES,
        _states((EvidenceOrigin.EVALUATED, EvidenceOutcome.PRESENT)),
    )
    with pytest.raises(LegacyManifestError, match="non-runtime"):
        AlgMulManifest.create(
            manifest_id="invalid-static",
            source=SourceMetadata.create(
                "C:/AlgMul.wl", _hash("source"), load_status=LoadStatus.STATIC_ONLY
            ),
            symbols=(symbol,),
            factories=(),
            registries=(),
            behaviors=(),
            messages=(),
            dispositions=(_disposition(symbol.record_id, symbol.partition),),
        )


def test_repeated_anchored_definition_literals_can_be_down_and_subvalues() -> None:
    symbol = SymbolRecord.create(
        "RepeatedLiteral",
        LegacyPartition.MATRICES,
        _states((EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT)),
    )
    definitions = (
        AlgMulDefinitionRecord.create(
            symbol.record_id,
            EvidenceOrigin.STATIC,
            DefinitionState.PRESENT_DEFINED,
            DefinitionKind.DOWN_VALUE,
            downvalue_count=1,
            source_span=SourceSpan.create(1, 1),
            definition_hash=_hash("first"),
            record_id="literal:first",
        ),
        AlgMulDefinitionRecord.create(
            symbol.record_id,
            EvidenceOrigin.STATIC,
            DefinitionState.PRESENT_DEFINED,
            DefinitionKind.SUB_VALUE,
            subvalue_count=1,
            source_span=SourceSpan.create(2, 2),
            definition_hash=_hash("second"),
            record_id="literal:second",
        ),
    )
    manifest = AlgMulManifest.create(
        manifest_id="mismatched-literals",
        source=SourceMetadata.create("C:/AlgMul.wl", _hash("source")),
        symbols=(symbol,),
        definitions=definitions,
        factories=(),
        registries=(),
        behaviors=(),
        messages=(),
        dispositions=(
            _disposition(symbol.record_id, symbol.partition),
            *(_disposition(item.record_id, symbol.partition) for item in definitions),
        ),
    )
    assert manifest.definitions == definitions


def test_repeated_static_source_sites_must_not_reuse_an_anchor() -> None:
    symbol = SymbolRecord.create(
        "DuplicateLiteralAnchor",
        LegacyPartition.MATRICES,
        _states((EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT)),
    )
    definitions = tuple(
        AlgMulDefinitionRecord.create(
            symbol.record_id,
            EvidenceOrigin.STATIC,
            DefinitionState.PRESENT_DEFINED,
            DefinitionKind.DOWN_VALUE,
            downvalue_count=1,
            source_span=SourceSpan.create(4, 4),
            definition_hash=_hash("shared source site"),
            record_id=f"duplicate-anchor:{index}",
        )
        for index in (1, 2)
    )
    with pytest.raises(LegacyManifestError, match="duplicate static source sites"):
        AlgMulManifest.create(
            manifest_id="duplicate-static-anchor",
            source=SourceMetadata.create("C:/AlgMul.wl", _hash("source")),
            symbols=(symbol,),
            definitions=definitions,
            factories=(),
            registries=(),
            behaviors=(),
            messages=(),
            dispositions=(
                _disposition(symbol.record_id, symbol.partition),
                *(
                    _disposition(item.record_id, symbol.partition)
                    for item in definitions
                ),
            ),
        )


@pytest.mark.parametrize(
    ("first_kind", "second_kind"),
    (
        (DefinitionKind.DOWN_VALUE, DefinitionKind.OWN_VALUE),
        (DefinitionKind.DOWN_VALUE, DefinitionKind.DOWN_VALUE),
    ),
)
def test_repeated_non_site_origin_definitions_are_always_rejected(
    first_kind: DefinitionKind,
    second_kind: DefinitionKind,
) -> None:
    symbol = SymbolRecord.create(
        "RepeatedEvaluatedSummary",
        LegacyPartition.MATRICES,
        _states((EvidenceOrigin.EVALUATED, EvidenceOutcome.PRESENT)),
    )

    def definition(kind: DefinitionKind, record_id: str) -> AlgMulDefinitionRecord:
        counts = {
            DefinitionKind.DOWN_VALUE: (1, 0),
            DefinitionKind.OWN_VALUE: (0, 1),
        }[kind]
        return AlgMulDefinitionRecord.create(
            symbol.record_id,
            EvidenceOrigin.EVALUATED,
            DefinitionState.PRESENT_DEFINED,
            kind,
            downvalue_count=counts[0],
            ownvalue_count=counts[1],
            record_id=record_id,
        )

    definitions = (
        definition(first_kind, "summary:one"),
        definition(second_kind, "summary:two"),
    )
    with pytest.raises(LegacyManifestError, match="repeated origin definitions"):
        AlgMulManifest.create(
            manifest_id="repeated-evaluated-summary",
            source=_runtime_source(_hash("runtime source")),
            symbols=(symbol,),
            definitions=definitions,
            factories=(),
            registries=(),
            behaviors=(),
            messages=(),
            dispositions=(
                _disposition(symbol.record_id, symbol.partition),
                *(
                    _disposition(item.record_id, symbol.partition)
                    for item in definitions
                ),
            ),
        )


def test_static_source_sites_cannot_launder_attributes_or_options() -> None:
    symbol = SymbolRecord.create(
        "AttributedStaticSite",
        LegacyPartition.MATRICES,
        _states((EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT)),
    )
    definition = AlgMulDefinitionRecord.create(
        symbol.record_id,
        EvidenceOrigin.STATIC,
        DefinitionState.PRESENT_DEFINED,
        DefinitionKind.DOWN_VALUE,
        downvalue_count=1,
        attributes=(SymbolAttribute.PROTECTED,),
        source_span=SourceSpan.create(1, 1),
        definition_hash=_hash("attributed literal"),
    )
    with pytest.raises(LegacyManifestError, match="static source sites"):
        AlgMulManifest.create(
            manifest_id="attributed-static-site",
            source=SourceMetadata.create("C:/AlgMul.wl", _hash("source")),
            symbols=(symbol,),
            definitions=(definition,),
            factories=(),
            registries=(),
            behaviors=(),
            messages=(),
            dispositions=(_disposition(symbol.record_id, symbol.partition),),
        )


def test_static_source_sites_cannot_mix_with_summary_records() -> None:
    symbol = SymbolRecord.create(
        "MixedStaticSite",
        LegacyPartition.MATRICES,
        _states((EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT)),
        definition_state=DefinitionState.PRESENT_DEFINED,
        attributes=(SymbolAttribute.PROTECTED,),
        source_span=SourceSpan.create(2, 2),
        definition_hash=_hash("summary"),
        summary_origin=EvidenceOrigin.STATIC,
    )
    source_site = AlgMulDefinitionRecord.create(
        symbol.record_id,
        EvidenceOrigin.STATIC,
        DefinitionState.PRESENT_DEFINED,
        DefinitionKind.DOWN_VALUE,
        downvalue_count=1,
        source_span=SourceSpan.create(1, 1),
        definition_hash=_hash("literal"),
        record_id="mixed:site",
    )
    summary = AlgMulDefinitionRecord.create(
        symbol.record_id,
        EvidenceOrigin.STATIC,
        DefinitionState.PRESENT_DEFINED,
        DefinitionKind.DOWN_VALUE,
        downvalue_count=1,
        attributes=(SymbolAttribute.PROTECTED,),
        source_span=SourceSpan.create(2, 2),
        definition_hash=_hash("summary"),
        record_id="mixed:summary",
    )
    with pytest.raises(LegacyManifestError, match="cannot mix"):
        AlgMulManifest.create(
            manifest_id="mixed-static-site",
            source=SourceMetadata.create("C:/AlgMul.wl", _hash("source")),
            symbols=(symbol,),
            definitions=(source_site, summary),
            factories=(),
            registries=(),
            behaviors=(),
            messages=(),
            dispositions=(
                _disposition(symbol.record_id, symbol.partition),
                _disposition(source_site.record_id, symbol.partition),
                _disposition(summary.record_id, symbol.partition),
            ),
        )


def test_wire_tampering_cannot_convert_static_site_into_summary() -> None:
    symbol = SymbolRecord.create(
        "WireStaticSite",
        LegacyPartition.MATRICES,
        _states((EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT)),
    )
    definition = AlgMulDefinitionRecord.create(
        symbol.record_id,
        EvidenceOrigin.STATIC,
        DefinitionState.PRESENT_DEFINED,
        DefinitionKind.DOWN_VALUE,
        downvalue_count=1,
        source_span=SourceSpan.create(7, 7),
        definition_hash=_hash("wire literal"),
    )
    manifest = AlgMulManifest.create(
        manifest_id="wire-static-site",
        source=SourceMetadata.create("C:/AlgMul.wl", _hash("source")),
        symbols=(symbol,),
        definitions=(definition,),
        factories=(),
        registries=(),
        behaviors=(),
        messages=(),
        dispositions=(
            _disposition(symbol.record_id, symbol.partition),
            _disposition(definition.record_id, symbol.partition),
        ),
    )
    registry = legacy_manifest_registry()
    wire = registry.to_record(manifest)
    tampered = json.loads(json.dumps(wire))
    first = json.loads(tampered["definitionPages"][0][0])
    first["downvalueCount"] = 2
    tampered["definitionPages"][0][0] = json.dumps(first, separators=(",", ":"))
    with pytest.raises((LegacyManifestError, SchemaError), match="parser_failed"):
        registry.from_record(tampered)


@pytest.mark.parametrize(
    ("kind", "counts", "span", "definition_hash"),
    (
        (DefinitionKind.DOWN_VALUE, (2, 0, 0, 0), SourceSpan.create(1, 1), _hash("a")),
        (DefinitionKind.DOWN_VALUE, (1, 0, 0, 0), None, _hash("b")),
        (DefinitionKind.DOWN_VALUE, (1, 0, 0, 0), SourceSpan.create(1, 1), None),
    ),
)
def test_static_source_site_rejects_nonliteral_or_unanchored_records(
    kind: DefinitionKind,
    counts: tuple[int, int, int, int],
    span: SourceSpan | None,
    definition_hash: SemanticHash | None,
) -> None:
    symbol = SymbolRecord.create(
        "BadStaticSite",
        LegacyPartition.MATRICES,
        _states((EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT)),
    )
    if span is None or definition_hash is None:
        with pytest.raises(LegacyManifestError, match="source"):
            AlgMulDefinitionRecord.create(
                symbol.record_id,
                EvidenceOrigin.STATIC,
                DefinitionState.PRESENT_DEFINED,
                kind,
                downvalue_count=counts[0],
                ownvalue_count=counts[1],
                upvalue_count=counts[2],
                subvalue_count=counts[3],
                source_span=span,
                definition_hash=definition_hash,
            )
        return
    definition = AlgMulDefinitionRecord.create(
        symbol.record_id,
        EvidenceOrigin.STATIC,
        DefinitionState.PRESENT_DEFINED,
        kind,
        downvalue_count=counts[0],
        ownvalue_count=counts[1],
        upvalue_count=counts[2],
        subvalue_count=counts[3],
        source_span=span,
        definition_hash=definition_hash,
    )
    with pytest.raises(LegacyManifestError, match="static source sites"):
        AlgMulManifest.create(
            manifest_id="bad-static-site",
            source=SourceMetadata.create("C:/AlgMul.wl", _hash("source")),
            symbols=(symbol,),
            definitions=(definition,),
            factories=(),
            registries=(),
            behaviors=(),
            messages=(),
            dispositions=(_disposition(symbol.record_id, symbol.partition),),
        )


def test_unknown_source_rejects_runtime_metadata_and_evaluated_provenance() -> None:
    with pytest.raises(LegacyManifestError, match="non-runtime source"):
        SourceMetadata.create("C:/AlgMul.wl", _hash("source"), kernel_version="12")
    symbol = SymbolRecord.create(
        "EvaluatedUnknown",
        LegacyPartition.MATRICES,
        _states((EvidenceOrigin.EVALUATED, EvidenceOutcome.PRESENT)),
    )
    with pytest.raises(LegacyManifestError, match="non-runtime capture"):
        AlgMulManifest.create(
            manifest_id="invalid-unknown",
            source=SourceMetadata.create("C:/AlgMul.wl", _hash("source")),
            symbols=(symbol,),
            factories=(),
            registries=(),
            behaviors=(),
            messages=(),
            dispositions=(_disposition(symbol.record_id, symbol.partition),),
        )


def test_definition_targets_are_independently_traced_and_cross_checked() -> None:
    symbol = SymbolRecord.create(
        "OsMul",
        LegacyPartition.MATRICES,
        _states((EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT)),
    )
    definitions = tuple(
        AlgMulDefinitionRecord.create(
            symbol.record_id,
            EvidenceOrigin.STATIC,
            DefinitionState.PRESENT_DEFINED,
            DefinitionKind.DOWN_VALUE,
            downvalue_count=1,
            source_span=SourceSpan.create(line, line),
            definition_hash=_hash(f"literal {line}"),
            record_id=f"literal:{line}",
        )
        for line in (2, 3)
    )
    base = dict(
        manifest_id="definition-trace",
        source=SourceMetadata.create("C:/AlgMul.wl", _hash("source")),
        symbols=(symbol,),
        definitions=definitions,
        factories=(),
        registries=(),
        behaviors=(),
        messages=(),
    )
    incomplete = AlgMulManifest.create(
        **base, dispositions=(_disposition(symbol.record_id, symbol.partition),)
    )
    with pytest.raises(LegacyManifestError, match="incomplete"):
        incomplete.validate_complete(owned_partitions=(LegacyPartition.MATRICES,))
    complete = AlgMulManifest.create(
        **base,
        dispositions=(
            _disposition(symbol.record_id, symbol.partition),
            *(_disposition(item.record_id, symbol.partition) for item in definitions),
        ),
    )
    complete.validate_complete(owned_partitions=(LegacyPartition.MATRICES,))
    contradictory = SymbolRecord.create(
        "AbsentRuntime",
        LegacyPartition.MATRICES,
        _states((EvidenceOrigin.EVALUATED, EvidenceOutcome.ABSENT)),
    )
    with pytest.raises(LegacyManifestError, match="contradicts"):
        AlgMulManifest.create(
            manifest_id="contradiction",
            source=_runtime_source(_hash("source")),
            symbols=(contradictory,),
            definitions=(
                AlgMulDefinitionRecord.create(
                    contradictory.record_id,
                    EvidenceOrigin.EVALUATED,
                    DefinitionState.PRESENT_DEFINED,
                    DefinitionKind.DOWN_VALUE,
                    downvalue_count=1,
                ),
            ),
            factories=(),
            registries=(),
            behaviors=(),
            messages=(),
            dispositions=(
                _disposition(contradictory.record_id, contradictory.partition),
            ),
        )


def test_message_origin_is_part_of_identity_and_canonical_order() -> None:
    source = _runtime_source(_hash("source"))
    static = MessageRecord.create(
        "load", "warning", "same", origin=EvidenceOrigin.STATIC
    )
    evaluated = MessageRecord.create(
        "load", "warning", "same", origin=EvidenceOrigin.EVALUATED
    )
    first = AlgMulManifest.create(
        manifest_id="message-origin",
        source=source,
        symbols=(),
        factories=(),
        registries=(),
        behaviors=(),
        messages=(evaluated, static),
        dispositions=(),
    )
    second = AlgMulManifest.create(
        manifest_id="message-origin",
        source=source,
        symbols=(),
        factories=(),
        registries=(),
        behaviors=(),
        messages=(static, evaluated),
        dispositions=(),
    )
    assert first.canonical_bytes() == second.canonical_bytes()


def test_message_records_are_first_class_display_disposition_targets() -> None:
    """Diagnostics cannot vanish from completeness accounting."""

    manifest = _manifest()
    message = manifest.messages[0]
    target_id = models._message_target_id(message)
    assert target_id == models._message_target_id(message)
    with pytest.raises(LegacyManifestError, match="incomplete disposition"):
        manifest.validate_complete(
            owned_partitions=(LegacyPartition.DISPLAY_REPORT_TABLES,)
        )
    completed = manifest.assign_disposition(
        target_id,
        Disposition.DEFER,
        owner="v0.1",
        note="captured diagnostic remains an explicit future parity obligation",
        api_ids=("api.legacy.capture_runtime",),
        test_ids=("test.legacy.evaluated_surface",),
        oracle_ids=(),
    )
    completed.validate_complete(
        owned_partitions=(LegacyPartition.DISPLAY_REPORT_TABLES,)
    )
    with pytest.raises(LegacyManifestError, match="does not name a record"):
        manifest.assign_disposition(
            "message:" + "0" * 64,
            Disposition.DEFER,
            owner="v0.1",
            note="unknown diagnostic target remains rejected by the manifest",
            api_ids=("api.legacy.capture_runtime",),
            test_ids=("test.legacy.evaluated_surface",),
            oracle_ids=(),
        )
