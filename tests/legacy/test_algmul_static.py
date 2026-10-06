"""Non-executing, source-faithful tests for the V00-083 static capture."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import cast

import pytest

from anyalgebra.legacy.algmul_static import (
    MAX_SOURCE_BYTES,
    PINNED_ALGMUL_SHA256,
    StaticCaptureError,
    _RUNTIME_LEAK_NAMES,
    _partition,
    capture_pinned_static,
    capture_static,
)
from anyalgebra.legacy.models import (
    EXPECTED_PROPERTY_NAMES,
    AlgMulDefinitionRecord,
    AlgMulManifest,
    BehaviorRecord,
    DefinitionKind,
    EvidenceOrigin,
    EvidenceOutcome,
    FactoryCallKind,
    FactoryRecord,
    LegacyPartition,
    LoadStatus,
    SymbolRecord,
    legacy_manifest_registry,
)


_STATIC_FIXTURE = (
    Path(__file__).parents[1] / "fixtures" / "legacy" / "algmul-static-manifest.json"
)


def _pinned_manifest() -> AlgMulManifest:
    value = legacy_manifest_registry().from_record(
        json.loads(_STATIC_FIXTURE.read_text(encoding="utf-8"))
    )
    return cast(AlgMulManifest, value)


def _states(
    record: SymbolRecord | FactoryRecord,
) -> set[tuple[EvidenceOrigin, EvidenceOutcome]]:
    return {(item.origin, item.outcome) for item in record.observations}


def _definitions(manifest: AlgMulManifest, name: str) -> list[AlgMulDefinitionRecord]:
    symbols = {item.name: item for item in manifest.symbols}
    return [
        item
        for item in manifest.definitions
        if item.symbol_id == symbols[name].record_id
    ]


def _spans(
    definitions: list[AlgMulDefinitionRecord],
) -> list[tuple[int, int]]:
    assert all(item.source_span is not None for item in definitions)
    return [
        (item.source_span.start_line, item.source_span.end_line)
        for item in definitions
        if item.source_span is not None
    ]


def _behavior_covers_line(record: BehaviorRecord, line: int) -> bool:
    match = re.search(r"lines (\d+)-(\d+);", record.description)
    assert match is not None
    return int(match.group(1)) <= line <= int(match.group(2))


def _behavior_start_line(record: BehaviorRecord) -> int:
    match = re.search(r"lines (\d+)-", record.description)
    assert match is not None
    return int(match.group(1))


def test_section_four_catalogue_classifier_is_complete_and_precedence_bound() -> None:
    """Every documented section-4 family has a representative classifier case."""

    representatives = {
        "alg": LegacyPartition.REGISTRY_BASES,
        "MakeAlg": LegacyPartition.ARBITRARY_STRUCTURES,
        "CMul": LegacyPartition.SCALAR_COMPOSITION_PRODUCTS,
        "SymToList": LegacyPartition.SYMBOLIC_LIST_CONVERSION,
        "ArrayToList": LegacyPartition.ARBITRARY_ARRAYS,
        "TMul": LegacyPartition.TENSOR_PRODUCTS,
        "MMul": LegacyPartition.MATRICES,
        "J2": LegacyPartition.JORDAN_MATRICES,
        "AntiComm": LegacyPartition.PRODUCT_OPERATORS,
        "Comm": LegacyPartition.PROPERTY_KERNELS,
        "AProperty1": LegacyPartition.PROPERTY_WRAPPERS,
        "MakeProperty": LegacyPartition.GENERATED_PROPERTY_API,
        "AElement": LegacyPartition.GENERIC_ELEMENTS,
        "BasisDecompose": LegacyPartition.SPAN_STRUCTURE_RECOVERY,
        "Conj": LegacyPartition.CONJUGATIONS_NORMS,
        "GAMul": LegacyPartition.CLIFFORD_GEOMETRIC_GRASSMANN,
        "SOnFund": LegacyPartition.LIE_GROUP_EXPERIMENTS,
        "PrintProperty": LegacyPartition.DISPLAY_REPORT_TABLES,
        "Times": LegacyPartition.GLOBAL_ALIASES_OPERATORS,
        "MakeT": LegacyPartition.INCOMPLETE_STUBS_COMMENTS,
    }
    assert {_partition(name) for name in representatives} == set(LegacyPartition)
    assert {name: _partition(name) for name in representatives} == representatives

    # Exact documented families must win over otherwise tempting spelling rules.
    assert _partition("TJ2") is LegacyPartition.JORDAN_MATRICES
    assert _partition("MAlternative") is LegacyPartition.PROPERTY_KERNELS
    assert _partition("TConj") is LegacyPartition.CONJUGATIONS_NORMS
    assert _partition("GAMul") is LegacyPartition.CLIFFORD_GEOMETRIC_GRASSMANN
    assert _partition("PrintProperty") is LegacyPartition.DISPLAY_REPORT_TABLES
    assert _partition("MTMul") is LegacyPartition.MATRICES
    assert _partition("Mul") is LegacyPartition.PRODUCT_OPERATORS
    assert _partition("MulOld") is LegacyPartition.PRODUCT_OPERATORS
    assert _partition("MulTable") is LegacyPartition.PRODUCT_OPERATORS
    assert _partition("CxCsMul") is LegacyPartition.TENSOR_PRODUCTS
    assert _partition("SymbolToTimes") is LegacyPartition.CLIFFORD_GEOMETRIC_GRASSMANN
    assert _partition("II") is LegacyPartition.CLIFFORD_GEOMETRIC_GRASSMANN
    assert _partition("ATProperty1") is LegacyPartition.PROPERTY_WRAPPERS
    assert _partition("IsNMTAlls") is LegacyPartition.DISPLAY_REPORT_TABLES
    assert _partition("MulToStruct") is LegacyPartition.SPAN_STRUCTURE_RECOVERY
    assert _partition("MulSolve") is LegacyPartition.PRODUCT_OPERATORS
    with pytest.raises(StaticCaptureError, match="unreviewed"):
        _partition("unknownLegacyStub", strict=True)
    assert _partition("e1") is LegacyPartition.REGISTRY_BASES
    assert _partition("G7") is LegacyPartition.REGISTRY_BASES
    assert _partition("ii") is LegacyPartition.REGISTRY_BASES
    assert _partition("if3") is LegacyPartition.REGISTRY_BASES
    assert _partition("\u00cf\u0095") is LegacyPartition.REGISTRY_BASES
    assert (
        _partition("alg1$", runtime_surface=True)
        is LegacyPartition.GLOBAL_ALIASES_OPERATORS
    )
    assert "alg1$" in _RUNTIME_LEAK_NAMES
    with pytest.raises(StaticCaptureError, match="unreviewed AlgMul catalogue"):
        _partition("alg1$", strict=True)
    with pytest.raises(StaticCaptureError, match="unreviewed evaluated"):
        _partition("runtimeImplementationLocal", runtime_surface=True)


@pytest.mark.parametrize(
    "name",
    (
        "MUnknown",
        "GammaFabricated",
        "SUFabricated",
        "NestedFabricated",
        "FooConjBar",
        "FooReport",
        "fabricatedRuntimeName",
    ),
)
def test_runtime_classifier_rejects_future_heuristic_lookalikes(name: str) -> None:
    """The evaluated receipt admits only exact reviewed runtime names."""

    with pytest.raises(StaticCaptureError, match="unreviewed evaluated"):
        _partition(name, runtime_surface=True)


@pytest.mark.legacy
def test_pinned_classifier_covers_catalogue_and_definitions_inherit_symbols() -> None:
    manifest = _pinned_manifest()
    symbols = {item.record_id: item for item in manifest.symbols}
    observed = {item.partition for item in manifest.symbols}
    assert observed == set(LegacyPartition)
    assert all(
        symbols[item.symbol_id].partition is _partition(symbols[item.symbol_id].name)
        for item in manifest.definitions
    )
    names = {item.name: item.partition for item in manifest.symbols}
    assert names["SymToList"] is LegacyPartition.SYMBOLIC_LIST_CONVERSION
    assert names["J2"] is LegacyPartition.JORDAN_MATRICES
    assert names["GAMul"] is LegacyPartition.CLIFFORD_GEOMETRIC_GRASSMANN
    assert names["e"] is LegacyPartition.REGISTRY_BASES
    assert names["MPowerAssociative"] is LegacyPartition.PROPERTY_KERNELS
    assert names["MTPowerAssociative"] is LegacyPartition.PROPERTY_KERNELS
    assert names["MTFlexible"] is LegacyPartition.PROPERTY_KERNELS
    assert names["MakeSOnFund"] is LegacyPartition.LIE_GROUP_EXPERIMENTS
    assert names["MakeSUnFund"] is LegacyPartition.LIE_GROUP_EXPERIMENTS
    assert names["MakeSPnFund"] is LegacyPartition.LIE_GROUP_EXPERIMENTS
    assert names["MulOld"] is LegacyPartition.PRODUCT_OPERATORS
    assert names["MulTable"] is LegacyPartition.PRODUCT_OPERATORS
    assert names["MulPairs"] is LegacyPartition.PRODUCT_OPERATORS
    assert names["SYMtoTLIST"] is LegacyPartition.INCOMPLETE_STUBS_COMMENTS
    assert {
        item.name
        for item in manifest.symbols
        if item.partition is LegacyPartition.INCOMPLETE_STUBS_COMMENTS
    } == {"GenMomenta", "GenMomentum", "SYMtoTLIST"}
    incomplete_definition_names = {
        symbols[item.symbol_id].name
        for item in manifest.definitions
        if symbols[item.symbol_id].partition
        is LegacyPartition.INCOMPLETE_STUBS_COMMENTS
    }
    assert incomplete_definition_names == {"GenMomenta", "GenMomentum", "SYMtoTLIST"}
    assert not {
        item.name
        for item in manifest.factories
        if item.partition is LegacyPartition.INCOMPLETE_STUBS_COMMENTS
    }
    assert {
        item.record_id
        for item in manifest.behaviors
        if item.partition is LegacyPartition.INCOMPLETE_STUBS_COMMENTS
    } == {
        "behavior:defect-comment:1:249:249:82d9fc7b0d3dbfbb",
        "behavior:defect-comment:2:264:264:c61fe5eff0f7a8f4",
        "behavior:defect-comment:3:862:862:45c326f39da7f093",
        "behavior:defect-comment:4:935:935:4c926707f5a89755",
        "behavior:defect-comment:5:1921:1921:c5a8288e4bb5e307",
        "behavior:defect-comment:6:2349:2349:4f071a502774cede",
        "behavior:defect-comment:7:2840:2840:31aea2d3a80f8320",
        "behavior:defect-comment:8:3821:3821:ba1d81348fba2ea3",
        "behavior:static-expression:MakeTChars:none:28:84:84:5a6976d6a43bc181",
    }


@pytest.mark.legacy
def test_pinned_expression_partitions_use_source_receipts_not_apparent_heads() -> None:
    manifest = _pinned_manifest()
    behaviors = {item.record_id: item for item in manifest.behaviors}
    for line in range(53, 61):
        records = [
            item
            for item in behaviors.values()
            if item.description.startswith("Static top-level executable expression;")
            and _behavior_covers_line(item, line)
        ]
        assert len(records) == 1
        assert records[0].partition is LegacyPartition.REGISTRY_BASES
    assert (
        behaviors[
            "behavior:static-expression:MakeTChars:none:28:84:84:5a6976d6a43bc181"
        ].partition
        is LegacyPartition.INCOMPLETE_STUBS_COMMENTS
    )
    assert (
        behaviors[
            "behavior:static-expression:Do:alg-alg-i-Table-SymToList-alg-alg-i-j-"
            "alg-alg-i-j-Length-alg-alg-i-i-Length-alg:82:404:404:604e53f1f9013c75"
        ].partition
        is LegacyPartition.REGISTRY_BASES
    )
    assert (
        behaviors[
            "behavior:static-expression:Range:none:784:3591:3591:72891c5f9298fc8e"
        ].partition
        is LegacyPartition.DISPLAY_REPORT_TABLES
    )


@pytest.mark.legacy
def test_pinned_capture_inventory() -> None:
    manifest = _pinned_manifest()

    assert manifest.source.content_hash.digest.upper() == PINNED_ALGMUL_SHA256
    assert manifest.source.source_bytes == 194012
    assert manifest.source.load_status is LoadStatus.STATIC_ONLY
    assert manifest.source.kernel_version is None
    assert not manifest.registries
    assert not manifest.messages
    assert len(manifest.definitions) == 688
    assert len(manifest.symbols) == 730  # 602 literal heads plus 128 intended shells
    assert _spans(_definitions(manifest, "MakeChar")) == [(196, 202), (203, 203)]
    assert _spans(_definitions(manifest, "MakeTMul")) == [(261, 270), (272, 272)]
    assert _spans(_definitions(manifest, "MakeAlg")) == [(739, 739)]
    assert _spans(_definitions(manifest, "MakeProperty")) == [(1750, 1805)]
    assert all(
        item.source_span is not None and item.definition_hash is not None
        for item in manifest.definitions
    )
    make_property = _definitions(manifest, "MakeProperty")[0]
    assert make_property.definition_hash is not None
    assert (
        make_property.definition_hash.digest
        == "5dfc78e334b758c3bfb5dc62c3bf90a9b26d9b95b2547ac5061f55fa8b26cb21"
    )


@pytest.mark.legacy
def test_dynamic_factories_are_callsite_evidence() -> None:
    manifest = _pinned_manifest()
    factories = {item.name: item for item in manifest.factories}
    symbols = {item.name: item for item in manifest.symbols}

    assert set(factories) == {
        "MakeChar",
        "MakeTMul",
        "MakeAlg",
        "MakeProperty",
        "ToExpression",
    }
    assert {name: len(factories[name].call_relationships) for name in factories} == {
        "MakeChar": 6,
        "MakeTMul": 1,
        "MakeAlg": 5,
        "MakeProperty": 8,
        "ToExpression": 160,
    }
    assert {item.call_kind for item in factories.values()} == {
        FactoryCallKind.GENERATES
    }
    assert all(":181" in item for item in factories["MakeProperty"].call_relationships)
    assert {(item.name, item.partition) for item in manifest.factories} >= {
        ("MakeChar", symbols["MakeChar"].partition),
        ("MakeTMul", symbols["MakeTMul"].partition),
        ("MakeAlg", symbols["MakeAlg"].partition),
    }
    assert factories["MakeProperty"].requested_names == EXPECTED_PROPERTY_NAMES
    assert _states(factories["MakeProperty"]) == {
        (EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT),
        (EvidenceOrigin.INTENDED, EvidenceOutcome.PRESENT),
    }
    for name in EXPECTED_PROPERTY_NAMES:
        assert _states(symbols[name]) == {
            (EvidenceOrigin.INTENDED, EvidenceOutcome.PRESENT)
        }
        assert not _definitions(manifest, name)
    assert not any(
        item.symbol_id
        in {symbols["ACommutative"].record_id, symbols["NJacobi"].record_id}
        for item in manifest.definitions
    )
    property_calls = factories["MakeProperty"].call_relationships
    property_receipt = next(
        item
        for item in manifest.behaviors
        if item.record_id == "behavior:static-callsite:MakeProperty"
    )
    assert property_receipt.input_snapshot is not None
    assert (
        property_receipt.input_snapshot.digest
        == hashlib.sha256("|".join(property_calls).encode("utf-8")).hexdigest()
    )


def test_newline_separated_multiline_and_repeated_definitions_are_top_level_only(
    tmp_path: Path,
) -> None:
    source = tmp_path / "syntax.wl"
    source.write_text(
        """(* outer (* Fake[x_] := x; *) comment *)
f[x_] := Module[{local},
    local = 1;
    x + local
]
f[x_, y_] := x + y
g[x_] := \"not a definition: h[x_] := x;\"
""",
        encoding="utf-8",
    )

    manifest = capture_static(source)
    names = {item.name for item in manifest.symbols}
    assert {"f", "g"} <= names
    assert not {"Fake", "h", "local"} & names
    definitions = _definitions(manifest, "f")
    assert len(definitions) == 2
    assert {item.definition_kind for item in definitions} == {DefinitionKind.DOWN_VALUE}
    assert {item.downvalue_count for item in definitions} == {1}
    assert len({item.definition_hash for item in definitions}) == 2
    assert [start for start, _ in _spans(definitions)] == [2, 6]


def test_usage_messages_are_not_function_ownvalues() -> None:
    manifest = _pinned_manifest()
    usage = [
        item
        for item in manifest.behaviors
        if item.record_id.startswith("behavior:usage-message:")
    ]
    assert len(usage) == 13
    assert all("slice sha256:" in item.description for item in usage)
    assert all(item.input_snapshot is not None for item in usage)
    assert all(
        item.input_snapshot is not None
        and item.input_snapshot.digest in item.description
        for item in usage
    )
    for item in _definitions(manifest, "TChars"):
        assert item.definition_kind is DefinitionKind.DOWN_VALUE
        assert item.ownvalue_count == 0


def test_unpinned_and_generic_makeproperty_are_static_only_without_invented_surface(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tiny.wl"
    source.write_text('MakeChar[x_] := x;\nMakeProperty["X"] := x;\n', encoding="utf-8")

    manifest = capture_static(source)

    assert manifest.source.load_status is LoadStatus.STATIC_ONLY
    assert manifest.source.content_hash.digest.upper() != PINNED_ALGMUL_SHA256
    assert not any(
        observation.origin is EvidenceOrigin.EVALUATED
        for record in manifest.symbols
        for observation in record.observations
    )
    assert not any(
        observation.origin is EvidenceOrigin.EVALUATED
        for record in manifest.factories
        for observation in record.observations
    )
    assert not any(item.name in EXPECTED_PROPERTY_NAMES for item in manifest.symbols)
    assert "MakeProperty" not in {item.name for item in manifest.factories}
    assert "ToExpression" not in {item.name for item in manifest.factories}
    assert not manifest.registries
    with pytest.raises(StaticCaptureError, match="hash"):
        capture_pinned_static(source)


def test_wrong_property_calls_or_missing_definition_do_not_invent_the_surface(
    tmp_path: Path,
) -> None:
    source = tmp_path / "wrong-property.wl"
    source.write_text(
        "\n".join(
            (
                'MakeProperty["Commutative", 99, "Comm"];',
                'MakeProperty["Associative", 3, "Assoc"];',
                'MakeProperty["Alternative", 2, "Alternative"];',
                'MakeProperty["Flexible", 2, "Flexible"];',
                'MakeProperty["PowerAssociative1", 1, "PowerAssociative1"];',
                'MakeProperty["PowerAssociative2", 1, "PowerAssociative2"];',
                'MakeProperty["JIdentity", 2, "JIdentity"];',
                'MakeProperty["Jacobi", 3, "JacobiIdentity"];',
            )
        ),
        encoding="utf-8",
    )
    manifest = capture_static(source)

    assert "MakeProperty" not in {item.name for item in manifest.factories}
    assert not {item.name for item in manifest.symbols} & set(EXPECTED_PROPERTY_NAMES)
    assert any(
        item.record_id == "behavior:static-callsite:MakeProperty"
        for item in manifest.behaviors
    )


def test_undefined_factory_calls_are_behaviors_not_factory_definitions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "undefined-factories.wl"
    source.write_text(
        'MakeChar[{1}, "X"];\nMakeTMul["C", "H"];\n'
        'MakeAlg["A", {1}, x];\nToExpression["MadeAtRuntime"];\n',
        encoding="utf-8",
    )
    manifest = capture_static(source)

    assert {item.name for item in manifest.factories} == {"ToExpression"}
    assert {
        item.record_id
        for item in manifest.behaviors
        if "static-callsite" in item.record_id
    } == {
        "behavior:static-callsite:MakeChar",
        "behavior:static-callsite:MakeTMul",
        "behavior:static-callsite:MakeAlg",
        "behavior:static-callsite:ToExpression",
    }


def test_tagset_is_an_upvalue_of_the_tag_not_a_downvalue_of_the_head(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tagset.wl"
    source.write_text(
        "tag /: head[x_] := x\nhead[x_] := x\nordinary[x_] := x\n",
        encoding="utf-8",
    )
    manifest = capture_static(source)

    tag = _definitions(manifest, "tag")
    head = _definitions(manifest, "head")
    ordinary = _definitions(manifest, "ordinary")
    assert {item.definition_kind for item in tag} == {DefinitionKind.UP_VALUE}
    assert {item.upvalue_count for item in tag} == {1}
    assert {item.definition_kind for item in head} == {DefinitionKind.DOWN_VALUE}
    assert {item.definition_kind for item in ordinary} == {DefinitionKind.DOWN_VALUE}
    assert {item.ownvalue_count for item in (*tag, *head, *ordinary)} == {0}


def test_curried_literal_is_a_subvalue_and_upset_is_rejected(tmp_path: Path) -> None:
    curried = tmp_path / "curried.wl"
    curried.write_text("f[x_][y_] := x + y\n", encoding="utf-8")
    definition = _definitions(capture_static(curried), "f")
    assert [item.definition_kind for item in definition] == [DefinitionKind.SUB_VALUE]
    assert definition[0].subvalue_count == 1

    upset = tmp_path / "upset.wl"
    upset.write_text("f[x_] ^= x\n", encoding="utf-8")
    with pytest.raises(StaticCaptureError, match="unsupported UpSet assignment"):
        capture_static(upset)


def test_static_effects_and_defect_comments_are_anchored_source_evidence() -> None:
    manifest = _pinned_manifest()
    static_expressions = [
        item
        for item in manifest.behaviors
        if item.record_id.startswith("behavior:static-expression:")
    ]
    required_effects = {
        601: ("Unprotect", "Star"),
        603: ("SetAttributes", "Star"),
        799: ("Unprotect", "CenterDot,Wedge,Diamond,CircleDot,CircleTimes,SmallCircle"),
        809: ("SetAttributes", "CenterDot"),
        810: ("SetAttributes", "Wedge"),
        811: ("SetAttributes", "Diamond"),
        812: ("SetAttributes", "CircleDot"),
        813: ("SetAttributes", "CircleTimes"),
        814: ("SetAttributes", "SmallCircle"),
        3166: ("Unprotect", "Times"),
        3169: ("Protect", "Times"),
        3170: ("Unprotect", "Power"),
        3173: ("Protect", "Power"),
    }
    for line, (head, targets) in required_effects.items():
        effect = next(
            item for item in static_expressions if _behavior_covers_line(item, line)
        )
        assert f"head={head};" in effect.description
        assert f"targets={targets};" in effect.description
        assert effect.partition is LegacyPartition.GLOBAL_ALIASES_OPERATORS
        assert effect.input_snapshot is not None
    assert not any(_behavior_covers_line(item, 2176) for item in static_expressions)
    assert {span[0] for span in _spans(_definitions(manifest, "Times"))} >= {3167, 3168}
    assert {span[0] for span in _spans(_definitions(manifest, "Power"))} >= {3171, 3172}
    assert any(
        item.partition is LegacyPartition.DISPLAY_REPORT_TABLES
        and "head=Print;" in item.description
        for item in static_expressions
    )

    defects = [
        item
        for item in manifest.behaviors
        if item.record_id.startswith("behavior:defect-comment:")
    ]
    assert len(defects) == 8
    assert [_behavior_start_line(item) for item in defects] == [
        249,
        264,
        862,
        935,
        1921,
        2349,
        2840,
        3821,
    ]
    assert all(item.input_snapshot is not None for item in defects)
    assert not any(_behavior_covers_line(item, 1954) for item in defects)


def test_comments_and_strings_never_become_effects_or_definitions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "comments.wl"
    source.write_text(
        "ClearAll[real]; (* outer (* WRONG fake[x_] := x; *) *)\n"
        '"ClearAll[string] and wrong"\n',
        encoding="utf-8",
    )
    manifest = capture_static(source)
    effects = [
        item
        for item in manifest.behaviors
        if item.record_id.startswith("behavior:static-expression:")
    ]
    assert len(effects) == 1
    assert "head=ClearAll; targets=real;" in effects[0].description
    assert "fake" not in {item.name for item in manifest.symbols}
    defects = [
        item
        for item in manifest.behaviors
        if item.record_id.startswith("behavior:defect-comment:")
    ]
    assert len(defects) == 1


def test_makeproperty_requires_literal_factory_and_exact_top_level_requests(
    tmp_path: Path,
) -> None:
    requests = (
        ('"Commutative", 2, "Comm"',),
        ('"Associative", 3, "Assoc"',),
        ('"Alternative", 2, "Alternative"',),
        ('"Flexible", 2, "Flexible"',),
        ('"PowerAssociative1", 1, "PowerAssociative1"',),
        ('"PowerAssociative2", 1, "PowerAssociative2"',),
        ('"JIdentity", 2, "JIdentity"',),
        ('"Jacobi", 3, "JacobiIdentity"',),
    )
    calls = "\n".join(
        f"(* leading comment *)\n  MakeProperty[{triple[0]}];" for triple in requests
    )
    valid = tmp_path / "valid-property.wl"
    valid.write_text(f"MakeProperty[a_, b_, c_] := a;\n{calls}\n", encoding="utf-8")
    manifest = capture_static(valid)
    assert [item.name for item in manifest.factories] == ["MakeProperty"]
    assert manifest.factories[0].requested_names == EXPECTED_PROPERTY_NAMES
    assert set(EXPECTED_PROPERTY_NAMES) <= {item.name for item in manifest.symbols}

    buried = tmp_path / "buried-property.wl"
    buried.write_text(
        f"MakeProperty[a_, b_, c_] := Module[{{x}}, {calls}]\n", encoding="utf-8"
    )
    buried_manifest = capture_static(buried)
    assert "MakeProperty" not in {item.name for item in buried_manifest.factories}
    assert not (
        set(EXPECTED_PROPERTY_NAMES) & {item.name for item in buried_manifest.symbols}
    )

    own_value = tmp_path / "own-value-property.wl"
    own_value.write_text(f"MakeProperty = 1;\n{calls}\n", encoding="utf-8")
    own_manifest = capture_static(own_value)
    assert "MakeProperty" not in {item.name for item in own_manifest.factories}

    curried = tmp_path / "curried-property.wl"
    curried.write_text(f"MakeProperty[a_][b_] := a;\n{calls}\n", encoding="utf-8")
    curried_manifest = capture_static(curried)
    assert "MakeProperty" not in {item.name for item in curried_manifest.factories}
    assert not (
        set(EXPECTED_PROPERTY_NAMES) & {item.name for item in curried_manifest.symbols}
    )


def test_nested_comments_receipt_each_direct_defect_marker(tmp_path: Path) -> None:
    source = tmp_path / "nested-defects.wl"
    source.write_text("(* WRONG outer (* TODO inner *) *)\n", encoding="utf-8")
    manifest = capture_static(source)
    defects = [
        item
        for item in manifest.behaviors
        if item.record_id.startswith("behavior:defect-comment:")
    ]
    assert len(defects) == 2
    assert all(_behavior_start_line(item) == 1 for item in defects)


def test_factory_requires_downvalue_not_ownvalue(tmp_path: Path) -> None:
    source = tmp_path / "not-a-factory.wl"
    source.write_text('MakeChar = 1;\nMakeChar["O"];\n', encoding="utf-8")
    manifest = capture_static(source)
    assert "MakeChar" not in {item.name for item in manifest.factories}
    assert any(
        item.record_id == "behavior:static-callsite:MakeChar"
        for item in manifest.behaviors
    )


def test_resolved_locations_and_bounded_open_read_are_canonical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "same.wl"
    source.write_text("f[x_] := x;\n", encoding="utf-8")
    absolute = capture_static(source)
    monkeypatch.chdir(tmp_path)
    relative = capture_static(Path("same.wl"))
    assert absolute.source.location == str(source.resolve())
    assert absolute.canonical_bytes() == relative.canonical_bytes()

    class BoundedReader:
        def __enter__(self) -> BoundedReader:
            return self

        def __exit__(
            self,
            exception_type: object,
            exception: object,
            traceback: object,
        ) -> None:
            del exception_type, exception, traceback

        def read(self, size: int = -1) -> bytes:
            assert size == MAX_SOURCE_BYTES + 1
            return b"f[x_] := x;\n"

    monkeypatch.setattr(
        Path, "read_bytes", lambda self: (_ for _ in ()).throw(AssertionError())
    )
    monkeypatch.setattr(Path, "open", lambda self, *args, **kwargs: BoundedReader())
    bounded = capture_static(Path("same.wl"))
    assert {item.name for item in bounded.symbols} == {"f"}


def test_capture_is_deterministic_bounded_and_never_executes_source(
    tmp_path: Path,
) -> None:
    deterministic = tmp_path / "deterministic.wl"
    deterministic.write_text("f[x_] := x;\n", encoding="utf-8")
    first = capture_static(deterministic)
    second = capture_static(deterministic)
    assert first.canonical_bytes() == second.canonical_bytes()
    with pytest.raises(StaticCaptureError, match="regular file"):
        capture_static(tmp_path)
    with pytest.raises(StaticCaptureError, match="regular file"):
        capture_static(tmp_path / "missing.wl")
    huge = tmp_path / "huge.wl"
    huge.write_bytes(b"x" * (8 * 1024 * 1024 + 1))
    with pytest.raises(StaticCaptureError, match="bounded"):
        capture_static(huge)
    for spelling, error in (
        ("f[x_] := x]", "bracket"),
        ("f[x_ := x", "bracket"),
        ('f[x_] := "unterminated', "string"),
        ("(* no close", "comment"),
    ):
        malformed = tmp_path / f"bad-{len(spelling)}.wl"
        malformed.write_text(spelling, encoding="utf-8")
        with pytest.raises(StaticCaptureError, match=error):
            capture_static(malformed)
    invalid_utf8 = tmp_path / "invalid-utf8.wl"
    invalid_utf8.write_bytes(b"f[x_] := \xff")
    with pytest.raises(StaticCaptureError, match="valid UTF-8"):
        capture_static(invalid_utf8)
    marker = tmp_path / "should-not-exist"
    dynamic = tmp_path / "dynamic.wl"
    dynamic.write_text(
        f'MakeChar[x_] := ToExpression["Created"][x];\n(* {marker} *)\n',
        encoding="utf-8",
    )
    manifest = capture_static(dynamic)
    assert not marker.exists()
    assert "Created" not in {item.name for item in manifest.symbols}
    assert {item.name for item in manifest.factories} == {"MakeChar", "ToExpression"}
    assert (
        len(
            next(
                item for item in manifest.factories if item.name == "MakeChar"
            ).call_relationships
        )
        == 0
    )


def test_top_level_comparison_is_static_expression_not_definition(
    tmp_path: Path,
) -> None:
    source = tmp_path / "comparison.wl"
    source.write_text("x == y;\n", encoding="utf-8")

    manifest = capture_static(source)

    assert not manifest.definitions
    assert not manifest.symbols
    expressions = [
        item
        for item in manifest.behaviors
        if item.record_id.startswith("behavior:static-expression:")
    ]
    assert len(expressions) == 1
    assert "head=x; targets=none;" in expressions[0].description
