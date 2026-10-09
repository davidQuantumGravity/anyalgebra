"""Materialize the complete, bounded AlgMul disposition ledgers.

The two evidence layers intentionally remain separate.  Their source and
kernel evidence have different epistemic status and the full union would exceed
the durable manifest model's per-collection limit.  This test is therefore the
canonical expansion algorithm for the compact, reviewed disposition rules.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

import pytest

from anyalgebra.core.parents import SemanticHash
from anyalgebra.legacy.algmul_static import (
    StaticCaptureError,
    _GLOBAL_OPERATOR_NAMES,
    _REGISTRY_BASE_NAMES,
    _RUNTIME_LEAK_NAMES,
    _partition,
)
from anyalgebra.legacy.models import (
    EXPECTED_PROPERTY_NAMES,
    AlgMulDefinitionRecord,
    AlgMulManifest,
    BehaviorRecord,
    DefinitionKind,
    DefinitionState,
    Disposition,
    DispositionRecord,
    EvidenceOrigin,
    EvidenceOutcome,
    FactoryCallKind,
    FactoryRecord,
    LegacyManifestError,
    LegacyPartition,
    MessageRecord,
    OracleStatus,
    RegistryRecord,
    SourceMetadata,
    SymbolObservation,
    SymbolRecord,
    _message_target_id,
    canonical_manifest_bytes,
    legacy_manifest_registry,
)


_ROOT = Path(__file__).resolve().parents[2]
_SURFACE = _ROOT / "docs/legacy/generated/algmul-surface-manifest.json"
_DISPOSITIONS = _ROOT / "docs/legacy/generated/algmul-dispositions.json"
_ORACLE_EVIDENCE = _ROOT / "docs/legacy/generated/algmul-oracle-evidence.json"
_STATIC_FIXTURE = _ROOT / "tests/fixtures/legacy/algmul-static-manifest.json"


def _v89_note(partition: LegacyPartition) -> str:
    return (
        "V00-089 canonical per-record assignment for "
        f"{partition.value}; legacy behavior remains non-oracular evidence."
    )


def _deferred_note(partition: LegacyPartition, release: str) -> str:
    return (
        f"Specialized {partition.value} work is explicitly deferred to {release} "
        "after the v0.0 generic contract."
    )


_CANONICAL_RULES: dict[
    LegacyPartition,
    tuple[str, str, tuple[str, ...], tuple[str, ...], tuple[str, ...], str, str],
] = {
    LegacyPartition.REGISTRY_BASES: (
        "redesign",
        "v0.0",
        ("api.parent.freeze", "api.legacy.validate_complete"),
        ("test.parents.identity_and_mismatch", "test.legacy.disposition_complete"),
        ("oracle.registry.immutable-parent",),
        "available",
        _v89_note(LegacyPartition.REGISTRY_BASES),
    ),
    LegacyPartition.ARBITRARY_STRUCTURES: (
        "preserve-concept",
        "v0.0",
        ("api.structure.build", "api.legacy.validate_complete"),
        ("test.structures.arbitrary_signature", "test.legacy.disposition_complete"),
        ("oracle.structure.explicit-table",),
        "available",
        _v89_note(LegacyPartition.ARBITRARY_STRUCTURES),
    ),
    LegacyPartition.SCALAR_COMPOSITION_PRODUCTS: (
        "preserve-exactly",
        "v0.0",
        ("api.structure.build", "api.legacy.validate_complete"),
        ("test.structures.arbitrary_signature", "test.legacy.disposition_complete"),
        ("oracle.scalar.table-fixture",),
        "available",
        _v89_note(LegacyPartition.SCALAR_COMPOSITION_PRODUCTS),
    ),
    LegacyPartition.SYMBOLIC_LIST_CONVERSION: (
        "preserve-concept",
        "v0.0",
        ("api.presentation.convert", "api.legacy.validate_complete"),
        ("test.presentations.round_trip", "test.legacy.disposition_complete"),
        ("oracle.presentation.round-trip",),
        "available",
        _v89_note(LegacyPartition.SYMBOLIC_LIST_CONVERSION),
    ),
    LegacyPartition.ARBITRARY_ARRAYS: (
        "redesign",
        "v0.0",
        ("api.matrix.construct", "api.legacy.validate_complete"),
        ("test.elements.sparse_canonical", "test.legacy.disposition_complete"),
        ("oracle.shape-canonicality",),
        "available",
        _v89_note(LegacyPartition.ARBITRARY_ARRAYS),
    ),
    LegacyPartition.TENSOR_PRODUCTS: (
        "preserve-concept",
        "v0.0",
        ("api.structure.build", "api.legacy.validate_complete"),
        ("test.analysis.basis_change_invariance", "test.legacy.disposition_complete"),
        ("oracle.tensor-product-contraction",),
        "available",
        _v89_note(LegacyPartition.TENSOR_PRODUCTS),
    ),
    LegacyPartition.MATRICES: (
        "preserve-concept",
        "v0.0",
        ("api.matrix.construct", "api.legacy.validate_complete"),
        ("test.partiality.no_silent_totalization", "test.legacy.disposition_complete"),
        ("oracle.matrix-entry-definition",),
        "available",
        _v89_note(LegacyPartition.MATRICES),
    ),
    LegacyPartition.JORDAN_MATRICES: (
        "defer",
        "v0.2",
        ("api.legacy.capture_static",),
        ("test.legacy.disposition_complete",),
        (),
        "pending",
        _deferred_note(LegacyPartition.JORDAN_MATRICES, "v0.2"),
    ),
    LegacyPartition.PRODUCT_OPERATORS: (
        "preserve-concept",
        "v0.0",
        ("api.validation.validate_law", "api.legacy.validate_complete"),
        (
            "test.validation.proved_disproved_inconclusive",
            "test.legacy.disposition_complete",
        ),
        ("oracle.parenthesized-term",),
        "available",
        _v89_note(LegacyPartition.PRODUCT_OPERATORS),
    ),
    LegacyPartition.PROPERTY_KERNELS: (
        "preserve-concept",
        "v0.0",
        ("api.validation.validate_law", "api.legacy.validate_complete"),
        ("test.legacy.generated_property_family", "test.legacy.disposition_complete"),
        ("oracle.finite-law-validation",),
        "available",
        _v89_note(LegacyPartition.PROPERTY_KERNELS),
    ),
    LegacyPartition.PROPERTY_WRAPPERS: (
        "replace",
        "v0.0",
        ("api.validation.validate_law", "api.legacy.validate_complete"),
        ("test.legacy.generated_property_family", "test.legacy.disposition_complete"),
        ("oracle.typed-law-presentation",),
        "available",
        _v89_note(LegacyPartition.PROPERTY_WRAPPERS),
    ),
    LegacyPartition.GENERATED_PROPERTY_API: (
        "replace",
        "v0.0",
        ("api.legacy.validate_complete",),
        ("test.legacy.generated_property_family", "test.legacy.disposition_complete"),
        ("oracle.typed-eight-law-registry",),
        "available",
        _v89_note(LegacyPartition.GENERATED_PROPERTY_API),
    ),
    LegacyPartition.GENERIC_ELEMENTS: (
        "preserve-concept",
        "v0.0",
        ("api.structure.evaluate", "api.legacy.validate_complete"),
        ("test.evidence.no_self_promotion", "test.legacy.disposition_complete"),
        ("oracle.seeded-domain",),
        "available",
        _v89_note(LegacyPartition.GENERIC_ELEMENTS),
    ),
    LegacyPartition.SPAN_STRUCTURE_RECOVERY: (
        "preserve-concept",
        "v0.0",
        ("api.linear.recover_structure_constants", "api.legacy.validate_complete"),
        ("test.linear.unique_recovery", "test.legacy.disposition_complete"),
        ("oracle.linear-witness",),
        "available",
        _v89_note(LegacyPartition.SPAN_STRUCTURE_RECOVERY),
    ),
    LegacyPartition.CONJUGATIONS_NORMS: (
        "defer",
        "v0.2",
        ("api.legacy.capture_static",),
        ("test.legacy.disposition_complete",),
        (),
        "pending",
        _deferred_note(LegacyPartition.CONJUGATIONS_NORMS, "v0.2"),
    ),
    LegacyPartition.CLIFFORD_GEOMETRIC_GRASSMANN: (
        "defer",
        "v0.3",
        ("api.legacy.capture_static",),
        ("test.legacy.disposition_complete",),
        (),
        "pending",
        _deferred_note(LegacyPartition.CLIFFORD_GEOMETRIC_GRASSMANN, "v0.3"),
    ),
    LegacyPartition.LIE_GROUP_EXPERIMENTS: (
        "defer",
        "v0.4",
        ("api.legacy.capture_static",),
        ("test.legacy.disposition_complete",),
        (),
        "pending",
        _deferred_note(LegacyPartition.LIE_GROUP_EXPERIMENTS, "v0.4"),
    ),
    LegacyPartition.DISPLAY_REPORT_TABLES: (
        "redesign",
        "v0.0",
        ("api.evidence.contract", "api.legacy.validate_complete"),
        ("test.evidence.no_self_promotion", "test.legacy.disposition_complete"),
        ("oracle.structured-report",),
        "available",
        _v89_note(LegacyPartition.DISPLAY_REPORT_TABLES),
    ),
    LegacyPartition.GLOBAL_ALIASES_OPERATORS: (
        "drop",
        "v0.0",
        ("api.parent.freeze", "api.legacy.validate_complete"),
        ("test.parents.identity_and_mismatch", "test.legacy.disposition_complete"),
        ("oracle.no-implicit-state",),
        "available",
        _v89_note(LegacyPartition.GLOBAL_ALIASES_OPERATORS),
    ),
    LegacyPartition.INCOMPLETE_STUBS_COMMENTS: (
        "defer",
        "v0.1",
        ("api.legacy.capture_static",),
        ("test.legacy.disposition_complete",),
        (),
        "pending",
        _deferred_note(LegacyPartition.INCOMPLETE_STUBS_COMMENTS, "v0.1"),
    ),
}

# V00-089 classifies every partition, but classification completeness is not
# implementation completion.  Only the six partitions below have adequate
# independent v0.0 evidence.  The remaining nondeferred dispositions retain
# their normative decision while naming the exact oracle gap.
_AVAILABLE_ORACLE_IDS: dict[LegacyPartition, tuple[str, ...]] = {
    LegacyPartition.ARBITRARY_STRUCTURES: (
        "oracle.structure.many-sorted-builder",
        "oracle.structure.callable-evaluation",
        "oracle.structure.explicit-constants",
    ),
    LegacyPartition.SYMBOLIC_LIST_CONVERSION: ("oracle.presentation.round-trip",),
    LegacyPartition.PROPERTY_KERNELS: (
        "oracle.property.kernel-ast",
        "oracle.property.legacy-parenthesization",
    ),
    LegacyPartition.PROPERTY_WRAPPERS: ("oracle.property.wrapper-evaluation",),
    LegacyPartition.GENERATED_PROPERTY_API: ("oracle.property.generated-api",),
    LegacyPartition.SPAN_STRUCTURE_RECOVERY: (
        "oracle.linear.span",
        "oracle.linear.recovery",
    ),
}
_PENDING_ORACLE_GAPS: dict[LegacyPartition, str] = {
    LegacyPartition.REGISTRY_BASES: (
        "V00-089 redesigns registry bases around immutable parents, but an "
        "independent registry-table and mutation-isolation oracle is still missing."
    ),
    LegacyPartition.SCALAR_COMPOSITION_PRODUCTS: (
        "V00-089 preserves named C, Cs, H, Hs, O, and Os fixtures exactly, but "
        "executable C/Cs tables and exhaustive independent named-product checks "
        "are still missing."
    ),
    LegacyPartition.ARBITRARY_ARRAYS: (
        "V00-089 redesigns arbitrary arrays with explicit shape and parent "
        "metadata, but an independent arbitrary-rank shape oracle is still missing."
    ),
    LegacyPartition.TENSOR_PRODUCTS: (
        "V00-089 preserves typed tensor-product concepts, but independent "
        "multiplication and contraction oracles for nested factors are still missing."
    ),
    LegacyPartition.MATRICES: (
        "V00-089 preserves typed matrix concepts, but independent entrywise and "
        "nonassociative matrix-product oracles are still missing."
    ),
    LegacyPartition.PRODUCT_OPERATORS: (
        "V00-089 preserves explicit parenthesized product operators, but "
        "anticommutator, nested-fold, and legacy operator-route oracles are "
        "still missing."
    ),
    LegacyPartition.GENERIC_ELEMENTS: (
        "V00-089 preserves generic-element intent with deterministic domains, but "
        "an independent seeded generation and shrinkage oracle is still missing."
    ),
    LegacyPartition.DISPLAY_REPORT_TABLES: (
        "V00-089 redesigns display output as structured reports, but an "
        "independent rendering and report-content oracle is still missing."
    ),
    LegacyPartition.GLOBAL_ALIASES_OPERATORS: (
        "V00-089 drops ambient aliases and global operator mutation, but an "
        "independent clean-process no-global-state oracle is still missing."
    ),
}
_CANONICAL_RULES = {
    partition: (
        rule[0],
        rule[1],
        rule[2],
        rule[3],
        (
            ()
            if partition in _PENDING_ORACLE_GAPS
            else _AVAILABLE_ORACLE_IDS.get(partition, rule[4])
        ),
        (OracleStatus.PENDING.value if partition in _PENDING_ORACLE_GAPS else rule[5]),
        _PENDING_ORACLE_GAPS.get(partition, rule[6]),
    )
    for partition, rule in _CANONICAL_RULES.items()
}

# CTX-030 permits only these registered legacy tests as disposition owners.
# Independent semantic evidence is separately path/hash/node bound in
# algmul-oracle-evidence.json and must never be replaced by invented test IDs.
_V89_TEST_IDS = frozenset(
    {
        "test.legacy.evaluated_surface",
        "test.legacy.generated_property_family",
        "test.legacy.disposition_complete",
    }
)
_V89_TESTS_BY_PARTITION: dict[LegacyPartition, tuple[str, ...]] = {
    LegacyPartition.REGISTRY_BASES: (
        "test.legacy.evaluated_surface",
        "test.legacy.disposition_complete",
    ),
    LegacyPartition.SCALAR_COMPOSITION_PRODUCTS: (
        "test.legacy.evaluated_surface",
        "test.legacy.disposition_complete",
    ),
    LegacyPartition.SYMBOLIC_LIST_CONVERSION: (
        "test.legacy.evaluated_surface",
        "test.legacy.disposition_complete",
    ),
    LegacyPartition.TENSOR_PRODUCTS: (
        "test.legacy.evaluated_surface",
        "test.legacy.disposition_complete",
    ),
    LegacyPartition.MATRICES: (
        "test.legacy.evaluated_surface",
        "test.legacy.disposition_complete",
    ),
    LegacyPartition.PROPERTY_KERNELS: (
        "test.legacy.generated_property_family",
        "test.legacy.disposition_complete",
    ),
    LegacyPartition.PROPERTY_WRAPPERS: (
        "test.legacy.generated_property_family",
        "test.legacy.disposition_complete",
    ),
    LegacyPartition.GENERATED_PROPERTY_API: (
        "test.legacy.generated_property_family",
        "test.legacy.disposition_complete",
    ),
    LegacyPartition.DISPLAY_REPORT_TABLES: (
        "test.legacy.evaluated_surface",
        "test.legacy.disposition_complete",
    ),
}
_CANONICAL_RULES = {
    partition: (
        rule[0],
        rule[1],
        rule[2],
        _V89_TESTS_BY_PARTITION.get(partition, ("test.legacy.disposition_complete",)),
        rule[4],
        rule[5],
        rule[6],
    )
    for partition, rule in _CANONICAL_RULES.items()
}


class SurfaceManifestError(ValueError):
    """A compact V00-089 evidence or disposition artifact is not canonical."""


def _read(path: Path, *, canonical: bool = True) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise SurfaceManifestError(f"{path.name} must contain one JSON object")
    expected = json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    if canonical and path.read_bytes() != expected:
        raise SurfaceManifestError(f"{path.name} is not canonical JSON")
    return cast(dict[str, object], value)


def _object(value: object, field: str) -> dict[str, object]:
    if type(value) is not dict:
        raise SurfaceManifestError(f"{field} must be an object")
    return cast(dict[str, object], value)


def _list(value: object, field: str) -> list[object]:
    if type(value) is not list:
        raise SurfaceManifestError(f"{field} must be a list")
    return cast(list[object], value)


def _text(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise SurfaceManifestError(f"{field} must be a nonempty built-in str")
    return value


def _int(value: object, field: str) -> int:
    if type(value) is not int:
        raise SurfaceManifestError(f"{field} must be an exact int")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _layer(surface: Mapping[str, object], layer_id: str) -> dict[str, object]:
    for raw in _list(surface["layers"], "layers"):
        layer = _object(raw, "layer")
        if layer.get("id") == layer_id:
            return layer
    raise SurfaceManifestError(f"missing layer {layer_id}")


def _provenance(surface: Mapping[str, object]) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for raw in _list(surface["provenance"], "provenance"):
        record = _object(raw, "provenance")
        if set(record) != {"id", "path", "sha256", "partition", "kind"}:
            raise SurfaceManifestError("provenance has an unexpected shape")
        identifier = _text(record["id"], "provenance.id")
        if identifier in records:
            raise SurfaceManifestError(f"duplicate provenance {identifier}")
        if _sha256(_relative(record["path"], "provenance.path")) != _text(
            record["sha256"], "provenance.sha256"
        ):
            raise SurfaceManifestError(f"provenance hash mismatch for {identifier}")
        records[identifier] = record
    expected_identifiers = (
        "runtime-load",
        "generated-lifts",
        "makealg",
        "makeproperty-intent",
    )
    if tuple(records) != expected_identifiers:
        raise SurfaceManifestError("provenance set is incomplete")
    return records


def _relative(path: object, field: str) -> Path:
    candidate = Path(_text(path, field))
    if candidate.is_absolute():
        raise SurfaceManifestError(f"{field} must be repository relative")
    resolved = (_ROOT / candidate).resolve()
    if not resolved.is_relative_to(_ROOT.resolve()):
        raise SurfaceManifestError(f"{field} escapes the repository")
    return resolved


def _rules(payload: Mapping[str, object]) -> dict[LegacyPartition, dict[str, object]]:
    records = _list(payload.get("rules"), "rules")
    rules: dict[LegacyPartition, dict[str, object]] = {}
    for raw in records:
        record = _object(raw, "rule")
        expected = {
            "partition",
            "disposition",
            "owner",
            "apiIds",
            "testIds",
            "oracleIds",
            "oracleStatus",
            "note",
        }
        if set(record) != expected:
            raise SurfaceManifestError("rule has an unexpected shape")
        try:
            partition = LegacyPartition(_text(record["partition"], "rule.partition"))
            Disposition(_text(record["disposition"], "rule.disposition"))
            OracleStatus(_text(record["oracleStatus"], "rule.oracleStatus"))
        except ValueError as error:
            raise SurfaceManifestError("rule has an unknown enum") from error
        if partition in rules:
            raise SurfaceManifestError("rule partitions must be unique")
        for link_field in ("apiIds", "testIds", "oracleIds"):
            links = _list(record[link_field], f"rule.{link_field}")
            if any(type(item) is not str or not item for item in links):
                raise SurfaceManifestError(f"rule.{link_field} has an invalid link")
            if len(set(cast(list[str], links))) != len(links):
                raise SurfaceManifestError(f"rule.{link_field} has duplicate links")
        rules[partition] = record
    if set(rules) != set(LegacyPartition):
        missing = sorted(item.value for item in set(LegacyPartition) - set(rules))
        raise SurfaceManifestError(f"unknown or unclassified partitions: {missing}")
    if tuple(rules) != tuple(LegacyPartition):
        raise SurfaceManifestError("rules are not in canonical partition order")
    return rules


def _validate_canonical_rule_contract(
    rules: Mapping[LegacyPartition, Mapping[str, object]],
) -> None:
    actual = {
        partition: (
            _text(record["disposition"], "rule.disposition"),
            _text(record["owner"], "rule.owner"),
            tuple(_list(record["apiIds"], "rule.apiIds")),
            tuple(_list(record["testIds"], "rule.testIds")),
            tuple(_list(record["oracleIds"], "rule.oracleIds")),
            _text(record["oracleStatus"], "rule.oracleStatus"),
            _text(record["note"], "rule.note"),
        )
        for partition, record in rules.items()
    }
    if actual != _CANONICAL_RULES:
        raise SurfaceManifestError("canonical rule contract drift")


def _validate_oracle_evidence(
    rules: Mapping[LegacyPartition, Mapping[str, object]],
    payload: Mapping[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    """Resolve every available oracle ID to external hash- and node-bound evidence."""

    evidence = _read(_ORACLE_EVIDENCE) if payload is None else dict(payload)
    if set(evidence) != {"schema", "purpose", "entries"}:
        raise SurfaceManifestError("oracle evidence has an unexpected shape")
    if evidence["schema"] != "anyalgebra.algmul-oracle-evidence/v1":
        raise SurfaceManifestError("unknown oracle evidence schema")
    expected_purpose = (
        "Externally anchored V00-089 evidence for the six disposition partitions "
        "with adequate independent v0.0 oracles. Pending rules deliberately have "
        "no oracle IDs. These records do not promote legacy observations into "
        "correctness or scientific claims."
    )
    if evidence["purpose"] != expected_purpose:
        raise SurfaceManifestError("oracle evidence purpose drift")

    records: dict[str, dict[str, object]] = {}
    for raw in _list(evidence["entries"], "oracleEvidence.entries"):
        record = _object(raw, "oracleEvidence.entry")
        if set(record) != {
            "id",
            "path",
            "sha256",
            "binding",
            "assertion",
            "limitation",
        }:
            raise SurfaceManifestError("oracle evidence entry has an unexpected shape")
        identifier = _text(record["id"], "oracleEvidence.id")
        if not identifier.startswith("oracle.") or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789.-"
            for character in identifier
        ):
            raise SurfaceManifestError("oracle evidence ID is not stable")
        if identifier in records:
            raise SurfaceManifestError(f"duplicate oracle evidence {identifier}")

        relative_path = _text(record["path"], "oracleEvidence.path")
        path = _relative(relative_path, "oracleEvidence.path")
        if path == Path(__file__).resolve():
            raise SurfaceManifestError("oracle evidence cannot cite its own validator")
        if not path.is_file():
            raise SurfaceManifestError(
                f"oracle evidence path is missing for {identifier}"
            )
        digest = _text(record["sha256"], "oracleEvidence.sha256")
        if (
            len(digest) != 64
            or digest != digest.upper()
            or any(character not in "0123456789ABCDEF" for character in digest)
        ):
            raise SurfaceManifestError("oracle evidence SHA256 is not canonical")
        if _sha256(path) != digest:
            raise SurfaceManifestError(
                f"oracle evidence hash mismatch for {identifier}"
            )
        if len(_text(record["assertion"], "oracleEvidence.assertion").strip()) < 20:
            raise SurfaceManifestError("oracle evidence assertion is not explicit")
        if len(_text(record["limitation"], "oracleEvidence.limitation").strip()) < 20:
            raise SurfaceManifestError("oracle evidence limitation is not explicit")

        binding = _object(record["binding"], "oracleEvidence.binding")
        if set(binding) != {"pytestNodes"}:
            raise SurfaceManifestError(
                "oracle evidence binding is not a pytest contract"
            )
        nodes = _list(binding["pytestNodes"], "oracleEvidence.pytestNodes")
        if not nodes or any(type(node) is not str or not node for node in nodes):
            raise SurfaceManifestError("oracle evidence needs exact pytest nodes")
        if len(set(cast(list[str], nodes))) != len(nodes):
            raise SurfaceManifestError("oracle evidence has duplicate pytest nodes")
        try:
            module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as error:
            raise SurfaceManifestError(
                "oracle evidence test source is unreadable"
            ) from error
        functions = {
            node.name
            for node in module.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        for raw_node in nodes:
            node_id = _text(raw_node, "oracleEvidence.pytestNode")
            node_path, separator, function_name = node_id.partition("::")
            if (
                separator != "::"
                or "::" in function_name
                or node_path != Path(relative_path).as_posix()
                or not function_name.startswith("test_")
                or function_name not in functions
            ):
                raise SurfaceManifestError(
                    f"oracle evidence pytest node is not source-bound: {node_id}"
                )
        records[identifier] = record

    referenced = {
        _text(item, "rule.oracleId")
        for rule in rules.values()
        for item in _list(rule["oracleIds"], "rule.oracleIds")
    }
    unknown = referenced - set(records)
    if unknown:
        raise SurfaceManifestError(
            f"unbacked oracle evidence link: {sorted(unknown)[0]}"
        )
    unreferenced = set(records) - referenced
    if unreferenced:
        raise SurfaceManifestError(
            f"unreferenced oracle evidence: {sorted(unreferenced)[0]}"
        )
    return records


def _authoritative_ids(path: Path, field: str) -> set[str]:
    if not path.is_file():
        pytest.skip("internal process records are not distributed with this checkout")
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not list:
        raise SurfaceManifestError(f"{field} registry must be a list")
    identifiers: set[str] = set()
    for raw in value:
        record = _object(raw, field)
        identifiers.add(_text(record.get("id"), f"{field}.id"))
    return identifiers


def _validate_documents(
    surface: Mapping[str, object], dispositions: Mapping[str, object]
) -> dict[LegacyPartition, dict[str, object]]:
    if surface.get("schema") != "anyalgebra.algmul-surface-manifest/v2":
        raise SurfaceManifestError("unknown surface schema")
    if dispositions.get("schema") != "anyalgebra.algmul-dispositions/v2":
        raise SurfaceManifestError("unknown disposition schema")
    if set(dispositions) != {
        "schema",
        "surfaceManifest",
        "oracleEvidenceRegistry",
        "selection",
        "rules",
        "assignments",
    }:
        raise SurfaceManifestError("dispositions have an unexpected shape")
    expected_surface = {
        "schema",
        "purpose",
        "source",
        "layers",
        "provenance",
        "runtimeBehaviors",
        "targetInventory",
        "limitations",
    }
    if set(surface) != expected_surface:
        raise SurfaceManifestError("surface has an unexpected shape")
    if surface["purpose"] != (
        "Canonical non-oracular V00-089 per-record parity ledger. Static, "
        "evaluated, and property-composite layers remain distinct because they "
        "have different provenance and the combined typed serializer is bounded."
    ):
        raise SurfaceManifestError("surface purpose drift")
    if _list(surface["limitations"], "limitations") != [
        "Legacy evidence is not a mathematical correctness or scientific oracle.",
        (
            "Every target is durably listed by layer, kind, partition, and "
            "origin; assignments live in the companion disposition ledger."
        ),
        (
            "The property composite expressly records required static, "
            "evaluated-absent, and intended obligations."
        ),
    ]:
        raise SurfaceManifestError("surface limitations drift")
    source = _object(surface["source"], "surface.source")
    if source != {
        "path": "${ANYALGEBRA_ALGMUL_SOURCE}",
        "sha256": "3A13C9F47092B5B9814AA08873FF4F1DE61D4635DF68FC7F3A039A7DA5B0D00D",
        "bytes": 194012,
    }:
        raise SurfaceManifestError("surface source binding drift")
    if Path(_text(source["path"], "surface.source.path")) != Path(
        "${ANYALGEBRA_ALGMUL_SOURCE}"
    ):
        raise SurfaceManifestError("surface source path is not exact")
    binding = _object(dispositions.get("surfaceManifest"), "surfaceManifest")
    if set(binding) != {"path", "sha256"}:
        raise SurfaceManifestError("surfaceManifest binding has an invalid shape")
    if _relative(binding["path"], "surfaceManifest.path") != _SURFACE:
        raise SurfaceManifestError("surfaceManifest path is not canonical")
    if _sha256(_SURFACE) != _text(binding["sha256"], "surfaceManifest.sha256"):
        raise SurfaceManifestError("surfaceManifest hash mismatch")
    oracle_binding = _object(
        dispositions.get("oracleEvidenceRegistry"), "oracleEvidenceRegistry"
    )
    if set(oracle_binding) != {"path", "sha256"}:
        raise SurfaceManifestError(
            "oracleEvidenceRegistry binding has an invalid shape"
        )
    if (
        _relative(oracle_binding["path"], "oracleEvidenceRegistry.path")
        != _ORACLE_EVIDENCE
    ):
        raise SurfaceManifestError("oracleEvidenceRegistry path is not canonical")
    if _sha256(_ORACLE_EVIDENCE) != _text(
        oracle_binding["sha256"], "oracleEvidenceRegistry.sha256"
    ):
        raise SurfaceManifestError("oracleEvidenceRegistry hash mismatch")
    if dispositions.get("selection") != (
        "One explicit assignment exists for every durable layer-target pair. "
        "Assignments bind layer, targetId, kind, partition, and rule; rules "
        "carry canonical owner, API/test links, and oracle status. Available "
        "oracle IDs resolve through the bound external evidence registry."
    ):
        raise SurfaceManifestError("selection drift")
    api_ids = _authoritative_ids(
        _ROOT / ".agents/project-process/api-contracts.json", "api"
    )
    test_ids = _authoritative_ids(
        _ROOT / ".agents/project-process/test-matrix.json", "test"
    )
    rules = _rules(dispositions)
    for rule in rules.values():
        rule_api_ids = {str(item) for item in _list(rule["apiIds"], "rule.apiIds")}
        rule_test_ids = {str(item) for item in _list(rule["testIds"], "rule.testIds")}
        if not rule_api_ids <= api_ids:
            raise SurfaceManifestError("rule has an unknown API identifier")
        if not rule_test_ids <= test_ids:
            raise SurfaceManifestError("rule has an unknown test identifier")
        if not rule_test_ids or not rule_test_ids <= _V89_TEST_IDS:
            raise SurfaceManifestError("rule has a non-CTX-030 owning test")
        disposition = Disposition(_text(rule["disposition"], "rule.disposition"))
        owner = _text(rule["owner"], "rule.owner")
        oracle_ids = _list(rule["oracleIds"], "rule.oracleIds")
        oracle_status = OracleStatus(_text(rule["oracleStatus"], "rule.oracleStatus"))
        if owner == "v0.0":
            if disposition is Disposition.DEFER:
                raise SurfaceManifestError("v0.0 owner cannot defer its disposition")
            if oracle_status is OracleStatus.AVAILABLE and not oracle_ids:
                raise SurfaceManifestError("available v0.0 oracle is unbacked")
            if oracle_status is OracleStatus.PENDING and (
                oracle_ids or len(_text(rule["note"], "rule.note").strip()) < 20
            ):
                raise SurfaceManifestError("pending v0.0 oracle gap is invalid")
        elif disposition is not Disposition.DEFER:
            raise SurfaceManifestError("later owner must be an explicit defer")
        elif oracle_ids or oracle_status is not OracleStatus.PENDING:
            raise SurfaceManifestError("unbacked future oracle link")
    _validate_oracle_evidence(rules)
    _validate_canonical_rule_contract(rules)
    _validate_surface_evidence(surface)
    _validate_inventory_and_assignments(surface, dispositions, rules)
    return rules


def _validate_surface_evidence(surface: Mapping[str, object]) -> None:
    """Check every non-inventory surface field against its pinned receipt role."""

    expected_static = {
        "id": "static",
        "origin": "static",
        "producer": "anyalgebra.legacy.algmul_static.capture_pinned_static",
        "symbols": 730,
        "definitions": 688,
        "factories": 5,
        "registries": 0,
        "behaviors": 213,
        "messages": 0,
        "requiredFactories": [
            "MakeChar",
            "MakeTMul",
            "MakeAlg",
            "MakeProperty",
            "ToExpression",
        ],
        "propertySurface": {
            "factory": "MakeProperty",
            "intendedNames": 128,
            "definitionPresent": False,
        },
    }
    expected_evaluated = {
        "id": "evaluated",
        "origin": "evaluated",
        "kernel": "12.0.0 for Microsoft Windows (64-bit) (April 6, 2019)",
        "symbols": 470,
        "registries": 20,
        "messages": 1,
        "loadStatus": "completed-with-missing-required-symbols",
        "missingFactory": "MakeProperty",
        "missingIntendedNames": 128,
        "messageTags": ["Syntax::sntx"],
        "messageSHA256": (
            "E1FB091A3742818B0A226A906F4B62C3B78CB14C344250154F6763FDF2072824"
        ),
        "registryNames": [
            "O",
            "H",
            "C",
            "O2",
            "H2",
            "C2",
            "O3",
            "H3",
            "C3",
            "Os",
            "Hs",
            "Cs",
            "Os2",
            "Hs2",
            "Cs2",
            "Os3",
            "Hs3",
            "Cs3",
            "CxH",
            "D",
        ],
    }
    if _list(surface["layers"], "layers") != [expected_static, expected_evaluated]:
        raise SurfaceManifestError("surface layer evidence drift")
    expected_provenance = {
        "runtime-load": (
            "tests/fixtures/legacy/runtime-load.json",
            "registry_bases",
            "runtime-load-and-registry",
        ),
        "generated-lifts": (
            "tests/fixtures/legacy/generated-lifts.json",
            "tensor_products",
            "MakeChar-and-MakeTMul-exercises",
        ),
        "makealg": (
            "tests/fixtures/legacy/makealg.json",
            "arbitrary_structures",
            "MakeAlg-and-constructor-exercises",
        ),
        "makeproperty-intent": (
            "tests/fixtures/legacy/makeproperty-intent.json",
            "generated_property_api",
            "missing-factory-and-128-name-intent",
        ),
    }
    provenance = _provenance(surface)
    for identifier, (path, partition, kind) in expected_provenance.items():
        record = provenance[identifier]
        if (
            record.get("path") != path
            or record.get("partition") != partition
            or record.get("kind") != kind
        ):
            raise SurfaceManifestError("provenance role drift")
    runtime = _read(
        _relative(provenance["runtime-load"]["path"], "runtime-load.path"),
        canonical=False,
    )
    runtime_source = _object(runtime["source"], "runtime-load.source")
    runtime_kernel = _object(runtime["kernel"], "runtime-load.kernel")
    runtime_load = _object(runtime["load"], "runtime-load.load")
    source = _object(surface["source"], "surface.source")
    if (
        Path(_text(runtime_source["path"], "runtime-load.source.path")).resolve()
        != Path(_text(source["path"], "surface.source.path")).resolve()
        or runtime_source.get("sha256") != source["sha256"]
        or runtime_source.get("bytes") != source["bytes"]
        or runtime_kernel.get("version") != expected_evaluated["kernel"]
        or runtime_load.get("status") != expected_evaluated["loadStatus"]
    ):
        raise SurfaceManifestError("runtime receipt binding drift")
    expected_behaviors = [
        {
            "id": "behavior:runtime-load",
            "partition": "registry_bases",
            "provenanceId": "runtime-load",
        },
        {
            "id": "behavior:generated-lifts",
            "partition": "tensor_products",
            "provenanceId": "generated-lifts",
        },
        {
            "id": "behavior:makealg",
            "partition": "arbitrary_structures",
            "provenanceId": "makealg",
        },
        {
            "id": "behavior:makeproperty-intent",
            "partition": "generated_property_api",
            "provenanceId": "makeproperty-intent",
        },
    ]
    if _list(surface["runtimeBehaviors"], "runtimeBehaviors") != expected_behaviors:
        raise SurfaceManifestError("runtime behavior binding drift")


def _validate_inventory_and_assignments(
    surface: Mapping[str, object],
    dispositions: Mapping[str, object],
    rules: Mapping[LegacyPartition, Mapping[str, object]],
) -> None:
    inventory = _object(surface["targetInventory"], "targetInventory")
    if (
        set(inventory) != {"schema", "layers"}
        or inventory.get("schema") != "anyalgebra.algmul-target-inventory/v1"
    ):
        raise SurfaceManifestError("target inventory schema drift")
    inventory_layers = _list(inventory["layers"], "targetInventory.layers")
    expected_layers = ("static", "evaluated", "property-composite")
    seen_layers: set[str] = set()
    layer_order: list[str] = []
    expected_assignments: set[tuple[str, str, str, str, str]] = set()
    for raw_layer in inventory_layers:
        layer = _object(raw_layer, "targetInventory.layer")
        if set(layer) != {"id", "targetCount", "targetSetSHA256", "records"}:
            raise SurfaceManifestError("target layer has an unexpected shape")
        identifier = _text(layer["id"], "targetInventory.layer.id")
        if identifier in seen_layers:
            raise SurfaceManifestError("duplicate target layer")
        seen_layers.add(identifier)
        layer_order.append(identifier)
        records = _list(layer["records"], "targetInventory.records")
        if len(records) != _int(layer["targetCount"], "targetInventory.targetCount"):
            raise SurfaceManifestError("target count drift")
        canonical = json.dumps(
            records, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        if hashlib.sha256(canonical).hexdigest().upper() != _text(
            layer["targetSetSHA256"], "targetSetSHA256"
        ):
            raise SurfaceManifestError("target set hash drift")
        if records != sorted(
            records,
            key=lambda value: (
                _text(
                    _object(value, "targetInventory.record")["targetId"],
                    "target.targetId",
                ),
                _text(_object(value, "targetInventory.record")["kind"], "target.kind"),
                _text(
                    _object(value, "targetInventory.record")["partition"],
                    "target.partition",
                ),
                _text(
                    _object(value, "targetInventory.record")["origin"], "target.origin"
                ),
            ),
        ):
            raise SurfaceManifestError(
                "target records are not deterministically ordered"
            )
        seen_targets: set[tuple[str, str]] = set()
        for raw_record in records:
            record = _object(raw_record, "targetInventory.record")
            if set(record) != {"targetId", "kind", "partition", "origin"}:
                raise SurfaceManifestError("target record has an unexpected shape")
            target_id = _text(record["targetId"], "target.targetId")
            kind = _text(record["kind"], "target.kind")
            partition = _text(record["partition"], "target.partition")
            origin = _text(record["origin"], "target.origin")
            if kind not in {
                "symbol",
                "definition",
                "factory",
                "registry",
                "behavior",
                "message",
            }:
                raise SurfaceManifestError("unknown target kind")
            if origin not in {
                "static",
                "static-or-intended",
                "evaluated",
                "evaluated-absent",
            }:
                raise SurfaceManifestError("unknown target origin")
            try:
                LegacyPartition(partition)
            except ValueError as error:
                raise SurfaceManifestError("unknown target partition") from error
            if (target_id, kind) in seen_targets:
                raise SurfaceManifestError("duplicate layer target")
            seen_targets.add((target_id, kind))
            expected_assignments.add(
                (identifier, target_id, kind, partition, partition)
            )
    if tuple(layer_order) != expected_layers:
        raise SurfaceManifestError("target inventory layer set drift")
    typed_layers = {
        "static": _typed_target_records(_static_manifest(surface)),
        "evaluated": _typed_target_records(_runtime_manifest(surface)),
        "property-composite": _typed_target_records(
            _property_composite_manifest(surface)
        ),
    }
    for identifier, expected_records in typed_layers.items():
        if _inventory_targets(surface, identifier) != expected_records:
            raise SurfaceManifestError("target inventory evidence origin drift")
    assignments = _list(dispositions.get("assignments"), "assignments")
    actual_assignments: set[tuple[str, str, str, str, str]] = set()
    for raw_assignment in assignments:
        assignment = _object(raw_assignment, "assignment")
        if set(assignment) != {"layer", "targetId", "kind", "partition", "rule"}:
            raise SurfaceManifestError("assignment has an unexpected shape")
        value = (
            _text(assignment["layer"], "assignment.layer"),
            _text(assignment["targetId"], "assignment.targetId"),
            _text(assignment["kind"], "assignment.kind"),
            _text(assignment["partition"], "assignment.partition"),
            _text(assignment["rule"], "assignment.rule"),
        )
        if value in actual_assignments:
            raise SurfaceManifestError("duplicate assignment")
        actual_assignments.add(value)
        if value[4] != value[3] or LegacyPartition(value[4]) not in rules:
            raise SurfaceManifestError("unknown assignment rule")
    layer_rank = {identifier: index for index, identifier in enumerate(expected_layers)}
    if assignments != sorted(
        assignments,
        key=lambda value: (
            layer_rank.get(
                _text(_object(value, "assignment")["layer"], "assignment.layer"), -1
            ),
            _text(_object(value, "assignment")["targetId"], "assignment.targetId"),
            _text(_object(value, "assignment")["kind"], "assignment.kind"),
            _text(_object(value, "assignment")["partition"], "assignment.partition"),
            _text(_object(value, "assignment")["rule"], "assignment.rule"),
        ),
    ):
        raise SurfaceManifestError("assignments are not deterministically ordered")
    if actual_assignments != expected_assignments:
        raise SurfaceManifestError("assignment target set drift")


def _record_targets(
    manifest: AlgMulManifest,
) -> tuple[tuple[str, LegacyPartition], ...]:
    symbols = {item.record_id: item.partition for item in manifest.symbols}
    return (
        tuple(symbols.items())
        + tuple(
            (item.record_id, symbols[item.symbol_id]) for item in manifest.definitions
        )
        + tuple((item.record_id, item.partition) for item in manifest.factories)
        + tuple((item.record_id, item.partition) for item in manifest.registries)
        + tuple((item.record_id, item.partition) for item in manifest.behaviors)
        + tuple(
            (_message_target_id(item), LegacyPartition.DISPLAY_REPORT_TABLES)
            for item in manifest.messages
        )
    )


def _observation_origin(
    observations: Iterable[SymbolObservation], *, symbol: bool
) -> str:
    """Reduce typed observations to the only durable inventory origin tags."""

    states = {(item.origin, item.outcome) for item in observations}
    if (EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT) in states or (
        EvidenceOrigin.INTENDED,
        EvidenceOutcome.PRESENT,
    ) in states:
        return "static-or-intended" if symbol else "static"
    if (EvidenceOrigin.EVALUATED, EvidenceOutcome.PRESENT) in states:
        return "evaluated"
    if (EvidenceOrigin.EVALUATED, EvidenceOutcome.ABSENT) in states:
        return "evaluated-absent"
    raise SurfaceManifestError("target has no durable origin state")


def _typed_target_records(
    manifest: AlgMulManifest,
) -> set[tuple[str, str, str, str]]:
    """Return exact target identity, partition, and evidence-derived origin."""

    partitions = {item.record_id: item.partition.value for item in manifest.symbols}
    return (
        {
            (
                item.record_id,
                "symbol",
                item.partition.value,
                _observation_origin(item.observations, symbol=True),
            )
            for item in manifest.symbols
        }
        | {
            (
                item.record_id,
                "definition",
                partitions[item.symbol_id],
                (
                    "evaluated-absent"
                    if item.origin is EvidenceOrigin.EVALUATED
                    and item.definition_state is DefinitionState.ABSENT
                    else item.origin.value
                ),
            )
            for item in manifest.definitions
        }
        | {
            (
                item.record_id,
                "factory",
                item.partition.value,
                _observation_origin(item.observations, symbol=False),
            )
            for item in manifest.factories
        }
        | {
            (item.record_id, "registry", item.partition.value, item.origin.value)
            for item in manifest.registries
        }
        | {
            (item.record_id, "behavior", item.partition.value, item.origin.value)
            for item in manifest.behaviors
        }
        | {
            (
                _message_target_id(item),
                "message",
                LegacyPartition.DISPLAY_REPORT_TABLES.value,
                item.origin.value,
            )
            for item in manifest.messages
        }
    )


def _typed_targets(manifest: AlgMulManifest) -> set[tuple[str, str, str]]:
    """Return the identity projection used by bounded disposition materialization."""

    return {
        (target_id, kind, partition)
        for target_id, kind, partition, _ in _typed_target_records(manifest)
    }


def _inventory_targets(
    surface: Mapping[str, object], layer_id: str
) -> set[tuple[str, str, str, str]]:
    inventory = _object(surface["targetInventory"], "targetInventory")
    for raw_layer in _list(inventory["layers"], "targetInventory.layers"):
        layer = _object(raw_layer, "targetInventory.layer")
        if layer.get("id") == layer_id:
            return {
                (
                    _text(record["targetId"], "target.targetId"),
                    _text(record["kind"], "target.kind"),
                    _text(record["partition"], "target.partition"),
                    _text(record["origin"], "target.origin"),
                )
                for raw_record in _list(layer["records"], "targetInventory.records")
                for record in (_object(raw_record, "targetInventory.record"),)
            }
    raise SurfaceManifestError(f"missing inventory layer {layer_id}")


def _with_dispositions(
    manifest: AlgMulManifest,
    rules: Mapping[LegacyPartition, Mapping[str, object]],
    assignments: Mapping[tuple[str, str, str], str],
    *,
    layer_id: str,
    owned_partitions: Iterable[LegacyPartition],
) -> AlgMulManifest:
    owned = frozenset(owned_partitions)
    records: list[DispositionRecord] = []
    for target_id, kind, partition_name in _typed_targets(manifest):
        partition = LegacyPartition(partition_name)
        if partition not in owned:
            continue
        rule_name = assignments.get((layer_id, target_id, kind))
        if rule_name is None:
            raise SurfaceManifestError(f"unclassified target {target_id}")
        if rule_name != partition.value:
            raise SurfaceManifestError(f"partition mismatch for {target_id}")
        rule = rules.get(partition)
        if rule is None:
            raise SurfaceManifestError(f"unknown assignment rule for {target_id}")
        records.append(
            DispositionRecord.create(
                target_id,
                partition,
                rule["disposition"],
                owner=rule["owner"],
                note=rule["note"],
                api_ids=tuple(_list(rule["apiIds"], "rule.apiIds")),
                test_ids=tuple(_list(rule["testIds"], "rule.testIds")),
                oracle_ids=tuple(_list(rule["oracleIds"], "rule.oracleIds")),
                oracle_status=rule["oracleStatus"],
            )
        )
    return AlgMulManifest.create(
        manifest_id=manifest.manifest_id,
        source=manifest.source,
        symbols=manifest.symbols,
        definitions=manifest.definitions,
        factories=manifest.factories,
        registries=manifest.registries,
        behaviors=manifest.behaviors,
        messages=manifest.messages,
        dispositions=tuple(records),
    )


def _assignment_map(
    dispositions: Mapping[str, object],
) -> dict[tuple[str, str, str], str]:
    """Return only the explicit durable layer-target assignment relation."""

    result: dict[tuple[str, str, str], str] = {}
    for raw in _list(dispositions["assignments"], "assignments"):
        record = _object(raw, "assignment")
        key = (
            _text(record["layer"], "assignment.layer"),
            _text(record["targetId"], "assignment.targetId"),
            _text(record["kind"], "assignment.kind"),
        )
        if key in result:
            raise SurfaceManifestError("duplicate assignment")
        result[key] = _text(record["rule"], "assignment.rule")
    return result


def _partition_manifest(
    manifest: AlgMulManifest, partition: LegacyPartition
) -> AlgMulManifest:
    """Build one bounded typed ledger for exactly one disposition partition."""

    symbols = tuple(item for item in manifest.symbols if item.partition is partition)
    symbol_ids = {item.record_id for item in symbols}
    return AlgMulManifest.create(
        manifest_id=f"{manifest.manifest_id}:{partition.value}",
        source=manifest.source,
        symbols=symbols,
        definitions=tuple(
            item for item in manifest.definitions if item.symbol_id in symbol_ids
        ),
        factories=tuple(
            item for item in manifest.factories if item.partition is partition
        ),
        registries=tuple(
            item for item in manifest.registries if item.partition is partition
        ),
        behaviors=tuple(
            item for item in manifest.behaviors if item.partition is partition
        ),
        messages=(
            manifest.messages
            if partition is LegacyPartition.DISPLAY_REPORT_TABLES
            else ()
        ),
        dispositions=(),
    )


def _add_observations(
    observations: tuple[SymbolObservation, ...],
    *requested: tuple[EvidenceOrigin, EvidenceOutcome],
) -> tuple[SymbolObservation, ...]:
    """Add only missing origin states without concealing a contradiction."""

    known = {item.origin: item.outcome for item in observations}
    result = list(observations)
    for origin, outcome in requested:
        if origin in known:
            if known[origin] is not outcome:
                raise SurfaceManifestError(
                    f"contradictory {origin.value} property state"
                )
            continue
        known[origin] = outcome
        result.append(SymbolObservation.create(origin, outcome))
    return tuple(result)


def _property_composite_manifest(surface: Mapping[str, object]) -> AlgMulManifest:
    """Join only the three required MakeProperty evidence states.

    ``validate_complete`` intentionally requires static, evaluated, and
    intended property states in one object.  This narrow join is therefore
    allowed for that factory only; all other static and evaluated records stay
    in their independent bounded ledgers.
    """

    static = _partition_manifest(
        _static_manifest(surface), LegacyPartition.GENERATED_PROPERTY_API
    )
    runtime = _partition_manifest(
        _runtime_manifest(surface), LegacyPartition.GENERATED_PROPERTY_API
    )
    symbols: list[SymbolRecord] = []
    definitions = list(static.definitions)
    for item in static.symbols:
        observations = item.observations
        if item.name == "MakeProperty":
            observations = _add_observations(
                observations,
                (EvidenceOrigin.EVALUATED, EvidenceOutcome.ABSENT),
                (EvidenceOrigin.INTENDED, EvidenceOutcome.PRESENT),
            )
        elif item.name in EXPECTED_PROPERTY_NAMES:
            observations = _add_observations(
                observations, (EvidenceOrigin.EVALUATED, EvidenceOutcome.ABSENT)
            )
        copied = SymbolRecord.create(
            item.name,
            item.partition,
            observations,
            definition_state=item.definition_state,
            definition_kind=item.definition_kind,
            downvalue_count=item.downvalue_count,
            ownvalue_count=item.ownvalue_count,
            upvalue_count=item.upvalue_count,
            subvalue_count=item.subvalue_count,
            attributes=item.attributes,
            option_hashes=item.option_hashes,
            source_span=item.source_span,
            definition_hash=item.definition_hash,
            summary_origin=item.summary_origin,
            record_id=item.record_id,
        )
        symbols.append(copied)
        if copied.name in EXPECTED_PROPERTY_NAMES:
            definitions.append(
                AlgMulDefinitionRecord.create(
                    copied.record_id,
                    EvidenceOrigin.EVALUATED,
                    DefinitionState.ABSENT,
                    DefinitionKind.NONE,
                )
            )
    factories = tuple(
        FactoryRecord.create(
            item.name,
            _add_observations(
                item.observations,
                (EvidenceOrigin.EVALUATED, EvidenceOutcome.ABSENT),
                (EvidenceOrigin.INTENDED, EvidenceOutcome.PRESENT),
            )
            if item.name == "MakeProperty"
            else item.observations,
            requested_names=item.requested_names,
            partition=item.partition,
            record_id=item.record_id,
            call_kind=item.call_kind,
            call_relationships=item.call_relationships,
        )
        for item in static.factories
    )
    return AlgMulManifest.create(
        manifest_id="algmul-property-composite-v00-089",
        source=_runtime_manifest(surface).source,
        symbols=tuple(symbols),
        definitions=tuple(definitions),
        factories=factories,
        registries=(),
        behaviors=(*static.behaviors, *runtime.behaviors),
        messages=(),
        dispositions=(),
    )


def _runtime_kind(
    counts: tuple[int, int, int, int],
) -> tuple[DefinitionState, DefinitionKind]:
    active = sum(item > 0 for item in counts)
    if active == 0:
        return (DefinitionState.PRESENT_UNDEFINED, DefinitionKind.NONE)
    if active > 1:
        return (DefinitionState.PRESENT_DEFINED, DefinitionKind.MIXED)
    return (
        DefinitionState.PRESENT_DEFINED,
        (
            DefinitionKind.DOWN_VALUE,
            DefinitionKind.OWN_VALUE,
            DefinitionKind.UP_VALUE,
            DefinitionKind.SUB_VALUE,
        )[next(index for index, item in enumerate(counts) if item > 0)],
    )


def _runtime_manifest(surface: Mapping[str, object]) -> AlgMulManifest:
    provenances = _provenance(surface)
    runtime_path = _relative(provenances["runtime-load"]["path"], "runtime.path")
    runtime = _read(runtime_path, canonical=False)
    source = _object(runtime["source"], "runtime.source")
    kernel = _object(runtime["kernel"], "runtime.kernel")
    load = _object(runtime["load"], "runtime.load")
    runtime_symbols = _list(runtime["symbols"], "runtime.symbols")
    symbols: list[SymbolRecord] = []
    definitions: list[AlgMulDefinitionRecord] = []
    seen: set[str] = set()
    for raw in runtime_symbols:
        record = _object(raw, "runtime.symbol")
        name = _text(record["name"], "runtime.symbol.name").split("`")[-1]
        if name in seen:
            raise SurfaceManifestError(f"duplicate evaluated name {name}")
        seen.add(name)
        counts = tuple(
            _int(record[field], f"runtime.symbol.{field}")
            for field in (
                "downValueCount",
                "ownValueCount",
                "upValueCount",
                "subValueCount",
            )
        )
        state, kind = _runtime_kind(cast(tuple[int, int, int, int], counts))
        symbol = SymbolRecord.create(
            name,
            _partition(name, runtime_surface=True),
            (
                SymbolObservation.create(
                    EvidenceOrigin.EVALUATED, EvidenceOutcome.PRESENT
                ),
            ),
            definition_state=state,
            definition_kind=kind,
            downvalue_count=counts[0],
            ownvalue_count=counts[1],
            upvalue_count=counts[2],
            subvalue_count=counts[3],
            summary_origin=EvidenceOrigin.EVALUATED,
        )
        symbols.append(symbol)
        definitions.append(
            AlgMulDefinitionRecord.create(
                symbol.record_id,
                EvidenceOrigin.EVALUATED,
                state,
                kind,
                downvalue_count=counts[0],
                ownvalue_count=counts[1],
                upvalue_count=counts[2],
                subvalue_count=counts[3],
            )
        )
    for name in ("MakeProperty", *EXPECTED_PROPERTY_NAMES):
        if name in seen:
            raise SurfaceManifestError(
                f"missing-property record unexpectedly present: {name}"
            )
        symbols.append(
            SymbolRecord.create(
                name,
                LegacyPartition.GENERATED_PROPERTY_API,
                (
                    SymbolObservation.create(
                        EvidenceOrigin.EVALUATED, EvidenceOutcome.ABSENT
                    ),
                ),
            )
        )
    registries = tuple(
        RegistryRecord.create(
            _text(record["name"], "registry.name"),
            tuple(_list(record["basisEntries"], "registry.basisEntries")),
            _text(record["multiplicationValue"], "registry.multiplicationValue"),
            basis_table_snapshot=SemanticHash(
                "sha256", _text(record["basisHash"], "registry.basisHash").lower()
            ),
            basis_table_hash=SemanticHash(
                "sha256", _text(record["basisHash"], "registry.basisHash").lower()
            ),
            origin=EvidenceOrigin.EVALUATED,
        )
        for raw in _list(
            _object(runtime["registries"], "registries")["algebras"],
            "registries.algebras",
        )
        for record in (_object(raw, "registry"),)
    )
    behaviors_list: list[BehaviorRecord] = []
    for raw in _list(surface["runtimeBehaviors"], "runtimeBehaviors"):
        record = _object(raw, "runtimeBehavior")
        behavior_id = _text(record["id"], "runtimeBehavior.id")
        if behavior_id == "behavior:runtime-load":
            continue
        provenance_id = _text(record["provenanceId"], "runtimeBehavior.provenanceId")
        receipt = provenances.get(provenance_id)
        if receipt is None:
            raise SurfaceManifestError(f"unknown behavior provenance {provenance_id}")
        behaviors_list.append(
            BehaviorRecord.create(
                behavior_id.removeprefix("behavior:"),
                "Pinned V00-089 provenance receipt; reproduction evidence only.",
                partition=_text(record["partition"], "runtimeBehavior.partition"),
                record_id=behavior_id,
                input_snapshot=SemanticHash(
                    "sha256", _text(receipt["sha256"], "provenance.sha256").lower()
                ),
                origin=EvidenceOrigin.EVALUATED,
            )
        )
    runtime_behavior = BehaviorRecord.create(
        "runtime-load",
        (
            "Pinned V00-085 evaluated load and registry receipt; "
            "reproduction evidence only."
        ),
        partition=LegacyPartition.REGISTRY_BASES,
        record_id="behavior:runtime-load",
        input_snapshot=SemanticHash(
            "sha256",
            _text(provenances["runtime-load"]["sha256"], "runtime.sha256").lower(),
        ),
        origin=EvidenceOrigin.EVALUATED,
    )
    factory = FactoryRecord.create(
        "MakeProperty",
        (SymbolObservation.create(EvidenceOrigin.EVALUATED, EvidenceOutcome.ABSENT),),
        requested_names=EXPECTED_PROPERTY_NAMES,
        partition=LegacyPartition.GENERATED_PROPERTY_API,
        call_kind=FactoryCallKind.GENERATES,
        call_relationships=("runtime:missing-after-load",),
    )
    messages = tuple(
        MessageRecord.create(
            "load",
            "error",
            _text(item, "load.message"),
            origin=EvidenceOrigin.EVALUATED,
        )
        for item in _list(load["messages"], "load.messages")
    )
    return AlgMulManifest.create(
        manifest_id="algmul-evaluated-v00-089",
        source=SourceMetadata.create(
            _text(source["path"], "runtime.source.path"),
            SemanticHash(
                "sha256", _text(source["sha256"], "runtime.source.sha256").lower()
            ),
            kernel_version=_text(kernel["version"], "kernel.version"),
            source_bytes=_int(source["bytes"], "runtime.source.bytes"),
            contexts=("AlgMul`",),
            load_status=_text(load["status"], "load.status"),
            initial_context=_text(load["initialContext"], "load.initialContext"),
            context_after_load=_text(load["contextAfterLoad"], "load.contextAfterLoad"),
            context_path_after_load=tuple(
                _list(load["contextPathAfterLoad"], "load.contextPathAfterLoad")
            ),
            load_result_head=_text(load["resultHead"], "load.resultHead"),
        ),
        symbols=tuple(symbols),
        definitions=tuple(definitions),
        factories=(factory,),
        registries=registries,
        behaviors=(*behaviors_list, runtime_behavior),
        messages=messages,
        dispositions=(),
    )


def _static_manifest(surface: Mapping[str, object]) -> AlgMulManifest:
    source = _object(surface["source"], "surface.source")
    value = legacy_manifest_registry().from_record(
        json.loads(_STATIC_FIXTURE.read_text(encoding="utf-8"))
    )
    if type(value) is not AlgMulManifest:
        raise SurfaceManifestError("static fixture has an invalid type")
    manifest = value
    if manifest.source.content_hash.digest.upper() != _text(
        source["sha256"], "surface.source.sha256"
    ):
        raise SurfaceManifestError("pinned AlgMul source hash mismatch")
    if manifest.source.source_bytes != _int(source["bytes"], "surface.source.bytes"):
        raise SurfaceManifestError("pinned AlgMul source byte count mismatch")
    return manifest


def test_surface_manifest_is_canonical_pinned_and_covers_all_prior_captures() -> None:
    surface = _read(_SURFACE)
    dispositions = _read(_DISPOSITIONS)
    _validate_documents(surface, dispositions)
    source = _object(surface["source"], "surface.source")
    static = _static_manifest(surface)
    assert static.source.content_hash.digest.upper() == source["sha256"]
    assert static.source.source_bytes == source["bytes"]
    assert len(_provenance(surface)) == 4
    layer = _layer(surface, "static")
    assert (
        len(static.symbols),
        len(static.definitions),
        len(static.factories),
        len(static.behaviors),
    ) == (
        layer["symbols"],
        layer["definitions"],
        layer["factories"],
        layer["behaviors"],
    )


def test_scalar_rule_preserves_named_tables_without_promoting_generic_dispatch() -> (
    None
):
    """The documented exact-fixture promise applies only to named products."""

    scalar_rule = _CANONICAL_RULES[LegacyPartition.SCALAR_COMPOSITION_PRODUCTS]
    assert scalar_rule[0] == "preserve-exactly"
    assert scalar_rule[1] == "v0.0"
    assert scalar_rule[4] == ()
    assert scalar_rule[5] == "pending"
    assert "executable C/Cs tables" in scalar_rule[6]
    assert "Preserve exactly under named convention fixtures" in (
        _ROOT / "docs/legacy/algmul-parity.md"
    ).read_text(encoding="utf-8")
    assert {_partition(name) for name in ("Mul", "MulOld", "MulTable")} == {
        LegacyPartition.PRODUCT_OPERATORS
    }


def test_durable_inventory_exactly_matches_all_typed_layers() -> None:
    surface = _read(_SURFACE)
    dispositions = _read(_DISPOSITIONS)
    _validate_documents(surface, dispositions)
    actual = {
        "static": _typed_target_records(_static_manifest(surface)),
        "evaluated": _typed_target_records(_runtime_manifest(surface)),
        "property-composite": _typed_target_records(
            _property_composite_manifest(surface)
        ),
    }
    for layer_id, targets in actual.items():
        assert _inventory_targets(surface, layer_id) == targets
    inventory = _object(surface["targetInventory"], "targetInventory")
    counts = {
        _text(layer["id"], "targetInventory.layer.id"): _int(
            layer["targetCount"], "targetInventory.targetCount"
        )
        for raw_layer in _list(inventory["layers"], "targetInventory.layers")
        for layer in (_object(raw_layer, "targetInventory.layer"),)
    }
    assert counts == {"static": 1636, "evaluated": 1095, "property-composite": 269}
    assert (
        len(_list(dispositions["assignments"], "assignments"))
        == sum(counts.values())
        == 3000
    )


def test_incomplete_partition_is_limited_to_reviewed_stubs_and_receipts() -> None:
    """Evaluated implementation locals are not silently mislabeled as stubs."""

    surface = _read(_SURFACE)
    incomplete = LegacyPartition.INCOMPLETE_STUBS_COMMENTS.value
    static_targets = {
        target
        for target in _typed_target_records(_static_manifest(surface))
        if target[2] == incomplete
    }
    assert static_targets == {
        ("symbol:GenMomenta", "symbol", incomplete, "static-or-intended"),
        ("symbol:GenMomentum", "symbol", incomplete, "static-or-intended"),
        ("symbol:SYMtoTLIST", "symbol", incomplete, "static-or-intended"),
        ("definition:symbol:GenMomenta:static:1", "definition", incomplete, "static"),
        ("definition:symbol:GenMomentum:static:1", "definition", incomplete, "static"),
        ("definition:symbol:SYMtoTLIST:static:1", "definition", incomplete, "static"),
        (
            "behavior:defect-comment:1:249:249:82d9fc7b0d3dbfbb",
            "behavior",
            incomplete,
            "static",
        ),
        (
            "behavior:defect-comment:2:264:264:c61fe5eff0f7a8f4",
            "behavior",
            incomplete,
            "static",
        ),
        (
            "behavior:defect-comment:3:862:862:45c326f39da7f093",
            "behavior",
            incomplete,
            "static",
        ),
        (
            "behavior:defect-comment:4:935:935:4c926707f5a89755",
            "behavior",
            incomplete,
            "static",
        ),
        (
            "behavior:defect-comment:5:1921:1921:c5a8288e4bb5e307",
            "behavior",
            incomplete,
            "static",
        ),
        (
            "behavior:defect-comment:6:2349:2349:4f071a502774cede",
            "behavior",
            incomplete,
            "static",
        ),
        (
            "behavior:defect-comment:7:2840:2840:31aea2d3a80f8320",
            "behavior",
            incomplete,
            "static",
        ),
        (
            "behavior:defect-comment:8:3821:3821:ba1d81348fba2ea3",
            "behavior",
            incomplete,
            "static",
        ),
        (
            "behavior:static-expression:MakeTChars:none:28:84:84:5a6976d6a43bc181",
            "behavior",
            incomplete,
            "static",
        ),
    }
    evaluated_targets = {
        target
        for target in _typed_target_records(_runtime_manifest(surface))
        if target[2] == incomplete
    }
    assert evaluated_targets == {
        ("symbol:SYMtoTLIST", "symbol", incomplete, "evaluated"),
        (
            "definition:symbol:SYMtoTLIST:evaluated",
            "definition",
            incomplete,
            "evaluated",
        ),
    }


def test_every_static_and_evaluated_target_receives_one_typed_disposition() -> None:
    surface = _read(_SURFACE)
    disposition_payload = _read(_DISPOSITIONS)
    rules = _validate_documents(surface, disposition_payload)
    assignments = _assignment_map(disposition_payload)
    for layer_id, manifest in (
        ("static", _static_manifest(surface)),
        ("evaluated", _runtime_manifest(surface)),
    ):
        all_targets = _record_targets(manifest)
        asserted: set[str] = set()
        for partition in LegacyPartition:
            if partition is LegacyPartition.GENERATED_PROPERTY_API:
                partitioned = _property_composite_manifest(surface)
                assignment_layer = "property-composite"
            else:
                partitioned = _partition_manifest(manifest, partition)
                assignment_layer = layer_id
            completed = _with_dispositions(
                partitioned,
                rules,
                assignments,
                layer_id=assignment_layer,
                owned_partitions=(partition,),
            )
            expected = {
                target_id
                for target_id, target_partition in all_targets
                if target_partition is partition
            }
            assigned = {item.target_id for item in completed.dispositions}
            assert expected <= assigned
            completed.validate_complete(owned_partitions=(partition,))
            assert canonical_manifest_bytes(completed)
            asserted.update(
                target_id for target_id in expected if target_id in assigned
            )
        assert asserted == {target_id for target_id, _ in all_targets}


def test_evaluated_layer_records_runtime_evidence_and_missing_property_surface() -> (
    None
):
    surface = _read(_SURFACE)
    manifest = _runtime_manifest(surface)
    evaluated = _layer(surface, "evaluated")
    assert len(manifest.symbols) == _int(
        evaluated["symbols"], "evaluated.symbols"
    ) + 1 + len(EXPECTED_PROPERTY_NAMES)
    assert len(manifest.definitions) == _int(evaluated["symbols"], "evaluated.symbols")
    assert len(manifest.registries) == _int(
        evaluated["registries"], "evaluated.registries"
    )
    assert len(manifest.messages) == _int(evaluated["messages"], "evaluated.messages")
    assert {item.name for item in manifest.registries} == set(
        _text(item, "evaluated.registryNames")
        for item in _list(evaluated["registryNames"], "evaluated.registryNames")
    )
    assert any("Syntax::sntx" in item.text for item in manifest.messages)
    message_targets = {
        target_id
        for target_id, kind, partition, origin in _typed_target_records(manifest)
        if kind == "message"
        and partition == LegacyPartition.DISPLAY_REPORT_TABLES.value
        and origin == EvidenceOrigin.EVALUATED.value
    }
    assert message_targets == {_message_target_id(manifest.messages[0])}
    assert hashlib.sha256(
        manifest.messages[0].text.encode()
    ).hexdigest().upper() == _text(
        evaluated["messageSHA256"], "evaluated.messageSHA256"
    )
    runtime = _read(
        _relative(_provenance(surface)["runtime-load"]["path"], "runtime.path"),
        canonical=False,
    )
    raw_registries = _list(
        _object(runtime["registries"], "registries")["algebras"],
        "registries.algebras",
    )
    expected_registries = {
        _text(record["name"], "registry.name"): record
        for raw in raw_registries
        for record in (_object(raw, "registry"),)
    }
    for registry in manifest.registries:
        raw = expected_registries[registry.name]
        assert registry.basis == tuple(
            _list(raw["basisEntries"], "registry.basisEntries")
        )
        assert registry.multiplication_symbol == raw["multiplicationValue"]
        assert registry.basis_table_hash is not None
        assert registry.basis_table_hash.digest.upper() == raw["basisHash"]
        assert raw["multiplicationRoute"] == f'AlgMul`alg["Mul"]["{registry.name}"]'
        assert raw["structureDotRoute"] == f'AlgMul`alg["StructDot"]["{registry.name}"]'
        assert raw["structureDotAvailable"] is False
        assert raw["structureDotHash"] is None
    absent = {
        item.name
        for item in manifest.symbols
        if item.observations[0].outcome is EvidenceOutcome.ABSENT
    }
    assert absent == {"MakeProperty", *EXPECTED_PROPERTY_NAMES}


def test_pinned_runtime_names_have_exact_reviewed_classifier_membership() -> None:
    """Runtime locals are enumerated receipts, while declared bases stay public."""

    surface = _read(_SURFACE)
    runtime = _read(
        _relative(_provenance(surface)["runtime-load"]["path"], "runtime.path"),
        canonical=False,
    )
    runtime_names = {
        _text(_object(raw, "runtime.symbol")["name"], "runtime.symbol.name").split("`")[
            -1
        ]
        for raw in _list(runtime["symbols"], "runtime.symbols")
    }
    basis_names = {
        "e1",
        "e2",
        "e3",
        "e4",
        "e5",
        "e6",
        "e7",
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "E6",
        "E7",
        "f1",
        "f2",
        "f3",
        "f4",
        "f5",
        "f6",
        "f7",
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
        "F7",
        "g1",
        "g2",
        "g3",
        "g4",
        "g5",
        "g6",
        "g7",
        "G1",
        "G2",
        "G3",
        "G4",
        "G5",
        "G6",
        "G7",
        "ii",
        "if1",
        "if2",
        "if3",
        "\u00cf\u0095",
    }
    assert len(_RUNTIME_LEAK_NAMES) == 147
    assert runtime_names >= _RUNTIME_LEAK_NAMES
    assert _RUNTIME_LEAK_NAMES.isdisjoint(basis_names)
    assert basis_names <= runtime_names
    assert basis_names <= _REGISTRY_BASE_NAMES
    assert {
        name
        for name in runtime_names
        if _partition(name, runtime_surface=True)
        is LegacyPartition.GLOBAL_ALIASES_OPERATORS
    } == _RUNTIME_LEAK_NAMES | (_GLOBAL_OPERATOR_NAMES & runtime_names)
    assert all(_partition(name, runtime_surface=True) for name in runtime_names)
    assert _partition("e1", runtime_surface=True) is LegacyPartition.REGISTRY_BASES
    assert _partition("G7", runtime_surface=True) is LegacyPartition.REGISTRY_BASES
    assert _partition("if1", runtime_surface=True) is LegacyPartition.REGISTRY_BASES
    assert (
        _partition("alg1$", runtime_surface=True)
        is LegacyPartition.GLOBAL_ALIASES_OPERATORS
    )
    assert (
        _partition("Z2", runtime_surface=True)
        is LegacyPartition.GLOBAL_ALIASES_OPERATORS
    )
    with pytest.raises(StaticCaptureError, match="unreviewed evaluated"):
        _partition("fabricatedRuntimeName", runtime_surface=True)


def test_property_composite_is_only_the_required_generated_property_ledger() -> None:
    surface = _read(_SURFACE)
    composite = _property_composite_manifest(surface)
    symbols = {item.name: item for item in composite.symbols}
    assert set(symbols) == {"MakeProperty", *EXPECTED_PROPERTY_NAMES}
    assert len(_typed_target_records(composite)) == 269
    for name in EXPECTED_PROPERTY_NAMES:
        assert {
            (observation.origin, observation.outcome)
            for observation in symbols[name].observations
        } == {
            (EvidenceOrigin.EVALUATED, EvidenceOutcome.ABSENT),
            (EvidenceOrigin.INTENDED, EvidenceOutcome.PRESENT),
        }
    make_property = symbols["MakeProperty"]
    assert {
        (observation.origin, observation.outcome)
        for observation in make_property.observations
    } == {
        (EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT),
        (EvidenceOrigin.EVALUATED, EvidenceOutcome.ABSENT),
        (EvidenceOrigin.INTENDED, EvidenceOutcome.PRESENT),
    }
    assert {
        item.symbol_id
        for item in composite.definitions
        if item.origin is EvidenceOrigin.EVALUATED
        and item.definition_state is DefinitionState.ABSENT
    } == {symbols[name].record_id for name in EXPECTED_PROPERTY_NAMES}
    assert {item.name for item in composite.factories} == {"MakeProperty"}
    behavior_ids = {item.record_id for item in composite.behaviors}
    assert "behavior:static-callsite:MakeProperty" in behavior_ids
    assert "behavior:makeproperty-intent" in behavior_ids
    assert not {
        "Alternative",
        "Comm",
        "JAlternative",
        "MAlternative",
        "PrintProperty",
    } & set(symbols)


def test_missing_or_tampered_rule_or_manifest_evidence_fails_closed() -> None:
    surface = _read(_SURFACE)
    dispositions = _read(_DISPOSITIONS)
    _validate_documents(surface, dispositions)
    damaged = copy.deepcopy(dispositions)
    rules = cast(list[object], damaged["rules"])
    rules.pop()
    with pytest.raises(SurfaceManifestError, match="unclassified"):
        _rules(damaged)
    bad_source = copy.deepcopy(surface)
    _object(bad_source["source"], "source")["sha256"] = "0" * 64
    with pytest.raises(SurfaceManifestError, match="source hash"):
        _static_manifest(bad_source)
    bad_binding = copy.deepcopy(dispositions)
    _object(bad_binding["surfaceManifest"], "surfaceManifest")["sha256"] = "0" * 64
    with pytest.raises(SurfaceManifestError, match="surfaceManifest hash"):
        _validate_documents(surface, bad_binding)
    bad_oracle_binding = copy.deepcopy(dispositions)
    _object(bad_oracle_binding["oracleEvidenceRegistry"], "oracleEvidenceRegistry")[
        "sha256"
    ] = "0" * 64
    with pytest.raises(SurfaceManifestError, match="oracleEvidenceRegistry hash"):
        _validate_documents(surface, bad_oracle_binding)
    completed = _with_dispositions(
        _static_manifest(surface),
        _rules(dispositions),
        _assignment_map(dispositions),
        layer_id="static",
        owned_partitions=(LegacyPartition.REGISTRY_BASES,),
    )
    object.__setattr__(completed.dispositions[0], "owner", "tampered")
    with pytest.raises(LegacyManifestError, match="semantic_hash"):
        canonical_manifest_bytes(completed)


def test_oracle_registry_is_external_hash_bound_and_exactly_referenced() -> None:
    dispositions = _read(_DISPOSITIONS)
    rules = _rules(dispositions)
    payload = _read(_ORACLE_EVIDENCE)
    records = _validate_oracle_evidence(rules, payload)
    assert len(records) == 10
    assert set(records) == {
        oracle_id
        for rule in rules.values()
        for oracle_id in _list(rule["oracleIds"], "rule.oracleIds")
    }
    assert all(
        _text(record["path"], "oracleEvidence.path")
        != "tests/legacy/test_disposition_complete.py"
        for record in records.values()
    )

    escaped = copy.deepcopy(payload)
    _object(_list(escaped["entries"], "entries")[0], "entry")["path"] = "../escape.py"
    with pytest.raises(SurfaceManifestError, match="escapes the repository"):
        _validate_oracle_evidence(rules, escaped)

    bad_hash = copy.deepcopy(payload)
    _object(_list(bad_hash["entries"], "entries")[0], "entry")["sha256"] = "0" * 64
    with pytest.raises(SurfaceManifestError, match="hash mismatch"):
        _validate_oracle_evidence(rules, bad_hash)

    bad_node = copy.deepcopy(payload)
    first = _object(_list(bad_node["entries"], "entries")[0], "entry")
    binding = _object(first["binding"], "binding")
    _list(binding["pytestNodes"], "pytestNodes")[0] = (
        f"{first['path']}::test_fabricated_oracle"
    )
    with pytest.raises(SurfaceManifestError, match="not source-bound"):
        _validate_oracle_evidence(rules, bad_node)

    unreferenced = copy.deepcopy(payload)
    fabricated = copy.deepcopy(_list(unreferenced["entries"], "entries")[0])
    _object(fabricated, "entry")["id"] = "oracle.fabricated"
    _list(unreferenced["entries"], "entries").append(fabricated)
    with pytest.raises(SurfaceManifestError, match="unreferenced oracle evidence"):
        _validate_oracle_evidence(rules, unreferenced)

    missing = copy.deepcopy(payload)
    _list(missing["entries"], "entries").pop()
    with pytest.raises(SurfaceManifestError, match="unbacked oracle evidence link"):
        _validate_oracle_evidence(rules, missing)

    self_citing = copy.deepcopy(payload)
    self_record = _object(_list(self_citing["entries"], "entries")[0], "entry")
    self_record["path"] = "tests/legacy/test_disposition_complete.py"
    self_record["sha256"] = _sha256(Path(__file__).resolve())
    self_record["binding"] = {
        "pytestNodes": [
            (
                "tests/legacy/test_disposition_complete.py::"
                "test_oracle_registry_is_external_hash_bound_and_exactly_referenced"
            )
        ]
    }
    with pytest.raises(SurfaceManifestError, match="own validator"):
        _validate_oracle_evidence(rules, self_citing)


def _inventory_layer(
    surface: Mapping[str, object], index: int = 0
) -> dict[str, object]:
    inventory = _object(surface["targetInventory"], "targetInventory")
    return _object(
        _list(inventory["layers"], "targetInventory.layers")[index],
        "targetInventory.layer",
    )


def _refresh_target_hash(surface: Mapping[str, object], index: int = 0) -> None:
    layer = _inventory_layer(surface, index)
    records = _list(layer["records"], "targetInventory.records")
    layer["targetSetSHA256"] = (
        hashlib.sha256(
            json.dumps(
                records, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        )
        .hexdigest()
        .upper()
    )


def _fabricate_inventory_origin(
    surface: dict[str, object], dispositions: dict[str, object]
) -> None:
    del dispositions
    records = _list(_inventory_layer(surface)["records"], "targetInventory.records")
    _object(records[0], "targetInventory.record")["origin"] = "fabricated"
    _refresh_target_hash(surface)


def _duplicate_inventory_target(
    surface: dict[str, object], dispositions: dict[str, object]
) -> None:
    del dispositions
    layer = _inventory_layer(surface)
    records = _list(layer["records"], "targetInventory.records")
    records.insert(0, copy.deepcopy(records[0]))
    layer["targetCount"] = _int(layer["targetCount"], "targetInventory.targetCount") + 1
    _refresh_target_hash(surface)


def _reorder_inventory_records(
    surface: dict[str, object], dispositions: dict[str, object]
) -> None:
    del dispositions
    records = _list(_inventory_layer(surface)["records"], "targetInventory.records")
    records[0], records[1] = records[1], records[0]
    _refresh_target_hash(surface)


def _reorder_assignments(
    surface: dict[str, object], dispositions: dict[str, object]
) -> None:
    del surface
    assignments = _list(dispositions["assignments"], "assignments")
    assignments[0], assignments[1] = assignments[1], assignments[0]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda surface, dispositions: dispositions["assignments"].pop(),
            "assignment target",
        ),
        (
            lambda surface, dispositions: dispositions["assignments"].append(
                copy.deepcopy(dispositions["assignments"][0])
            ),
            "duplicate assignment",
        ),
        (
            lambda surface, dispositions: dispositions["assignments"][0].__setitem__(
                "layer", "unknown"
            ),
            "assignment target",
        ),
        (
            lambda surface, dispositions: dispositions["assignments"][0].__setitem__(
                "kind", "unknown"
            ),
            "assignment target",
        ),
        (
            lambda surface, dispositions: dispositions["rules"][0].__setitem__(
                "partition", "unknown"
            ),
            "unknown enum",
        ),
        (
            lambda surface, dispositions: dispositions["rules"][0].__setitem__(
                "apiIds", ["api.unknown"]
            ),
            "unknown API",
        ),
        (
            lambda surface, dispositions: dispositions["rules"][0].__setitem__(
                "testIds", ["test.unknown"]
            ),
            "unknown test",
        ),
        (
            lambda surface, dispositions: dispositions["rules"][0].__setitem__(
                "owner", "v9.9"
            ),
            "later owner must be an explicit defer",
        ),
        (
            lambda surface, dispositions: dispositions["rules"][0].__setitem__(
                "disposition", "preserve-concept"
            ),
            "canonical rule contract drift",
        ),
        (
            lambda surface, dispositions: dispositions["rules"][0].__setitem__(
                "apiIds", ["api.structure.build", "api.legacy.validate_complete"]
            ),
            "canonical rule contract drift",
        ),
        (
            lambda surface, dispositions: dispositions["rules"][1].__setitem__(
                "oracleIds", ["oracle.fabricated"]
            ),
            "unbacked oracle evidence link",
        ),
        (
            lambda surface, dispositions: dispositions["rules"][1].__setitem__(
                "oracleIds", ["oracle.scalar.table-fixture"]
            ),
            "unbacked oracle evidence link",
        ),
        (
            lambda surface, dispositions: dispositions["rules"][7].__setitem__(
                "owner", "someday"
            ),
            "canonical rule contract drift",
        ),
        (
            lambda surface, dispositions: surface["provenance"][0].__setitem__(
                "path", "../escape.json"
            ),
            "escapes the repository",
        ),
        (
            lambda surface, dispositions: surface["provenance"][0].__setitem__(
                "unvalidated", True
            ),
            "provenance has an unexpected shape",
        ),
        (
            lambda surface, dispositions: surface["limitations"].__setitem__(
                0, "fabricated limitation"
            ),
            "surface limitations drift",
        ),
        (
            lambda surface, dispositions: surface["layers"][1].__setitem__(
                "kernel", "tampered"
            ),
            "layer evidence",
        ),
        (
            lambda surface, dispositions: surface["targetInventory"]["layers"][0][
                "records"
            ][0].__setitem__("partition", "matrices"),
            "target set hash",
        ),
        (_fabricate_inventory_origin, "unknown target origin"),
        (_duplicate_inventory_target, "duplicate layer target"),
        (_reorder_inventory_records, "deterministically ordered"),
        (_reorder_assignments, "deterministically ordered"),
        (
            lambda surface, dispositions: dispositions.__setitem__("unvalidated", True),
            "dispositions have an unexpected shape",
        ),
    ],
)
def test_durable_ledger_rejects_assignment_and_evidence_tampering(
    mutator: object, message: str
) -> None:
    surface = _read(_SURFACE)
    dispositions = _read(_DISPOSITIONS)
    mutated_surface = copy.deepcopy(surface)
    mutated_dispositions = copy.deepcopy(dispositions)
    assert callable(mutator)
    mutator(mutated_surface, mutated_dispositions)
    with pytest.raises(SurfaceManifestError, match=message):
        _validate_documents(mutated_surface, mutated_dispositions)


def test_empty_partition_is_a_bounded_validated_ledger() -> None:
    surface = _read(_SURFACE)
    dispositions = _read(_DISPOSITIONS)
    rules = _validate_documents(surface, dispositions)
    static_jordan = _partition_manifest(
        _static_manifest(surface), LegacyPartition.JORDAN_MATRICES
    )
    assert _record_targets(static_jordan)
    jordan_rule = rules[LegacyPartition.JORDAN_MATRICES]
    assert jordan_rule["owner"] == "v0.2"
    assert jordan_rule["disposition"] == Disposition.DEFER.value
    empty = _partition_manifest(
        _property_composite_manifest(surface), LegacyPartition.JORDAN_MATRICES
    )
    assert not _record_targets(empty)
    completed = _with_dispositions(
        empty,
        rules,
        _assignment_map(dispositions),
        layer_id="property-composite",
        owned_partitions=(LegacyPartition.JORDAN_MATRICES,),
    )
    completed.validate_complete(owned_partitions=(LegacyPartition.JORDAN_MATRICES,))
