"""Closed, immutable records for non-oracular AlgMul parity evidence.

These records describe what a pinned legacy capture observed; they do not claim
that a legacy result is mathematically correct.  The only decoder is the
allow-listed :class:`~anyalgebra.persistence.registry.SerializerRegistry`
below.  Input data therefore cannot name imports, execute callables, or create
arbitrary objects.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import TypeVar, cast

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import SemanticHash
from anyalgebra.persistence.canonical import canonical_json
from anyalgebra.persistence.registry import JSONValue, SerializerRegistry


_TAG = "anyalgebra.legacy.algmul_manifest"
_VERSION = 1
_MAX_ITEMS = 2_048
_PAGE_ITEMS = 240
_MAX_TEXT = 4_096
_MAX_COMPACT_RECORD = 65_536

PROPERTY_PREFIXES = (
    "A",
    "AT",
    "AM",
    "AMT",
    "IsA",
    "IsAT",
    "IsAM",
    "IsAMT",
    "N",
    "NT",
    "NM",
    "NMT",
    "IsN",
    "IsNT",
    "IsNM",
    "IsNMT",
)
PROPERTY_FAMILIES = (
    "Commutative",
    "Associative",
    "Alternative",
    "Flexible",
    "PowerAssociative1",
    "PowerAssociative2",
    "JIdentity",
    "Jacobi",
)
EXPECTED_PROPERTY_NAMES = tuple(
    prefix + family for prefix in PROPERTY_PREFIXES for family in PROPERTY_FAMILIES
)


class LegacyManifestError(AnyAlgebraError, ValueError):
    """A value violates the closed legacy-manifest evidence contract."""

    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid legacy manifest {field}: {reason}")


class LegacyPartition(StrEnum):
    """The durable partitions from the AlgMul capability catalogue."""

    REGISTRY_BASES = "registry_bases"
    ARBITRARY_STRUCTURES = "arbitrary_structures"
    SCALAR_COMPOSITION_PRODUCTS = "scalar_composition_products"
    SYMBOLIC_LIST_CONVERSION = "symbolic_list_conversion"
    ARBITRARY_ARRAYS = "arbitrary_arrays"
    TENSOR_PRODUCTS = "tensor_products"
    MATRICES = "matrices"
    JORDAN_MATRICES = "jordan_matrices"
    PRODUCT_OPERATORS = "product_operators"
    PROPERTY_KERNELS = "property_kernels"
    PROPERTY_WRAPPERS = "property_wrappers"
    GENERATED_PROPERTY_API = "generated_property_api"
    GENERIC_ELEMENTS = "generic_elements"
    SPAN_STRUCTURE_RECOVERY = "span_structure_recovery"
    CONJUGATIONS_NORMS = "conjugations_norms"
    CLIFFORD_GEOMETRIC_GRASSMANN = "clifford_geometric_grassmann"
    LIE_GROUP_EXPERIMENTS = "lie_group_experiments"
    DISPLAY_REPORT_TABLES = "display_report_tables"
    GLOBAL_ALIASES_OPERATORS = "global_aliases_operators"
    INCOMPLETE_STUBS_COMMENTS = "incomplete_stubs_comments"


class EvidenceOrigin(StrEnum):
    """The surface on which an observation was made or requested."""

    STATIC = "static"
    EVALUATED = "evaluated"
    INTENDED = "intended"


class EvidenceOutcome(StrEnum):
    """Whether that origin contains the observed item."""

    PRESENT = "present"
    ABSENT = "absent"


class Disposition(StrEnum):
    """The v0.0 treatment of a legacy partition, never a correctness claim."""

    PRESERVE_EXACTLY = "preserve-exactly"
    PRESERVE_CONCEPT = "preserve-concept"
    REDESIGN = "redesign"
    REPLACE = "replace"
    DEFER = "defer"
    DROP = "drop"


ParityDisposition = Disposition


class DefinitionState(StrEnum):
    """Static definition status, separate from runtime symbol presence."""

    PRESENT_DEFINED = "present-defined"
    PRESENT_UNDEFINED = "present-undefined"
    ABSENT = "absent"


class DefinitionKind(StrEnum):
    """The structural Wolfram definition form recorded for a symbol."""

    NONE = "none"
    UNKNOWN = "unknown"
    OWN_VALUE = "own-value"
    DOWN_VALUE = "down-value"
    UP_VALUE = "up-value"
    SUB_VALUE = "sub-value"
    MIXED = "mixed"


class LoadStatus(StrEnum):
    """Whether the pinned source was observed loading successfully."""

    UNKNOWN = "unknown"
    STATIC_ONLY = "static-only"
    LOADED = "loaded"
    FAILED = "failed"
    COMPLETED_WITH_MISSING_REQUIRED_SYMBOLS = "completed-with-missing-required-symbols"


class FactoryCallKind(StrEnum):
    """A bounded relationship between a factory and another captured surface."""

    INVOKES = "invokes"
    GENERATES = "generates"


class SymbolAttribute(StrEnum):
    """Supported semantic Wolfram attributes, never arbitrary attribute text."""

    FLAT = "flat"
    LISTABLE = "listable"
    ONE_IDENTITY = "one-identity"
    ORDERLESS = "orderless"
    PROTECTED = "protected"
    TEMPORARY = "temporary"


class OracleStatus(StrEnum):
    """Whether an independently reproducible oracle link is available."""

    AVAILABLE = "available"
    PENDING = "pending"


def _text(value: object, field: str, *, identifier: bool = False) -> str:
    if type(value) is not str or not value:
        raise LegacyManifestError(field, "must be a nonempty exact built-in str")
    if len(value) > _MAX_TEXT:
        raise LegacyManifestError(field, "text limit exceeded")
    if any("\ud800" <= character <= "\udfff" for character in value):
        raise LegacyManifestError(field, "contains an invalid Unicode scalar")
    if identifier and value != value.strip():
        raise LegacyManifestError(field, "must not have leading or trailing whitespace")
    return value


def _hash(value: object, field: str) -> SemanticHash:
    if type(value) is not SemanticHash:
        raise LegacyManifestError(field, "must be an exact SemanticHash")
    try:
        return SemanticHash(value.algorithm, value.digest)
    except Exception as error:
        raise LegacyManifestError(field, "has an invalid SemanticHash") from error


def _enum(value: object, enum_type: type[Enum], field: str) -> Enum:
    if type(value) is enum_type:
        return value
    if type(value) is not str:
        raise LegacyManifestError(field, "must be an exact enum value")
    try:
        return enum_type(value)
    except ValueError as error:
        raise LegacyManifestError(field, "has an unknown enum value") from error


def _tuple(value: object, field: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise LegacyManifestError(field, "must be an exact tuple")
    if len(value) > _MAX_ITEMS:
        raise LegacyManifestError(field, "item limit exceeded")
    return value


def _names(value: object, field: str, *, expected: bool = False) -> tuple[str, ...]:
    names = tuple(_text(item, field, identifier=True) for item in _tuple(value, field))
    if len(set(names)) != len(names):
        raise LegacyManifestError(field, "contains duplicate names")
    if expected and names != EXPECTED_PROPERTY_NAMES:
        raise LegacyManifestError(field, "must be the exact intended 128-name surface")
    return names


def _records(value: object, item_type: type[object], field: str) -> tuple[object, ...]:
    items = _tuple(value, field)
    if any(type(item) is not item_type for item in items):
        raise LegacyManifestError(
            field, f"must contain exact {item_type.__name__} values"
        )
    return items


def _mapping(value: object, keys: frozenset[str], field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise LegacyManifestError(field, "has an invalid record shape")
    return value


def _hash_record(value: SemanticHash) -> dict[str, str]:
    return {"algorithm": value.algorithm, "digest": value.digest}


def _parse_hash(value: object, field: str) -> SemanticHash:
    record = _mapping(value, frozenset(("algorithm", "digest")), field)
    try:
        return SemanticHash(
            _text(record["algorithm"], field, identifier=True),
            _text(record["digest"], field, identifier=True),
        )
    except Exception as error:
        raise LegacyManifestError(field, "has an invalid semantic hash") from error


@dataclass(frozen=True, slots=True, init=False, repr=False)
class SourceMetadata:
    """Pinned source identity and optional evaluated-kernel metadata."""

    location: str
    content_hash: SemanticHash
    kernel_version: str | None
    source_bytes: int
    contexts: tuple[str, ...]
    load_status: LoadStatus
    initial_context: str | None
    context_after_load: str | None
    context_path_after_load: tuple[str, ...]
    load_result_head: str | None

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise LegacyManifestError("source", "is factory-owned")

    @classmethod
    def create(
        cls,
        location: object,
        content_hash: object,
        *,
        kernel_version: object = None,
        source_bytes: object = 0,
        contexts: object = (),
        load_status: object = LoadStatus.UNKNOWN,
        initial_context: object = None,
        context_after_load: object = None,
        context_path_after_load: object = (),
        load_result_head: object = None,
    ) -> SourceMetadata:
        value = object.__new__(SourceMetadata)
        object.__setattr__(value, "location", _text(location, "source.location"))
        object.__setattr__(value, "content_hash", _hash(content_hash, "source.hash"))
        object.__setattr__(
            value,
            "kernel_version",
            None
            if kernel_version is None
            else _text(kernel_version, "source.kernel_version"),
        )
        object.__setattr__(
            value,
            "initial_context",
            None
            if initial_context is None
            else _text(initial_context, "source.initial_context", identifier=True),
        )
        object.__setattr__(
            value,
            "context_after_load",
            None
            if context_after_load is None
            else _text(
                context_after_load, "source.context_after_load", identifier=True
            ),
        )
        object.__setattr__(
            value,
            "context_path_after_load",
            _names(context_path_after_load, "source.context_path_after_load"),
        )
        object.__setattr__(
            value,
            "load_result_head",
            None
            if load_result_head is None
            else _text(load_result_head, "source.load_result_head", identifier=True),
        )
        if type(source_bytes) is not int or not 0 <= source_bytes <= 100_000_000:
            raise LegacyManifestError(
                "source.source_bytes", "must be a bounded exact int"
            )
        object.__setattr__(value, "source_bytes", source_bytes)
        object.__setattr__(value, "contexts", _names(contexts, "source.contexts"))
        checked_status = cast(
            LoadStatus, _enum(load_status, LoadStatus, "source.load_status")
        )
        runtime_statuses = {
            LoadStatus.LOADED,
            LoadStatus.FAILED,
            LoadStatus.COMPLETED_WITH_MISSING_REQUIRED_SYMBOLS,
        }
        if checked_status in (LoadStatus.UNKNOWN, LoadStatus.STATIC_ONLY) and (
            value.kernel_version is not None
            or value.initial_context is not None
            or value.context_after_load is not None
            or value.context_path_after_load
            or value.load_result_head is not None
        ):
            raise LegacyManifestError(
                "source.load_status", "non-runtime source cannot carry runtime metadata"
            )
        if checked_status in runtime_statuses and (
            value.kernel_version is None
            or value.initial_context is None
            or value.context_after_load is None
            or not value.context_path_after_load
            or value.load_result_head is None
        ):
            raise LegacyManifestError(
                "source.load_status",
                "runtime source requires complete runtime metadata",
            )
        object.__setattr__(value, "load_status", checked_status)
        return value

    def __repr__(self) -> str:
        return "SourceMetadata()"

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("SourceMetadata cannot be subclassed")


@dataclass(frozen=True, slots=True, init=False, repr=False)
class SourceSpan:
    """One bounded, inclusive source line span for a static definition."""

    start_line: int
    end_line: int

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise LegacyManifestError("source_span", "is factory-owned")

    @classmethod
    def create(cls, start_line: object, end_line: object) -> SourceSpan:
        if (
            type(start_line) is not int
            or type(end_line) is not int
            or not 1 <= start_line <= end_line <= 1_000_000
        ):
            raise LegacyManifestError("source_span", "must be finite and ordered")
        value = object.__new__(SourceSpan)
        object.__setattr__(value, "start_line", start_line)
        object.__setattr__(value, "end_line", end_line)
        return value

    def __repr__(self) -> str:
        return "SourceSpan()"

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("SourceSpan cannot be subclassed")


@dataclass(frozen=True, slots=True, init=False, repr=False)
class SymbolObservation:
    """One present/absent observation at static, evaluated, or intended scope."""

    origin: EvidenceOrigin
    outcome: EvidenceOutcome

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise LegacyManifestError("observation", "is factory-owned")

    @classmethod
    def create(cls, origin: object, outcome: object) -> SymbolObservation:
        value = object.__new__(SymbolObservation)
        object.__setattr__(
            value,
            "origin",
            cast(EvidenceOrigin, _enum(origin, EvidenceOrigin, "origin")),
        )
        object.__setattr__(
            value,
            "outcome",
            cast(EvidenceOutcome, _enum(outcome, EvidenceOutcome, "outcome")),
        )
        return value

    def __repr__(self) -> str:
        return "SymbolObservation()"

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("SymbolObservation cannot be subclassed")


def _observations(value: object, field: str) -> tuple[SymbolObservation, ...]:
    observations = cast(
        tuple[SymbolObservation, ...], _records(value, SymbolObservation, field)
    )
    if not observations:
        raise LegacyManifestError(field, "must not be empty")
    pairs = tuple((item.origin, item.outcome) for item in observations)
    if len(set(pairs)) != len(pairs):
        raise LegacyManifestError(field, "contains duplicate observations")
    if len({item.origin for item in observations}) != len(observations):
        raise LegacyManifestError(field, "contains contradictory origin outcomes")
    return tuple(
        sorted(observations, key=lambda item: (item.origin.value, item.outcome.value))
    )


def _definition_shape(
    state: DefinitionState,
    kind: DefinitionKind,
    counts: tuple[int, int, int, int],
    attributes: tuple[SymbolAttribute, ...],
    option_hashes: tuple[SemanticHash, ...],
    source_span: SourceSpan | None,
    definition_hash: SemanticHash | None,
    field: str,
) -> None:
    if state is DefinitionState.ABSENT and (
        kind is not DefinitionKind.NONE
        or any(counts)
        or attributes
        or option_hashes
        or source_span is not None
        or definition_hash is not None
    ):
        raise LegacyManifestError(field, "absent summaries carry no semantic detail")
    if state is DefinitionState.PRESENT_UNDEFINED and (
        kind is not DefinitionKind.NONE
        or any(counts)
        or source_span is not None
        or definition_hash is not None
    ):
        raise LegacyManifestError(
            field, "undefined summaries carry no structural detail"
        )
    if state is DefinitionState.PRESENT_DEFINED and kind is DefinitionKind.NONE:
        raise LegacyManifestError(field, "defined summaries require a definition kind")
    required = {
        DefinitionKind.OWN_VALUE: (0, 1, 0, 0),
        DefinitionKind.DOWN_VALUE: (1, 0, 0, 0),
        DefinitionKind.UP_VALUE: (0, 0, 1, 0),
        DefinitionKind.SUB_VALUE: (0, 0, 0, 1),
    }
    if kind in required and any(
        (count > 0) != bool(expected)
        for count, expected in zip(counts, required[kind], strict=True)
    ):
        raise LegacyManifestError(field, "kind and counts disagree")
    if kind is DefinitionKind.MIXED and sum(count > 0 for count in counts) < 2:
        raise LegacyManifestError(field, "mixed kind needs two families")


@dataclass(frozen=True, slots=True, init=False, repr=False)
class SymbolRecord:
    """One legacy symbol with explicit partition and evidence-state distinctions."""

    record_id: str
    name: str
    partition: LegacyPartition
    observations: tuple[SymbolObservation, ...]
    definition_state: DefinitionState
    definition_kind: DefinitionKind
    downvalue_count: int
    ownvalue_count: int
    upvalue_count: int
    subvalue_count: int
    attributes: tuple[SymbolAttribute, ...]
    option_hashes: tuple[SemanticHash, ...]
    source_span: SourceSpan | None
    definition_hash: SemanticHash | None
    summary_origin: EvidenceOrigin | None

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise LegacyManifestError("symbol", "is factory-owned")

    @classmethod
    def create(
        cls,
        name: object,
        partition: object,
        observations: object,
        *,
        definition_state: object = DefinitionState.ABSENT,
        definition_kind: object = None,
        downvalue_count: object = 0,
        ownvalue_count: object = 0,
        upvalue_count: object = 0,
        subvalue_count: object = 0,
        attributes: object = (),
        option_hashes: object = (),
        source_span: object = None,
        definition_hash: object = None,
        summary_origin: object = None,
        record_id: object | None = None,
    ) -> SymbolRecord:
        checked_state = cast(
            DefinitionState,
            _enum(definition_state, DefinitionState, "symbol.definition_state"),
        )
        checked_kind = cast(
            DefinitionKind,
            _enum(
                (
                    DefinitionKind.NONE
                    if checked_state
                    in (DefinitionState.ABSENT, DefinitionState.PRESENT_UNDEFINED)
                    else DefinitionKind.UNKNOWN
                )
                if definition_kind is None
                else definition_kind,
                DefinitionKind,
                "symbol.definition_kind",
            ),
        )
        counts = (downvalue_count, ownvalue_count, upvalue_count, subvalue_count)
        if any(
            type(count) is not int or not 0 <= count <= _MAX_ITEMS for count in counts
        ):
            raise LegacyManifestError(
                "symbol.definition_counts", "must be bounded exact ints"
            )
        counts = cast(tuple[int, int, int, int], counts)
        checked_attributes = tuple(
            cast(SymbolAttribute, _enum(item, SymbolAttribute, "symbol.attributes"))
            for item in _tuple(attributes, "symbol.attributes")
        )
        if len(set(checked_attributes)) != len(checked_attributes):
            raise LegacyManifestError("symbol.attributes", "contains duplicate values")
        checked_options = tuple(
            _hash(item, "symbol.option_hashes")
            for item in _tuple(option_hashes, "symbol.option_hashes")
        )
        if len(set(checked_options)) != len(checked_options):
            raise LegacyManifestError(
                "symbol.option_hashes", "contains duplicate values"
            )
        checked_summary_origin = (
            None
            if summary_origin is None
            else cast(
                EvidenceOrigin,
                _enum(summary_origin, EvidenceOrigin, "symbol.summary_origin"),
            )
        )
        if source_span is not None and type(source_span) is not SourceSpan:
            raise LegacyManifestError(
                "symbol.source_span", "must be an exact SourceSpan"
            )
        checked_span = (
            None
            if source_span is None
            else SourceSpan.create(source_span.start_line, source_span.end_line)
        )
        checked_hash = (
            None
            if definition_hash is None
            else _hash(definition_hash, "symbol.definition_hash")
        )
        _definition_shape(
            checked_state,
            checked_kind,
            counts,
            checked_attributes,
            checked_options,
            checked_span,
            checked_hash,
            "symbol.definition",
        )
        if checked_summary_origin is EvidenceOrigin.STATIC and (
            (checked_span is None) != (checked_hash is None)
            or (
                checked_state is DefinitionState.PRESENT_DEFINED
                and (checked_span is None or checked_hash is None)
            )
        ):
            raise LegacyManifestError(
                "symbol.definition", "static definitions require both source anchors"
            )
        if checked_summary_origin not in (None, EvidenceOrigin.STATIC) and (
            checked_span is not None or checked_hash is not None
        ):
            raise LegacyManifestError(
                "symbol.definition", "non-static summaries cannot carry source anchors"
            )
        checked_observations = _observations(observations, "symbol.observations")
        value = object.__new__(SymbolRecord)
        checked_name = _text(name, "symbol.name", identifier=True)
        object.__setattr__(
            value,
            "record_id",
            _text(
                f"symbol:{checked_name}" if record_id is None else record_id,
                "symbol.record_id",
                identifier=True,
            ),
        )
        object.__setattr__(value, "name", checked_name)
        object.__setattr__(
            value,
            "partition",
            cast(
                LegacyPartition, _enum(partition, LegacyPartition, "symbol.partition")
            ),
        )
        object.__setattr__(value, "observations", checked_observations)
        object.__setattr__(value, "definition_state", checked_state)
        object.__setattr__(value, "definition_kind", checked_kind)
        object.__setattr__(value, "downvalue_count", downvalue_count)
        object.__setattr__(value, "ownvalue_count", ownvalue_count)
        object.__setattr__(value, "upvalue_count", upvalue_count)
        object.__setattr__(value, "subvalue_count", subvalue_count)
        object.__setattr__(value, "attributes", tuple(sorted(checked_attributes)))
        object.__setattr__(
            value, "option_hashes", tuple(sorted(checked_options, key=str))
        )
        object.__setattr__(value, "source_span", checked_span)
        object.__setattr__(value, "definition_hash", checked_hash)
        if summary_origin is None and (
            checked_state is not DefinitionState.ABSENT
            or checked_kind is not DefinitionKind.NONE
            or any(counts)
            or checked_attributes
            or checked_options
            or checked_span is not None
            or checked_hash is not None
        ):
            raise LegacyManifestError(
                "symbol.summary_origin", "is required for aggregate definition detail"
            )
        object.__setattr__(
            value,
            "summary_origin",
            checked_summary_origin,
        )
        return value

    def __repr__(self) -> str:
        return "SymbolRecord()"

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("SymbolRecord cannot be subclassed")


@dataclass(frozen=True, slots=True, init=False, repr=False)
class AlgMulDefinitionRecord:
    """One per-origin structural definition summary for a captured symbol."""

    record_id: str
    symbol_id: str
    origin: EvidenceOrigin
    definition_state: DefinitionState
    definition_kind: DefinitionKind
    downvalue_count: int
    ownvalue_count: int
    upvalue_count: int
    subvalue_count: int
    attributes: tuple[SymbolAttribute, ...]
    option_hashes: tuple[SemanticHash, ...]
    source_span: SourceSpan | None
    definition_hash: SemanticHash | None

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise LegacyManifestError("definition", "is factory-owned")

    @classmethod
    def create(
        cls,
        symbol_id: object,
        origin: object,
        definition_state: object,
        definition_kind: object,
        *,
        downvalue_count: object = 0,
        ownvalue_count: object = 0,
        upvalue_count: object = 0,
        subvalue_count: object = 0,
        attributes: object = (),
        option_hashes: object = (),
        source_span: object = None,
        definition_hash: object = None,
        record_id: object | None = None,
    ) -> AlgMulDefinitionRecord:
        checked_origin = cast(
            EvidenceOrigin, _enum(origin, EvidenceOrigin, "definition.origin")
        )
        checked_state = cast(
            DefinitionState,
            _enum(definition_state, DefinitionState, "definition.state"),
        )
        checked_kind = cast(
            DefinitionKind, _enum(definition_kind, DefinitionKind, "definition.kind")
        )
        counts = (downvalue_count, ownvalue_count, upvalue_count, subvalue_count)
        if any(type(item) is not int or not 0 <= item <= _MAX_ITEMS for item in counts):
            raise LegacyManifestError("definition.counts", "must be bounded exact ints")
        counts = cast(tuple[int, int, int, int], counts)
        if source_span is not None and type(source_span) is not SourceSpan:
            raise LegacyManifestError(
                "definition.source", "must be an exact SourceSpan"
            )
        checked_span = (
            None
            if source_span is None
            else SourceSpan.create(source_span.start_line, source_span.end_line)
        )
        checked_hash = (
            None
            if definition_hash is None
            else _hash(definition_hash, "definition.hash")
        )
        attrs = tuple(
            cast(SymbolAttribute, _enum(item, SymbolAttribute, "definition.attributes"))
            for item in _tuple(attributes, "definition.attributes")
        )
        if len(set(attrs)) != len(attrs):
            raise LegacyManifestError(
                "definition.attributes", "contains duplicate values"
            )
        checked_options = tuple(
            _hash(item, "definition.option_hashes")
            for item in _tuple(option_hashes, "definition.option_hashes")
        )
        if len(set(checked_options)) != len(checked_options):
            raise LegacyManifestError(
                "definition.option_hashes", "contains duplicate values"
            )
        _definition_shape(
            checked_state,
            checked_kind,
            counts,
            attrs,
            checked_options,
            checked_span,
            checked_hash,
            "definition",
        )
        if checked_origin is not EvidenceOrigin.STATIC and (
            checked_span is not None or checked_hash is not None
        ):
            raise LegacyManifestError(
                "definition.source", "only static summaries may carry source anchors"
            )
        if checked_origin is EvidenceOrigin.STATIC and (
            (checked_span is None) != (checked_hash is None)
            or (
                checked_state is DefinitionState.PRESENT_DEFINED
                and (checked_span is None or checked_hash is None)
            )
        ):
            raise LegacyManifestError(
                "definition.source", "static definitions require both source anchors"
            )
        value = object.__new__(AlgMulDefinitionRecord)
        checked_symbol_id = _text(symbol_id, "definition.symbol_id", identifier=True)
        object.__setattr__(
            value,
            "record_id",
            _text(
                f"definition:{checked_symbol_id}:{checked_origin.value}"
                if record_id is None
                else record_id,
                "definition.record_id",
                identifier=True,
            ),
        )
        object.__setattr__(value, "symbol_id", checked_symbol_id)
        object.__setattr__(value, "origin", checked_origin)
        object.__setattr__(value, "definition_state", checked_state)
        object.__setattr__(value, "definition_kind", checked_kind)
        object.__setattr__(value, "downvalue_count", downvalue_count)
        object.__setattr__(value, "ownvalue_count", ownvalue_count)
        object.__setattr__(value, "upvalue_count", upvalue_count)
        object.__setattr__(value, "subvalue_count", subvalue_count)
        object.__setattr__(value, "attributes", tuple(sorted(attrs)))
        object.__setattr__(
            value,
            "option_hashes",
            tuple(sorted(checked_options, key=str)),
        )
        object.__setattr__(
            value,
            "source_span",
            checked_span,
        )
        object.__setattr__(
            value,
            "definition_hash",
            checked_hash,
        )
        return value

    def __repr__(self) -> str:
        return "AlgMulDefinitionRecord()"

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("AlgMulDefinitionRecord cannot be subclassed")


@dataclass(frozen=True, slots=True, init=False, repr=False)
class FactoryRecord:
    """A dynamic factory and its explicitly requested generated-name surface."""

    record_id: str
    name: str
    partition: LegacyPartition
    observations: tuple[SymbolObservation, ...]
    requested_names: tuple[str, ...]
    call_kind: FactoryCallKind
    call_relationships: tuple[str, ...]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise LegacyManifestError("factory", "is factory-owned")

    @classmethod
    def create(
        cls,
        name: object,
        observations: object,
        *,
        requested_names: object,
        partition: object = LegacyPartition.GENERATED_PROPERTY_API,
        record_id: object | None = None,
        call_kind: object = FactoryCallKind.GENERATES,
        call_relationships: object = (),
    ) -> FactoryRecord:
        checked_name = _text(name, "factory.name", identifier=True)
        checked_observations = _observations(observations, "factory.observations")
        checked_partition = cast(
            LegacyPartition, _enum(partition, LegacyPartition, "factory.partition")
        )
        checked_names = _names(
            requested_names,
            "factory.requested_names",
            expected=checked_name == "MakeProperty",
        )
        checked_relationships = _names(call_relationships, "factory.call_relationships")
        value = object.__new__(FactoryRecord)
        object.__setattr__(
            value,
            "record_id",
            _text(
                f"factory:{checked_name}" if record_id is None else record_id,
                "factory.record_id",
                identifier=True,
            ),
        )
        object.__setattr__(value, "name", checked_name)
        object.__setattr__(value, "partition", checked_partition)
        object.__setattr__(value, "observations", checked_observations)
        object.__setattr__(value, "requested_names", checked_names)
        object.__setattr__(
            value,
            "call_kind",
            cast(
                FactoryCallKind, _enum(call_kind, FactoryCallKind, "factory.call_kind")
            ),
        )
        object.__setattr__(value, "call_relationships", checked_relationships)
        return value

    def __repr__(self) -> str:
        return "FactoryRecord()"

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("FactoryRecord cannot be subclassed")


@dataclass(frozen=True, slots=True, init=False, repr=False)
class RegistryRecord:
    """One evaluated legacy registry entry and its multiplication route."""

    record_id: str
    name: str
    basis: tuple[str, ...]
    multiplication_symbol: str
    structure_dot_symbol: str | None
    partition: LegacyPartition
    basis_table_snapshot: SemanticHash | None
    basis_table_hash: SemanticHash | None
    origin: EvidenceOrigin

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise LegacyManifestError("registry", "is factory-owned")

    @classmethod
    def create(
        cls,
        name: object,
        basis: object,
        multiplication_symbol: object,
        *,
        partition: object = LegacyPartition.REGISTRY_BASES,
        record_id: object | None = None,
        basis_table_snapshot: object = None,
        basis_table_hash: object = None,
        origin: object = EvidenceOrigin.EVALUATED,
        structure_dot_symbol: object = None,
    ) -> RegistryRecord:
        checked_basis = (
            (_text(basis, "registry.basis"),)
            if type(basis) is str
            else _names(basis, "registry.basis")
        )
        if not checked_basis:
            raise LegacyManifestError("registry.basis", "must not be empty")
        value = object.__new__(RegistryRecord)
        checked_name = _text(name, "registry.name")
        object.__setattr__(
            value,
            "record_id",
            _text(
                f"registry:{checked_name}" if record_id is None else record_id,
                "registry.record_id",
                identifier=True,
            ),
        )
        object.__setattr__(value, "name", checked_name)
        object.__setattr__(value, "basis", checked_basis)
        object.__setattr__(
            value,
            "multiplication_symbol",
            _text(multiplication_symbol, "registry.multiplication_symbol"),
        )
        object.__setattr__(
            value,
            "partition",
            cast(
                LegacyPartition, _enum(partition, LegacyPartition, "registry.partition")
            ),
        )
        checked_snapshot = (
            None
            if basis_table_snapshot is None
            else _hash(basis_table_snapshot, "registry.basis_table_snapshot")
        )
        checked_table_hash = (
            None
            if basis_table_hash is None
            else _hash(basis_table_hash, "registry.basis_table_hash")
        )
        if (checked_snapshot is None) != (checked_table_hash is None):
            raise LegacyManifestError(
                "registry.basis_table", "requires both snapshot and hash links"
            )
        object.__setattr__(value, "basis_table_snapshot", checked_snapshot)
        object.__setattr__(value, "basis_table_hash", checked_table_hash)
        object.__setattr__(
            value,
            "origin",
            cast(EvidenceOrigin, _enum(origin, EvidenceOrigin, "registry.origin")),
        )
        object.__setattr__(
            value,
            "structure_dot_symbol",
            None
            if structure_dot_symbol is None
            else _text(structure_dot_symbol, "registry.structure_dot_symbol"),
        )
        return value

    def __repr__(self) -> str:
        return "RegistryRecord()"

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("RegistryRecord cannot be subclassed")


@dataclass(frozen=True, slots=True, init=False, repr=False)
class BehaviorRecord:
    """A classified behavior or side effect without executable legacy code."""

    record_id: str
    name: str
    description: str
    partition: LegacyPartition
    input_snapshot: SemanticHash | None
    output_snapshot: SemanticHash | None
    origin: EvidenceOrigin

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise LegacyManifestError("behavior", "is factory-owned")

    @classmethod
    def create(
        cls,
        name: object,
        description: object,
        *,
        partition: object = LegacyPartition.REGISTRY_BASES,
        record_id: object | None = None,
        input_snapshot: object = None,
        output_snapshot: object = None,
        origin: object = EvidenceOrigin.STATIC,
    ) -> BehaviorRecord:
        value = object.__new__(BehaviorRecord)
        checked_name = _text(name, "behavior.name", identifier=True)
        object.__setattr__(
            value,
            "record_id",
            _text(
                f"behavior:{checked_name}" if record_id is None else record_id,
                "behavior.record_id",
                identifier=True,
            ),
        )
        object.__setattr__(
            value,
            "origin",
            cast(EvidenceOrigin, _enum(origin, EvidenceOrigin, "behavior.origin")),
        )
        object.__setattr__(value, "name", checked_name)
        object.__setattr__(
            value, "description", _text(description, "behavior.description")
        )
        object.__setattr__(
            value,
            "partition",
            cast(
                LegacyPartition, _enum(partition, LegacyPartition, "behavior.partition")
            ),
        )
        object.__setattr__(
            value,
            "input_snapshot",
            None
            if input_snapshot is None
            else _hash(input_snapshot, "behavior.input_snapshot"),
        )
        object.__setattr__(
            value,
            "output_snapshot",
            None
            if output_snapshot is None
            else _hash(output_snapshot, "behavior.output_snapshot"),
        )
        return value

    def __repr__(self) -> str:
        return "BehaviorRecord()"

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("BehaviorRecord cannot be subclassed")


@dataclass(frozen=True, slots=True, init=False, repr=False)
class MessageRecord:
    """One capture message, retained as text rather than an executable object."""

    stage: str
    severity: str
    text: str
    origin: EvidenceOrigin

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise LegacyManifestError("message", "is factory-owned")

    @classmethod
    def create(
        cls,
        stage: object,
        severity: object,
        text: object,
        *,
        origin: object = EvidenceOrigin.STATIC,
    ) -> MessageRecord:
        value = object.__new__(MessageRecord)
        object.__setattr__(
            value, "stage", _text(stage, "message.stage", identifier=True)
        )
        object.__setattr__(
            value, "severity", _text(severity, "message.severity", identifier=True)
        )
        object.__setattr__(value, "text", _text(text, "message.text"))
        object.__setattr__(
            value,
            "origin",
            cast(EvidenceOrigin, _enum(origin, EvidenceOrigin, "message.origin")),
        )
        return value

    def __repr__(self) -> str:
        return "MessageRecord()"

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("MessageRecord cannot be subclassed")


def _message_target_id(message: MessageRecord) -> str:
    """Return the stable disposition target for one captured diagnostic."""

    payload = json.dumps(
        {
            "stage": message.stage,
            "severity": message.severity,
            "text": message.text,
            "origin": message.origin.value,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"message:{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True, slots=True, init=False, repr=False)
class DispositionRecord:
    """A v0.0 owner/disposition, rationale, and trace links for one record."""

    target_id: str
    partition: LegacyPartition
    disposition: Disposition
    owner: str
    note: str
    oracle_status: OracleStatus
    api_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    oracle_ids: tuple[str, ...]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise LegacyManifestError("disposition", "is factory-owned")

    @classmethod
    def create(
        cls,
        target_id: object,
        partition: object,
        disposition: object,
        *,
        owner: object,
        note: object,
        api_ids: object = (),
        test_ids: object = (),
        oracle_ids: object,
        oracle_status: object = None,
    ) -> DispositionRecord:
        checked_api_ids = _names(api_ids, "disposition.api_ids")
        checked_test_ids = _names(test_ids, "disposition.test_ids")
        checked_oracle_ids = _names(oracle_ids, "disposition.oracle_ids")
        checked_kind = cast(
            Disposition, _enum(disposition, Disposition, "disposition.kind")
        )
        checked_owner = _text(owner, "disposition.owner", identifier=True)
        checked_note = _text(note, "disposition.note")
        checked_oracle_status = cast(
            OracleStatus,
            _enum(
                OracleStatus.AVAILABLE
                if oracle_status is None and checked_oracle_ids
                else OracleStatus.PENDING
                if oracle_status is None
                else oracle_status,
                OracleStatus,
                "disposition.oracle_status",
            ),
        )
        if checked_kind is Disposition.DEFER and checked_owner == "v0.0":
            raise LegacyManifestError(
                "disposition.owner", "deferred entries need a later owner"
            )
        if checked_owner == "v0.0" and (not checked_api_ids or not checked_test_ids):
            raise LegacyManifestError(
                "disposition", "v0.0 ownership requires API and test links"
            )
        if checked_oracle_status is OracleStatus.AVAILABLE and not checked_oracle_ids:
            raise LegacyManifestError(
                "disposition.oracle_ids",
                "available oracle status requires at least one evidence link",
            )
        if checked_oracle_status is OracleStatus.PENDING and checked_oracle_ids:
            raise LegacyManifestError(
                "disposition.oracle_ids",
                "pending oracle status cannot claim an evidence link",
            )
        if (
            checked_owner == "v0.0"
            and checked_oracle_status is OracleStatus.PENDING
            and len(checked_note.strip()) < 20
        ):
            raise LegacyManifestError(
                "disposition.note",
                "pending v0.0 oracle needs a precise gap explanation",
            )
        value = object.__new__(DispositionRecord)
        object.__setattr__(
            value,
            "target_id",
            _text(target_id, "disposition.target_id", identifier=True),
        )
        object.__setattr__(
            value,
            "partition",
            cast(
                LegacyPartition,
                _enum(partition, LegacyPartition, "disposition.partition"),
            ),
        )
        object.__setattr__(
            value,
            "disposition",
            checked_kind,
        )
        object.__setattr__(
            value,
            "owner",
            checked_owner,
        )
        object.__setattr__(value, "note", checked_note)
        object.__setattr__(value, "oracle_status", checked_oracle_status)
        object.__setattr__(value, "api_ids", checked_api_ids)
        object.__setattr__(value, "test_ids", checked_test_ids)
        object.__setattr__(value, "oracle_ids", checked_oracle_ids)
        return value

    def __repr__(self) -> str:
        return "DispositionRecord()"

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("DispositionRecord cannot be subclassed")


def _is_static_source_site(definition: AlgMulDefinitionRecord) -> bool:
    """Recognise one literal static source site, never an aggregate summary."""

    required = {
        DefinitionKind.OWN_VALUE: (0, 1, 0, 0),
        DefinitionKind.DOWN_VALUE: (1, 0, 0, 0),
        DefinitionKind.UP_VALUE: (0, 0, 1, 0),
        DefinitionKind.SUB_VALUE: (0, 0, 0, 1),
    }
    counts = (
        definition.downvalue_count,
        definition.ownvalue_count,
        definition.upvalue_count,
        definition.subvalue_count,
    )
    return (
        definition.origin is EvidenceOrigin.STATIC
        and definition.definition_state is DefinitionState.PRESENT_DEFINED
        and definition.definition_kind in required
        and counts == required[definition.definition_kind]
        and not definition.attributes
        and not definition.option_hashes
        and definition.source_span is not None
        and definition.definition_hash is not None
    )


@dataclass(frozen=True, slots=True, init=False, repr=False)
class AlgMulManifest:
    """The schema-v1 immutable capture ledger for AlgMul's generic surfaces."""

    manifest_id: str
    source: SourceMetadata
    symbols: tuple[SymbolRecord, ...]
    definitions: tuple[AlgMulDefinitionRecord, ...]
    factories: tuple[FactoryRecord, ...]
    registries: tuple[RegistryRecord, ...]
    behaviors: tuple[BehaviorRecord, ...]
    messages: tuple[MessageRecord, ...]
    dispositions: tuple[DispositionRecord, ...]
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise LegacyManifestError("manifest", "is factory-owned")

    @classmethod
    def create(
        cls,
        *,
        manifest_id: object,
        source: object,
        symbols: object,
        definitions: object = (),
        factories: object,
        registries: object,
        behaviors: object,
        messages: object,
        dispositions: object,
    ) -> AlgMulManifest:
        if type(source) is not SourceMetadata:
            raise LegacyManifestError("source", "must be an exact SourceMetadata")
        checked_symbols = cast(
            tuple[SymbolRecord, ...], _records(symbols, SymbolRecord, "symbols")
        )
        checked_factories = cast(
            tuple[FactoryRecord, ...], _records(factories, FactoryRecord, "factories")
        )
        checked_definitions = cast(
            tuple[AlgMulDefinitionRecord, ...],
            _records(definitions, AlgMulDefinitionRecord, "definitions"),
        )
        checked_registries = cast(
            tuple[RegistryRecord, ...],
            _records(registries, RegistryRecord, "registries"),
        )
        checked_behaviors = cast(
            tuple[BehaviorRecord, ...], _records(behaviors, BehaviorRecord, "behaviors")
        )
        checked_messages = cast(
            tuple[MessageRecord, ...], _records(messages, MessageRecord, "messages")
        )
        checked_dispositions = cast(
            tuple[DispositionRecord, ...],
            _records(dispositions, DispositionRecord, "dispositions"),
        )
        checked_source = SourceMetadata.create(
            source.location,
            _hash(source.content_hash, "source.hash"),
            kernel_version=source.kernel_version,
            source_bytes=source.source_bytes,
            contexts=source.contexts,
            load_status=source.load_status,
            initial_context=source.initial_context,
            context_after_load=source.context_after_load,
            context_path_after_load=source.context_path_after_load,
            load_result_head=source.load_result_head,
        )
        checked_symbols = tuple(_copy_symbol(item) for item in checked_symbols)
        checked_definitions = tuple(
            _copy_definition(item) for item in checked_definitions
        )
        checked_factories = tuple(_copy_factory(item) for item in checked_factories)
        checked_registries = tuple(_copy_registry(item) for item in checked_registries)
        checked_behaviors = tuple(_copy_behavior(item) for item in checked_behaviors)
        checked_messages = tuple(_copy_message(item) for item in checked_messages)
        checked_dispositions = tuple(
            _copy_disposition(item) for item in checked_dispositions
        )
        _unique(checked_symbols, "symbols", lambda item: item.record_id)
        _unique(checked_symbols, "symbols", lambda item: item.name)
        _unique(checked_definitions, "definitions", lambda item: item.record_id)
        _unique(checked_factories, "factories", lambda item: item.record_id)
        _unique(checked_registries, "registries", lambda item: item.record_id)
        _unique(
            checked_registries,
            "registries",
            lambda item: (item.origin, item.name),
        )
        _unique(checked_behaviors, "behaviors", lambda item: item.record_id)
        _unique(checked_dispositions, "dispositions", lambda item: item.target_id)
        _unique(
            checked_messages,
            "messages",
            lambda item: (item.stage, item.severity, item.text, item.origin),
        )
        symbol_targets = {item.record_id: item.partition for item in checked_symbols}
        if not {item.symbol_id for item in checked_definitions} <= set(symbol_targets):
            raise LegacyManifestError(
                "definitions", "contains an unknown symbol record"
            )
        definition_targets = {
            item.record_id: symbol_targets[item.symbol_id]
            for item in checked_definitions
        }
        targets = (
            symbol_targets
            | definition_targets
            | {item.record_id: item.partition for item in checked_factories}
            | {item.record_id: item.partition for item in checked_registries}
            | {item.record_id: item.partition for item in checked_behaviors}
            | {
                _message_target_id(item): LegacyPartition.DISPLAY_REPORT_TABLES
                for item in checked_messages
            }
        )
        if len(targets) != (
            len(checked_symbols)
            + len(checked_definitions)
            + len(checked_factories)
            + len(checked_registries)
            + len(checked_behaviors)
            + len(checked_messages)
        ):
            raise LegacyManifestError("records", "record IDs must be globally unique")
        symbols_by_id = {item.record_id: item for item in checked_symbols}
        for definition in checked_definitions:
            outcome = {
                item.origin: item.outcome
                for item in symbols_by_id[definition.symbol_id].observations
            }.get(definition.origin)
            if (
                outcome is None
                or (
                    definition.definition_state is DefinitionState.ABSENT
                    and outcome is not EvidenceOutcome.ABSENT
                )
                or (
                    definition.definition_state is not DefinitionState.ABSENT
                    and outcome is not EvidenceOutcome.PRESENT
                )
            ):
                raise LegacyManifestError(
                    "definitions", "contradicts symbol observation"
                )
        definitions_by_origin: dict[
            tuple[str, EvidenceOrigin], list[AlgMulDefinitionRecord]
        ] = {}
        for definition in checked_definitions:
            definitions_by_origin.setdefault(
                (definition.symbol_id, definition.origin), []
            ).append(definition)
        for definitions_at_origin in definitions_by_origin.values():
            symbol = symbols_by_id[definitions_at_origin[0].symbol_id]
            if (
                definitions_at_origin[0].origin is EvidenceOrigin.STATIC
                and symbol.summary_origin is not EvidenceOrigin.STATIC
                and not all(
                    _is_static_source_site(item) for item in definitions_at_origin
                )
            ):
                raise LegacyManifestError(
                    "definitions", "static source sites must be anchored unit literals"
                )
            if len(definitions_at_origin) == 1:
                continue
            static_site_flags = tuple(
                _is_static_source_site(item) for item in definitions_at_origin
            )
            if any(static_site_flags) and not all(static_site_flags):
                raise LegacyManifestError(
                    "definitions", "cannot mix static source sites and summaries"
                )
            if all(static_site_flags):
                anchors = {
                    (
                        item.source_span.start_line,
                        item.source_span.end_line,
                        item.definition_hash,
                    )
                    for item in definitions_at_origin
                    if item.source_span is not None and item.definition_hash is not None
                }
                if len(anchors) != len(definitions_at_origin):
                    raise LegacyManifestError(
                        "definitions", "contains duplicate static source sites"
                    )
                continue
            raise LegacyManifestError(
                "definitions",
                (
                    "repeated origin definitions must be one summary or static "
                    "source sites"
                ),
            )
        for symbol in checked_symbols:
            if symbol.summary_origin is None:
                continue
            summaries = [
                item
                for item in checked_definitions
                if item.symbol_id == symbol.record_id
                and item.origin is symbol.summary_origin
            ]
            if len(summaries) != 1:
                raise LegacyManifestError(
                    "definitions", "summary origin needs one definition record"
                )
            summary = summaries[0]
            if (
                summary.definition_state is not symbol.definition_state
                or summary.definition_kind is not symbol.definition_kind
                or (
                    summary.downvalue_count,
                    summary.ownvalue_count,
                    summary.upvalue_count,
                    summary.subvalue_count,
                )
                != (
                    symbol.downvalue_count,
                    symbol.ownvalue_count,
                    symbol.upvalue_count,
                    symbol.subvalue_count,
                )
                or summary.attributes != symbol.attributes
                or summary.option_hashes != symbol.option_hashes
                or summary.source_span != symbol.source_span
                or summary.definition_hash != symbol.definition_hash
            ):
                raise LegacyManifestError(
                    "definitions", "summary does not match its definition record"
                )
        dispositions_by_target = {item.target_id: item for item in checked_dispositions}
        if not set(dispositions_by_target) <= set(targets):
            raise LegacyManifestError(
                "dispositions", "contains an unknown target record"
            )
        if any(
            dispositions_by_target[record_id].partition is not targets[record_id]
            for record_id in dispositions_by_target
        ):
            raise LegacyManifestError(
                "dispositions", "target partition does not match the record"
            )
        has_evaluated = (
            any(
                observation.origin is EvidenceOrigin.EVALUATED
                for symbol in checked_symbols
                for observation in symbol.observations
            )
            or any(
                observation.origin is EvidenceOrigin.EVALUATED
                for factory in checked_factories
                for observation in factory.observations
            )
            or any(
                item.origin is EvidenceOrigin.EVALUATED for item in checked_registries
            )
            or any(
                item.origin is EvidenceOrigin.EVALUATED for item in checked_behaviors
            )
            or any(item.origin is EvidenceOrigin.EVALUATED for item in checked_messages)
            or any(
                item.origin is EvidenceOrigin.EVALUATED for item in checked_definitions
            )
        )
        if (
            checked_source.load_status in (LoadStatus.UNKNOWN, LoadStatus.STATIC_ONLY)
            and has_evaluated
        ):
            raise LegacyManifestError(
                "source.load_status",
                "non-runtime capture cannot contain evaluated data",
            )
        return _manifest_instance(
            _text(manifest_id, "manifest_id", identifier=True),
            checked_source,
            tuple(
                sorted(checked_symbols, key=lambda item: (item.name, item.record_id))
            ),
            tuple(
                sorted(
                    checked_definitions,
                    key=lambda item: (item.symbol_id, item.record_id),
                )
            ),
            tuple(
                sorted(checked_factories, key=lambda item: (item.name, item.record_id))
            ),
            tuple(
                sorted(checked_registries, key=lambda item: (item.name, item.record_id))
            ),
            tuple(
                sorted(checked_behaviors, key=lambda item: (item.name, item.record_id))
            ),
            tuple(
                sorted(
                    checked_messages,
                    key=lambda item: (
                        item.stage,
                        item.severity,
                        item.text,
                        item.origin,
                    ),
                )
            ),
            tuple(
                sorted(
                    checked_dispositions,
                    key=lambda item: (item.partition.value, item.target_id),
                )
            ),
        )

    def canonical_bytes(self) -> bytes:
        return canonical_manifest_bytes(self)

    def assign_disposition(
        self,
        record_id: object,
        disposition: object,
        *,
        api_ids: object,
        test_ids: object,
        oracle_ids: object,
        note: object,
        owner: object = "v0.0",
    ) -> AlgMulManifest:
        """Return a new manifest with one record-targeted disposition replaced."""
        verified = _verified_manifest(self)
        target = _text(record_id, "disposition.target_id", identifier=True)
        symbol_partitions = {
            item.record_id: item.partition for item in verified.symbols
        }
        records = (
            symbol_partitions
            | {
                item.record_id: symbol_partitions[item.symbol_id]
                for item in verified.definitions
            }
            | {item.record_id: item.partition for item in verified.factories}
            | {item.record_id: item.partition for item in verified.registries}
            | {item.record_id: item.partition for item in verified.behaviors}
            | {
                _message_target_id(item): LegacyPartition.DISPLAY_REPORT_TABLES
                for item in verified.messages
            }
        )
        if target not in records:
            raise LegacyManifestError("disposition.target_id", "does not name a record")
        replacement = DispositionRecord.create(
            target,
            records[target],
            disposition,
            owner=owner,
            note=note,
            api_ids=api_ids,
            test_ids=test_ids,
            oracle_ids=oracle_ids,
        )
        return AlgMulManifest.create(
            manifest_id=verified.manifest_id,
            source=verified.source,
            symbols=verified.symbols,
            definitions=verified.definitions,
            factories=verified.factories,
            registries=verified.registries,
            behaviors=verified.behaviors,
            messages=verified.messages,
            dispositions=(
                replacement,
                *(item for item in verified.dispositions if item.target_id != target),
            ),
        )

    def validate_complete(self, *, owned_partitions: object) -> None:
        """Require one disposition for each requested owned record partition."""
        verified = _verified_manifest(self)
        checked_partitions = _tuple(owned_partitions, "owned_partitions")
        partitions = {
            cast(
                LegacyPartition,
                _enum(item, LegacyPartition, "owned_partitions"),
            )
            for item in checked_partitions
        }
        symbol_partitions = {
            item.record_id: item.partition for item in verified.symbols
        }
        records = (
            tuple((item.record_id, item.partition) for item in verified.symbols)
            + tuple(
                (item.record_id, symbol_partitions[item.symbol_id])
                for item in verified.definitions
            )
            + tuple((item.record_id, item.partition) for item in verified.factories)
            + tuple((item.record_id, item.partition) for item in verified.registries)
            + tuple((item.record_id, item.partition) for item in verified.behaviors)
            + tuple(
                (_message_target_id(item), LegacyPartition.DISPLAY_REPORT_TABLES)
                for item in verified.messages
            )
        )
        missing = tuple(
            record_id
            for record_id, partition in records
            if partition in partitions
            and record_id not in {entry.target_id for entry in verified.dispositions}
        )
        if missing:
            raise LegacyManifestError(
                "dispositions", "incomplete disposition assignments"
            )
        dispositions_by_target = {
            item.target_id: item for item in verified.dispositions
        }
        unlinked = tuple(
            record_id
            for record_id, partition in records
            if partition in partitions
            and (
                not dispositions_by_target[record_id].api_ids
                or not dispositions_by_target[record_id].test_ids
            )
        )
        if unlinked:
            raise LegacyManifestError(
                "dispositions", "completed assignments require API and test links"
            )
        invalid_ownership = tuple(
            record_id
            for record_id, partition in records
            if partition in partitions
            and (
                (
                    dispositions_by_target[record_id].disposition
                    is not Disposition.DEFER
                    and (
                        dispositions_by_target[record_id].owner != "v0.0"
                        or (
                            dispositions_by_target[record_id].oracle_status
                            is OracleStatus.AVAILABLE
                            and not dispositions_by_target[record_id].oracle_ids
                        )
                        or (
                            dispositions_by_target[record_id].oracle_status
                            is OracleStatus.PENDING
                            and (
                                dispositions_by_target[record_id].oracle_ids
                                or len(dispositions_by_target[record_id].note.strip())
                                < 20
                            )
                        )
                    )
                )
                or (
                    dispositions_by_target[record_id].disposition is Disposition.DEFER
                    and (
                        dispositions_by_target[record_id].owner == "v0.0"
                        or (
                            dispositions_by_target[record_id].oracle_ids
                            and dispositions_by_target[record_id].oracle_status
                            is not OracleStatus.AVAILABLE
                        )
                        or (
                            not dispositions_by_target[record_id].oracle_ids
                            and (
                                dispositions_by_target[record_id].oracle_status
                                is not OracleStatus.PENDING
                                or len(dispositions_by_target[record_id].note.strip())
                                < 20
                            )
                        )
                    )
                )
            )
        )
        if invalid_ownership:
            raise LegacyManifestError(
                "dispositions", "completed ownership or oracle links are invalid"
            )
        if LegacyPartition.GENERATED_PROPERTY_API not in partitions:
            return
        factories = [item for item in verified.factories if item.name == "MakeProperty"]
        symbol_groups: dict[str, list[SymbolRecord]] = {}
        for symbol in verified.symbols:
            symbol_groups.setdefault(symbol.name, []).append(symbol)
        makers = symbol_groups.get("MakeProperty", [])
        if len(factories) != 1 or len(makers) != 1:
            raise LegacyManifestError("properties", "missing MakeProperty capture")
        required = {
            (EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT),
            (EvidenceOrigin.EVALUATED, EvidenceOutcome.ABSENT),
            (EvidenceOrigin.INTENDED, EvidenceOutcome.PRESENT),
        }
        factory_states = {
            (item.origin, item.outcome) for item in factories[0].observations
        }
        if (
            factories[0].partition is not LegacyPartition.GENERATED_PROPERTY_API
            or not required <= factory_states
        ):
            raise LegacyManifestError(
                "properties", "MakeProperty states are incomplete"
            )
        maker = makers[0]
        maker_states = {(item.origin, item.outcome) for item in maker.observations}
        maker_definitions = [
            item
            for item in verified.definitions
            if item.symbol_id == maker.record_id
            and item.origin is EvidenceOrigin.STATIC
        ]
        if (
            maker.partition is not LegacyPartition.GENERATED_PROPERTY_API
            or (
                not any(
                    item.definition_state is DefinitionState.PRESENT_DEFINED
                    for item in maker_definitions
                )
            )
            or not required <= maker_states
        ):
            raise LegacyManifestError(
                "properties", "MakeProperty definition is incomplete"
            )
        property_symbols = {
            name: symbol_groups.get(name, []) for name in EXPECTED_PROPERTY_NAMES
        }
        if any(len(items) != 1 for items in property_symbols.values()):
            raise LegacyManifestError("properties", "missing intended property records")
        expected_states = {
            (EvidenceOrigin.EVALUATED, EvidenceOutcome.ABSENT),
            (EvidenceOrigin.INTENDED, EvidenceOutcome.PRESENT),
        }
        if any(
            item.partition is not LegacyPartition.GENERATED_PROPERTY_API
            or not any(
                definition.symbol_id == item.record_id
                for definition in verified.definitions
            )
            or any(
                definition.symbol_id == item.record_id
                and definition.definition_state is not DefinitionState.ABSENT
                for definition in verified.definitions
            )
            or not expected_states
            <= {(state.origin, state.outcome) for state in item.observations}
            for items in property_symbols.values()
            for item in items
        ):
            raise LegacyManifestError(
                "properties", "property symbol states are incomplete"
            )

    def __repr__(self) -> str:
        return "AlgMulManifest()"

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("AlgMulManifest cannot be subclassed")


_Item = TypeVar("_Item")


def _unique(
    items: tuple[_Item, ...], field: str, key: Callable[[_Item], object]
) -> None:
    names = [key(item) for item in items]
    if len(set(names)) != len(names):
        raise LegacyManifestError(field, "contains duplicate names")


def _copy_observation(value: SymbolObservation) -> SymbolObservation:
    return SymbolObservation.create(value.origin, value.outcome)


def _copy_symbol(value: SymbolRecord) -> SymbolRecord:
    return SymbolRecord.create(
        value.name,
        value.partition,
        tuple(_copy_observation(item) for item in value.observations),
        definition_state=value.definition_state,
        definition_kind=value.definition_kind,
        downvalue_count=value.downvalue_count,
        ownvalue_count=value.ownvalue_count,
        upvalue_count=value.upvalue_count,
        subvalue_count=value.subvalue_count,
        attributes=value.attributes,
        option_hashes=value.option_hashes,
        source_span=value.source_span,
        definition_hash=value.definition_hash,
        summary_origin=value.summary_origin,
        record_id=value.record_id,
    )


def _copy_definition(value: AlgMulDefinitionRecord) -> AlgMulDefinitionRecord:
    return AlgMulDefinitionRecord.create(
        value.symbol_id,
        value.origin,
        value.definition_state,
        value.definition_kind,
        downvalue_count=value.downvalue_count,
        ownvalue_count=value.ownvalue_count,
        upvalue_count=value.upvalue_count,
        subvalue_count=value.subvalue_count,
        attributes=value.attributes,
        option_hashes=value.option_hashes,
        source_span=value.source_span,
        definition_hash=value.definition_hash,
        record_id=value.record_id,
    )


def _copy_factory(value: FactoryRecord) -> FactoryRecord:
    return FactoryRecord.create(
        value.name,
        tuple(_copy_observation(item) for item in value.observations),
        requested_names=value.requested_names,
        partition=value.partition,
        record_id=value.record_id,
        call_kind=value.call_kind,
        call_relationships=value.call_relationships,
    )


def _copy_registry(value: RegistryRecord) -> RegistryRecord:
    return RegistryRecord.create(
        value.name,
        value.basis,
        value.multiplication_symbol,
        partition=value.partition,
        record_id=value.record_id,
        basis_table_snapshot=value.basis_table_snapshot,
        basis_table_hash=value.basis_table_hash,
        origin=value.origin,
        structure_dot_symbol=value.structure_dot_symbol,
    )


def _copy_behavior(value: BehaviorRecord) -> BehaviorRecord:
    return BehaviorRecord.create(
        value.name,
        value.description,
        partition=value.partition,
        record_id=value.record_id,
        input_snapshot=value.input_snapshot,
        output_snapshot=value.output_snapshot,
        origin=value.origin,
    )


def _copy_message(value: MessageRecord) -> MessageRecord:
    return MessageRecord.create(
        value.stage, value.severity, value.text, origin=value.origin
    )


def _copy_disposition(value: DispositionRecord) -> DispositionRecord:
    return DispositionRecord.create(
        value.target_id,
        value.partition,
        value.disposition,
        owner=value.owner,
        note=value.note,
        oracle_status=value.oracle_status,
        api_ids=value.api_ids,
        test_ids=value.test_ids,
        oracle_ids=value.oracle_ids,
    )


def _manifest_instance(
    manifest_id: str,
    source: SourceMetadata,
    symbols: tuple[SymbolRecord, ...],
    definitions: tuple[AlgMulDefinitionRecord, ...],
    factories: tuple[FactoryRecord, ...],
    registries: tuple[RegistryRecord, ...],
    behaviors: tuple[BehaviorRecord, ...],
    messages: tuple[MessageRecord, ...],
    dispositions: tuple[DispositionRecord, ...],
) -> AlgMulManifest:
    value = object.__new__(AlgMulManifest)
    for name, item in (
        ("manifest_id", manifest_id),
        ("source", source),
        ("symbols", symbols),
        ("definitions", definitions),
        ("factories", factories),
        ("registries", registries),
        ("behaviors", behaviors),
        ("messages", messages),
        ("dispositions", dispositions),
        ("semantic_hash", SemanticHash("sha256", "0" * 64)),
    ):
        object.__setattr__(value, name, item)
    object.__setattr__(
        value,
        "semantic_hash",
        SemanticHash(
            "sha256", hashlib.sha256(_unverified_canonical_bytes(value)).hexdigest()
        ),
    )
    return value


def _observation_record(value: SymbolObservation) -> dict[str, str]:
    return {"origin": value.origin.value, "outcome": value.outcome.value}


def _source_record(value: SourceMetadata) -> dict[str, object]:
    return {
        "location": value.location,
        "contentHash": _hash_record(value.content_hash),
        "kernelVersion": value.kernel_version,
        "sourceBytes": value.source_bytes,
        "contexts": list(value.contexts),
        "loadStatus": value.load_status.value,
        "initialContext": value.initial_context,
        "contextAfterLoad": value.context_after_load,
        "contextPathAfterLoad": list(value.context_path_after_load),
        "loadResultHead": value.load_result_head,
    }


def _compact(value: dict[str, object]) -> str:
    compact = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    if len(compact) > _MAX_COMPACT_RECORD:
        raise LegacyManifestError("pages", "compact record limit exceeded")
    return compact


def _pages(items: list[str]) -> list[list[str]]:
    return [
        items[index : index + _PAGE_ITEMS]
        for index in range(0, len(items), _PAGE_ITEMS)
    ]


def _manifest_record(value: AlgMulManifest) -> dict[str, object]:
    """Build a record only for already-validated internal construction."""
    return {
        "schemaType": _TAG,
        "schemaVersion": _VERSION,
        "id": value.manifest_id,
        "source": _source_record(value.source),
        "symbolPages": _pages(
            [
                _compact(
                    {
                        "recordId": item.record_id,
                        "name": item.name,
                        "partition": item.partition.value,
                        "observations": [
                            _observation_record(observation)
                            for observation in item.observations
                        ],
                        "definitionState": item.definition_state.value,
                        "definitionKind": item.definition_kind.value,
                        "downvalueCount": item.downvalue_count,
                        "ownvalueCount": item.ownvalue_count,
                        "upvalueCount": item.upvalue_count,
                        "subvalueCount": item.subvalue_count,
                        "attributes": [
                            attribute.value for attribute in item.attributes
                        ],
                        "optionHashes": [
                            _hash_record(option) for option in item.option_hashes
                        ],
                        "sourceSpan": None
                        if item.source_span is None
                        else {
                            "startLine": item.source_span.start_line,
                            "endLine": item.source_span.end_line,
                        },
                        "definitionHash": None
                        if item.definition_hash is None
                        else _hash_record(item.definition_hash),
                        "summaryOrigin": None
                        if item.summary_origin is None
                        else item.summary_origin.value,
                    }
                )
                for item in value.symbols
            ]
        ),
        "definitionPages": _pages(
            [
                _compact(
                    {
                        "recordId": item.record_id,
                        "symbolId": item.symbol_id,
                        "origin": item.origin.value,
                        "definitionState": item.definition_state.value,
                        "definitionKind": item.definition_kind.value,
                        "downvalueCount": item.downvalue_count,
                        "ownvalueCount": item.ownvalue_count,
                        "upvalueCount": item.upvalue_count,
                        "subvalueCount": item.subvalue_count,
                        "attributes": [
                            attribute.value for attribute in item.attributes
                        ],
                        "optionHashes": [
                            _hash_record(item) for item in item.option_hashes
                        ],
                        "sourceSpan": None
                        if item.source_span is None
                        else {
                            "startLine": item.source_span.start_line,
                            "endLine": item.source_span.end_line,
                        },
                        "definitionHash": None
                        if item.definition_hash is None
                        else _hash_record(item.definition_hash),
                    }
                )
                for item in value.definitions
            ]
        ),
        "factories": [
            {
                "recordId": item.record_id,
                "name": item.name,
                "partition": item.partition.value,
                "observations": [
                    _observation_record(observation)
                    for observation in item.observations
                ],
                "requestedNamePages": _pages(list(item.requested_names)),
                "callKind": item.call_kind.value,
                "callRelationships": list(item.call_relationships),
            }
            for item in value.factories
        ],
        "registries": [
            {
                "recordId": item.record_id,
                "name": item.name,
                "basis": list(item.basis),
                "multiplicationSymbol": item.multiplication_symbol,
                "structureDotSymbol": item.structure_dot_symbol,
                "partition": item.partition.value,
                "basisTableSnapshot": (
                    None
                    if item.basis_table_snapshot is None
                    else _hash_record(item.basis_table_snapshot)
                ),
                "basisTableHash": (
                    None
                    if item.basis_table_hash is None
                    else _hash_record(item.basis_table_hash)
                ),
                "origin": item.origin.value,
            }
            for item in value.registries
        ],
        "behaviors": [
            {
                "recordId": item.record_id,
                "name": item.name,
                "description": item.description,
                "partition": item.partition.value,
                "inputSnapshot": (
                    None
                    if item.input_snapshot is None
                    else _hash_record(item.input_snapshot)
                ),
                "outputSnapshot": (
                    None
                    if item.output_snapshot is None
                    else _hash_record(item.output_snapshot)
                ),
                "origin": item.origin.value,
            }
            for item in value.behaviors
        ],
        "messages": [
            {
                "stage": item.stage,
                "severity": item.severity,
                "text": item.text,
                "origin": item.origin.value,
            }
            for item in value.messages
        ],
        "dispositionPages": _pages(
            [
                _compact(
                    {
                        "targetId": item.target_id,
                        "partition": item.partition.value,
                        "disposition": item.disposition.value,
                        "owner": item.owner,
                        "note": item.note,
                        "oracleStatus": item.oracle_status.value,
                        "apiIds": list(item.api_ids),
                        "testIds": list(item.test_ids),
                        "oracleIds": list(item.oracle_ids),
                    }
                )
                for item in value.dispositions
            ]
        ),
    }


def manifest_record(value: object) -> dict[str, object]:
    """Return a fresh, integrity-checked v1 tagged record."""
    if type(value) is not AlgMulManifest:
        raise LegacyManifestError("manifest", "must be an exact AlgMulManifest")
    return _manifest_record(_verified_manifest(value))


def _body(record: dict[str, object]) -> dict[str, JSONValue]:
    return {
        key: cast(JSONValue, item)
        for key, item in record.items()
        if key not in {"schemaType", "schemaVersion"}
    }


def _encode_manifest(value: AlgMulManifest) -> dict[str, JSONValue]:
    return _body(manifest_record(_verified_manifest(value)))


def _encode_manifest_unverified(value: AlgMulManifest) -> dict[str, JSONValue]:
    return _body(_manifest_record(value))


def _unverified_canonical_bytes(value: AlgMulManifest) -> bytes:
    registry = SerializerRegistry().with_codec(
        _TAG, _VERSION, AlgMulManifest, _encode_manifest_unverified, _parse_manifest
    )
    return canonical_json(value, registry=registry)


def _verified_manifest(value: AlgMulManifest) -> AlgMulManifest:
    try:
        stored_hash = _hash(value.semantic_hash, "manifest.semantic_hash")
        current_hash = SemanticHash(
            "sha256", hashlib.sha256(_unverified_canonical_bytes(value)).hexdigest()
        )
    except Exception as error:
        raise LegacyManifestError(
            "manifest.semantic_hash", "cannot encode the current manifest state"
        ) from error
    if stored_hash != current_hash:
        raise LegacyManifestError(
            "manifest.semantic_hash", "does not match canonical bytes"
        )
    rebuilt = AlgMulManifest.create(
        manifest_id=value.manifest_id,
        source=value.source,
        symbols=value.symbols,
        definitions=value.definitions,
        factories=value.factories,
        registries=value.registries,
        behaviors=value.behaviors,
        messages=value.messages,
        dispositions=value.dispositions,
    )
    if current_hash != rebuilt.semantic_hash:
        raise LegacyManifestError(
            "manifest.semantic_hash", "does not match canonical bytes"
        )
    return rebuilt


def _parse_observations(value: object, field: str) -> tuple[SymbolObservation, ...]:
    if type(value) is not tuple:
        raise LegacyManifestError(field, "has an invalid record shape")
    return tuple(
        SymbolObservation.create(
            _mapping(item, frozenset(("origin", "outcome")), field)["origin"],
            _mapping(item, frozenset(("origin", "outcome")), field)["outcome"],
        )
        for item in value
    )


def _parse_source(value: object) -> SourceMetadata:
    fields = _mapping(
        value,
        frozenset(
            (
                "location",
                "contentHash",
                "kernelVersion",
                "sourceBytes",
                "contexts",
                "loadStatus",
                "initialContext",
                "contextAfterLoad",
                "contextPathAfterLoad",
                "loadResultHead",
            )
        ),
        "source",
    )
    kernel = fields["kernelVersion"]
    if kernel is not None and type(kernel) is not str:
        raise LegacyManifestError(
            "source.kernel_version", "has an invalid record shape"
        )
    return SourceMetadata.create(
        fields["location"],
        _parse_hash(fields["contentHash"], "source.hash"),
        kernel_version=kernel,
        source_bytes=fields["sourceBytes"],
        contexts=fields["contexts"],
        load_status=fields["loadStatus"],
        initial_context=fields["initialContext"],
        context_after_load=fields["contextAfterLoad"],
        context_path_after_load=fields["contextPathAfterLoad"],
        load_result_head=fields["loadResultHead"],
    )


def _parse_items(
    value: object, field: str, parser: Callable[[object], object]
) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise LegacyManifestError(field, "has an invalid record shape")
    if len(value) > _MAX_ITEMS:
        raise LegacyManifestError(field, "item limit exceeded")
    return tuple(parser(item) for item in value)


def _freeze_wire(value: object) -> object:
    if type(value) in (type(None), bool, int, str):
        return value
    if type(value) is list:
        return tuple(_freeze_wire(item) for item in value)
    if type(value) is dict:
        return {key: _freeze_wire(item) for key, item in value.items()}
    raise LegacyManifestError("pages", "contains an unsafe compact value")


def _no_duplicate_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate compact JSON key")
        result[key] = value
    return result


def _parse_pages(
    value: object, field: str, parser: Callable[[object], object]
) -> tuple[object, ...]:
    if type(value) is not tuple or len(value) > 8:
        raise LegacyManifestError(field, "has an invalid page shape")
    parsed: list[object] = []
    for page in value:
        if type(page) is not tuple or len(page) > _PAGE_ITEMS:
            raise LegacyManifestError(field, "has an invalid page size")
        for item in page:
            if type(item) is not str or len(item) > _MAX_COMPACT_RECORD:
                raise LegacyManifestError(field, "has an invalid compact record")
            try:
                wire = _freeze_wire(
                    json.loads(item, object_pairs_hook=_no_duplicate_json_object)
                )
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise LegacyManifestError(
                    field, "has an invalid compact record"
                ) from error
            parsed.append(parser(wire))
    if len(parsed) > _MAX_ITEMS:
        raise LegacyManifestError(field, "logical item limit exceeded")
    return tuple(parsed)


def _parse_name_pages(value: object, field: str) -> tuple[str, ...]:
    """Parse bounded non-JSON pages used by large factory name surfaces."""
    if type(value) is not tuple or len(value) > 8:
        raise LegacyManifestError(field, "has an invalid page shape")
    names: list[str] = []
    for page in value:
        if type(page) is not tuple or len(page) > _PAGE_ITEMS:
            raise LegacyManifestError(field, "has an invalid page size")
        names.extend(_text(item, field, identifier=True) for item in page)
    return _names(tuple(names), field)


def _parse_symbol(value: object) -> SymbolRecord:
    fields = _mapping(
        value,
        frozenset(
            (
                "name",
                "recordId",
                "partition",
                "observations",
                "definitionState",
                "definitionKind",
                "downvalueCount",
                "ownvalueCount",
                "upvalueCount",
                "subvalueCount",
                "attributes",
                "optionHashes",
                "sourceSpan",
                "definitionHash",
                "summaryOrigin",
            )
        ),
        "symbol",
    )
    span = fields["sourceSpan"]
    parsed_span = (
        None
        if span is None
        else SourceSpan.create(
            _mapping(span, frozenset(("startLine", "endLine")), "symbol.source_span")[
                "startLine"
            ],
            _mapping(span, frozenset(("startLine", "endLine")), "symbol.source_span")[
                "endLine"
            ],
        )
    )
    return SymbolRecord.create(
        fields["name"],
        fields["partition"],
        _parse_observations(fields["observations"], "symbol.observations"),
        definition_state=fields["definitionState"],
        definition_kind=fields["definitionKind"],
        downvalue_count=fields["downvalueCount"],
        ownvalue_count=fields["ownvalueCount"],
        upvalue_count=fields["upvalueCount"],
        subvalue_count=fields["subvalueCount"],
        attributes=fields["attributes"],
        option_hashes=tuple(
            _parse_hash(item, "symbol.option_hashes")
            for item in _tuple(fields["optionHashes"], "symbol.option_hashes")
        ),
        source_span=parsed_span,
        definition_hash=None
        if fields["definitionHash"] is None
        else _parse_hash(fields["definitionHash"], "symbol.definition_hash"),
        summary_origin=fields["summaryOrigin"],
        record_id=fields["recordId"],
    )


def _parse_factory(value: object) -> FactoryRecord:
    fields = _mapping(
        value,
        frozenset(
            (
                "recordId",
                "name",
                "partition",
                "observations",
                "requestedNamePages",
                "callKind",
                "callRelationships",
            )
        ),
        "factory",
    )
    return FactoryRecord.create(
        fields["name"],
        _parse_observations(fields["observations"], "factory.observations"),
        requested_names=_parse_name_pages(
            fields["requestedNamePages"], "factory.requested_names"
        ),
        partition=fields["partition"],
        record_id=fields["recordId"],
        call_kind=fields["callKind"],
        call_relationships=fields["callRelationships"],
    )


def _parse_definition(value: object) -> AlgMulDefinitionRecord:
    fields = _mapping(
        value,
        frozenset(
            (
                "recordId",
                "symbolId",
                "origin",
                "definitionState",
                "definitionKind",
                "downvalueCount",
                "ownvalueCount",
                "upvalueCount",
                "subvalueCount",
                "attributes",
                "optionHashes",
                "sourceSpan",
                "definitionHash",
            )
        ),
        "definition",
    )
    span = fields["sourceSpan"]
    parsed_span = (
        None
        if span is None
        else SourceSpan.create(
            _mapping(
                span, frozenset(("startLine", "endLine")), "definition.source_span"
            )["startLine"],
            _mapping(
                span, frozenset(("startLine", "endLine")), "definition.source_span"
            )["endLine"],
        )
    )
    return AlgMulDefinitionRecord.create(
        fields["symbolId"],
        fields["origin"],
        fields["definitionState"],
        fields["definitionKind"],
        downvalue_count=fields["downvalueCount"],
        ownvalue_count=fields["ownvalueCount"],
        upvalue_count=fields["upvalueCount"],
        subvalue_count=fields["subvalueCount"],
        attributes=fields["attributes"],
        option_hashes=tuple(
            _parse_hash(item, "definition.option_hashes")
            for item in _tuple(fields["optionHashes"], "definition.option_hashes")
        ),
        source_span=parsed_span,
        definition_hash=None
        if fields["definitionHash"] is None
        else _parse_hash(fields["definitionHash"], "definition.hash"),
        record_id=fields["recordId"],
    )


def _parse_registry(value: object) -> RegistryRecord:
    fields = _mapping(
        value,
        frozenset(
            (
                "recordId",
                "name",
                "basis",
                "multiplicationSymbol",
                "structureDotSymbol",
                "partition",
                "basisTableSnapshot",
                "basisTableHash",
                "origin",
            )
        ),
        "registry",
    )
    return RegistryRecord.create(
        fields["name"],
        _tuple(fields["basis"], "registry.basis"),
        fields["multiplicationSymbol"],
        structure_dot_symbol=fields["structureDotSymbol"],
        partition=fields["partition"],
        record_id=fields["recordId"],
        basis_table_snapshot=(
            None
            if fields["basisTableSnapshot"] is None
            else _parse_hash(
                fields["basisTableSnapshot"], "registry.basis_table_snapshot"
            )
        ),
        basis_table_hash=(
            None
            if fields["basisTableHash"] is None
            else _parse_hash(fields["basisTableHash"], "registry.basis_table_hash")
        ),
        origin=fields["origin"],
    )


def _parse_behavior(value: object) -> BehaviorRecord:
    fields = _mapping(
        value,
        frozenset(
            (
                "recordId",
                "name",
                "description",
                "partition",
                "inputSnapshot",
                "outputSnapshot",
                "origin",
            )
        ),
        "behavior",
    )
    return BehaviorRecord.create(
        fields["name"],
        fields["description"],
        partition=fields["partition"],
        record_id=fields["recordId"],
        input_snapshot=(
            None
            if fields["inputSnapshot"] is None
            else _parse_hash(fields["inputSnapshot"], "behavior.input_snapshot")
        ),
        output_snapshot=(
            None
            if fields["outputSnapshot"] is None
            else _parse_hash(fields["outputSnapshot"], "behavior.output_snapshot")
        ),
        origin=fields["origin"],
    )


def _parse_message(value: object) -> MessageRecord:
    fields = _mapping(
        value, frozenset(("stage", "severity", "text", "origin")), "message"
    )
    return MessageRecord.create(
        fields["stage"], fields["severity"], fields["text"], origin=fields["origin"]
    )


def _parse_disposition(value: object) -> DispositionRecord:
    fields = _mapping(
        value,
        frozenset(
            (
                "partition",
                "targetId",
                "disposition",
                "owner",
                "note",
                "oracleStatus",
                "apiIds",
                "testIds",
                "oracleIds",
            )
        ),
        "disposition",
    )
    return DispositionRecord.create(
        fields["targetId"],
        fields["partition"],
        fields["disposition"],
        owner=fields["owner"],
        note=fields["note"],
        oracle_status=fields["oracleStatus"],
        api_ids=fields["apiIds"],
        test_ids=fields["testIds"],
        oracle_ids=fields["oracleIds"],
    )


def _parse_manifest(record: object) -> AlgMulManifest:
    fields = _mapping(
        record,
        frozenset(
            (
                "schemaType",
                "schemaVersion",
                "id",
                "source",
                "symbolPages",
                "definitionPages",
                "factories",
                "registries",
                "behaviors",
                "messages",
                "dispositionPages",
            )
        ),
        "manifest",
    )
    return AlgMulManifest.create(
        manifest_id=fields["id"],
        source=_parse_source(fields["source"]),
        symbols=_parse_pages(fields["symbolPages"], "symbols", _parse_symbol),
        definitions=_parse_pages(
            fields["definitionPages"], "definitions", _parse_definition
        ),
        factories=_parse_items(fields["factories"], "factories", _parse_factory),
        registries=_parse_items(fields["registries"], "registries", _parse_registry),
        behaviors=_parse_items(fields["behaviors"], "behaviors", _parse_behavior),
        messages=_parse_items(fields["messages"], "messages", _parse_message),
        dispositions=_parse_pages(
            fields["dispositionPages"], "dispositions", _parse_disposition
        ),
    )


LEGACY_MANIFEST_REGISTRY = SerializerRegistry().with_codec(
    _TAG, _VERSION, AlgMulManifest, _encode_manifest, _parse_manifest
)


def legacy_manifest_registry() -> SerializerRegistry:
    """Return the exact v1 allow-list; no input selects migrations or imports."""
    return LEGACY_MANIFEST_REGISTRY


def canonical_manifest_bytes(value: object) -> bytes:
    """Return canonical bytes whose SHA-256 is ``AlgMulManifest.semantic_hash``."""
    if type(value) is not AlgMulManifest:
        raise LegacyManifestError("manifest", "must be an exact AlgMulManifest")
    return canonical_json(_verified_manifest(value), registry=LEGACY_MANIFEST_REGISTRY)


# Public v0.0 names for the durable definition/surface contracts.
AlgMulSurfaceManifest = AlgMulManifest
