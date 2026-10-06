"""Checked, content-addressed complete and interrupted enumeration reports."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import TypeAlias

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import SemanticHash

from .bounds import EnumerationBounds
from .filtering import compile_constraint_filter
from .pruning import PrunedEnumeration
from .reference import ReferenceEnumeration
from .spec import CensusSpec


_SCHEMA_VERSION = 1
_COMPLETE_TAG = "anyalgebra.census.report.complete"
_INCOMPLETE_TAG = "anyalgebra.census.report.incomplete"
_JSON_INTEGER_LIMIT = (1 << 63) - 1


class EnumerationReportError(AnyAlgebraError, ValueError):
    """An enumeration report source or accounting equation was invalid."""

    def __init__(self, *, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid enumeration report {field}: {reason}")


def _hash_record(value: SemanticHash) -> dict[str, str]:
    return {"algorithm": value.algorithm, "digest": value.digest}


def _bounds_record(value: EnumerationBounds) -> dict[str, int | None]:
    return {
        "maxCandidates": value.max_candidates,
        "maxMemoryBytes": value.max_memory_bytes,
        "maxObservedMilliseconds": value.max_observed_milliseconds,
        "maxOrbits": value.max_orbits,
        "maxWorkUnits": value.max_work_units,
    }


def _integer_record(value: int) -> int | dict[str, int | str]:
    """Avoid interpreter decimal-digit limits for exact combinatorial counts."""
    if value <= _JSON_INTEGER_LIMIT:
        return value
    return {"digits": format(value, "x"), "radix": 16}


def _rejection_counts(value: object) -> tuple[tuple[str, int], ...]:
    if type(value) is not tuple:
        raise EnumerationReportError(
            field="rejection_counts", reason="must be an exact canonical tuple"
        )
    result: list[tuple[str, int]] = []
    for item in value:
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or not item[0]
            or type(item[1]) is not int
            or item[1] <= 0
        ):
            raise EnumerationReportError(
                field="rejection_counts", reason="contains an invalid entry"
            )
        result.append((item[0], item[1]))
    if tuple(result) != tuple(sorted(result)) or len({key for key, _ in result}) != len(
        result
    ):
        raise EnumerationReportError(
            field="rejection_counts", reason="must be unique and sorted"
        )
    return tuple(result)


def _count(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise EnumerationReportError(
            field=field, reason="must be a non-negative exact built-in int"
        )
    return value


def _common(
    *,
    spec: CensusSpec,
    total: int,
    examined: int,
    rejected: int,
    pruned: int,
    emitted: int,
    rejection_counts: tuple[tuple[str, int], ...],
    algorithm: str,
) -> tuple[int, int, int, int, int, tuple[tuple[str, int], ...], str]:
    if type(spec) is not CensusSpec:
        raise EnumerationReportError(field="spec", reason="must be exact CensusSpec")
    checked = (
        _count(total, "total_candidate_count"),
        _count(examined, "examined_candidate_count"),
        _count(rejected, "rejected_candidate_count"),
        _count(pruned, "pruned_candidate_count"),
        _count(emitted, "emitted_candidate_count"),
    )
    if type(algorithm) is not str or not algorithm:
        raise EnumerationReportError(field="algorithm", reason="must be stable text")
    reasons = _rejection_counts(rejection_counts)
    if checked[1] != checked[2] + checked[4]:
        raise EnumerationReportError(
            field="accounting", reason="examined must equal rejected plus emitted"
        )
    if sum(count for _, count in reasons) < checked[2]:
        raise EnumerationReportError(
            field="rejection_counts",
            reason="cannot account for every rejected candidate",
        )
    return (*checked, reasons, algorithm)


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class CompleteEnumerationReport:
    """Proof-shaped report whose accounting covers the entire declared corpus."""

    spec: CensusSpec
    bounds: EnumerationBounds
    status: str
    total_candidate_count: int
    examined_candidate_count: int
    rejected_candidate_count: int
    pruned_candidate_count: int
    emitted_candidate_count: int
    rejection_counts: tuple[tuple[str, int], ...]
    algorithm: str
    algorithm_version: int
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise EnumerationReportError(field="complete", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("CompleteEnumerationReport cannot be subclassed")

    def __eq__(self, other: object) -> bool:
        return type(other) is CompleteEnumerationReport and _identity(
            self
        ) == _identity(other)

    def __hash__(self) -> int:
        return int.from_bytes(
            bytes.fromhex(self.semantic_hash.digest[:16]), "big", signed=True
        )

    def __repr__(self) -> str:
        return (
            "CompleteEnumerationReport("
            f"semantic_hash={str(self.semantic_hash)!r}, "
            f"emitted_candidate_count={self.emitted_candidate_count})"
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class IncompleteEnumerationReport:
    """Bounded-prefix report that structurally requires a frontier and reason."""

    spec: CensusSpec
    bounds: EnumerationBounds
    status: str
    total_candidate_count: int
    examined_candidate_count: int
    rejected_candidate_count: int
    pruned_candidate_count: int
    emitted_candidate_count: int
    rejection_counts: tuple[tuple[str, int], ...]
    next_candidate_index: int
    stop_reason: str
    algorithm: str
    algorithm_version: int
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise EnumerationReportError(field="incomplete", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("IncompleteEnumerationReport cannot be subclassed")

    def __eq__(self, other: object) -> bool:
        return type(other) is IncompleteEnumerationReport and _identity(
            self
        ) == _identity(other)

    def __hash__(self) -> int:
        return int.from_bytes(
            bytes.fromhex(self.semantic_hash.digest[:16]), "big", signed=True
        )

    def __repr__(self) -> str:
        return (
            "IncompleteEnumerationReport("
            f"semantic_hash={str(self.semantic_hash)!r}, "
            f"next_candidate_index={self.next_candidate_index}, "
            f"stop_reason={self.stop_reason!r})"
        )


EnumerationReport: TypeAlias = CompleteEnumerationReport | IncompleteEnumerationReport


def _identity(value: EnumerationReport) -> tuple[object, ...]:
    common = (
        value.spec,
        value.bounds,
        value.status,
        value.total_candidate_count,
        value.examined_candidate_count,
        value.rejected_candidate_count,
        value.pruned_candidate_count,
        value.emitted_candidate_count,
        value.rejection_counts,
        value.algorithm,
        value.algorithm_version,
        value.semantic_hash,
    )
    if type(value) is IncompleteEnumerationReport:
        return (*common, value.next_candidate_index, value.stop_reason)
    return common


def _body(value: EnumerationReport) -> dict[str, object]:
    body: dict[str, object] = {
        "algorithm": value.algorithm,
        "algorithmVersion": value.algorithm_version,
        "bounds": _bounds_record(value.bounds),
        "emittedCandidateCount": _integer_record(value.emitted_candidate_count),
        "examinedCandidateCount": _integer_record(value.examined_candidate_count),
        "prunedCandidateCount": _integer_record(value.pruned_candidate_count),
        "rejectedCandidateCount": _integer_record(value.rejected_candidate_count),
        "rejectionCounts": [
            {"count": count, "reason": reason}
            for reason, count in value.rejection_counts
        ],
        "specHash": _hash_record(value.spec.semantic_hash),
        "status": value.status,
        "totalCandidateCount": _integer_record(value.total_candidate_count),
    }
    if type(value) is IncompleteEnumerationReport:
        body["nextCandidateIndex"] = _integer_record(value.next_candidate_index)
        body["stopReason"] = value.stop_reason
    return body


def _tag(value: EnumerationReport) -> str:
    return (
        _COMPLETE_TAG if type(value) is CompleteEnumerationReport else _INCOMPLETE_TAG
    )


def _semantic_hash(value: EnumerationReport) -> SemanticHash:
    record = {
        "schemaType": _tag(value),
        "schemaVersion": _SCHEMA_VERSION,
        **_body(value),
    }
    encoded = json.dumps(
        record,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return SemanticHash("sha256", hashlib.sha256(encoded).hexdigest())


def _assign_common(
    value: EnumerationReport,
    spec: CensusSpec,
    common: tuple[int, int, int, int, int, tuple[tuple[str, int], ...], str],
    status: str,
) -> None:
    total, examined, rejected, pruned, emitted, reasons, algorithm = common
    for field, item in (
        ("spec", spec),
        ("bounds", spec.bounds),
        ("status", status),
        ("total_candidate_count", total),
        ("examined_candidate_count", examined),
        ("rejected_candidate_count", rejected),
        ("pruned_candidate_count", pruned),
        ("emitted_candidate_count", emitted),
        ("rejection_counts", reasons),
        ("algorithm", algorithm),
        ("algorithm_version", 1),
        ("semantic_hash", SemanticHash("sha256", "0" * 64)),
    ):
        object.__setattr__(value, field, item)


def _complete(
    spec: CensusSpec,
    *,
    total: int,
    examined: int,
    rejected: int,
    pruned: int,
    emitted: int,
    rejection_counts: tuple[tuple[str, int], ...],
    algorithm: str,
) -> CompleteEnumerationReport:
    common = _common(
        spec=spec,
        total=total,
        examined=examined,
        rejected=rejected,
        pruned=pruned,
        emitted=emitted,
        rejection_counts=rejection_counts,
        algorithm=algorithm,
    )
    if total != examined + pruned:
        raise EnumerationReportError(
            field="accounting", reason="complete report does not cover the corpus"
        )
    value = object.__new__(CompleteEnumerationReport)
    _assign_common(value, spec, common, "complete")
    object.__setattr__(value, "semantic_hash", _semantic_hash(value))
    return value


def _incomplete(
    spec: CensusSpec,
    *,
    total: int,
    examined: int,
    rejected: int,
    pruned: int,
    emitted: int,
    rejection_counts: tuple[tuple[str, int], ...],
    next_candidate_index: int,
    stop_reason: str,
    algorithm: str,
) -> IncompleteEnumerationReport:
    common = _common(
        spec=spec,
        total=total,
        examined=examined,
        rejected=rejected,
        pruned=pruned,
        emitted=emitted,
        rejection_counts=rejection_counts,
        algorithm=algorithm,
    )
    frontier = _count(next_candidate_index, "next_candidate_index")
    if examined + pruned >= total or frontier != examined + pruned:
        raise EnumerationReportError(
            field="accounting", reason="incomplete frontier is inconsistent"
        )
    if type(stop_reason) is not str or not stop_reason:
        raise EnumerationReportError(field="stop_reason", reason="must be stable text")
    value = object.__new__(IncompleteEnumerationReport)
    _assign_common(value, spec, common, "bounded_incomplete")
    object.__setattr__(value, "next_candidate_index", frontier)
    object.__setattr__(value, "stop_reason", stop_reason)
    object.__setattr__(value, "semantic_hash", _semantic_hash(value))
    return value


def _reason_counts(reasons: list[str]) -> tuple[tuple[str, int], ...]:
    return tuple((reason, reasons.count(reason)) for reason in sorted(set(reasons)))


def report_reference_enumeration(
    enumeration: ReferenceEnumeration,
) -> EnumerationReport:
    """Filter one raw batch and construct the only structurally valid report type."""
    if type(enumeration) is not ReferenceEnumeration:
        raise EnumerationReportError(
            field="enumeration", reason="must be exact ReferenceEnumeration"
        )
    compiled = compile_constraint_filter(enumeration.spec)
    results = tuple(compiled.evaluate(item) for item in enumeration.candidates)
    rejected = sum(not result.accepted for result in results)
    emitted = len(results) - rejected
    reasons = _reason_counts(
        [reason for result in results for reason in result.rejection_reasons]
    )
    if enumeration.complete:
        return _complete(
            enumeration.spec,
            total=enumeration.total_candidate_count,
            examined=enumeration.examined_count,
            rejected=rejected,
            pruned=0,
            emitted=emitted,
            rejection_counts=reasons,
            algorithm="anyalgebra.census.reference",
        )
    assert enumeration.next_candidate_index is not None
    assert enumeration.stop_reason is not None
    return _incomplete(
        enumeration.spec,
        total=enumeration.total_candidate_count,
        examined=enumeration.examined_count,
        rejected=rejected,
        pruned=0,
        emitted=emitted,
        rejection_counts=reasons,
        next_candidate_index=enumeration.next_candidate_index,
        stop_reason=enumeration.stop_reason,
        algorithm="anyalgebra.census.reference",
    )


def report_pruned_enumeration(
    enumeration: PrunedEnumeration,
) -> CompleteEnumerationReport:
    """Construct a complete report from a fully accounted pruned DFS result."""
    if type(enumeration) is not PrunedEnumeration or not enumeration.complete:
        raise EnumerationReportError(
            field="enumeration", reason="must be an exact complete PrunedEnumeration"
        )
    return _complete(
        enumeration.spec,
        total=enumeration.spec.core.candidate_count,
        examined=enumeration.examined_candidate_count,
        rejected=0,
        pruned=enumeration.skipped_candidate_count,
        emitted=len(enumeration.accepted_candidates),
        rejection_counts=(),
        algorithm="anyalgebra.census.local_prefix_pruning",
    )


def enumeration_report_record(value: EnumerationReport) -> dict[str, object]:
    """Return one fresh report record with a type-specific schema tag."""
    if type(value) not in (CompleteEnumerationReport, IncompleteEnumerationReport):
        raise EnumerationReportError(field="report", reason="must be an exact report")
    if _semantic_hash(value) != value.semantic_hash:
        raise EnumerationReportError(field="semantic_hash", reason="content drift")
    return {
        "schemaType": _tag(value),
        "schemaVersion": _SCHEMA_VERSION,
        **_body(value),
        "contentHash": _hash_record(value.semantic_hash),
    }


def enumeration_report_canonical_bytes(value: EnumerationReport) -> bytes:
    """Return deterministic UTF-8 JSON bytes for one checked report."""
    return json.dumps(
        enumeration_report_record(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


__all__ = (
    "CompleteEnumerationReport",
    "EnumerationReport",
    "EnumerationReportError",
    "IncompleteEnumerationReport",
    "enumeration_report_canonical_bytes",
    "enumeration_report_record",
    "report_pruned_enumeration",
    "report_reference_enumeration",
)
