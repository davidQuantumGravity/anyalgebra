"""Simple deterministic, unpruned enumeration of finite operation tables."""

from __future__ import annotations

from dataclasses import dataclass

from anyalgebra.core.errors import AnyAlgebraError

from .spec import CensusSpec, CensusSpecCore
from .table_codes import unrank_operation_table


REFERENCE_BATCH_LIMIT = 1_000_000
_CANDIDATE_RECORD_OVERHEAD = 128
_BYTES_PER_WORKING_DIGIT = 16


class ReferenceEnumerationError(AnyAlgebraError, ValueError):
    """A reference-enumeration input or internal record was invalid."""

    def __init__(self, *, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid reference enumeration {field}: {reason}")


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ReferenceCandidate:
    """One raw operation table paired with its canonical candidate index."""

    candidate_index: int
    outputs: tuple[int, ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ReferenceEnumerationError(field="candidate", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("ReferenceCandidate cannot be subclassed")

    @classmethod
    def _create(cls, core: CensusSpecCore, candidate_index: int) -> ReferenceCandidate:
        value = object.__new__(ReferenceCandidate)
        object.__setattr__(value, "candidate_index", candidate_index)
        object.__setattr__(
            value, "outputs", unrank_operation_table(core, candidate_index)
        )
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is ReferenceCandidate and (
            self.candidate_index,
            self.outputs,
        ) == (other.candidate_index, other.outputs)

    def __repr__(self) -> str:
        return (
            "ReferenceCandidate("
            f"candidate_index={self.candidate_index}, cell_count={len(self.outputs)})"
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ReferenceEnumeration:
    """Immutable exact result of one bounded raw reference-enumeration pass."""

    spec: CensusSpec
    core: CensusSpecCore
    candidates: tuple[ReferenceCandidate, ...]
    total_candidate_count: int
    examined_count: int
    emitted_count: int
    next_candidate_index: int | None
    complete: bool
    stop_reason: str | None
    workspace_bytes_per_candidate: int
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ReferenceEnumerationError(field="result", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("ReferenceEnumeration cannot be subclassed")

    @classmethod
    def _create(
        cls,
        spec: CensusSpec,
        candidates: tuple[ReferenceCandidate, ...],
        *,
        total: int,
        stop_reason: str | None,
        workspace_bytes: int,
    ) -> ReferenceEnumeration:
        emitted = len(candidates)
        complete = emitted == total
        if complete:
            stop_reason = None
        value = object.__new__(ReferenceEnumeration)
        for field, item in (
            ("spec", spec),
            ("core", spec.core),
            ("candidates", candidates),
            ("total_candidate_count", total),
            ("examined_count", emitted),
            ("emitted_count", emitted),
            ("next_candidate_index", None if complete else emitted),
            ("complete", complete),
            ("stop_reason", stop_reason),
            ("workspace_bytes_per_candidate", workspace_bytes),
        ):
            object.__setattr__(value, field, item)
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is ReferenceEnumeration and (
            self.spec,
            self.candidates,
            self.total_candidate_count,
            self.examined_count,
            self.emitted_count,
            self.next_candidate_index,
            self.complete,
            self.stop_reason,
            self.workspace_bytes_per_candidate,
        ) == (
            other.spec,
            other.candidates,
            other.total_candidate_count,
            other.examined_count,
            other.emitted_count,
            other.next_candidate_index,
            other.complete,
            other.stop_reason,
            other.workspace_bytes_per_candidate,
        )

    def __repr__(self) -> str:
        return (
            "ReferenceEnumeration("
            f"examined_count={self.examined_count}, "
            f"total_candidate_count_bits={self.total_candidate_count.bit_length()}, "
            f"complete={self.complete}, stop_reason={self.stop_reason!r})"
        )


def _limit(spec: CensusSpec, total: int, workspace: int) -> tuple[int, str | None]:
    """Choose the first smallest deterministic budget in canonical order."""
    if total == 0:
        return 0, None
    memory_capacity = spec.bounds.max_memory_bytes // workspace
    selected = total
    reason: str | None = None
    for capacity, field in (
        (spec.bounds.max_candidates, "max_candidates"),
        (spec.bounds.max_work_units, "max_work_units"),
        (memory_capacity, "max_memory_bytes"),
        (REFERENCE_BATCH_LIMIT, "implementation_batch_limit"),
    ):
        if capacity < selected:
            selected = capacity
            reason = field
    return selected, reason


def enumerate_reference(spec: CensusSpec) -> ReferenceEnumeration:
    """Materialize the bounded prefix of raw tables in canonical index order."""
    if type(spec) is not CensusSpec:
        raise ReferenceEnumerationError(field="spec", reason="must be exact CensusSpec")
    workspace = (
        _CANDIDATE_RECORD_OVERHEAD
        + _BYTES_PER_WORKING_DIGIT * spec.core.input_tuple_count
    )
    total = spec.core.candidate_count
    stop, reason = _limit(spec, total, workspace)
    candidates = tuple(
        ReferenceCandidate._create(spec.core, index) for index in range(stop)
    )
    return ReferenceEnumeration._create(
        spec,
        candidates,
        total=total,
        stop_reason=reason,
        workspace_bytes=workspace,
    )


__all__ = (
    "REFERENCE_BATCH_LIMIT",
    "ReferenceCandidate",
    "ReferenceEnumeration",
    "ReferenceEnumerationError",
    "enumerate_reference",
)
