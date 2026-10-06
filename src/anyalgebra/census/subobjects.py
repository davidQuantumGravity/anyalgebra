"""Exact closed subsets and congruences of one bounded finite carrier."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
import json
from typing import NoReturn

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import SemanticHash
from anyalgebra.structures.outcomes import UnsupportedCapability

from .spec import CensusSpecCore
from .table_codes import OperationTableCodeError, rank_operation_table


_SCHEMA_VERSION = 1


class FiniteSubobjectAnalysisError(AnyAlgebraError, ValueError):
    """A finite subobject request or retained result is malformed."""

    def __init__(self, *, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid finite subobject analysis {field}: {reason}")


@dataclass(frozen=True, slots=True)
class FiniteSubobjectBounds:
    """Hard preflight limits for a complete subset and partition search."""

    max_carrier_size: int = 8
    max_subset_candidates: int = 65_535
    max_partition_candidates: int = 10_000
    max_closure_checks: int = 2_000_000
    max_compatibility_checks: int = 20_000_000

    def __post_init__(self) -> None:
        for field in (
            "max_carrier_size",
            "max_subset_candidates",
            "max_partition_candidates",
            "max_closure_checks",
            "max_compatibility_checks",
        ):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise FiniteSubobjectAnalysisError(
                    field=field, reason="must be a positive exact int"
                )


Partition = tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class FiniteSubobjectAnalysis:
    """Complete deterministic closed-subset and congruence enumeration."""

    core: CensusSpecCore
    outputs: tuple[int, ...]
    bounds: FiniteSubobjectBounds
    substructures: tuple[tuple[int, ...], ...]
    congruences: tuple[Partition, ...]
    substructure_status: str
    congruence_status: str
    ideal_status: str
    derivation_status: str
    hypotheses: tuple[str, ...]
    subset_candidates_examined: int
    closure_assignments_checked: int
    partition_candidates_examined: int
    compatibility_pairs_checked: int
    complete: bool
    algorithm: str
    algorithm_version: int
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise FiniteSubobjectAnalysisError(
            field="subobjects", reason="is factory-owned"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("FiniteSubobjectAnalysis cannot be subclassed")

    def __eq__(self, other: object) -> bool:
        return type(other) is FiniteSubobjectAnalysis and _identity(self) == _identity(
            other
        )


def _identity(value: FiniteSubobjectAnalysis) -> tuple[object, ...]:
    return (
        value.core,
        value.outputs,
        value.bounds,
        value.substructures,
        value.congruences,
        value.substructure_status,
        value.congruence_status,
        value.ideal_status,
        value.derivation_status,
        value.hypotheses,
        value.subset_candidates_examined,
        value.closure_assignments_checked,
        value.partition_candidates_examined,
        value.compatibility_pairs_checked,
        value.complete,
        value.algorithm,
        value.algorithm_version,
        value.semantic_hash,
    )


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


def _validated(
    core: object, outputs: object, options: object
) -> tuple[CensusSpecCore, tuple[int, ...], FiniteSubobjectBounds]:
    if type(core) is not CensusSpecCore:
        raise FiniteSubobjectAnalysisError(
            field="core", reason="must be exact CensusSpecCore"
        )
    if core.candidate_count == 0:
        raise FiniteSubobjectAnalysisError(
            field="outputs", reason="core has no operation tables"
        )
    if type(outputs) is not tuple:
        raise FiniteSubobjectAnalysisError(
            field="outputs", reason="must be exact tuple"
        )
    try:
        rank_operation_table(core, outputs)
    except OperationTableCodeError as error:
        raise FiniteSubobjectAnalysisError(
            field="outputs", reason="invalid table"
        ) from error
    if options is None:
        bounds = FiniteSubobjectBounds()
    elif type(options) is FiniteSubobjectBounds:
        bounds = FiniteSubobjectBounds(
            max_carrier_size=options.max_carrier_size,
            max_subset_candidates=options.max_subset_candidates,
            max_partition_candidates=options.max_partition_candidates,
            max_closure_checks=options.max_closure_checks,
            max_compatibility_checks=options.max_compatibility_checks,
        )
    else:
        raise FiniteSubobjectAnalysisError(
            field="options", reason="must be exact FiniteSubobjectBounds or None"
        )
    return core, outputs, bounds


def _apply(outputs: tuple[int, ...], size: int, arguments: tuple[int, ...]) -> int:
    index = 0
    for argument in arguments:
        index = index * size + argument
    return outputs[index]


def _bell_number(size: int) -> int:
    row = [1]
    for _ in range(size):
        next_row = [row[-1]]
        for index in range(1, len(row) + 1):
            next_row.append(next_row[-1] + row[index - 1])
        row = next_row
    return row[0]


def _partitions(size: int) -> tuple[Partition, ...]:
    if size == 0:
        return ((),)
    labels = [0 for _ in range(size)]
    found: list[Partition] = []

    def extend(position: int, maximum: int) -> None:
        if position == size:
            found.append(
                tuple(
                    tuple(index for index, label in enumerate(labels) if label == block)
                    for block in range(maximum + 1)
                )
            )
            return
        for label in range(maximum + 2):
            labels[position] = label
            extend(position + 1, max(maximum, label))

    extend(1, 0)
    return tuple(sorted(found, key=lambda partition: (-len(partition), partition)))


def _preflight(core: CensusSpecCore, bounds: FiniteSubobjectBounds) -> None:
    size = core.carrier_size
    if size > bounds.max_carrier_size:
        raise FiniteSubobjectAnalysisError(
            field="bounds", reason="declared carrier-size bound exceeded"
        )
    subset_count = (1 << size) - 1
    partition_count = _bell_number(size)
    closure_checks = sum(
        len(tuple(index for index in range(size) if mask & (1 << index))) ** core.arity
        for mask in range(1, 1 << size)
    )
    tuple_count = size**core.arity
    compatibility_upper_bound = partition_count * tuple_count * tuple_count
    checks = (
        (subset_count, bounds.max_subset_candidates, "subset-candidate"),
        (partition_count, bounds.max_partition_candidates, "partition-candidate"),
        (closure_checks, bounds.max_closure_checks, "closure-check"),
        (
            compatibility_upper_bound,
            bounds.max_compatibility_checks,
            "compatibility-check",
        ),
    )
    for actual, limit, name in checks:
        if actual > limit:
            raise FiniteSubobjectAnalysisError(
                field="bounds", reason=f"declared {name} bound exceeded"
            )


def analyze_finite_subobjects(
    core: CensusSpecCore,
    outputs: tuple[int, ...],
    *,
    options: FiniteSubobjectBounds | None = None,
) -> FiniteSubobjectAnalysis:
    """Enumerate every nonempty closed subset and compatible partition."""
    checked_core, checked_outputs, bounds = _validated(core, outputs, options)
    _preflight(checked_core, bounds)
    size = checked_core.carrier_size
    arity = checked_core.arity
    closed: list[tuple[int, ...]] = []
    closure_checks = 0
    for mask in range(1, 1 << size):
        subset = tuple(index for index in range(size) if mask & (1 << index))
        is_closed = True
        for arguments in product(subset, repeat=arity):
            is_closed &= _apply(checked_outputs, size, arguments) in subset
            closure_checks += 1
        if is_closed:
            closed.append(subset)

    congruences: list[Partition] = []
    compatibility_checks = 0
    carrier_tuples = tuple(product(range(size), repeat=arity))
    partitions = _partitions(size)
    for partition in partitions:
        classes = {
            element: block_index
            for block_index, block in enumerate(partition)
            for element in block
        }
        compatible = True
        for left in carrier_tuples:
            for right in carrier_tuples:
                if all(
                    classes[left[index]] == classes[right[index]]
                    for index in range(arity)
                ):
                    compatible &= (
                        classes[_apply(checked_outputs, size, left)]
                        == classes[_apply(checked_outputs, size, right)]
                    )
                    compatibility_checks += 1
        if compatible:
            congruences.append(partition)

    value = object.__new__(FiniteSubobjectAnalysis)
    for field, item in (
        ("core", checked_core),
        ("outputs", checked_outputs),
        ("bounds", bounds),
        ("substructures", tuple(closed)),
        ("congruences", tuple(congruences)),
        ("substructure_status", "complete"),
        ("congruence_status", "complete"),
        ("ideal_status", "unsupported_no_module"),
        ("derivation_status", "unsupported_no_module"),
        (
            "hypotheses",
            (
                "one total operation on a finite carrier",
                "substructures are nonempty operation-closed subsets",
                "congruences are operation-compatible equivalence relations",
                "no additive or scalar module structure is present",
            ),
        ),
        ("subset_candidates_examined", (1 << size) - 1),
        ("closure_assignments_checked", closure_checks),
        ("partition_candidates_examined", len(partitions)),
        ("compatibility_pairs_checked", compatibility_checks),
        ("complete", True),
        ("algorithm", "anyalgebra.finite_carrier_subobjects_exhaustive"),
        ("algorithm_version", 1),
        ("semantic_hash", SemanticHash("sha256", "0" * 64)),
    ):
        object.__setattr__(value, field, item)
    object.__setattr__(value, "semantic_hash", _semantic_hash(_body(value)))
    return value


def finite_carrier_ideals(core: CensusSpecCore, outputs: tuple[int, ...]) -> NoReturn:
    """Reject linear ideals on a carrier lacking module operations."""
    _validated(core, outputs, None)
    raise UnsupportedCapability("linear-ideals", context="bare finite carrier")


def finite_carrier_derivations(
    core: CensusSpecCore, outputs: tuple[int, ...]
) -> NoReturn:
    """Reject linear derivations on a carrier lacking module operations."""
    _validated(core, outputs, None)
    raise UnsupportedCapability("linear-derivations", context="bare finite carrier")


def _bounds_body(value: FiniteSubobjectBounds) -> dict[str, int]:
    return {
        "maxCarrierSize": value.max_carrier_size,
        "maxClosureChecks": value.max_closure_checks,
        "maxCompatibilityChecks": value.max_compatibility_checks,
        "maxPartitionCandidates": value.max_partition_candidates,
        "maxSubsetCandidates": value.max_subset_candidates,
    }


def _body(value: FiniteSubobjectAnalysis) -> dict[str, object]:
    return {
        "algorithm": value.algorithm,
        "algorithmVersion": value.algorithm_version,
        "arity": value.core.arity,
        "bounds": _bounds_body(value.bounds),
        "carrierSize": value.core.carrier_size,
        "closureAssignmentsChecked": value.closure_assignments_checked,
        "complete": value.complete,
        "congruenceStatus": value.congruence_status,
        "congruences": [[list(block) for block in item] for item in value.congruences],
        "derivationStatus": value.derivation_status,
        "idealStatus": value.ideal_status,
        "hypotheses": list(value.hypotheses),
        "outputs": list(value.outputs),
        "partitionCandidatesExamined": value.partition_candidates_examined,
        "compatibilityPairsChecked": value.compatibility_pairs_checked,
        "schemaType": "anyalgebra.census.finite_subobject_analysis",
        "schemaVersion": _SCHEMA_VERSION,
        "subsetCandidatesExamined": value.subset_candidates_examined,
        "substructures": [list(item) for item in value.substructures],
        "substructureStatus": value.substructure_status,
    }


def finite_subobject_record(value: FiniteSubobjectAnalysis) -> dict[str, object]:
    if type(value) is not FiniteSubobjectAnalysis:
        raise FiniteSubobjectAnalysisError(
            field="subobjects", reason="must be exact FiniteSubobjectAnalysis"
        )
    expected = analyze_finite_subobjects(
        value.core, value.outputs, options=value.bounds
    )
    if value != expected:
        raise FiniteSubobjectAnalysisError(field="subobjects", reason="content drift")
    body = _body(value)
    semantic_hash = _semantic_hash(body)
    if semantic_hash != value.semantic_hash:
        raise FiniteSubobjectAnalysisError(
            field="semantic_hash", reason="content drift"
        )
    return {
        **body,
        "contentHash": {
            "algorithm": semantic_hash.algorithm,
            "digest": semantic_hash.digest,
        },
    }


def finite_subobject_canonical_bytes(value: FiniteSubobjectAnalysis) -> bytes:
    return _encoded(finite_subobject_record(value))


__all__ = (
    "FiniteSubobjectAnalysis",
    "FiniteSubobjectAnalysisError",
    "FiniteSubobjectBounds",
    "Partition",
    "analyze_finite_subobjects",
    "finite_carrier_derivations",
    "finite_carrier_ideals",
    "finite_subobject_canonical_bytes",
    "finite_subobject_record",
)
