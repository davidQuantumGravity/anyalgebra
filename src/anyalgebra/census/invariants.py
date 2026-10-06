"""Exact, rejection-only invariant snapshots for finite operation tables."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import SemanticHash

from .equivalence import EquivalencePolicy
from .partition import compute_invariant_partition
from .spec import CensusSpecCore
from .table_codes import OperationTableCodeError, rank_operation_table


_SCHEMA_VERSION = 1


class InvariantComparisonError(AnyAlgebraError, ValueError):
    """A rejection-only invariant input or retained record was invalid."""

    def __init__(self, *, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid invariant comparison {field}: {reason}")


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


def _hash_record(value: SemanticHash) -> dict[str, str]:
    return {"algorithm": value.algorithm, "digest": value.digest}


def _flat_binary(outputs: tuple[int, ...], size: int, left: int, right: int) -> int:
    return outputs[left * size + right]


def _law_profile(
    core: CensusSpecCore, outputs: tuple[int, ...]
) -> tuple[tuple[str, bool | int], ...]:
    size = core.carrier_size
    idempotent_applicable = core.arity > 0
    idempotent = idempotent_applicable and all(
        outputs[sum(element * size**power for power in range(core.arity))] == element
        for element in range(size)
    )
    binary = core.arity == 2
    commutative = binary and all(
        _flat_binary(outputs, size, left, right)
        == _flat_binary(outputs, size, right, left)
        for left in range(size)
        for right in range(size)
    )
    associative = binary and all(
        _flat_binary(
            outputs,
            size,
            _flat_binary(outputs, size, left, middle),
            right,
        )
        == _flat_binary(
            outputs,
            size,
            left,
            _flat_binary(outputs, size, middle, right),
        )
        for left in range(size)
        for middle in range(size)
        for right in range(size)
    )
    left_identities = 0
    right_identities = 0
    two_sided_identities = 0
    if binary:
        for candidate in range(size):
            left = all(
                _flat_binary(outputs, size, candidate, item) == item
                for item in range(size)
            )
            right = all(
                _flat_binary(outputs, size, item, candidate) == item
                for item in range(size)
            )
            left_identities += int(left)
            right_identities += int(right)
            two_sided_identities += int(left and right)
    return (
        ("associative", associative),
        ("associative_applicable", binary),
        ("commutative", commutative),
        ("commutative_applicable", binary),
        ("idempotent", idempotent),
        ("idempotent_applicable", idempotent_applicable),
        ("left_identity_count", left_identities),
        ("right_identity_count", right_identities),
        ("two_sided_identity_count", two_sided_identities),
    )


def _sort_blocks(
    policy: EquivalencePolicy,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((block.name, block.elements) for block in policy.sort_blocks)


def _block_color_sizes(
    policy: EquivalencePolicy, colors: tuple[int, ...]
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    result: list[tuple[str, tuple[int, ...]]] = []
    for block in policy.sort_blocks:
        counts = tuple(
            sorted(
                sum(colors[element] == color for element in block.elements)
                for color in set(colors[element] for element in block.elements)
            )
        )
        result.append((block.name, counts))
    return tuple(result)


def _snapshot_body(value: TableInvariantSnapshot) -> dict[str, object]:
    return {
        "arity": value.arity,
        "blockColorClassSizes": value.block_color_class_sizes,
        "carrierSize": value.carrier_size,
        "colorClassSizes": value.color_class_sizes,
        "distinguishedColorProfile": value.distinguished_color_profile,
        "distinguishedElements": value.distinguished_elements,
        "fixedElements": value.fixed_elements,
        "lawProfile": value.law_profile,
        "outputMultiplicityHistogram": value.output_multiplicity_histogram,
        "policyKind": value.policy_kind,
        "schemaType": "anyalgebra.census.table_invariant_snapshot",
        "schemaVersion": _SCHEMA_VERSION,
        "sortBlocks": value.sort_blocks,
    }


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class TableInvariantSnapshot:
    """Independently computed exact data that may reject, but never identify."""

    core: CensusSpecCore
    equivalence: EquivalencePolicy
    outputs: tuple[int, ...]
    carrier_size: int
    arity: int
    distinguished_elements: tuple[tuple[str, int], ...]
    policy_kind: str
    fixed_elements: tuple[int, ...]
    sort_blocks: tuple[tuple[str, tuple[int, ...]], ...]
    law_profile: tuple[tuple[str, bool | int], ...]
    output_multiplicity_histogram: tuple[int, ...]
    color_class_sizes: tuple[int, ...]
    block_color_class_sizes: tuple[tuple[str, tuple[int, ...]], ...]
    distinguished_color_profile: tuple[tuple[str, int], ...]
    semantic_hash: SemanticHash
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise InvariantComparisonError(field="snapshot", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("TableInvariantSnapshot cannot be subclassed")

    @classmethod
    def _create(
        cls,
        core: CensusSpecCore,
        outputs: tuple[int, ...],
        policy: EquivalencePolicy,
    ) -> TableInvariantSnapshot:
        partition = compute_invariant_partition(core, outputs, policy)
        value = object.__new__(TableInvariantSnapshot)
        for field, item in (
            ("core", core),
            ("equivalence", policy),
            ("outputs", outputs),
            ("carrier_size", core.carrier_size),
            ("arity", core.arity),
            ("distinguished_elements", core.distinguished_elements),
            ("policy_kind", policy.kind),
            ("fixed_elements", policy.fixed_elements),
            ("sort_blocks", _sort_blocks(policy)),
            ("law_profile", _law_profile(core, outputs)),
            (
                "output_multiplicity_histogram",
                tuple(
                    sorted(
                        outputs.count(element) for element in range(core.carrier_size)
                    )
                ),
            ),
            (
                "color_class_sizes",
                tuple(sorted(len(elements) for _, elements in partition.blocks)),
            ),
            (
                "block_color_class_sizes",
                _block_color_sizes(policy, partition.colors),
            ),
            (
                "distinguished_color_profile",
                tuple(
                    (name, partition.colors[element])
                    for name, element in core.distinguished_elements
                ),
            ),
            ("semantic_hash", SemanticHash("sha256", "0" * 64)),
        ):
            object.__setattr__(value, field, item)
        object.__setattr__(
            value, "semantic_hash", _semantic_hash(_snapshot_body(value))
        )
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is TableInvariantSnapshot and (
            self.semantic_hash,
            self.outputs,
            self.core,
            self.equivalence,
        ) == (
            other.semantic_hash,
            other.outputs,
            other.core,
            other.equivalence,
        )

    def __repr__(self) -> str:
        return (
            "TableInvariantSnapshot("
            f"semantic_hash={str(self.semantic_hash)!r}, "
            f"carrier_size={self.carrier_size}, arity={self.arity})"
        )


def _validate_input(
    core: object,
    outputs: object,
    policy: object,
    *,
    side: str,
) -> tuple[CensusSpecCore, tuple[int, ...], EquivalencePolicy]:
    if type(core) is not CensusSpecCore:
        raise InvariantComparisonError(
            field=f"{side}_core", reason="must be an exact CensusSpecCore"
        )
    if core.candidate_count == 0:
        raise InvariantComparisonError(
            field=f"{side}_outputs", reason="the declared core has no operation tables"
        )
    if type(policy) is not EquivalencePolicy or policy.core is not core:
        raise InvariantComparisonError(
            field=f"{side}_equivalence",
            reason="must belong to the literal table core",
        )
    if type(outputs) is not tuple:
        raise InvariantComparisonError(
            field=f"{side}_outputs", reason="must be an exact tuple"
        )
    try:
        rank_operation_table(core, outputs)
    except OperationTableCodeError as error:
        raise InvariantComparisonError(
            field=f"{side}_outputs", reason="is not a valid table for the core"
        ) from error
    return core, outputs, policy


def compute_table_invariants(
    core: CensusSpecCore,
    outputs: tuple[int, ...],
    equivalence: EquivalencePolicy,
) -> TableInvariantSnapshot:
    """Recompute every rejection field from one exact table and policy."""
    checked_core, checked_outputs, checked_policy = _validate_input(
        core, outputs, equivalence, side="table"
    )
    return TableInvariantSnapshot._create(checked_core, checked_outputs, checked_policy)


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class InvariantCheck:
    """One named independently recomputed equality test."""

    name: str
    left_value: object
    right_value: object
    matches: bool
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise InvariantComparisonError(field="check", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("InvariantCheck cannot be subclassed")

    @classmethod
    def _create(cls, name: str, left: object, right: object) -> InvariantCheck:
        value = object.__new__(InvariantCheck)
        object.__setattr__(value, "name", name)
        object.__setattr__(value, "left_value", left)
        object.__setattr__(value, "right_value", right)
        object.__setattr__(value, "matches", left == right)
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is InvariantCheck and (
            self.name,
            self.left_value,
            self.right_value,
            self.matches,
        ) == (other.name, other.left_value, other.right_value, other.matches)


_CHECK_FIELDS = (
    "carrier_size",
    "arity",
    "distinguished_elements",
    "policy_kind",
    "fixed_elements",
    "sort_blocks",
    "law_profile",
    "output_multiplicity_histogram",
    "color_class_sizes",
    "block_color_class_sizes",
    "distinguished_color_profile",
)


def _checks(
    left: TableInvariantSnapshot, right: TableInvariantSnapshot
) -> tuple[InvariantCheck, ...]:
    return tuple(
        InvariantCheck._create(name, getattr(left, name), getattr(right, name))
        for name in _CHECK_FIELDS
    )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class InvariantComparison:
    """A rejection or no-rejection result that never claims isomorphism."""

    left: TableInvariantSnapshot
    right: TableInvariantSnapshot
    checks: tuple[InvariantCheck, ...]
    rejected: bool
    first_mismatch: str | None
    status: str
    evidence_role: str
    complete_invariant: bool
    semantic_hash: SemanticHash
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise InvariantComparisonError(field="comparison", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("InvariantComparison cannot be subclassed")

    @classmethod
    def _create(
        cls, left: TableInvariantSnapshot, right: TableInvariantSnapshot
    ) -> InvariantComparison:
        checks = _checks(left, right)
        mismatch = next((check.name for check in checks if not check.matches), None)
        value = object.__new__(InvariantComparison)
        for field, item in (
            ("left", left),
            ("right", right),
            ("checks", checks),
            ("rejected", mismatch is not None),
            ("first_mismatch", mismatch),
            ("status", "rejected" if mismatch is not None else "not_rejected"),
            ("evidence_role", "rejection_only"),
            ("complete_invariant", False),
            ("semantic_hash", SemanticHash("sha256", "0" * 64)),
        ):
            object.__setattr__(value, field, item)
        object.__setattr__(
            value, "semantic_hash", _semantic_hash(_comparison_body(value))
        )
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is InvariantComparison and (
            self.left,
            self.right,
            self.checks,
            self.rejected,
            self.first_mismatch,
            self.status,
            self.evidence_role,
            self.complete_invariant,
            self.semantic_hash,
        ) == (
            other.left,
            other.right,
            other.checks,
            other.rejected,
            other.first_mismatch,
            other.status,
            other.evidence_role,
            other.complete_invariant,
            other.semantic_hash,
        )

    def __repr__(self) -> str:
        return (
            "InvariantComparison("
            f"semantic_hash={str(self.semantic_hash)!r}, status={self.status!r}, "
            f"first_mismatch={self.first_mismatch!r})"
        )


def _comparison_body(value: InvariantComparison) -> dict[str, object]:
    return {
        "checks": [
            {
                "left": check.left_value,
                "matches": check.matches,
                "name": check.name,
                "right": check.right_value,
            }
            for check in value.checks
        ],
        "completeInvariant": value.complete_invariant,
        "evidenceRole": value.evidence_role,
        "firstMismatch": value.first_mismatch,
        "left": _snapshot_body(value.left),
        "rejected": value.rejected,
        "right": _snapshot_body(value.right),
        "schemaType": "anyalgebra.census.invariant_comparison",
        "schemaVersion": _SCHEMA_VERSION,
        "status": value.status,
    }


def compare_table_invariants(
    left_core: CensusSpecCore,
    left_outputs: tuple[int, ...],
    left_equivalence: EquivalencePolicy,
    right_core: CensusSpecCore,
    right_outputs: tuple[int, ...],
    right_equivalence: EquivalencePolicy,
) -> InvariantComparison:
    """Independently compute both sides and return only rejection evidence."""
    checked_left = _validate_input(
        left_core, left_outputs, left_equivalence, side="left"
    )
    checked_right = _validate_input(
        right_core, right_outputs, right_equivalence, side="right"
    )
    left = TableInvariantSnapshot._create(*checked_left)
    right = TableInvariantSnapshot._create(*checked_right)
    return InvariantComparison._create(left, right)


def _recompute_snapshot(value: TableInvariantSnapshot) -> TableInvariantSnapshot:
    if type(value) is not TableInvariantSnapshot:
        raise InvariantComparisonError(
            field="snapshot", reason="must be an exact TableInvariantSnapshot"
        )
    return compute_table_invariants(value.core, value.outputs, value.equivalence)


def invariant_comparison_record(value: InvariantComparison) -> dict[str, object]:
    """Return a fresh canonical record after independently recomputing both sides."""
    if type(value) is not InvariantComparison:
        raise InvariantComparisonError(
            field="comparison", reason="must be an exact InvariantComparison"
        )
    left = _recompute_snapshot(value.left)
    right = _recompute_snapshot(value.right)
    expected = InvariantComparison._create(left, right)
    if value != expected:
        raise InvariantComparisonError(field="comparison", reason="content drift")
    body = _comparison_body(value)
    semantic_hash = _semantic_hash(body)
    if semantic_hash != value.semantic_hash:
        raise InvariantComparisonError(field="semantic_hash", reason="content drift")
    return {**body, "contentHash": _hash_record(semantic_hash)}


def invariant_comparison_canonical_bytes(value: InvariantComparison) -> bytes:
    """Return deterministic UTF-8 canonical JSON bytes for one checked result."""
    return _encoded(invariant_comparison_record(value))


__all__ = (
    "InvariantCheck",
    "InvariantComparison",
    "InvariantComparisonError",
    "TableInvariantSnapshot",
    "compare_table_invariants",
    "compute_table_invariants",
    "invariant_comparison_canonical_bytes",
    "invariant_comparison_record",
)
