"""Disjoint, proof-shaped outcomes for finite-table isomorphism comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import TypeAlias

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import SemanticHash

from .canonical import CanonicalLabel
from .certificates import (
    CanonicalCertificate,
    CanonicalCertificateError,
    verify_canonical_certificate,
)
from .equivalence import EquivalencePolicy
from .invariants import (
    InvariantCheck,
    InvariantComparison,
    compare_table_invariants,
    invariant_comparison_record,
)
from .partition import compute_invariant_partition
from .permutations import generate_allowed_permutations
from .spec import CensusSpecCore
from .table_codes import OperationTableCodeError, rank_operation_table
from .transport import TableTransportCertificate, transport_operation_table


_SCHEMA_VERSION = 1
_BOUND_NAMES = frozenset(
    ("max_observed_milliseconds", "max_permutations", "max_work_units")
)


class IsomorphismOutcomeError(AnyAlgebraError, ValueError):
    """An outcome lacked the exact evidence required by its status."""

    def __init__(self, *, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid isomorphism outcome {field}: {reason}")


def _hash_record(value: SemanticHash) -> dict[str, str]:
    return {"algorithm": value.algorithm, "digest": value.digest}


def _policy_record(value: EquivalencePolicy) -> dict[str, object]:
    return {
        "fixedElements": list(value.fixed_elements),
        "kind": value.kind,
        "sortBlocks": [
            {"elements": list(block.elements), "name": block.name}
            for block in value.sort_blocks
        ],
    }


def _encoded(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _body_hash(body: dict[str, object]) -> SemanticHash:
    return SemanticHash("sha256", hashlib.sha256(_encoded(body)).hexdigest())


def _valid_table(core: CensusSpecCore, outputs: tuple[int, ...], field: str) -> None:
    if type(outputs) is not tuple:
        raise IsomorphismOutcomeError(field=field, reason="must be an exact tuple")
    try:
        rank_operation_table(core, outputs)
    except OperationTableCodeError as error:
        raise IsomorphismOutcomeError(
            field=field, reason="is not a valid table for the outcome core"
        ) from error


def _common_body(
    *,
    schema_type: str,
    status: str,
    complete: bool,
    core: CensusSpecCore,
    equivalence: EquivalencePolicy,
) -> dict[str, object]:
    return {
        "arity": core.arity,
        "carrierSize": core.carrier_size,
        "complete": complete,
        "equivalence": _policy_record(equivalence),
        "schemaType": schema_type,
        "schemaVersion": _SCHEMA_VERSION,
        "status": status,
    }


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class Isomorphic:
    """Positive result requiring one replayable operation-preserving bijection."""

    core: CensusSpecCore
    equivalence: EquivalencePolicy
    status: str
    complete: bool
    left_outputs: tuple[int, ...]
    right_outputs: tuple[int, ...]
    permutation_images: tuple[int, ...]
    witness: TableTransportCertificate
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise IsomorphismOutcomeError(field="isomorphic", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("Isomorphic cannot be subclassed")

    @classmethod
    def from_transport(cls, witness: TableTransportCertificate) -> Isomorphic:
        """Construct only from an exact full-cell transport certificate."""
        if cls is not Isomorphic:
            raise IsomorphismOutcomeError(
                field="isomorphic", reason="factory requires exact type"
            )
        if type(witness) is not TableTransportCertificate:
            raise IsomorphismOutcomeError(
                field="witness",
                reason="must be an exact TableTransportCertificate",
            )
        if not witness.verify_cells():
            raise IsomorphismOutcomeError(
                field="witness", reason="operation-preserving replay failed"
            )
        value = object.__new__(Isomorphic)
        for field, item in (
            ("core", witness.core),
            ("equivalence", witness.permutation.policy),
            ("status", "isomorphic"),
            ("complete", True),
            ("left_outputs", witness.source_outputs),
            ("right_outputs", witness.target_outputs),
            ("permutation_images", witness.permutation.images),
            ("witness", witness),
            ("semantic_hash", SemanticHash("sha256", "0" * 64)),
        ):
            object.__setattr__(value, field, item)
        object.__setattr__(value, "semantic_hash", _body_hash(_body(value)))
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is Isomorphic and (
            self.core,
            self.equivalence,
            self.left_outputs,
            self.right_outputs,
            self.permutation_images,
            self.semantic_hash,
        ) == (
            other.core,
            other.equivalence,
            other.left_outputs,
            other.right_outputs,
            other.permutation_images,
            other.semantic_hash,
        )

    def __hash__(self) -> int:
        return int.from_bytes(
            bytes.fromhex(self.semantic_hash.digest[:16]), "big", signed=True
        )

    def __repr__(self) -> str:
        return f"Isomorphic(semantic_hash={str(self.semantic_hash)!r})"


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class NonIsomorphic:
    """Complete negative result from unequal exhaustive canonical identifiers."""

    core: CensusSpecCore
    equivalence: EquivalencePolicy
    status: str
    complete: bool
    left_label: CanonicalLabel
    right_label: CanonicalLabel
    left_canonical_id: SemanticHash
    right_canonical_id: SemanticHash
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise IsomorphismOutcomeError(field="non_isomorphic", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("NonIsomorphic cannot be subclassed")

    @classmethod
    def from_canonical_labels(
        cls, left: CanonicalLabel, right: CanonicalLabel
    ) -> NonIsomorphic:
        """Construct from two checked exhaustive labels with distinct IDs."""
        if cls is not NonIsomorphic:
            raise IsomorphismOutcomeError(
                field="non_isomorphic", reason="factory requires exact type"
            )
        _validate_negative_labels(left, right)
        value = object.__new__(NonIsomorphic)
        for field, item in (
            ("core", left.core),
            ("equivalence", left.equivalence),
            ("status", "non_isomorphic"),
            ("complete", True),
            ("left_label", left),
            ("right_label", right),
            ("left_canonical_id", left.canonical_id),
            ("right_canonical_id", right.canonical_id),
            ("semantic_hash", SemanticHash("sha256", "0" * 64)),
        ):
            object.__setattr__(value, field, item)
        object.__setattr__(value, "semantic_hash", _body_hash(_body(value)))
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is NonIsomorphic and (
            self.core,
            self.equivalence,
            self.left_canonical_id,
            self.right_canonical_id,
            self.semantic_hash,
        ) == (
            other.core,
            other.equivalence,
            other.left_canonical_id,
            other.right_canonical_id,
            other.semantic_hash,
        )

    def __hash__(self) -> int:
        return int.from_bytes(
            bytes.fromhex(self.semantic_hash.digest[:16]), "big", signed=True
        )

    def __repr__(self) -> str:
        return f"NonIsomorphic(semantic_hash={str(self.semantic_hash)!r})"


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class Inconclusive:
    """Bounded comparison result that structurally retains unfinished work."""

    core: CensusSpecCore
    equivalence: EquivalencePolicy
    status: str
    complete: bool
    left_outputs: tuple[int, ...]
    right_outputs: tuple[int, ...]
    bound_name: str
    bound_limit: int
    examined_count: int
    remaining_count: int
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise IsomorphismOutcomeError(field="inconclusive", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("Inconclusive cannot be subclassed")

    @classmethod
    def create(
        cls,
        *,
        core: CensusSpecCore,
        equivalence: EquivalencePolicy,
        left_outputs: tuple[int, ...],
        right_outputs: tuple[int, ...],
        bound_name: str,
        bound_limit: int,
        examined_count: int,
        remaining_count: int,
    ) -> Inconclusive:
        if cls is not Inconclusive:
            raise IsomorphismOutcomeError(
                field="inconclusive", reason="factory requires exact type"
            )
        _validate_inconclusive(
            core,
            equivalence,
            left_outputs,
            right_outputs,
            bound_name,
            bound_limit,
            examined_count,
            remaining_count,
        )
        value = object.__new__(Inconclusive)
        for field, item in (
            ("core", core),
            ("equivalence", equivalence),
            ("status", "bounded_inconclusive"),
            ("complete", False),
            ("left_outputs", left_outputs),
            ("right_outputs", right_outputs),
            ("bound_name", bound_name),
            ("bound_limit", bound_limit),
            ("examined_count", examined_count),
            ("remaining_count", remaining_count),
            ("semantic_hash", SemanticHash("sha256", "0" * 64)),
        ):
            object.__setattr__(value, field, item)
        object.__setattr__(value, "semantic_hash", _body_hash(_body(value)))
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is Inconclusive and (
            self.core,
            self.equivalence,
            self.left_outputs,
            self.right_outputs,
            self.bound_name,
            self.bound_limit,
            self.examined_count,
            self.remaining_count,
            self.semantic_hash,
        ) == (
            other.core,
            other.equivalence,
            other.left_outputs,
            other.right_outputs,
            other.bound_name,
            other.bound_limit,
            other.examined_count,
            other.remaining_count,
            other.semantic_hash,
        )

    def __hash__(self) -> int:
        return int.from_bytes(
            bytes.fromhex(self.semantic_hash.digest[:16]), "big", signed=True
        )

    def __repr__(self) -> str:
        return (
            "Inconclusive("
            f"semantic_hash={str(self.semantic_hash)!r}, "
            f"bound_name={self.bound_name!r}, remaining_count={self.remaining_count})"
        )


IsomorphismOutcome: TypeAlias = Isomorphic | NonIsomorphic | Inconclusive


def _validate_negative_labels(left: object, right: object) -> None:
    if type(left) is not CanonicalLabel or type(right) is not CanonicalLabel:
        raise IsomorphismOutcomeError(
            field="labels", reason="must be exact CanonicalLabel values"
        )
    if left.core is not right.core or left.equivalence is not right.equivalence:
        raise IsomorphismOutcomeError(
            field="equivalence",
            reason="labels must share one literal comparison policy",
        )
    try:
        verify_canonical_certificate(CanonicalCertificate.from_label(left))
        verify_canonical_certificate(CanonicalCertificate.from_label(right))
    except CanonicalCertificateError as error:
        raise IsomorphismOutcomeError(
            field="labels", reason="canonical certificate replay failed"
        ) from error
    if left.canonical_id == right.canonical_id:
        raise IsomorphismOutcomeError(
            field="canonical_id", reason="equal identifiers are not negative evidence"
        )


def _exact_count(value: object, field: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive):
        qualifier = "positive" if positive else "non-negative"
        raise IsomorphismOutcomeError(
            field=field, reason=f"must be a {qualifier} exact built-in int"
        )
    return value


def _validate_inconclusive(
    core: object,
    equivalence: object,
    left_outputs: tuple[int, ...],
    right_outputs: tuple[int, ...],
    bound_name: object,
    bound_limit: object,
    examined_count: object,
    remaining_count: object,
) -> None:
    if type(core) is not CensusSpecCore:
        raise IsomorphismOutcomeError(
            field="core", reason="must be an exact CensusSpecCore"
        )
    if core.candidate_count == 0:
        raise IsomorphismOutcomeError(
            field="tables", reason="the declared core has no operation tables"
        )
    if type(equivalence) is not EquivalencePolicy or equivalence.core is not core:
        raise IsomorphismOutcomeError(
            field="equivalence", reason="must belong to the literal outcome core"
        )
    _valid_table(core, left_outputs, "left_outputs")
    _valid_table(core, right_outputs, "right_outputs")
    if type(bound_name) is not str or bound_name not in _BOUND_NAMES:
        raise IsomorphismOutcomeError(
            field="bound_name", reason="must be an allow-listed bound"
        )
    limit = _exact_count(bound_limit, "bound_limit")
    examined = _exact_count(examined_count, "examined_count")
    _exact_count(remaining_count, "remaining_count", positive=True)
    if examined != limit:
        raise IsomorphismOutcomeError(
            field="examined_count", reason="must equal the exhausted bound limit"
        )


def _body(value: IsomorphismOutcome) -> dict[str, object]:
    if type(value) is Isomorphic:
        if (
            type(value.witness) is not TableTransportCertificate
            or not value.witness.verify_cells()
            or value.core is not value.witness.core
            or value.equivalence is not value.witness.permutation.policy
            or value.left_outputs != value.witness.source_outputs
            or value.right_outputs != value.witness.target_outputs
            or value.permutation_images != value.witness.permutation.images
            or value.status != "isomorphic"
            or value.complete is not True
        ):
            raise IsomorphismOutcomeError(
                field="witness", reason="positive result content drift"
            )
        return {
            **_common_body(
                schema_type="anyalgebra.census.isomorphism.isomorphic",
                status=value.status,
                complete=value.complete,
                core=value.core,
                equivalence=value.equivalence,
            ),
            "leftOutputs": list(value.left_outputs),
            "proofKind": "verified_operation_preserving_bijection",
            "rightOutputs": list(value.right_outputs),
            "witnessPermutation": list(value.permutation_images),
        }
    if type(value) is NonIsomorphic:
        _validate_negative_labels(value.left_label, value.right_label)
        if (
            value.core is not value.left_label.core
            or value.equivalence is not value.left_label.equivalence
            or value.left_canonical_id != value.left_label.canonical_id
            or value.right_canonical_id != value.right_label.canonical_id
            or value.status != "non_isomorphic"
            or value.complete is not True
        ):
            raise IsomorphismOutcomeError(
                field="labels", reason="negative result content drift"
            )
        return {
            **_common_body(
                schema_type="anyalgebra.census.isomorphism.non_isomorphic",
                status=value.status,
                complete=value.complete,
                core=value.core,
                equivalence=value.equivalence,
            ),
            "leftCanonicalId": _hash_record(value.left_canonical_id),
            "proofKind": "exhaustive_canonical_identifier_mismatch",
            "rightCanonicalId": _hash_record(value.right_canonical_id),
        }
    if type(value) is Inconclusive:
        _validate_inconclusive(
            value.core,
            value.equivalence,
            value.left_outputs,
            value.right_outputs,
            value.bound_name,
            value.bound_limit,
            value.examined_count,
            value.remaining_count,
        )
        if value.status != "bounded_inconclusive" or value.complete is not False:
            raise IsomorphismOutcomeError(
                field="status", reason="inconclusive result content drift"
            )
        return {
            **_common_body(
                schema_type="anyalgebra.census.isomorphism.inconclusive",
                status=value.status,
                complete=value.complete,
                core=value.core,
                equivalence=value.equivalence,
            ),
            "bound": {
                "examinedCount": value.examined_count,
                "limit": value.bound_limit,
                "name": value.bound_name,
                "remainingCount": value.remaining_count,
            },
            "leftOutputs": list(value.left_outputs),
            "proofKind": "bounded_search_frontier",
            "rightOutputs": list(value.right_outputs),
        }
    raise IsomorphismOutcomeError(
        field="outcome", reason="must be an exact isomorphism outcome"
    )


def isomorphism_outcome_record(value: IsomorphismOutcome) -> dict[str, object]:
    """Return a fresh type-specific record after revalidating its evidence."""
    body = _body(value)
    semantic_hash = _body_hash(body)
    if semantic_hash != value.semantic_hash:
        raise IsomorphismOutcomeError(field="semantic_hash", reason="content drift")
    return {**body, "contentHash": _hash_record(semantic_hash)}


def isomorphism_outcome_canonical_bytes(value: IsomorphismOutcome) -> bytes:
    """Return exact deterministic UTF-8 JSON bytes for one checked outcome."""
    return _encoded(isomorphism_outcome_record(value))


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class IsomorphismSearchResult:
    """Deterministic exhaustive search receipt with an optional positive map."""

    core: CensusSpecCore
    equivalence: EquivalencePolicy
    left_outputs: tuple[int, ...]
    right_outputs: tuple[int, ...]
    found: bool
    complete: bool
    witness: TableTransportCertificate | None
    outcome: Isomorphic | None
    total_allowed_count: int
    color_compatible_count: int
    partition_rejected_count: int
    permutations_examined: int
    stop_reason: str
    strategy: str
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise IsomorphismOutcomeError(field="search", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("IsomorphismSearchResult cannot be subclassed")

    @classmethod
    def _create(
        cls,
        *,
        core: CensusSpecCore,
        equivalence: EquivalencePolicy,
        left_outputs: tuple[int, ...],
        right_outputs: tuple[int, ...],
        witness: TableTransportCertificate | None,
        total_allowed_count: int,
        color_compatible_count: int,
        permutations_examined: int,
    ) -> IsomorphismSearchResult:
        found = witness is not None
        outcome = Isomorphic.from_transport(witness) if witness is not None else None
        stop_reason = (
            "witness_found" if found else "exhausted_color_compatible_bijections"
        )
        value = object.__new__(IsomorphismSearchResult)
        for field, item in (
            ("core", core),
            ("equivalence", equivalence),
            ("left_outputs", left_outputs),
            ("right_outputs", right_outputs),
            ("found", found),
            ("complete", True),
            ("witness", witness),
            ("outcome", outcome),
            ("total_allowed_count", total_allowed_count),
            ("color_compatible_count", color_compatible_count),
            (
                "partition_rejected_count",
                total_allowed_count - color_compatible_count,
            ),
            ("permutations_examined", permutations_examined),
            ("stop_reason", stop_reason),
            ("strategy", "color_compatible_lexicographic_v1"),
            ("semantic_hash", SemanticHash("sha256", "0" * 64)),
        ):
            object.__setattr__(value, field, item)
        object.__setattr__(value, "semantic_hash", _body_hash(_search_body(value)))
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is IsomorphismSearchResult and (
            self.core,
            self.equivalence,
            self.left_outputs,
            self.right_outputs,
            self.found,
            self.complete,
            self.witness,
            self.outcome,
            self.total_allowed_count,
            self.color_compatible_count,
            self.partition_rejected_count,
            self.permutations_examined,
            self.stop_reason,
            self.strategy,
            self.semantic_hash,
        ) == (
            other.core,
            other.equivalence,
            other.left_outputs,
            other.right_outputs,
            other.found,
            other.complete,
            other.witness,
            other.outcome,
            other.total_allowed_count,
            other.color_compatible_count,
            other.partition_rejected_count,
            other.permutations_examined,
            other.stop_reason,
            other.strategy,
            other.semantic_hash,
        )

    def __repr__(self) -> str:
        return (
            "IsomorphismSearchResult("
            f"semantic_hash={str(self.semantic_hash)!r}, found={self.found}, "
            f"permutations_examined={self.permutations_examined})"
        )


def _search_inputs(
    core: object,
    left_outputs: object,
    right_outputs: object,
    equivalence: object,
) -> tuple[CensusSpecCore, tuple[int, ...], tuple[int, ...], EquivalencePolicy]:
    if type(core) is not CensusSpecCore:
        raise IsomorphismOutcomeError(
            field="core", reason="must be an exact CensusSpecCore"
        )
    if core.candidate_count == 0:
        raise IsomorphismOutcomeError(
            field="tables", reason="the declared core has no operation tables"
        )
    if type(equivalence) is not EquivalencePolicy or equivalence.core is not core:
        raise IsomorphismOutcomeError(
            field="equivalence", reason="must belong to the literal search core"
        )
    if type(left_outputs) is not tuple:
        raise IsomorphismOutcomeError(
            field="left_outputs", reason="must be an exact tuple"
        )
    if type(right_outputs) is not tuple:
        raise IsomorphismOutcomeError(
            field="right_outputs", reason="must be an exact tuple"
        )
    _valid_table(core, left_outputs, "left_outputs")
    _valid_table(core, right_outputs, "right_outputs")
    return core, left_outputs, right_outputs, equivalence


def search_isomorphism(
    core: CensusSpecCore,
    left_outputs: tuple[int, ...],
    right_outputs: tuple[int, ...],
    equivalence: EquivalencePolicy,
) -> IsomorphismSearchResult:
    """Return the first lexicographic color-compatible preserving bijection."""
    checked_core, left, right, policy = _search_inputs(
        core, left_outputs, right_outputs, equivalence
    )
    left_partition = compute_invariant_partition(checked_core, left, policy)
    right_partition = compute_invariant_partition(checked_core, right, policy)
    group = generate_allowed_permutations(policy)
    compatible = tuple(
        permutation
        for permutation in group
        if all(
            left_partition.colors[old] == right_partition.colors[permutation(old)]
            for old in range(checked_core.carrier_size)
        )
    )
    witness: TableTransportCertificate | None = None
    examined = 0
    for permutation in compatible:
        examined += 1
        certificate = transport_operation_table(checked_core, left, permutation)
        if certificate.target_outputs == right:
            witness = certificate
            break
    return IsomorphismSearchResult._create(
        core=checked_core,
        equivalence=policy,
        left_outputs=left,
        right_outputs=right,
        witness=witness,
        total_allowed_count=len(group),
        color_compatible_count=len(compatible),
        permutations_examined=examined,
    )


def _search_body(value: IsomorphismSearchResult) -> dict[str, object]:
    status = "isomorphic" if value.found else "search_exhausted_no_map"
    return {
        **_common_body(
            schema_type="anyalgebra.census.isomorphism.search",
            status=status,
            complete=value.complete,
            core=value.core,
            equivalence=value.equivalence,
        ),
        "colorCompatibleCount": value.color_compatible_count,
        "leftOutputs": list(value.left_outputs),
        "outcomeHash": _hash_record(value.outcome.semantic_hash)
        if value.outcome is not None
        else None,
        "partitionRejectedCount": value.partition_rejected_count,
        "permutationsExamined": value.permutations_examined,
        "rightOutputs": list(value.right_outputs),
        "stopReason": value.stop_reason,
        "strategy": value.strategy,
        "totalAllowedCount": value.total_allowed_count,
        "witnessPermutation": list(value.witness.permutation.images)
        if value.witness is not None
        else None,
    }


def isomorphism_search_record(value: IsomorphismSearchResult) -> dict[str, object]:
    """Re-run the deterministic search before returning its canonical record."""
    if type(value) is not IsomorphismSearchResult:
        raise IsomorphismOutcomeError(
            field="search", reason="must be an exact IsomorphismSearchResult"
        )
    expected = search_isomorphism(
        value.core, value.left_outputs, value.right_outputs, value.equivalence
    )
    if value != expected:
        raise IsomorphismOutcomeError(field="search", reason="content drift")
    body = _search_body(value)
    semantic_hash = _body_hash(body)
    if semantic_hash != value.semantic_hash:
        raise IsomorphismOutcomeError(field="semantic_hash", reason="content drift")
    return {**body, "contentHash": _hash_record(semantic_hash)}


def isomorphism_search_canonical_bytes(value: IsomorphismSearchResult) -> bytes:
    """Return deterministic UTF-8 canonical JSON for one replayed search."""
    return _encoded(isomorphism_search_record(value))


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class InvariantNonIsomorphic:
    """Complete rejection from the first exact invariant mismatch."""

    core: CensusSpecCore
    equivalence: EquivalencePolicy
    status: str
    complete: bool
    proof_kind: str
    comparison: InvariantComparison
    mismatch: InvariantCheck
    permutations_examined: int
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise IsomorphismOutcomeError(
            field="invariant_negative", reason="is factory-owned"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("InvariantNonIsomorphic cannot be subclassed")

    @classmethod
    def _create(cls, comparison: InvariantComparison) -> InvariantNonIsomorphic:
        if type(comparison) is not InvariantComparison or not comparison.rejected:
            raise IsomorphismOutcomeError(
                field="invariant_comparison",
                reason="must be an exact rejected comparison",
            )
        mismatch = next(check for check in comparison.checks if not check.matches)
        value = object.__new__(InvariantNonIsomorphic)
        for field, item in (
            ("core", comparison.left.core),
            ("equivalence", comparison.left.equivalence),
            ("status", "non_isomorphic"),
            ("complete", True),
            ("proof_kind", "invariant_mismatch"),
            ("comparison", comparison),
            ("mismatch", mismatch),
            ("permutations_examined", 0),
            ("semantic_hash", SemanticHash("sha256", "0" * 64)),
        ):
            object.__setattr__(value, field, item)
        object.__setattr__(
            value, "semantic_hash", _body_hash(_comparison_result_body(value))
        )
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is InvariantNonIsomorphic and (
            self.comparison,
            self.mismatch,
            self.permutations_examined,
            self.semantic_hash,
        ) == (
            other.comparison,
            other.mismatch,
            other.permutations_examined,
            other.semantic_hash,
        )

    def __hash__(self) -> int:
        return int.from_bytes(
            bytes.fromhex(self.semantic_hash.digest[:16]), "big", signed=True
        )

    def __repr__(self) -> str:
        return (
            "InvariantNonIsomorphic("
            f"semantic_hash={str(self.semantic_hash)!r}, "
            f"mismatch={self.mismatch.name!r})"
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ExhaustiveNonIsomorphic:
    """Complete no-map receipt accounting for the entire allowed group."""

    core: CensusSpecCore
    equivalence: EquivalencePolicy
    status: str
    complete: bool
    proof_kind: str
    search: IsomorphismSearchResult
    total_allowed_count: int
    partition_rejected_count: int
    operation_checked_count: int
    checked_count: int
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise IsomorphismOutcomeError(
            field="exhaustive_negative", reason="is factory-owned"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("ExhaustiveNonIsomorphic cannot be subclassed")

    @classmethod
    def _create(cls, search: IsomorphismSearchResult) -> ExhaustiveNonIsomorphic:
        _validate_exhaustive_search(search)
        value = object.__new__(ExhaustiveNonIsomorphic)
        for field, item in (
            ("core", search.core),
            ("equivalence", search.equivalence),
            ("status", "non_isomorphic"),
            ("complete", True),
            ("proof_kind", "exhaustive_no_map"),
            ("search", search),
            ("total_allowed_count", search.total_allowed_count),
            ("partition_rejected_count", search.partition_rejected_count),
            ("operation_checked_count", search.permutations_examined),
            ("checked_count", search.total_allowed_count),
            ("semantic_hash", SemanticHash("sha256", "0" * 64)),
        ):
            object.__setattr__(value, field, item)
        object.__setattr__(
            value, "semantic_hash", _body_hash(_comparison_result_body(value))
        )
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is ExhaustiveNonIsomorphic and (
            self.search,
            self.total_allowed_count,
            self.partition_rejected_count,
            self.operation_checked_count,
            self.checked_count,
            self.semantic_hash,
        ) == (
            other.search,
            other.total_allowed_count,
            other.partition_rejected_count,
            other.operation_checked_count,
            other.checked_count,
            other.semantic_hash,
        )

    def __hash__(self) -> int:
        return int.from_bytes(
            bytes.fromhex(self.semantic_hash.digest[:16]), "big", signed=True
        )

    def __repr__(self) -> str:
        return (
            "ExhaustiveNonIsomorphic("
            f"semantic_hash={str(self.semantic_hash)!r}, "
            f"checked_count={self.checked_count})"
        )


ComparisonOutcome: TypeAlias = (
    Isomorphic | InvariantNonIsomorphic | ExhaustiveNonIsomorphic | Inconclusive
)


def _validate_exhaustive_search(search: object) -> None:
    if type(search) is not IsomorphismSearchResult:
        raise IsomorphismOutcomeError(
            field="search", reason="must be an exact IsomorphismSearchResult"
        )
    if (
        search.found
        or search.witness is not None
        or search.outcome is not None
        or not search.complete
        or search.permutations_examined != search.color_compatible_count
        or search.partition_rejected_count + search.permutations_examined
        != search.total_allowed_count
    ):
        raise IsomorphismOutcomeError(
            field="search", reason="must exhaust every allowed bijection with no map"
        )


def _comparison_result_body(
    value: InvariantNonIsomorphic | ExhaustiveNonIsomorphic,
) -> dict[str, object]:
    if isinstance(value, InvariantNonIsomorphic):
        return {
            **_common_body(
                schema_type="anyalgebra.census.isomorphism.invariant_negative",
                status=value.status,
                complete=value.complete,
                core=value.core,
                equivalence=value.equivalence,
            ),
            "comparisonHash": _hash_record(value.comparison.semantic_hash),
            "mismatch": {
                "left": value.mismatch.left_value,
                "name": value.mismatch.name,
                "right": value.mismatch.right_value,
            },
            "permutationsExamined": value.permutations_examined,
            "proofKind": value.proof_kind,
        }
    return {
        **_common_body(
            schema_type="anyalgebra.census.isomorphism.exhaustive_negative",
            status=value.status,
            complete=value.complete,
            core=value.core,
            equivalence=value.equivalence,
        ),
        "checkedCount": value.checked_count,
        "operationCheckedCount": value.operation_checked_count,
        "partitionRejectedCount": value.partition_rejected_count,
        "proofKind": value.proof_kind,
        "searchHash": _hash_record(value.search.semantic_hash),
        "totalAllowedCount": value.total_allowed_count,
    }


def _bounded_count(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise IsomorphismOutcomeError(
            field="max_permutations",
            reason="must be None or a non-negative exact built-in int",
        )
    return value


def compare_isomorphism(
    core: CensusSpecCore,
    left_outputs: tuple[int, ...],
    right_outputs: tuple[int, ...],
    equivalence: EquivalencePolicy,
    *,
    max_permutations: int | None = None,
) -> ComparisonOutcome:
    """Return first invariant rejection, a map, full no-map, or bounded result."""
    limit = _bounded_count(max_permutations)
    checked_core, left, right, policy = _search_inputs(
        core, left_outputs, right_outputs, equivalence
    )
    invariant = compare_table_invariants(
        checked_core, left, policy, checked_core, right, policy
    )
    if invariant.rejected:
        return InvariantNonIsomorphic._create(invariant)

    left_partition = compute_invariant_partition(checked_core, left, policy)
    right_partition = compute_invariant_partition(checked_core, right, policy)
    group = generate_allowed_permutations(policy)
    compatible = tuple(
        permutation
        for permutation in group
        if all(
            left_partition.colors[old] == right_partition.colors[permutation(old)]
            for old in range(checked_core.carrier_size)
        )
    )
    search_limit = len(compatible) if limit is None else min(limit, len(compatible))
    witness: TableTransportCertificate | None = None
    examined = 0
    for permutation in compatible[:search_limit]:
        examined += 1
        certificate = transport_operation_table(checked_core, left, permutation)
        if certificate.target_outputs == right:
            witness = certificate
            break
    if witness is not None:
        return Isomorphic.from_transport(witness)
    if examined == len(compatible):
        search = IsomorphismSearchResult._create(
            core=checked_core,
            equivalence=policy,
            left_outputs=left,
            right_outputs=right,
            witness=None,
            total_allowed_count=len(group),
            color_compatible_count=len(compatible),
            permutations_examined=examined,
        )
        return ExhaustiveNonIsomorphic._create(search)
    if limit is None:
        raise IsomorphismOutcomeError(
            field="search", reason="unbounded search stopped before exhaustion"
        )
    return Inconclusive.create(
        core=checked_core,
        equivalence=policy,
        left_outputs=left,
        right_outputs=right,
        bound_name="max_permutations",
        bound_limit=limit,
        examined_count=examined,
        remaining_count=len(compatible) - examined,
    )


def comparison_outcome_record(value: ComparisonOutcome) -> dict[str, object]:
    """Return canonical evidence after replaying each result's exact source."""
    if isinstance(value, Isomorphic | Inconclusive):
        return isomorphism_outcome_record(value)
    if type(value) is InvariantNonIsomorphic:
        expected_comparison = compare_table_invariants(
            value.comparison.left.core,
            value.comparison.left.outputs,
            value.comparison.left.equivalence,
            value.comparison.right.core,
            value.comparison.right.outputs,
            value.comparison.right.equivalence,
        )
        invariant_comparison_record(value.comparison)
        expected_invariant = InvariantNonIsomorphic._create(expected_comparison)
        if value != expected_invariant:
            raise IsomorphismOutcomeError(
                field="invariant_negative", reason="content drift"
            )
    elif type(value) is ExhaustiveNonIsomorphic:
        _validate_exhaustive_search(value.search)
        isomorphism_search_record(value.search)
        expected_search = search_isomorphism(
            value.search.core,
            value.search.left_outputs,
            value.search.right_outputs,
            value.search.equivalence,
        )
        expected_exhaustive = ExhaustiveNonIsomorphic._create(expected_search)
        if value != expected_exhaustive:
            raise IsomorphismOutcomeError(
                field="exhaustive_negative", reason="content drift"
            )
    else:
        raise IsomorphismOutcomeError(
            field="outcome", reason="must be an exact comparison outcome"
        )
    body = _comparison_result_body(value)
    semantic_hash = _body_hash(body)
    if semantic_hash != value.semantic_hash:
        raise IsomorphismOutcomeError(field="semantic_hash", reason="content drift")
    return {**body, "contentHash": _hash_record(semantic_hash)}


def comparison_outcome_canonical_bytes(value: ComparisonOutcome) -> bytes:
    """Return deterministic UTF-8 canonical JSON for one replayed comparison."""
    return _encoded(comparison_outcome_record(value))


__all__ = (
    "ComparisonOutcome",
    "ExhaustiveNonIsomorphic",
    "Inconclusive",
    "InvariantNonIsomorphic",
    "Isomorphic",
    "IsomorphismOutcome",
    "IsomorphismOutcomeError",
    "IsomorphismSearchResult",
    "NonIsomorphic",
    "compare_isomorphism",
    "comparison_outcome_canonical_bytes",
    "comparison_outcome_record",
    "isomorphism_outcome_canonical_bytes",
    "isomorphism_outcome_record",
    "isomorphism_search_canonical_bytes",
    "isomorphism_search_record",
    "search_isomorphism",
)
