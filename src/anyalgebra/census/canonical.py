"""Brute-force canonical operation-table labels under explicit relabelings."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import SemanticHash

from .equivalence import EquivalencePolicy
from .permutations import (
    PermutationGenerationError,
    generate_allowed_permutations,
)
from .spec import CensusSpecCore
from .table_codes import OperationTableCodeError, rank_operation_table
from .transport import TableTransportCertificate, transport_operation_table


class CanonicalLabelError(AnyAlgebraError, ValueError):
    """A canonical search input or internal orbit count was invalid."""

    def __init__(self, *, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid canonical label {field}: {reason}")


def _policy_record(policy: EquivalencePolicy) -> dict[str, object]:
    return {
        "fixedElements": list(policy.fixed_elements),
        "kind": policy.kind,
        "sortBlocks": [
            {"elements": list(block.elements), "name": block.name}
            for block in policy.sort_blocks
        ],
    }


def _canonical_bytes(
    core: CensusSpecCore,
    outputs: tuple[int, ...],
    policy: EquivalencePolicy,
) -> bytes:
    record = {
        "arity": core.arity,
        "carrierSize": core.carrier_size,
        "equivalence": _policy_record(policy),
        "outputs": list(outputs),
        "schemaType": "anyalgebra.census.canonical_table",
        "schemaVersion": 1,
    }
    return json.dumps(
        record,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class CanonicalLabel:
    """Least orbit image plus first lexicographic transport certificate."""

    core: CensusSpecCore
    equivalence: EquivalencePolicy
    source_outputs: tuple[int, ...]
    canonical_outputs: tuple[int, ...]
    canonical_bytes: bytes
    canonical_id: SemanticHash
    certificate: TableTransportCertificate
    permutations_checked: int
    orbit_size: int
    stabilizer_size: int
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise CanonicalLabelError(field="label", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("CanonicalLabel cannot be subclassed")

    @classmethod
    def _create(
        cls,
        core: CensusSpecCore,
        policy: EquivalencePolicy,
        source: tuple[int, ...],
        canonical: tuple[int, ...],
        certificate: TableTransportCertificate,
        *,
        checked: int,
        orbit_size: int,
        stabilizer_size: int,
    ) -> CanonicalLabel:
        if orbit_size * stabilizer_size != checked:
            raise CanonicalLabelError(
                field="orbit", reason="orbit-stabilizer count mismatch"
            )
        encoded = _canonical_bytes(core, canonical, policy)
        value = object.__new__(CanonicalLabel)
        for field, item in (
            ("core", core),
            ("equivalence", policy),
            ("source_outputs", source),
            ("canonical_outputs", canonical),
            ("canonical_bytes", encoded),
            (
                "canonical_id",
                SemanticHash("sha256", hashlib.sha256(encoded).hexdigest()),
            ),
            ("certificate", certificate),
            ("permutations_checked", checked),
            ("orbit_size", orbit_size),
            ("stabilizer_size", stabilizer_size),
        ):
            object.__setattr__(value, field, item)
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is CanonicalLabel and (
            self.core,
            self.equivalence,
            self.source_outputs,
            self.canonical_outputs,
            self.canonical_bytes,
            self.canonical_id,
            self.certificate,
            self.permutations_checked,
            self.orbit_size,
            self.stabilizer_size,
        ) == (
            other.core,
            other.equivalence,
            other.source_outputs,
            other.canonical_outputs,
            other.canonical_bytes,
            other.canonical_id,
            other.certificate,
            other.permutations_checked,
            other.orbit_size,
            other.stabilizer_size,
        )

    def __repr__(self) -> str:
        return (
            "CanonicalLabel("
            f"canonical_id={str(self.canonical_id)!r}, "
            f"orbit_size={self.orbit_size}, "
            f"permutations_checked={self.permutations_checked})"
        )


def canonicalize_operation_table(
    core: CensusSpecCore,
    source_outputs: tuple[int, ...],
    equivalence: EquivalencePolicy,
) -> CanonicalLabel:
    """Choose the least allowed table image and first lexicographic witness."""
    if type(core) is not CensusSpecCore:
        raise CanonicalLabelError(field="core", reason="must be exact CensusSpecCore")
    if core.candidate_count == 0:
        raise CanonicalLabelError(
            field="source_outputs", reason="the declared core has no operation tables"
        )
    if type(equivalence) is not EquivalencePolicy or equivalence.core is not core:
        raise CanonicalLabelError(
            field="equivalence", reason="must belong to the literal canonical core"
        )
    if type(source_outputs) is not tuple:
        raise CanonicalLabelError(
            field="source_outputs", reason="must be an exact tuple"
        )
    try:
        rank_operation_table(core, source_outputs)
    except OperationTableCodeError as error:
        raise CanonicalLabelError(
            field="source_outputs", reason="is not a valid table for the core"
        ) from error
    try:
        group = generate_allowed_permutations(equivalence)
    except PermutationGenerationError as error:
        raise CanonicalLabelError(
            field="equivalence", reason="allowed group cannot be generated"
        ) from error

    images: set[tuple[int, ...]] = set()
    chosen: TableTransportCertificate | None = None
    stabilizer = 0
    for permutation in group:
        certificate = transport_operation_table(core, source_outputs, permutation)
        image = certificate.target_outputs
        images.add(image)
        if image == source_outputs:
            stabilizer += 1
        if chosen is None or image < chosen.target_outputs:
            chosen = certificate
    if chosen is None:
        raise CanonicalLabelError(field="equivalence", reason="allowed group is empty")
    return CanonicalLabel._create(
        core,
        equivalence,
        source_outputs,
        chosen.target_outputs,
        chosen,
        checked=len(group),
        orbit_size=len(images),
        stabilizer_size=stabilizer,
    )


__all__ = (
    "CanonicalLabel",
    "CanonicalLabelError",
    "canonicalize_operation_table",
)
