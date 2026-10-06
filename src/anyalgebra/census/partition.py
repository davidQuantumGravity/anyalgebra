"""Relabeling-invariant element partitions for sound canonical-search ordering."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json

from anyalgebra.core.errors import AnyAlgebraError

from .canonical import CanonicalLabel
from .equivalence import EquivalencePolicy
from .permutations import CarrierPermutation, generate_allowed_permutations
from .spec import CensusSpecCore
from .table_codes import OperationTableCodeError, rank_operation_table
from .transport import TableTransportCertificate, transport_operation_table


class InvariantPartitionError(AnyAlgebraError, ValueError):
    """A partition input or refinement contract was invalid."""

    def __init__(self, *, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid invariant partition {field}: {reason}")


def _canonical_colors(signatures: tuple[object, ...]) -> tuple[int, ...]:
    ordered = tuple(
        sorted(
            set(signatures),
            key=lambda signature: json.dumps(
                signature,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
    )
    color_for = {signature: color for color, signature in enumerate(ordered)}
    return tuple(color_for[signature] for signature in signatures)


def _initial_colors(core: CensusSpecCore, policy: EquivalencePolicy) -> tuple[int, ...]:
    block_names = [""] * core.carrier_size
    for block in policy.sort_blocks:
        for element in block.elements:
            block_names[element] = block.name
    fixed = set(policy.fixed_elements)
    names: list[list[str]] = [[] for _ in range(core.carrier_size)]
    for name, element in core.distinguished_elements:
        names[element].append(name)
    signatures: tuple[object, ...] = tuple(
        (
            block_names[element],
            element if element in fixed else -1,
            tuple(sorted(names[element])),
        )
        for element in range(core.carrier_size)
    )
    return _canonical_colors(signatures)


def _refinement_signatures(
    core: CensusSpecCore,
    outputs: tuple[int, ...],
    colors: tuple[int, ...],
) -> tuple[object, ...]:
    cells = tuple(
        (inputs, outputs[index])
        for index, inputs in enumerate(
            product(range(core.carrier_size), repeat=core.arity)
        )
    )
    signatures: list[object] = []
    for element in range(core.carrier_size):
        output_role = tuple(
            sorted(
                tuple(colors[argument] for argument in inputs)
                for inputs, output in cells
                if output == element
            )
        )
        input_roles = tuple(
            tuple(
                sorted(
                    (
                        colors[output],
                        tuple(colors[argument] for argument in inputs),
                    )
                    for inputs, output in cells
                    if inputs[position] == element
                )
            )
            for position in range(core.arity)
        )
        if core.arity == 0:
            diagonal: tuple[int, ...] = ()
        else:
            diagonal_index = 0
            for _ in range(core.arity):
                diagonal_index = diagonal_index * core.carrier_size + element
            diagonal_output = outputs[diagonal_index]
            diagonal = (
                int(diagonal_output == element),
                colors[diagonal_output],
            )
        signatures.append((colors[element], output_role, input_roles, diagonal))
    return tuple(signatures)


def _blocks(colors: tuple[int, ...]) -> tuple[tuple[int, tuple[int, ...]], ...]:
    return tuple(
        (color, tuple(index for index, item in enumerate(colors) if item == color))
        for color in sorted(set(colors))
    )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class InvariantPartition:
    """A deterministic equivariant color partition, never a complete invariant."""

    core: CensusSpecCore
    equivalence: EquivalencePolicy
    source_outputs: tuple[int, ...]
    colors: tuple[int, ...]
    blocks: tuple[tuple[int, tuple[int, ...]], ...]
    initial_color_count: int
    color_count: int
    refinement_rounds: int
    refinement_passes: int
    signature_evaluations: int
    stable: bool
    complete_invariant: bool
    search_role: str
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise InvariantPartitionError(field="partition", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("InvariantPartition cannot be subclassed")

    @classmethod
    def _create(
        cls,
        core: CensusSpecCore,
        policy: EquivalencePolicy,
        outputs: tuple[int, ...],
        colors: tuple[int, ...],
        *,
        initial_color_count: int,
        refinement_rounds: int,
        refinement_passes: int,
        signature_evaluations: int,
        stable: bool,
    ) -> InvariantPartition:
        value = object.__new__(InvariantPartition)
        for field, item in (
            ("core", core),
            ("equivalence", policy),
            ("source_outputs", outputs),
            ("colors", colors),
            ("blocks", _blocks(colors)),
            ("initial_color_count", initial_color_count),
            ("color_count", len(set(colors))),
            ("refinement_rounds", refinement_rounds),
            ("refinement_passes", refinement_passes),
            ("signature_evaluations", signature_evaluations),
            ("stable", stable),
            ("complete_invariant", False),
            ("search_role", "ordering_only"),
        ):
            object.__setattr__(value, field, item)
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is InvariantPartition and (
            self.core,
            self.equivalence,
            self.source_outputs,
            self.colors,
            self.blocks,
            self.initial_color_count,
            self.color_count,
            self.refinement_rounds,
            self.refinement_passes,
            self.signature_evaluations,
            self.stable,
            self.complete_invariant,
            self.search_role,
        ) == (
            other.core,
            other.equivalence,
            other.source_outputs,
            other.colors,
            other.blocks,
            other.initial_color_count,
            other.color_count,
            other.refinement_rounds,
            other.refinement_passes,
            other.signature_evaluations,
            other.stable,
            other.complete_invariant,
            other.search_role,
        )

    def __repr__(self) -> str:
        return (
            "InvariantPartition("
            f"carrier_size={self.core.carrier_size}, color_count={self.color_count}, "
            f"refinement_rounds={self.refinement_rounds}, stable={self.stable})"
        )


def _inputs(
    core: CensusSpecCore,
    source_outputs: tuple[int, ...],
    equivalence: EquivalencePolicy,
    max_refinement_rounds: int | None,
) -> None:
    if type(core) is not CensusSpecCore:
        raise InvariantPartitionError(
            field="core", reason="must be an exact CensusSpecCore"
        )
    if core.candidate_count == 0:
        raise InvariantPartitionError(
            field="source_outputs", reason="the declared core has no operation tables"
        )
    if type(equivalence) is not EquivalencePolicy or equivalence.core is not core:
        raise InvariantPartitionError(
            field="equivalence", reason="must belong to the literal partition core"
        )
    if type(source_outputs) is not tuple:
        raise InvariantPartitionError(
            field="source_outputs", reason="must be an exact tuple"
        )
    try:
        rank_operation_table(core, source_outputs)
    except OperationTableCodeError as error:
        raise InvariantPartitionError(
            field="source_outputs", reason="is not a valid table for the core"
        ) from error
    if max_refinement_rounds is not None and (
        type(max_refinement_rounds) is not int
        or max_refinement_rounds < 0
        or max_refinement_rounds > core.carrier_size
    ):
        raise InvariantPartitionError(
            field="max_refinement_rounds",
            reason="must be None or a non-negative built-in int at most carrier size",
        )


def compute_invariant_partition(
    core: CensusSpecCore,
    source_outputs: tuple[int, ...],
    equivalence: EquivalencePolicy,
    *,
    max_refinement_rounds: int | None = None,
) -> InvariantPartition:
    """Refine equivariant element colors to stability or an explicit round bound."""
    _inputs(core, source_outputs, equivalence, max_refinement_rounds)
    colors = _initial_colors(core, equivalence)
    initial_count = len(set(colors))
    if core.carrier_size == 0:
        return InvariantPartition._create(
            core,
            equivalence,
            source_outputs,
            colors,
            initial_color_count=0,
            refinement_rounds=0,
            refinement_passes=0,
            signature_evaluations=0,
            stable=True,
        )
    if max_refinement_rounds == 0:
        return InvariantPartition._create(
            core,
            equivalence,
            source_outputs,
            colors,
            initial_color_count=initial_count,
            refinement_rounds=0,
            refinement_passes=0,
            signature_evaluations=0,
            stable=initial_count == core.carrier_size,
        )

    limit = (
        core.carrier_size if max_refinement_rounds is None else max_refinement_rounds
    )
    rounds = 0
    passes = 0
    stable = len(set(colors)) == core.carrier_size
    while not stable and passes < limit:
        signatures = _refinement_signatures(core, source_outputs, colors)
        passes += 1
        refined = _canonical_colors(signatures)
        if refined == colors:
            stable = True
            break
        colors = refined
        rounds += 1
        if len(set(colors)) == core.carrier_size:
            stable = True
    return InvariantPartition._create(
        core,
        equivalence,
        source_outputs,
        colors,
        initial_color_count=initial_count,
        refinement_rounds=rounds,
        refinement_passes=passes,
        signature_evaluations=passes * core.carrier_size,
        stable=stable,
    )


def _target_color_sequence(
    partition: InvariantPartition, permutation: CarrierPermutation
) -> tuple[int, ...]:
    inverse = permutation.inverse()
    return tuple(partition.colors[inverse(new)] for new in range(len(inverse.images)))


def order_permutations_by_partition(
    partition: InvariantPartition,
) -> tuple[CarrierPermutation, ...]:
    """Order, but never remove, every relabeling allowed by the partition policy."""
    if type(partition) is not InvariantPartition:
        raise InvariantPartitionError(
            field="partition", reason="must be an exact InvariantPartition"
        )
    group = generate_allowed_permutations(partition.equivalence)
    return tuple(
        sorted(
            group,
            key=lambda permutation: (
                _target_color_sequence(partition, permutation),
                permutation.images,
            ),
        )
    )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class PartitionGuidedCanonicalization:
    """Canonical result plus transparent ordering-only work evidence."""

    label: CanonicalLabel
    partition: InvariantPartition
    permutations_checked: int
    permutations_pruned: int
    best_update_count: int
    strategy: str
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise InvariantPartitionError(
            field="canonicalization", reason="is factory-owned"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("PartitionGuidedCanonicalization cannot be subclassed")

    @classmethod
    def _create(
        cls,
        label: CanonicalLabel,
        partition: InvariantPartition,
        *,
        best_update_count: int,
    ) -> PartitionGuidedCanonicalization:
        value = object.__new__(PartitionGuidedCanonicalization)
        for field, item in (
            ("label", label),
            ("partition", partition),
            ("permutations_checked", label.permutations_checked),
            ("permutations_pruned", 0),
            ("best_update_count", best_update_count),
            ("strategy", "invariant_partition_ordered_exhaustive_v1"),
        ):
            object.__setattr__(value, field, item)
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is PartitionGuidedCanonicalization and (
            self.label,
            self.partition,
            self.permutations_checked,
            self.permutations_pruned,
            self.best_update_count,
            self.strategy,
        ) == (
            other.label,
            other.partition,
            other.permutations_checked,
            other.permutations_pruned,
            other.best_update_count,
            other.strategy,
        )

    def __repr__(self) -> str:
        return (
            "PartitionGuidedCanonicalization("
            f"canonical_id={str(self.label.canonical_id)!r}, "
            f"permutations_checked={self.permutations_checked}, "
            "permutations_pruned=0)"
        )


def canonicalize_with_invariant_partition(
    core: CensusSpecCore,
    source_outputs: tuple[int, ...],
    equivalence: EquivalencePolicy,
    *,
    max_refinement_rounds: int | None = None,
) -> PartitionGuidedCanonicalization:
    """Canonicalize exhaustively after deterministic invariant-based ordering."""
    partition = compute_invariant_partition(
        core,
        source_outputs,
        equivalence,
        max_refinement_rounds=max_refinement_rounds,
    )
    group = order_permutations_by_partition(partition)
    chosen: TableTransportCertificate | None = None
    images: set[tuple[int, ...]] = set()
    stabilizer = 0
    best_updates = 0
    for permutation in group:
        certificate = transport_operation_table(core, source_outputs, permutation)
        image = certificate.target_outputs
        images.add(image)
        if image == source_outputs:
            stabilizer += 1
        if chosen is None or image < chosen.target_outputs:
            chosen = certificate
            best_updates += 1
        elif (
            image == chosen.target_outputs
            and permutation.images < chosen.permutation.images
        ):
            chosen = certificate
    if chosen is None:
        raise InvariantPartitionError(
            field="equivalence", reason="allowed permutation group is empty"
        )
    label = CanonicalLabel._create(
        core,
        equivalence,
        source_outputs,
        chosen.target_outputs,
        chosen,
        checked=len(group),
        orbit_size=len(images),
        stabilizer_size=stabilizer,
    )
    return PartitionGuidedCanonicalization._create(
        label, partition, best_update_count=best_updates
    )


__all__ = (
    "InvariantPartition",
    "InvariantPartitionError",
    "PartitionGuidedCanonicalization",
    "canonicalize_with_invariant_partition",
    "compute_invariant_partition",
    "order_permutations_by_partition",
)
