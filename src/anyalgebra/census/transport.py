"""Exact conjugation transport of finite operation tables under relabeling."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from anyalgebra.core.errors import AnyAlgebraError

from .permutations import CarrierPermutation
from .spec import CensusSpecCore
from .table_codes import OperationTableCodeError, rank_operation_table


class TableTransportError(AnyAlgebraError, ValueError):
    """A transport input or cell certificate was invalid."""

    def __init__(self, *, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid operation-table transport {field}: {reason}")


def _flat_index(arguments: tuple[int, ...], size: int) -> int:
    index = 0
    for argument in arguments:
        index = index * size + argument
    return index


def _verify(
    core: CensusSpecCore,
    source: tuple[int, ...],
    target: tuple[int, ...],
    permutation: CarrierPermutation,
) -> bool:
    for old_flat, old_inputs in enumerate(
        product(range(core.carrier_size), repeat=core.arity)
    ):
        new_inputs = tuple(permutation(item) for item in old_inputs)
        new_flat = _flat_index(new_inputs, core.carrier_size)
        if target[new_flat] != permutation(source[old_flat]):
            return False
    return True


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class TableTransportCertificate:
    """Cell-complete witness for one explicit old-to-new table conjugation."""

    core: CensusSpecCore
    source_outputs: tuple[int, ...]
    target_outputs: tuple[int, ...]
    permutation: CarrierPermutation
    direction: str
    checked_cell_count: int
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TableTransportError(field="certificate", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("TableTransportCertificate cannot be subclassed")

    @classmethod
    def _create(
        cls,
        core: CensusSpecCore,
        source: tuple[int, ...],
        target: tuple[int, ...],
        permutation: CarrierPermutation,
    ) -> TableTransportCertificate:
        if not _verify(core, source, target, permutation):
            raise TableTransportError(
                field="certificate", reason="cell verification failed"
            )
        value = object.__new__(TableTransportCertificate)
        for field, item in (
            ("core", core),
            ("source_outputs", source),
            ("target_outputs", target),
            ("permutation", permutation),
            ("direction", "old_index_to_new_index_conjugation"),
            ("checked_cell_count", core.input_tuple_count),
        ):
            object.__setattr__(value, field, item)
        return value

    def verify_cells(self) -> bool:
        """Independently replay the defining equation over every old input cell."""
        return _verify(
            self.core,
            self.source_outputs,
            self.target_outputs,
            self.permutation,
        )

    def __eq__(self, other: object) -> bool:
        return type(other) is TableTransportCertificate and (
            self.core,
            self.source_outputs,
            self.target_outputs,
            self.permutation,
            self.direction,
            self.checked_cell_count,
        ) == (
            other.core,
            other.source_outputs,
            other.target_outputs,
            other.permutation,
            other.direction,
            other.checked_cell_count,
        )

    def __repr__(self) -> str:
        return (
            "TableTransportCertificate("
            f"size={self.core.carrier_size}, arity={self.core.arity}, "
            f"checked_cell_count={self.checked_cell_count})"
        )


def transport_operation_table(
    core: CensusSpecCore,
    source_outputs: tuple[int, ...],
    permutation: CarrierPermutation,
) -> TableTransportCertificate:
    """Conjugate one table so ``p(f(x)) == f'(p(x))`` at every input."""
    if type(core) is not CensusSpecCore:
        raise TableTransportError(field="core", reason="must be exact CensusSpecCore")
    if core.candidate_count == 0:
        raise TableTransportError(
            field="source_outputs", reason="the declared core has no operation tables"
        )
    if (
        type(permutation) is not CarrierPermutation
        or permutation.policy.core is not core
    ):
        raise TableTransportError(
            field="permutation", reason="must belong to the literal transport core"
        )
    if type(source_outputs) is not tuple:
        raise TableTransportError(
            field="source_outputs", reason="must be an exact flat tuple"
        )
    try:
        rank_operation_table(core, source_outputs)
    except OperationTableCodeError as error:
        raise TableTransportError(
            field="source_outputs", reason="is not a valid table for the core"
        ) from error
    target = [0] * core.input_tuple_count
    for old_flat, old_inputs in enumerate(
        product(range(core.carrier_size), repeat=core.arity)
    ):
        new_inputs = tuple(permutation(item) for item in old_inputs)
        target[_flat_index(new_inputs, core.carrier_size)] = permutation(
            source_outputs[old_flat]
        )
    return TableTransportCertificate._create(
        core, source_outputs, tuple(target), permutation
    )


__all__ = (
    "TableTransportCertificate",
    "TableTransportError",
    "transport_operation_table",
)
