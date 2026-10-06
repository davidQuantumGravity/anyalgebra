"""Exact lexicographic indexing for bounded finite operation tables."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import cast

from anyalgebra.core.errors import AnyAlgebraError

from .spec import CensusSpecCore


class OperationTableCodeError(AnyAlgebraError, ValueError):
    """An operation-table digit sequence or candidate index was invalid."""

    def __init__(self, *, field: str, reason: str, index: int | None = None) -> None:
        self.field = field
        self.reason = reason
        self.index = index
        location = field if index is None else f"{field}[{index}]"
        super().__init__(f"invalid operation table code {location}: {reason}")


def _core(value: object) -> CensusSpecCore:
    if type(value) is not CensusSpecCore:
        raise OperationTableCodeError(
            field="core", reason="must be an exact CensusSpecCore"
        )
    return value


def _iterator(value: object) -> Iterator[object]:
    if isinstance(value, str | bytes):
        raise OperationTableCodeError(
            field="outputs", reason="must be an iterable of table digits"
        )
    try:
        return iter(cast(Iterable[object], value))
    except Exception as error:
        raise OperationTableCodeError(
            field="outputs", reason="must be an iterable of table digits"
        ) from error


def rank_operation_table(core: CensusSpecCore, outputs: Iterable[int]) -> int:
    """Return the base-``n`` rank with the first flat cell most significant."""
    checked = _core(core)
    if checked.candidate_count == 0:
        raise OperationTableCodeError(
            field="outputs", reason="the declared core has no operation tables"
        )
    iterator = _iterator(outputs)
    rank = 0
    try:
        for index in range(checked.input_tuple_count):
            try:
                digit = next(iterator)
            except StopIteration as error:
                raise OperationTableCodeError(
                    field="outputs", reason="wrong cell count", index=index
                ) from error
            if type(digit) is not int:
                raise OperationTableCodeError(
                    field="outputs",
                    reason="digits must be exact built-in ints, excluding bool",
                    index=index,
                )
            if digit < 0 or digit >= checked.carrier_size:
                raise OperationTableCodeError(
                    field="outputs",
                    reason="digit is outside the carrier range",
                    index=index,
                )
            rank = rank * checked.carrier_size + digit
        try:
            next(iterator)
        except StopIteration:
            return rank
        raise OperationTableCodeError(
            field="outputs",
            reason="wrong cell count",
            index=checked.input_tuple_count,
        )
    except OperationTableCodeError:
        raise
    except Exception as error:
        raise OperationTableCodeError(
            field="outputs", reason="iterable traversal failed"
        ) from error


def unrank_operation_table(
    core: CensusSpecCore, candidate_index: int
) -> tuple[int, ...]:
    """Return the unique flat output tuple at one validated candidate index."""
    checked = _core(core)
    if type(candidate_index) is not int or candidate_index < 0:
        raise OperationTableCodeError(
            field="candidate_index",
            reason="must be a non-negative built-in int, excluding bool",
        )
    if candidate_index >= checked.candidate_count:
        raise OperationTableCodeError(
            field="candidate_index", reason="is outside the candidate range"
        )
    if checked.input_tuple_count == 0:
        return ()

    remainder = candidate_index
    outputs = [0] * checked.input_tuple_count
    for index in range(checked.input_tuple_count - 1, -1, -1):
        remainder, outputs[index] = divmod(remainder, checked.carrier_size)
    if remainder:
        raise OperationTableCodeError(
            field="candidate_index", reason="internal rank decomposition failed"
        )
    return tuple(outputs)


__all__ = (
    "OperationTableCodeError",
    "rank_operation_table",
    "unrank_operation_table",
)
