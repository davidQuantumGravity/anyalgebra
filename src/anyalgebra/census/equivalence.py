"""Immutable policies describing allowed finite-carrier relabelings."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import factorial
from typing import cast

from anyalgebra.core.errors import AnyAlgebraError

from .spec import CensusSpecCore


_MAX_DECLARATIONS = 256
_MAX_NAME_CODE_POINTS = 256


class EquivalencePolicyError(AnyAlgebraError, ValueError):
    """A relabeling policy or candidate permutation was malformed."""

    def __init__(self, *, field: str, reason: str, index: int | None = None) -> None:
        self.field = field
        self.reason = reason
        self.index = index
        location = field if index is None else f"{field}[{index}]"
        super().__init__(f"invalid equivalence policy {location}: {reason}")


def _name(value: object) -> str:
    if type(value) is not str or not value:
        raise EquivalencePolicyError(
            field="name", reason="must be a non-empty exact built-in str"
        )
    if value != value.strip():
        raise EquivalencePolicyError(field="name", reason="has outer whitespace")
    if len(value) > _MAX_NAME_CODE_POINTS:
        raise EquivalencePolicyError(field="name", reason="text limit exceeded")
    if any("\ud800" <= character <= "\udfff" for character in value):
        raise EquivalencePolicyError(
            field="name", reason="contains an invalid Unicode scalar"
        )
    return value


def _bounded_snapshot(value: object, *, field: str) -> tuple[object, ...]:
    if isinstance(value, str | bytes):
        raise EquivalencePolicyError(field=field, reason="must be an iterable")
    try:
        iterator = iter(cast(Iterable[object], value))
    except Exception as error:
        raise EquivalencePolicyError(
            field=field, reason="must be an iterable"
        ) from error
    snapshot: list[object] = []
    try:
        for _ in range(_MAX_DECLARATIONS + 1):
            snapshot.append(next(iterator))
    except StopIteration:
        return tuple(snapshot)
    except Exception as error:
        raise EquivalencePolicyError(
            field=field, reason="iterable snapshot failed"
        ) from error
    raise EquivalencePolicyError(field=field, reason="declaration limit exceeded")


def _indices(value: object, *, field: str, allow_empty: bool) -> tuple[int, ...]:
    source = _bounded_snapshot(value, field=field)
    if not source and not allow_empty:
        raise EquivalencePolicyError(field=field, reason="must be non-empty")
    checked: list[int] = []
    seen: set[int] = set()
    for index, item in enumerate(source):
        if type(item) is not int or item < 0:
            raise EquivalencePolicyError(
                field=field,
                reason="must contain non-negative built-in ints, excluding bool",
                index=index,
            )
        if item in seen:
            raise EquivalencePolicyError(
                field=field, reason="contains a duplicate element", index=index
            )
        seen.add(item)
        checked.append(item)
    return tuple(sorted(checked))


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class SortBlock:
    """One named nonempty carrier subset preserved by relabelings."""

    name: str
    elements: tuple[int, ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise EquivalencePolicyError(field="sort_block", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("SortBlock cannot be subclassed")

    @classmethod
    def create(cls, *, name: str, elements: Iterable[int]) -> SortBlock:
        if cls is not SortBlock:
            raise EquivalencePolicyError(
                field="sort_block", reason="factory requires exact type"
            )
        value = object.__new__(SortBlock)
        object.__setattr__(value, "name", _name(name))
        object.__setattr__(
            value, "elements", _indices(elements, field="elements", allow_empty=False)
        )
        return value

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is SortBlock
            and self.name == other.name
            and self.elements == other.elements
        )

    def __repr__(self) -> str:
        return f"SortBlock(name={self.name!r}, size={len(self.elements)})"


def _blocks(
    core: CensusSpecCore, value: Iterable[SortBlock] | None
) -> tuple[SortBlock, ...]:
    if value is None:
        if core.carrier_size == 0:
            return ()
        return (SortBlock.create(name="carrier", elements=range(core.carrier_size)),)
    source = _bounded_snapshot(value, field="sort_blocks")
    blocks: list[SortBlock] = []
    names: set[str] = set()
    owners: dict[int, int] = {}
    for index, item in enumerate(source):
        if type(item) is not SortBlock:
            raise EquivalencePolicyError(
                field="sort_blocks",
                reason="must contain exact SortBlock values",
                index=index,
            )
        if item.name in names:
            raise EquivalencePolicyError(
                field="sort_blocks", reason="duplicate block name", index=index
            )
        names.add(item.name)
        for element in item.elements:
            if element >= core.carrier_size:
                raise EquivalencePolicyError(
                    field="sort_blocks",
                    reason="element is outside the carrier range",
                    index=index,
                )
            if element in owners:
                raise EquivalencePolicyError(
                    field="sort_blocks",
                    reason="duplicate carrier element across blocks",
                    index=index,
                )
            owners[element] = index
        blocks.append(item)
    if set(owners) != set(range(core.carrier_size)):
        raise EquivalencePolicyError(
            field="sort_blocks", reason="block partition is incomplete"
        )
    return tuple(sorted(blocks, key=lambda block: block.name))


def _fixed(core: CensusSpecCore, value: Iterable[int]) -> tuple[int, ...]:
    explicit = _indices(value, field="fixed_elements", allow_empty=True)
    for index, element in enumerate(explicit):
        if element >= core.carrier_size:
            raise EquivalencePolicyError(
                field="fixed_elements",
                reason="element is outside the carrier range",
                index=index,
            )
    distinguished = {element for _, element in core.distinguished_elements}
    return tuple(sorted(distinguished.union(explicit)))


def _permutation_count(
    blocks: tuple[SortBlock, ...], fixed_elements: tuple[int, ...]
) -> int:
    fixed = set(fixed_elements)
    result = 1
    for block in blocks:
        result *= factorial(sum(element not in fixed for element in block.elements))
    return result


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class EquivalencePolicy:
    """Exact carrier-relabeling group declaration with membership checking."""

    core: CensusSpecCore
    kind: str
    carrier_size: int
    fixed_elements: tuple[int, ...]
    sort_blocks: tuple[SortBlock, ...]
    permutation_count: int
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise EquivalencePolicyError(field="policy", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("EquivalencePolicy cannot be subclassed")

    @classmethod
    def literal(cls, core: CensusSpecCore) -> EquivalencePolicy:
        checked = _core(core)
        blocks = _blocks(checked, None)
        return cls._create(
            "literal", checked, tuple(range(checked.carrier_size)), blocks
        )

    @classmethod
    def relabeling(
        cls,
        core: CensusSpecCore,
        *,
        fixed_elements: Iterable[int] = (),
        sort_blocks: Iterable[SortBlock] | None = None,
    ) -> EquivalencePolicy:
        checked = _core(core)
        blocks = _blocks(checked, sort_blocks)
        fixed = _fixed(checked, fixed_elements)
        return cls._create("carrier_relabeling", checked, fixed, blocks)

    @classmethod
    def _create(
        cls,
        kind: str,
        core: CensusSpecCore,
        fixed: tuple[int, ...],
        blocks: tuple[SortBlock, ...],
    ) -> EquivalencePolicy:
        if cls is not EquivalencePolicy:
            raise EquivalencePolicyError(
                field="policy", reason="factory requires exact type"
            )
        value = object.__new__(EquivalencePolicy)
        object.__setattr__(value, "core", core)
        object.__setattr__(value, "kind", kind)
        object.__setattr__(value, "carrier_size", core.carrier_size)
        object.__setattr__(value, "fixed_elements", fixed)
        object.__setattr__(value, "sort_blocks", blocks)
        object.__setattr__(
            value, "permutation_count", _permutation_count(blocks, fixed)
        )
        return value

    def allows(self, permutation: tuple[int, ...]) -> bool:
        """Return membership for one valid old-index to new-index bijection."""
        if type(permutation) is not tuple or len(permutation) != self.carrier_size:
            raise EquivalencePolicyError(
                field="permutation", reason="must be an exact tuple of carrier size"
            )
        if any(type(element) is not int for element in permutation):
            raise EquivalencePolicyError(
                field="permutation", reason="must contain exact built-in ints"
            )
        if set(permutation) != set(range(self.carrier_size)):
            raise EquivalencePolicyError(
                field="permutation", reason="must be a carrier bijection"
            )
        if any(permutation[element] != element for element in self.fixed_elements):
            return False
        if self.kind == "literal":
            return permutation == tuple(range(self.carrier_size))
        return all(
            permutation[element] in block.elements
            for block in self.sort_blocks
            for element in block.elements
        )

    def __eq__(self, other: object) -> bool:
        return type(other) is EquivalencePolicy and (
            self.core,
            self.kind,
            self.carrier_size,
            self.fixed_elements,
            self.sort_blocks,
            self.permutation_count,
        ) == (
            other.core,
            other.kind,
            other.carrier_size,
            other.fixed_elements,
            other.sort_blocks,
            other.permutation_count,
        )

    def __repr__(self) -> str:
        return (
            "EquivalencePolicy("
            f"kind={self.kind!r}, carrier_size={self.carrier_size}, "
            f"fixed_count={len(self.fixed_elements)}, "
            f"block_count={len(self.sort_blocks)}, "
            f"permutation_count={self.permutation_count})"
        )


def _core(value: object) -> CensusSpecCore:
    if type(value) is not CensusSpecCore:
        raise EquivalencePolicyError(
            field="core", reason="must be an exact CensusSpecCore"
        )
    return value
