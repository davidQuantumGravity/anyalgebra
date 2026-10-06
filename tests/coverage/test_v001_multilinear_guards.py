"""Exercise multilinear normalization failures that protect public factories."""

from __future__ import annotations

from collections.abc import ItemsView, Iterator, Mapping
from typing import Any, cast

import pytest

import anyalgebra.algebra.multilinear as multilinear
from anyalgebra.algebra.multilinear import (
    BasisProductTable,
    FiniteMultilinearStructure,
    MultilinearDefinitionError,
    StructureConstants,
)
from anyalgebra.core.domains import Domain, DomainElement, QQ, ZZ
from anyalgebra.core.modules import Basis, FreeModule


def _module(rank: int = 1, *, domain: Domain[object] | None = None) -> FreeModule:
    coefficient_domain: Domain[object] = ZZ() if domain is None else domain
    return FreeModule(
        coefficient_domain,
        Basis(
            tuple(f"e{i}" for i in range(rank)),
            coefficient_domain=coefficient_domain,
        ),
    )


def test_sparse_entries_reject_strings_and_nonpairs() -> None:
    module = _module()
    with pytest.raises(MultilinearDefinitionError, match="iterable"):
        StructureConstants.from_sparse((module.basis,), module.basis, "entry")  # type: ignore[arg-type]
    with pytest.raises(MultilinearDefinitionError, match="pair"):
        StructureConstants.from_sparse(
            (module.basis,),
            module.basis,
            cast(Any, (((0, 0), ZZ().element(1), "extra"),)),
        )


def test_coefficient_protocol_and_parent_lookup_failures_are_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()

    class ExplodingMeta(type):
        def __instancecheck__(cls, instance: object) -> bool:
            del cls, instance
            raise RuntimeError("private")

    class ExplodingProtocol(metaclass=ExplodingMeta):
        pass

    monkeypatch.setattr(multilinear, "DomainElement", ExplodingProtocol)
    with pytest.raises(MultilinearDefinitionError, match="protocol check failed"):
        multilinear._require_domain_element(object(), module.domain, entry_index=0)
    monkeypatch.undo()

    class BadParent:
        @property
        def parent(self) -> object:
            raise RuntimeError("private")

        @property
        def value(self) -> object:
            return 1

    with pytest.raises(MultilinearDefinitionError, match="parent lookup failed"):
        multilinear._require_domain_element(BadParent(), module.domain, entry_index=0)


class _Element:
    def __init__(self, parent: object, value: object) -> None:
        self._parent = parent
        self._value = value

    @property
    def parent(self) -> object:
        return self._parent

    @property
    def value(self) -> object:
        return self._value


class _OddDomain:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.calls = 0

    def normalize(self, value: object) -> object:
        return value

    def element(self, value: object) -> object:
        self.calls += 1
        if self.mode == "raise" or (self.mode == "second-raise" and self.calls > 1):
            raise RuntimeError("private")
        if self.mode == "invalid":
            return object()
        if isinstance(value, _Element) and value.parent is self:
            return value
        return _Element(self, value)


@pytest.mark.parametrize("mode", ["raise", "invalid"])
def test_zero_probe_never_turns_domain_failure_into_a_zero_proof(mode: str) -> None:
    domain = _OddDomain(mode)
    coefficient = cast(DomainElement[object], _Element(domain, 7))
    assert not multilinear._is_proven_zero(coefficient, cast(Domain[object], domain))


def test_structure_constant_equality_rejects_wrong_shape_and_metadata() -> None:
    module = _module()
    coefficient = ZZ().element(1)
    left = StructureConstants.from_sparse(
        (module.basis,), module.basis, (((0, 0), coefficient),)
    )
    empty = StructureConstants.from_sparse((module.basis,), module.basis, ())
    assert left != object()
    assert left != empty
    assert left == left

    qq_module = _module(domain=cast(Domain[object], QQ()))
    foreign = StructureConstants.from_sparse(
        (qq_module.basis,), qq_module.basis, (((0, 0), QQ().element(1)),)
    )
    assert left != foreign


def test_table_cells_reject_outer_mapping_bad_pairs_and_duplicate_coordinates() -> None:
    module = _module()
    with pytest.raises(MultilinearDefinitionError, match="ordered iterable"):
        BasisProductTable.from_cells(module, 1, cast(Any, {0: {}}))
    with pytest.raises(MultilinearDefinitionError, match="exact pair"):
        BasisProductTable.from_cells(module, 1, cast(Any, (_BadPairMapping(),)))
    with pytest.raises(MultilinearDefinitionError, match="duplicate"):
        BasisProductTable.from_cells(module, 1, cast(Any, (_DuplicateMapping(),)))
    table = BasisProductTable.from_cells(module, 1, ({},))
    assert table != object()


class _BadPairMapping(Mapping[int, object]):
    def __getitem__(self, key: int) -> object:
        raise KeyError(key)

    def __iter__(self) -> Iterator[int]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self) -> ItemsView[int, object]:
        return cast(ItemsView[int, object], ((0, ZZ().element(1), "extra"),))


class _DuplicateMapping(_BadPairMapping):
    def items(self) -> ItemsView[int, object]:
        return cast(
            ItemsView[int, object],
            ((0, ZZ().element(1)), (0, ZZ().element(2))),
        )


def test_literal_basis_vector_normalizes_domain_and_element_failures() -> None:
    raising = cast(Domain[object], _OddDomain("raise"))
    raising_module = _module(domain=raising)
    with pytest.raises(MultilinearDefinitionError, match="coefficient one"):
        multilinear._literal_basis_vector(raising_module, 0)

    invalid = cast(Domain[object], _OddDomain("invalid"))
    invalid_module = _module(domain=invalid)
    with pytest.raises(MultilinearDefinitionError, match="literal coefficient one"):
        multilinear._literal_basis_vector(invalid_module, 0)

    ordinary_module = _module()
    with pytest.raises(MultilinearDefinitionError, match="basis SparseElement"):
        multilinear._literal_basis_vector(ordinary_module, ordinary_module.rank)


def test_cell_element_normalizes_a_late_parent_failure() -> None:
    domain = cast(Domain[object], _OddDomain("valid"))
    module = _module(domain=domain)

    class StatefulElement(_Element):
        def __init__(self, parent: object, value: object) -> None:
            super().__init__(parent, value)
            self.lookups = 0

        @property
        def parent(self) -> object:
            self.lookups += 1
            if self.lookups > 1:
                raise RuntimeError("private")
            return self._parent

    coefficient = StatefulElement(domain, 1)
    with pytest.raises(MultilinearDefinitionError, match="coordinates are malformed"):
        multilinear._cell_element({0: coefficient}, module, cell_index=0)


def test_structure_and_table_routes_reject_foreign_domains_and_mappings() -> None:
    module = _module()
    qq_module = _module(domain=cast(Domain[object], QQ()))
    constants = StructureConstants.from_sparse((qq_module.basis,), qq_module.basis, ())
    with pytest.raises(MultilinearDefinitionError, match="coefficient parent"):
        FiniteMultilinearStructure.from_structure_constants(module, constants)
    with pytest.raises(MultilinearDefinitionError, match="exactly one"):
        FiniteMultilinearStructure.from_tables(
            module, cast(Any, {"operation": object()})
        )
    operation = multilinear.MultilinearOperation.from_structure_constants(
        StructureConstants.from_sparse((module.basis,), module.basis, ())
    )
    assert operation != object()
