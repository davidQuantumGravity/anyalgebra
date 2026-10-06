"""Contract tests for sparse arbitrary-arity multilinear structure constants."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from anyalgebra.algebra.multilinear import (
    FiniteMultilinearStructure,
    MultilinearDefinitionError,
    MultilinearOperation,
    StructureConstants,
)
from anyalgebra.core.domains import Domain, QQ, ZZ
from anyalgebra.core.modules import Basis, FreeModule


def _module(rank: int, *, domain: Domain[object] | None = None) -> FreeModule:
    coefficient_domain = ZZ() if domain is None else domain
    return FreeModule(
        coefficient_domain,
        Basis(
            tuple(f"e{index}" for index in range(rank)),
            coefficient_domain=coefficient_domain,
        ),
    )


def _constants(module: FreeModule, arity: int, entries: object) -> StructureConstants:
    return StructureConstants.from_sparse(
        (module.basis for _ in range(arity)), module.basis, cast(Any, entries)
    )


def test_binary_constants_are_sparse_canonical_and_evaluate_to_module_elements() -> (
    None
):
    module = _module(2)
    constants = _constants(
        module,
        2,
        (
            ((1, 0, 1), ZZ().element(3)),
            ((0, 1, 0), ZZ().element(-2)),
            ((1, 1, 1), ZZ().element(0)),
        ),
    )
    structure = FiniteMultilinearStructure.from_structure_constants(module, constants)

    assert (
        constants.key_shape
        == "(input_index_0, ..., input_index_{arity-1}, output_index)"
    )
    assert constants.entries == (
        ((0, 1, 0), ZZ().element(-2)),
        ((1, 0, 1), ZZ().element(3)),
    )
    assert structure.operation.coefficients_for_basis(1, 0) == ((1, ZZ().element(3)),)
    assert tuple(structure.evaluate_basis(0, 1).coordinates().items()) == (
        (0, ZZ().element(-2)),
    )
    assert structure.evaluate_basis(1, 1) == module.zero()


def test_output_coordinates_follow_declared_basis_index_order() -> None:
    module = _module(2)
    structure = FiniteMultilinearStructure.from_structure_constants(
        module,
        _constants(
            module,
            2,
            (
                ((0, 0, 1), ZZ().element(3)),
                ((0, 0, 0), ZZ().element(-2)),
            ),
        ),
    )

    assert structure.operation.coefficients_for_basis(0, 0) == (
        (0, ZZ().element(-2)),
        (1, ZZ().element(3)),
    )


def test_unary_ternary_and_nullary_constants_preserve_the_declared_arity() -> None:
    module = _module(2)
    unary = FiniteMultilinearStructure.from_structure_constants(
        module,
        _constants(module, 1, (((0, 1), ZZ().element(5)),)),
    )
    ternary = FiniteMultilinearStructure.from_structure_constants(
        module,
        _constants(module, 3, (((1, 0, 1, 0), ZZ().element(7)),)),
    )
    nullary = FiniteMultilinearStructure.from_structure_constants(
        module,
        _constants(module, 0, (((1,), ZZ().element(11)),)),
    )

    assert unary.arity == 1
    assert tuple(unary.evaluate_basis(0).coordinates().items()) == (
        (1, ZZ().element(5)),
    )
    assert ternary.arity == 3
    assert tuple(ternary.evaluate_basis(1, 0, 1).coordinates().items()) == (
        (0, ZZ().element(7)),
    )
    assert nullary.arity == 0
    assert tuple(nullary.evaluate_basis().coordinates().items()) == (
        (1, ZZ().element(11)),
    )


def test_257_ary_singleton_structure_never_allocates_a_dense_tensor() -> None:
    module = _module(1)
    inputs = (0,) * 257
    constants = _constants(module, 257, (((*inputs, 0), ZZ().element(1)),))
    structure = FiniteMultilinearStructure.from_structure_constants(module, constants)

    assert constants.arity == 257
    assert len(constants.entries) == 1
    assert tuple(structure.evaluate_basis(*inputs).coordinates().items()) == (
        (0, ZZ().element(1)),
    )


def test_empty_and_explicit_zero_constants_have_empty_sparse_support() -> None:
    module = _module(2)
    empty = _constants(module, 2, ())
    zero = _constants(module, 2, (((0, 0, 0), ZZ().element(0)),))

    assert empty.entries == ()
    assert zero.entries == ()
    assert (
        FiniteMultilinearStructure.from_structure_constants(
            module, empty
        ).evaluate_basis(0, 0)
        == module.zero()
    )


def test_basis_evaluation_rejects_wrong_arity_and_noncanonical_indices() -> None:
    module = _module(1)
    structure = FiniteMultilinearStructure.from_structure_constants(
        module, _constants(module, 2, ())
    )

    with pytest.raises(MultilinearDefinitionError, match="wrong number"):
        structure.evaluate_basis(0)
    with pytest.raises(MultilinearDefinitionError, match="input basis range"):
        structure.evaluate_basis(True, 0)


@pytest.mark.parametrize(
    ("entry", "reason"),
    (
        (([0, 0], ZZ().element(1)), "exact tuple"),
        (((0,), ZZ().element(1)), "wrong length"),
        (((True, 0), ZZ().element(1)), "exact non-negative"),
        (((-1, 0), ZZ().element(1)), "exact non-negative"),
        (((1, 0), ZZ().element(1)), "outside the declared basis range"),
    ),
)
def test_rejects_wrong_key_shape_and_index_boundaries(
    entry: object, reason: str
) -> None:
    module = _module(1)
    with pytest.raises(MultilinearDefinitionError, match=reason):
        _constants(module, 1, (entry,))


def test_rejects_wrong_bases_coefficients_and_foreign_parent_data() -> None:
    module = _module(1)
    qq_module = _module(1, domain=QQ())
    with pytest.raises(MultilinearDefinitionError, match="exact Basis"):
        StructureConstants.from_sparse((object(),), module.basis, ())  # type: ignore[arg-type]
    with pytest.raises(MultilinearDefinitionError, match="output_basis"):
        StructureConstants.from_sparse((module.basis,), object(), ())  # type: ignore[arg-type]
    with pytest.raises(MultilinearDefinitionError, match="coefficient parent"):
        StructureConstants.from_sparse((module.basis,), qq_module.basis, ())
    with pytest.raises(MultilinearDefinitionError, match="exact domain element"):
        _constants(module, 1, (((0, 0), 1),))
    with pytest.raises(MultilinearDefinitionError, match="literal coefficient parent"):
        _constants(module, 1, (((0, 0), QQ().element(1)),))
    constants = _constants(module, 1, ())
    with pytest.raises(MultilinearDefinitionError, match="literal module basis"):
        FiniteMultilinearStructure.from_structure_constants(_module(1), constants)


def test_snapshots_mapping_and_iterable_once_and_rejects_duplicate_keys() -> None:
    module = _module(1)
    entries = [((0, 0), ZZ().element(2))]
    constants = _constants(module, 1, (item for item in entries))
    entries.clear()
    from_mapping = _constants(module, 1, {(0, 0): ZZ().element(3)})

    assert constants.entries == (((0, 0), ZZ().element(2)),)
    assert from_mapping.entries == (((0, 0), ZZ().element(3)),)
    with pytest.raises(MultilinearDefinitionError, match="duplicate"):
        _constants(
            module,
            1,
            (((0, 0), ZZ().element(2)), ((0, 0), ZZ().element(2))),
        )
    with pytest.raises(MultilinearDefinitionError, match="duplicate"):
        _constants(
            module,
            1,
            (((0, 0), ZZ().element(0)), ((0, 0), ZZ().element(2))),
        )
    with pytest.raises(MultilinearDefinitionError, match="exact tuple"):
        _constants(module, 1, ([((0, 0), ZZ().element(2))],))


def test_rank_zero_boundary_accepts_only_empty_sparse_constants() -> None:
    module = _module(0)
    constants = _constants(module, 0, ())
    structure = FiniteMultilinearStructure.from_structure_constants(module, constants)

    assert structure.evaluate_basis() == module.zero()
    with pytest.raises(
        MultilinearDefinitionError, match="outside the declared basis range"
    ):
        _constants(module, 0, (((0,), ZZ().element(1)),))


def test_direct_construction_and_factory_type_bypasses_are_rejected() -> None:
    module = _module(1)
    constants = _constants(module, 1, ())
    with pytest.raises(MultilinearDefinitionError, match="from_sparse"):
        StructureConstants((), module.basis, ())
    with pytest.raises(MultilinearDefinitionError, match="from_structure_constants"):
        MultilinearOperation(constants)
    with pytest.raises(MultilinearDefinitionError, match="from_structure_constants"):
        FiniteMultilinearStructure(module, constants)
    with pytest.raises(MultilinearDefinitionError, match="exact StructureConstants"):
        MultilinearOperation.from_structure_constants(object())  # type: ignore[arg-type]
    with pytest.raises(MultilinearDefinitionError, match="exact FreeModule"):
        FiniteMultilinearStructure.from_structure_constants(
            object(),  # type: ignore[arg-type]
            constants,
        )
    with pytest.raises(MultilinearDefinitionError, match="exact StructureConstants"):
        FiniteMultilinearStructure.from_structure_constants(
            module,
            object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("name", ("", " ", " name", "name ", 1, True))
def test_structure_name_validation(name: object) -> None:
    module = _module(1)
    with pytest.raises(MultilinearDefinitionError, match="name"):
        FiniteMultilinearStructure.from_structure_constants(
            module,
            _constants(module, 1, ()),
            name=name,  # type: ignore[arg-type]
        )


def test_normalizes_iterators_and_uncertain_coefficients() -> None:
    module = _module(1)

    def broken() -> object:
        yield ((0, 0), ZZ().element(1))
        raise RuntimeError("broken iterator")

    with pytest.raises(MultilinearDefinitionError, match="snapshotted"):
        _constants(module, 1, broken())
    domain = _UncertainDomain()
    uncertain_module = _module(1, domain=domain)
    coefficient = domain.element("not provably zero")
    constants = _constants(uncertain_module, 1, (((0, 0), coefficient),))

    assert constants.entries == (((0, 0), coefficient),)
    assert constants != _constants(
        uncertain_module, 1, (((0, 0), domain.element("not provably zero")),)
    )


def test_records_are_frozen_slotted_unhashable_and_distinguish_module_parents() -> None:
    module = _module(1)
    constants = _constants(module, 1, (((0, 0), ZZ().element(1)),))
    operation = MultilinearOperation.from_structure_constants(constants)
    left = FiniteMultilinearStructure.from_structure_constants(module, constants)
    other_module = _module(1)
    right = FiniteMultilinearStructure.from_structure_constants(
        other_module,
        _constants(other_module, 1, (((0, 0), ZZ().element(1)),)),
    )

    for value in (constants, operation, left):
        assert not hasattr(value, "__dict__")
        with pytest.raises(TypeError, match="unhashable"):
            hash(value)
    with pytest.raises(FrozenInstanceError):
        constants.entries = ()  # type: ignore[misc]
    assert left != right


class _UncertainElement:
    """A deliberately unhashable coefficient whose equality cannot prove zero."""

    __hash__ = None  # type: ignore[assignment]

    def __init__(self, parent: _UncertainDomain, value: object) -> None:
        self._parent = parent
        self._value = value

    @property
    def parent(self) -> _UncertainDomain:
        return self._parent

    @property
    def value(self) -> object:
        return self._value

    def __eq__(self, other: object) -> object:  # type: ignore[override]
        del other
        raise RuntimeError("equality is unavailable")


class _UncertainDomain:
    """A minimal exact parent whose elements deliberately have no zero test."""

    def normalize(self, value: object) -> object:
        return value

    def element(self, value: object) -> _UncertainElement:
        if type(value) is _UncertainElement and value.parent is self:
            return value
        return _UncertainElement(self, value)
