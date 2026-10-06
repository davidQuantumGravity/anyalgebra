"""Independent exact properties for public multilinear element evaluation."""

from __future__ import annotations

from itertools import product
from typing import cast

import pytest

from anyalgebra.algebra.multilinear import (
    FiniteMultilinearStructure,
    StructureConstants,
    evaluate_multilinear,
)
from anyalgebra.core.domains import DomainElement, QQ, ZZ
from anyalgebra.core.elements import SparseElement
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.presentations.basis_change import ExactBasisIsomorphism, change_basis


def _module(
    *, rational: bool = False, labels: tuple[str, str] = ("e0", "e1")
) -> FreeModule:
    domain = QQ() if rational else ZZ()
    return FreeModule(domain, Basis(labels, coefficient_domain=domain))


def _generated_structure(module: FreeModule, arity: int) -> FiniteMultilinearStructure:
    entries: list[tuple[tuple[int, ...], DomainElement[object]]] = []
    for inputs in product(range(module.rank), repeat=arity):
        weighted_inputs = sum(
            (position + 1) * (index + 1) for position, index in enumerate(inputs)
        )
        for output in range(module.rank):
            coefficient = (weighted_inputs + 2 * (output + 1) + arity) % 5 - 2
            if coefficient:
                entries.append(((*inputs, output), module.domain.element(coefficient)))
    constants = StructureConstants.from_sparse(
        tuple(module.basis for _ in range(arity)), module.basis, entries
    )
    return FiniteMultilinearStructure.from_structure_constants(module, constants)


def _dense_basis_oracle(
    structure: FiniteMultilinearStructure, elements: tuple[SparseElement, ...]
) -> SparseElement:
    """Expand every basis tuple without reading sparse constant entries."""
    coordinate_maps = tuple(element.coordinates() for element in elements)
    result = structure.module.zero()
    for indices in product(range(structure.module.rank), repeat=structure.arity):
        if any(
            basis_index not in coordinates
            for basis_index, coordinates in zip(indices, coordinate_maps, strict=True)
        ):
            continue
        term = structure.evaluate_basis(*indices)
        for basis_index, coordinates in zip(indices, coordinate_maps, strict=True):
            term = term.scale(coordinates[basis_index])
        result = result.add(term)
    return result


def _controls(module: FreeModule) -> tuple[SparseElement, ...]:
    return (
        module.zero(),
        module.element({0: 1}),
        module.element({1: -1}),
        module.element({0: 1, 1: 1}),
        module.element({0: 2, 1: -1}),
    )


@pytest.mark.exhaustive
@pytest.mark.parametrize("arity", range(5))
def test_generated_sparse_inputs_agree_with_independent_dense_oracle(
    arity: int,
) -> None:
    module = _module()
    structure = _generated_structure(module, arity)
    controls = _controls(module)

    checked = 0
    for elements in product(controls, repeat=arity):
        assert evaluate_multilinear(structure, *elements) == _dense_basis_oracle(
            structure, elements
        )
        checked += 1

    assert checked == len(controls) ** arity


@pytest.mark.parametrize("arity", (1, 2, 3, 4))
def test_evaluator_is_additive_and_integer_homogeneous_in_every_slot(
    arity: int,
) -> None:
    module = _module()
    structure = _generated_structure(module, arity)
    fixed = [module.element({0: 1, 1: -1}) for _ in range(arity)]
    left = module.element({0: 2, 1: 1})
    right = module.element({0: -1, 1: 3})

    for slot in range(arity):
        summed = fixed.copy()
        summed[slot] = left.add(right)
        left_arguments = fixed.copy()
        left_arguments[slot] = left
        right_arguments = fixed.copy()
        right_arguments[slot] = right

        assert evaluate_multilinear(structure, *summed) == evaluate_multilinear(
            structure, *left_arguments
        ).add(evaluate_multilinear(structure, *right_arguments))

        scaled = fixed.copy()
        scaled[slot] = left.scale(-3)
        assert evaluate_multilinear(structure, *scaled) == evaluate_multilinear(
            structure, *left_arguments
        ).scale(-3)


def _shear(source: FreeModule) -> ExactBasisIsomorphism:
    target = _module(rational=True, labels=("f0", "f1"))
    return ExactBasisIsomorphism.from_images(
        source,
        target,
        (target.element({0: 1, 1: 1}), target.element({1: 1})),
        (source.element({0: 1, 1: -1}), source.element({1: 1})),
    )


@pytest.mark.exhaustive
@pytest.mark.parametrize("arity", range(4))
def test_exact_basis_transport_commutes_with_evaluation(arity: int) -> None:
    source_module = _module(rational=True)
    source = _generated_structure(source_module, arity)
    shear = _shear(source_module)
    transported_constants = cast(
        StructureConstants, change_basis(source.operation.constants, shear)
    )
    target = FiniteMultilinearStructure.from_structure_constants(
        shear.target, transported_constants
    )
    controls = _controls(source_module)[:4]

    checked = 0
    for source_elements in product(controls, repeat=arity):
        target_elements = tuple(
            cast(SparseElement, change_basis(element, shear))
            for element in source_elements
        )
        source_result = evaluate_multilinear(source, *source_elements)
        expected = cast(SparseElement, change_basis(source_result, shear))

        assert evaluate_multilinear(target, *target_elements) == expected
        checked += 1

    assert checked == len(controls) ** arity


def test_repeated_evaluation_is_deterministic_and_retains_literal_output_parent() -> (
    None
):
    module = _module(rational=True)
    structure = _generated_structure(module, 3)
    elements = _controls(module)[1:4]

    first = evaluate_multilinear(structure, *elements)
    second = evaluate_multilinear(structure, *elements)

    assert first == second
    assert first.parent is module
    assert tuple(first.coordinates()) == tuple(second.coordinates())
