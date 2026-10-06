"""Exact public evaluation on arbitrary tables and composition fixtures."""

from __future__ import annotations

import json
from typing import Any, TypedDict, cast

from anyalgebra.algebra.multilinear import (
    FiniteMultilinearStructure,
    StructureConstants,
    evaluate_multilinear,
)
from anyalgebra.core.domains import ZZ
from anyalgebra.core.elements import SparseElement
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.fixtures.composition import (
    octonion_fixture,
    quaternion_fixture,
    split_octonion_fixture,
    split_quaternion_fixture,
)


class MultilinearEvaluationExample(TypedDict):
    """Objects used by the executable example and its tests."""

    generic: FiniteMultilinearStructure
    left_nested: SparseElement
    right_nested: SparseElement
    ternary_result: SparseElement
    quaternion: FiniteMultilinearStructure
    quaternion_product: SparseElement
    octonion: FiniteMultilinearStructure
    octonion_product: SparseElement
    split_zero_divisors: dict[str, SparseElement]
    identity_checks: dict[str, bool]


def _structure(
    module: FreeModule,
    arity: int,
    entries: tuple[tuple[tuple[int, ...], int], ...],
    *,
    name: str,
) -> FiniteMultilinearStructure:
    constants = StructureConstants.from_sparse(
        tuple(module.basis for _ in range(arity)),
        module.basis,
        tuple((key, module.domain.element(value)) for key, value in entries),
    )
    return FiniteMultilinearStructure.from_structure_constants(
        module, constants, name=name
    )


def _identity_check(structure: FiniteMultilinearStructure) -> bool:
    one = structure.module.element({0: 1})
    value = structure.module.element(
        {index: index + 1 for index in range(structure.module.rank)}
    )
    return (
        evaluate_multilinear(structure, one, value) == value
        and evaluate_multilinear(structure, value, one) == value
    )


def build_example() -> MultilinearEvaluationExample:
    """Build unrelated exact operations and four named composition controls."""
    module = FreeModule(ZZ(), Basis(("a", "b"), coefficient_domain=ZZ()))
    generic = _structure(
        module,
        2,
        (((0, 0, 1), 1), ((1, 0, 0), 1)),
        name="example.nonassociative.v1",
    )
    ternary = _structure(
        module,
        3,
        (((0, 1, 0, 1), 3),),
        name="example.ternary.v1",
    )
    a = module.element({0: 1})
    left_nested = evaluate_multilinear(generic, evaluate_multilinear(generic, a, a), a)
    right_nested = evaluate_multilinear(generic, a, evaluate_multilinear(generic, a, a))
    ternary_result = evaluate_multilinear(
        ternary,
        module.element({0: 2}),
        module.element({1: 5}),
        module.element({0: -1}),
    )

    quaternion = quaternion_fixture()
    quaternion_product = evaluate_multilinear(
        quaternion,
        quaternion.module.element({0: 1, 1: 2}),
        quaternion.module.element({2: 1, 3: 1}),
    )
    octonion = octonion_fixture()
    octonion_product = evaluate_multilinear(
        octonion,
        octonion.module.element({0: 1, 1: 1}),
        octonion.module.element({2: 1, 4: 1}),
    )

    split_quaternion = split_quaternion_fixture()
    split_octonion = split_octonion_fixture()
    split_zero_divisors = {
        cast(str, split_quaternion.name): evaluate_multilinear(
            split_quaternion,
            split_quaternion.module.element({0: 1, 2: 1}),
            split_quaternion.module.element({0: 1, 2: -1}),
        ),
        cast(str, split_octonion.name): evaluate_multilinear(
            split_octonion,
            split_octonion.module.element({0: 1, 4: 1}),
            split_octonion.module.element({0: 1, 4: -1}),
        ),
    }
    fixtures = (quaternion, split_quaternion, octonion, split_octonion)
    identity_checks = {
        cast(str, structure.name): _identity_check(structure) for structure in fixtures
    }
    return {
        "generic": generic,
        "left_nested": left_nested,
        "right_nested": right_nested,
        "ternary_result": ternary_result,
        "quaternion": quaternion,
        "quaternion_product": quaternion_product,
        "octonion": octonion,
        "octonion_product": octonion_product,
        "split_zero_divisors": split_zero_divisors,
        "identity_checks": identity_checks,
    }


def _integer_coordinates(element: SparseElement) -> list[list[int]]:
    return [
        [index, cast(Any, coefficient).value]
        for index, coefficient in element.coordinates().items()
    ]


def run_example() -> dict[str, object]:
    """Return deterministic exact facts without promoting a research claim."""
    example = build_example()
    return {
        "generic_nonassociative": {
            "fixture": cast(str, example["generic"].name),
            "left_nested": _integer_coordinates(example["left_nested"]),
            "right_nested": _integer_coordinates(example["right_nested"]),
        },
        "ternary_result": _integer_coordinates(example["ternary_result"]),
        "quaternion": {
            "fixture": cast(str, example["quaternion"].name),
            "product": _integer_coordinates(example["quaternion_product"]),
        },
        "octonion": {
            "fixture": cast(str, example["octonion"].name),
            "product": _integer_coordinates(example["octonion_product"]),
        },
        "split_zero_divisors": {
            name: _integer_coordinates(value)
            for name, value in example["split_zero_divisors"].items()
        },
        "identity_checks": example["identity_checks"],
        "scope": (
            "exact finite-table evaluation controls only; no classification or "
            "physics claim"
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), sort_keys=True, separators=(",", ":")))
