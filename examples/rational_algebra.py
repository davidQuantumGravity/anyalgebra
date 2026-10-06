"""A small exact QQ noncommutative algebra; this example makes no research claim."""

from __future__ import annotations

import json
from itertools import product
from typing import Any, TypedDict, cast

from anyalgebra.algebra.multilinear import (
    BasisProductTable,
    FiniteMultilinearStructure,
    StructureConstants,
)
from anyalgebra.core.domains import QQ, Domain
from anyalgebra.core.elements import SparseElement
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.presentations.basis_change import ExactBasisIsomorphism, change_basis


class RationalAlgebraExample(TypedDict):
    """Public objects needed to replay table, product, and transport facts."""

    basis_products: dict[str, SparseElement]
    domain: Domain[object]
    expected_generic_product: SparseElement
    generic_product: SparseElement
    isomorphism: ExactBasisIsomorphism
    module: FreeModule
    structure: FiniteMultilinearStructure
    table: BasisProductTable
    table_round_trip: BasisProductTable
    transported_constants: StructureConstants


def multiply(
    constants: StructureConstants, left: SparseElement, right: SparseElement
) -> SparseElement:
    """Evaluate a binary sparse tensor on two generic exact module elements."""
    module = left.parent
    if right.parent is not module:
        raise ValueError("generic product requires one literal module parent")
    result = module.zero()
    for left_index, left_coefficient in left.coordinates().items():
        for right_index, right_coefficient in right.coordinates().items():
            for output_index, coefficient in constants.coefficients_for_basis(
                left_index, right_index
            ):
                term = module.element({output_index: coefficient})
                result = result.add(
                    term.scale(left_coefficient).scale(right_coefficient)
                )
    return result


def _coordinates(value: SparseElement) -> list[list[object]]:
    """Serialize exact QQ coordinates as stable numerator/denominator pairs."""
    return [
        [
            index,
            cast(Any, coefficient).value.numerator,
            cast(Any, coefficient).value.denominator,
        ]
        for index, coefficient in value.coordinates().items()
    ]


def build_example() -> RationalAlgebraExample:
    """Construct a rank-two QQ algebra with a visibly noncommutative product."""
    domain = QQ()
    module = FreeModule(domain, Basis(("p", "q"), coefficient_domain=domain))
    table = BasisProductTable.from_cells(
        module,
        2,
        (
            module.zero(),
            module.element({0: (3, 2)}),
            module.element({1: (-1, 2)}),
            module.element({1: 1}),
        ),
    )
    structure = FiniteMultilinearStructure.from_tables(module, table)
    constants = structure.operation.constants
    generic_product = multiply(
        constants, module.element({0: 1, 1: 2}), module.element({0: 2, 1: -1})
    )
    expected_generic_product = module.element({0: (-3, 2), 1: -4})
    table_round_trip = BasisProductTable.from_cells(
        module,
        2,
        tuple(
            structure.evaluate_basis(*indices)
            for indices in product(range(2), repeat=2)
        ),
    )
    target = FreeModule(domain, Basis(("u", "v"), coefficient_domain=domain))
    isomorphism = ExactBasisIsomorphism.from_images(
        module,
        target,
        (target.element({1: 1}), target.element({0: 1})),
        (module.element({1: 1}), module.element({0: 1})),
    )
    transported_constants = cast(
        StructureConstants, change_basis(constants, isomorphism)
    )
    return {
        "basis_products": {
            "p_p": structure.evaluate_basis(0, 0),
            "p_q": structure.evaluate_basis(0, 1),
            "q_p": structure.evaluate_basis(1, 0),
            "q_q": structure.evaluate_basis(1, 1),
        },
        "domain": domain,
        "expected_generic_product": expected_generic_product,
        "generic_product": generic_product,
        "isomorphism": isomorphism,
        "module": module,
        "structure": structure,
        "table": table,
        "table_round_trip": table_round_trip,
        "transported_constants": transported_constants,
    }


def run_example() -> dict[str, object]:
    """Return deterministic exact facts for the finite infrastructure example."""
    example = build_example()
    transported = example["transported_constants"]
    return {
        "basis_products": {
            name: _coordinates(value)
            for name, value in example["basis_products"].items()
        },
        "basis_change": {
            "certificate": example["isomorphism"].certificate.algorithm,
            "certificate_counts": [
                example["isomorphism"].certificate.source_rank,
                example["isomorphism"].certificate.target_rank,
                example["isomorphism"].certificate.expected_forward_after_inverse_count,
                example[
                    "isomorphism"
                ].certificate.evaluated_forward_after_inverse_count,
                example["isomorphism"].certificate.expected_inverse_after_forward_count,
                example[
                    "isomorphism"
                ].certificate.evaluated_inverse_after_forward_count,
            ],
            "exact_certified": example["isomorphism"].certificate.exact_certified,
            "transported_constants": [
                [
                    *key,
                    cast(Any, coefficient).value.numerator,
                    cast(Any, coefficient).value.denominator,
                ]
                for key, coefficient in transported.entries
            ],
        },
        "generic_product": _coordinates(example["generic_product"]),
        "generic_product_matches_expected": (
            example["generic_product"] == example["expected_generic_product"]
        ),
        "scope": (
            "finite exact algebra infrastructure example only; "
            "no research claim verified"
        ),
        "table_constants_round_trip": example["table_round_trip"] == example["table"],
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), sort_keys=True, separators=(",", ":")))
