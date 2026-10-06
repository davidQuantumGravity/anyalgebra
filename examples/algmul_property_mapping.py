"""Show the corrected typed replacement for AlgMul's generated properties.

The legacy ``MakeProperty`` surface is inventory evidence only.  This example
constructs the replacement law from typed operations, validates it on two
hand-checkable finite structures, and keeps the proof and counterexample
bounds explicit.  It makes no claim that the pinned Mathematica load generated
the absent legacy functions and no scientific claim.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Final

from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.legacy.property_mapping import (
    PropertyMappingError,
    intended_property_members,
    makeproperty_law,
    property_factory_record,
    property_laws,
    property_surfaces,
)
from anyalgebra.structures.operations import Operation
from anyalgebra.structures.signatures import OperationSymbol, Signature
from anyalgebra.structures.structure import Structure, StructureBuilder
from anyalgebra.validation.validate import Disproved, Proved, validate_law


ENUMERATION_POLICY: Final = "finite_substitution_domain_lexicographic"
SCOPE: Final = (
    "typed finite replacement behavior only; legacy captures are secondary "
    "reproduction evidence and no scientific claim is verified"
)


def _binary_structure(
    name: str, rule: Callable[[int, int], int]
) -> tuple[Structure, OperationSymbol]:
    """Build one literal two-element operation from its four table cells."""
    bit = Sort("bit")
    symbol = OperationSymbol(name, (bit, bit), bit)
    carrier = FiniteCarrier((0, 1), sort=bit)
    operation = Operation.from_table(
        symbol,
        ((bit, carrier),),
        tuple(
            ((left, right), rule(left, right)) for left in (0, 1) for right in (0, 1)
        ),
    )
    structure = (
        StructureBuilder(Signature((bit,), (symbol,)))
        .with_carrier(bit, carrier)
        .with_operation(symbol, operation)
        .freeze()
    )
    return structure, symbol


def run_example() -> dict[str, object]:
    """Return deterministic mapping, positive, negative, and boundary facts."""
    meet, meet_symbol = _binary_structure("and", lambda left, right: left & right)
    left_projection, projection_symbol = _binary_structure(
        "left_projection", lambda left, _right: left
    )
    proved = validate_law(meet, makeproperty_law("Commutative", meet_symbol))
    disproved = validate_law(
        left_projection, makeproperty_law("Commutative", projection_symbol)
    )
    assert type(proved) is Proved
    assert type(disproved) is Disproved and disproved.witness is not None

    try:
        makeproperty_law("Jacobi", meet_symbol)
    except PropertyMappingError as error:
        boundary = {"field": error.field, "reason": error.reason}
    else:  # pragma: no cover - the checked API contract requires this rejection.
        raise AssertionError("Jacobi accepted no explicit addition or zero")

    members = intended_property_members()
    defect_names = sorted(
        member.name for member in members if member.caveat.startswith("legacy_defect:")
    )
    factory = property_factory_record()
    return {
        "inventory": {
            "factory": factory.name,
            "factory_legacy_status": factory.legacy_status,
            "law_count": len(property_laws()),
            "surface_count": len(property_surfaces()),
            "intended_name_count": len(members),
            "unique_name_count": len({member.name for member in members}),
            "localized_defect_names": defect_names,
        },
        "positive": {
            "law": "Commutative",
            "outcome": type(proved).__name__,
            "expected_assignments": proved.expected_assignments,
            "evaluated_assignments": proved.evaluated_assignments,
            "enumeration_policy": proved.enumeration_policy,
        },
        "negative": {
            "law": "Commutative",
            "outcome": type(disproved).__name__,
            "expected_assignments": disproved.expected_assignments,
            "evaluated_assignments": disproved.evaluated_assignments,
            "witness_indices": list(disproved.witness.substitution_indices),
            "enumeration_policy": disproved.enumeration_policy,
        },
        "boundary": boundary,
        "scope": SCOPE,
    }


def main() -> None:
    """Print one stable JSON record suitable for exact documentation tests."""
    print(json.dumps(run_example(), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
