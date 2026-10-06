"""Explicit presentation routes; this executable example makes no research claim."""

from __future__ import annotations

import json
from typing import TypedDict, cast

from anyalgebra.algebra.multilinear import BasisProductTable, FiniteMultilinearStructure
from anyalgebra.core.domains import QQ
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.presentations.adapters import BasisExpression, PresentationAdapterBundle
from anyalgebra.presentations.convert import (
    ConversionAmbiguityError,
    convert,
    plan_conversion,
)
from anyalgebra.presentations.graph import Conversion


class PresentationsExample(TypedDict):
    """Public objects retained so integration can replay each named route."""

    ambiguity_candidates: tuple[tuple[tuple[str, str], ...], ...]
    bundle: PresentationAdapterBundle
    module: FreeModule
    symbolic: BasisExpression
    symbolic_round_trip: BasisExpression
    table: BasisProductTable
    table_round_trip: BasisProductTable


def build_example() -> PresentationsExample:
    """Build two explicit bidirectional routes and one intentional ambiguity."""
    domain = QQ()
    module = FreeModule(domain, Basis(("e", "f"), coefficient_domain=domain))
    bundle = PresentationAdapterBundle.for_module(module, "example.presentations", 2)
    symbolic = BasisExpression.from_terms(module, (("e", (1, 2)), ("f", -1)))
    coordinates = convert(
        symbolic,
        bundle.coordinate_kind,
        graph=bundle.graph,
        source_kind=bundle.symbolic_kind,
    )
    symbolic_round_trip = cast(
        BasisExpression,
        convert(
            coordinates.value,
            bundle.symbolic_kind,
            graph=bundle.graph,
            source_kind=bundle.coordinate_kind,
        ).value,
    )
    table = BasisProductTable.from_cells(
        module,
        2,
        (
            module.element({0: 1}),
            module.element({1: 1}),
            module.zero(),
            module.element({0: -1}),
        ),
    )
    structure = convert(
        table,
        bundle.structure_kind,
        graph=bundle.graph,
        source_kind=bundle.table_kind,
    )
    assert type(structure.value) is FiniteMultilinearStructure
    table_round_trip = cast(
        BasisProductTable,
        convert(
            structure.value,
            bundle.table_kind,
            graph=bundle.graph,
            source_kind=bundle.structure_kind,
        ).value,
    )
    alternate = Conversion.from_callable(
        "example.presentations.alternate-symbolic-coordinate",
        bundle.symbolic_kind,
        bundle.coordinate_kind,
        bundle.coordinates_from_symbolic,
    )
    try:
        plan_conversion(
            bundle.symbolic_kind,
            bundle.coordinate_kind,
            graph=bundle.graph.with_conversion(alternate),
        )
    except ConversionAmbiguityError as error:
        ambiguity_candidates = error.candidates
    else:  # pragma: no cover - guards the public ambiguity contract.
        raise RuntimeError("the deliberately parallel conversion route was selected")
    return {
        "ambiguity_candidates": ambiguity_candidates,
        "bundle": bundle,
        "module": module,
        "symbolic": symbolic,
        "symbolic_round_trip": symbolic_round_trip,
        "table": table,
        "table_round_trip": table_round_trip,
    }


def run_example() -> dict[str, object]:
    """Return compact JSON-safe route facts, not opaque values or proof claims."""
    example = build_example()
    bundle = example["bundle"]
    return {
        "ambiguity": {
            "automatic_route_selected": False,
            "candidate_routes": [
                [list(witness) for witness in candidate]
                for candidate in example["ambiguity_candidates"]
            ],
            "container_shape_inference": "not attempted; source kind is explicit",
        },
        "scope": "infrastructure example only; no research claim verified",
        "symbolic_coordinate": {
            "round_trip_equal": example["symbolic_round_trip"] == example["symbolic"],
            "round_trip_id": bundle.symbolic_to_coordinate.round_trip_id,
            "route": [
                [bundle.symbolic_to_coordinate.stable_id, "forward"],
                [bundle.symbolic_to_coordinate.stable_id, "inverse"],
            ],
            "validity_domain": bundle.symbolic_to_coordinate.validity_domain,
        },
        "table_structure": {
            "round_trip_equal": example["table_round_trip"] == example["table"],
            "round_trip_id": bundle.table_to_structure.round_trip_id,
            "route": [
                [bundle.table_to_structure.stable_id, "forward"],
                [bundle.table_to_structure.stable_id, "inverse"],
            ],
            "validity_domain": bundle.table_to_structure.validity_domain,
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), sort_keys=True, separators=(",", ":")))
