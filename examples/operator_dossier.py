"""Exact presentation and operator-recovery dossier; no scientific claim is made."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TypedDict, cast

from anyalgebra.algebra.multilinear import BasisProductTable, FiniteMultilinearStructure
from anyalgebra.core.domains import QQ
from anyalgebra.core.elements import SparseElement
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.linear.matrix import Matrix, MatrixSpace
from anyalgebra.linear.recovery import RecoveryFailure, RecoveryReport, RecoverySuccess
from anyalgebra.linear.span import NonClosed
from anyalgebra.presentations.adapters import BasisExpression, PresentationAdapterBundle
from anyalgebra.presentations.basis_change import ExactBasisIsomorphism
from anyalgebra.presentations.basis_change import change_basis
from anyalgebra.presentations.convert import (
    ConversionAmbiguityError,
    convert,
    plan_conversion,
)
from anyalgebra.presentations.graph import Conversion

if __package__ is None:  # Supports direct execution from the examples directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples import operator_recovery


class PresentationScenario(TypedDict):
    """Explicit conversions retained with their source representations."""

    ambiguity_candidates: tuple[tuple[tuple[str, str], ...], ...]
    bundle: PresentationAdapterBundle
    symbolic: BasisExpression
    symbolic_round_trip: BasisExpression
    table: BasisProductTable
    table_round_trip: BasisProductTable


class OperatorDossier(TypedDict):
    """Exact inputs and outcomes retained for bounded replay."""

    basis_change: ExactBasisIsomorphism
    basis_change_value: SparseElement
    matrix: Matrix
    matrix_space: MatrixSpace
    presentation: PresentationScenario
    recovery: operator_recovery.OperatorRecoveryExample


def _presentation_scenario() -> PresentationScenario:
    """Build named routes, including an intentional unselected ambiguity."""
    domain = QQ()
    module = FreeModule(domain, Basis(("e", "f"), coefficient_domain=domain))
    bundle = PresentationAdapterBundle.for_module(module, "example.operator-dossier", 2)
    symbolic = BasisExpression.from_terms(module, (("e", 1), ("f", -2)))
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
        "example.operator-dossier.alternate-symbolic-coordinate",
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
        raise RuntimeError("the deliberately parallel route was selected")
    return {
        "ambiguity_candidates": ambiguity_candidates,
        "bundle": bundle,
        "symbolic": symbolic,
        "symbolic_round_trip": symbolic_round_trip,
        "table": table,
        "table_round_trip": table_round_trip,
    }


def _basis_change() -> tuple[ExactBasisIsomorphism, SparseElement]:
    """Construct a certified permutation rather than inferring a coordinate route."""
    domain = QQ()
    source = FreeModule(domain, Basis(("e", "f"), coefficient_domain=domain))
    target = FreeModule(domain, Basis(("u", "v"), coefficient_domain=domain))
    change = ExactBasisIsomorphism.from_images(
        source,
        target,
        (target.element({1: 1}), target.element({0: 1})),
        (source.element({1: 1}), source.element({0: 1})),
    )
    return change, source.element({0: 1, 1: -2})


def dependency_replays(family: tuple[Matrix, ...], relation: Matrix) -> bool:
    """Replay the retained dependency relation with the current public helper."""
    return operator_recovery.dependency_replays(family, relation)


def nonclosure_witness_replays(report: RecoveryReport) -> bool:
    """Replay the separating functional with the current public helper."""
    return operator_recovery.nonclosure_witness_replays(report)


def _coordinates(value: SparseElement) -> list[list[int]]:
    """Serialize exact sparse coordinates without exposing implementation state."""
    return [
        [
            index,
            cast(Any, coefficient).value.numerator,
            cast(Any, coefficient).value.denominator,
        ]
        for index, coefficient in value.coordinates().items()
    ]


def _matrix_entries(value: Matrix) -> list[list[list[int]]]:
    """Serialize the matrix only after its explicit MatrixSpace construction."""
    return [
        [
            [cast(Any, entry).value.numerator, cast(Any, entry).value.denominator]
            for entry in row
        ]
        for row in value.entries
    ]


def build_example() -> OperatorDossier:
    """Build the finite, exact dossier using current public APIs only."""
    domain = QQ()
    matrix_space = MatrixSpace(1, 2, domain)
    matrix = matrix_space.element(((1, 2),))
    basis_change, basis_change_value = _basis_change()
    return {
        "basis_change": basis_change,
        "basis_change_value": basis_change_value,
        "matrix": matrix,
        "matrix_space": matrix_space,
        "presentation": _presentation_scenario(),
        "recovery": operator_recovery.build_example(),
    }


def run_example() -> dict[str, object]:
    """Return deterministic route, parent, closure, and witness facts."""
    dossier = build_example()
    presentation = dossier["presentation"]
    bundle = presentation["bundle"]
    basis_change = dossier["basis_change"]
    basis_change_value = dossier["basis_change_value"]
    transported = cast(SparseElement, change_basis(basis_change_value, basis_change))
    restored = change_basis(transported, basis_change.reversed())
    recovery = dossier["recovery"]
    closed = recovery["closed"]["report"]
    dependent = recovery["dependent"]["report"]
    nonclosed = recovery["nonclosed"]["report"]
    if type(closed) is not RecoverySuccess:
        raise RuntimeError("closed family did not return unique constants")
    if type(dependent) is not RecoveryFailure or not dependent.dependencies:
        raise RuntimeError("dependent family did not retain a relation")
    if (
        type(nonclosed) is not RecoveryFailure
        or type(nonclosed.closure_evidence) is not NonClosed
    ):
        raise RuntimeError("nonclosed family did not retain closure evidence")
    return {
        "basis_change": {
            "algorithm": basis_change.certificate.algorithm,
            "exact_inverse": restored == basis_change_value,
            "source_coordinates": _coordinates(basis_change_value),
            "target_coordinates": _coordinates(transported),
        },
        "matrix": {
            "columns": dossier["matrix_space"].columns,
            "entries": _matrix_entries(dossier["matrix"]),
            "entry_parent": "QQ",
            "rows": dossier["matrix_space"].rows,
        },
        "presentation": {
            "ambiguity_auto_selected": False,
            "candidate_route_count": len(presentation["ambiguity_candidates"]),
            "container_shape_inference": "not attempted; source kind is explicit",
            "symbolic_coordinate_round_trip": presentation["symbolic_round_trip"]
            == presentation["symbolic"],
            "symbolic_coordinate_route": [
                [bundle.symbolic_to_coordinate.stable_id, "forward"],
                [bundle.symbolic_to_coordinate.stable_id, "inverse"],
            ],
            "symbolic_coordinate_validity_domain": (
                bundle.symbolic_to_coordinate.validity_domain
            ),
            "table_structure_round_trip": presentation["table_round_trip"]
            == presentation["table"],
            "table_structure_route": [
                [bundle.table_to_structure.stable_id, "forward"],
                [bundle.table_to_structure.stable_id, "inverse"],
            ],
            "table_structure_validity_domain": (
                bundle.table_to_structure.validity_domain
            ),
        },
        "recovery": {
            "closed": {
                "constants_returned": True,
                "reconstruction_exact": [
                    check.exact for check in closed.reconstruction_checks
                ],
            },
            "dependent": {
                "constants_returned": False,
                "dependency_replays": dependency_replays(
                    recovery["dependent"]["family"], dependent.dependencies[0]
                ),
            },
            "nonclosed": {
                "constants_returned": False,
                "first_pair": [
                    nonclosed.closure_evidence.left_index,
                    nonclosed.closure_evidence.right_index,
                ],
                "witness_replays": nonclosure_witness_replays(nonclosed),
            },
        },
        "scope": (
            "finite exact infrastructure dossier only; "
            "no scientific or legacy correctness claim verified"
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), sort_keys=True, separators=(",", ":")))
