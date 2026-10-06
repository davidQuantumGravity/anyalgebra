"""Executable contract for the integrated presentation/operator dossier."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from anyalgebra.core.domains import QQ
from anyalgebra.linear.recovery import RecoveryFailure, RecoverySuccess
from anyalgebra.linear.span import NonClosed
from examples import operator_dossier


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_dossier_replays_explicit_routes_matrix_parent_and_recovery_witnesses() -> None:
    """The bounded dossier preserves each route and exact recovery distinction."""
    dossier = operator_dossier.build_example()

    presentation = dossier["presentation"]
    assert presentation["symbolic_round_trip"] == presentation["symbolic"]
    assert presentation["table_round_trip"] == presentation["table"]
    assert len(presentation["ambiguity_candidates"]) == 2

    basis_change = dossier["basis_change"]
    transported = basis_change.forward(dossier["basis_change_value"])
    assert basis_change.inverse(transported) == dossier["basis_change_value"]
    assert basis_change.certificate.exact_certified is True

    matrix = dossier["matrix"]
    assert matrix.parent is dossier["matrix_space"]
    assert matrix.entries == ((QQ().element(1), QQ().element(2)),)

    recovery = dossier["recovery"]
    assert type(recovery["closed"]["report"]) is RecoverySuccess
    assert type(recovery["dependent"]["report"]) is RecoveryFailure
    assert type(recovery["nonclosed"]["report"]) is RecoveryFailure
    assert operator_dossier.dependency_replays(
        recovery["dependent"]["family"], recovery["dependent"]["report"].dependencies[0]
    )
    assert type(recovery["nonclosed"]["report"].closure_evidence) is NonClosed
    assert operator_dossier.nonclosure_witness_replays(recovery["nonclosed"]["report"])


def test_dossier_payload_is_deterministic_and_claim_bounded() -> None:
    """Only exact, replayable outcomes appear in the compact public payload."""
    payload = operator_dossier.run_example()
    symbolic_domain = "literal-module:example.operator-dossier"
    table_domain = f"{symbolic_domain};arity:2"
    scope = (
        "finite exact infrastructure dossier only; "
        "no scientific or legacy correctness claim verified"
    )
    assert payload == {
        "basis_change": {
            "algorithm": "explicit-basis-composites-v1",
            "exact_inverse": True,
            "source_coordinates": [[0, 1, 1], [1, -2, 1]],
            "target_coordinates": [[0, -2, 1], [1, 1, 1]],
        },
        "matrix": {
            "columns": 2,
            "entries": [[[1, 1], [2, 1]]],
            "entry_parent": "QQ",
            "rows": 1,
        },
        "presentation": {
            "ambiguity_auto_selected": False,
            "candidate_route_count": 2,
            "container_shape_inference": "not attempted; source kind is explicit",
            "symbolic_coordinate_round_trip": True,
            "symbolic_coordinate_route": [
                ["example.operator-dossier.symbolic-to-coordinate", "forward"],
                ["example.operator-dossier.symbolic-to-coordinate", "inverse"],
            ],
            "symbolic_coordinate_validity_domain": symbolic_domain,
            "table_structure_round_trip": True,
            "table_structure_route": [
                ["example.operator-dossier.table-to-structure", "forward"],
                ["example.operator-dossier.table-to-structure", "inverse"],
            ],
            "table_structure_validity_domain": table_domain,
        },
        "recovery": {
            "closed": {
                "constants_returned": True,
                "reconstruction_exact": [True, True, True, True],
            },
            "dependent": {
                "constants_returned": False,
                "dependency_replays": True,
            },
            "nonclosed": {
                "constants_returned": False,
                "first_pair": [0, 0],
                "witness_replays": True,
            },
        },
        "scope": scope,
    }
    assert operator_dossier.run_example() == payload


def test_dossier_cli_is_cwd_independent_compact_json(tmp_path: Path) -> None:
    """The example is an executable artifact instead of an import-only fixture."""
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "examples" / "operator_dossier.py"),
    ]
    result = subprocess.run(
        command, cwd=tmp_path, check=False, capture_output=True, text=True
    )
    expected = operator_dossier.run_example()
    assert result.returncode == 0
    assert result.stderr == ""
    assert (
        result.stdout
        == json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n"
    )
    assert "0x" not in result.stdout and "<callable" not in result.stdout
