"""Executable receipt checks for the experimental compact-F4 construction."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from examples import f4_albert_certificate


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_receipt_has_exact_dimension_closure_and_compactness() -> None:
    report = f4_albert_certificate.run_example()

    assert report["coordinate_dimension"] == 27
    assert report["derivation_dimension_certificate"] == {
        "unknowns": 729,
        "nonzero_constraints": 9063,
        "prime": 1_000_003,
        "modular_rank_lower_bound": 677,
        "inner_derivation_nullity_lower_bound": 52,
        "certified_rational_rank": 677,
        "certified_nullity": 52,
    }
    assert report["lie_closure"] == {
        "generator_count": 52,
        "unordered_pairs_checked_including_diagonal": 1378,
        "all_commutators_in_exact_span": True,
    }
    assert report["compactness"] == {
        "trace_gram_ldl_positive_pivots": 52,
        "minimum_pivot": {"numerator": 3, "denominator": 2},
        "maximum_pivot": {"numerator": 6, "denominator": 1},
    }


def test_numerical_exponential_receipt_is_tightly_bounded_and_labelled() -> None:
    check = f4_albert_certificate.run_example()["one_parameter_numerical_check"]

    assert max(check["absolute_deltas"].values()) < 1e-11
    assert check["role"] == (
        "numerical integration check, not a global group-coordinate proof"
    )


def test_cli_is_deterministic_json() -> None:
    expected = f4_albert_certificate.run_example()
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "examples" / "f4_albert_certificate.py"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,  # a hang guard, not a performance budget
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == expected
    assert (
        completed.stdout
        == json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n"
    )
