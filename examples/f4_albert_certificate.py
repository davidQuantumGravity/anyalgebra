"""Emit a deterministic structural receipt for the experimental F4 model."""

from __future__ import annotations

import json
from fractions import Fraction
from typing import Any

from anyalgebra.experimental.exceptional import (
    albert_derivation_dimension_certificate,
    albert_determinant,
    albert_trace,
    albert_trace_form,
    f4_inner_derivation_basis,
    matrix_commutator,
    one_parameter_action,
    operator_coordinates,
    positive_definite_ldl_pivots,
    trace_operator_gram,
)


def _rational(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def run_example() -> dict[str, Any]:
    generators = f4_inner_derivation_basis()
    certificate = albert_derivation_dimension_certificate()
    closure_pairs = 0
    for left_index, left in enumerate(generators):
        for right in generators[left_index:]:
            operator_coordinates(matrix_commutator(left, right), generators)
            closure_pairs += 1

    pivots = positive_definite_ldl_pivots(trace_operator_gram(generators))
    element = tuple(Fraction((index % 5) - 2, index + 1) for index in range(27))
    transformed = one_parameter_action(generators[0], element, 0.37, terms=48)
    before = {
        "trace": float(albert_trace(element)),
        "trace_form": float(albert_trace_form(element, element)),
        "determinant": float(albert_determinant(element)),
    }
    after = {
        "trace": albert_trace(transformed),
        "trace_form": albert_trace_form(transformed, transformed),
        "determinant": albert_determinant(transformed),
    }

    return {
        "construction": "Der(H_3(O)) in the algmul.O.v1 convention",
        "coordinate_dimension": 27,
        "derivation_dimension_certificate": {
            "unknowns": certificate.unknowns,
            "nonzero_constraints": certificate.nonzero_constraints,
            "prime": certificate.prime,
            "modular_rank_lower_bound": certificate.modular_rank,
            "inner_derivation_nullity_lower_bound": (
                certificate.inner_derivation_dimension
            ),
            "certified_rational_rank": certificate.rational_rank,
            "certified_nullity": certificate.nullity,
        },
        "lie_closure": {
            "generator_count": len(generators),
            "unordered_pairs_checked_including_diagonal": closure_pairs,
            "all_commutators_in_exact_span": True,
        },
        "compactness": {
            "trace_gram_ldl_positive_pivots": len(pivots),
            "minimum_pivot": _rational(min(pivots)),
            "maximum_pivot": _rational(max(pivots)),
        },
        "one_parameter_numerical_check": {
            "parameter": 0.37,
            "terms": 48,
            "before": before,
            "after": after,
            "absolute_deltas": {
                name: abs(after[name] - before[name]) for name in before
            },
            "role": "numerical integration check, not a global group-coordinate proof",
        },
        "claim_scope": (
            "exact finite structural reproduction of the compact F4 derivation "
            "algebra of the real division Albert algebra, plus one bounded "
            "floating-point exponential-action check"
        ),
    }


def main() -> None:
    print(json.dumps(run_example(), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
