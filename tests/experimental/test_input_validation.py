"""Refused inputs and small exact facts for the Albert and magic-square modules.

Valid constructions are tested where each object is built.  This file checks
that the public entry points reject what they must, and pins a few helper
results that no other test looked at.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from anyalgebra.experimental.exceptional import (
    albert_basis,
    apply_operator,
    f4_inner_derivation_basis,
    octonion_basis_product,
    octonion_multiply,
    octonion_real_part,
    one_parameter_action,
    operator_coordinates,
    positive_definite_ldl_pivots,
    trace_operator_gram,
)
from anyalgebra.experimental.magic_square import (
    jordan_dimension,
    render_magic_square_markdown,
    triality_dimension,
)


def test_octonion_and_albert_helpers_validate_their_vectors() -> None:
    one = (1, 0, 0, 0, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="exactly 8 coordinates"):
        octonion_multiply((1, 2), one)
    with pytest.raises(ValueError, match="finite"):
        octonion_multiply((float("inf"), 0, 0, 0, 0, 0, 0, 0), one)
    with pytest.raises(TypeError, match="Fractions, ints, or floats"):
        octonion_multiply(("1", 0, 0, 0, 0, 0, 0, 0), one)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"0\.\.7"):
        octonion_basis_product(8, 0)
    with pytest.raises(ValueError, match=r"0\.\.7"):
        octonion_basis_product(True, 0)
    assert octonion_real_part((Fraction(3), *([Fraction()] * 7))) == 3
    # Floats are accepted by the generic product and stay floats.
    assert octonion_multiply((0.5, 0, 0, 0, 0, 0, 0, 0), one)[0] == 0.5

    derivation = f4_inner_derivation_basis()[0]
    with pytest.raises(ValueError, match="exactly 27 rows"):
        apply_operator(derivation[:26], albert_basis()[0])
    floating = tuple(tuple(float(entry) for entry in row) for row in derivation)
    with pytest.raises(TypeError, match="Fraction entries"):
        operator_coordinates(floating, f4_inner_derivation_basis())
    with pytest.raises(TypeError, match="Fraction entries"):
        trace_operator_gram((floating,))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="built-in float"):
        one_parameter_action(derivation, albert_basis()[1], 1)
    with pytest.raises(ValueError, match="terms"):
        one_parameter_action(derivation, albert_basis()[1], 0.1, terms=0)


def test_operator_coordinates_in_a_basis_that_needs_elimination() -> None:
    first, second, third = f4_inner_derivation_basis()[:3]
    mixed = tuple(
        tuple(a + b for a, b in zip(row_a, row_b, strict=True))
        for row_a, row_b in zip(first, second, strict=True)
    )
    basis = (first, mixed)
    assert operator_coordinates(second, basis) == (Fraction(-1), Fraction(1))
    assert operator_coordinates(mixed, basis) == (Fraction(0), Fraction(1))
    with pytest.raises(ValueError, match="outside the declared span"):
        operator_coordinates(third, basis)
    with pytest.raises(ValueError, match="linearly dependent"):
        operator_coordinates(first, (first, mixed, second))


def test_exact_ldl_pivots_on_coupled_and_indefinite_matrices() -> None:
    two, one, zero = Fraction(2), Fraction(1), Fraction(0)
    assert positive_definite_ldl_pivots(((two, one), (one, two))) == (
        Fraction(2),
        Fraction(3, 2),
    )
    assert positive_definite_ldl_pivots(
        ((Fraction(4), two, zero), (two, Fraction(5), one), (zero, one, Fraction(3)))
    ) == (Fraction(4), Fraction(4), Fraction(11, 4))
    with pytest.raises(ValueError, match="not positive definite"):
        positive_definite_ldl_pivots(((one, two), (two, one)))
    with pytest.raises(ValueError, match="square"):
        positive_definite_ldl_pivots(((one, zero),))
    with pytest.raises(ValueError, match="symmetric"):
        positive_definite_ldl_pivots(((one, one), (zero, one)))
    with pytest.raises(TypeError, match="Fraction entries"):
        positive_definite_ldl_pivots(((1,),))  # type: ignore[arg-type]


def test_magic_square_helpers_validate_and_render() -> None:
    with pytest.raises(TypeError, match="catalogue key"):
        triality_dimension(3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="2 or 3"):
        jordan_dimension(4, "O")
    with pytest.raises(TypeError, match="catalogue key"):
        jordan_dimension(2, 3)  # type: ignore[arg-type]
    assert jordan_dimension(3, "O") == 27
    assert jordan_dimension(2, "O") == 10

    compact = render_magic_square_markdown(3, ("R", "C", "H", "O"))
    rows = compact.splitlines()
    assert rows[0] == "| A \\ B | R | C | H | O |"
    assert len(rows) == 6
    assert "e8(-248)" in rows[-1] or "e8" in rows[-1]
    assert len(render_magic_square_markdown(2).splitlines()) == 9
