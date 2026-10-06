"""Exact structural tests for the experimental Albert-algebra/F4 toolkit."""

from __future__ import annotations

from fractions import Fraction
from math import isclose

from anyalgebra.experimental.exceptional import (
    ALBERT_DIMENSION,
    F4_DIMENSION,
    AlbertElement,
    Operator,
    albert_basis,
    albert_determinant,
    albert_derivation_dimension_certificate,
    albert_identity,
    albert_jordan_product,
    albert_trace,
    albert_trace_form,
    apply_operator,
    f4_inner_derivation_basis,
    is_albert_derivation,
    is_trace_form_skew,
    jordan_associator,
    matrix_commutator,
    nested_difference_action,
    nested_left_action,
    nested_right_action,
    octonion_basis,
    octonion_multiply,
    one_parameter_action,
    operator_coordinates,
    trace_operator_gram,
    positive_definite_ldl_pivots,
)


def test_albert_basis_and_identity_are_exact() -> None:
    basis = albert_basis()
    identity = albert_identity()

    assert len(basis) == ALBERT_DIMENSION == 27
    assert albert_trace(identity) == Fraction(3)
    for vector in basis:
        assert albert_jordan_product(identity, vector) == vector


def test_jordan_product_is_commutative_and_has_the_jordan_identity_on_basis() -> None:
    basis = albert_basis()

    for left in basis:
        for right in basis:
            assert albert_jordan_product(left, right) == albert_jordan_product(
                right, left
            )

    # This is a bounded exact basis-vector check, not a polarization proof of
    # the quartic Jordan identity on arbitrary linear combinations.
    for x in basis:
        x2 = albert_jordan_product(x, x)
        for y in basis:
            assert albert_jordan_product(
                albert_jordan_product(x2, y), x
            ) == albert_jordan_product(x2, albert_jordan_product(y, x))


def test_jordan_associator_is_the_inner_derivation_action() -> None:
    basis = albert_basis()
    h1, element, h2 = basis[1], basis[10], basis[19]
    derivations = f4_inner_derivation_basis()

    associator = jordan_associator(h1, element, h2)
    expected = apply_operator(
        matrix_commutator(
            # The public basis contains [L_x,L_y] operators; recover the two
            # required multiplication operators through the defining action.
            _left_operator(h2, basis),
            _left_operator(h1, basis),
        ),
        element,
    )

    assert associator == expected
    assert len(derivations) == F4_DIMENSION == 52


def _left_operator(
    element: AlbertElement, basis: tuple[AlbertElement, ...]
) -> Operator:
    """Construct a test oracle directly from the public Jordan product."""
    columns = tuple(albert_jordan_product(element, vector) for vector in basis)
    return tuple(tuple(column[row] for column in columns) for row in range(27))


def test_all_52_generators_are_exact_derivations_and_compact() -> None:
    derivations = f4_inner_derivation_basis()

    assert len(derivations) == 52
    assert all(is_albert_derivation(operator) for operator in derivations)
    assert all(is_trace_form_skew(operator) for operator in derivations)
    assert all(
        apply_operator(operator, albert_identity()) == (Fraction(0),) * 27
        for operator in derivations
    )


def test_full_derivation_space_has_dimension_52_by_two_sided_rank_bounds() -> None:
    certificate = albert_derivation_dimension_certificate()

    assert certificate.unknowns == 27 * 27
    assert certificate.modular_rank == 677
    assert certificate.rational_rank == 677
    assert certificate.nullity == 52
    assert certificate.inner_derivation_dimension == 52
    assert certificate.prime == 1_000_003


def test_compact_trace_gram_is_exactly_positive_definite() -> None:
    gram = trace_operator_gram(f4_inner_derivation_basis())
    pivots = positive_definite_ldl_pivots(gram)

    assert len(pivots) == 52
    assert all(pivot > 0 for pivot in pivots)


def test_f4_generator_commutators_close_in_the_52_dimensional_span() -> None:
    basis = f4_inner_derivation_basis()
    checked = 0

    for left_index, left in enumerate(basis):
        for right in basis[left_index:]:
            coordinates = operator_coordinates(matrix_commutator(left, right), basis)
            assert len(coordinates) == 52
            checked += 1

    assert checked == 52 * 53 // 2


def test_exponentiated_generator_preserves_the_three_basic_invariants() -> None:
    generator = f4_inner_derivation_basis()[0]
    element = tuple(Fraction((index % 5) - 2, index + 1) for index in range(27))
    transformed = one_parameter_action(generator, element, 0.37, terms=48)

    assert isclose(
        float(albert_trace(element)), albert_trace(transformed), abs_tol=1e-12
    )
    assert isclose(
        float(albert_trace_form(element, element)),
        albert_trace_form(transformed, transformed),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    assert isclose(
        float(albert_determinant(element)),
        albert_determinant(transformed),
        rel_tol=1e-11,
        abs_tol=1e-11,
    )


def test_algmul_nested_action_order_is_explicit_on_octonions() -> None:
    one, e1, e2, e3, *_ = octonion_basis()
    del one

    left = nested_left_action((e1, e2), e3, octonion_multiply)
    right = nested_right_action((e1, e2), e3, octonion_multiply)

    assert left == octonion_multiply(e1, octonion_multiply(e2, e3))
    assert right == octonion_multiply(octonion_multiply(e3, e2), e1)
    assert left != right
    assert nested_difference_action((e1, e2), e3, octonion_multiply) == tuple(
        -left[index] - right[index] for index in range(8)
    )
