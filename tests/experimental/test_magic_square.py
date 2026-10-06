"""Tests for the experimental 2-by-2 and 3-by-3 magic-square catalogue."""

from __future__ import annotations

from math import comb

from anyalgebra.experimental.magic_square import (
    composition_algebra,
    composition_algebra_keys,
    jordan_dimension,
    magic_square,
    n2_magic_square_entry,
    n2_orthogonal_basis,
    n2_orthogonal_certificate,
    n3_magic_square_entry,
    n3_triality_decomposition,
    triality_dimension,
)


def test_composition_algebra_registry_records_division_and_split_signatures() -> None:
    assert composition_algebra_keys() == ("R", "C", "Cs", "H", "Hs", "O", "Os")
    assert composition_algebra("R").signature == (1, 0)
    assert composition_algebra("C").signature == (2, 0)
    assert composition_algebra("Cs").signature == (1, 1)
    assert composition_algebra("H").signature == (4, 0)
    assert composition_algebra("Hs").signature == (2, 2)
    assert composition_algebra("O").signature == (8, 0)
    assert composition_algebra("Os").signature == (4, 4)


def test_jordan_carrier_dimensions_distinguish_n2_from_n3() -> None:
    assert jordan_dimension(2, "C") == 4
    assert jordan_dimension(2, "O") == 10
    assert jordan_dimension(3, "O") == 27


def test_triality_dimensions_give_the_compact_e8_completion() -> None:
    assert tuple(triality_dimension(key) for key in ("R", "C", "H", "O")) == (
        0,
        2,
        9,
        28,
    )
    assert n3_triality_decomposition("O", "O") == (28, 28, 192)
    assert sum(n3_triality_decomposition("O", "O")) == 248


def test_triality_decomposition_matches_all_division_and_split_entries() -> None:
    for left in composition_algebra_keys():
        for right in composition_algebra_keys():
            assert sum(n3_triality_decomposition(left, right)) == (
                n3_magic_square_entry(left, right).lie_algebra_dimension
            )


def test_requested_compact_n2_ladder_reaches_spin16() -> None:
    r_c = n2_magic_square_entry("R", "C")
    c_o = n2_magic_square_entry("C", "O")
    o_o = n2_magic_square_entry("O", "O")

    assert (r_c.carrier_dimension, r_c.lie_algebra, r_c.group) == (
        4,
        "so(3)",
        "Spin(3)",
    )
    assert (c_o.carrier_dimension, c_o.lie_algebra_dimension) == (20, 45)
    assert (c_o.lie_algebra, c_o.group) == ("so(10)", "Spin(10)")
    assert (o_o.carrier_dimension, o_o.lie_algebra_dimension) == (80, 120)
    assert (o_o.lie_algebra, o_o.group) == ("so(16)", "Spin(16)")
    assert o_o.decomposition == (21, 36, 63)


def test_n2_allows_both_division_and_split_composition_algebras() -> None:
    assert n2_magic_square_entry("Cs", "O").signature == (9, 1)
    assert n2_magic_square_entry("Cs", "O").lie_algebra == "so(9,1)"
    assert n2_magic_square_entry("Os", "O").lie_algebra == "so(12,4)"
    assert n2_magic_square_entry("Cs", "Os").lie_algebra == "so(5,5)"

    table = magic_square(2)
    assert len(table) == 7
    assert all(len(row) == 7 for row in table)
    for row, left in zip(table, composition_algebra_keys(), strict=True):
        for entry, right in zip(row, composition_algebra_keys(), strict=True):
            reverse = n2_magic_square_entry(right, left)
            expected = comb(
                composition_algebra(left).dimension
                + composition_algebra(right).dimension,
                2,
            )
            assert entry.lie_algebra_dimension == expected
            assert entry.lie_algebra == reverse.lie_algebra


def test_n2_spin16_standard_operators_close_exactly() -> None:
    basis = n2_orthogonal_basis("O", "O")
    certificate = n2_orthogonal_certificate("O", "O")

    assert len(basis) == 120
    assert all(len(matrix) == 16 and len(matrix[0]) == 16 for matrix in basis)
    assert certificate.generator_count == 120
    assert certificate.metric_skew_generators == 120
    assert certificate.commutators_checked == 120 * 121 // 2
    assert certificate.closed


def test_compact_n3_square_contains_the_exceptional_sequence_and_e8() -> None:
    assert n3_magic_square_entry("R", "O").lie_algebra == "f4(-52)"
    assert n3_magic_square_entry("C", "O").lie_algebra == "e6(-78)"
    assert n3_magic_square_entry("H", "O").lie_algebra == "e7(-133)"

    e8 = n3_magic_square_entry("O", "O")
    assert e8.carrier_dimension == 216
    assert e8.lie_algebra_dimension == 248
    assert e8.lie_algebra == "e8(-248)"
    assert e8.group == "E8(-248)"
    assert e8.decomposition == (14, 52, 182)


def test_n3_half_split_and_fully_split_real_forms_are_catalogued() -> None:
    assert n3_magic_square_entry("Cs", "O").lie_algebra == "e6(-26)"
    assert n3_magic_square_entry("Hs", "O").lie_algebra == "e7(-25)"
    assert n3_magic_square_entry("Os", "O").lie_algebra == "e8(-24)"
    assert n3_magic_square_entry("Cs", "Cs").lie_algebra == ("sl(3,R)+sl(3,R)")
    assert n3_magic_square_entry("Hs", "Os").lie_algebra == "e7(7)"
    assert n3_magic_square_entry("Os", "Os").lie_algebra == "e8(8)"


def test_full_n3_catalogue_is_symmetric_and_dimensionally_exact() -> None:
    table = magic_square(3)
    assert len(table) == 7
    assert all(len(row) == 7 for row in table)

    for row, left in zip(table, composition_algebra_keys(), strict=True):
        for entry, right in zip(row, composition_algebra_keys(), strict=True):
            reverse = n3_magic_square_entry(right, left)
            assert entry.lie_algebra == reverse.lie_algebra
            assert entry.lie_algebra_dimension == reverse.lie_algebra_dimension
            assert sum(entry.decomposition) == entry.lie_algebra_dimension


def test_full_n3_catalogue_matches_the_three_published_real_form_tables() -> None:
    # Cacciatori--Cerchiai--Marrani, arXiv:1208.6153, Tables 1--3.
    keys = composition_algebra_keys()
    expected = (
        ("so(3)", "su(3)", "sl(3,R)", "sp(3)", "sp(6,R)", "f4(-52)", "f4(4)"),
        (
            "su(3)",
            "su(3)+su(3)",
            "sl(3,C)_R",
            "su(6)",
            "su(3,3)",
            "e6(-78)",
            "e6(2)",
        ),
        (
            "sl(3,R)",
            "sl(3,C)_R",
            "sl(3,R)+sl(3,R)",
            "su*(6)",
            "sl(6,R)",
            "e6(-26)",
            "e6(6)",
        ),
        (
            "sp(3)",
            "su(6)",
            "su*(6)",
            "so(12)",
            "so*(12)",
            "e7(-133)",
            "e7(-5)",
        ),
        (
            "sp(6,R)",
            "su(3,3)",
            "sl(6,R)",
            "so*(12)",
            "so(6,6)",
            "e7(-25)",
            "e7(7)",
        ),
        (
            "f4(-52)",
            "e6(-78)",
            "e6(-26)",
            "e7(-133)",
            "e7(-25)",
            "e8(-248)",
            "e8(-24)",
        ),
        (
            "f4(4)",
            "e6(2)",
            "e6(6)",
            "e7(-5)",
            "e7(7)",
            "e8(-24)",
            "e8(8)",
        ),
    )

    actual = tuple(
        tuple(n3_magic_square_entry(left, right).lie_algebra for right in keys)
        for left in keys
    )
    assert actual == expected

    for left in keys:
        for right in keys:
            entry = n3_magic_square_entry(left, right)
            assert "Cacciatori--Cerchiai--Marrani 2012" in entry.source
            assert entry.group_scope == "connected global form unspecified"


def test_magic_square_rejects_unsupported_matrix_size_and_algebra() -> None:
    try:
        magic_square(4)
    except ValueError as error:
        assert "2 or 3" in str(error)
    else:
        raise AssertionError("unsupported matrix size was accepted")

    try:
        composition_algebra("S")
    except ValueError as error:
        assert "unknown composition algebra" in str(error)
    else:
        raise AssertionError("unknown composition algebra was accepted")
