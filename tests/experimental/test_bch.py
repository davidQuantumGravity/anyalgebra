"""Exact symbolic tests for the BCH implementation."""

from __future__ import annotations

from fractions import Fraction

import pytest

from anyalgebra.experimental.bch import (
    bch_associative_word_coefficients,
    bch_exact_certificate,
    bch_hall_terms,
)


def test_quartic_free_lie_basis_has_three_terms_but_one_bch_coefficient() -> None:
    quartic = tuple(
        term for term in bch_hall_terms(4, include_zero=True) if term.degree == 4
    )

    assert tuple(term.expression for term in quartic) == (
        "[X,[X,[X,Y]]]",
        "[Y,[X,[X,Y]]]",
        "[Y,[Y,[Y,X]]]",
    )
    assert tuple(term.coefficient for term in quartic) == (
        Fraction(0),
        Fraction(-1, 24),
        Fraction(0),
    )


def test_exact_associative_log_reconstructs_the_hall_polynomial() -> None:
    certificate = bch_exact_certificate(4)

    assert certificate.free_lie_dimensions == (2, 1, 2, 3)
    assert certificate.homogeneous_nonzero_hall_terms == (2, 1, 2, 1)
    assert certificate.associative_words_checked == 30
    assert certificate.hall_basis_terms_checked == 8
    assert certificate.reconstruction_exact
    assert dict(bch_associative_word_coefficients(4))["XXYY"] == Fraction(1, 24)


@pytest.mark.parametrize("degree", [0, 5, True])
def test_bch_symbolic_tools_reject_unsupported_degrees(degree: object) -> None:
    with pytest.raises(ValueError, match="degree"):
        bch_hall_terms(degree)  # type: ignore[arg-type]
