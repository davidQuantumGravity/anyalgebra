"""Bounded exhaustive properties for the canonical exact-domain boundary.

The bounds below are deliberately finite and deterministic.  They complement
the focused examples in :mod:`test_zz`, :mod:`test_qq`, and
:mod:`test_rational_kernel` without relying on a property-testing dependency.
"""

from __future__ import annotations

from math import gcd

import pytest

from anyalgebra.core.domains import IntegerDomain, QQ, RationalDomain, ZZ
from anyalgebra.core.errors import DomainConstructionError
from anyalgebra.core.rational import Rational


# The exhaustive finite fixture contains 25 integers and 600 nonzero-
# denominator rational literals.  The separate boundary fixture exercises
# arbitrary-precision normalization without making the nested loop unbounded.
INTEGER_BOX_BOUND = 12
INTEGER_BOX = range(-INTEGER_BOX_BOUND, INTEGER_BOX_BOUND + 1)
ZZ_BOX_CASE_COUNT = len(INTEGER_BOX)
LARGE_INTEGER_BOUNDARIES = (
    -(10**200),
    -(2**127),
    -1,
    0,
    1,
    2**127,
    10**200,
)
LARGE_RATIONAL_DENOMINATORS = (-(10**200), -(2**127), -1, 1, 2**127, 10**200)
RATIONAL_PAIR_CASE_COUNT = len(INTEGER_BOX) * (len(INTEGER_BOX) - 1)
LARGE_RATIONAL_PAIR_CASE_COUNT = len(LARGE_INTEGER_BOUNDARIES) * len(
    LARGE_RATIONAL_DENOMINATORS
)
QQ_INVALID_INPUT_REASON = (
    "expected a Rational, a built-in int, or a pair of built-in ints"
)
RAW_NUMBERS: tuple[object, ...] = (-12, -1, 0, 1, 12, -0.5, 0.5, 1.0)
RAW_NUMBER_CASE_COUNT = len(RAW_NUMBERS)
INVALID_INPUT_MATRIX: tuple[tuple[IntegerDomain | RationalDomain, object, str], ...] = (
    (ZZ(), True, "expected a built-in int, excluding bool"),
    (ZZ(), False, "expected a built-in int, excluding bool"),
    (ZZ(), 1.0, "expected a built-in int, excluding bool"),
    (ZZ(), "1", "expected a built-in int, excluding bool"),
    (ZZ(), (1, 1), "expected a built-in int, excluding bool"),
    (ZZ(), Rational(1, 1), "expected a built-in int, excluding bool"),
    (QQ(), True, QQ_INVALID_INPUT_REASON),
    (QQ(), False, QQ_INVALID_INPUT_REASON),
    (QQ(), 1.0, QQ_INVALID_INPUT_REASON),
    (QQ(), "1/2", QQ_INVALID_INPUT_REASON),
    (QQ(), [1, 2], QQ_INVALID_INPUT_REASON),
    (QQ(), (1,), QQ_INVALID_INPUT_REASON),
    (QQ(), (1, 2, 3), QQ_INVALID_INPUT_REASON),
    (QQ(), (True, 1), QQ_INVALID_INPUT_REASON),
    (QQ(), (1, False), QQ_INVALID_INPUT_REASON),
    (QQ(), (1.0, 2), QQ_INVALID_INPUT_REASON),
    (QQ(), (1, 2.0), QQ_INVALID_INPUT_REASON),
    (QQ(), ZZ().element(1), "cannot implicitly import an element from another parent"),
)
INVALID_INPUT_CASE_COUNT = len(INVALID_INPUT_MATRIX)


def test_zz_constructor_equivalence_is_exhaustive_in_the_declared_box() -> None:
    """Every declared native integer has exactly one ZZ value and hash class."""
    case_count = 0
    for value in INTEGER_BOX:
        case_count += 1
        element = ZZ().element(value)
        rebuilt = ZZ().element(element)

        assert element.parent is ZZ(), f"ZZ parent changed for input {value!r}"
        assert element.value == value, f"ZZ value changed for input {value!r}"
        assert rebuilt is element, f"ZZ did not rewrap input {value!r} idempotently"
        assert element == rebuilt, f"ZZ equality failed for input {value!r}"
        assert hash(element) == hash(rebuilt), f"ZZ hash failed for input {value!r}"
        assert len({element, rebuilt}) == 1, f"ZZ set class split for input {value!r}"

    assert case_count == ZZ_BOX_CASE_COUNT


def test_zz_large_integer_boundaries_remain_exact() -> None:
    """ZZ's direct exact boundary does not use bounded machine integers."""
    for value in LARGE_INTEGER_BOUNDARIES:
        element = ZZ().element(value)

        assert element.value == value, f"ZZ lost exactness at boundary {value!r}"
        assert ZZ().normalize(value) == value, (
            f"ZZ normalized boundary {value!r} incorrectly"
        )


def test_qq_constructor_equivalence_and_normal_form_are_exhaustive() -> None:
    """All 600 declared rational literals agree across pair and payload forms."""
    case_count = 0
    first_element_by_payload: dict[Rational, object] = {}
    all_pair_elements: list[object] = []
    for numerator in INTEGER_BOX:
        for denominator in INTEGER_BOX:
            if denominator == 0:
                continue

            case_count += 1
            payload = Rational(numerator, denominator)
            from_pair = QQ().element((numerator, denominator))
            from_payload = QQ().element(payload)
            all_pair_elements.append(from_pair)

            assert from_pair.value == payload, (
                "QQ pair construction disagreed with Rational payload for "
                f"({numerator!r}, {denominator!r})"
            )
            assert from_pair == from_payload, (
                "QQ constructor forms did not compare equal for "
                f"({numerator!r}, {denominator!r})"
            )
            assert hash(from_pair) == hash(from_payload), (
                "QQ equal constructor forms had different hashes for "
                f"({numerator!r}, {denominator!r})"
            )
            assert len({from_pair, from_payload}) == 1, (
                "QQ equal constructor forms occupied different set classes for "
                f"({numerator!r}, {denominator!r})"
            )
            assert payload.denominator > 0, (
                "Rational denominator was not positive for "
                f"({numerator!r}, {denominator!r})"
            )
            if payload.numerator == 0:
                assert (payload.numerator, payload.denominator) == (0, 1), (
                    "Rational zero did not have its sole normal form for "
                    f"({numerator!r}, {denominator!r})"
                )
            else:
                assert gcd(abs(payload.numerator), payload.denominator) == 1, (
                    "Rational payload was not gcd-reduced for "
                    f"({numerator!r}, {denominator!r})"
                )

            first_element = first_element_by_payload.get(payload)
            if first_element is None:
                first_element_by_payload[payload] = from_pair
            else:
                assert first_element == from_pair, (
                    "Equivalent QQ raw pairs did not compare equal for canonical "
                    f"payload {payload!r}"
                )
                assert from_pair == first_element, (
                    f"QQ equality was asymmetric for canonical payload {payload!r}"
                )
                assert hash(first_element) == hash(from_pair), (
                    "Equivalent QQ raw pairs had different hashes for canonical "
                    f"payload {payload!r}"
                )
                assert len({first_element, from_pair}) == 1, (
                    "Equivalent QQ raw pairs occupied different set classes for "
                    f"canonical payload {payload!r}"
                )

    assert case_count == RATIONAL_PAIR_CASE_COUNT
    expected_payloads = {
        Rational(numerator, denominator)
        for numerator in INTEGER_BOX
        for denominator in INTEGER_BOX
        if denominator != 0
    }
    assert set(first_element_by_payload) == expected_payloads
    assert len(first_element_by_payload) == len(expected_payloads)
    assert len(set(all_pair_elements)) == len(expected_payloads)


def test_qq_integer_constructor_forms_agree_for_every_declared_integer() -> None:
    """An integral rational has equivalent int, pair, and Rational forms."""
    case_count = 0
    for value in INTEGER_BOX:
        case_count += 1
        from_int = QQ().element(value)
        from_pair = QQ().element((value, 1))
        from_payload = QQ().element(Rational(value, 1))

        assert from_int == from_pair == from_payload, (
            f"QQ integral constructor forms disagree for {value!r}"
        )
        assert hash(from_int) == hash(from_pair) == hash(from_payload), (
            f"QQ integral constructor hash mismatch for {value!r}"
        )
        assert len({from_int, from_pair, from_payload}) == 1, (
            f"QQ integral constructor set class split for {value!r}"
        )

    assert case_count == ZZ_BOX_CASE_COUNT


def test_qq_large_boundaries_preserve_gcd_sign_and_zero_invariants() -> None:
    """Large inputs retain the same canonical rational normalization rules."""
    case_count = 0
    for numerator in LARGE_INTEGER_BOUNDARIES:
        for denominator in LARGE_RATIONAL_DENOMINATORS:
            case_count += 1
            payload = QQ().element((numerator, denominator)).value

            assert payload.denominator > 0, (
                "QQ large-boundary denominator was not positive for "
                f"({numerator!r}, {denominator!r})"
            )
            if numerator == 0:
                assert (payload.numerator, payload.denominator) == (0, 1), (
                    "QQ large-boundary zero was not normalized for "
                    f"({numerator!r}, {denominator!r})"
                )
            else:
                assert gcd(abs(payload.numerator), payload.denominator) == 1, (
                    "QQ large-boundary payload was not reduced for "
                    f"({numerator!r}, {denominator!r})"
                )

    assert case_count == LARGE_RATIONAL_PAIR_CASE_COUNT


def test_python_equality_does_not_cross_exact_parent_or_raw_number_boundaries() -> None:
    """Python equality never invokes an undeclared ZZ-to-QQ coercion route."""
    cross_parent_case_count = 0
    for value in INTEGER_BOX:
        cross_parent_case_count += 1
        zz_element = ZZ().element(value)
        qq_element = QQ().element(value)
        zz_as_object: object = zz_element
        qq_as_object: object = qq_element

        assert zz_as_object != qq_element, (
            f"ZZ and QQ compared equal without coercion for {value!r}"
        )
        assert qq_as_object != zz_element, (
            f"QQ and ZZ compared equal without coercion for {value!r}"
        )

    raw_number_case_count = 0
    for raw_number in RAW_NUMBERS:
        raw_number_case_count += 1
        raw_as_object: object = raw_number
        assert ZZ().element(1) != raw_number, (
            f"ZZ element compared equal to raw Python number {raw_number!r}"
        )
        assert raw_as_object != ZZ().element(1), (
            f"Raw Python number {raw_number!r} compared equal to a ZZ element"
        )
        assert QQ().element(1) != raw_number, (
            f"QQ element compared equal to raw Python number {raw_number!r}"
        )
        assert raw_as_object != QQ().element(1), (
            f"Raw Python number {raw_number!r} compared equal to a QQ element"
        )

    assert cross_parent_case_count == ZZ_BOX_CASE_COUNT
    assert raw_number_case_count == RAW_NUMBER_CASE_COUNT


@pytest.mark.parametrize(("domain", "raw", "expected_reason"), INVALID_INPUT_MATRIX)
def test_exact_domain_invalid_input_matrix_is_typed_and_explicit(
    domain: IntegerDomain | RationalDomain, raw: object, expected_reason: str
) -> None:
    """The declared invalid-input matrix cannot enter either exact parent."""
    with pytest.raises(DomainConstructionError) as caught:
        domain.element(raw)

    assert caught.value.domain is domain
    assert caught.value.reason == expected_reason


def test_declared_fixture_sizes_are_stable_and_derivable() -> None:
    """Reported finite-suite counts are derived from the fixture definitions."""
    assert ZZ_BOX_CASE_COUNT == 25
    assert RATIONAL_PAIR_CASE_COUNT == 600
    assert LARGE_RATIONAL_PAIR_CASE_COUNT == 42
    assert RAW_NUMBER_CASE_COUNT == 8
    assert INVALID_INPUT_CASE_COUNT == 18


def test_qq_zero_denominator_remains_a_distinct_typed_boundary() -> None:
    """A syntactically valid pair with denominator zero is not silently repaired."""
    with pytest.raises(ZeroDivisionError, match="denominator must not be zero"):
        QQ().element((1, 0))
