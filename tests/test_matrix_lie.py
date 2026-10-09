"""Matrix Lie algebras over the composition algebras against the known table.

Sudbery, "Division algebras, (pseudo)orthogonal groups and spinors" (1984);
Barton and Sudbery, "Magic squares and matrix models of Lie algebras" (2003);
Baez, "The Octonions", section 3:

    su(2, K) = so(2), su(2), sp(2) = so(5), so(9)        for K = R, C, H, O
    sl(2, K) = so(2,1), so(3,1), so(5,1), so(9,1)
    su(3, K) = so(3), su(3), sp(3), f4
    sl(3, K) = sl(3,R), sl(3,C), su*(6), e6(-26)

The inertia of the Killing form is (noncompact, compact, 0) dimensions.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

import anyalgebra.composition as ca
import anyalgebra.easy as aa
import anyalgebra.matrices as am
import anyalgebra.matrix_lie as ml

Maker = Callable[[], aa.Algebra]
TABLE: tuple[tuple[str, int, Maker, int, tuple[int, int, int]], ...] = (
    ("su", 2, ca.reals, 1, (0, 0, 1)),
    ("su", 2, ca.complexes, 3, (0, 3, 0)),
    ("su", 2, ca.quaternions, 10, (0, 10, 0)),
    ("su", 2, ca.octonions, 36, (0, 36, 0)),
    ("sl", 2, ca.reals, 3, (2, 1, 0)),
    ("sl", 2, ca.complexes, 6, (3, 3, 0)),
    ("sl", 2, ca.quaternions, 15, (5, 10, 0)),
    ("sl", 2, ca.octonions, 45, (9, 36, 0)),
    ("su", 3, ca.reals, 3, (0, 3, 0)),
    ("su", 3, ca.complexes, 8, (0, 8, 0)),
    ("su", 3, ca.quaternions, 21, (0, 21, 0)),
    ("sl", 3, ca.reals, 8, (5, 3, 0)),
    ("sl", 3, ca.complexes, 16, (8, 8, 0)),
    ("sl", 3, ca.quaternions, 35, (14, 21, 0)),
)


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(
            r, id=f"{r[0]}{r[1]}-{r[3]}", marks=[pytest.mark.slow] * (r[3] > 30)
        )
        for r in TABLE
    ],
)
def test_generated_algebras_match_the_classical_identifications(
    row: tuple[str, int, Maker, int, tuple[int, int, int]],
) -> None:
    kind, size, make, dimension, signature = row
    build = ml.su if kind == "su" else ml.sl
    found = build(size, make())
    assert found.dimension == dimension
    assert found.killing_signature() == signature
    assert found.jacobi_holds()


def test_generator_bases_have_the_expected_sizes() -> None:
    for make, d in (
        (ca.reals, 1),
        (ca.complexes, 2),
        (ca.quaternions, 4),
        (ca.octonions, 8),
    ):
        source = make()
        for n in (2, 3):
            skew = ml.antihermitian_traceless(source, n)
            hermitian = ml.hermitian_traceless(source, n)
            assert len(skew) == d * n * (n - 1) // 2 + (n - 1) * (d - 1)
            assert len(hermitian) == d * n * (n - 1) // 2 + (n - 1)
            assert all(x.is_antihermitian() and not x.trace() for x in skew)
            assert all(x.is_hermitian() and not x.trace() for x in hermitian)


def test_actions_are_the_stated_linear_maps() -> None:
    quaternions = ca.quaternions()
    matrix = am.matrix(quaternions, [[quaternions.e1, 1], [-1, 0]])
    operator = ml.column_action(matrix)
    assert len(operator) == 8 and len(operator[0]) == 8
    vector = am.matrix(quaternions, [[quaternions.e2], [3]])
    image = am.coordinates(matrix @ vector)
    flat = am.coordinates(vector)
    assert (
        tuple(sum(a * b for a, b in zip(row, flat, strict=True)) for row in operator)
        == image
    )
    octonions = ca.octonions()
    skew = ml.antihermitian_traceless(octonions, 3)[0]
    derivation = ml.hermitian_action(skew)
    assert len(derivation) == 27
    assert sum(derivation[i][i] for i in range(27)) == 0


@pytest.mark.slow
@pytest.mark.parametrize(("kind", "dimension"), (("su", 52), ("sl", 78)))
def test_three_by_three_octonionic_matrices_generate_f4_and_e6(
    kind: str, dimension: int
) -> None:
    found = ml.dimension_certificate(3, ca.octonions(), kind=kind)
    assert found.independent == dimension
    assert found.name == f"{kind}(3,O)"


def test_requests_without_a_meaning_fail_with_typed_errors() -> None:
    octonions, sedenions = ca.octonions(), ca.sedenions()
    no_unit = aa.wrap(
        aa.algebra(("a",), {("a", "a"): "a"}, unit=None).structure,
        unit=None,
        conjugation_signs=(1,),
    )
    cases: tuple[tuple[Callable[[], object], str], ...] = (
        (lambda: ml.su(1, octonions), "shape"),
        (lambda: ml.su(4, octonions), "unsupported"),
        (lambda: ml.sl(3, sedenions), "unsupported"),
        (lambda: ml.dimension_certificate(3, octonions, kind="so"), "unsupported"),
        (lambda: ml.hermitian_action(am.unit(no_unit, 1, 0, 0, no_unit.a)), "unit"),
    )
    for attempt, code in cases:
        with pytest.raises(aa.EasyError) as caught:
            attempt()
        assert caught.value.code == code
    small = ml.dimension_certificate(2, ca.quaternions())
    assert (small.independent, small.name) == (10, "su(2,H)")
    assert ml.dimension_certificate(2, ca.complexes(), kind="sl").independent == 6
