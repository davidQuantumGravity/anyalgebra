"""Cartan subalgebras, roots, Cartan types and real forms from structure constants."""

from __future__ import annotations

from fractions import Fraction

import pytest

import anyalgebra.composition as ca
import anyalgebra.lie as al
import anyalgebra.lie_structure as ls
import anyalgebra.matrix_lie as ml
from anyalgebra.easy import EasyError
from anyalgebra.lie_structure import Gauss


def _algebra(
    dimension: int, brackets: dict[tuple[int, int], list[int]], name: str = "g"
) -> al.LieAlgebra:
    both = dict(brackets)
    for (i, j), value in brackets.items():
        both[(j, i)] = [-entry for entry in value]
    return al.LieAlgebra(both, dimension, name=name)


def test_gaussian_rationals() -> None:
    i = Gauss(0, 1)
    assert i * i == -1 and (1 + i) * (1 - i) == 2 and (1 + i) / i == 1 - i
    assert Gauss(1, 2).conjugate() == Gauss(1, -2) and -i == Gauss(0, -1)
    assert i - 1 == Gauss(-1, 1) and 2 * i == i + i and not Gauss() and bool(i)
    assert [
        str(z) for z in (Gauss(2), i, -i, Gauss(0, 3), Gauss(1, -2), Gauss(1, 1))
    ] == [
        "2",
        "i",
        "-i",
        "3*i",
        "1 - 2*i",
        "1 + i",
    ]
    assert repr(Gauss(Fraction(1, 2))) == "Gauss(1/2)" and hash(i) == hash(Gauss(0, 1))
    assert (i == "i") is False
    for wrong in (
        lambda: i + "x",
        lambda: i - "x",
        lambda: "x" - i,
        lambda: i * "x",
        lambda: i / 0,
    ):
        with pytest.raises(TypeError):
            wrong()  # type: ignore[no-untyped-call]


CARTAN_TYPES = [
    ("A", 1),
    ("A", 4),
    ("B", 2),
    ("B", 3),
    ("B", 5),
    ("C", 3),
    ("C", 5),
    ("D", 4),
    ("D", 6),
    ("E", 6),
    ("E", 7),
    ("E", 8),
    ("F", 4),
    ("G", 2),
]


@pytest.mark.parametrize(("family", "size"), CARTAN_TYPES)
def test_every_cartan_matrix_is_recognized(family: str, size: int) -> None:
    matrix = al.cartan_matrix(family, size)
    assert ls.classify(matrix) == ((family, size),)
    # The answer does not depend on the numbering of the simple roots.
    order = list(reversed(range(size)))
    shuffled = [[matrix[i][j] for j in order] for i in order]
    assert ls.classify(shuffled) == ((family, size),)


def test_a_reducible_cartan_matrix_splits_into_components() -> None:
    matrix = [[2, 0, -1], [0, 2, 0], [-1, 0, 2]]
    assert ls.classify(matrix) == (("A", 2), ("A", 1))
    assert ls.classify([]) == ()


IDENTIFIED = [
    (lambda: al.sl(2), "A1, real form sl(2,R)"),
    (lambda: al.so(3), "A1, real form su(2)"),
    (lambda: al.su(1, 1), "A1, real form sl(2,R)"),
    (lambda: al.sl(3), "A2, real form sl(3,R)"),
    (lambda: al.su(3), "A2, real form su(3)"),
    (lambda: al.su(2, 1), "A2, real form su(2,1)"),
    (lambda: al.so(4), "A1+A1, real form su(2) + su(2)"),
    (lambda: al.so(3, 1), "A1+A1, real form sl(2,C)"),
    (lambda: al.so(2, 2), "A1+A1, real form sl(2,R) + sl(2,R)"),
    (lambda: al.so(5), "B2, real form so(5)"),
    (lambda: al.so(4, 1), "B2, real form so(4,1)"),
    (lambda: al.sp(2), "B2, real form so(3,2)"),
    (lambda: al.sp(3), "C3, real form sp(6,R)"),
    (lambda: al.so(7), "B3, real form so(7)"),
    (lambda: al.so(8), "D4, real form so(8)"),
    (lambda: al.so(4, 4), "D4, real form so(4,4)"),
    (lambda: al.so(6, 2), "D4, real form so(6,2) or so*(8)"),
    (lambda: al.su(2, 2), "A3, real form su(2,2)"),
    (lambda: ml.su(2, ca.octonions()), "B4, real form so(9)"),
    (lambda: ml.sl(2, ca.octonions()), "D5, real form so(9,1)"),
    (lambda: ml.su(3, ca.quaternions()), "C3, real form sp(3)"),
    (lambda: ml.sl(3, ca.quaternions()), "A5, real form su*(6)"),
    (lambda: ml.multiplication_algebra(ca.octonions()), "D4, real form so(8)"),
]


@pytest.mark.parametrize(
    ("build", "expected"), IDENTIFIED, ids=[text for _, text in IDENTIFIED]
)
def test_identify_names_the_algebra_from_its_structure_constants(
    build: object, expected: str
) -> None:
    assert callable(build)
    assert ls.identify(build()) == expected


@pytest.mark.slow
def test_identify_the_octonionic_exceptional_algebras() -> None:
    octonions = ca.octonions()
    assert ls.identify(ml.su(3, octonions)) == "F4, real form f4"
    assert ls.identify(ml.sl(3, octonions)) == "E6, real form e6(-26)"


def test_roots_and_chevalley_generators_of_sl3_and_so5() -> None:
    for algebra, matrix, split in (
        (al.sl(3), ((2, -1), (-1, 2)), True),
        (al.so(5), None, False),
        (al.sp(3), None, True),
    ):
        found = ls.root_decomposition(algebra)
        cartan = found.cartan_matrix()
        if matrix is not None:
            assert cartan == matrix
        assert found.is_split is split and found.rank == len(found.simple)
        assert len(found.roots) == 2 * len(found.positive)
        assert len(found.roots) + found.rank == algebra.dimension
        generators = found.chevalley_generators()
        for i, (e_i, f_i, h_i) in enumerate(generators):
            assert ls.bracket(algebra, e_i, f_i) == h_i
            for j, (e_j, f_j, h_j) in enumerate(generators):
                assert ls.bracket(algebra, h_i, e_j) == tuple(
                    cartan[j][i] * value for value in e_j
                )
                assert ls.bracket(algebra, h_i, f_j) == tuple(
                    -cartan[j][i] * value for value in f_j
                )
                assert not any(ls.bracket(algebra, h_i, h_j))
                if i != j:
                    assert not any(ls.bracket(algebra, e_i, f_j))
        if split:
            assert all(not value.im for e, _, _ in generators for value in e)
    so5 = ls.root_decomposition(al.so(5))
    assert so5.type == "B2" and so5.kinds == ("imaginary", "imaginary")
    assert so5.cartan_integer(so5.simple[0], so5.simple[0]) == 2
    assert so5.cartan_integer(so5.simple[0], tuple(-v for v in so5.simple[0])) == -2
    assert repr(so5) == "RootDecomposition(so(5), type=B2, roots=8)"


def test_reductive_and_abelian_algebras_have_a_torus_part() -> None:
    gl3 = ls.root_decomposition(al.gl(3))
    assert gl3.type == "A2+T1" and len(gl3.roots) == 6
    abelian = _algebra(2, {}, "abelian")
    found = ls.root_decomposition(abelian)
    assert found.type == "T2" and found.roots == () and found.is_split
    assert ls.root_decomposition(_algebra(0, {})).type == "0"


def test_cartan_subalgebra_rank_and_radical() -> None:
    assert ls.rank(al.sl(3)) == 2 and ls.rank(al.so(5)) == 2 and ls.rank(al.gl(3)) == 3
    assert ls.rank(al.so(4, 1)) == 2 and ls.rank(al.su(3)) == 2
    for algebra in (al.sl(3), al.so(4)):
        cartan = ls.cartan_subalgebra(algebra)
        assert all(not any(algebra.bracket(x, y)) for x in cartan for y in cartan)
    # The Heisenberg algebra is nilpotent: it is its own Cartan subalgebra.
    heisenberg = _algebra(3, {(0, 1): [0, 0, 1]}, "heis")
    assert ls.rank(heisenberg) == 3 and len(ls.radical(heisenberg)) == 3
    # The two-dimensional nonabelian algebra [h, e] = e is solvable of rank 1.
    affine = _algebra(2, {(0, 1): [0, 1]}, "aff")
    assert (
        ls.cartan_subalgebra(affine) == ((Fraction(1), Fraction(0)),)
        and len(ls.radical(affine)) == 2
    )
    assert ls.radical(al.sl(2)) == () and len(ls.radical(al.gl(3))) == 1


def test_errors_are_named() -> None:
    affine = _algebra(2, {(0, 1): [0, 1]}, "aff")
    with pytest.raises(EasyError, match="not semisimple") as caught:
        ls.identify(affine)
    assert caught.value.code == "semisimple"
    # [h, e1] = e1 and [h, e2] = e2: the root space of 1 has dimension two.
    wide = _algebra(3, {(0, 1): [0, 1, 0], (0, 2): [0, 0, 1]}, "wide")
    with pytest.raises(EasyError, match="dimension above one") as caught:
        ls.root_decomposition(wide)
    assert caught.value.code == "reductive"
    # so(3) in a basis whose vectors, sums and differences all have irrational
    # length: the eigenvalues are i times a square root, outside Q(i).
    b = [[1, 2, 0], [0, 1, 3], [5, 0, 1]]
    so3 = al.so(3)
    matrices = so3.matrices
    assert matrices is not None
    skewed = al.from_matrices(
        [
            [
                [
                    sum(c * m[r][s] for c, m in zip(row, matrices, strict=True))
                    for s in range(3)
                ]
                for r in range(3)
            ]
            for row in b
        ],
        name="skewed",
    )
    with pytest.raises(EasyError, match=r"splits over Q\(i\)") as caught:
        ls.root_decomposition(skewed)
    assert caught.value.code == "split"
    assert ls.rank(skewed) == 1 and skewed.killing_signature() == (0, 3, 0)
    # The Heisenberg algebra has nilpotent, not diagonalizable, adjoint maps.
    with pytest.raises(EasyError, match=r"splits over Q\(i\)"):
        ls.root_decomposition(_algebra(3, {(0, 1): [0, 0, 1]}, "heis"))


def test_real_form_tables() -> None:
    assert ls.real_forms("E", 6, -26) == ("e6(-26)",) and ls.real_forms(
        "E", 6, -78
    ) == ("e6",)
    assert ls.real_forms("E", 7, 7) == ("e7(7)",) and ls.real_forms("G", 2, -14) == (
        "g2",
    )
    assert ls.real_forms("F", 4, -20) == ("f4(-20)",) and ls.real_forms(
        "E", 7, -25
    ) == ("e7(-25)",)
    assert ls.real_forms("A", 3, -5) == ("su*(4)",) and ls.real_forms("A", 3, 3) == (
        "sl(4,R)",
    )
    assert ls.real_forms("C", 2, -2) == ("sp(1,1)",) and ls.real_forms("C", 2, 2) == (
        "sp(4,R)",
    )
    assert ls.real_forms("D", 9, -9) == ("so(12,6)", "so*(18)")
    assert ls.real_forms("B", 4, 100) == ()
    with pytest.raises(EasyError):
        ls.real_forms("Z", 3, 0)
