"""The second round of the family modules: what the first versions left open."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from itertools import combinations

import pytest

import anyalgebra.clifford as ac
import anyalgebra.composition as ca
import anyalgebra.jordan as aj
import anyalgebra.lie as al
import anyalgebra.matrices as am
import anyalgebra.matrix_lie as ml
from anyalgebra.easy import Algebra, EasyError, algebra

# Lounesto, "Clifford Algebras and Spinors", table of Cl(p, q): the number of
# commuting involutions, the real dimension of a minimal left ideal, and the
# division ring.  Generators 1..p square to +1.
IDEALS = (
    ((1, 0), 1, 1, "R"),
    ((0, 1), 0, 2, "C"),
    ((2, 0), 1, 2, "R"),
    ((1, 1), 1, 2, "R"),
    ((0, 2), 0, 4, "H"),
    ((3, 0), 1, 4, "C"),
    ((0, 3), 1, 4, "H"),
    ((3, 1), 2, 4, "R"),
    ((1, 3), 1, 8, "H"),
    ((2, 2), 2, 4, "R"),
    ((4, 0), 1, 8, "H"),
    ((0, 4), 1, 8, "H"),
    ((5, 0), 2, 8, "H"),
)


@pytest.mark.parametrize(("signature", "count", "dimension", "ring"), IDEALS)
def test_primitive_idempotents_match_the_classification(
    signature: tuple[int, int], count: int, dimension: int, ring: str
) -> None:
    source = ac.clifford(*signature)
    blades = ac.commuting_involutions(source)
    assert len(blades) == count
    assert all(b * b == 1 for b in blades)
    assert all(a * b == b * a for a, b in combinations(blades, 2))
    idempotents = ac.primitive_idempotents(source)
    assert len(idempotents) == 2**count
    assert all(ac.is_idempotent(f) for f in idempotents)
    assert sum(idempotents[1:], idempotents[0]) == 1
    assert all(not a * b for a, b in combinations(idempotents, 2))
    f = idempotents[0]
    assert len(ac.left_ideal(f)) == dimension and ac.division_ring(f) == ring


@pytest.mark.parametrize("signature", [(3, 1), (1, 3), (2, 2), (3, 0), (0, 2)])
def test_the_spinor_representation_satisfies_the_clifford_relations(
    signature: tuple[int, int],
) -> None:
    p, q = signature
    matrices = ac.spinor_representation(ac.clifford(p, q))
    size = len(matrices[0])
    one = tuple(tuple(Fraction(int(i == j)) for j in range(size)) for i in range(size))

    times = al._multiply

    for a, first in enumerate(matrices):
        for b, second in enumerate(matrices):
            total = tuple(
                tuple(x + y for x, y in zip(r, s, strict=True))
                for r, s in zip(times(first, second), times(second, first), strict=True)
            )
            eta = 0 if a != b else (2 if a < p else -2)
            assert total == tuple(tuple(eta * v for v in row) for row in one)


def test_majorana_matrices_exist_exactly_where_the_ring_is_real() -> None:
    assert len(ac.majorana_matrices(3, 1)[0]) == 4
    assert len(ac.majorana_matrices(1, 1)[0]) == 2
    with pytest.raises(EasyError, match="spinors over H"):
        ac.majorana_matrices(1, 3)
    with pytest.raises(EasyError, match="spinors over C"):
        ac.majorana_matrices(3, 0)


def test_idempotent_rejections() -> None:
    source = ac.clifford(3, 1)
    wrong: tuple[tuple[str, Callable[[], object]], ...] = (
        ("shape", lambda: ac.idempotent([])),
        ("shape", lambda: ac.idempotent([source.e1], [2])),
        ("idempotent", lambda: ac.division_ring(source.e1)),
        ("idempotent", lambda: ac.division_ring(source.zero)),
        ("idempotent", lambda: ac.division_ring(source.e1 * 0 + 1)),
    )
    for code, call in wrong:
        with pytest.raises(EasyError) as caught:
            call()
        assert caught.value.code == code
    assert ac.primitive_idempotents(ac.clifford(0, 1)) == (ac.clifford(0, 1).unit,) or (
        len(ac.primitive_idempotents(ac.clifford(0, 1))) == 1
    )
    assert not ac.is_idempotent(source.e1)
    chosen = ac.idempotent([source.e1], [-1])
    assert chosen == (1 - source.e1) / 2 and ac.is_idempotent(chosen)
    assert (
        len(ac.spinor_representation(source, ac.primitive_idempotents(source)[1])) == 4
    )


def test_the_dirac_basis_swaps_the_diagonal_matrix() -> None:
    gammas = ac.gamma_matrices(1, 3)
    weyl = [ac.as_matrix(g) for g in gammas]
    chiral = ac.as_matrix(ac.chirality(gammas))
    dirac, s = ac.dirac_basis(gammas)
    one = am.identity(ac.complex_numbers(), 4)
    assert s @ s == one * 2
    # In the new basis gamma 0 is the old chirality matrix, which is diagonal.
    assert dirac[0] == chiral and dirac[0] != weyl[0]
    for a, first in enumerate(dirac):
        for b, second in enumerate(dirac):
            eta = 0 if a != b else (2 if a < 1 else -2)
            assert first @ second + second @ first == one * eta
    assert ac.change_basis(dirac, s, s / 2) == tuple(weyl)
    spacelike, s2 = ac.dirac_basis(gammas, 1)
    assert spacelike[1] @ spacelike[1] == -one and s2 != s
    assert not ac.is_real(weyl) and not ac.is_imaginary(weyl)
    assert ac.is_real([weyl[0], weyl[1]]) and ac.is_imaginary([weyl[2] @ weyl[0]])
    with pytest.raises(EasyError, match="even number"):
        ac.dirac_basis(ac.gamma_matrices(3, 0))
    with pytest.raises(EasyError, match="not the inverse"):
        ac.change_basis(weyl, s, s)


def test_vectors_and_slashed_matrices_convert_both_ways() -> None:
    gammas = ac.gamma_matrices(1, 3)
    vector = (Fraction(1), Fraction(-2), Fraction(3, 2), Fraction(0))
    slashed = ac.slash(gammas, vector)
    assert ac.unslash(gammas, slashed) == vector
    # p-slash squared is the Minkowski square of p.
    assert slashed @ slashed == am.identity(ac.complex_numbers(), 4) * (
        1 - 4 - Fraction(9, 4)
    )
    with pytest.raises(EasyError, match="one component"):
        ac.slash(gammas, [1, 2])
    with pytest.raises(EasyError, match="not a combination"):
        ac.unslash(gammas, am.identity(ac.complex_numbers(), 4))


def test_three_tensor_factors_and_their_conjugations() -> None:
    c, h, o = ca.complexes(), ca.complexes(), ca.quaternions()
    triple = ca.tensor(ca.tensor(c, h), o)
    assert triple.rank == 16 and triple.labels[-1] == "e1.f1.g3"
    assert ca.factors_conjugation([c, h], [1]) == ca.factor_conjugation(
        c, h, second=True
    )
    signs = ca.factors_conjugation([c, h, o], [0, 2])
    assert len(signs) == 16 and ca.is_involution(triple, signs)
    everything = ca.factors_conjugation([c, h, o], [0, 1, 2])
    assert everything == triple.conjugation_signs and ca.is_involution(triple)
    x = triple["e1.f1.g3"]
    assert ca.conjugate(x, signs) == x and ca.conjugate(x, everything) == -x
    matrix = am.matrix(triple, [[x, 0], [0, x]])
    assert matrix.dagger(signs) == matrix and matrix.is_antihermitian(everything)
    with pytest.raises(EasyError, match="existing factors"):
        ca.factors_conjugation([c, h], [2])
    with pytest.raises(EasyError, match="no conjugation"):
        ca.factors_conjugation([ac.grassmann(1), _plain()], [1])


def _plain() -> Algebra:
    return algebra(("1", "u"), {("1", "1"): "1", ("1", "u"): "u", ("u", "1"): "u"})


def test_multiplication_algebras_and_so_over_an_algebra() -> None:
    h, o = ca.quaternions(), ca.octonions()
    assert ml.multiplication_algebra(h).killing_signature() == (0, 3, 0)
    both = ml.multiplication_algebra(h, right=True)
    assert both.name == "mult_LR(H)" and both.killing_signature() == (0, 6, 0)
    assert ml.multiplication_algebra(h, left=False, right=True).dimension == 3
    # Left multiplication by imaginary octonions generates so(8).
    so8 = ml.multiplication_algebra(o)
    assert so8.dimension == 28 and so8.killing_signature() == (0, 28, 0)
    assert ml.so(3, ca.reals()).killing_signature() == (0, 3, 0)
    assert ml.so(3, ca.complexes()).killing_signature() == (3, 3, 0)
    assert ml.so(4, ca.complexes()).dimension == 12
    wrong: tuple[tuple[str, Callable[[], object]], ...] = (
        ("unsupported", lambda: ml.so(3, h)),
        ("unsupported", lambda: ml.so(3, o)),
        ("shape", lambda: ml.so(1, ca.reals())),
        ("shape", lambda: ml.multiplication_algebra(h, left=False)),
        ("shape", lambda: ml.multiplication_algebra(ca.reals())),
    )
    for code, call in wrong:
        with pytest.raises(EasyError) as caught:
            call()
        assert caught.value.code == code


@pytest.mark.slow
def test_three_by_three_octonionic_algebras_have_exact_signatures() -> None:
    octonions = ca.octonions()
    f4 = ml.su(3, octonions)
    assert f4.dimension == 52 and f4.killing_signature() == (0, 52, 0)
    e6 = ml.sl(3, octonions)
    # The real form E6(-26): 26 noncompact and 52 compact directions.
    assert e6.dimension == 78 and e6.killing_signature() == (26, 52, 0)


def test_jordan_reports() -> None:
    r, c, h = ca.reals(), ca.complexes(), ca.quaternions()
    assert aj.proof_table([r, c, h], (2, 3)).splitlines() == [
        "   J2      J3",
        "R  Jordan  Jordan",
        "C  Jordan  Jordan",
        "H  Jordan  Jordan",
    ]
    assert aj.report(aj.hermitian(h, 2)).splitlines() == [
        "J2(H):",
        "  dimension        6",
        "  commutative      True",
        "  Jordan identity  True",
        "  Jordan algebra   True",
    ]
    assert "Jordan algebra   False" in aj.report(h)


@pytest.mark.slow
def test_the_octonionic_proof_table_stops_at_three() -> None:
    row = aj.proof_table([ca.octonions()], (2, 3, 4)).splitlines()[1]
    assert row == "O  Jordan      Jordan      not Jordan"


def test_exact_finite_transformations() -> None:
    minkowski = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
    turn = al.rotation(3, 1, 0, 1, Fraction(1, 2))
    assert turn[0][:2] == (Fraction(3, 5), Fraction(4, 5))
    assert al.preserves_form(turn, minkowski)
    boost = al.rotation(3, 1, 0, 3, Fraction(1, 2))
    assert (boost[0][0], boost[0][3]) == (Fraction(5, 3), Fraction(-4, 3))
    assert al.preserves_form(boost, minkowski)
    assert not al.preserves_form([[2, 0], [0, 1]], [[1, 0], [0, 1]])
    # The Cayley transform of every generator of so(2, 2) preserves its form.
    eta = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]
    matrices = al.so(2, 2).matrices
    assert matrices is not None
    for generator in matrices:
        half = [[value / 3 for value in row] for row in generator]
        assert al.preserves_form(al.cayley(half), eta)
    assert al.inverse([[2, 1], [1, 1]]) == al._exact([[1, -1], [-1, 2]])
    assert al.inverse([[0, 1], [1, 0]]) == al._exact([[0, 1], [1, 0]])
    wrong: tuple[Callable[[], object], ...] = (
        lambda: al.inverse([[1, 2], [2, 4]]),
        lambda: al.cayley([[1, 0], [0, 1]]),
        lambda: al.rotation(2, 0, 0, 0, 1),
        lambda: al.rotation(1, 1, 0, 1, 1),
    )
    for call in wrong:
        with pytest.raises(EasyError):
            call()
