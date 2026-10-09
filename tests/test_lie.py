"""Lie algebras and root systems against the classical tables.

Dimensions, Killing-form inertia and root counts are standard (Helgason,
*Differential Geometry, Lie Groups, and Symmetric Spaces*, ch. X; Humphreys,
*Introduction to Lie Algebras and Representation Theory*, sections 12 and 24).
For a real semisimple algebra the Killing form is negative on a maximal
compact subalgebra and positive on its complement.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

import pytest

import anyalgebra.easy as aa
import anyalgebra.lie as al

# algebra, dimension, (positive, negative, zero) inertia of the Killing form
CLASSICAL: tuple[tuple[Callable[[], al.LieAlgebra], int, tuple[int, int, int]], ...] = (
    (lambda: al.so(3), 3, (0, 3, 0)),
    (lambda: al.so(4), 6, (0, 6, 0)),
    (lambda: al.so(5), 10, (0, 10, 0)),
    (lambda: al.so(2, 1), 3, (2, 1, 0)),
    (lambda: al.so(3, 1), 6, (3, 3, 0)),
    (lambda: al.so(2, 2), 6, (4, 2, 0)),
    (lambda: al.so(4, 1), 10, (4, 6, 0)),
    (lambda: al.sl(2), 3, (2, 1, 0)),
    (lambda: al.sl(3), 8, (5, 3, 0)),
    (lambda: al.sp(1), 3, (2, 1, 0)),
    (lambda: al.sp(2), 10, (6, 4, 0)),
    (lambda: al.su(2), 3, (0, 3, 0)),
    (lambda: al.su(3), 8, (0, 8, 0)),
    (lambda: al.su(1, 1), 3, (2, 1, 0)),
    (lambda: al.su(2, 1), 8, (4, 4, 0)),
)


@pytest.mark.parametrize("row", CLASSICAL, ids=[str(n) for n in range(len(CLASSICAL))])
def test_classical_algebras_have_the_tabulated_dimension_and_real_form(
    row: tuple[Callable[[], al.LieAlgebra], int, tuple[int, int, int]],
) -> None:
    make, dimension, signature = row
    found = make()
    assert found.dimension == dimension
    assert found.jacobi_holds()
    assert found.killing_signature() == signature
    assert found.is_semisimple()
    assert found.derived_dimension() == dimension and found.center_dimension() == 0


def _f(*values: int) -> tuple[Fraction, ...]:
    return tuple(Fraction(value) for value in values)


def test_gl_is_reductive_with_a_one_dimensional_center() -> None:
    found = al.gl(2)
    assert found.dimension == 4 and found.jacobi_holds()
    assert not found.is_semisimple() and found.killing_signature() == (2, 1, 1)
    assert found.center_dimension() == 1 and found.derived_dimension() == 3
    assert found.labels == ("E11", "E12", "E21", "E22")


def test_brackets_and_the_convenience_handle() -> None:
    rotations = al.so(3)
    assert rotations.labels == ("L1_2", "L1_3", "L2_3")
    assert rotations.basis_bracket(0, 1) == _f(0, 0, -1)
    assert rotations.bracket((1, 0, 0), (0, 2, 0)) == _f(0, 0, -2)
    assert rotations.adjoint(0)[2][1] == -1
    handle = rotations.as_algebra()
    x, y, z = handle.basis
    assert x * y == -z and aa.assoc(x, y, y) != 0 and handle.unit is None
    assert (x * y) * z + (y * z) * x + (z * x) * y == handle.zero
    assert repr(rotations) == "LieAlgebra('so(3)', dimension=3)"


def test_structure_constants_and_closure_from_matrices() -> None:
    h, e, f = [[1, 0], [0, -1]], [[0, 1], [0, 0]], [[0, 0], [1, 0]]
    found = al.from_matrices([h, e, f], name="sl2", labels=("h", "e", "f"))
    assert found.basis_bracket(0, 1) == _f(0, 2, 0)
    assert found.basis_bracket(0, 2) == _f(0, 0, -2)
    assert found.basis_bracket(1, 2) == _f(1, 0, 0)
    assert found.matrices is not None and len(found.matrices) == 3
    generated = al.generated_by([e, f])
    assert generated.dimension == 3 and generated.killing_signature() == (2, 1, 0)
    lx = [[0, 0, 0], [0, 0, -1], [0, 1, 0]]
    ly = [[0, 0, 1], [0, 0, 0], [-1, 0, 0]]
    assert len(al.lie_closure([lx, ly])) == 3
    assert len(al.lie_closure([lx, lx])) == 1


def test_inertia_handles_zero_diagonals() -> None:
    hyperbolic = [[Fraction(0), Fraction(1)], [Fraction(1), Fraction(0)]]
    assert al.inertia(hyperbolic) == (1, 1, 0)
    assert al.inertia([[Fraction(0)] * 2] * 2) == (0, 0, 2)
    mixed = [[Fraction(v) for v in row] for row in ([0, 0, 1], [0, -3, 0], [1, 0, 0])]
    assert al.inertia(mixed) == (1, 2, 0)


ROOTS = (
    ("A", 1, 2), ("A", 2, 6), ("A", 4, 20), ("B", 2, 8), ("B", 3, 18), ("C", 3, 18),
    ("D", 4, 24), ("D", 5, 40), ("G", 2, 12), ("F", 4, 48), ("E", 6, 72), ("E", 7, 126),
    ("E", 8, 240),
)  # fmt: skip


@pytest.mark.parametrize(("family", "rank", "count"), ROOTS)
def test_root_counts(family: str, rank: int, count: int) -> None:
    found = al.root_system(family, rank)
    assert found.count == count and len(found.positive) == count // 2
    assert found.dimension == count + rank
    assert all(min(root) >= 0 and sum(root) >= 1 for root in found.positive)
    assert repr(found) == f"RootSystem({family}{rank}, roots={count})"


WEYL = (
    ("A", 1, (3,), 4), ("A", 2, (1, 0), 3), ("A", 2, (1, 1), 8), ("A", 2, (2, 0), 6),
    ("B", 2, (1, 0), 5), ("B", 2, (0, 1), 4), ("C", 3, (1, 0, 0), 6),
    ("C", 3, (0, 0, 1), 14), ("D", 4, (1, 0, 0, 0), 8), ("D", 4, (0, 0, 1, 0), 8),
    ("D", 4, (0, 1, 0, 0), 28), ("G", 2, (1, 0), 7), ("G", 2, (0, 1), 14),
    ("F", 4, (0, 0, 0, 1), 26), ("F", 4, (1, 0, 0, 0), 52),
    ("E", 6, (1, 0, 0, 0, 0, 0), 27),
    ("E", 6, (0, 1, 0, 0, 0, 0), 78), ("E", 7, (0, 0, 0, 0, 0, 0, 1), 56),
    ("E", 7, (1, 0, 0, 0, 0, 0, 0), 133), ("E", 8, (0, 0, 0, 0, 0, 0, 0, 1), 248),
)  # fmt: skip


@pytest.mark.parametrize(("family", "rank", "weight", "dimension"), WEYL)
def test_weyl_dimensions(
    family: str, rank: int, weight: tuple[int, ...], dimension: int
) -> None:
    system = al.root_system(family, rank)
    assert system.weyl_dimension(weight) == dimension
    assert system.weyl_dimension((0,) * rank) == 1


def test_requests_without_a_meaning_fail_with_typed_errors() -> None:
    e, f = [[0, 1], [0, 0]], [[0, 0], [1, 0]]
    cases: tuple[tuple[Callable[[], object], str], ...] = (
        (lambda: al.from_matrices([e, f]), "closure"),
        (lambda: al.from_matrices([e, e]), "dependent"),
        (lambda: al.from_matrices([]), "shape"),
        (lambda: al.from_matrices([[[1, 2]]]), "shape"),
        (lambda: al.from_matrices([e, [[1]]]), "shape"),
        (lambda: al.lie_closure([]), "shape"),
        (lambda: al.lie_closure([e, [[1]]]), "shape"),
        (lambda: al.lie_closure([e, f], limit=2), "bound"),
        (lambda: al.so(1), "shape"),
        (lambda: al.sl(1), "shape"),
        (lambda: al.sp(0), "shape"),
        (lambda: al.su(1), "shape"),
        (lambda: al.cartan_matrix("E", 5), "type"),
        (lambda: al.cartan_matrix("X", 2), "type"),
        (lambda: al.root_system("A", 2).weyl_dimension((1,)), "weight"),
        (lambda: al.root_system("A", 2).weyl_dimension((1, -1)), "weight"),
        (lambda: al.LieAlgebra({(0, 1): (1, 0)}, 2), "bracket"),
        (lambda: al.LieAlgebra({(0, 2): (1, 0)}, 2), "shape"),
        (lambda: al.LieAlgebra({}, 2, labels=("a",)), "shape"),
    )
    for attempt, code in cases:
        with pytest.raises(aa.EasyError) as caught:
            attempt()
        assert caught.value.code == code
    abelian = al.LieAlgebra({}, 2)
    assert abelian.jacobi_holds() and abelian.center_dimension() == 2
    assert not abelian.is_semisimple() and abelian.derived_dimension() == 0
    broken = al.LieAlgebra(
        {(0, 1): (0, 0, 1), (1, 0): (0, 0, -1), (0, 2): (1, 0, 0), (2, 0): (-1, 0, 0)},
        3,
    )
    assert not broken.jacobi_holds()
