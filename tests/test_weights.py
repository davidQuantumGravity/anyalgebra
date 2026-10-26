"""Weights, characters, tensor products and global structure.

The expected values are the standard ones, as tabulated by Slansky, "Group
theory for unified model building", Phys. Rep. 79 (1981), and by Feger and
Kephart, "LieART", Comput. Phys. Commun. 192 (2015).
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

import pytest

import anyalgebra.weights as aw
from anyalgebra.easy import EasyError

DIMENSIONS = [
    ("A1", (3,), 4),
    ("A2", (1, 1), 8),
    ("A2", (2, 0), 6),
    ("A2", (2, 1), 15),
    ("A4", (0, 1, 0, 0), 10),
    ("B2", (0, 1), 4),
    ("B3", (0, 0, 1), 8),
    ("B4", (0, 0, 0, 1), 16),
    ("C3", (0, 0, 1), 14),
    ("D4", (0, 1, 0, 0), 28),
    ("D5", (0, 0, 0, 0, 1), 16),
    ("D5", (0, 0, 0, 0, 2), 126),
    ("G2", (1, 0), 7),
    ("G2", (0, 1), 14),
    ("G2", (2, 0), 27),
    ("F4", (0, 0, 0, 1), 26),
    ("F4", (1, 0, 0, 0), 52),
    ("E6", (1, 0, 0, 0, 0, 0), 27),
    ("E6", (0, 1, 0, 0, 0, 0), 78),
    ("E6", (0, 0, 1, 0, 0, 0), 351),
    ("E7", (0, 0, 0, 0, 0, 0, 1), 56),
    ("E7", (1, 0, 0, 0, 0, 0, 0), 133),
]


@pytest.mark.parametrize(("name", "weight", "dimension"), DIMENSIONS)
def test_dimensions_and_weight_counts(
    name: str, weight: tuple[int, ...], dimension: int
) -> None:
    kind = aw.lie_type(name)
    assert kind.dimension(weight) == dimension
    if dimension <= 60:
        assert sum(kind.character(weight).values()) == dimension


def _dims(kind: aw.LieType, content: aw.Character) -> list[int]:
    return sorted(
        dimension
        for weight, count in content.items()
        for dimension in [kind.dimension(weight)] * count
    )


def test_tensor_products_match_the_tables() -> None:
    a2 = aw.lie_type("A2")
    three, bar, eight = (1, 0), (0, 1), (1, 1)
    assert _dims(a2, a2.tensor(three, three)) == [3, 6]
    assert _dims(a2, a2.tensor(three, bar)) == [1, 8]
    assert _dims(a2, a2.tensor(eight, eight)) == [1, 8, 8, 10, 10, 27]
    assert a2.show(a2.tensor(three, three)) == "6 + 3*"
    g2 = aw.lie_type("G2")
    assert _dims(g2, g2.tensor((1, 0), (1, 0))) == [1, 7, 14, 27]
    e6 = aw.lie_type("E6")
    first = (1, 0, 0, 0, 0, 0)
    assert _dims(e6, e6.tensor(first, first)) == [27, 351, 351]
    assert e6.show(e6.tensor(first, first)) == "351 + 351' + 27*"
    assert _dims(e6, e6.tensor(first, e6.dual(first))) == [1, 78, 650]
    e7 = aw.lie_type("E7")
    small = (0, 0, 0, 0, 0, 0, 1)
    assert _dims(e7, e7.tensor(small, small)) == [1, 133, 1463, 1539]
    d5 = aw.lie_type("D5")
    spinor = (0, 0, 0, 0, 1)
    assert _dims(d5, d5.tensor(spinor, spinor)) == [10, 120, 126]
    # The product does not depend on the order of the factors.
    assert a2.tensor((2, 0), (1, 1)) == a2.tensor((1, 1), (2, 0))


def test_symmetric_and_exterior_powers() -> None:
    e6 = aw.lie_type("E6")
    first = (1, 0, 0, 0, 0, 0)
    assert _dims(e6, e6.symmetric_power(first, 2)) == [27, 351]
    assert _dims(e6, e6.exterior_power(first, 2)) == [351]
    # The cubic invariant of the 27.
    assert _dims(e6, e6.symmetric_power(first, 3)) == [1, 650, 3003]
    e7 = aw.lie_type("E7")
    small = (0, 0, 0, 0, 0, 0, 1)
    assert _dims(e7, e7.symmetric_power(small, 2)) == [133, 1463]
    assert _dims(e7, e7.exterior_power(small, 2)) == [1, 1539]
    a4 = aw.lie_type("A4")
    five = (1, 0, 0, 0)
    assert a4.exterior_power(five, 2) == {aw._weight((0, 1, 0, 0)): 1}
    assert a4.exterior_power(five, 5) == {aw._weight((0, 0, 0, 0)): 1}
    assert a4.exterior_power(five, 6) == {} and a4.show({}) == "0"
    assert a4.symmetric_power(five, 0) == {aw._weight((0, 0, 0, 0)): 1}
    for kind, weight in ((aw.lie_type("B3"), (0, 0, 1)), (aw.lie_type("C2"), (1, 0))):
        both = dict(kind.symmetric_power(weight, 2))
        for key, count in kind.exterior_power(weight, 2).items():
            both[key] = both.get(key, 0) + count
        assert both == kind.tensor(weight, weight)
    with pytest.raises(EasyError):
        a4.symmetric_power(five, -1)


def test_multiplicities_by_freudenthal() -> None:
    zero2, zero6 = aw._weight((0, 0)), aw._weight((0,) * 6)
    assert aw.lie_type("A2").character((1, 1))[zero2] == 2
    assert aw.lie_type("G2").character((0, 1))[zero2] == 2
    assert aw.lie_type("G2").character((1, 0))[zero2] == 1
    assert aw.lie_type("E6").dominant_character((0, 1, 0, 0, 0, 0))[zero6] == 6
    b2 = aw.lie_type("B2")
    content = b2.dominant_character((1, 1))
    assert b2.dimension((1, 1)) == 16 and content[aw._weight((0, 1))] == 2
    a2 = aw.lie_type("A2")
    assert a2.decompose(a2.character((2, 1))) == {aw._weight((2, 1)): 1}
    with pytest.raises(EasyError, match="do not form a character"):
        a2.decompose({aw._weight((1, 0)): -1})
    with pytest.raises(EasyError, match="do not form a character"):
        a2.decompose({aw._weight((Fraction(1, 2), 0)): 1})


REALITY = [
    ("A1", (1,), "pseudoreal", 1),
    ("A1", (2,), "real", 4),
    ("A2", (1, 0), "complex", 1),
    ("A2", (1, 1), "real", 6),
    ("B2", (0, 1), "pseudoreal", 1),
    ("B3", (0, 0, 1), "real", 2),
    ("C2", (1, 0), "pseudoreal", 1),
    ("D4", (0, 0, 0, 1), "real", 2),
    ("D5", (0, 0, 0, 0, 1), "complex", 4),
    ("D6", (0, 0, 0, 0, 0, 1), "pseudoreal", 8),
    ("G2", (1, 0), "real", 2),
    ("F4", (0, 0, 0, 1), "real", 6),
    ("E6", (1, 0, 0, 0, 0, 0), "complex", 6),
    ("E7", (0, 0, 0, 0, 0, 0, 1), "pseudoreal", 12),
]


@pytest.mark.parametrize(("name", "weight", "reality", "index"), REALITY)
def test_reality_and_dynkin_index(
    name: str, weight: tuple[int, ...], reality: str, index: int
) -> None:
    kind = aw.lie_type(name)
    assert kind.reality(weight) == reality and kind.index(weight) == index


def test_adjoint_index_is_twice_the_dual_coxeter_number() -> None:
    for name, coxeter in (("A3", 4), ("B3", 5), ("C3", 4), ("D4", 6), ("G2", 4)):
        kind = aw.lie_type(name)
        adjoint = kind.highest_root()
        assert kind.dimension(adjoint) == kind.algebra_dimension
        assert (
            kind.index(adjoint) == 2 * coxeter and kind.casimir(adjoint) == 2 * coxeter
        )
    for name, coxeter in (("F4", 9), ("E6", 12), ("E7", 18)):
        kind = aw.lie_type(name)
        assert kind.casimir(kind.highest_root()) == 2 * coxeter


def test_centre_congruency_and_global_forms() -> None:
    centres = {"A1": (2,), "A2": (3,), "A3": (4,), "B3": (2,), "C3": (2,), "D4": (2, 2)}
    centres |= {"D5": (4,), "E6": (3,), "E7": (2,), "G2": (), "F4": ()}
    for name, expected in centres.items():
        assert aw.lie_type(name).center() == expected
    a2 = aw.lie_type("A2")
    assert a2.congruency((1, 0)) != a2.congruency((0, 1))
    assert a2.congruency((1, 1)) == (0,) and a2.congruency((3, 0)) == (0,)
    for kind, weights in (
        (a2, ((1, 0), (0, 1), (2, 1))),
        (aw.lie_type("D4"), ((1, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))),
    ):
        # Every root lies in the root lattice, and the class is additive.
        assert all(not any(kind.congruency(root)) for root in kind.simple_roots)
        for first in weights:
            for second in weights:
                total = tuple(a + b for a, b in zip(first, second, strict=True))
                assert kind.congruency(total) == tuple(
                    (a + b) % n
                    for a, b, n in zip(
                        kind.congruency(first),
                        kind.congruency(second),
                        kind.center(),
                        strict=True,
                    )
                )
    d4 = aw.lie_type("D4")
    vector, spinor, adjoint = (1, 0, 0, 0), (0, 0, 0, 1), (0, 1, 0, 0)
    # Spin(8), SO(8) and the adjoint group PSO(8).
    assert d4.fundamental_group([vector, spinor]) == 1
    assert d4.fundamental_group([vector]) == 2 and d4.acting_center([vector]) == 2
    assert d4.fundamental_group([adjoint]) == 4 and d4.acting_center([adjoint]) == 1
    e6 = aw.lie_type("E6")
    assert e6.fundamental_group([(0, 1, 0, 0, 0, 0)]) == 3
    assert e6.fundamental_group([(1, 0, 0, 0, 0, 0)]) == 1
    assert aw.lie_type("G2").fundamental_group([(1, 0)]) == 1


def test_smith_normal_form() -> None:
    assert aw.smith_normal_form([[2, 0], [0, 3]])[0] == [1, 6]
    assert aw.smith_normal_form([[2, 4], [4, 2]])[0] == [2, 6]
    assert aw.smith_normal_form([[0, 0], [0, 0]])[0] == [0, 0]
    factors, columns = aw.smith_normal_form([[2, -1], [-1, 2]])
    assert (
        factors == [1, 3]
        and abs(columns[0][0] * columns[1][1] - columns[0][1] * columns[1][0]) == 1
    )


def test_reductive_types_carry_charges() -> None:
    model = aw.lie_type("A2 + A1 + U1")
    assert model.name == "A2+A1+U1" and model.rank == 4 and not model.is_simple
    assert model.algebra_dimension == 12 and repr(model) == "LieType(A2+A1+U1)"
    quark = (1, 0, 1, Fraction(1, 3))
    assert model.dimension(quark) == 6 and model.label(quark) == "(3,2)(1/3)"
    assert model.label((0, 1, 0, -1)) == "(3*,1)(-1)"
    product = model.tensor(quark, (0, 1, 1, Fraction(-1, 3)))
    assert model.show(product) == "(8,3)(0) + (8,1)(0) + (1,3)(0) + (1,1)(0)"
    assert model.total_dimension(product) == 36
    assert model.dual(quark) == aw._weight((0, 1, 1, Fraction(-1, 3)))
    assert model.reality(quark) == "complex"
    torus = aw.lie_type("U1")
    assert torus.dimension((2,)) == 1 and torus.label((2,)) == "1(2)"
    assert torus.character((2,)) == {aw._weight((2,)): 1} and torus.center() == ()
    assert aw.lie_type("A1+A1").key == aw.LieType([("A", 1), ("A", 1)]).key
    assert aw.lie_type("A1") == aw.lie_type("a1") and aw.lie_type("A1") != "A1"
    assert len({aw.lie_type("A1"), aw.lie_type("A1")}) == 1
    a1 = aw.lie_type("A1")
    assert a1.orbit((2,)) == (aw._weight((2,)), aw._weight((-2,)))
    assert a1.to_dominant((-3,)) == (aw._weight((3,)), -1)
    assert a1.inner((1,), (1,)) == Fraction(1, 2)


def test_rejections() -> None:
    a2, model = aw.lie_type("A2"), aw.lie_type("A2+U1")
    wrong: tuple[tuple[str, Callable[[], object]], ...] = (
        ("type", lambda: aw.lie_type("X2")),
        ("type", lambda: aw.lie_type("A")),
        ("type", lambda: aw.lie_type("D3")),
        ("type", lambda: aw.LieType([])),
        ("weight", lambda: a2.dimension((1,))),
        ("weight", lambda: a2.dimension((1, -1))),
        ("weight", lambda: a2.character((Fraction(1, 2), 0))),
        ("type", lambda: model.index((1, 0, 0))),
        ("type", lambda: model.highest_root()),
    )
    for code, call in wrong:
        with pytest.raises(EasyError) as caught:
            call()
        assert caught.value.code == code
