"""Subalgebras, branching rules and chains.

The expected rules are the standard ones tabulated by Slansky, Phys. Rep. 79
(1981), tables 14 to 58.  Bars are written ``*``; which member of a dual
pair carries it is a convention, so the tests compare dimensions and charges
where the convention could matter.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from fractions import Fraction

import pytest

import anyalgebra.branching as ab
import anyalgebra.lie as al
import anyalgebra.lie_structure as ls
import anyalgebra.weights as aw
from anyalgebra.easy import EasyError

T = aw.lie_type


def _terms(step: ab.Embedding, weight: tuple[int, ...]) -> list[str]:
    return sorted(step.show(weight).split(" = ")[1].replace("*", "").split(" + "))


def _one(source: str, target: str) -> ab.Embedding:
    (found,) = [
        step for step in ab.maximal_subalgebras(T(source)) if step.target.name == target
    ]
    return found


RULES = [
    ("E6", "D5+U1", (1, 0, 0, 0, 0, 0), ["1(4)", "10(-2)", "16(1)"]),
    ("E6", "A5+A1", (1, 0, 0, 0, 0, 0), ["(15,1)", "(6,2)"]),
    ("E6", "A2+A2+A2", (1, 0, 0, 0, 0, 0), ["(1,3,3)", "(3,1,3)", "(3,3,1)"]),
    ("E6", "F4", (1, 0, 0, 0, 0, 0), ["1", "26"]),
    ("E6", "F4", (0, 1, 0, 0, 0, 0), ["26", "52"]),
    ("E7", "E6+U1", (0, 0, 0, 0, 0, 0, 1), ["1(-3)", "1(3)", "27(-1)", "27(1)"]),
    ("E7", "D6+A1", (0, 0, 0, 0, 0, 0, 1), ["(12,2)", "(32,1)"]),
    ("E7", "A7", (0, 0, 0, 0, 0, 0, 1), ["28", "28"]),
    ("E7", "A7", (1, 0, 0, 0, 0, 0, 0), ["63", "70"]),
    ("F4", "B4", (0, 0, 0, 1), ["1", "16", "9"]),
    ("F4", "B4", (1, 0, 0, 0), ["16", "36"]),
    ("F4", "C3+A1", (0, 0, 0, 1), ["(14,1)", "(6,2)"]),
    ("F4", "A2+A2", (0, 0, 0, 1), ["(1,8)", "(3,3)", "(3,3)"]),
    ("G2", "A2", (1, 0), ["1", "3", "3"]),
    ("G2", "A2", (0, 1), ["3", "3", "8"]),
    ("G2", "A1+A1", (1, 0), ["(2,2)", "(3,1)"]),
    ("G2", "A1", (1, 0), ["7"]),
    ("G2", "A1", (0, 1), ["11", "3"]),
    ("B3", "G2", (0, 0, 1), ["1", "7"]),
    ("B3", "G2", (1, 0, 0), ["7"]),
    ("D4", "B3", (1, 0, 0, 0), ["1", "7"]),
    ("D4", "B3", (0, 0, 0, 1), ["8"]),
    ("D5", "B4", (0, 0, 0, 0, 1), ["16"]),
    ("D5", "A4+U1", (1, 0, 0, 0, 0), ["5(-2)", "5(2)"]),
    ("A4", "A2+A1+U1", (1, 0, 0, 0), ["(1,2)(-3)", "(3,1)(2)"]),
    ("A3", "C2", (0, 1, 0), ["1", "5"]),
    ("A5", "C3", (0, 1, 0, 0, 0), ["1", "14"]),
    ("A4", "B2", (1, 0, 0, 0), ["5"]),
    ("C3", "A1", (1, 0, 0), ["6"]),
    ("C3", "A2+U1", (1, 0, 0), ["3(-1)", "3(1)"]),
]


@pytest.mark.parametrize(("source", "target", "weight", "expected"), RULES)
def test_branching_rules_match_the_tables(
    source: str, target: str, weight: tuple[int, ...], expected: list[str]
) -> None:
    candidates = [
        step for step in ab.maximal_subalgebras(T(source)) if step.target.name == target
    ]
    assert candidates
    found = [_terms(step, weight) for step in candidates]
    # The overall sign of a U1 charge is a convention.
    flipped = [sorted(_flip(term) for term in terms) for terms in found]
    assert sorted(expected) in found or sorted(expected) in flipped


def _flip(term: str) -> str:
    match = re.fullmatch(r"(.*)\((-?)(\d+)\)", term)
    if match is None or "," in match.group(3):
        return term
    head, sign, number = match.groups()
    return f"{head}({'' if sign or number == '0' else '-'}{number})"


@pytest.mark.parametrize("name", ["A2", "A3", "B2", "B3", "C3", "D4", "G2", "F4", "E6"])
def test_every_listed_subalgebra_preserves_dimensions(name: str) -> None:
    source = T(name)
    weights = [source.highest_root()]
    for k in range(source.rank):
        fundamental = [0] * source.rank
        fundamental[k] = 1
        weights.append(aw._weight(fundamental))
    for step in ab.maximal_subalgebras(source):
        assert step.source == source and step.target.rank <= source.rank
        for weight in weights:
            content = step.branch(weight)
            assert step.target.total_dimension(content) == source.dimension(weight)
        # The adjoint representation contains the adjoint of the subalgebra.
        adjoint = step.branch(source.highest_root())
        inside = sum(
            count
            for weight, count in adjoint.items()
            if step.target.dimension(weight) > 1 or not any(weight)
        )
        assert inside >= 1


def test_marks_regular_subalgebras_and_levi_charges() -> None:
    assert ab.marks(T("E6")) == (1, 2, 2, 3, 2, 1)
    assert ab.marks(T("G2")) == (3, 2) and ab.marks(T("F4")) == (2, 3, 4, 2)
    # Deleting the extra node gives the algebra back.
    same = ab.regular(T("B3"), 3)
    assert same.target == T("B3") and same.branch((0, 0, 1)) == {
        aw._weight((0, 0, 1)): 1
    }
    # A node with a composite mark gives a subalgebra that is not maximal.
    lower = ab.regular(T("F4"), 2)
    assert lower.target.name == "A3+A1" and "node 2 of the extended" in lower.label
    assert lower.target.total_dimension(lower.branch((0, 0, 0, 1))) == 26
    levi = ab.regular(T("A2"), 0, extended=False)
    assert levi.target.name == "A1+U1" and levi.show((1, 0)) in (
        "3 = 2(1) + 1(-2)",
        "3 = 2(-1) + 1(2)",
    )
    torus = ab.regular(T("A1"), 0, extended=False)
    assert torus.target.name == "U1" and torus.show((2,)) == "3 = 1(2) + 1(0) + 1(-2)"
    assert repr(levi).startswith("Embedding(A2 > A1+U1")


def test_embedding_indices() -> None:
    def index(step: ab.Embedding) -> list[int]:
        return sorted(int(value) for value in step.index())

    assert index(ab.principal_sl2(T("A2"))) == [4]
    assert index(ab.principal_sl2(T("G2"))) == [28]
    assert index(_one("G2", "A1+A1")) == [1, 3]
    assert index(_one("G2", "A2")) == [1] and index(_one("B3", "G2")) == [1]
    assert index(_one("F4", "B4")) == [1] and index(_one("E6", "F4")) == [1]
    assert index(_one("A3", "C2")) == [1] and index(_one("A4", "B2")) == [2]
    # The principal A1 of the largest exceptional type: Dynkin's index 1240.
    assert index(ab.principal_sl2(T("E8"))) == [1240]


def test_subalgebras_given_by_a_representation() -> None:
    a2, a1 = T("A2"), T("A1")
    # SU(3) in SO(8) by the adjoint representation: all three 8s stay irreducible.
    octet = ab.via_representation(a2, [(1, 1)])
    assert octet.source == T("D4") and "by representation" in octet.label
    for weight in ((1, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)):
        assert octet.show(weight) == "8 = 8"
    assert octet.show((0, 1, 0, 0)) == "28 = 10 + 10* + 8"
    # SU(2) in Sp(4) by the 4, and in SO(5) by the 5.
    assert ab.via_representation(a1, [(3,)]).source == T("C2")
    assert ab.via_representation(a1, [(4,)]).show((0, 1)) == "4 = 4"
    # SU(3) x SU(2) in SU(6) by the (3, 2).
    product = ab.via_representation(T("A2+A1"), [(1, 0, 1)])
    assert product.source == T("A5")
    assert sorted(product.show((0, 1, 0, 0, 0)).split(" = ")[1].split(" + ")) == [
        "(3*,3)",
        "(6,1)",
    ]
    # A reducible representation: SO(7) in SO(8) by 7 + 1.
    assert ab.via_representation(T("B3"), [(1, 0, 0), (0, 0, 0)]).source == T("D4")
    assert ab.via_representation(a2, [(1, 0)]).source == T("A2")
    wrong: tuple[Callable[[], object], ...] = (
        lambda: ab.via_representation(a2, [(1, 0)], "so"),
        lambda: ab.via_representation(a1, [(2,)], "sp"),
        lambda: ab.via_representation(T("A1+U1"), [(1, 0)]),
        lambda: ab.via_representation(a1, [(1,), (1,)], "so"),
    )
    for call in wrong:
        with pytest.raises(EasyError):
            call()


def test_foldings_and_the_principal_subalgebra() -> None:
    assert ab.folding(T("D4")).show((1, 0, 0, 0)) == "8 = 7 + 1"
    assert ab.folding(T("D4")).show((0, 1, 0, 0)) == "28 = 14 + 2 x 7"
    assert ab.folding(T("E6")).target == T("F4")
    with pytest.raises(EasyError, match="no folding"):
        ab.folding(T("A2"))
    # The adjoint representation restricts to spins given by the exponents.
    for name, exponents in (
        ("A2", (1, 2)),
        ("B2", (1, 3)),
        ("G2", (1, 5)),
        ("F4", (1, 5, 7, 11)),
    ):
        kind = T(name)
        content = ab.principal_sl2(kind).branch(kind.highest_root())
        assert sorted(int(weight[0]) // 2 for weight in content) == list(exponents)


def test_chains_compose_projections() -> None:
    found = ab.chains(T("D5"), T("A2+A1+U1+U1"), max_steps=2)
    labels = [chain.label for chain in found]
    assert "D5 > A4+U1 > A2+A1+U1+U1" in labels
    chain = found[labels.index("D5 > A4+U1 > A2+A1+U1+U1")]
    content = chain.branch((0, 0, 0, 0, 1))
    # One family: six multiplets, sixteen states.
    assert len(content) == 6 and chain.target.total_dimension(content) == 16
    assert sorted(chain.target.dimension(weight) for weight in content) == [
        1,
        1,
        2,
        3,
        3,
        6,
    ]
    first = _one("D5", "A4+U1")
    second = ab.on_factor(first.target, 0, _one("A4", "A2+A1+U1"))
    assert first.then(second).branch((0, 0, 0, 0, 1)) == content
    pati_salam = ab.chains(T("D5"), T("A3+A1+A1"), max_steps=1)
    assert [chain.show((0, 0, 0, 0, 1)) for chain in pati_salam] in (
        ["16* = (4,2,1) + (4*,1,2)"],
        ["16* = (4*,2,1) + (4,1,2)"],
        ["16* = (4,1,2) + (4*,2,1)"],
        ["16* = (4*,1,2) + (4,2,1)"],
    )
    assert ab.chains(T("A2"), T("E6")) == [] and ab.chains(T("G2"), T("B4")) == []
    with pytest.raises(EasyError, match="starts from another type"):
        first.then(first)
    with pytest.raises(EasyError, match="does not start from that factor"):
        ab.on_factor(T("A2+A1"), 1, _one("A4", "A2+A1+U1"))


def test_regular_subalgebras_by_type() -> None:
    assert set(ab.regular_subalgebras(T("G2"))) == {"G2", "A2", "A1+A1", "A1"}
    assert set(ab.regular_subalgebras(T("B2"))) == {"B2", "A1+A1", "A1"}
    f4 = ab.regular_subalgebras(T("F4"))
    assert {"F4", "B4", "D4", "C3+A1", "A2+A2", "A3+A1", "B3", "C3"} <= set(f4)
    assert f4[0] in ("F4", "B4", "D4") and "G2" not in f4


def test_a_subalgebra_given_by_explicit_elements() -> None:
    so5 = al.so(5)
    inside = [
        tuple(int(i == k) for i in range(so5.dimension))
        for k, label in enumerate(so5.labels)
        if "5" not in label
    ]
    step = ab.from_subalgebra(so5, inside)
    assert step.source == T("B2") and step.target == T("A1+A1")
    assert step.show((1, 0)) == "5 = (2,2) + (1,1)"
    assert step.show((0, 1)) == "4 = (2,1) + (1,2)" and "explicit" in step.label
    # sl(2) in the corner of sl(3), from the two matrix units E12 and E21.
    sl3 = al.sl(3)
    corner = [tuple(int(i == k) for i in range(8)) for k in (0, 2)]
    small = ab.from_subalgebra(sl3, corner)
    assert small.source == T("A2") and small.target == T("A1")
    assert (
        small.show((1, 0)) == "3 = 2 + 1" and small.show((1, 1)) == "8 = 3 + 2 x 2 + 1"
    )
    sub, basis = ls.subalgebra(sl3, corner, name="corner")
    assert sub.name == "corner" and sub.dimension == 3 and len(basis) == 3
    wrong: tuple[Callable[[], object], ...] = (
        lambda: ls.subalgebra(sl3, []),
        lambda: ls.subalgebra(sl3, [(1, 0)]),
        lambda: ls.root_decomposition(sl3, torus=[aw._weight(corner[0])]),
        lambda: ab.from_subalgebra(al.gl(2), [(1, 0, 0, 0)]),
    )
    for call in wrong:
        with pytest.raises(EasyError):
            call()


def test_shapes_are_checked() -> None:
    wrong: tuple[Callable[[], object], ...] = (
        lambda: ab.Embedding(T("A2"), T("A1"), [[1]]),
        lambda: ab.regular(T("A2"), 5),
        lambda: ab.regular(T("A2+A1"), 0),
        lambda: ab.marks(T("A2+U1")),
    )
    for call in wrong:
        with pytest.raises(EasyError):
            call()
    step = ab.principal_sl2(T("A2"))
    assert step.project((1, 0)) == (Fraction(2),)


REAL_FORMS = [
    ("A", 1), ("A", 2), ("A", 3), ("A", 5), ("B", 2), ("B", 4), ("C", 3), ("C", 4),
    ("D", 4), ("D", 5), ("G", 2), ("F", 4), ("E", 6), ("E", 7), ("E", 8),
]  # fmt: skip


@pytest.mark.parametrize(("family", "size"), REAL_FORMS)
def test_maximal_compact_subalgebras_have_the_dimension_the_character_gives(
    family: str, size: int
) -> None:
    whole = T(f"{family}{size}").algebra_dimension
    for character, names in ls._real_forms(family, size).items():
        for name in names:
            compact = ls.maximal_compact(name)
            dimension = 0 if compact == "0" else T(compact).algebra_dimension
            assert 2 * dimension == whole - character, name


def test_maximal_compact_names() -> None:
    table = {
        "so(9,1)": "B4",
        "so(3,1)": "A1",
        "su(2,2)": "A1+A1+U1",
        "sl(2,R)": "U1",
        "sl(3,R)": "A1",
        "su*(6)": "C3",
        "sp(4,R)": "A1+U1",
        "sp(1,1)": "A1+A1",
        "so*(8)": "A3+U1",
        "e6(-26)": "F4",
        "e7(-25)": "E6+U1",
        "f4(-20)": "B4",
        "su(3)": "A2",
        "so(8)": "D4",
        "sp(3)": "C3",
    }
    for name, expected in table.items():
        assert ls.maximal_compact(name) == expected
    for bad in ("su(x)", "so*(7)", "banana", "sl(3,C)"):
        with pytest.raises(EasyError):
            ls.maximal_compact(bad)
