"""Many-sorted structures, symbolic elements, comparison and rich display."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

import pytest

import anyalgebra.composition as ca
import anyalgebra.easy as aa
from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.polynomials import Polynomial, symbols
from anyalgebra.structures.laws import Equation, Law
from anyalgebra.structures.operations import Operation, PartialOperation
from anyalgebra.structures.outcomes import Indeterminate
from anyalgebra.structures.signatures import OperationSymbol, Signature
from anyalgebra.structures.structure import Structure, StructureBuilder
from anyalgebra.structures.terms import Term, Variable
from anyalgebra.validation.validate import validate_law

CORNERS = "abc"


def _triangle() -> aa.FiniteStructure:
    return aa.structure(
        {"turn": (0, 1, 2), "corner": tuple(CORNERS)},
        {
            "add": ("turn turn -> turn", lambda m, n: (m + n) % 3),
            "move": (
                "turn corner -> corner",
                lambda n, p: CORNERS[(CORNERS.index(p) + n) % 3],
            ),
            "flip": ("corner -> corner", {("a",): "a", ("b",): "c", ("c",): "b"}),
            "zero": ("-> turn", lambda: 0),
            "median": ("turn turn turn -> turn", lambda a, b, c: sorted((a, b, c))[1]),
            "half": ("turn -> turn", {(0,): 0, (2,): 1}),
        },
        name="triangle",
    )


def test_a_structure_with_two_sorts_and_operations_of_any_arity() -> None:
    s = _triangle()
    assert s.move(1, "a") == "b" and s.zero() == 0 and s.median(0, 2, 1) == 1
    assert s.apply("flip", "b") == "c" and s.half(1) is None
    assert s.move(s.half(1), "a") is None
    assert s.report().splitlines() == [
        "triangle: 2 sorts, 6 operations",
        "  turn: 0, 1, 2",
        "  corner: a, b, c",
        "  add: turn turn -> turn, total",
        "  move: turn corner -> corner, total",
        "  flip: corner -> corner, total",
        "  zero: -> turn, total",
        "  median: turn turn turn -> turn, total",
        "  half: turn -> turn, partial",
    ]
    assert s.table("move").splitlines() == [
        "move    a  b  c",
        "0       a  b  c",
        "1       b  c  a",
        "2       c  a  b",
    ]
    assert s.table("move", rules=True).splitlines()[:2] == [
        " move │  a  b  c",
        "──────┼──────────",
    ]
    assert repr(s) == "FiniteStructure('triangle', 2 sorts, 6 operations)"


def test_laws_are_functions_and_fail_with_a_counterexample() -> None:
    s = _triangle()
    action = s.check(
        "action",
        lambda m, n, p: (s.move(m, s.move(n, p)), s.move(s.add(m, n), p)),
        "turn turn corner",
    )
    assert action and str(action) == "action: holds for all 27 assignments"
    found = s.check(
        "flip commutes",
        lambda n, p: (s.flip(s.move(n, p)), s.move(n, s.flip(p))),
        ("turn", "corner"),
    )
    assert not found and found.witness == (1, "a")
    assert str(found) == (
        "flip commutes fails at n = 1, p = a: left side = c, right side = b"
    )
    halves = s.check("halving", lambda n: (s.add(s.half(n), s.half(n)), n), "turn")
    assert (
        str(halves) == "halving fails at n = 1: left side = undefined, right side = 1"
    )
    assert str(s.check("small", lambda n: n < 2, "turn")) == "small fails at n = 2"
    assert str(s.check("zero is a turn", lambda: s.zero() == 0, "")) == (
        "zero is a turn: holds for all 1 assignment"
    )
    one_sort = aa.structure({"bit": (0, 1)}, {"and": ("bit bit -> bit", min)})
    assert one_sort.check("commutative", lambda x, y: (min(x, y), min(y, x)))
    assert one_sort.report().startswith("S: 1 sort, 1 operation")


def test_structure_rejections() -> None:
    s = _triangle()
    wrong: tuple[tuple[str, Callable[[], object]], ...] = (
        ("signature", lambda: aa.structure({"a": (0,)}, {"f": ("a a", min)})),
        ("signature", lambda: aa.structure({"a": (0,)}, {"f": ("a -> b", abs)})),
        ("table", lambda: aa.structure({"a": (0, 0)}, {})),
        ("table", lambda: aa.structure({"a": (0, 1)}, {"f": ("a -> a", lambda x: 5)})),
        ("symbol", lambda: s.apply("spin", 1)),
        ("symbol", lambda: s.move(1)),
        ("symbol", lambda: s.move("a", 1)),
        ("table", lambda: s.table("flip")),
        ("law", lambda: s.check("vague", lambda n: True)),
        ("law", lambda: s.check("short", lambda n, p: True, "turn")),
        ("law", lambda: s.check("odd", lambda n: 3, "turn")),
    )
    for code, call in wrong:
        with pytest.raises(aa.EasyError) as caught:
            call()
        assert caught.value.code == code
    with pytest.raises(AttributeError):
        _ = s.spin


def test_generic_elements_multiply_as_polynomials() -> None:
    quaternions = aa.quaternions()
    x, y = quaternions.generic("a"), quaternions.generic("b")
    assert str(x) == "a0 + a1*i + a2*j + a3*k"
    assert str(x.norm()) == "a0^2 + a1^2 + a2^2 + a3^2"
    assert (x * y).norm() == x.norm() * y.norm()
    assert str((x * y).coefficients["i"]) == "a0*b1 + a1*b0 + a2*b3 - a3*b2"
    assert x * y - y * x == aa.comm(x, y) and aa.comm(x, x) == 0 and not aa.comm(x, x)
    i, j = quaternions.i, quaternions.j
    assert str(i * x) == "-a1 + a0*i - a3*j + a2*k"
    assert str(x * i - i * x) == "2*a3*j - 2*a2*k"
    assert str(x - x) == "0" and str(-x + x) == "0" and str(1 - x + x) == "1"
    assert str(2 * x / 2 + j - x) == "j" and (x + 1 - x) == 1 and x != y
    assert (x**2).coefficients["1"] == x.coordinates[0] ** 2 - sum(
        (value**2 for value in x.coordinates[1:]), Polynomial()
    )
    (t,) = symbols("t")
    assert str(t * i + (t + 1) * quaternions.generic("c") * 0 + x * t) == (
        "a0*t + (a1*t + t)*i + a2*t*j + a3*t*k"
    )
    assert str(i - x * 0 + (t - 1)) == "t - 1 + i"
    assert str((t + 1) * i + (t - 1)) == "t - 1 + (t + 1)*i"
    assert str(i * (t + 1) + (i - (t - 1)) * 0 + ((t + 2) - i) * 0) == "(t + 1)*i"
    assert str(i + t) == "t + i"
    assert repr(x.conj()) == "H(a0 - a1*i - a2*j - a3*k)"
    chosen = x.subs(a0=1, a1=0, a2=Fraction(1, 2), a3=0)
    assert chosen.element() == 1 + j / 2 and chosen == 1 + j / 2
    assert (1 + j / 2 == chosen) and (j == x) is False


def test_generic_rejections() -> None:
    quaternions, octonions = aa.quaternions(), aa.octonions()
    x = quaternions.generic()
    cross = aa.algebra(("x", "y"), {}, name="null", unit=None)
    wrong: tuple[tuple[str, Callable[[], object]], ...] = (
        ("parent", lambda: x + octonions.generic()),
        ("parent", lambda: x * octonions.e1),
        ("symbolic", lambda: x.element()),
        ("power", lambda: x**0),
        ("coordinates", lambda: aa.Generic(quaternions, [Polynomial()])),
        ("unit", lambda: cross.generic() + 1),
        ("involution", lambda: cross.generic().conj()),
        ("norm", lambda: ca_norm_failure()),
    )
    for code, call in wrong:
        with pytest.raises(aa.EasyError) as caught:
            call()
        assert caught.value.code == code
    assert (x == octonions.e1) is False and (cross.generic() == 0) is False
    assert (cross.generic() == "x") is False and (x == "x") is False
    bad: tuple[Callable[[], object], ...] = (
        lambda: x + "x",
        lambda: x - "x",
        lambda: "x" - x,
        lambda: x * "x",
        lambda: "x" * x,
        lambda: x / 0,
        lambda: x / x,
    )
    for call in bad:
        with pytest.raises(TypeError):
            call()


def ca_norm_failure() -> object:
    table = {("1", "1"): "1", ("1", "u"): "u", ("u", "1"): "u", ("u", "u"): "u"}
    algebra = aa.Algebra(
        aa.algebra(("1", "u"), table).structure, conjugation_signs=(1, 1)
    )
    return algebra.generic().norm()


def test_identities_hold_for_every_element_or_fail_on_a_named_one() -> None:
    octonions = aa.octonions()
    alternative = octonions.identity("alternative", lambda x, y: aa.assoc(x, x, y))
    assert alternative and str(alternative) == (
        "alternative: an identity in 16 symbols, so it holds for every element"
    )
    assert octonions.identity(
        "Moufang", lambda x, y, z: (x * (y * (x * z)), ((x * y) * x) * z)
    )
    assert octonions.identity(
        "eight squares", lambda x, y: ((x * y).norm(), x.norm() * y.norm())
    )
    found = octonions.identity("associative", lambda x, y, z: aa.assoc(x, y, z))
    assert not found and str(found) == (
        "associative fails at x = e1, y = e2, z = e4: the difference is 2*e7"
    )
    assert found.witness == (octonions.e1, octonions.e2, octonions.e4)
    sedenions = aa.wrap(ca.sedenions().structure, name="S")
    assert sedenions.identity("flexible", lambda x, y: aa.assoc(x, y, x))
    assert not sedenions.identity("alternative", lambda x, y: aa.assoc(x, x, y))
    quaternions = aa.quaternions()
    # Not multilinear, and zero on every basis element: no basis witness.
    hidden = quaternions.identity(
        "hidden", lambda x: x.coordinates[1] * x.coordinates[2] if _generic(x) else 0
    )
    assert str(hidden) == "hidden fails: the difference is a1*a2"
    with pytest.raises(aa.EasyError, match="at most"):
        quaternions.identity("many", lambda a, b, c, d, e, f, g, h, i: 0)


def _generic(value: object) -> bool:
    return isinstance(value, aa.Generic)


def test_comparison_rules_and_html() -> None:
    quaternions, octonions = aa.quaternions(), aa.octonions()
    assert aa.compare(quaternions, octonions).splitlines() == [
        "                      H        O",
        "dimension             4        8",
        "commutative           False    False",
        "associative           True     False",
        "unit                  present  present",
        "center dimension      1        1",
        "nucleus dimension     4        1",
        "derivation dimension  3        14",
    ]
    with pytest.raises(aa.EasyError):
        aa.compare()
    assert quaternions.table(rules=True).splitlines() == [
        "   │   1   i   j   k",
        "───┼─────────────────",
        " 1 │   1   i   j   k",
        " i │   i  -1   k  -j",
        " j │   j  -k  -1   i",
        " k │   k   j  -i  -1",
    ]
    assert quaternions.table().splitlines()[1] == "1     1   i   j   k"
    x = octonions.e3 / 2 - 2 * octonions.e7
    assert f"{x:h}" == "½ e<sub>3</sub> &minus; 2 e<sub>7</sub>"
    assert f"{-quaternions.i:h}" == "&minus;i" and f"{quaternions.zero:h}" == "0"
    shown = quaternions._repr_html_()
    assert shown.startswith("<table><caption>H: row times column</caption>")
    assert shown.count("<tr>") == 5 and "&minus;1" in shown
    game = aa.magma((0, 1), lambda a, b: a if a == b else None, name="<M>")
    assert "&lt;M&gt;" in game._repr_html_() and "&middot;" in game._repr_html_()
    assert game.table(rules=True).splitlines()[0] == " * │  0  1"


def _bits(rule: Operation | PartialOperation, symbol: OperationSymbol) -> Structure:
    bit = symbol.output
    builder = StructureBuilder(Signature((bit,), (symbol,)))
    builder.with_carrier(bit, FiniteCarrier((0, 1), sort=bit))
    builder.with_operation(symbol, rule)
    return builder.freeze()


def test_kernel_validation_reports_read_as_sentences() -> None:
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    symbol = OperationSymbol("op", (bit, bit), bit)
    x, y = Variable("x", bit), Variable("y", bit)
    tx, ty = Term.variable(x), Term.variable(y)
    law = Law(
        "commutative",
        (x, y),
        Equation(Term.apply(symbol, tx, ty), Term.apply(symbol, ty, tx)),
    )

    def table(rule: object) -> Operation:
        assert callable(rule)
        cells = tuple(((a, b), rule(a, b)) for a in (0, 1) for b in (0, 1))
        return Operation.from_table(symbol, ((bit, carrier),), cells)

    proved = validate_law(_bits(table(lambda a, b: a & b), symbol), law)
    assert str(proved) == "Proved: commutative holds on all 4 assignments"
    assert aa.explain(proved) == (
        "commutative holds: op(x, y) = op(y, x) for all 4 assignments"
    )
    disproved = validate_law(_bits(table(lambda a, _b: a), symbol), law)
    assert str(disproved) == (
        "Disproved: commutative fails at assignment indices (0, 1), "
        "found after 2 of 4 assignments"
    )
    assert aa.explain(disproved) == (
        "commutative fails at x = 0, y = 1: op(x, y) = 0, but op(y, x) = 1"
    )
    unary = OperationSymbol("limit", (bit,), bit)
    vague = PartialOperation.from_callable(
        unary, ((bit, carrier),), lambda _value: Indeterminate("limit")
    )
    undecided = validate_law(
        _bits(vague, unary), Law("fixed", (x,), Equation(Term.apply(unary, tx), tx))
    )
    assert str(undecided) == "Inconclusive: fixed has 2 undecidable assignments among 2"
    assert aa.explain(undecided) == (
        "fixed is undecided: 2 of 2 assignments could not be evaluated"
    )
    nothing = PartialOperation.from_table(
        unary, ((bit, carrier),), (), undefined_marker=object()
    )
    missing = validate_law(
        _bits(nothing, unary), Law("fixed", (x,), Equation(Term.apply(unary, tx), tx))
    )
    assert aa.explain(missing) == (
        "fixed fails at x = 0: limit(x) = undefined, but x = 0"
    )
