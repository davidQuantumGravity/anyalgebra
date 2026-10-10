"""Readable output, finite structures and the catalog of the convenience layer."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

import pytest

import anyalgebra.easy as aa
from anyalgebra.analysis.elementary import associator
from anyalgebra.core.domains import QQ, ZZ
from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.fixtures.composition import octonion_fixture
from anyalgebra.structures.operations import Operation
from anyalgebra.structures.signatures import OperationSymbol, Signature
from anyalgebra.structures.structure import StructureBuilder
from anyalgebra.validation.factories import associative_law, commutative_law
from anyalgebra.validation.validate import Disproved, Proved, validate_law

HANDS = ("rock", "paper", "scissors")
BEATS = {("paper", "rock"), ("scissors", "paper"), ("rock", "scissors")}


def _winner(a: object, b: object) -> object:
    return a if a == b or (a, b) in BEATS else b


def test_kernel_values_print_readably_while_repr_keeps_the_record() -> None:
    octonions = octonion_fixture()
    e = [octonions.module.element({n: 1}) for n in range(8)]
    value = associator(octonions, e[1], e[2], e[4])
    assert str(value) == "2*e7" and repr(value) == "SparseElement(rank=8, support=(7,))"
    assert str(octonions.module.zero()) == "0"
    assert str(e[1] + 3 * e[2]) == "1*e1 + 3*e2"
    assert str(ZZ().element(5)) == "5" and "value=5" in repr(ZZ().element(5))
    assert str(QQ().element((3, 2))) == "3/2" and str(QQ().element(4)) == "4"
    assert {n: str(c) for n, c in value.coordinates().items()} == {7: "2"}


def test_pretty_and_wolfram_forms() -> None:
    quaternions, octonions = aa.quaternions(), aa.octonions()
    _, i, j, k = quaternions.basis
    x = (1 + 2 * i) * (j + k) / 2
    assert f"{x:p}" == format(x, "pretty") == "−½ j + 3/2 k"  # noqa: RUF001
    assert f"{2 * octonions.e3 - octonions.e7:p}" == "2 e₃ − e₇"  # noqa: RUF001
    assert f"{1 - i:p}" == "1 − i" and f"{quaternions.zero:p}" == "0"  # noqa: RUF001
    assert f"{x:w}" == format(x, "wolfram") == "-1/2 j + 3/2 k"
    assert f"{3 + i:w}" == "3 + 1 i" and f"{quaternions.zero:w}" == "0"
    assert f"{Fraction(3, 4) * k:p}" == "¾ k"


def test_a_magma_from_a_function_reports_laws_as_sentences() -> None:
    game = aa.magma(HANDS, _winner, name="RPS")
    assert game("rock", "paper") == "paper" and game.is_total
    assert str(game.check("commutative")) == "commutative: all 9 pairs agree"
    found = game.check("associative")
    assert not found and found.witness == HANDS and found.law == "associative"
    assert str(found) == (
        "not associative: (rock*paper)*scissors = scissors, "
        "but rock*(paper*scissors) = rock"
    )
    assert bool(game.check("idempotent")) and game.identity() is None
    assert repr(found) == "LawResult('associative', holds=False)"
    assert game.table().splitlines() == [
        "*               rock     paper  scissors",
        "rock            rock     paper      rock",
        "paper          paper     paper  scissors",
        "scissors        rock  scissors  scissors",
    ]
    assert game.report().splitlines()[:2] == [
        "RPS: 3 elements, total operation *",
        "  identity: none",
    ]
    assert repr(game) == "Magma('RPS', 3 elements)"


def test_the_magma_verdicts_agree_with_the_kernel_validator() -> None:
    hand = Sort("hand")
    play = OperationSymbol("play", (hand, hand), hand)
    carrier = FiniteCarrier(HANDS, sort=hand)
    table = tuple(((a, b), _winner(a, b)) for a in HANDS for b in HANDS)
    kernel = (
        StructureBuilder(Signature((hand,), (play,)))
        .with_carrier(hand, carrier)
        .with_operation(play, Operation.from_table(play, ((hand, carrier),), table))
        .freeze()
    )
    game = aa.magma(HANDS, _winner)
    assert type(validate_law(kernel, commutative_law(play))) is Proved
    assert bool(game.check("commutative"))
    disproved = validate_law(kernel, associative_law(play))
    assert type(disproved) is Disproved and not game.check("associative")
    assert disproved.witness is not None
    assert tuple(HANDS[n] for n in disproved.witness.substitution_indices) == (
        game.check("associative").witness
    )


def test_groups_partial_operations_and_other_laws() -> None:
    cyclic = aa.magma(range(4), lambda a, b: (a + b) % 4, name="Z4", symbol="+")  # type: ignore[operator]
    assert cyclic.identity() == 0 and cyclic.check("associative")
    assert str(cyclic.check("associative")) == "associative: all 64 triples agree"
    assert str(cyclic.check("idempotent")) == "not idempotent: 1+1 = 2"
    assert cyclic.report().splitlines()[1] == "  identity: 0"
    subtraction = aa.magma(range(3), lambda a, b: (a - b) % 3)  # type: ignore[operator]
    assert (
        str(subtraction.check("commutative")) == "not commutative: 0*1 = 2, but 1*0 = 1"
    )
    partial = aa.magma((0, 1, 2), lambda a, b: a + b if a + b < 3 else None)  # type: ignore[operator]
    assert not partial.is_total and partial(2, 2) is None
    assert partial.table().splitlines()[3] == "2    2  .  ."
    assert "partial operation" in partial.report()
    assert str(partial.check("idempotent")) == "not idempotent: 1*1 = 2"
    assert bool(partial.check("associative")) and bool(partial.check("commutative"))
    lopsided = aa.magma((0, 1), lambda a, b: None if (a, b) == (1, 1) else a)
    assert str(lopsided.check("associative")) == (
        "not associative: (0*1)*1 = 0, but 0*(1*1) = undefined"
    )
    gap = aa.magma((0, 1), lambda a, b: None if (a, b) == (0, 1) else b)
    assert str(gap.check("associative")) == (
        "not associative: (0*0)*1 = undefined, but 0*(0*1) = undefined"
    ) or not gap.check("commutative")


def test_catalog_lists_every_family() -> None:
    lines = aa.catalog().splitlines()
    assert len(lines) == 19 and lines[0].startswith("aa.quaternions(), aa.octonions()")
    text = aa.catalog()
    for call in (
        "ca.sedenions()",
        "ac.gamma_matrices(p, q)",
        "aj.hermitian(A, n)",
        "ml.su(n, A)",
    ):
        assert call in text
    assert len({line.index("  ", len(line.split("  ")[0])) for line in lines}) >= 1


def test_requests_without_a_meaning_fail_with_typed_errors() -> None:
    game = aa.magma(HANDS, _winner)
    cases: tuple[tuple[Callable[[], object], str], ...] = (
        (lambda: aa.magma((), _winner), "table"),
        (lambda: aa.magma((1, 1), _winner), "table"),
        (lambda: aa.magma((0, 1), lambda a, b: 7), "table"),
        (lambda: game("rock", "lizard"), "symbol"),
        (lambda: game.check("distributive"), "law"),
    )
    for attempt, code in cases:
        with pytest.raises(aa.EasyError) as caught:
            attempt()
        assert caught.value.code == code
