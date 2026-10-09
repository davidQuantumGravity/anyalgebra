"""Subtraction, negation, scaling and operators for exact matrices."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from anyalgebra.core.domains import QQ, ZZ
from anyalgebra.linear.matrix import MatrixDefinitionError, MatrixSpace


def test_subtract_negate_scale_and_operators_agree_entrywise() -> None:
    space = MatrixSpace(2, 2, QQ())
    a = space.element(((1, 2), (3, 4)))
    b = space.element(((0, 1), (1, 0)))
    assert a - b == a.subtract(b) == space.element(((1, 1), (2, 4)))
    assert -a == a.negate() == space.element(((-1, -2), (-3, -4)))
    assert a + b == a.add(b)
    assert a @ b == a.matmul(b) == space.element(((2, 1), (4, 3)))
    assert a.scale((1, 2)) == space.element((((1, 2), 1), ((3, 2), 2)))
    assert a.scale(QQ().element(2)) == a + a
    assert (a @ b) - (b @ a) == space.element(((-1, -3), (3, 1)))


def test_arithmetic_keeps_the_literal_parent_rules() -> None:
    space = MatrixSpace(1, 1, QQ())
    other = MatrixSpace(1, 1, QQ())
    a = space.element(((1,),))
    attempts: tuple[Callable[[], object], ...] = (
        lambda: a.subtract(other.element(((1,),))),
        lambda: a - 1,
    )
    for attempt in attempts:
        with pytest.raises(MatrixDefinitionError, match="literal same MatrixSpace"):
            attempt()
    with pytest.raises(MatrixDefinitionError, match="scalar construction failed"):
        a.scale("not a number")
    with pytest.raises(MatrixDefinitionError, match="scalar construction failed"):
        a.scale(ZZ().element(2))


def test_entry_failures_are_normalized() -> None:
    class Entry:
        def __init__(self, parent: object) -> None:
            self.parent = parent

        def negate(self) -> object:
            raise RuntimeError("private")

    class Parent:
        def element(self, value: object) -> object:
            return Entry(self) if not isinstance(value, Entry) else value

    space = MatrixSpace(1, 1, Parent())
    with pytest.raises(MatrixDefinitionError, match="negation failed"):
        space.element(((0,),)).negate()
