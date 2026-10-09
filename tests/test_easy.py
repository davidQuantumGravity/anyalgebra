"""The convenience layer agrees with the kernel and prints in several forms."""

from __future__ import annotations

from fractions import Fraction
from collections.abc import Callable
from itertools import product
from typing import Any, cast

import pytest

import anyalgebra.easy as aa
from anyalgebra.algebra.multilinear import (
    FiniteMultilinearStructure,
    evaluate_multilinear,
)
from anyalgebra.core.domains import ZZ
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.fixtures import composition

HANDLES = (aa.quaternions, aa.split_quaternions, aa.octonions, aa.split_octonions)
FIXTURES = (
    composition.quaternion_fixture,
    composition.split_quaternion_fixture,
    composition.octonion_fixture,
    composition.split_octonion_fixture,
)


def _cross() -> aa.Algebra:
    table: dict[tuple[str, str], aa.Product] = {
        ("x", "y"): "z",
        ("y", "x"): "-z",
        ("y", "z"): "x",
        ("z", "y"): "-x",
        ("z", "x"): {"y": 1},
        ("x", "z"): {"y": -1},
        ("x", "x"): 0,
    }
    return aa.algebra(("x", "y", "z"), table, name="cross", unit=None)


@pytest.mark.parametrize(
    ("handle", "fixture"), list(zip(HANDLES, FIXTURES, strict=True))
)
def test_star_agrees_with_the_kernel_on_the_whole_basis_grid(
    handle: Callable[[], aa.Algebra],
    fixture: Callable[[], FiniteMultilinearStructure],
) -> None:
    algebra = handle()
    source = fixture()
    for left, right in product(range(algebra.rank), repeat=2):
        expected = evaluate_multilinear(
            source, source.module.element({left: 1}), source.module.element({right: 1})
        )
        found = algebra.basis[left] * algebra.basis[right]
        assert found.vector == tuple(
            Fraction(cast(Any, expected.coordinates()[n]).value)
            if n in expected.coordinates()
            else 0
            for n in range(algebra.rank)
        )


def test_operators_on_quaternions() -> None:
    quaternions = aa.quaternions()
    one, i, j, k = quaternions.basis
    assert i * j == k and (i * j) * k == -1 and (i * j) * k == -1
    x = (1 + 2 * i) * (j + k) / 2
    assert x == quaternions(0, 0, Fraction(-1, 2), Fraction(3, 2))
    assert x.coefficients == {"j": Fraction(-1, 2), "k": Fraction(3, 2)}
    assert 1 - i == one - i and i - 1 == -(1 - i) and +i == i
    assert i * Fraction(1, 2) == Fraction(1, 2) * i == i / 2
    assert i**2 == -1 and i**3 == -i and i**1 == i
    assert bool(i) and not bool(quaternions.zero)
    assert i != j and i != "i" and quaternions.zero == 0
    assert aa.comm(i, j) == 2 * k and aa.anticomm(i, j) == 0
    assert aa.assoc(i, j, k) == 0
    assert x.raw.parent is quaternions.structure.module
    assert (1 + i).conj() == 1 - i and (1 + i).norm() == 2
    assert (1 + i) * (1 + i).inv() == 1


def test_octonions_are_not_associative_and_split_forms_have_zero_divisors() -> None:
    octonions = aa.octonions()
    e = octonions.basis
    assert aa.assoc(e[1], e[2], e[4]) == 2 * e[7]
    assert aa.comm(e[1], e[2]) == 2 * e[3]
    split = aa.split_octonions()
    u, v = split(1, 0, 0, 0, 1, 0, 0, 0), split(1, 0, 0, 0, -1, 0, 0, 0)
    assert u * v == 0 and u.norm() == 0
    with pytest.raises(aa.EasyError, match="norm zero") as caught:
        u.inv()
    assert caught.value.code == "division"


def test_four_ways_to_name_basis_elements() -> None:
    octonions = aa.octonions()
    assert (
        octonions.e3 is octonions.basis[3]
        and octonions["e3"] is octonions.e3
        and octonions[3] is octonions.e3
        and octonions[-1] is octonions.e7
    )
    assert (
        octonions.one is octonions.basis[0]
        and octonions["one"] is octonions.one
        and octonions.unit is octonions.one
    )
    namespace: dict[str, object] = {}
    assert octonions.inject(namespace) == (
        "e1",
        "e2",
        "e3",
        "e4",
        "e5",
        "e6",
        "e7",
        "one",
    )
    assert namespace["e5"] is octonions.e5
    names = octonions.inject()
    assert "e7" in names and globals()["e7"] is octonions.e7
    for bad in (8, "e9", 1.5):
        with pytest.raises(aa.EasyError) as caught:
            octonions[bad]  # type: ignore[index]
        assert caught.value.code == "symbol"
    with pytest.raises(AttributeError):
        _ = octonions.e9
    with pytest.raises(AttributeError):
        _ = octonions._private
    assert repr(octonions) == "Algebra('O', rank=8)"


def test_print_forms_have_fixed_outputs() -> None:
    quaternions = aa.quaternions()
    _, i, j, k = quaternions.basis
    x = (1 + 2 * i) * (j + k) / 2
    assert str(x) == f"{x}" == f"{x:s}" == format(x, "sum") == "-1/2*j + 3/2*k"
    assert f"{x:v}" == format(x, "vector") == "[0, 0, -1/2, 3/2]"
    assert f"{x:l}" == r"-\frac{1}{2}\,j + \frac{3}{2}\,k"
    assert f"{x:d}" == "{j: -1/2, k: 3/2}"
    assert repr(x) == "H(-1/2*j + 3/2*k)"
    assert x._repr_latex_() == r"$-\frac{1}{2}\,j + \frac{3}{2}\,k$"
    assert str(quaternions.zero) == "0" and f"{quaternions.zero:v}" == "[0, 0, 0, 0]"
    assert str(1 - i + 2 * k) == "1 - i + 2*k" and str(-quaternions.one) == "-1"
    assert str(Fraction(3, 2) + i) == "3/2 + i"
    octonions = aa.octonions()
    assert f"{2 * octonions.e3 - octonions.e7:l}" == r"2\,e_{3} - e_{7}"
    assert set(aa.forms()) >= {"sum", "vector", "latex", "sparse", "s", "v", "l", "d"}
    with pytest.raises(aa.EasyError, match="unknown print form") as caught:
        format(x, "nope")
    assert caught.value.code == "form"


def test_default_form_per_block_per_handle_and_per_session() -> None:
    quaternions = aa.quaternions()
    k = quaternions.k
    with aa.display("vector"):
        assert str(k) == "[0, 0, 0, 1]"
        with aa.display("sparse"):
            assert str(k) == "{k: 1}"
        assert str(k) == "[0, 0, 0, 1]"
    assert str(k) == "k"
    quaternions.display = "vector"
    assert str(k) == "[0, 0, 0, 1]" and f"{k:s}" == "k"
    with aa.display("sum"):
        assert str(k) == "k"
    quaternions.display = None
    aa.set_display("latex")
    try:
        assert str(2 * k) == r"2\,k"
    finally:
        aa.set_display("sum")
    assert (
        aa.Algebra(quaternions.structure, display="vector").basis[1].__str__()
        == "[0, 1, 0, 0]"
    )
    with pytest.raises(aa.EasyError):
        aa.set_display("nope")
    with pytest.raises(aa.EasyError), aa.display("nope"):
        pass  # pragma: no cover
    with pytest.raises(aa.EasyError):
        aa.Algebra(quaternions.structure, display="nope")


def test_a_new_form_is_one_function() -> None:
    @aa.register_form("wolfram-test")
    def wolfram(x: aa.Element) -> str:
        return " + ".join(f"{value} {label}" for label, value in x.coefficients.items())

    quaternions = aa.quaternions()
    assert format(2 * quaternions.i + quaternions.k, "wolfram-test") == "2 i + 1 k"
    with pytest.raises(aa.EasyError):
        aa.register_form()
    with pytest.raises(aa.EasyError):
        aa.register_form("")


def test_tables_and_reports() -> None:
    quaternions = aa.quaternions()
    assert quaternions.table().splitlines() == [
        "      1   i   j   k",
        "1     1   i   j   k",
        "i     i  -1   k  -j",
        "j     j  -k  -1   i",
        "k     k   j  -i  -1",
    ]
    assert "[-1, 0, 0, 0]" in quaternions.table("vector")
    assert quaternions.report().splitlines() == [
        "H: basis 1, i, j, k",
        "  dimension             4",
        "  commutative           False",
        "  associative           True",
        "  unit                  present",
        "  center dimension      1",
        "  nucleus dimension     4",
        "  derivation dimension  3",
    ]
    assert "derivation dimension  14" in aa.octonions().report()


def test_a_user_table_without_a_unit() -> None:
    cross = _cross()
    x, y, z = cross.basis
    assert x * y == z and y * x == -z and z * x == y and x * x == cross.zero
    assert aa.assoc(x, x, y) == y
    assert cross.unit is None and "one" not in cross.symbols()
    attempts: tuple[Callable[[], object], ...] = (
        lambda: x + 1,
        lambda: 1 - x,
        lambda: cross.scalar(2),
    )
    for attempt in attempts:
        with pytest.raises(aa.EasyError, match="declares no unit") as caught:
            attempt()
        assert caught.value.code == "unit"
    assert x != 0 and (x == 1) is False
    with pytest.raises(aa.EasyError, match="declares no conjugation"):
        x.conj()
    assert "derivation dimension  3" in cross.report()


def test_elements_of_two_algebras_never_mix() -> None:
    first, second = aa.quaternions(), aa.quaternions()
    assert first.i != second.i
    attempts: tuple[Callable[[], object], ...] = (
        lambda: first.i + second.i,
        lambda: first.i * second.i,
    )
    for attempt in attempts:
        with pytest.raises(aa.EasyError, match="cannot combine") as caught:
            attempt()
        assert caught.value.code == "parent"
    with pytest.raises(aa.EasyError):
        aa.Element(first, second.i.raw)


def test_unsupported_operands_and_arguments_fail_with_typed_errors() -> None:
    quaternions = aa.quaternions()
    i = quaternions.i
    attempts: tuple[Callable[[], object], ...] = (
        lambda: i + "j",
        lambda: "j" + i,
        lambda: i - "j",
        lambda: "j" - i,
        lambda: i * "j",
        lambda: "j" * i,
        lambda: i / "j",
        lambda: i + 1.5,
    )
    for attempt in attempts:
        with pytest.raises(TypeError):
            attempt()
    for attempt, code in (
        (lambda: i / 0, "division"),
        (lambda: i**0, "power"),
        (lambda: i ** Fraction(1, 2), "power"),
        (lambda: quaternions(1, 2), "coordinates"),
    ):
        with pytest.raises(aa.EasyError) as caught:
            attempt()
        assert caught.value.code == code


def test_table_constructor_rejects_malformed_input() -> None:
    for labels, table in (
        ((), {}),
        (("x", "x"), {}),
        (("x", ""), {}),
        (("x",), {("x", "q"): "x"}),
        (("x",), {("x",): "x"}),
        (("x",), {("x", "x"): "q"}),
        (("x",), {("x", "x"): {"q": 1}}),
        (("x",), {("x", "x"): 2}),
    ):
        with pytest.raises(aa.EasyError) as caught:
            aa.algebra(labels, table)  # type: ignore[arg-type]
        assert caught.value.code == "table"
    one_dimensional = aa.algebra(("1",), {("1", "1"): "+1"})
    assert one_dimensional.one * one_dimensional.one == 1


def test_wrap_and_over_rationals() -> None:
    source = composition.octonion_fixture()
    handle = aa.wrap(
        source, name="Oct", conjugation_signs=(1, -1, -1, -1, -1, -1, -1, -1)
    )
    assert handle.name == "Oct" and handle.e1.conj() == -handle.e1
    assert aa.over_rationals(handle.structure) is handle.structure
    assert aa.wrap(source).name == "algmul.O.v1"
    with pytest.raises(aa.EasyError) as caught:
        aa.wrap(source, conjugation_signs=(1, 2))
    assert caught.value.code == "involution"
    with pytest.raises(aa.EasyError) as caught:
        aa.over_rationals(object())  # type: ignore[arg-type]
    assert caught.value.code == "structure"
    bad = aa.algebra(("1", "a"), {("1", "1"): "1", ("a", "a"): "a"}, name="bad")
    signed = aa.wrap(bad.structure, conjugation_signs=(1, -1))
    with pytest.raises(aa.EasyError) as caught:
        signed.a.norm()
    assert caught.value.code == "norm"
    module = FreeModule(ZZ(), Basis(("p",), coefficient_domain=ZZ()))
    assert module.rank == 1
