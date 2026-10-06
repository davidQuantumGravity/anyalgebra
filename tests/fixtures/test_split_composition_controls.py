"""Exact split-composition isotropy, zero-divisor, and idempotent controls."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from anyalgebra.algebra.multilinear import (
    FiniteMultilinearStructure,
    StructureConstants,
    evaluate_multilinear,
)
from anyalgebra.core.domains import QQ
from anyalgebra.core.elements import SparseElement
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.core.rational import Rational
from anyalgebra.fixtures.composition import (
    octonion_fixture,
    quaternion_fixture,
    split_octonion_fixture,
    split_quaternion_fixture,
)
from anyalgebra.fixtures.composition_ops import composition_operations


_SPLIT_CONTROLS = (
    (split_quaternion_fixture, 2, "algmul.Hs.v1"),
    (split_octonion_fixture, 4, "algmul.Os.v1"),
)


def _qq_lift(value: FiniteMultilinearStructure) -> FiniteMultilinearStructure:
    domain = QQ()
    module = FreeModule(
        domain, Basis(value.module.basis.labels, coefficient_domain=domain)
    )
    constants = StructureConstants.from_sparse(
        (module.basis, module.basis),
        module.basis,
        tuple(
            (key, domain.element(coefficient.value))
            for key, coefficient in value.operation.constants.entries
        ),
    )
    return FiniteMultilinearStructure.from_structure_constants(module, constants)


def _conjugate(value: SparseElement) -> SparseElement:
    return value.parent.element(
        {
            index: (
                coefficient.value
                if index == 0
                else cast(Rational, coefficient.value).negate()
            )
            for index, coefficient in value.coordinates().items()
        }
    )


def _norm(algebra: FiniteMultilinearStructure, value: SparseElement) -> object:
    product_value = evaluate_multilinear(algebra, value, _conjugate(value))
    assert all(index == 0 for index in product_value.coordinates())
    coefficient = product_value.coordinates().get(0)
    return QQ().element(0).value if coefficient is None else coefficient.value


@pytest.mark.parametrize(("factory", "generator", "identifier"), _SPLIT_CONTROLS)
def test_integer_isotropic_zero_divisor_pair_is_exact(
    factory: object, generator: int, identifier: str
) -> None:
    algebra = factory()  # type: ignore[operator]
    operations = composition_operations(algebra)
    one = algebra.module.element({0: 1})
    plus = algebra.module.element({0: 1, generator: 1})
    minus = algebra.module.element({0: 1, generator: -1})

    assert algebra.name == identifier
    assert plus != algebra.module.zero() and minus != algebra.module.zero()
    assert operations.quadratic_norm(plus).value == 0
    assert operations.quadratic_norm(minus).value == 0
    assert operations.multiply(plus, minus) == algebra.module.zero()
    assert operations.multiply(minus, plus) == algebra.module.zero()
    assert plus.add(minus) == one.scale(2)


@pytest.mark.parametrize(("factory", "generator", "identifier"), _SPLIT_CONTROLS)
def test_rational_complementary_idempotents_are_nontrivial_and_orthogonal(
    factory: object, generator: int, identifier: str
) -> None:
    algebra = _qq_lift(factory())  # type: ignore[operator]
    half = (1, 2)
    p = algebra.module.element({0: half, generator: half})
    q = algebra.module.element({0: half, generator: (-1, 2)})
    one = algebra.module.element({0: 1})
    zero = algebra.module.zero()

    assert identifier.startswith("algmul.")
    assert p not in (zero, one) and q not in (zero, one)
    assert evaluate_multilinear(algebra, p, p) == p
    assert evaluate_multilinear(algebra, q, q) == q
    assert evaluate_multilinear(algebra, p, q) == zero
    assert evaluate_multilinear(algebra, q, p) == zero
    assert p.add(q) == one
    assert _norm(algebra, p) == QQ().element(0).value
    assert _norm(algebra, q) == QQ().element(0).value


def _inverse_requires_nonzero_norm(
    algebra: FiniteMultilinearStructure, value: SparseElement
) -> SparseElement:
    if _norm(algebra, value) == QQ().element(0).value:
        raise ZeroDivisionError("split composition inverse requires nonzero norm")
    raise NotImplementedError("test helper only exercises the zero-norm boundary")


@pytest.mark.parametrize(("factory", "generator", "_identifier"), _SPLIT_CONTROLS)
def test_zero_norm_explicitly_blocks_division(
    factory: object, generator: int, _identifier: str
) -> None:
    algebra = _qq_lift(factory())  # type: ignore[operator]
    isotropic = algebra.module.element({0: 1, generator: 1})
    with pytest.raises(ZeroDivisionError, match="nonzero norm"):
        _inverse_requires_nonzero_norm(algebra, isotropic)


@pytest.mark.parametrize(
    ("factory", "generator", "identifier"),
    (
        (quaternion_fixture, 1, "algmul.H.v1"),
        (octonion_fixture, 1, "algmul.O.v1"),
    ),
)
def test_split_witness_pattern_does_not_generalize_to_ordinary_forms(
    factory: object, generator: int, identifier: str
) -> None:
    algebra = factory()  # type: ignore[operator]
    operations = composition_operations(algebra)
    plus = algebra.module.element({0: 1, generator: 1})
    minus = algebra.module.element({0: 1, generator: -1})

    assert algebra.name == identifier
    assert operations.quadratic_norm(plus).value == 2
    assert operations.multiply(plus, minus) == algebra.module.element({0: 2})


def test_documented_witness_code_is_executable_and_exact() -> None:
    guide = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "guides"
        / "split-composition-controls.md"
    ).read_text(encoding="utf-8")
    marker = "```python\n"
    code = guide.split(marker, 1)[1].split("\n```", 1)[0]
    namespace: dict[str, object] = {}
    exec(compile(code, "split-composition-controls.md", "exec"), namespace)

    assert namespace["report"] == {
        "algmul.Hs.v1": {
            "generator": "j",
            "norms": [0, 0],
            "products_are_zero": [True, True],
        },
        "algmul.Os.v1": {
            "generator": "e4",
            "norms": [0, 0],
            "products_are_zero": [True, True],
        },
    }
