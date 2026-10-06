"""Contract tests for one immutable, typed coercion edge."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from anyalgebra.core.coercions import CoercionMap
from anyalgebra.core.domains import Domain, DomainElement, QQ, ZZ
from anyalgebra.core.errors import CoercionError
from anyalgebra.core.rational import Rational


def _zz_to_qq(value: DomainElement[int]) -> DomainElement[Rational]:
    """Embed one normalized ZZ element as the corresponding QQ element."""
    return QQ().element(value.value)


def _same_zz_to_qq(value: DomainElement[int]) -> DomainElement[Rational]:
    """A separately allocated callable with the same declared map meaning."""
    return QQ().element(value.value)


def _zz_to_zz(value: DomainElement[int]) -> DomainElement[int]:
    """Deliberately violate the declared QQ target for a boundary test."""
    return ZZ().element(value.value)


def _exact_embedding(
    forward: Callable[[DomainElement[int]], DomainElement[Rational]] = _zz_to_qq,
) -> CoercionMap[int, Rational]:
    """Build the canonical exact ZZ-to-QQ edge for the map-only slice."""
    return CoercionMap(
        id="core.zz_to_qq.v1",
        source=ZZ(),
        target=QQ(),
        forward=forward,
        injective=True,
        surjective=False,
        exact=True,
        lossless=True,
        cost=1,
        assumptions=("native exact integers",),
    )


def test_exact_zz_to_qq_map_is_immutable_and_applies_directly() -> None:
    """A declared exact embedding owns stable metadata and preserves parents."""
    coercion = _exact_embedding()

    assert coercion.id == "core.zz_to_qq.v1"
    assert coercion.stable_id == coercion.id
    assert coercion.source is ZZ()
    assert coercion.target is QQ()
    assert coercion.injective is True
    assert coercion.surjective is False
    assert coercion.exact is True
    assert coercion.lossless is True
    assert coercion.cost == 1
    assert coercion.assumptions == ("native exact integers",)

    result = coercion.apply(ZZ().element(7))
    assert result.parent is QQ()
    assert result.value.numerator == 7
    assert result.value.denominator == 1

    with pytest.raises(FrozenInstanceError):
        coercion.cost = 2  # type: ignore[misc]


def test_callable_object_identity_is_not_map_semantic_identity() -> None:
    """Distinct function objects do not alter the declared map identity or hash."""
    first = _exact_embedding(_zz_to_qq)
    second = _exact_embedding(_same_zz_to_qq)

    assert first.forward is not second.forward
    assert first == second
    assert hash(first) == hash(second)
    assert first.stable_id == second.stable_id


@pytest.mark.parametrize(
    ("changes", "expected_message"),
    [
        ({"id": ""}, "id"),
        ({"id": " core.zz_to_qq.v1 "}, "id"),
        ({"source": object()}, "source"),
        ({"target": object()}, "target"),
        ({"forward": None}, "forward"),
        ({"injective": 1}, "injective"),
        ({"surjective": 1}, "surjective"),
        ({"exact": 1}, "exact"),
        ({"lossless": 1}, "lossless"),
        ({"injective": True, "exact": False, "lossless": True}, "lossless"),
        ({"injective": False, "exact": True, "lossless": True}, "lossless"),
        ({"cost": -1}, "cost"),
        ({"cost": True}, "cost"),
        ({"cost": 0.5}, "cost"),
        ({"assumptions": ("",)}, "assumptions"),
        ({"assumptions": ["mutable"]}, "assumptions"),
        ({"assumptions": {"unordered"}}, "assumptions"),
        ({"assumptions": {"mapping": "is unordered"}}, "assumptions"),
    ],
)
def test_constructor_rejects_malformed_semantic_metadata(
    changes: dict[str, object], expected_message: str
) -> None:
    """Invalid declarations cannot form a coercion edge for later graph work."""
    arguments: dict[str, object] = {
        "id": "core.zz_to_qq.v1",
        "source": ZZ(),
        "target": QQ(),
        "forward": _zz_to_qq,
        "injective": True,
        "surjective": False,
        "exact": True,
        "lossless": True,
        "cost": 1,
        "assumptions": (),
    }
    arguments.update(changes)

    with pytest.raises((TypeError, ValueError), match=expected_message):
        CoercionMap(
            id=cast(str, arguments["id"]),
            source=cast(Domain[int], arguments["source"]),
            target=cast(Domain[Rational], arguments["target"]),
            forward=cast(
                Callable[[DomainElement[int]], DomainElement[Rational]],
                arguments["forward"],
            ),
            injective=cast(bool, arguments["injective"]),
            surjective=cast(bool, arguments["surjective"]),
            exact=cast(bool, arguments["exact"]),
            lossless=cast(bool, arguments["lossless"]),
            cost=cast(int, arguments["cost"]),
            assumptions=cast(tuple[str, ...], arguments["assumptions"]),
        )


def test_direct_application_rejects_wrong_source_or_wrong_target_result() -> None:
    """The local edge boundary detects parent mistakes before graph work exists."""
    coercion = _exact_embedding()

    with pytest.raises(CoercionError, match="source"):
        coercion.apply(cast(DomainElement[int], QQ().element(3)))

    wrong_target = CoercionMap(
        id="test.bad_target.v1",
        source=ZZ(),
        target=QQ(),
        forward=cast(
            Callable[[DomainElement[int]], DomainElement[Rational]], _zz_to_zz
        ),
        injective=True,
        surjective=False,
        exact=True,
        lossless=True,
        cost=0,
    )
    with pytest.raises(CoercionError, match="target"):
        wrong_target.apply(ZZ().element(3))
