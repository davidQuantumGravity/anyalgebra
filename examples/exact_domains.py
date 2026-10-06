"""Exercise the bounded v0.0 exact-domain and coercion infrastructure.

This example reports software behavior only.  It does not verify a research
claim or a mathematical statement beyond the displayed exact constructions.
"""

from __future__ import annotations

import json

from anyalgebra.core.coercions import (
    CoercionGraph,
    CoercionMap,
    coerce,
    common_parent,
)
from anyalgebra.core.domains import DomainElement, QQ, ZZ
from anyalgebra.core.errors import LossyCoercionError
from anyalgebra.core.rational import Rational


def _zz_to_qq(value: DomainElement[int]) -> DomainElement[Rational]:
    """Embed an exact integer into the canonical rational parent."""
    return QQ().element(value.value)


def _rational_pair(value: object) -> list[int]:
    """Return one exact rational payload as JSON-compatible coordinates."""
    if not isinstance(value, Rational):
        raise TypeError("expected an exact Rational payload")
    return [value.numerator, value.denominator]


def main() -> int:
    """Run exact construction, planning, application, and loss diagnostics."""
    lossy_calls: list[tuple[int, int]] = []

    def qq_to_zz_lossy(value: DomainElement[Rational]) -> DomainElement[int]:
        """Provide an explicit projection that implicit coercion must not call."""
        lossy_calls.append((value.value.numerator, value.value.denominator))
        return ZZ().element(value.value.numerator // value.value.denominator)

    zz_to_qq = CoercionMap(
        id="core.zz_to_qq.v1",
        source=ZZ(),
        target=QQ(),
        forward=_zz_to_qq,
        injective=True,
        exact=True,
        lossless=True,
    )
    qq_to_zz = CoercionMap(
        id="example.qq_to_zz.lossy.v1",
        source=QQ(),
        target=ZZ(),
        forward=qq_to_zz_lossy,
        injective=False,
        exact=True,
        lossless=False,
    )
    graph = CoercionGraph().with_map(zz_to_qq).with_map(qq_to_zz)

    integer = ZZ().element(6)
    rational = QQ().element((4, 6))
    identity_integer = coerce(integer, ZZ(), graph=graph)
    identity_rational = coerce(rational, QQ(), graph=graph)
    embedded = coerce(integer, QQ(), graph=graph)

    plan = common_parent((integer, rational), graph=graph)
    common_values = plan.apply((integer, rational))

    prohibited_loss: dict[str, object]
    try:
        coerce(rational, ZZ(), graph=graph)
    except LossyCoercionError as error:
        prohibited_loss = {
            "callable_executed": bool(lossy_calls),
            "error_type": type(error).__name__,
            "map_id_reported": qq_to_zz.stable_id in error.reason,
        }
    else:  # pragma: no cover - the example test requires this branch to be impossible
        raise AssertionError("a lossy coercion was applied implicitly")

    payload = {
        "canonical_parents": {"QQ": QQ() is QQ(), "ZZ": ZZ() is ZZ()},
        "canonical_values": {
            "QQ": _rational_pair(rational.value),
            "ZZ": integer.value,
        },
        "common_parent": {
            "route_ids": [list(route) for route in plan.route_ids],
            "target": "QQ" if plan.target is QQ() else "unexpected",
            "values": [_rational_pair(value.value) for value in common_values],
        },
        "exact_embedding": {
            "parent": "QQ" if embedded.parent is QQ() else "unexpected",
            "route_ids": list(graph.plan(ZZ(), QQ()).route_ids[0]),
            "value": _rational_pair(embedded.value),
        },
        "identity": {
            "QQ_same_object": identity_rational is rational,
            "ZZ_same_object": identity_integer is integer,
        },
        "prohibited_loss": prohibited_loss,
        "scope": "infrastructure example only; no research claim verified",
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
