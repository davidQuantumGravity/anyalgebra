"""Contract tests for exact, identity-sensitive common-parent planning."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass

import pytest

from anyalgebra.core.coercions import (
    CoercionGraph,
    CoercionMap,
    CoercionPlan,
    common_parent,
)
from anyalgebra.core.domains import DomainElement, QQ, ZZ
from anyalgebra.core.errors import (
    CoercionAmbiguityError,
    CoercionError,
    LossyCoercionError,
)
from anyalgebra.core.rational import Rational


@dataclass(frozen=True, slots=True, eq=False)
class FixtureDomain:
    """Small identity-sensitive parent fixture with no arithmetic claims."""

    label: str

    def normalize(self, value: object) -> object:
        """Accept a fixture value without changing its identity."""
        return value

    def element(self, value: object) -> FixtureElement:
        """Construct one parent-aware fixture element."""
        return FixtureElement(parent=self, value=value)


@dataclass(frozen=True, slots=True)
class FixtureElement:
    """Minimal immutable scalar fixture."""

    parent: FixtureDomain
    value: object


def _forward(value: DomainElement[object]) -> DomainElement[object]:
    """Return the input only where source/target identity is irrelevant to planning."""
    return value


def _edge(
    identifier: str,
    source: FixtureDomain,
    target: FixtureDomain,
    *,
    lossless: bool = True,
    cost: int = 1,
) -> CoercionMap[object, object]:
    """Construct a declared fixture edge with distinct lossless/lossy states."""
    return CoercionMap(
        id=identifier,
        source=source,
        target=target,
        forward=_forward,
        injective=lossless,
        exact=lossless,
        lossless=lossless,
        cost=cost,
    )


def _graph(*edges: CoercionMap[object, object]) -> CoercionGraph:
    """Register edges in deliberately supplied order for ordering assertions."""
    graph = CoercionGraph()
    for edge in edges:
        graph = graph.with_map(edge)
    return graph


def _route_ids(plan: CoercionPlan) -> tuple[tuple[str, ...], ...]:
    """Project plan routes to their portable map identifiers."""
    return plan.route_ids


def _zz_to_qq(value: DomainElement[int]) -> DomainElement[Rational]:
    """Embed a canonical integer as a canonical rational for the public control."""
    return QQ().element(value.value)


def _zz_qq_graph() -> CoercionGraph:
    """Return the one declared automatic exact scalar embedding."""
    return CoercionGraph().with_map(
        CoercionMap(
            id="core.zz_to_qq.v1",
            source=ZZ(),
            target=QQ(),
            forward=_zz_to_qq,
            injective=True,
            exact=True,
            lossless=True,
            cost=7,
        )
    )


def test_zz_and_qq_have_one_exact_order_independent_common_parent_plan() -> None:
    """The declared ``ZZ -> QQ`` embedding supplies the one automatic plan."""
    graph = _zz_qq_graph()

    forward = common_parent((ZZ().element(2), QQ().element((3, 5))), graph=graph)
    reverse = common_parent((QQ().element((3, 5)), ZZ().element(2)), graph=graph)

    assert forward.target is QQ()
    assert reverse.target is QQ()
    assert _route_ids(forward) == (("core.zz_to_qq.v1",), ())
    assert _route_ids(reverse) == ((), ("core.zz_to_qq.v1",))
    assert forward.route_costs == (7, 0)
    assert reverse.route_costs == (0, 7)
    assert forward.total_cost == reverse.total_cost == 7
    assert forward.lossless is True


def test_identity_is_lossless_for_single_duplicate_and_all_same_parent_inputs() -> None:
    """Same-parent inputs retain their literal parent despite downstream nodes."""
    parent = FixtureDomain("A")
    downstream = FixtureDomain("B")
    graph = _graph(
        _edge("b.to.a", downstream, parent, cost=1),
        _edge("a.to.b", parent, downstream, cost=99),
    )
    value = parent.element("x")

    for values in ((value,), (value, value), (value, parent.element("y"))):
        plan = common_parent(values, graph=graph)
        assert plan.target is parent
        assert plan.routes == tuple(() for _ in values)
        assert plan.route_costs == tuple(0 for _ in values)
        assert plan.total_cost == 0


def test_least_common_parent_excludes_downstream_reachable_candidates() -> None:
    """A chain ``A -> C -> D`` chooses ``C`` rather than treating ``D`` as tied."""
    source = FixtureDomain("A")
    target = FixtureDomain("C")
    downstream = FixtureDomain("D")
    graph = _graph(
        _edge("c.to.d", target, downstream, cost=1),
        _edge("a.to.c", source, target, cost=11),
    )

    plan = common_parent((source.element(1), target.element(2)), graph=graph)

    assert plan.target is target
    assert _route_ids(plan) == (("a.to.c",), ())
    assert plan.route_costs == (11, 0)
    assert plan.total_cost == 11


def test_parallel_and_diamond_lossless_routes_are_ambiguity_not_cost_selection() -> (
    None
):
    """Multiple complete routes to one selected target remain a typed ambiguity."""
    source = FixtureDomain("A")
    left = FixtureDomain("L")
    right = FixtureDomain("R")
    target = FixtureDomain("D")
    graph = _graph(
        _edge("right.join", right, target, cost=1),
        _edge("source.right", source, right, cost=1),
        _edge("left.join", left, target, cost=10),
        _edge("source.left", source, left, cost=10),
    )

    with pytest.raises(CoercionAmbiguityError) as diamond:
        common_parent((source.element(1), target.element(2)), graph=graph)
    assert "left.join" in diamond.value.reason
    assert "right.join" in diamond.value.reason
    assert "source.left" in diamond.value.reason
    assert "source.right" in diamond.value.reason

    parallel = _graph(
        _edge("z.parallel", source, target, cost=0),
        _edge("a.parallel", source, target, cost=100),
    )
    with pytest.raises(CoercionAmbiguityError, match=r"a\.parallel.*z\.parallel"):
        common_parent((source.element(1), target.element(2)), graph=parallel)


def test_incomparable_least_common_parents_are_ambiguity_with_stable_witnesses() -> (
    None
):
    """Two incomparable least candidates cannot be selected by edge name or cost."""
    left_source = FixtureDomain("left-source")
    right_source = FixtureDomain("right-source")
    left_target = FixtureDomain("left-target")
    right_target = FixtureDomain("right-target")
    graph = _graph(
        _edge("right.to.right_target", right_source, right_target, cost=1),
        _edge("left.to.right_target", left_source, right_target, cost=50),
        _edge("right.to.left_target", right_source, left_target, cost=100),
        _edge("left.to.left_target", left_source, left_target, cost=2),
    )

    with pytest.raises(CoercionAmbiguityError) as result:
        common_parent((left_source.element(1), right_source.element(2)), graph=graph)
    assert "left.to.left_target" in result.value.reason
    assert "right.to.left_target" in result.value.reason
    assert "left.to.right_target" in result.value.reason
    assert "right.to.right_target" in result.value.reason


def test_lossy_edges_are_ignored_and_lossy_only_failure_is_distinct() -> None:
    """Lossless candidates win; a lossy-only reverse map cannot plan implicitly."""
    source = FixtureDomain("A")
    shared = FixtureDomain("C")
    lossy = FixtureDomain("D")
    graph = _graph(
        _edge("c.to.d.lossy", shared, lossy, lossless=False),
        _edge("a.to.c", source, shared),
    )

    plan = common_parent((source.element(1), shared.element(2)), graph=graph)
    assert plan.target is shared
    assert _route_ids(plan) == (("a.to.c",), ())

    reverse_only = _graph(_edge("qq.to.zz.lossy", shared, source, lossless=False))
    with pytest.raises(LossyCoercionError, match=r"qq\.to\.zz\.lossy"):
        common_parent((shared.element(1), source.element(2)), graph=reverse_only)


def test_unreachable_empty_and_invalid_inputs_fail_without_a_selected_parent() -> None:
    """No-route, no-input, and malformed-input diagnostics remain distinct."""
    left = FixtureDomain("left")
    right = FixtureDomain("right")

    with pytest.raises(CoercionError, match="no declared common parent"):
        common_parent((left.element(1), right.element(2)), graph=CoercionGraph())
    with pytest.raises(ValueError, match="at least one"):
        common_parent((), graph=CoercionGraph())
    with pytest.raises(TypeError, match="DomainElement"):
        common_parent((object(),), graph=CoercionGraph())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="CoercionGraph"):
        common_parent((left.element(1),), graph=object())  # type: ignore[arg-type]


def test_cycles_literal_parent_identity_and_immutable_plan_routes_are_preserved() -> (
    None
):
    """Cycle traversal is finite, equal labels do not merge, and plans are frozen."""
    source = FixtureDomain("A")
    middle = FixtureDomain("B")
    target = FixtureDomain("C")
    equal_label_other_parent = FixtureDomain("A")
    graph = _graph(
        _edge("middle.source", middle, source),
        _edge("source.middle", source, middle, cost=3),
        _edge("middle.target", middle, target, cost=4),
    )

    plan = common_parent((source.element(1), target.element(2)), graph=graph)
    assert plan.target is target
    assert _route_ids(plan) == (("source.middle", "middle.target"), ())
    assert plan.route_costs == (7, 0)
    assert plan.total_cost == 7
    with pytest.raises(CoercionError, match="no declared common parent"):
        common_parent(
            (equal_label_other_parent.element(1), target.element(2)), graph=graph
        )
    with pytest.raises(FrozenInstanceError):
        plan.target = source  # type: ignore[misc]
    with pytest.raises(AttributeError):
        plan.routes[0].append(graph.topology[0])  # type: ignore[attr-defined]


def test_mutually_reachable_candidates_remain_an_ambiguity() -> None:
    """A lossless strongly connected pair has no silent preferred representative."""
    left = FixtureDomain("A")
    right = FixtureDomain("B")
    graph = _graph(
        _edge("right.to.left", right, left, cost=1),
        _edge("left.to.right", left, right, cost=99),
    )

    with pytest.raises(CoercionAmbiguityError) as result:
        common_parent((left.element(1), right.element(2)), graph=graph)

    assert "left.to.right" in result.value.reason
    assert "right.to.left" in result.value.reason
