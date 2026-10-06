"""Contract tests for immutable, deterministic coercion graph topology."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from anyalgebra.core.coercions import CoercionGraph, CoercionMap
from anyalgebra.core.domains import DomainElement


@dataclass(frozen=True, slots=True, eq=False)
class FixtureDomain:
    """Small identity-sensitive domain fixture for graph-only tests."""

    label: str

    def normalize(self, value: object) -> object:
        """Accept the fixture payload without defining arithmetic."""
        return value

    def element(self, value: object) -> FixtureElement:
        """Construct a parent-owned fixture element."""
        return FixtureElement(parent=self, value=value)


@dataclass(frozen=True, slots=True)
class FixtureElement:
    """Minimal parent-aware fixture scalar."""

    parent: FixtureDomain
    value: object


def _forward(value: DomainElement[object]) -> DomainElement[object]:
    """Provide an unused, typed graph-edge implementation."""
    return value


def _map(
    identifier: str,
    source: FixtureDomain,
    target: FixtureDomain,
    *,
    cost: int = 1,
) -> CoercionMap[object, object]:
    """Build one exact lossless fixture edge with declared stable metadata."""
    return CoercionMap(
        id=identifier,
        source=source,
        target=target,
        forward=_forward,
        injective=True,
        exact=True,
        lossless=True,
        cost=cost,
    )


def _path_ids(
    paths: tuple[tuple[CoercionMap[object, object], ...], ...],
) -> tuple[tuple[str, ...], ...]:
    """Project graph paths to their stable identifiers for concise assertions."""
    return tuple(tuple(edge.stable_id for edge in path) for path in paths)


def test_empty_and_single_edge_topology_are_immutable_and_parent_sensitive() -> None:
    """The empty graph is usable, and registration returns a new frozen graph."""
    source = FixtureDomain("source")
    target = FixtureDomain("target")
    edge = _map("fixture.source_to_target", source, target)
    empty = CoercionGraph()

    assert empty.topology == ()
    assert empty.paths(source, target) == ()

    graph = empty.with_map(edge)

    assert empty.topology == ()
    assert graph.topology == (edge,)
    assert _path_ids(graph.paths(source, target)) == (("fixture.source_to_target",),)
    with pytest.raises(AttributeError):
        graph.topology.append(edge)  # type: ignore[attr-defined]

    same_label_distinct_parent = FixtureDomain("source")
    assert graph.paths(same_label_distinct_parent, target) == ()


def test_duplicate_registration_rejects_conflicting_stable_id_metadata() -> None:
    """Stable IDs bind immutable semantic metadata, not callable identity."""
    source = FixtureDomain("source")
    target = FixtureDomain("target")
    edge = _map("fixture.edge", source, target)
    graph = CoercionGraph().with_map(edge)

    assert graph.with_map(_map("fixture.edge", source, target)) is graph

    with pytest.raises(ValueError, match=r"fixture\.edge"):
        graph.with_map(_map("fixture.edge", source, target, cost=2))

    equal_but_distinct_source = FixtureDomain("source")
    with pytest.raises(ValueError, match=r"fixture\.edge"):
        graph.with_map(_map("fixture.edge", equal_but_distinct_source, target))


def test_topology_and_paths_do_not_depend_on_registration_order() -> None:
    """Stable edge IDs, not incidental insertion order, determine graph output."""
    source = FixtureDomain("source")
    left = FixtureDomain("left")
    right = FixtureDomain("right")
    target = FixtureDomain("target")
    edges = (
        _map("left.join", left, target),
        _map("source.left", source, left),
        _map("right.join", right, target),
        _map("source.right", source, right),
    )
    forward = CoercionGraph()
    reverse = CoercionGraph()
    for edge in edges:
        forward = forward.with_map(edge)
    for edge in reversed(edges):
        reverse = reverse.with_map(edge)

    assert tuple(edge.stable_id for edge in forward.topology) == (
        "left.join",
        "right.join",
        "source.left",
        "source.right",
    )
    assert forward.topology == reverse.topology
    assert _path_ids(forward.paths(source, target)) == (
        ("source.left", "left.join"),
        ("source.right", "right.join"),
    )
    assert _path_ids(forward.paths(source, target)) == _path_ids(
        reverse.paths(source, target)
    )


def test_parallel_edges_and_diamond_paths_are_all_sorted_by_edge_id_sequences() -> None:
    """Parallel edges remain distinct, while a diamond exposes both full routes."""
    source = FixtureDomain("source")
    left = FixtureDomain("left")
    right = FixtureDomain("right")
    target = FixtureDomain("target")
    graph = CoercionGraph()
    for edge in (
        _map("z.parallel", source, target),
        _map("a.parallel", source, target),
        _map("left.join", left, target),
        _map("source.left", source, left),
        _map("right.join", right, target),
        _map("source.right", source, right),
    ):
        graph = graph.with_map(edge)

    assert _path_ids(graph.paths(source, target)) == (
        ("a.parallel",),
        ("source.left", "left.join"),
        ("source.right", "right.join"),
        ("z.parallel",),
    )


def test_cycle_enumeration_is_simple_and_bounded_by_number_of_edges() -> None:
    """Cycles never cause repeated-parent expansion or nonterminating searches."""
    source = FixtureDomain("source")
    middle = FixtureDomain("middle")
    cycle = FixtureDomain("cycle")
    target = FixtureDomain("target")
    graph = CoercionGraph()
    for edge in (
        _map("source.middle", source, middle),
        _map("middle.cycle", middle, cycle),
        _map("cycle.source", cycle, source),
        _map("middle.target", middle, target),
    ):
        graph = graph.with_map(edge)

    assert _path_ids(graph.paths(source, target)) == (
        ("source.middle", "middle.target"),
    )
    assert graph.paths(source, target, max_steps=1) == ()
    assert _path_ids(graph.paths(source, target, max_steps=2)) == (
        ("source.middle", "middle.target"),
    )


def test_path_bounds_identity_and_invalid_endpoints_are_explicit() -> None:
    """``max_steps`` is an exact nonnegative built-in edge-count bound."""
    source = FixtureDomain("source")
    target = FixtureDomain("target")
    graph = CoercionGraph().with_map(_map("source.target", source, target))

    assert _path_ids(graph.paths(source, target, max_steps=1)) == (("source.target",),)
    assert graph.paths(source, target, max_steps=0) == ()
    assert graph.paths(source, source, max_steps=0) == ((),)
    assert graph.paths(source, source, max_steps=7) == ((),)
    assert graph.paths(target, source, max_steps=3) == ()

    for invalid_bound in (-1, True, 1.5):
        with pytest.raises((TypeError, ValueError), match="max_steps"):
            graph.paths(source, target, max_steps=invalid_bound)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="source"):
        graph.paths(object(), target)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="target"):
        graph.paths(source, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="CoercionMap"):
        CoercionGraph().with_map(object())  # type: ignore[arg-type]
