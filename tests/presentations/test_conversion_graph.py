"""Contract tests for immutable declaration-only presentation conversions."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from anyalgebra.presentations.graph import (
    Conversion,
    ConversionDefinitionError,
    ConversionGraph,
    PresentationKind,
)
from anyalgebra.core.errors import AnyAlgebraError


class _TrackedCallable:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        self.calls += 1
        return object()


class _BrokenIterator:
    def __iter__(self) -> Iterator[str]:
        yield "first"
        raise RuntimeError("secret iterator payload")


class _HostileIterable:
    def __iter__(self) -> Iterator[object]:
        raise ValueError("secret iter payload")


class _HostileNext:
    def __iter__(self) -> Iterator[object]:
        yield "first"
        raise ValueError("secret next payload")


class _StringSubclass(str):
    pass


class _CountingNames:
    def __init__(self, count: int) -> None:
        self.count = count
        self.pulls = 0

    def __iter__(self) -> Iterator[str]:
        for index in range(self.count):
            self.pulls += 1
            yield f"g{index}"


class _KindSubclass(PresentationKind):
    pass


class _ConversionSubclass(Conversion):
    pass


class _GraphSubclass(ConversionGraph):
    pass


def _kind(id: str) -> PresentationKind:
    return PresentationKind.from_id(id)


def _conversion(
    id: str,
    source: PresentationKind,
    target: PresentationKind,
    *,
    forward: object | None = None,
    **kwargs: object,
) -> Conversion:
    return Conversion.from_callable(
        id,
        source,
        target,
        (lambda value: value) if forward is None else forward,
        **kwargs,
    )


def test_empty_single_and_persistent_declaration_order() -> None:
    source = _kind("symbolic")
    target = _kind("coordinates")
    edge = _conversion("symbolic.coordinates", source, target)
    empty = ConversionGraph.empty()
    graph = empty.with_conversion(edge)

    assert empty.edges == ()
    assert graph.edges == (edge,)
    assert graph.topology == (edge,)
    assert graph.outgoing(source) == (edge,)
    assert graph.incoming(target) == (edge,)
    assert graph.neighbors(source) == (target,)
    with pytest.raises(AttributeError):
        graph.edges.append(edge)  # type: ignore[attr-defined]
    with pytest.raises(FrozenInstanceError):
        graph._edges = ()  # type: ignore[misc]


def test_literal_kinds_parallel_cycle_self_and_inverse_declarations() -> None:
    first = _kind("same-looking")
    distinct = _kind("same-looking")
    middle = _kind("middle")
    forward = _conversion(
        "first.middle",
        first,
        middle,
        inverse=lambda value: value,
        validity_domain="finite-fixture",
        round_trip_id="first.middle.round-trip",
        guarantees=("declared",),
    )
    parallel = _conversion("first.middle.alt", first, middle, cost=0)
    cycle = _conversion("middle.first", middle, first, cost=10**9)
    self_edge = _conversion("middle.middle", middle, middle)
    graph = ConversionGraph.from_conversions((forward, parallel, cycle, self_edge))

    assert graph.outgoing(first) == (forward, parallel)
    assert graph.outgoing(distinct) == (forward, parallel)
    assert graph.neighbors(first) == (middle, middle)
    assert graph.outgoing(middle) == (cycle, self_edge)
    assert forward.inverse_direction == "target_to_source"
    assert forward.validity_domain == "finite-fixture"
    assert forward.round_trip_id == "first.middle.round-trip"
    assert forward.guarantees == ("declared",)
    assert parallel.cost == 0
    assert cycle.cost == 10**9


def test_registration_never_executes_user_callables_and_rejects_invalid_metadata() -> (
    None
):
    source = _kind("source")
    target = _kind("target")
    forward = _TrackedCallable()
    admissibility = _TrackedCallable()
    inverse = _TrackedCallable()
    edge = _conversion(
        "source.target",
        source,
        target,
        forward=forward,
        admissibility=admissibility,
        inverse=inverse,
    )
    ConversionGraph.empty().with_conversion(edge)
    assert forward.calls == admissibility.calls == inverse.calls == 0

    for cost in (True, -1, 1.5):
        with pytest.raises(ConversionDefinitionError, match="cost"):
            _conversion("bad.cost", source, target, cost=cost)
    for field, kwargs in (
        ("forward", {"forward": object()}),
        ("admissibility", {"admissibility": object()}),
        ("inverse", {"inverse": object()}),
    ):
        with pytest.raises(ConversionDefinitionError, match=field):
            _conversion("bad.callable", source, target, **kwargs)


def test_metadata_bounds_iterator_sanitation_and_duplicate_ids() -> None:
    source = _kind("source")
    target = _kind("target")
    with pytest.raises(
        ConversionDefinitionError, match="could not be iterated"
    ) as caught:
        _conversion("broken.metadata", source, target, guarantees=_BrokenIterator())
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    with pytest.raises(ConversionDefinitionError, match="maximum"):
        _conversion("long.metadata", source, target, guarantees=("g",) * 65)
    with pytest.raises(ConversionDefinitionError, match="duplicate"):
        _conversion("duplicate.metadata", source, target, guarantees=("g", "g"))
    with pytest.raises(ConversionDefinitionError, match="validity_domain"):
        _conversion("invalid.domain", source, target, validity_domain=" ")

    first = _conversion("shared.id", source, target)
    second = _conversion("shared.id", target, source)
    graph = ConversionGraph.empty().with_conversion(first)
    with pytest.raises(ConversionDefinitionError, match="duplicate stable id"):
        graph.with_conversion(second)

    def broken_conversions() -> Iterator[Conversion]:
        yield first
        raise RuntimeError("secret conversion payload")

    with pytest.raises(
        ConversionDefinitionError, match="could not be iterated"
    ) as graph_error:
        ConversionGraph.from_conversions(broken_conversions())
    assert graph_error.value.__cause__ is None
    assert graph_error.value.__context__ is None


def test_kind_stable_id_and_factory_only_executable_record_semantics() -> None:
    source = _kind("source")
    target = _kind("target")
    edge = _conversion("source.target", source, target)
    graph = ConversionGraph.empty().with_conversion(edge)
    other_edge = _conversion("source.target.other", source, target)
    other_graph = ConversionGraph.empty().with_conversion(edge)

    assert source == _kind("source")
    assert edge != other_edge
    assert graph != other_graph
    for value in (source, edge, graph):
        assert "0x" not in repr(value)
        with pytest.raises(TypeError, match="unhashable"):
            hash(value)
    with pytest.raises(ConversionDefinitionError, match="from_id"):
        PresentationKind("source")
    with pytest.raises(ConversionDefinitionError, match="from_callable"):
        Conversion("id", source, target, lambda value: value)
    with pytest.raises(ConversionDefinitionError, match="empty"):
        ConversionGraph()


def test_inverse_round_trip_metadata_and_exact_factory_dispatch() -> None:
    source = _kind("source")
    target = _kind("target")
    derived = _conversion(
        "source.target.derived", source, target, inverse=lambda value: value
    )
    explicit = _conversion(
        "source.target.explicit",
        source,
        target,
        inverse=lambda value: value,
        round_trip_id="named-round-trip",
    )
    assert derived.round_trip_id == "source.target.derived"
    assert explicit.round_trip_id == "named-round-trip"
    with pytest.raises(ConversionDefinitionError, match="round_trip_id"):
        _conversion("source.target.no-inverse", source, target, round_trip_id="bad")

    with pytest.raises(ConversionDefinitionError, match="exact PresentationKind"):
        _KindSubclass.from_id("subclass")
    with pytest.raises(ConversionDefinitionError, match="exact Conversion"):
        _ConversionSubclass.from_callable(
            "subclass.conversion", source, target, lambda value: value
        )
    with pytest.raises(ConversionDefinitionError, match="exact ConversionGraph"):
        _GraphSubclass.empty()
    with pytest.raises(ConversionDefinitionError, match="exact ConversionGraph"):
        _GraphSubclass.from_conversions(())


def test_kind_identity_validation_and_semantic_endpoint_reconstruction() -> None:
    for invalid in ("", " ", _StringSubclass("symbolic")):
        with pytest.raises(ConversionDefinitionError, match="id") as caught:
            PresentationKind.from_id(invalid)
        assert isinstance(caught.value, AnyAlgebraError)
    source = _kind("symbolic")
    target = _kind("coordinates")
    graph = ConversionGraph.empty().with_conversion(
        _conversion("symbolic.coordinates", source, target)
    )
    reconstructed_source = _kind("symbolic")
    reconstructed_target = _kind("coordinates")
    assert reconstructed_source == source
    assert graph.outgoing(reconstructed_source) == graph.edges
    assert graph.incoming(reconstructed_target) == graph.edges
    assert graph.outgoing(_kind("different")) == ()
    with pytest.raises(ConversionDefinitionError, match="source_kind"):
        _conversion("bad.source", cast(PresentationKind, []), target)
    with pytest.raises(ConversionDefinitionError, match="target_kind"):
        _conversion("bad.target", source, cast(PresentationKind, tuple))


def test_conversion_option_boundaries_and_metadata_pull_limits() -> None:
    source = _kind("source")
    target = _kind("target")
    absent = _conversion("source.target", source, target)
    present = _conversion(
        "source.target.inverse", source, target, inverse=lambda value: value
    )
    admissible = _conversion(
        "source.target.admissible", source, target, admissibility=lambda value: True
    )
    assert absent.inverse is None
    assert absent.inverse_direction is None
    assert present.inverse is not None
    assert present.inverse_direction == "target_to_source"
    assert absent.admissibility is None
    assert admissible.admissibility is not None

    exact = _CountingNames(64)
    assert _conversion(
        "exact.guarantees", source, target, guarantees=exact
    ).guarantees == tuple(f"g{index}" for index in range(64))
    assert exact.pulls == 64
    overlong = _CountingNames(65)
    with pytest.raises(ConversionDefinitionError, match="maximum"):
        _conversion("overlong.guarantees", source, target, guarantees=overlong)
    assert overlong.pulls == 65


def test_hostile_iterator_sanitation_and_graph_edge_boundaries() -> None:
    source = _kind("source")
    target = _kind("target")
    for hostile in (_HostileIterable(), _HostileNext()):
        with pytest.raises(
            ConversionDefinitionError, match="could not be iterated"
        ) as caught:
            _conversion("hostile.metadata", source, target, guarantees=hostile)
        assert "secret" not in str(caught.value)
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None

    def edges(count: int) -> Iterator[Conversion]:
        for index in range(count):
            yield _conversion(f"edge.{index}", source, target)

    graph = ConversionGraph.from_conversions(edges(512))
    assert len(graph.edges) == 512
    with pytest.raises(ConversionDefinitionError, match="maximum"):
        ConversionGraph.from_conversions(edges(513))

    def hostile_edges() -> Iterator[Conversion]:
        yield _conversion("valid.first", source, target)
        raise ValueError("secret graph payload")

    with pytest.raises(
        ConversionDefinitionError, match="could not be iterated"
    ) as caught:
        ConversionGraph.from_conversions(hostile_edges())
    assert "secret" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
