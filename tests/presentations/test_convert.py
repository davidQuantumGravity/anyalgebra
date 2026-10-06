"""Tests for explicit, non-speculative presentation conversion routing."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import FrozenInstanceError
import pytest

from anyalgebra.presentations.convert import (
    ConversionAmbiguityError,
    ConversionApplicationError,
    ConversionError,
    ConversionNotAdmissibleError,
    ConversionPlan,
    ConversionRouteError,
    ConversionSearchLimitError,
    ConvertedValue,
    RouteStep,
    convert,
    plan,
    plan_conversion,
)
from anyalgebra.presentations.graph import Conversion, ConversionGraph, PresentationKind


def _kind(name: str) -> PresentationKind:
    return PresentationKind.from_id(name)


def _edge(
    name: str, source: PresentationKind, target: PresentationKind, **kwargs: object
) -> Conversion:
    return Conversion.from_callable(name, source, target, lambda value: value, **kwargs)


def _graph(*edges: Conversion) -> ConversionGraph:
    return ConversionGraph.from_conversions(edges)


def test_identity_direct_and_multistep_application_are_explicit() -> None:
    first = _kind("first")
    second = _kind("second")
    third = _kind("third")
    calls = {"identity": 0, "first": 0, "second": 0}

    def first_transform(value: int) -> int:
        calls["first"] += 1
        return value + 1

    def second_transform(value: int) -> int:
        calls["second"] += 1
        return value * 2

    direct = Conversion.from_callable("first.second", first, second, first_transform)
    second_edge = Conversion.from_callable(
        "second.third",
        second,
        third,
        second_transform,
        guarantees=("second-step",),
    )
    graph = _graph(direct, second_edge)
    identity = convert(3, first, graph=graph, source_kind=first)
    assert identity.value == 3
    assert identity.plan.steps == ()
    assert calls == {"identity": 0, "first": 0, "second": 0}
    direct_value = convert(3, second, graph=graph, source_kind=first)
    assert direct_value.value == 4
    multi = convert(3, third, graph=graph, source_kind=first)
    assert multi.value == 8
    assert multi.route_witness == (
        ("first.second", "forward"),
        ("second.third", "forward"),
    )
    assert multi.plan.guarantees == ("second-step",)
    assert multi.plan.cost == 2
    assert multi.cost == 2
    assert multi.guarantees == ("second-step",)
    assert multi.validity_domains == ("all", "all")


def test_ambiguity_cost_and_explicit_disambiguation_never_call_candidates() -> None:
    source = _kind("source")
    target = _kind("target")
    calls = {"low": 0, "high": 0}

    def low_transform(value: object) -> object:
        calls["low"] += 1
        return value

    def high_transform(value: object) -> object:
        calls["high"] += 1
        return value

    low = Conversion.from_callable(
        "low",
        source,
        target,
        low_transform,
        cost=0,
    )
    high = Conversion.from_callable(
        "high",
        source,
        target,
        high_transform,
        cost=99,
    )
    graph = _graph(low, high)
    with pytest.raises(ConversionAmbiguityError) as caught:
        plan_conversion(source, target, graph=graph)
    assert caught.value.candidates == ((("low", "forward"),), (("high", "forward"),))
    assert calls == {"low": 0, "high": 0}
    selected = convert(
        source,
        target,
        graph=graph,
        route=(RouteStep.from_conversion(high, "forward"),),
        source_kind=source,
    )
    assert selected.value is source
    assert calls == {"low": 0, "high": 1}


def test_route_validation_inverse_and_round_trip_metadata() -> None:
    source = _kind("source")
    target = _kind("target")
    edge = Conversion.from_callable(
        "source.target",
        source,
        target,
        lambda value: f"F{value}",
        inverse=lambda value: value[1:],
        cost=7,
        guarantees=("forward-declared", "inverse-declared"),
        validity_domain="fixture",
        round_trip_id="source-target-round-trip",
    )
    graph = _graph(edge)
    forward = convert("x", target, graph=graph, source_kind=source)
    backward = convert(
        forward.value,
        source,
        graph=graph,
        source_kind=target,
        route=(RouteStep.from_conversion(edge, "inverse"),),
    )
    assert backward.value == "x"
    assert backward.source_kind == target
    assert backward.target_kind == source
    assert backward.route_witness == (("source.target", "inverse"),)
    assert backward.cost == 7
    assert backward.guarantees == ("forward-declared", "inverse-declared")
    assert forward.plan.round_trip_ids == ("source-target-round-trip",)
    assert backward.plan.validity_domains == ("fixture",)
    assert backward.round_trip_ids == ("source-target-round-trip",)
    missing = _edge("missing.inverse", source, target)
    with pytest.raises(ConversionRouteError, match="not declared"):
        RouteStep.from_conversion(missing, "inverse")
    detached = _edge("detached", source, target)
    with pytest.raises(ConversionRouteError, match="unregistered"):
        plan_conversion(
            source,
            target,
            graph=graph,
            route=(RouteStep.from_conversion(detached, "forward"),),
        )


def test_admissibility_and_application_failures_are_typed_and_safe() -> None:
    source = _kind("source")
    target = _kind("target")
    false_edge = Conversion.from_callable(
        "false", source, target, lambda value: value, admissibility=lambda value: False
    )
    with pytest.raises(ConversionNotAdmissibleError):
        convert(1, target, graph=_graph(false_edge), source_kind=source)
    non_bool = Conversion.from_callable(
        "non-bool", source, target, lambda value: value, admissibility=lambda value: 1
    )
    with pytest.raises(ConversionApplicationError, match="non-bool"):
        convert(1, target, graph=_graph(non_bool), source_kind=source)
    exploding = Conversion.from_callable(
        "explode",
        source,
        target,
        lambda value: (_ for _ in ()).throw(ValueError("secret")),
    )
    with pytest.raises(ConversionApplicationError) as caught:
        convert(1, target, graph=_graph(exploding), source_kind=source)
    assert "secret" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_no_route_cycles_explicit_errors_and_record_boundaries() -> None:
    source = _kind("source")
    middle = _kind("middle")
    target = _kind("target")
    cycle = _edge("source.middle", source, middle)
    back = _edge("middle.source", middle, source)
    self_edge = _edge("middle.middle", middle, middle)
    graph = _graph(cycle, back, self_edge)
    with pytest.raises(ConversionRouteError, match="no declared route"):
        plan_conversion(source, target, graph=graph)
    disconnected = RouteStep.from_conversion(cycle, "forward")
    with pytest.raises(ConversionRouteError, match="does not end"):
        plan_conversion(source, target, graph=graph, route=(disconnected,))
    with pytest.raises(ConversionRouteError, match="raw string"):
        plan_conversion(source, target, graph=graph, route="source.middle")
    plan = ConversionPlan.from_steps(source, middle, (disconnected,))
    result = ConvertedValue._from_applied("value", plan)
    for record in (disconnected, plan, result):
        assert not hasattr(record, "__dict__")
        assert "0x" not in repr(record)
        with pytest.raises(TypeError, match="unhashable"):
            hash(record)
    with pytest.raises(FrozenInstanceError):
        plan.target_kind = source  # type: ignore[misc]
    with pytest.raises(ConversionRouteError, match="from_conversion"):
        RouteStep(cycle, "forward")
    with pytest.raises(ConversionRouteError, match="from_steps"):
        ConversionPlan(source, middle, ())
    with pytest.raises(ConversionRouteError, match="produced"):
        ConvertedValue("value", plan)


def test_identity_competes_with_declared_self_cycle_and_tuple_bounds() -> None:
    source = _kind("source")
    middle = _kind("middle")
    self_edge = _edge("source.self", source, source)
    to_middle = _edge("source.middle", source, middle)
    back = _edge("middle.source", middle, source)
    graph = _graph(self_edge, to_middle, back)
    with pytest.raises(ConversionAmbiguityError) as caught:
        plan_conversion(source, source, graph=graph)
    assert caught.value.candidates == (
        (),
        (("source.self", "forward"),),
        (("source.middle", "forward"), ("middle.source", "forward")),
    )
    step = RouteStep.from_conversion(to_middle, "forward")
    with pytest.raises(ConversionSearchLimitError) as caught_limit:
        plan_conversion(source, middle, graph=graph, route=(step,) * 513)
    assert caught_limit.value.max_steps == 512
    assert caught_limit.value.max_candidates == 512
    assert caught_limit.value.observed == 513
    assert caught_limit.value.phase == "explicit route steps"


def test_explicit_route_preflight_and_hostile_iterators_never_call_transforms() -> None:
    source = _kind("source")
    middle = _kind("middle")
    target = _kind("target")
    calls = {"edge": 0}

    def transform(value: object) -> object:
        calls["edge"] += 1
        return value

    edge = Conversion.from_callable("source.middle", source, middle, transform)
    disconnected_edge = _edge("target.target", target, target)
    graph = _graph(edge, disconnected_edge)
    step = RouteStep.from_conversion(edge, "forward")
    disconnected_step = RouteStep.from_conversion(disconnected_edge, "forward")
    with pytest.raises(ConversionRouteError, match="exact ConversionGraph"):
        plan_conversion(source, middle, graph=object(), route=(step,))
    with pytest.raises(ConversionRouteError, match="exact PresentationKind"):
        plan_conversion(source, object(), graph=graph, route=(step,))
    with pytest.raises(ConversionRouteError, match="does not end"):
        convert("value", target, graph=graph, source_kind=source, route=(step,))
    with pytest.raises(ConversionRouteError, match="disconnected"):
        plan_conversion(
            source,
            target,
            graph=graph,
            route=(step, disconnected_step),
        )
    with pytest.raises(ConversionRouteError, match="disconnected"):
        plan_conversion(middle, middle, graph=graph, route=(step,))
    with pytest.raises(ConversionRouteError, match="exact PresentationKind"):
        convert("value", target, graph=graph, source_kind=object(), route=(step,))
    assert calls == {"edge": 0}

    def hostile() -> Iterator[RouteStep]:
        yield step
        raise ValueError("secret route iterator payload")

    with pytest.raises(ConversionRouteError, match="could not be iterated") as caught:
        plan_conversion(source, middle, graph=graph, route=hostile())
    assert "secret" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None

    def infinite() -> Iterator[RouteStep]:
        while True:
            yield step

    with pytest.raises(ConversionSearchLimitError) as caught_limit:
        plan_conversion(source, middle, graph=graph, route=infinite())
    assert caught_limit.value.observed == 513
    assert caught_limit.value.phase == "explicit route steps"


def test_inverse_and_admissibility_exceptions_plus_safe_result_repr() -> None:
    source = _kind("source")
    target = _kind("target")
    inverse_edge = Conversion.from_callable(
        "inverse.explodes",
        source,
        target,
        lambda value: value,
        inverse=lambda value: (_ for _ in ()).throw(RuntimeError("secret inverse")),
        admissibility=lambda value: (_ for _ in ()).throw(
            ValueError("secret admissibility")
        ),
    )
    graph = _graph(inverse_edge)
    inverse_step = RouteStep.from_conversion(inverse_edge, "inverse")
    with pytest.raises(ConversionApplicationError) as caught:
        convert("value", source, graph=graph, source_kind=target, route=(inverse_step,))
    assert caught.value.stage == "admissibility"
    assert "secret" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None

    exploding_inverse = Conversion.from_callable(
        "inverse.transform",
        source,
        target,
        lambda value: value,
        inverse=lambda value: (_ for _ in ()).throw(RuntimeError("secret inverse")),
    )
    with pytest.raises(ConversionApplicationError) as inverse_error:
        convert(
            "value",
            source,
            graph=_graph(exploding_inverse),
            source_kind=target,
            route=(RouteStep.from_conversion(exploding_inverse, "inverse"),),
        )
    assert inverse_error.value.direction == "inverse"
    assert inverse_error.value.stage == "transform"

    class Noisy:
        def __repr__(self) -> str:
            return "secret-payload-0xabc"

    result = convert(Noisy(), source, graph=ConversionGraph.empty(), source_kind=source)
    assert "secret-payload" not in repr(result)
    assert "0x" not in repr(result)


def test_reconstructed_kinds_and_exact_factory_subclass_holes() -> None:
    source = _kind("source")
    target = _kind("target")
    edge = _edge("source.target", source, target)
    graph = _graph(edge)
    reconstructed = convert(
        "value", _kind("target"), graph=graph, source_kind=_kind("source")
    )
    assert reconstructed.value == "value"

    class RouteStepSubclass(RouteStep):
        pass

    class PlanSubclass(ConversionPlan):
        pass

    class ValueSubclass(ConvertedValue):
        pass

    with pytest.raises(ConversionRouteError, match="exact RouteStep"):
        RouteStepSubclass.from_conversion(edge, "forward")
    step = RouteStep.from_conversion(edge, "forward")
    with pytest.raises(ConversionRouteError, match="exact ConversionPlan"):
        PlanSubclass.from_steps(source, target, (step,))
    plan = ConversionPlan.from_steps(source, target, (step,))
    with pytest.raises(ConversionRouteError, match="exact ConvertedValue"):
        ValueSubclass._from_applied("value", plan)


def test_error_taxonomy_and_application_messages_are_not_route_diagnostics() -> None:
    route_error = ConversionRouteError(reason="fixture")
    ambiguity = ConversionAmbiguityError(())
    application = ConversionApplicationError(
        step_index=0,
        edge_id="edge",
        direction="forward",
        stage="transform",
    )
    rejected = ConversionNotAdmissibleError(
        step_index=0,
        edge_id="edge",
        direction="forward",
    )
    assert isinstance(route_error, ConversionError)
    assert isinstance(route_error, ValueError)
    assert isinstance(ambiguity, ConversionRouteError)
    assert isinstance(application, ConversionError)
    assert not isinstance(application, ConversionRouteError)
    assert not isinstance(application, ValueError)
    assert isinstance(rejected, ConversionError)
    assert not isinstance(rejected, ConversionRouteError)
    assert not isinstance(rejected, ValueError)
    assert "invalid conversion route" not in str(application)
    assert "invalid conversion route" not in str(rejected)


def test_plan_length_boundary_hostile_direction_and_result_preflight() -> None:
    kinds = tuple(_kind(f"kind.{index}") for index in range(513))
    steps = tuple(
        RouteStep.from_conversion(
            _edge(f"edge.{index}", kinds[index], kinds[index + 1]), "forward"
        )
        for index in range(512)
    )
    accepted = ConversionPlan.from_steps(kinds[0], kinds[-1], steps)
    assert len(accepted.steps) == 512
    with pytest.raises(ConversionSearchLimitError) as caught:
        ConversionPlan.from_steps(kinds[0], kinds[-1], (*steps, steps[-1]))
    assert caught.value.observed == 513
    assert caught.value.phase == "explicit route steps"

    edge = _edge("hostile.direction", kinds[0], kinds[1])

    class HostileDirection:
        calls = 0

        def __eq__(self, other: object) -> bool:
            del other
            type(self).calls += 1
            raise AssertionError("hostile equality must not run")

    class StringSubclass(str):
        pass

    with pytest.raises(ConversionRouteError, match="direction"):
        RouteStep.from_conversion(edge, HostileDirection())
    with pytest.raises(ConversionRouteError, match="direction"):
        RouteStep.from_conversion(edge, StringSubclass("forward"))
    assert HostileDirection.calls == 0
    with pytest.raises(ConversionRouteError, match="exact ConversionPlan"):
        ConvertedValue._from_applied("value", object())  # type: ignore[arg-type]


def test_all_topological_candidates_and_inverse_candidates_remain_ambiguous() -> None:
    source = _kind("source")
    middle = _kind("middle")
    target = _kind("target")
    calls = {
        "direct": 0,
        "first": 0,
        "second": 0,
        "direct_admissibility": 0,
        "first_admissibility": 0,
        "second_admissibility": 0,
    }

    def direct(value: object) -> object:
        calls["direct"] += 1
        return value

    def first(value: object) -> object:
        calls["first"] += 1
        return value

    def second(value: object) -> object:
        calls["second"] += 1
        return value

    def direct_admissibility(value: object) -> bool:
        del value
        calls["direct_admissibility"] += 1
        return True

    def first_admissibility(value: object) -> bool:
        del value
        calls["first_admissibility"] += 1
        return True

    def second_admissibility(value: object) -> bool:
        del value
        calls["second_admissibility"] += 1
        return True

    direct_edge = Conversion.from_callable(
        "direct", source, target, direct, admissibility=direct_admissibility, cost=0
    )
    first_edge = Conversion.from_callable(
        "first", source, middle, first, admissibility=first_admissibility, cost=99
    )
    second_edge = Conversion.from_callable(
        "second", middle, target, second, admissibility=second_admissibility, cost=1
    )
    graph = _graph(direct_edge, first_edge, second_edge)
    with pytest.raises(ConversionAmbiguityError) as caught:
        plan_conversion(source, target, graph=graph)
    assert caught.value.candidates == (
        (("direct", "forward"),),
        (("first", "forward"), ("second", "forward")),
    )
    assert calls == {
        "direct": 0,
        "first": 0,
        "second": 0,
        "direct_admissibility": 0,
        "first_admissibility": 0,
        "second_admissibility": 0,
    }

    inverse_edge = Conversion.from_callable(
        "inverse",
        source,
        target,
        lambda value: value,
        inverse=lambda value: value,
    )
    reverse_edge = Conversion.from_callable(
        "reverse", target, source, lambda value: value
    )
    with pytest.raises(ConversionAmbiguityError) as inverse_ambiguity:
        plan_conversion(target, source, graph=_graph(inverse_edge, reverse_edge))
    assert inverse_ambiguity.value.candidates == (
        (("inverse", "inverse"),),
        (("reverse", "forward"),),
    )


def test_candidate_overflow_never_truncates_declared_self_routes() -> None:
    source = _kind("source")
    graph = _graph(
        *tuple(_edge(f"self.{index}", source, source) for index in range(512))
    )
    with pytest.raises(ConversionSearchLimitError) as caught:
        plan_conversion(source, source, graph=graph)
    assert caught.value.phase == "candidate routes"
    assert caught.value.observed == 513
    assert caught.value.max_steps == 512
    assert caught.value.max_candidates == 512


def test_explicit_empty_self_selection_inverse_and_plan_alias() -> None:
    source = _kind("source")
    target = _kind("target")
    self_edge = Conversion.from_callable(
        "self", source, source, lambda value: f"s:{value}"
    )
    inverse_edge = Conversion.from_callable(
        "source.target",
        source,
        target,
        lambda value: f"f:{value}",
        inverse=lambda value: f"i:{value}",
    )
    graph = _graph(self_edge, inverse_edge)
    selected = convert(
        "payload",
        source,
        graph=graph,
        source_kind=source,
        route=(RouteStep.from_conversion(self_edge, "forward"),),
    )
    assert selected.value == "s:payload"
    assert selected.source_kind is source
    assert selected.target_kind is source
    assert selected.route_witness == (("self", "forward"),)
    assert selected.validity_domains == ("all",)
    assert selected.round_trip_ids == ()
    inverse_plan = plan(target, source, graph=graph)
    assert inverse_plan.route_witness == (("source.target", "inverse"),)
    assert (
        convert("payload", source, graph=graph, source_kind=target).value == "i:payload"
    )
    assert (
        plan_conversion(source, source, graph=ConversionGraph.empty(), route=()).steps
        == ()
    )
    with pytest.raises(ConversionRouteError, match="does not end"):
        plan_conversion(source, target, graph=ConversionGraph.empty(), route=())
