"""Contract tests for exact coercion-plan application and diagnostics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass

import pytest

from anyalgebra.core.coercions import (
    CoercionGraph,
    CoercionMap,
    CoercionPlan,
    CoercionStepError,
    coerce,
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
    """Identity-sensitive test parent with no implicit conversions."""

    label: str

    def normalize(self, value: object) -> object:
        """Return an already-exact fixture payload."""
        return value

    def element(self, value: object) -> FixtureElement:
        """Construct one immutable element owned by this literal parent."""
        return FixtureElement(parent=self, value=value)


@dataclass(frozen=True, slots=True)
class FixtureElement:
    """Minimal immutable domain element used by application tests."""

    parent: FixtureDomain
    value: object


def _edge(
    identifier: str,
    source: FixtureDomain,
    target: FixtureDomain,
    forward: object,
    *,
    lossless: bool = True,
    cost: int = 1,
) -> CoercionMap[object, object]:
    """Build one fixture map while keeping callable typing local to the test."""
    return CoercionMap(
        id=identifier,
        source=source,
        target=target,
        forward=forward,  # type: ignore[arg-type]
        injective=lossless,
        exact=lossless,
        lossless=lossless,
        cost=cost,
    )


def _graph(*edges: CoercionMap[object, object]) -> CoercionGraph:
    """Register a collection of declared maps in the supplied order."""
    graph = CoercionGraph()
    for edge in edges:
        graph = graph.with_map(edge)
    return graph


def test_identity_and_unique_zz_to_qq_coercions_preserve_exact_values() -> None:
    """Identity returns the same object and the canonical embedding remains exact."""

    def zz_to_qq(value: DomainElement[int]) -> DomainElement[Rational]:
        return QQ().element(value.value)

    graph = CoercionGraph().with_map(
        CoercionMap(
            id="core.zz_to_qq.v1",
            source=ZZ(),
            target=QQ(),
            forward=zz_to_qq,
            injective=True,
            exact=True,
            lossless=True,
        )
    )
    integer = ZZ().element(7)
    rational = QQ().element((3, 5))

    assert coerce(integer, ZZ(), graph=graph) is integer
    converted = coerce(integer, QQ(), graph=graph)
    assert converted.parent is QQ()
    assert converted.value == Rational(7)
    assert coerce(rational, QQ(), graph=graph) is rational


def test_plan_applies_multiple_identity_and_multistep_routes_in_operand_order() -> None:
    """A resolved plan applies each stored edge exactly in its operand route."""
    source = FixtureDomain("source")
    middle = FixtureDomain("middle")
    target = FixtureDomain("target")
    calls: list[tuple[str, object]] = []

    def first(value: DomainElement[object]) -> FixtureElement:
        calls.append(("source.middle", value.value))
        return middle.element(f"m({value.value})")

    def second(value: DomainElement[object]) -> FixtureElement:
        calls.append(("middle.target", value.value))
        return target.element(f"t({value.value})")

    first_edge = _edge("source.middle", source, middle, first)
    second_edge = _edge("middle.target", middle, target, second)
    plan = CoercionPlan(
        sources=(source, target),
        target=target,
        routes=((first_edge, second_edge), ()),
    )
    already_target = target.element("ready")

    result = plan.apply((source.element("x"), already_target))

    assert type(result) is tuple
    assert result[0] == target.element("t(m(x))")
    assert result[1] is already_target
    assert calls == [("source.middle", "x"), ("middle.target", "m(x)")]

    calls.clear()
    implicit = coerce(
        source.element("y"), target, graph=_graph(second_edge, first_edge)
    )
    assert implicit == target.element("t(m(y))")
    assert calls == [("source.middle", "y"), ("middle.target", "m(y)")]
    with pytest.raises(AttributeError):
        result.append(already_target)  # type: ignore[attr-defined]
    with pytest.raises(FrozenInstanceError):
        plan.target = source  # type: ignore[misc]


def test_plan_rejects_arity_and_literal_parent_mismatch_before_any_step() -> None:
    """All input ownership is validated before an application can have effects."""
    source = FixtureDomain("A")
    equal_label_other = FixtureDomain("A")
    target = FixtureDomain("B")
    equal_target_other = FixtureDomain("B")
    calls: list[object] = []

    def convert(value: DomainElement[object]) -> FixtureElement:
        calls.append(value.value)
        return target.element(value.value)

    edge = _edge("a.to.b", source, target, convert)
    plan = CoercionPlan(sources=(source, target), target=target, routes=((edge,), ()))

    with pytest.raises(ValueError, match=r"expected 2.*received 0"):
        plan.apply(())
    with pytest.raises(CoercionError, match=r"operand 0.*source parent"):
        plan.apply((equal_label_other.element(1), target.element(2)))
    with pytest.raises(CoercionError, match=r"operand 1.*source parent"):
        plan.apply((source.element(1), equal_target_other.element(2)))
    with pytest.raises(TypeError, match=r"operand 0.*DomainElement"):
        plan.apply((object(), target.element(2)))  # type: ignore[arg-type]
    assert calls == []


def test_wrong_intermediate_parent_has_stable_step_diagnostics() -> None:
    """A map returning the wrong parent is wrapped with operand and map identity."""
    source = FixtureDomain("source")
    middle = FixtureDomain("middle")
    target = FixtureDomain("target")

    def wrong(value: DomainElement[object]) -> FixtureElement:
        return source.element(value.value)

    first = _edge("source.middle", source, middle, wrong)
    second = _edge(
        "middle.target",
        middle,
        target,
        lambda value: target.element(value.value),
    )
    plan = CoercionPlan(sources=(source,), target=target, routes=((first, second),))

    with pytest.raises(CoercionStepError) as caught:
        plan.apply((source.element(4),))

    error = caught.value
    assert error.operand_index == 0
    assert error.step_index == 0
    assert error.map_id == "source.middle"
    assert error.cause_type == "CoercionError"
    assert "operand 0" in error.reason
    assert "step 0" in error.reason
    assert "source.middle" in error.reason
    assert isinstance(error.__cause__, CoercionError)


def test_wrong_final_output_parent_identifies_the_final_route_step() -> None:
    """The final map must return an element of the plan's literal target parent."""
    source = FixtureDomain("source")
    middle = FixtureDomain("middle")
    target = FixtureDomain("target")
    equal_label_wrong_target = FixtureDomain("target")

    first = _edge(
        "source.middle",
        source,
        middle,
        lambda value: middle.element(value.value),
    )
    second = _edge(
        "middle.target",
        middle,
        target,
        lambda value: equal_label_wrong_target.element(value.value),
    )
    plan = CoercionPlan(sources=(source,), target=target, routes=((first, second),))

    with pytest.raises(CoercionStepError) as caught:
        plan.apply((source.element(4),))

    assert caught.value.step_index == 1
    assert caught.value.map_id == "middle.target"
    assert "target parent does not own the result" in caught.value.detail
    assert isinstance(caught.value.__cause__, CoercionError)


def test_map_exception_is_wrapped_without_losing_stable_step_identity() -> None:
    """Implementation exceptions cannot escape without the declared edge witness."""
    source = FixtureDomain("source")
    target = FixtureDomain("target")

    def explode(value: DomainElement[object]) -> FixtureElement:
        raise ArithmeticError(f"cannot transport {value.value}")

    graph = _graph(_edge("source.target", source, target, explode))

    with pytest.raises(CoercionStepError) as caught:
        coerce(source.element("x"), target, graph=graph)

    error = caught.value
    assert (error.operand_index, error.step_index, error.map_id) == (
        0,
        0,
        "source.target",
    )
    assert error.cause_type == "ArithmeticError"
    assert error.detail == "cannot transport x"
    assert isinstance(error.__cause__, ArithmeticError)


def test_implicit_coerce_refuses_ambiguous_lossy_and_missing_routes() -> None:
    """Costs and registration order never choose ambiguity, loss, or no route."""
    source = FixtureDomain("source")
    target = FixtureDomain("target")

    def forward(value: DomainElement[object]) -> FixtureElement:
        return target.element(value.value)

    ambiguous = _graph(
        _edge("z.route", source, target, forward, cost=0),
        _edge("a.route", source, target, forward, cost=100),
    )
    reverse_registration = _graph(
        _edge("a.route", source, target, forward, cost=100),
        _edge("z.route", source, target, forward, cost=0),
    )

    with pytest.raises(CoercionAmbiguityError) as ambiguity:
        coerce(source.element(1), target, graph=ambiguous)
    assert "a.route" in ambiguity.value.reason
    assert "z.route" in ambiguity.value.reason
    assert ambiguity.value.reason.index("a.route") < ambiguity.value.reason.index(
        "z.route"
    )
    with pytest.raises(CoercionAmbiguityError) as reverse_ambiguity:
        coerce(source.element(1), target, graph=reverse_registration)
    assert reverse_ambiguity.value.reason == ambiguity.value.reason

    lossy = _graph(
        _edge("source.target.lossy", source, target, forward, lossless=False)
    )
    with pytest.raises(LossyCoercionError, match=r"source\.target\.lossy"):
        coerce(source.element(1), target, graph=lossy)

    with pytest.raises(CoercionError, match="no declared coercion route"):
        coerce(source.element(1), target, graph=CoercionGraph())


def test_coerce_validates_value_target_and_graph_without_callable_effects() -> None:
    """Malformed calls fail before path search or user callable invocation."""
    source = FixtureDomain("source")
    target = FixtureDomain("target")
    called = False

    def convert(value: DomainElement[object]) -> FixtureElement:
        nonlocal called
        called = True
        return target.element(value.value)

    graph = _graph(_edge("source.target", source, target, convert))

    with pytest.raises(TypeError, match=r"value.*DomainElement"):
        coerce(object(), target, graph=graph)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"target.*Domain"):
        coerce(source.element(1), object(), graph=graph)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"graph.*CoercionGraph"):
        coerce(source.element(1), target, graph=object())  # type: ignore[arg-type]
    assert called is False
