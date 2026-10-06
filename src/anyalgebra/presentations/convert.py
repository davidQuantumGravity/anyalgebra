"""Explicit presentation-conversion routing and application.

Topology belongs to :mod:`anyalgebra.presentations.graph`; this module only
turns one explicit or uniquely declared finite route into an application plan.
It never infers a presentation kind from a Python value and never uses cost or
admissibility execution to break a topological ambiguity.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import cast

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.presentations.graph import Conversion, ConversionGraph, PresentationKind


_MAX_ROUTE_STEPS = 512
_MAX_CANDIDATES = 512


class ConversionError(AnyAlgebraError):
    """Base class for typed presentation conversion failures."""


class ConversionRouteError(ConversionError, ValueError):
    """A route request was malformed, disconnected, or unreachable."""

    def __init__(self, *, reason: str) -> None:
        """Keep an inert, deterministic route diagnostic."""
        self.reason = reason
        super().__init__(f"invalid conversion route: {reason}")


class ConversionAmbiguityError(ConversionRouteError):
    """Automatic routing found more than one declared topology route."""

    def __init__(self, candidates: tuple[tuple[tuple[str, str], ...], ...]) -> None:
        """Retain every bounded stable-ID/direction witness in declaration order."""
        self.candidates = candidates
        super().__init__(reason=f"ambiguous declared routes: {candidates}")


class ConversionNotAdmissibleError(ConversionError):
    """A selected route step rejected the current payload by exact ``False``."""

    def __init__(self, *, step_index: int, edge_id: str, direction: str) -> None:
        """Record only stable route location metadata."""
        self.step_index = step_index
        self.edge_id = edge_id
        self.direction = direction
        self.reason = (
            f"step {step_index} edge {edge_id!r} direction {direction!r} "
            "is not admissible"
        )
        super().__init__(f"conversion not admissible: {self.reason}")


class ConversionApplicationError(ConversionError):
    """A selected admissibility or transform callable raised or returned invalidly."""

    def __init__(
        self,
        *,
        step_index: int,
        edge_id: str,
        direction: str,
        stage: str,
        cause_type: str | None = None,
    ) -> None:
        """Expose safe stable step context without caller exception payloads."""
        self.step_index = step_index
        self.edge_id = edge_id
        self.direction = direction
        self.stage = stage
        self.cause_type = cause_type
        detail = (
            f"step {step_index} edge {edge_id!r} direction {direction!r} {stage} failed"
        )
        if cause_type is not None:
            detail = f"{detail} ({cause_type})"
        self.reason = detail
        super().__init__(f"conversion application failed: {detail}")


class ConversionSearchLimitError(ConversionError):
    """Finite topology search exceeded a declared non-success bound."""

    def __init__(
        self,
        *,
        max_steps: int,
        max_candidates: int,
        observed: int,
        phase: str,
    ) -> None:
        """Retain exact bound/phase metadata without a partial-success claim."""
        self.max_steps = max_steps
        self.max_candidates = max_candidates
        self.observed = observed
        self.phase = phase
        super().__init__(
            f"search limit in {phase}: observed {observed}, "
            f"max_steps={max_steps}, max_candidates={max_candidates}"
        )


def _safe_exception_name(error: Exception) -> str:
    """Return a payload-free stable exception type label."""
    name = type(error).__name__
    return name if type(name) is str and name.isidentifier() else "exception"


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class RouteStep:
    """One explicit traversal of a declared conversion in one direction."""

    conversion: Conversion
    direction: str
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked records; use :meth:`from_conversion`."""
        del args, kwargs
        raise ConversionRouteError(reason="use RouteStep.from_conversion")

    @classmethod
    def from_conversion(cls, conversion: object, direction: object) -> RouteStep:
        """Freeze an exact forward or declared-inverse traversal selector."""
        if cls is not RouteStep:
            raise ConversionRouteError(reason="factory requires exact RouteStep")
        if type(conversion) is not Conversion:
            raise ConversionRouteError(
                reason="step conversion must be an exact Conversion"
            )
        if type(direction) is not str or direction not in ("forward", "inverse"):
            raise ConversionRouteError(
                reason="step direction must be 'forward' or 'inverse'"
            )
        if direction == "inverse" and conversion.inverse is None:
            raise ConversionRouteError(reason="inverse direction is not declared")
        step = object.__new__(cls)
        object.__setattr__(step, "conversion", conversion)
        object.__setattr__(step, "direction", direction)
        return step

    @property
    def source_kind(self) -> PresentationKind:
        """Return the semantic traversal source kind."""
        return (
            self.conversion.source_kind
            if self.direction == "forward"
            else self.conversion.target_kind
        )

    @property
    def target_kind(self) -> PresentationKind:
        """Return the semantic traversal target kind."""
        return (
            self.conversion.target_kind
            if self.direction == "forward"
            else self.conversion.source_kind
        )

    @property
    def witness(self) -> tuple[str, str]:
        """Return the stable edge-ID/direction selector witness."""
        return self.conversion.stable_id, self.direction

    def __eq__(self, other: object) -> bool:
        """Keep callable-bearing selector records identity-only."""
        return self is other

    def __repr__(self) -> str:
        """Render only safe route metadata."""
        return (
            f"RouteStep(edge_id={self.conversion.stable_id!r}, "
            f"direction={self.direction!r})"
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ConversionPlan:
    """One immutable, connected route between explicit presentation kinds."""

    source_kind: PresentationKind
    target_kind: PresentationKind
    steps: tuple[RouteStep, ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked records; use :meth:`from_steps`."""
        del args, kwargs
        raise ConversionRouteError(reason="use ConversionPlan.from_steps")

    @classmethod
    def from_steps(
        cls, source_kind: object, target_kind: object, steps: object
    ) -> ConversionPlan:
        """Freeze a fully connected exact tuple of route steps."""
        if cls is not ConversionPlan:
            raise ConversionRouteError(reason="factory requires exact ConversionPlan")
        _require_kind(source_kind, "source_kind")
        _require_kind(target_kind, "target_kind")
        if type(steps) is not tuple:
            raise ConversionRouteError(reason="route steps must be an exact tuple")
        if len(steps) > _MAX_ROUTE_STEPS:
            raise ConversionSearchLimitError(
                max_steps=_MAX_ROUTE_STEPS,
                max_candidates=_MAX_CANDIDATES,
                observed=len(steps),
                phase="explicit route steps",
            )
        current = source_kind
        for index, step in enumerate(steps):
            if type(step) is not RouteStep:
                raise ConversionRouteError(
                    reason=f"route step {index} must be an exact RouteStep"
                )
            if step.source_kind != current:
                raise ConversionRouteError(reason=f"route step {index} is disconnected")
            current = step.target_kind
        if current != target_kind:
            raise ConversionRouteError(reason="route does not end at target kind")
        plan = object.__new__(cls)
        object.__setattr__(plan, "source_kind", source_kind)
        object.__setattr__(plan, "target_kind", target_kind)
        object.__setattr__(plan, "steps", steps)
        return plan

    @property
    def route_witness(self) -> tuple[tuple[str, str], ...]:
        """Return exact stable edge-ID/direction witnesses in route order."""
        return tuple(step.witness for step in self.steps)

    @property
    def cost(self) -> int:
        """Return transparent declared route cost without choosing by it."""
        return sum(step.conversion.cost for step in self.steps)

    @property
    def guarantees(self) -> tuple[str, ...]:
        """Aggregate declared guarantees in route order without inventing proof."""
        return tuple(
            guarantee for step in self.steps for guarantee in step.conversion.guarantees
        )

    @property
    def validity_domains(self) -> tuple[str, ...]:
        """Return named declared validity domains in route order."""
        return tuple(step.conversion.validity_domain for step in self.steps)

    @property
    def round_trip_ids(self) -> tuple[str, ...]:
        """Return declared inverse-associated names without certifying them."""
        return tuple(
            step.conversion.round_trip_id
            for step in self.steps
            if step.conversion.round_trip_id is not None
        )

    def __eq__(self, other: object) -> bool:
        """Keep plans identity-only because route steps own executable declarations."""
        return self is other

    def __repr__(self) -> str:
        """Render only bounded safe route metadata."""
        return f"ConversionPlan(step_count={len(self.steps)}, cost={self.cost})"


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ConvertedValue:
    """One raw payload transported through an explicit immutable conversion plan."""

    value: object = field(repr=False)
    plan: ConversionPlan
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked records; application owns this result boundary."""
        del args, kwargs
        raise ConversionRouteError(reason="ConvertedValue is produced by convert")

    @classmethod
    def _from_applied(cls, value: object, plan: ConversionPlan) -> ConvertedValue:
        """Allocate a result after its plan has applied every selected step."""
        if cls is not ConvertedValue:
            raise ConversionRouteError(reason="factory requires exact ConvertedValue")
        if type(plan) is not ConversionPlan:
            raise ConversionRouteError(
                reason="application result requires an exact ConversionPlan"
            )
        record = object.__new__(cls)
        object.__setattr__(record, "value", value)
        object.__setattr__(record, "plan", plan)
        return record

    @property
    def source_kind(self) -> PresentationKind:
        """Return the explicit plan source kind."""
        return self.plan.source_kind

    @property
    def target_kind(self) -> PresentationKind:
        """Return the explicit plan target kind."""
        return self.plan.target_kind

    @property
    def route_witness(self) -> tuple[tuple[str, str], ...]:
        """Return the applied exact route witness."""
        return self.plan.route_witness

    @property
    def cost(self) -> int:
        """Return transparent summed declared route cost."""
        return self.plan.cost

    @property
    def guarantees(self) -> tuple[str, ...]:
        """Return ordered declared guarantees without adding certification."""
        return self.plan.guarantees

    @property
    def validity_domains(self) -> tuple[str, ...]:
        """Return ordered declared validity-domain names."""
        return self.plan.validity_domains

    @property
    def round_trip_ids(self) -> tuple[str, ...]:
        """Return declared round-trip names without claiming equality proof."""
        return self.plan.round_trip_ids

    def __eq__(self, other: object) -> bool:
        """Treat arbitrary-payload application results as identity-only."""
        return self is other

    def __repr__(self) -> str:
        """Avoid arbitrary transported-payload reprs."""
        return (
            f"ConvertedValue(step_count={len(self.plan.steps)}, cost={self.plan.cost})"
        )


def _require_kind(value: object, field: str) -> PresentationKind:
    """Require an exact explicit semantic kind without inspecting payload shape."""
    if type(value) is not PresentationKind:
        raise ConversionRouteError(reason=f"{field} must be an exact PresentationKind")
    return value


def _require_graph(value: object) -> ConversionGraph:
    """Require the exact immutable topology type for all route operations."""
    if type(value) is not ConversionGraph:
        raise ConversionRouteError(reason="graph must be an exact ConversionGraph")
    return value


def _snapshot_steps(route: object) -> tuple[RouteStep, ...]:
    """Bound one explicit route iterable and sanitize arbitrary iterator failures."""
    if type(route) is tuple:
        values = route
        if len(values) > _MAX_ROUTE_STEPS:
            raise ConversionSearchLimitError(
                max_steps=_MAX_ROUTE_STEPS,
                max_candidates=_MAX_CANDIDATES,
                observed=len(values),
                phase="explicit route steps",
            )
    else:
        if isinstance(route, str | bytes):
            raise ConversionRouteError(reason="route must not be a raw string iterable")
        failure: str | None
        try:
            iterator = iter(cast(Iterable[object], route))
        except Exception as error:
            failure = _safe_exception_name(error)
            iterator = None
        else:
            failure = None
        if iterator is None:
            raise ConversionRouteError(
                reason=f"route could not be iterated ({failure})"
            )
        gathered: list[object] = []
        failed = False
        try:
            for index, value in enumerate(iterator):
                if index >= _MAX_ROUTE_STEPS:
                    raise ConversionSearchLimitError(
                        max_steps=_MAX_ROUTE_STEPS,
                        max_candidates=_MAX_CANDIDATES,
                        observed=index + 1,
                        phase="explicit route steps",
                    )
                gathered.append(value)
        except (ConversionRouteError, ConversionSearchLimitError):
            raise
        except Exception as error:
            failed = True
            failure = _safe_exception_name(error)
        if failed:
            raise ConversionRouteError(
                reason=f"route could not be iterated ({failure})"
            )
        values = tuple(gathered)
    for index, value in enumerate(values):
        if type(value) is not RouteStep:
            raise ConversionRouteError(
                reason=f"route step {index} must be an exact RouteStep"
            )
    return cast(tuple[RouteStep, ...], values)


def _registered_plan(
    source_kind: PresentationKind,
    target_kind: PresentationKind,
    graph: ConversionGraph,
    steps: tuple[RouteStep, ...],
) -> ConversionPlan:
    """Validate explicit edge registration and endpoints before any callable use."""
    for index, step in enumerate(steps):
        if not any(edge is step.conversion for edge in graph.edges):
            raise ConversionRouteError(
                reason=f"route step {index} names an unregistered edge"
            )
    return ConversionPlan.from_steps(source_kind, target_kind, steps)


def _candidate_routes(
    source_kind: PresentationKind,
    target_kind: PresentationKind,
    graph: ConversionGraph,
) -> tuple[tuple[RouteStep, ...], ...]:
    """Enumerate all bounded finite simple routes in graph declaration order."""
    found: list[tuple[RouteStep, ...]] = [()] if source_kind == target_kind else []

    def append(route: tuple[RouteStep, ...]) -> None:
        if len(found) >= _MAX_CANDIDATES:
            raise ConversionSearchLimitError(
                max_steps=_MAX_ROUTE_STEPS,
                max_candidates=_MAX_CANDIDATES,
                observed=len(found) + 1,
                phase="candidate routes",
            )
        found.append(route)

    def walk(
        current: PresentationKind,
        visited: tuple[PresentationKind, ...],
        route: tuple[RouteStep, ...],
    ) -> None:
        if len(route) >= _MAX_ROUTE_STEPS:
            raise ConversionSearchLimitError(
                max_steps=_MAX_ROUTE_STEPS,
                max_candidates=_MAX_CANDIDATES,
                observed=len(route) + 1,
                phase="route steps",
            )
        for edge in graph.edges:
            candidates: tuple[RouteStep, ...] = ()
            if edge.source_kind == current:
                candidates += (RouteStep.from_conversion(edge, "forward"),)
            if edge.inverse is not None and edge.target_kind == current:
                candidates += (RouteStep.from_conversion(edge, "inverse"),)
            for step in candidates:
                next_kind = step.target_kind
                next_route = (*route, step)
                if next_kind == target_kind:
                    append(next_route)
                    continue
                if any(next_kind == prior for prior in visited):
                    continue
                walk(next_kind, (*visited, next_kind), next_route)

    walk(source_kind, (source_kind,), ())
    return tuple(found)


def plan_conversion(
    source_kind: object,
    target_kind: object,
    *,
    graph: object,
    route: object | None = None,
) -> ConversionPlan:
    """Return an explicit route plan or the one uniquely declared topology route."""
    source = _require_kind(source_kind, "source_kind")
    target = _require_kind(target_kind, "target_kind")
    topology = _require_graph(graph)
    if route is not None:
        return _registered_plan(source, target, topology, _snapshot_steps(route))
    candidates = _candidate_routes(source, target, topology)
    if not candidates:
        raise ConversionRouteError(reason="no declared route reaches target kind")
    if len(candidates) != 1:
        raise ConversionAmbiguityError(
            tuple(tuple(step.witness for step in candidate) for candidate in candidates)
        )
    return ConversionPlan.from_steps(source, target, candidates[0])


def plan(
    source_kind: object,
    target_kind: object,
    *,
    graph: object,
    route: object | None = None,
) -> ConversionPlan:
    """Alias :func:`plan_conversion` for the documented planning boundary."""
    return plan_conversion(source_kind, target_kind, graph=graph, route=route)


def _apply_plan(value: object, plan: ConversionPlan) -> object:
    """Apply one already-preflighted plan with stepwise safe diagnostics."""
    current = value
    for index, step in enumerate(plan.steps):
        conversion = step.conversion
        if conversion.admissibility is not None:
            admissibility_failure: str | None = None
            try:
                admissible = conversion.admissibility(current)
            except Exception as error:
                admissible = None
                admissibility_failure = _safe_exception_name(error)
            if admissibility_failure is not None:
                raise ConversionApplicationError(
                    step_index=index,
                    edge_id=conversion.stable_id,
                    direction=step.direction,
                    stage="admissibility",
                    cause_type=admissibility_failure,
                )
            if type(admissible) is not bool:
                raise ConversionApplicationError(
                    step_index=index,
                    edge_id=conversion.stable_id,
                    direction=step.direction,
                    stage="admissibility returned non-bool",
                )
            if not admissible:
                raise ConversionNotAdmissibleError(
                    step_index=index,
                    edge_id=conversion.stable_id,
                    direction=step.direction,
                )
        function = (
            conversion.forward if step.direction == "forward" else conversion.inverse
        )
        assert function is not None
        transform_failure: str | None = None
        try:
            current = function(current)
        except Exception as error:
            transform_failure = _safe_exception_name(error)
        if transform_failure is not None:
            raise ConversionApplicationError(
                step_index=index,
                edge_id=conversion.stable_id,
                direction=step.direction,
                stage="transform",
                cause_type=transform_failure,
            )
    return current


def convert(
    value: object,
    target_kind: object,
    *,
    graph: object,
    source_kind: object,
    route: object | None = None,
) -> ConvertedValue:
    """Transport an arbitrary payload through an explicit source-kind contract."""
    conversion_plan = plan_conversion(
        source_kind, target_kind, graph=graph, route=route
    )
    return ConvertedValue._from_applied(
        _apply_plan(value, conversion_plan), conversion_plan
    )
