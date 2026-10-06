"""Immutable coercion topology, exact planning, and checked application.

The graph records declared directed maps and deterministically enumerates
bounded simple paths.  Common-parent planning considers only declared lossless
paths and returns only a unique least common target with one path per input.
Resolved plans apply with literal-parent checks and stable step diagnostics;
no boundary in this module silently selects an ambiguous or lossy route.  A
map's semantic identity is its explicit ``id`` and declared metadata; a Python
callable is executable implementation detail, not a stable identity component.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, cast

from .domains import Domain, DomainElement
from .errors import CoercionAmbiguityError, CoercionError, LossyCoercionError


S = TypeVar("S")
T = TypeVar("T")


class CoercionStepError(CoercionError):
    """A declared coercion map failed at one stable plan position.

    The operand index, zero-based route step, and declared map ID remain
    available as structured diagnostics.  The original exception is retained
    through normal exception chaining; this wrapper adds plan context without
    pretending that an implementation failure is mathematical undefinedness.
    """

    def __init__(
        self,
        *,
        operand_index: int,
        step_index: int,
        map_id: str,
        cause: Exception,
    ) -> None:
        """Record deterministic route context and the implementation detail."""
        self.operand_index = operand_index
        self.step_index = step_index
        self.map_id = map_id
        self.cause_type = type(cause).__name__
        self.detail = str(cause)
        super().__init__(
            "coercion application failed at "
            f"operand {operand_index}, step {step_index}, map {map_id!r}: "
            f"{self.cause_type}: {self.detail}"
        )


@dataclass(frozen=True, slots=True)
class CoercionMap(Generic[S, T]):
    """One immutable, declared map from a source parent to a target parent.

    ``id`` is an explicitly supplied stable semantic identifier.  It must be
    globally unique only when a later :class:`CoercionGraph` registers maps;
    this value object does not make a global registry claim.  ``forward`` is
    intentionally excluded from equality and hashing: separately allocated
    callables can implement the same declared map, and their Python identities
    are neither portable nor mathematical evidence.

    A map is ``lossless`` only when its declaration is both exact and
    injective.  Exact non-injective maps (for example, a quotient) are allowed
    as explicit maps but are not lossless.  ``surjective`` records a separate
    declared property and does not imply losslessness.
    """

    id: str
    source: Domain[S]
    target: Domain[T]
    forward: Callable[[DomainElement[S]], DomainElement[T]] = field(
        repr=False, compare=False, hash=False
    )
    injective: bool = False
    surjective: bool = False
    exact: bool = False
    lossless: bool = False
    cost: int = 1
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate and freeze the complete declared map definition eagerly."""
        if (
            type(self.id) is not str
            or not self.id.strip()
            or self.id != self.id.strip()
        ):
            raise ValueError(
                "coercion id must be a non-empty string without outer whitespace"
            )
        if not isinstance(self.source, Domain):
            raise TypeError("coercion source must implement Domain")
        if not isinstance(self.target, Domain):
            raise TypeError("coercion target must implement Domain")
        if not callable(self.forward):
            raise TypeError("coercion forward must be callable")

        for name in ("injective", "surjective", "exact", "lossless"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"coercion {name} must be a bool")
        if type(self.cost) is not int or self.cost < 0:
            raise ValueError("coercion cost must be a non-negative built-in int")
        if self.lossless and (not self.exact or not self.injective):
            raise ValueError("a lossless coercion must be exact and injective")

        if type(self.assumptions) is not tuple:
            raise TypeError("coercion assumptions must be an exact tuple")
        assumptions = self.assumptions
        if any(
            type(assumption) is not str or not assumption.strip()
            for assumption in assumptions
        ):
            raise ValueError("coercion assumptions must contain non-empty strings")
        object.__setattr__(self, "assumptions", assumptions)

    @property
    def stable_id(self) -> str:
        """Return the declared portable identity, independent of ``forward``."""
        return self.id

    def __hash__(self) -> int:
        """Hash the declared semantic identifier, never a callable object."""
        return hash(self.id)

    def apply(self, value: DomainElement[S]) -> DomainElement[T]:
        """Apply this one declared edge with minimal local parent validation.

        This is not the public ``coerce`` operation: it neither selects paths
        nor decides whether a lossy map is permissible.  The later application
        layer owns graph planning and stepwise application diagnostics.
        """
        if not isinstance(value, DomainElement) or value.parent is not self.source:
            raise CoercionError(
                f"coercion {self.id!r} source parent does not own the input"
            )
        result = self.forward(value)
        if not isinstance(result, DomainElement) or result.parent is not self.target:
            raise CoercionError(
                f"coercion {self.id!r} target parent does not own the result"
            )
        return result


@dataclass(frozen=True, slots=True)
class CoercionPlan:
    """One immutable, fully determined lossless common-parent plan.

    ``sources`` and ``routes`` have matching operand order.  Each route starts
    at its corresponding source and ends at the literal ``target`` parent;
    the empty route is the lossless identity map.  Costs are transparent route
    metadata only: this value type never uses them to rank distinct routes.
    """

    sources: tuple[Domain[object], ...]
    target: Domain[object]
    routes: tuple[tuple[CoercionMap[object, object], ...], ...]

    def __post_init__(self) -> None:
        """Validate that every stored route proves one lossless transport."""
        if type(self.sources) is not tuple or not self.sources:
            raise ValueError("coercion plan requires a non-empty tuple of sources")
        if type(self.routes) is not tuple or len(self.routes) != len(self.sources):
            raise ValueError("coercion plan routes must match its sources")
        _validate_endpoint(self.target, name="target")
        for source, route in zip(self.sources, self.routes, strict=True):
            _validate_endpoint(source, name="source")
            if type(route) is not tuple:
                raise TypeError("coercion plan routes must be exact tuples")
            current = source
            for edge in route:
                if not isinstance(edge, CoercionMap):
                    raise TypeError(
                        "coercion plan routes must contain CoercionMap values"
                    )
                if not edge.lossless:
                    raise ValueError("coercion plan routes must be lossless")
                if edge.source is not current:
                    raise ValueError(
                        "coercion plan route has disconnected parent endpoints"
                    )
                current = edge.target
            if current is not self.target:
                raise ValueError("coercion plan route does not end at its target")

    @property
    def route_ids(self) -> tuple[tuple[str, ...], ...]:
        """Return immutable stable map-ID witnesses in operand order."""
        return tuple(tuple(edge.stable_id for edge in route) for route in self.routes)

    @property
    def route_costs(self) -> tuple[int, ...]:
        """Return each transparent route cost without using it as a preference."""
        return tuple(sum(edge.cost for edge in route) for route in self.routes)

    @property
    def total_cost(self) -> int:
        """Return the sum of declared route costs for evidence/reporting only."""
        return sum(self.route_costs)

    @property
    def lossless(self) -> bool:
        """State the invariant established by this planning boundary."""
        return True

    def apply(
        self, values: Iterable[DomainElement[object]]
    ) -> tuple[DomainElement[object], ...]:
        """Apply every stored route after atomic arity and parent validation.

        All inputs are materialized and checked against the plan's literal
        source-parent identities before any user map callable runs.  This
        prevents an invalid later operand from leaving an externally visible
        partially applied route.  Each implementation failure is then wrapped
        with stable operand, step, and map-ID context while retaining its
        original exception as ``__cause__``.
        """
        items = tuple(values)
        if len(items) != len(self.sources):
            raise ValueError(
                "coercion plan application expected "
                f"{len(self.sources)} values but received {len(items)}"
            )
        for index, (value, source) in enumerate(zip(items, self.sources, strict=True)):
            if not isinstance(value, DomainElement):
                raise TypeError(
                    f"coercion plan operand {index} must implement DomainElement"
                )
            if value.parent is not source:
                raise CoercionError(
                    f"coercion plan operand {index} source parent does not own "
                    "the input"
                )

        results: list[DomainElement[object]] = []
        for operand_index, (value, route) in enumerate(
            zip(items, self.routes, strict=True)
        ):
            current = value
            for step_index, edge in enumerate(route):
                try:
                    current = edge.apply(current)
                except Exception as error:
                    raise CoercionStepError(
                        operand_index=operand_index,
                        step_index=step_index,
                        map_id=edge.stable_id,
                        cause=error,
                    ) from error
            results.append(current)
        return tuple(results)


@dataclass(frozen=True, slots=True, init=False)
class CoercionGraph:
    """A persistent directed set of coercion maps with deterministic paths.

    Graph topology is stored only as an immutable tuple ordered by stable map
    ID.  Parent endpoints are compared by object identity throughout: two
    separately constructed parents with equal-looking labels are not silently
    identified.  ``paths`` enumerates every simple directed edge path within
    an edge-count bound; deciding which of those paths may be used is a later
    planning boundary.
    """

    _topology: tuple[CoercionMap[object, object], ...]

    def __init__(self) -> None:
        """Construct the one empty immutable graph value."""
        object.__setattr__(self, "_topology", ())

    @classmethod
    def _from_topology(
        cls, topology: tuple[CoercionMap[object, object], ...]
    ) -> CoercionGraph:
        """Freeze a previously validated tuple in canonical stable-ID order."""
        graph = object.__new__(cls)
        object.__setattr__(graph, "_topology", topology)
        return graph

    @property
    def topology(self) -> tuple[CoercionMap[object, object], ...]:
        """Return the immutable, stable-ID-sorted declared edge topology."""
        return self._topology

    def with_map(self, map_: CoercionMap[Any, Any]) -> CoercionGraph:
        """Return a graph extended by one map without mutating this graph.

        Re-registering a map with the same stable ID and all the same declared
        metadata is idempotent, even if it uses another callable object.  The
        same stable ID with different metadata is rejected rather than silently
        overwriting a semantic declaration.  Distinct stable IDs are allowed to
        be parallel edges between the same parents.
        """
        if not isinstance(map_, CoercionMap):
            raise TypeError("graph maps must be CoercionMap instances")
        candidate = cast(CoercionMap[object, object], map_)
        for existing in self._topology:
            if existing.stable_id == candidate.stable_id:
                if _same_declared_map(existing, candidate):
                    return self
                raise ValueError(
                    "conflicting coercion metadata for stable id "
                    f"{candidate.stable_id!r}"
                )
        return self._from_topology(
            tuple(sorted((*self._topology, candidate), key=lambda edge: edge.stable_id))
        )

    def paths(
        self,
        source: Domain[object],
        target: Domain[object],
        *,
        max_steps: int | None = None,
    ) -> tuple[tuple[CoercionMap[object, object], ...], ...]:
        """Enumerate deterministic simple directed paths from source to target.

        ``max_steps`` counts edges and must be a nonnegative built-in ``int``;
        ``None`` uses the finite topology size as a complete simple-path bound.
        A source identical to the target yields the empty identity path, even
        at zero steps.  An unreachable target yields an empty tuple.  Repeated
        parent identities are never expanded, so cycles cannot create an
        infinite search or non-simple route.
        """
        _validate_endpoint(source, name="source")
        _validate_endpoint(target, name="target")
        limit = _validated_max_steps(max_steps, topology_size=len(self._topology))
        if source is target:
            return ((),)

        found: list[tuple[CoercionMap[object, object], ...]] = []

        def walk(
            current: Domain[object],
            visited: tuple[Domain[object], ...],
            path: tuple[CoercionMap[object, object], ...],
        ) -> None:
            if len(path) == limit:
                return
            for edge in self._topology:
                if edge.source is not current:
                    continue
                next_parent = edge.target
                if any(parent is next_parent for parent in visited):
                    continue
                next_path = (*path, edge)
                if next_parent is target:
                    found.append(next_path)
                    continue
                walk(next_parent, (*visited, next_parent), next_path)

        walk(source, (source,), ())
        return tuple(
            sorted(found, key=lambda path: tuple(edge.stable_id for edge in path))
        )

    def plan(self, source: Domain[object], target: Domain[object]) -> CoercionPlan:
        """Resolve one unique lossless route between literal parent identities.

        Lossy paths never compete with a unique lossless path.  If no lossless
        route exists, a loss-specific diagnostic names every declared path;
        multiple lossless routes remain ambiguity regardless of edge costs.
        """
        _validate_endpoint(source, name="source")
        _validate_endpoint(target, name="target")
        route = _unique_lossless_route(self, source, target)
        return CoercionPlan(sources=(source,), target=target, routes=(route,))


def coerce(
    value: DomainElement[object],
    target: Domain[object],
    *,
    graph: CoercionGraph,
) -> DomainElement[object]:
    """Transport one value through a unique declared lossless coercion route.

    The value's literal parent supplies the route source.  Identity transport
    returns the same immutable element.  This boundary never resolves route
    ambiguity by cost or applies a declared lossy conversion implicitly.
    """
    if not isinstance(value, DomainElement):
        raise TypeError("coerce value must implement DomainElement")
    _validate_endpoint(target, name="target")
    if not isinstance(graph, CoercionGraph):
        raise TypeError("graph must be a CoercionGraph")
    source = value.parent
    _validate_endpoint(source, name="value parent")
    return graph.plan(source, target).apply((value,))[0]


def common_parent(
    values: Iterable[DomainElement[object]], *, graph: CoercionGraph
) -> CoercionPlan:
    """Plan a unique, least, lossless common parent for ``values``.

    Candidate parents are the finite identities declared by graph endpoints,
    plus any input parent absent from that topology.  A candidate is admissible
    only if every input has at least one lossless simple path to it; identity
    paths are lossless.  Of the admissible candidates, a target is *least*
    when no distinct admissible candidate can itself reach it losslessly.  This
    prevents a downstream extension from competing with an earlier common
    parent in a chain such as ``A -> C -> D``.  Mutual reachability is not
    treated as strict dominance: distinct parents in a lossless strongly
    connected component remain an explicit ambiguity.

    The result exists only if exactly one least target remains and every input
    has exactly one lossless path to it.  Costs never resolve an ambiguity.
    The function plans only: it neither applies maps nor accepts a user policy.
    """
    if not isinstance(graph, CoercionGraph):
        raise TypeError("graph must be a CoercionGraph")

    items = tuple(values)
    if not items:
        raise ValueError("common_parent requires at least one DomainElement")
    sources = tuple(_element_parent(value) for value in items)
    identity_parent = sources[0]
    if all(source is identity_parent for source in sources):
        return CoercionPlan(
            sources=sources,
            target=identity_parent,
            routes=tuple(() for _ in sources),
        )
    candidates = _deterministic_parents(graph, sources)

    lossless_routes = {
        id(candidate): tuple(
            _lossless_paths(graph, source, candidate) for source in sources
        )
        for candidate in candidates
    }
    admissible = tuple(
        candidate
        for candidate in candidates
        if all(lossless_routes[id(candidate)][index] for index in range(len(sources)))
    )
    if not admissible:
        _raise_no_lossless_common_parent(graph, sources, candidates)

    least = tuple(
        candidate
        for candidate in admissible
        if not any(
            other is not candidate
            and _lossless_paths(graph, other, candidate)
            and not _lossless_paths(graph, candidate, other)
            for other in admissible
        )
    )
    if len(least) != 1:
        raise CoercionAmbiguityError(
            "ambiguous incomparable least common parents: "
            f"{_candidate_witnesses(least, lossless_routes)}"
        )

    target = least[0]
    routes = lossless_routes[id(target)]
    ambiguous_routes = tuple(
        (index, route_set)
        for index, route_set in enumerate(routes)
        if len(route_set) != 1
    )
    if ambiguous_routes:
        raise CoercionAmbiguityError(
            "ambiguous lossless coercion routes to the unique least common "
            f"parent: {_route_witnesses(ambiguous_routes)}"
        )
    return CoercionPlan(
        sources=sources,
        target=target,
        routes=tuple(route_set[0] for route_set in routes),
    )


def _element_parent(value: object) -> Domain[object]:
    """Return one validated literal element parent without constructing values."""
    if not isinstance(value, DomainElement):
        raise TypeError("common_parent values must implement DomainElement")
    parent = value.parent
    _validate_endpoint(parent, name="value parent")
    return cast(Domain[object], parent)


def _deterministic_parents(
    graph: CoercionGraph, sources: tuple[Domain[object], ...]
) -> tuple[Domain[object], ...]:
    """Enumerate endpoint identities in stable edge order, then absent sources."""
    parents: list[Domain[object]] = []

    def add(parent: Domain[object]) -> None:
        if not any(existing is parent for existing in parents):
            parents.append(parent)

    for edge in graph.topology:
        add(edge.source)
        add(edge.target)
    for source in sources:
        add(source)
    return tuple(parents)


def _lossless_paths(
    graph: CoercionGraph, source: Domain[object], target: Domain[object]
) -> tuple[tuple[CoercionMap[object, object], ...], ...]:
    """Return the complete finite simple-path subset whose every edge is lossless."""
    return tuple(
        path
        for path in graph.paths(source, target)
        if all(edge.lossless for edge in path)
    )


def _unique_lossless_route(
    graph: CoercionGraph, source: Domain[object], target: Domain[object]
) -> tuple[CoercionMap[object, object], ...]:
    """Return exactly one lossless path or raise a deterministic typed failure."""
    all_routes = graph.paths(source, target)
    lossless_routes = tuple(
        route for route in all_routes if all(edge.lossless for edge in route)
    )
    if len(lossless_routes) == 1:
        return lossless_routes[0]
    if len(lossless_routes) > 1:
        raise CoercionAmbiguityError(
            f"ambiguous lossless coercion routes: {_path_id_witnesses(lossless_routes)}"
        )
    if all_routes:
        raise LossyCoercionError(
            "no lossless coercion route; declared routes require explicit "
            f"conversion: {_path_id_witnesses(all_routes)}"
        )
    raise CoercionError("no declared coercion route reaches the target parent")


def _path_id_witnesses(
    routes: tuple[tuple[CoercionMap[object, object], ...], ...],
) -> tuple[tuple[str, ...], ...]:
    """Project already deterministically ordered routes to stable map IDs."""
    return tuple(tuple(edge.stable_id for edge in route) for route in routes)


def _raise_no_lossless_common_parent(
    graph: CoercionGraph,
    sources: tuple[Domain[object], ...],
    candidates: tuple[Domain[object], ...],
) -> None:
    """Raise a loss-specific error only when a declared lossy common route exists."""
    lossy_ids: list[str] = []
    for candidate in candidates:
        routes = tuple(graph.paths(source, candidate) for source in sources)
        if not all(routes):
            continue
        for route_set in routes:
            for route in route_set:
                lossy_ids.extend(edge.stable_id for edge in route if not edge.lossless)
        if lossy_ids:
            raise LossyCoercionError(
                "no lossless common parent; declared lossy routes require "
                "explicit conversion: "
                f"{tuple(sorted(set(lossy_ids)))}"
            )
    raise CoercionError("no declared common parent is reachable by lossless coercions")


def _route_witnesses(
    routes: tuple[tuple[int, tuple[tuple[CoercionMap[object, object], ...], ...]], ...],
) -> tuple[tuple[int, tuple[tuple[str, ...], ...]], ...]:
    """Format deterministic operand-indexed route-ID witnesses for diagnostics."""
    return tuple(
        (
            index,
            tuple(tuple(edge.stable_id for edge in path) for path in route_set),
        )
        for index, route_set in routes
    )


def _candidate_witnesses(
    candidates: tuple[Domain[object], ...],
    route_sets: dict[
        int,
        tuple[tuple[tuple[CoercionMap[object, object], ...], ...], ...],
    ],
) -> tuple[tuple[int, tuple[tuple[tuple[str, ...], ...], ...]], ...]:
    """Format candidate-indexed complete paths without relying on parent reprs."""
    return tuple(
        (
            index,
            tuple(
                tuple(tuple(edge.stable_id for edge in path) for path in paths)
                for paths in route_sets[id(candidate)]
            ),
        )
        for index, candidate in enumerate(candidates)
    )


def _same_declared_map(
    left: CoercionMap[object, object], right: CoercionMap[object, object]
) -> bool:
    """Compare declared map metadata while preserving literal parent identity."""
    return (
        left.source is right.source
        and left.target is right.target
        and left.injective == right.injective
        and left.surjective == right.surjective
        and left.exact == right.exact
        and left.lossless == right.lossless
        and left.cost == right.cost
        and left.assumptions == right.assumptions
    )


def _validate_endpoint(endpoint: object, *, name: str) -> None:
    """Reject non-domain path endpoints before graph traversal begins."""
    if not isinstance(endpoint, Domain):
        raise TypeError(f"{name} must implement Domain")


def _validated_max_steps(max_steps: int | None, *, topology_size: int) -> int:
    """Normalize the finite path edge bound without accepting ``bool`` as int."""
    if max_steps is None:
        return topology_size
    if type(max_steps) is not int or max_steps < 0:
        raise ValueError("max_steps must be a non-negative built-in int or None")
    return max_steps
