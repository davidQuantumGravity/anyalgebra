"""Immutable declarations of explicit presentation-conversion topology.

This module records only directed conversion edges.  It does not select a
route, test admissibility, apply a callable, or certify an inverse; those are
separate later boundaries.  In particular, ordinary Python container shape is
never a presentation kind or a conversion rule.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import cast

from anyalgebra.core.errors import AnyAlgebraError


_MAX_METADATA = 64
_MAX_EDGES = 512


class ConversionDefinitionError(AnyAlgebraError, ValueError):
    """A presentation-kind, conversion, or topology declaration was malformed."""

    def __init__(self, *, field: str, reason: str, index: int | None = None) -> None:
        """Store deterministic, address-free declaration diagnostics."""
        self.field = field
        self.reason = reason
        self.index = index
        location = field if index is None else f"{field}[{index}]"
        super().__init__(f"invalid conversion declaration {location}: {reason}")


def _require_name(value: object, *, field: str) -> str:
    """Require a stable non-empty built-in string without outer whitespace."""
    if type(value) is not str or not value or value != value.strip():
        raise ConversionDefinitionError(
            field=field, reason="must be a non-empty trimmed built-in str"
        )
    return value


def _safe_exception_name(error: Exception) -> str:
    """Return a payload-free stable type label for a user iterator failure."""
    name = type(error).__name__
    return name if type(name) is str and name.isidentifier() else "exception"


def _bounded_strings(value: object, *, field: str) -> tuple[str, ...]:
    """Snapshot bounded inert string metadata once with sanitized failures."""
    if isinstance(value, str | bytes):
        raise ConversionDefinitionError(field=field, reason="must be an iterable")
    failure: str | None
    try:
        iterator = iter(cast(Iterable[object], value))
    except Exception as error:
        iterator = None
        failure = _safe_exception_name(error)
    else:
        failure = None
    if iterator is None:
        raise ConversionDefinitionError(
            field=field, reason=f"could not be iterated ({failure})"
        )
    items: list[str] = []
    failed = False
    try:
        for index, item in enumerate(iterator):
            if index >= _MAX_METADATA:
                raise ConversionDefinitionError(
                    field=field, reason=f"exceeds maximum {_MAX_METADATA} entries"
                )
            items.append(_require_name(item, field=field))
    except ConversionDefinitionError:
        raise
    except Exception as error:
        failed = True
        failure = _safe_exception_name(error)
    if failed:
        raise ConversionDefinitionError(
            field=field, reason=f"could not be iterated ({failure})"
        )
    if len(set(items)) != len(items):
        raise ConversionDefinitionError(
            field=field, reason="contains duplicate entries"
        )
    return tuple(items)


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class PresentationKind:
    """One explicit, stable presentation-kind identity.

    Exact stable IDs compare equal across independent reconstruction.  This is
    deliberately semantic metadata rather than an inference from a Python
    container, class, or object identity.
    """

    id: str
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked construction; use :meth:`from_id`."""
        del args, kwargs
        raise ConversionDefinitionError(
            field="presentation_kind", reason="use PresentationKind.from_id"
        )

    @classmethod
    def from_id(cls, id: object) -> PresentationKind:
        """Create one explicit presentation-kind identity."""
        if cls is not PresentationKind:
            raise ConversionDefinitionError(
                field="presentation_kind",
                reason="factory requires exact PresentationKind",
            )
        kind = object.__new__(cls)
        object.__setattr__(kind, "id", _require_name(id, field="id"))
        return kind

    def __eq__(self, other: object) -> bool:
        """Compare only the explicit stable kind identity."""
        return type(other) is PresentationKind and self.id == other.id

    def __repr__(self) -> str:
        """Expose only stable kind metadata, never incidental addresses."""
        return f"PresentationKind(id={self.id!r})"


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class Conversion:
    """One declared directed conversion edge without execution or certification."""

    id: str
    source_kind: PresentationKind
    target_kind: PresentationKind
    forward: Callable[..., object]
    admissibility: Callable[..., object] | None
    inverse: Callable[..., object] | None
    inverse_direction: str | None
    cost: int
    guarantees: tuple[str, ...]
    validity_domain: str
    round_trip_id: str | None
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked construction; use :meth:`from_callable`."""
        del args, kwargs
        raise ConversionDefinitionError(
            field="conversion", reason="use Conversion.from_callable"
        )

    @classmethod
    def from_callable(
        cls,
        id: object,
        source_kind: object,
        target_kind: object,
        forward: object,
        *,
        admissibility: object | None = None,
        inverse: object | None = None,
        cost: object = 1,
        guarantees: object = (),
        validity_domain: object = "all",
        round_trip_id: object | None = None,
    ) -> Conversion:
        """Freeze one explicit edge after static, non-executing validation."""
        if cls is not Conversion:
            raise ConversionDefinitionError(
                field="conversion", reason="factory requires exact Conversion"
            )
        stable_id = _require_name(id, field="id")
        if type(source_kind) is not PresentationKind:
            raise ConversionDefinitionError(
                field="source_kind", reason="must be an exact PresentationKind"
            )
        if type(target_kind) is not PresentationKind:
            raise ConversionDefinitionError(
                field="target_kind", reason="must be an exact PresentationKind"
            )
        if not callable(forward):
            raise ConversionDefinitionError(field="forward", reason="must be callable")
        if admissibility is not None and not callable(admissibility):
            raise ConversionDefinitionError(
                field="admissibility", reason="must be None or callable"
            )
        if inverse is not None and not callable(inverse):
            raise ConversionDefinitionError(
                field="inverse", reason="must be None or callable"
            )
        declared_inverse_direction = None if inverse is None else "target_to_source"
        if type(cost) is not int or cost < 0:
            raise ConversionDefinitionError(
                field="cost", reason="must be a non-negative built-in int"
            )
        domain = _require_name(validity_domain, field="validity_domain")
        if inverse is None:
            if round_trip_id is not None:
                raise ConversionDefinitionError(
                    field="round_trip_id", reason="requires an inverse callable"
                )
            declared_round_trip_id = None
        elif round_trip_id is None:
            declared_round_trip_id = stable_id
        else:
            declared_round_trip_id = _require_name(round_trip_id, field="round_trip_id")
        declared_guarantees = _bounded_strings(guarantees, field="guarantees")

        conversion = object.__new__(cls)
        object.__setattr__(conversion, "id", stable_id)
        object.__setattr__(conversion, "source_kind", source_kind)
        object.__setattr__(conversion, "target_kind", target_kind)
        object.__setattr__(conversion, "forward", forward)
        object.__setattr__(conversion, "admissibility", admissibility)
        object.__setattr__(conversion, "inverse", inverse)
        object.__setattr__(conversion, "inverse_direction", declared_inverse_direction)
        object.__setattr__(conversion, "cost", cost)
        object.__setattr__(conversion, "guarantees", declared_guarantees)
        object.__setattr__(conversion, "validity_domain", domain)
        object.__setattr__(conversion, "round_trip_id", declared_round_trip_id)
        return conversion

    @property
    def stable_id(self) -> str:
        """Return the declared edge identity without inspecting callables."""
        return self.id

    def __eq__(self, other: object) -> bool:
        """Keep executable declarations identity-only and unhashable."""
        return self is other

    def __repr__(self) -> str:
        """Render stable route metadata without endpoint or callable reprs."""
        return (
            f"Conversion(id={self.id!r}, cost={self.cost}, "
            f"has_inverse={self.inverse is not None})"
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ConversionGraph:
    """Persistent declaration-order topology of explicit conversion edges."""

    _edges: tuple[Conversion, ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked topology; use :meth:`empty`."""
        del args, kwargs
        raise ConversionDefinitionError(
            field="graph", reason="use ConversionGraph.empty"
        )

    @classmethod
    def empty(cls) -> ConversionGraph:
        """Create an empty immutable conversion topology."""
        if cls is not ConversionGraph:
            raise ConversionDefinitionError(
                field="graph", reason="factory requires exact ConversionGraph"
            )
        graph = object.__new__(cls)
        object.__setattr__(graph, "_edges", ())
        return graph

    @classmethod
    def from_conversions(cls, conversions: object) -> ConversionGraph:
        """Build one bounded topology from an ordered iterable of declarations."""
        if cls is not ConversionGraph:
            raise ConversionDefinitionError(
                field="graph", reason="factory requires exact ConversionGraph"
            )
        if isinstance(conversions, str | bytes):
            raise ConversionDefinitionError(
                field="conversions", reason="must be an iterable"
            )
        failure: str | None
        try:
            iterator = iter(cast(Iterable[object], conversions))
        except Exception as error:
            iterator = None
            failure = _safe_exception_name(error)
        else:
            failure = None
        if iterator is None:
            raise ConversionDefinitionError(
                field="conversions", reason=f"could not be iterated ({failure})"
            )
        graph = cls.empty()
        failed = False
        try:
            for index, conversion in enumerate(iterator):
                if index >= _MAX_EDGES:
                    raise ConversionDefinitionError(
                        field="conversions",
                        reason=f"exceeds maximum {_MAX_EDGES} entries",
                    )
                graph = graph.with_conversion(conversion)
        except ConversionDefinitionError:
            raise
        except Exception as error:
            failed = True
            failure = _safe_exception_name(error)
        if failed:
            raise ConversionDefinitionError(
                field="conversions", reason=f"could not be iterated ({failure})"
            )
        return graph

    @property
    def edges(self) -> tuple[Conversion, ...]:
        """Return declared edges in literal registration order."""
        return self._edges

    @property
    def topology(self) -> tuple[Conversion, ...]:
        """Alias the immutable edge declaration order for later planners."""
        return self._edges

    def with_conversion(self, conversion: object) -> ConversionGraph:
        """Return a new graph with one explicit edge, never mutating this graph."""
        if type(conversion) is not Conversion:
            raise ConversionDefinitionError(
                field="conversion", reason="must be an exact Conversion"
            )
        if len(self._edges) >= _MAX_EDGES:
            raise ConversionDefinitionError(
                field="graph", reason=f"exceeds maximum {_MAX_EDGES} edges"
            )
        for existing in self._edges:
            if existing.stable_id == conversion.stable_id:
                raise ConversionDefinitionError(
                    field="conversion", reason="duplicate stable id"
                )
        graph = object.__new__(type(self))
        object.__setattr__(graph, "_edges", (*self._edges, conversion))
        return graph

    def outgoing(self, kind: object) -> tuple[Conversion, ...]:
        """Inspect direct outgoing declarations without selecting a route."""
        if type(kind) is not PresentationKind:
            raise ConversionDefinitionError(
                field="kind", reason="must be an exact PresentationKind"
            )
        return tuple(edge for edge in self._edges if edge.source_kind == kind)

    def incoming(self, kind: object) -> tuple[Conversion, ...]:
        """Inspect direct incoming declarations without selecting a route."""
        if type(kind) is not PresentationKind:
            raise ConversionDefinitionError(
                field="kind", reason="must be an exact PresentationKind"
            )
        return tuple(edge for edge in self._edges if edge.target_kind == kind)

    def neighbors(self, kind: object) -> tuple[PresentationKind, ...]:
        """Inspect ordered direct target kinds; parallel edges remain visible."""
        return tuple(edge.target_kind for edge in self.outgoing(kind))

    def __eq__(self, other: object) -> bool:
        """Treat topology declarations as identity-only executable records."""
        return self is other

    def __repr__(self) -> str:
        """Expose only deterministic edge-count metadata."""
        return f"ConversionGraph(edge_count={len(self._edges)})"
