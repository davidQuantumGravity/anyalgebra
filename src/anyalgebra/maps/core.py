"""Explicit, record-only map declarations with literal parent endpoints.

This v0.0 slice records declared callable maps and inert metadata only.  It
does not apply a callable, infer conversions from Python shape, establish
linearity/multilinearity, check preservation, construct inverse maps, plan
routes, adapt backends, or change bases.  Those obligations remain separate
later tasks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Self, cast

from anyalgebra.algebra.multilinear import FiniteMultilinearStructure
from anyalgebra.algebra.tensor import TensorProductParent
from anyalgebra.core.domains import Domain
from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.modules import FreeModule
from anyalgebra.core.parents import FiniteCarrier
from anyalgebra.structures.signatures import OperationSymbol, RelationSymbol
from anyalgebra.structures.structure import Structure


class MapDefinitionError(AnyAlgebraError, ValueError):
    """A map record's endpoint or inert declaration metadata was malformed."""

    def __init__(self, *, field: str, reason: str, index: int | None = None) -> None:
        """Retain deterministic location data without rendering caller values."""
        self.field = field
        self.reason = reason
        self.index = index
        location = field if index is None else f"{field}[{index}]"
        super().__init__(f"invalid map {location}: {reason}")


_MAX_GUARANTEES = 64
_MAX_PRESERVED_SYMBOLS = 64
_MAX_MULTILINEAR_SOURCES = 512


def _exception_name(error: Exception) -> str:
    """Return safe exception type metadata without retaining payload text."""
    name = type(error).__name__
    return name if name.isidentifier() else "exception"


def _bounded_snapshot(
    values: object, *, field: str, maximum: int
) -> tuple[object, ...]:
    """Consume at most one item beyond a declared metadata bound."""
    iterator_error: str | None = None
    try:
        iterator = iter(cast(Iterable[object], values))
    except Exception as error:
        iterator_error = _exception_name(error)
    if iterator_error is not None:
        raise MapDefinitionError(
            field=field, reason=f"must be an iterable ({iterator_error})"
        )
    snapshot: list[object] = []
    for _ in range(maximum + 1):
        iteration_error: str | None = None
        try:
            snapshot.append(next(iterator))
        except StopIteration:
            break
        except Exception as error:
            iteration_error = _exception_name(error)
        if iteration_error is not None:
            raise MapDefinitionError(
                field=field, reason=f"could not be iterated ({iteration_error})"
            )
    if len(snapshot) > maximum:
        raise MapDefinitionError(field=field, reason="exceeds the declared maximum")
    return tuple(snapshot)


def _is_declared_parent(candidate: object) -> bool:
    """Recognize only current exact parent categories, never shape-like values."""
    if type(candidate) in (
        FreeModule,
        FiniteCarrier,
        TensorProductParent,
        Structure,
        FiniteMultilinearStructure,
    ):
        return True
    if isinstance(candidate, type):
        return False
    try:
        return (
            isinstance(candidate, Domain)
            and callable(candidate.normalize)
            and callable(candidate.element)
        )
    except Exception:
        return False


def _require_parent(value: object, *, field: str, index: int | None = None) -> object:
    """Require one current declared parent without coercing or rebuilding it."""
    if not _is_declared_parent(value):
        raise MapDefinitionError(
            field=field, reason="must be a supported declared parent", index=index
        )
    return value


def _require_callable(function: object) -> Callable[..., object]:
    """Check callable metadata but never execute it in this record-only slice."""
    if not callable(function):
        raise MapDefinitionError(field="function", reason="must be callable")
    return cast(Callable[..., object], function)


def _string_metadata(values: object, *, field: str) -> tuple[str, ...]:
    """Freeze bounded unique exact string metadata in declared order."""
    if isinstance(values, str | bytes):
        raise MapDefinitionError(field=field, reason="must be an iterable of strings")
    snapshot = _bounded_snapshot(values, field=field, maximum=_MAX_GUARANTEES)
    validated: list[str] = []
    for index, value in enumerate(snapshot):
        if type(value) is not str or not value or value != value.strip():
            raise MapDefinitionError(
                field=field,
                reason="must contain non-empty trimmed exact built-in str values",
                index=index,
            )
        if value in validated:
            raise MapDefinitionError(
                field=field, reason="contains a duplicate", index=index
            )
        validated.append(value)
    return tuple(validated)


def _operation_metadata(values: object) -> tuple[OperationSymbol, ...]:
    """Freeze bounded exact operation-preservation claims without proof."""
    snapshot = _bounded_snapshot(
        values, field="preserved_operations", maximum=_MAX_PRESERVED_SYMBOLS
    )
    validated: list[OperationSymbol] = []
    for index, value in enumerate(snapshot):
        if type(value) is not OperationSymbol:
            raise MapDefinitionError(
                field="preserved_operations",
                reason="must contain exact OperationSymbol values",
                index=index,
            )
        if any(prior == value for prior in validated):
            raise MapDefinitionError(
                field="preserved_operations", reason="contains a duplicate", index=index
            )
        validated.append(value)
    return tuple(validated)


def _relation_metadata(values: object) -> tuple[RelationSymbol, ...]:
    """Freeze bounded exact relation-preservation claims without proof."""
    snapshot = _bounded_snapshot(
        values, field="preserved_relations", maximum=_MAX_PRESERVED_SYMBOLS
    )
    validated: list[RelationSymbol] = []
    for index, value in enumerate(snapshot):
        if type(value) is not RelationSymbol:
            raise MapDefinitionError(
                field="preserved_relations",
                reason="must contain exact RelationSymbol values",
                index=index,
            )
        if any(prior == value for prior in validated):
            raise MapDefinitionError(
                field="preserved_relations", reason="contains a duplicate", index=index
            )
        validated.append(value)
    return tuple(validated)


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class Map:
    """One callable-backed record with exact source/target parent declarations."""

    source: object
    target: object
    function: Callable[..., object]
    guarantees: tuple[str, ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked records; use :meth:`from_callable`."""
        del args, kwargs
        raise MapDefinitionError(field="map", reason="use Map.from_callable")

    @classmethod
    def from_callable(
        cls,
        source: object,
        target: object,
        function: object,
        *,
        guarantees: object = (),
    ) -> Self:
        """Freeze literal endpoints and inert guarantees without map application."""
        source = _require_parent(source, field="source")
        target = _require_parent(target, field="target")
        function = _require_callable(function)
        guarantees = _string_metadata(guarantees, field="guarantees")
        record = object.__new__(cls)
        object.__setattr__(record, "source", source)
        object.__setattr__(record, "target", target)
        object.__setattr__(record, "function", function)
        object.__setattr__(record, "guarantees", guarantees)
        return record

    def __eq__(self, other: object) -> bool:
        """Treat executable declarations as identity-only semantic records."""
        return self is other

    def __repr__(self) -> str:
        """Avoid callable and endpoint reprs, which can contain unstable addresses."""
        return f"{type(self).__name__}(guarantee_count={len(self.guarantees)})"


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class LinearMap(Map):
    """A map record whose caller declares linearity; no proof is performed here."""

    @classmethod
    def from_callable(
        cls,
        source: object,
        target: object,
        function: object,
        *,
        guarantees: object = (),
    ) -> LinearMap:
        """Freeze an exact module-to-module linearity declaration only."""
        if type(source) is not FreeModule:
            raise MapDefinitionError(
                field="source", reason="must be an exact FreeModule"
            )
        if type(target) is not FreeModule:
            raise MapDefinitionError(
                field="target", reason="must be an exact FreeModule"
            )
        function = _require_callable(function)
        guarantees = _string_metadata(guarantees, field="guarantees")
        record = object.__new__(cls)
        object.__setattr__(record, "source", source)
        object.__setattr__(record, "target", target)
        object.__setattr__(record, "function", function)
        object.__setattr__(record, "guarantees", guarantees)
        return record


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class MultilinearMap:
    """One ordered-source callable record whose multilinearity is only declared."""

    sources: tuple[object, ...]
    target: object
    function: Callable[..., object]
    guarantees: tuple[str, ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked records; use :meth:`from_callable`."""
        del args, kwargs
        raise MapDefinitionError(
            field="multilinear_map", reason="use MultilinearMap.from_callable"
        )

    @classmethod
    def from_callable(
        cls,
        sources: object,
        target: object,
        function: object,
        *,
        guarantees: object = (),
    ) -> MultilinearMap:
        """Freeze nullary through bounded high-arity exact source parent tuples."""
        if type(target) is not FreeModule:
            raise MapDefinitionError(
                field="target", reason="must be an exact FreeModule"
            )
        function = _require_callable(function)
        raw_sources = _bounded_snapshot(
            sources, field="sources", maximum=_MAX_MULTILINEAR_SOURCES
        )
        validated_sources: list[FreeModule] = []
        for index, value in enumerate(raw_sources):
            if type(value) is not FreeModule:
                raise MapDefinitionError(
                    field="sources",
                    reason="must contain exact FreeModule values",
                    index=index,
                )
            validated_sources.append(value)
        guarantees = _string_metadata(guarantees, field="guarantees")
        record = object.__new__(cls)
        object.__setattr__(record, "sources", tuple(validated_sources))
        object.__setattr__(record, "target", target)
        object.__setattr__(record, "function", function)
        object.__setattr__(record, "guarantees", guarantees)
        return record

    def __eq__(self, other: object) -> bool:
        """Treat executable declarations as identity-only semantic records."""
        return self is other

    def __repr__(self) -> str:
        """Avoid callable and endpoint reprs, which can contain unstable addresses."""
        return (
            f"MultilinearMap(arity={len(self.sources)}, "
            f"guarantee_count={len(self.guarantees)})"
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class Morphism:
    """A structure-map record with unproved operation/relation preservation claims.

    The documented ``Morphism`` name owns this record seam.  Homomorphism and
    isomorphism validation reports are intentionally deferred to V00-050.
    """

    source: Structure
    target: Structure
    components: tuple[Map, ...]
    guarantees: tuple[str, ...]
    preserved_operations: tuple[OperationSymbol, ...]
    preserved_relations: tuple[RelationSymbol, ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked records; use :meth:`from_components`."""
        del args, kwargs
        raise MapDefinitionError(
            field="morphism",
            reason="use Morphism.from_components or one-sorted Morphism.from_callable",
        )

    @classmethod
    def from_components(
        cls,
        source: object,
        target: object,
        components: object,
        *,
        guarantees: object = (),
        preserved_operations: object = (),
        preserved_relations: object = (),
    ) -> Morphism:
        """Freeze one literal carrier-map component per declared signature sort."""
        if type(source) is not Structure:
            raise MapDefinitionError(
                field="source", reason="must be an exact Structure"
            )
        if type(target) is not Structure:
            raise MapDefinitionError(
                field="target", reason="must be an exact Structure"
            )
        if source.signature != target.signature:
            raise MapDefinitionError(
                field="target", reason="must have the same structural signature"
            )
        raw_components = _bounded_snapshot(
            components, field="components", maximum=len(source.carriers)
        )
        if len(raw_components) != len(source.carriers):
            raise MapDefinitionError(
                field="components",
                reason="must contain one component per signature sort",
            )
        validated: list[Map] = []
        for index, (component, source_carrier, target_carrier) in enumerate(
            zip(raw_components, source.carriers, target.carriers, strict=True)
        ):
            if type(component) is not Map:
                raise MapDefinitionError(
                    field="components",
                    reason="must contain exact Map values",
                    index=index,
                )
            if (
                component.source is not source_carrier
                or component.target is not target_carrier
            ):
                raise MapDefinitionError(
                    field="components",
                    reason=(
                        "must retain the literal source and target carrier "
                        "in sort order"
                    ),
                    index=index,
                )
            validated.append(component)
        guarantees = _string_metadata(guarantees, field="guarantees")
        operations = _operation_metadata(preserved_operations)
        relations = _relation_metadata(preserved_relations)
        record = object.__new__(cls)
        object.__setattr__(record, "source", source)
        object.__setattr__(record, "target", target)
        object.__setattr__(record, "components", tuple(validated))
        object.__setattr__(record, "guarantees", guarantees)
        object.__setattr__(record, "preserved_operations", operations)
        object.__setattr__(record, "preserved_relations", relations)
        return record

    @classmethod
    def from_callable(
        cls,
        source: object,
        target: object,
        function: object,
        *,
        guarantees: object = (),
        preserved_operations: object = (),
        preserved_relations: object = (),
    ) -> Morphism:
        """Build the sound one-sorted convenience form without sort guessing."""
        if type(source) is not Structure:
            raise MapDefinitionError(
                field="source", reason="must be an exact Structure"
            )
        if type(target) is not Structure:
            raise MapDefinitionError(
                field="target", reason="must be an exact Structure"
            )
        if source.signature != target.signature:
            raise MapDefinitionError(
                field="target", reason="must have the same structural signature"
            )
        if len(source.carriers) != 1 or len(target.carriers) != 1:
            raise MapDefinitionError(
                field="source" if len(source.carriers) != 1 else "target",
                reason=(
                    "one-sorted convenience requires exactly one sort on each endpoint"
                ),
            )
        component = Map.from_callable(source.carriers[0], target.carriers[0], function)
        return cls.from_components(
            source,
            target,
            (component,),
            guarantees=guarantees,
            preserved_operations=preserved_operations,
            preserved_relations=preserved_relations,
        )

    def __eq__(self, other: object) -> bool:
        """Treat a declared family of executable components as identity-only."""
        return self is other

    def __repr__(self) -> str:
        """Avoid callable and parent reprs, which can expose addresses."""
        return f"Morphism(component_count={len(self.components)})"
