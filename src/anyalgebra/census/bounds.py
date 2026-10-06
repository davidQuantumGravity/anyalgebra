"""Exact resource ceilings and canonical ordering for finite censuses."""

from __future__ import annotations

from dataclasses import dataclass

from anyalgebra.core.errors import AnyAlgebraError


_MAX_LIMIT = (1 << 63) - 1


class CensusBoundsError(AnyAlgebraError, ValueError):
    """A census bound or ordering record was malformed."""

    def __init__(self, *, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid census bounds {field}: {reason}")


class CensusBoundExceeded(CensusBoundsError):
    """A fixed-cost preflight exceeded one explicitly declared ceiling."""

    def __init__(self, *, field: str, limit: int, requested: int) -> None:
        self.limit = limit
        self.requested = requested
        AnyAlgebraError.__init__(
            self,
            f"census preflight {field} exceeds declared limit {limit}: "
            f"requested {requested}",
        )
        self.field = field
        self.reason = "declared limit exceeded"


def _limit(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise CensusBoundsError(
            field=field,
            reason="must be a non-negative built-in int, excluding bool",
        )
    if value > _MAX_LIMIT:
        raise CensusBoundsError(field=field, reason="hard integer limit exceeded")
    return value


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class EnumerationBounds:
    """Hard semantic ceilings plus a separately labelled time observation."""

    max_candidates: int
    max_orbits: int
    max_work_units: int
    max_memory_bytes: int
    max_observed_milliseconds: int | None
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise CensusBoundsError(field="bounds", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("EnumerationBounds cannot be subclassed")

    @classmethod
    def create(
        cls,
        *,
        max_candidates: int,
        max_orbits: int,
        max_work_units: int,
        max_memory_bytes: int,
        max_observed_milliseconds: int | None = None,
    ) -> EnumerationBounds:
        if cls is not EnumerationBounds:
            raise CensusBoundsError(
                field="bounds", reason="factory requires exact type"
            )
        values = (
            ("max_candidates", _limit(max_candidates, "max_candidates")),
            ("max_orbits", _limit(max_orbits, "max_orbits")),
            ("max_work_units", _limit(max_work_units, "max_work_units")),
            ("max_memory_bytes", _limit(max_memory_bytes, "max_memory_bytes")),
        )
        observed = (
            None
            if max_observed_milliseconds is None
            else _limit(max_observed_milliseconds, "max_observed_milliseconds")
        )
        value = object.__new__(EnumerationBounds)
        for field, item in values:
            object.__setattr__(value, field, item)
        object.__setattr__(value, "max_observed_milliseconds", observed)
        return value

    def preflight(
        self,
        *,
        candidate_count: int,
        orbit_count: int,
        work_units: int,
        memory_bytes: int,
    ) -> None:
        """Check fixed-cost estimates in canonical field order before work."""
        checks = (
            ("candidate_count", candidate_count, self.max_candidates),
            ("orbit_count", orbit_count, self.max_orbits),
            ("work_units", work_units, self.max_work_units),
            ("memory_bytes", memory_bytes, self.max_memory_bytes),
        )
        for field, raw_requested, limit in checks:
            requested = _limit(raw_requested, field)
            if requested > limit:
                raise CensusBoundExceeded(field=field, limit=limit, requested=requested)

    def __eq__(self, other: object) -> bool:
        return type(other) is EnumerationBounds and (
            self.max_candidates,
            self.max_orbits,
            self.max_work_units,
            self.max_memory_bytes,
            self.max_observed_milliseconds,
        ) == (
            other.max_candidates,
            other.max_orbits,
            other.max_work_units,
            other.max_memory_bytes,
            other.max_observed_milliseconds,
        )

    def __repr__(self) -> str:
        return (
            "EnumerationBounds("
            f"max_candidates={self.max_candidates}, "
            f"max_orbits={self.max_orbits}, "
            f"max_work_units={self.max_work_units}, "
            f"max_memory_bytes={self.max_memory_bytes}, "
            f"max_observed_milliseconds={self.max_observed_milliseconds})"
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class CensusOrdering:
    """One versioned, fully explicit reference enumeration order."""

    candidate_order: str
    input_tuple_order: str
    constraint_order: str
    permutation_order: str
    output_digit_significance: str
    schema_version: int
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise CensusBoundsError(field="ordering", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("CensusOrdering cannot be subclassed")

    @classmethod
    def reference(cls) -> CensusOrdering:
        if cls is not CensusOrdering:
            raise CensusBoundsError(
                field="ordering", reason="factory requires exact type"
            )
        value = object.__new__(CensusOrdering)
        for field, item in (
            ("candidate_order", "lexicographic_flat_outputs"),
            ("input_tuple_order", "lexicographic_indices"),
            ("constraint_order", "canonical_constraint_key"),
            ("permutation_order", "lexicographic_images"),
            ("output_digit_significance", "first_cell_most_significant"),
            ("schema_version", 1),
        ):
            object.__setattr__(value, field, item)
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is CensusOrdering and (
            self.candidate_order,
            self.input_tuple_order,
            self.constraint_order,
            self.permutation_order,
            self.output_digit_significance,
            self.schema_version,
        ) == (
            other.candidate_order,
            other.input_tuple_order,
            other.constraint_order,
            other.permutation_order,
            other.output_digit_significance,
            other.schema_version,
        )

    def __repr__(self) -> str:
        return "CensusOrdering(reference-v1)"
