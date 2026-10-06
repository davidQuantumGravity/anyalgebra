"""Deterministic finite many-sorted substitution domains for exact checking."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import product
from typing import cast

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.structures.structure import Structure
from anyalgebra.structures.terms import Variable


_MAX_VARIABLES = 512
_MAX_ASSIGNMENTS = 65_536


class SubstitutionDomainError(AnyAlgebraError, ValueError):
    """A finite substitution-domain declaration was malformed or too large."""

    def __init__(
        self,
        reason: str,
        *,
        field: str | None = None,
        index: int | None = None,
        kind: str | None = None,
        observed: int | None = None,
        maximum: int | None = None,
    ) -> None:
        """Retain stable coordinates and bounds without rendering caller values."""
        self.reason = reason
        self.field = field
        self.index = index
        self.kind = kind
        self.observed = observed
        self.maximum = maximum
        detail = reason if field is None else f"{field}: {reason}"
        if index is not None:
            detail = f"{detail}; index={index}"
        if kind is not None:
            detail = f"{detail}; kind={kind}"
        if observed is not None and maximum is not None:
            detail = f"{detail}; observed={observed}; maximum={maximum}"
        super().__init__(f"invalid finite substitution domain: {detail}")


def _exception_name(error: Exception) -> str:
    """Return a sanitized exception type for hostile one-pass inputs."""
    name = type(error).__name__
    return name if type(name) is str and name.isidentifier() else "exception"


def _snapshot(
    value: object,
    *,
    field: str,
    maximum: int = _MAX_VARIABLES,
    mapping_items: bool = False,
) -> tuple[object, ...]:
    """Consume at most one bounded inert declaration iterable exactly once."""
    if isinstance(value, str | bytes):
        raise SubstitutionDomainError(
            "must be an iterable, not a raw string", field=field
        )
    source: object = value
    if mapping_items and isinstance(value, Mapping):
        try:
            source = value.items()
        except Exception as error:
            raise SubstitutionDomainError(
                f"could not be snapshotted ({_exception_name(error)})", field=field
            ) from None
    try:
        iterator = iter(cast(Iterable[object], source))
    except Exception as error:
        raise SubstitutionDomainError(
            f"could not be snapshotted ({_exception_name(error)})", field=field
        ) from None
    result: list[object] = []
    for _ in range(maximum + 1):
        try:
            result.append(next(iterator))
        except StopIteration:
            break
        except Exception as error:
            raise SubstitutionDomainError(
                f"could not be snapshotted ({_exception_name(error)})", field=field
            ) from None
    if len(result) > maximum:
        raise SubstitutionDomainError(
            "exceeds declared maximum",
            field=field,
            kind=field,
            observed=len(result),
            maximum=maximum,
        )
    return tuple(result)


def _variables(value: object) -> tuple[Variable, ...]:
    """Validate one ordered, duplicate-free exact variable declaration."""
    raw = _snapshot(value, field="variables")
    variables: list[Variable] = []
    for index, item in enumerate(raw):
        if type(item) is not Variable:
            raise SubstitutionDomainError(
                "must contain exact Variable values", field="variables", index=index
            )
        if any(item == prior for prior in variables):
            raise SubstitutionDomainError(
                "contains duplicate structurally equal variables",
                field="variables",
                index=index,
            )
        variables.append(item)
    return tuple(variables)


def _bindings(value: object) -> tuple[tuple[Sort, FiniteCarrier], ...]:
    """Validate bounded explicit sort/carrier bindings without dict coercion."""
    raw = _snapshot(value, field="carriers", mapping_items=True)
    bindings: list[tuple[Sort, FiniteCarrier]] = []
    for index, item in enumerate(raw):
        if type(item) is not tuple or len(item) != 2:
            raise SubstitutionDomainError(
                "entries must be exact (Sort, FiniteCarrier) pairs",
                field="carriers",
                index=index,
            )
        sort, carrier = item
        if type(sort) is not Sort:
            raise SubstitutionDomainError(
                "binding sort must be an exact Sort", field="carriers", index=index
            )
        if type(carrier) is not FiniteCarrier:
            raise SubstitutionDomainError(
                "binding carrier must be an exact FiniteCarrier",
                field="carriers",
                index=index,
            )
        if carrier.sort != sort:
            raise SubstitutionDomainError(
                "carrier sort does not match its binding sort",
                field="carriers",
                index=index,
            )
        if any(sort == prior_sort for prior_sort, _ in bindings):
            raise SubstitutionDomainError(
                "contains duplicate structurally equal sort bindings",
                field="carriers",
                index=index,
            )
        bindings.append((sort, carrier))
    return tuple(bindings)


def _canonical_bindings(
    variables: tuple[Variable, ...], bindings: tuple[tuple[Sort, FiniteCarrier], ...]
) -> tuple[tuple[Sort, FiniteCarrier], ...]:
    """Align one binding per required sort in first variable-declaration order."""
    required: list[Sort] = []
    for variable in variables:
        if not any(variable.sort == prior for prior in required):
            required.append(variable.sort)
    canonical: list[tuple[Sort, FiniteCarrier]] = []
    for sort in required:
        matching = tuple(
            carrier for bound_sort, carrier in bindings if bound_sort == sort
        )
        if not matching:
            raise SubstitutionDomainError(
                "missing required carrier binding", field="carriers"
            )
        assert len(matching) == 1
        canonical.append((sort, matching[0]))
    if len(bindings) != len(canonical):
        raise SubstitutionDomainError(
            "contains an extra sort binding", field="carriers"
        )
    return tuple(canonical)


def _assignment_count(
    variables: tuple[Variable, ...], bindings: tuple[tuple[Sort, FiniteCarrier], ...]
) -> int:
    """Compute the exact Cartesian cardinality before any substitution is yielded."""
    carrier_for_sort = tuple(bindings)
    count = 1
    for variable in variables:
        carrier = next(
            carrier for sort, carrier in carrier_for_sort if sort == variable.sort
        )
        count *= len(carrier)
        if count > _MAX_ASSIGNMENTS:
            raise SubstitutionDomainError(
                "Cartesian assignment count exceeds declared maximum",
                field="variables",
                kind="assignments",
                observed=count,
                maximum=_MAX_ASSIGNMENTS,
            )
    return count


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class Substitution(Mapping[Variable, object]):
    """One immutable sorted assignment directly consumable by ``evaluate_term``."""

    variables: tuple[Variable, ...]
    member_values: tuple[object, ...]
    carrier_indices: tuple[int, ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked mappings; only a finite domain creates assignments."""
        del args, kwargs
        raise SubstitutionDomainError("use FiniteSubstitutionDomain enumeration")

    @classmethod
    def _create(
        cls,
        variables: tuple[Variable, ...],
        member_values: tuple[object, ...],
        carrier_indices: tuple[int, ...],
    ) -> Substitution:
        if cls is not Substitution:
            raise SubstitutionDomainError("factory requires exact Substitution")
        value = object.__new__(cls)
        object.__setattr__(value, "variables", variables)
        object.__setattr__(value, "member_values", member_values)
        object.__setattr__(value, "carrier_indices", carrier_indices)
        return value

    def __getitem__(self, variable: Variable) -> object:
        """Resolve a variable structurally in declaration order without hashing."""
        if type(variable) is not Variable:
            raise KeyError(variable)
        for candidate, value in zip(self.variables, self.member_values, strict=True):
            if candidate == variable:
                return value
        raise KeyError(variable)

    def __iter__(self) -> Iterator[Variable]:
        """Iterate variables in the domain's declared order."""
        return iter(self.variables)

    def __len__(self) -> int:
        """Return the exact finite number of variable bindings."""
        return len(self.variables)

    def __eq__(self, other: object) -> bool:
        """Keep arbitrary-member assignments conservative and identity-only."""
        return self is other

    def __repr__(self) -> str:
        """Expose only safe shape/index metadata, never arbitrary member reprs."""
        return (
            "Substitution("
            f"variable_count={len(self.variables)}, "
            f"carrier_indices={self.carrier_indices})"
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class FiniteSubstitutionDomain:
    """A repeatable exact Cartesian domain over declared sorted variables."""

    variables: tuple[Variable, ...]
    carrier_bindings: tuple[tuple[Sort, FiniteCarrier], ...]
    assignment_count: int
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject unchecked domains; use a public canonicalizing factory."""
        del args, kwargs
        raise SubstitutionDomainError(
            "use FiniteSubstitutionDomain.from_bindings or from_structure"
        )

    @classmethod
    def _create(
        cls,
        variables: tuple[Variable, ...],
        bindings: tuple[tuple[Sort, FiniteCarrier], ...],
    ) -> FiniteSubstitutionDomain:
        if cls is not FiniteSubstitutionDomain:
            raise SubstitutionDomainError(
                "factory requires exact FiniteSubstitutionDomain"
            )
        value = object.__new__(cls)
        object.__setattr__(value, "variables", variables)
        object.__setattr__(value, "carrier_bindings", bindings)
        object.__setattr__(
            value, "assignment_count", _assignment_count(variables, bindings)
        )
        return value

    @classmethod
    def from_bindings(
        cls, variables: object, carriers: object
    ) -> FiniteSubstitutionDomain:
        """Freeze explicit required-sort carrier bindings in variable order."""
        if cls is not FiniteSubstitutionDomain:
            raise SubstitutionDomainError(
                "factory requires exact FiniteSubstitutionDomain"
            )
        declarations = _variables(variables)
        canonical = _canonical_bindings(declarations, _bindings(carriers))
        return cls._create(declarations, canonical)

    @classmethod
    def from_structure(
        cls, variables: object, structure: object
    ) -> FiniteSubstitutionDomain:
        """Use the exact carrier declarations already frozen by one structure."""
        if cls is not FiniteSubstitutionDomain:
            raise SubstitutionDomainError(
                "factory requires exact FiniteSubstitutionDomain"
            )
        if type(structure) is not Structure:
            raise SubstitutionDomainError(
                "must be an exact Structure", field="structure"
            )
        return cls.from_bindings(
            variables,
            tuple(zip(structure.signature.sorts, structure.carriers, strict=True)),
        )

    def _carriers_by_variable(self) -> tuple[FiniteCarrier, ...]:
        """Resolve retained canonical carriers in exact variable declaration order."""
        return tuple(
            next(
                carrier
                for sort, carrier in self.carrier_bindings
                if sort == variable.sort
            )
            for variable in self.variables
        )

    def __iter__(self) -> Iterator[Substitution]:
        """Yield each Cartesian assignment once, rightmost variables fastest."""
        carriers = self._carriers_by_variable()
        for indices in product(*(range(len(carrier)) for carrier in carriers)):
            yield Substitution._create(
                self.variables,
                tuple(
                    carrier.items[index]
                    for carrier, index in zip(carriers, indices, strict=True)
                ),
                indices,
            )

    def __len__(self) -> int:
        """Return the preflighted exact Cartesian assignment count."""
        return self.assignment_count

    def __getitem__(self, index: int) -> Substitution:
        """Return one deterministic assignment by its lexicographic index."""
        if type(index) is not int:
            raise TypeError("substitution index must be an exact built-in int")
        normalized = index + self.assignment_count if index < 0 else index
        if normalized < 0 or normalized >= self.assignment_count:
            raise IndexError("substitution index out of range")
        carriers = self._carriers_by_variable()
        indices = [0] * len(carriers)
        remainder = normalized
        for position in range(len(carriers) - 1, -1, -1):
            remainder, indices[position] = divmod(remainder, len(carriers[position]))
        frozen_indices = tuple(indices)
        return Substitution._create(
            self.variables,
            tuple(
                carrier.items[item_index]
                for carrier, item_index in zip(carriers, frozen_indices, strict=True)
            ),
            frozen_indices,
        )

    def __repr__(self) -> str:
        """Render only counts, never arbitrary carrier item or label values."""
        return (
            "FiniteSubstitutionDomain("
            f"variable_count={len(self.variables)}, "
            f"assignment_count={self.assignment_count})"
        )
