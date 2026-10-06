"""Allow-listed immutable constraints for finite operation-table censuses."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from anyalgebra.core.errors import AnyAlgebraError

from .spec import CensusSpecCore


_MAX_CONSTRAINTS = 256
_MAX_TERM_ARGUMENTS = 64
_MAX_TERM_DEPTH = 64
_MAX_TERM_NODES = 4_096
_IDENTITY_SIDES = frozenset(("left", "right", "two_sided"))


class ConstraintError(AnyAlgebraError, ValueError):
    """A declarative census constraint failed a stable validation boundary."""

    def __init__(self, *, field: str, reason: str, index: int | None = None) -> None:
        self.field = field
        self.reason = reason
        self.index = index
        location = field if index is None else f"{field}[{index}]"
        super().__init__(f"invalid census constraint {location}: {reason}")


def _index(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ConstraintError(
            field=field, reason="must be a non-negative built-in int, excluding bool"
        )
    return value


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class CensusTerm:
    """One variable, carrier constant, or application of the sole operation."""

    kind: str
    index: int | None
    arguments: tuple[CensusTerm, ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ConstraintError(field="term", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("CensusTerm cannot be subclassed")

    @classmethod
    def variable(cls, index: int) -> CensusTerm:
        return cls._create("variable", _index(index, "variable"), ())

    @classmethod
    def constant(cls, element: int) -> CensusTerm:
        return cls._create("constant", _index(element, "constant"), ())

    @classmethod
    def apply(cls, *arguments: CensusTerm) -> CensusTerm:
        if len(arguments) > _MAX_TERM_ARGUMENTS:
            raise ConstraintError(field="term", reason="operation arity limit exceeded")
        for index, argument in enumerate(arguments):
            if type(argument) is not CensusTerm:
                raise ConstraintError(
                    field="term",
                    reason="arguments must contain exact CensusTerm values",
                    index=index,
                )
        depth = 1 + max((_term_depth(argument) for argument in arguments), default=0)
        if depth > _MAX_TERM_DEPTH:
            raise ConstraintError(field="term", reason="term depth limit exceeded")
        size = 1 + sum(_term_size(argument) for argument in arguments)
        if size > _MAX_TERM_NODES:
            raise ConstraintError(field="term", reason="term node limit exceeded")
        return cls._create("apply", None, arguments)

    @classmethod
    def _create(
        cls, kind: str, index: int | None, arguments: tuple[CensusTerm, ...]
    ) -> CensusTerm:
        if cls is not CensusTerm:
            raise ConstraintError(field="term", reason="factory requires exact type")
        value = object.__new__(CensusTerm)
        object.__setattr__(value, "kind", kind)
        object.__setattr__(value, "index", index)
        object.__setattr__(value, "arguments", arguments)
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is CensusTerm and _term_key(self) == _term_key(other)

    def __repr__(self) -> str:
        return f"CensusTerm(kind={self.kind!r}, size={_term_size(self)})"


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class CensusConstraint:
    """One allow-listed, data-only predicate on a finite operation table."""

    kind: str
    parameters: tuple[tuple[str, int | str], ...]
    left: CensusTerm | None
    right: CensusTerm | None
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ConstraintError(field="constraint", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("CensusConstraint cannot be subclassed")

    @classmethod
    def _create(
        cls,
        kind: str,
        parameters: tuple[tuple[str, int | str], ...] = (),
        left: CensusTerm | None = None,
        right: CensusTerm | None = None,
    ) -> CensusConstraint:
        if cls is not CensusConstraint:
            raise ConstraintError(
                field="constraint", reason="factory requires exact type"
            )
        value = object.__new__(CensusConstraint)
        object.__setattr__(value, "kind", kind)
        object.__setattr__(value, "parameters", parameters)
        object.__setattr__(value, "left", left)
        object.__setattr__(value, "right", right)
        return value

    @classmethod
    def nullary_value(cls, *, element: int) -> CensusConstraint:
        return cls._create("nullary_value", (("element", _index(element, "element")),))

    @classmethod
    def identity(cls, *, element: int, side: str) -> CensusConstraint:
        if type(side) is not str or side not in _IDENTITY_SIDES:
            raise ConstraintError(field="side", reason="is not an allow-listed side")
        return cls._create(
            "identity", (("element", _index(element, "element")), ("side", side))
        )

    @classmethod
    def commutative(cls) -> CensusConstraint:
        return cls._create("commutative")

    @classmethod
    def idempotent(cls) -> CensusConstraint:
        return cls._create("idempotent")

    @classmethod
    def quasigroup(cls) -> CensusConstraint:
        return cls._create("quasigroup")

    @classmethod
    def equation(
        cls, left: CensusTerm, right: CensusTerm, *, variable_count: int
    ) -> CensusConstraint:
        if type(left) is not CensusTerm or type(right) is not CensusTerm:
            raise ConstraintError(
                field="equation", reason="sides must be exact CensusTerm values"
            )
        count = _index(variable_count, "variable_count")
        if count > _MAX_TERM_ARGUMENTS:
            raise ConstraintError(
                field="variable_count", reason="variable limit exceeded"
            )
        return cls._create(
            "equation", (("variable_count", count),), left=left, right=right
        )

    def __eq__(self, other: object) -> bool:
        return type(other) is CensusConstraint and _constraint_key(
            self
        ) == _constraint_key(other)

    def __repr__(self) -> str:
        return f"CensusConstraint(kind={self.kind!r})"


def _term_key(term: CensusTerm | None) -> tuple[object, ...]:
    if term is None:
        return ("none",)
    return (
        term.kind,
        -1 if term.index is None else term.index,
        tuple(_term_key(argument) for argument in term.arguments),
    )


def _term_size(term: CensusTerm) -> int:
    return 1 + sum(_term_size(argument) for argument in term.arguments)


def _term_depth(term: CensusTerm) -> int:
    return 1 + max((_term_depth(argument) for argument in term.arguments), default=0)


def _constraint_key(constraint: CensusConstraint) -> tuple[object, ...]:
    return (
        constraint.kind,
        constraint.parameters,
        _term_key(constraint.left),
        _term_key(constraint.right),
    )


def _validate_term(
    term: CensusTerm,
    core: CensusSpecCore,
    variable_count: int,
    *,
    constraint_index: int,
) -> None:
    if term.kind == "variable":
        assert term.index is not None
        if term.index >= variable_count:
            raise ConstraintError(
                field="constraints",
                reason="term variable is outside the declared variable range",
                index=constraint_index,
            )
        return
    if term.kind == "constant":
        assert term.index is not None
        if term.index >= core.carrier_size:
            raise ConstraintError(
                field="constraints",
                reason="term constant is outside the carrier range",
                index=constraint_index,
            )
        return
    if term.kind != "apply":
        raise ConstraintError(
            field="constraints", reason="unknown term kind", index=constraint_index
        )
    if len(term.arguments) != core.arity:
        raise ConstraintError(
            field="constraints",
            reason="term application has the wrong operation arity",
            index=constraint_index,
        )
    for argument in term.arguments:
        _validate_term(
            argument, core, variable_count, constraint_index=constraint_index
        )


def _parameter(constraint: CensusConstraint, name: str) -> int | str:
    for key, value in constraint.parameters:
        if key == name:
            return value
    raise ConstraintError(field="constraint", reason="missing internal parameter")


def _validate_constraint(
    constraint: CensusConstraint, core: CensusSpecCore, index: int
) -> None:
    if constraint.kind == "nullary_value":
        if core.arity != 0:
            raise ConstraintError(
                field="constraints",
                reason="nullary value requires a nullary operation",
                index=index,
            )
        element = cast(int, _parameter(constraint, "element"))
        if element >= core.carrier_size:
            raise ConstraintError(
                field="constraints",
                reason="element is outside the carrier range",
                index=index,
            )
        return
    if constraint.kind in {"commutative", "idempotent", "quasigroup", "identity"}:
        if core.arity != 2:
            raise ConstraintError(
                field="constraints",
                reason="constraint requires a binary operation",
                index=index,
            )
        if constraint.kind == "identity":
            element = cast(int, _parameter(constraint, "element"))
            if element >= core.carrier_size:
                raise ConstraintError(
                    field="constraints",
                    reason="element is outside the carrier range",
                    index=index,
                )
        return
    if constraint.kind == "equation":
        count = cast(int, _parameter(constraint, "variable_count"))
        assert constraint.left is not None and constraint.right is not None
        _validate_term(constraint.left, core, count, constraint_index=index)
        _validate_term(constraint.right, core, count, constraint_index=index)
        return
    raise ConstraintError(
        field="constraints", reason="unknown constraint kind", index=index
    )


def _snapshot_constraints(value: object) -> tuple[object, ...]:
    if isinstance(value, str | bytes):
        raise ConstraintError(
            field="constraints", reason="must be an iterable of declarations"
        )
    try:
        iterator = iter(cast(Iterable[object], value))
    except Exception as error:
        raise ConstraintError(
            field="constraints", reason="must be an iterable"
        ) from error
    snapshot: list[object] = []
    try:
        for _ in range(_MAX_CONSTRAINTS + 1):
            snapshot.append(next(iterator))
    except StopIteration:
        return tuple(snapshot)
    except Exception as error:
        raise ConstraintError(
            field="constraints", reason="iterable snapshot failed"
        ) from error
    raise ConstraintError(field="constraints", reason="declaration limit exceeded")


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ConstraintSet:
    """Canonical immutable set of validated allow-listed constraints."""

    core: CensusSpecCore
    constraints: tuple[CensusConstraint, ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ConstraintError(field="constraint_set", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("ConstraintSet cannot be subclassed")

    @classmethod
    def create(
        cls, core: CensusSpecCore, constraints: Iterable[CensusConstraint]
    ) -> ConstraintSet:
        if cls is not ConstraintSet:
            raise ConstraintError(
                field="constraint_set", reason="factory requires exact type"
            )
        if type(core) is not CensusSpecCore:
            raise ConstraintError(
                field="core", reason="must be an exact CensusSpecCore"
            )
        source = _snapshot_constraints(constraints)
        unique: dict[tuple[object, ...], CensusConstraint] = {}
        for index, item in enumerate(source):
            if type(item) is not CensusConstraint:
                raise ConstraintError(
                    field="constraints",
                    reason="must contain exact CensusConstraint values",
                    index=index,
                )
            _validate_constraint(item, core, index)
            unique.setdefault(_constraint_key(item), item)
        canonical = tuple(unique[key] for key in sorted(unique))
        value = object.__new__(ConstraintSet)
        object.__setattr__(value, "core", core)
        object.__setattr__(value, "constraints", canonical)
        return value

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is ConstraintSet
            and self.core == other.core
            and self.constraints == other.constraints
        )

    def __repr__(self) -> str:
        kinds = tuple(constraint.kind for constraint in self.constraints)
        return f"ConstraintSet(count={len(self.constraints)}, kinds={kinds!r})"
