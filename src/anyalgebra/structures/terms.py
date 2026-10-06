"""Typed immutable syntax trees for many-sorted operation terms.

Terms are syntax only: they preserve operation-tree shape and infer result
sorts, but deliberately do not provide interpretation or evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import Sort
from anyalgebra.structures.signatures import OperationSymbol


class VariableDefinitionError(AnyAlgebraError, ValueError):
    """A typed variable declaration was malformed."""

    def __init__(self, *, field: str, reason: str) -> None:
        """Record an address-free variable-definition diagnostic."""
        self.field = field
        self.reason = reason
        super().__init__(f"invalid variable {field}: {reason}")


class TermDefinitionError(AnyAlgebraError, ValueError):
    """A public term-construction request violated the syntax contract."""

    def __init__(
        self,
        *,
        field: str,
        reason: str,
        index: int | None = None,
        expected_sort: Sort | None = None,
        actual_sort: Sort | None = None,
    ) -> None:
        """Record a deterministic diagnostic without rendering arbitrary values."""
        self.field = field
        self.reason = reason
        self.index = index
        self.expected_sort = expected_sort
        self.actual_sort = actual_sort
        location = field if index is None else f"{field}[{index}]"
        if expected_sort is not None and actual_sort is not None:
            detail = (
                f"expected sort {expected_sort.name!r}, got sort {actual_sort.name!r}"
            )
        else:
            detail = reason
        super().__init__(f"invalid term {location}: {detail}")


def _require_variable_name(value: object) -> str:
    """Validate an exact, nonempty, unpadded variable name."""
    if type(value) is not str or not value or value != value.strip():
        raise VariableDefinitionError(
            field="name",
            reason="must be a non-empty trimmed built-in str",
        )
    return value


def _require_variable_sort(value: object) -> Sort:
    """Validate an exact variable sort while retaining its literal instance."""
    if type(value) is not Sort:
        raise VariableDefinitionError(field="sort", reason="must be an exact Sort")
    return value


@dataclass(frozen=True, slots=True, init=False)
class Variable:
    """An immutable variable leaf with one declared many-sorted type."""

    name: str
    sort: Sort

    def __init__(self, name: str, sort: Sort) -> None:
        """Validate and retain one literal exact sort instance."""
        object.__setattr__(self, "name", _require_variable_name(name))
        object.__setattr__(self, "sort", _require_variable_sort(sort))


@dataclass(frozen=True, slots=True, init=False)
class Term:
    """A factory-constructed, typed operation tree.

    ``variable_value`` is set only for variable leaves.  ``symbol`` and
    ``arguments`` are set only for operation applications; nullary operations
    therefore have an empty argument tuple.  The public constructor always
    rejects so no caller can form an ill-sorted node without a factory check.
    """

    variable_value: Variable | None
    symbol: OperationSymbol | None
    arguments: tuple[Term, ...]
    sort: Sort

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Reject direct construction in favor of checked public factories."""
        del args, kwargs
        raise TermDefinitionError(
            field="term",
            reason="must be constructed with Term.variable() or Term.apply()",
        )

    @classmethod
    def variable(cls, variable: Variable) -> Term:
        """Construct one typed variable leaf from an exact ``Variable``."""
        if type(variable) is not Variable:
            raise TermDefinitionError(
                field="variable", reason="must be an exact Variable"
            )
        return cls._make(
            variable_value=variable,
            symbol=None,
            arguments=(),
            sort=variable.sort,
        )

    @classmethod
    def apply(cls, symbol: OperationSymbol, *arguments: Term) -> Term:
        """Construct a checked operation application with inferred result sort."""
        if type(symbol) is not OperationSymbol:
            raise TermDefinitionError(
                field="symbol", reason="must be an exact OperationSymbol"
            )
        if len(arguments) != symbol.arity:
            raise TermDefinitionError(
                field="arguments",
                reason=(f"expected {symbol.arity} arguments, got {len(arguments)}"),
            )
        for index, (argument, expected_sort) in enumerate(
            zip(arguments, symbol.inputs, strict=True)
        ):
            if type(argument) is not Term:
                raise TermDefinitionError(
                    field="arguments",
                    reason="must contain exact Term values",
                    index=index,
                )
            if argument.sort != expected_sort:
                raise TermDefinitionError(
                    field="arguments",
                    reason="argument sort does not match the operation input sort",
                    index=index,
                    expected_sort=expected_sort,
                    actual_sort=argument.sort,
                )
        return cls._make(
            variable_value=None,
            symbol=symbol,
            arguments=arguments,
            sort=symbol.output,
        )

    @classmethod
    def _make(
        cls,
        *,
        variable_value: Variable | None,
        symbol: OperationSymbol | None,
        arguments: tuple[Term, ...],
        sort: Sort,
    ) -> Term:
        """Create a node only after a public factory has proven its invariant."""
        term = object.__new__(cls)
        object.__setattr__(term, "variable_value", variable_value)
        object.__setattr__(term, "symbol", symbol)
        object.__setattr__(term, "arguments", arguments)
        object.__setattr__(term, "sort", sort)
        return term
