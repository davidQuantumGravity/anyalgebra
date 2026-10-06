"""Checked syntax-only factories for common quantified algebra laws.

The functions in this module create :class:`~anyalgebra.structures.laws.Law`
values.  They neither interpret operations nor validate a law on a structure.
Every operation is an explicit caller-supplied symbol: names and container
membership are never used to discover an operation.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import cast

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import Sort
from anyalgebra.structures.laws import Equation, Law, LawDefinitionError
from anyalgebra.structures.signatures import OperationSymbol
from anyalgebra.structures.terms import Term, Variable


_MAX_DECLARATIONS = 512


class LawFactoryError(AnyAlgebraError, ValueError):
    """A generic-law factory received malformed syntax-level input."""

    def __init__(
        self,
        *,
        field: str,
        reason: str,
        index: int | None = None,
        conflicting_index: int | None = None,
        kind: str | None = None,
        observed: int | None = None,
        maximum: int | None = None,
    ) -> None:
        """Record an address-free deterministic declaration error."""
        self.field = field
        self.reason = reason
        self.index = index
        self.conflicting_index = conflicting_index
        self.kind = kind
        self.observed = observed
        self.maximum = maximum
        location = field if index is None else f"{field}[{index}]"
        detail = f"invalid law factory {location}: {reason}"
        if conflicting_index is not None:
            assert index is not None
            detail = (
                f"invalid law factory {field}: {reason} at declared indices "
                f"{conflicting_index} and {index}"
            )
        if kind is not None:
            detail = f"{detail}; kind={kind}"
        if observed is not None and maximum is not None:
            detail = f"{detail}; observed={observed}; maximum={maximum}"
        super().__init__(detail)


def _require_name(value: object) -> str:
    """Validate the law name without accepting subclasses or padded strings."""
    if type(value) is not str or not value or value != value.strip():
        raise LawFactoryError(
            field="name", reason="must be a non-empty trimmed built-in str"
        )
    return value


def _require_partial_semantics(value: object) -> str:
    """Validate the two syntax-level policies understood by ``Law``."""
    if type(value) is not str or value not in {"strong", "definedness_conditional"}:
        raise LawFactoryError(
            field="partial_semantics",
            reason="must be one of 'strong' or 'definedness_conditional'",
        )
    return value


def _require_closed_binary(
    value: object, *, field: str = "operation"
) -> OperationSymbol:
    """Require one exact binary endomorphism symbol without name inference."""
    if type(value) is not OperationSymbol:
        raise LawFactoryError(field=field, reason="must be an exact OperationSymbol")
    if (
        value.arity != 2
        or value.inputs[0] != value.output
        or value.inputs[1] != value.output
    ):
        raise LawFactoryError(
            field=field, reason="must be a closed binary OperationSymbol"
        )
    return value


def _require_zero(value: object, *, sort: Sort) -> OperationSymbol:
    """Require an explicit zero constant for the additive Jacobi equation."""
    if type(value) is not OperationSymbol:
        raise LawFactoryError(field="zero", reason="must be an exact OperationSymbol")
    if value.arity != 0 or value.output != sort:
        raise LawFactoryError(
            field="zero",
            reason="must be a nullary OperationSymbol with the bracket sort output",
        )
    return value


def _variables(sort: Sort, *names: str) -> tuple[Variable, ...]:
    """Return fresh stable-name variables over the operation's literal output sort."""
    return tuple(Variable(name, sort) for name in names)


def _apply(symbol: OperationSymbol, *arguments: Term) -> Term:
    """Apply an already compatibility-checked explicit operation symbol."""
    return Term.apply(symbol, *arguments)


def _from_components(
    name: object,
    variables: Iterable[Variable],
    conclusion: object,
    *,
    operations: Iterable[OperationSymbol],
    premises: Iterable[Equation] = (),
    partial_semantics: object = "strong",
) -> Law:
    """Validate explicit syntax declarations and construct one immutable law."""
    checked_name = _require_name(name)
    checked_variables = _snapshot_variables(variables)
    checked_conclusion = _require_equation(conclusion, field="conclusion")
    checked_operations = _snapshot_operations(operations)
    checked_premises = _snapshot_equations(premises, field="premises")
    checked_semantics = _require_partial_semantics(partial_semantics)
    _validate_term_operations(checked_conclusion.left, checked_operations)
    _validate_term_operations(checked_conclusion.right, checked_operations)
    for premise in checked_premises:
        _validate_term_operations(premise.left, checked_operations)
        _validate_term_operations(premise.right, checked_operations)
    try:
        return Law(
            checked_name,
            checked_variables,
            checked_conclusion,
            hypotheses=checked_premises,
            partial_semantics=checked_semantics,
        )
    except LawDefinitionError as error:
        raise LawFactoryError(
            field=error.field,
            reason=error.reason,
            index=error.index,
            conflicting_index=error.conflicting_index,
        ) from error


def _snapshot_variables(values: Iterable[Variable]) -> tuple[Variable, ...]:
    """Snapshot one possibly one-pass quantifier iterable with exact validation."""
    snapshot = _snapshot(
        values,
        field="variables",
        description="Variable",
    )
    names: dict[str, int] = {}
    for index, value in enumerate(snapshot):
        if type(value) is not Variable:
            raise LawFactoryError(
                field="variables",
                reason="must contain exact Variable values",
                index=index,
            )
        prior = names.get(value.name)
        if prior is not None:
            raise LawFactoryError(
                field="variables",
                reason="duplicate quantified variable name",
                index=index,
                conflicting_index=prior,
            )
        names[value.name] = index
    return cast(tuple[Variable, ...], snapshot)


def _snapshot_operations(
    values: Iterable[OperationSymbol],
) -> tuple[OperationSymbol, ...]:
    """Snapshot explicitly supplied operations without accepting an operation lookup."""
    snapshot = _snapshot(
        values,
        field="operations",
        description="OperationSymbol",
    )
    for index, value in enumerate(snapshot):
        if type(value) is not OperationSymbol:
            raise LawFactoryError(
                field="operations",
                reason="must contain exact OperationSymbol values",
                index=index,
            )
        if any(value is prior for prior in snapshot[:index]):
            raise LawFactoryError(
                field="operations",
                reason="contains a duplicate explicit OperationSymbol declaration",
                index=index,
            )
    return cast(tuple[OperationSymbol, ...], snapshot)


def _snapshot_equations(
    values: Iterable[Equation], *, field: str
) -> tuple[Equation, ...]:
    """Snapshot ordered premise equations without evaluating their contents."""
    snapshot = _snapshot(values, field=field, description="Equation")
    for index, value in enumerate(snapshot):
        if type(value) is not Equation:
            raise LawFactoryError(
                field=field, reason="must contain exact Equation values", index=index
            )
    return cast(tuple[Equation, ...], snapshot)


def _snapshot(
    values: Iterable[object], *, field: str, description: str
) -> tuple[object, ...]:
    """Consume one declaration iterable at most ``_MAX_DECLARATIONS + 1`` times."""
    if isinstance(values, str | bytes):
        raise LawFactoryError(
            field=field,
            reason=f"must be an iterable of exact {description} values",
        )
    try:
        iterator = iter(values)
    except Exception:
        iterator = None
    if iterator is None:
        raise LawFactoryError(
            field=field,
            reason=(
                f"could not be snapshotted as an iterable of exact {description} values"
            ),
        ) from None
    snapshot: list[object] = []
    failed = False
    for _ in range(_MAX_DECLARATIONS + 1):
        try:
            snapshot.append(next(iterator))
        except StopIteration:
            break
        except Exception:
            failed = True
            break
    if failed:
        raise LawFactoryError(
            field=field,
            reason=(
                f"could not be snapshotted as an iterable of exact {description} values"
            ),
        ) from None
    if len(snapshot) > _MAX_DECLARATIONS:
        raise LawFactoryError(
            field=field,
            reason="exceeds declared maximum",
            kind=field,
            observed=len(snapshot),
            maximum=_MAX_DECLARATIONS,
        )
    return tuple(snapshot)


def _require_equation(value: object, *, field: str) -> Equation:
    """Require one checked equation before traversing its terms."""
    if type(value) is not Equation:
        raise LawFactoryError(field=field, reason="must be an exact Equation")
    return value


def _symbols_in(term: Term) -> Iterator[OperationSymbol]:
    """Yield each literal operation symbol in a term in depth-first order."""
    if term.symbol is not None:
        yield term.symbol
    for argument in term.arguments:
        yield from _symbols_in(argument)


def _validate_term_operations(
    term: Term, operations: tuple[OperationSymbol, ...]
) -> None:
    """Require every term application to use one supplied symbol by identity."""
    for symbol in _symbols_in(term):
        if not any(symbol is declared for declared in operations):
            raise LawFactoryError(
                field="operations",
                reason="does not declare an operation used by the law",
            )


def associative_law(
    operation: OperationSymbol,
    *,
    name: str = "associativity",
    partial_semantics: str = "strong",
) -> Law:
    """Return ``(x*y)*z = x*(y*z)`` for one explicit closed binary operation."""
    product = _require_closed_binary(operation)
    x, y, z = _variables(product.output, "x", "y", "z")
    x_term, y_term, z_term = (Term.variable(item) for item in (x, y, z))
    return _from_components(
        name,
        (x, y, z),
        Equation(
            _apply(product, _apply(product, x_term, y_term), z_term),
            _apply(product, x_term, _apply(product, y_term, z_term)),
        ),
        operations=(product,),
        partial_semantics=partial_semantics,
    )


def commutative_law(
    operation: OperationSymbol,
    *,
    name: str = "commutativity",
    partial_semantics: str = "strong",
) -> Law:
    """Return ``x*y = y*x`` for one explicit closed binary operation."""
    product = _require_closed_binary(operation)
    x, y = _variables(product.output, "x", "y")
    x_term, y_term = (Term.variable(item) for item in (x, y))
    return _from_components(
        name,
        (x, y),
        Equation(_apply(product, x_term, y_term), _apply(product, y_term, x_term)),
        operations=(product,),
        partial_semantics=partial_semantics,
    )


def left_alternativity_law(
    operation: OperationSymbol,
    *,
    name: str = "left_alternativity",
    partial_semantics: str = "strong",
) -> Law:
    """Return ``(x*x)*y = x*(x*y)`` for one explicit closed binary operation."""
    product = _require_closed_binary(operation)
    x, y = _variables(product.output, "x", "y")
    x_term, y_term = (Term.variable(item) for item in (x, y))
    return _from_components(
        name,
        (x, y),
        Equation(
            _apply(product, _apply(product, x_term, x_term), y_term),
            _apply(product, x_term, _apply(product, x_term, y_term)),
        ),
        operations=(product,),
        partial_semantics=partial_semantics,
    )


def right_alternativity_law(
    operation: OperationSymbol,
    *,
    name: str = "right_alternativity",
    partial_semantics: str = "strong",
) -> Law:
    """Return ``(x*y)*y = x*(y*y)`` for one explicit closed binary operation."""
    product = _require_closed_binary(operation)
    x, y = _variables(product.output, "x", "y")
    x_term, y_term = (Term.variable(item) for item in (x, y))
    return _from_components(
        name,
        (x, y),
        Equation(
            _apply(product, _apply(product, x_term, y_term), y_term),
            _apply(product, x_term, _apply(product, y_term, y_term)),
        ),
        operations=(product,),
        partial_semantics=partial_semantics,
    )


def alternative_laws(
    operation: OperationSymbol,
    *,
    partial_semantics: str = "strong",
) -> tuple[Law, Law]:
    """Return left then right alternative laws for one explicit product."""
    product = _require_closed_binary(operation)
    return (
        left_alternativity_law(product, partial_semantics=partial_semantics),
        right_alternativity_law(product, partial_semantics=partial_semantics),
    )


def flexible_law(
    operation: OperationSymbol,
    *,
    name: str = "flexibility",
    partial_semantics: str = "strong",
) -> Law:
    """Return ``(x*y)*x = x*(y*x)`` for one explicit closed binary operation."""
    product = _require_closed_binary(operation)
    x, y = _variables(product.output, "x", "y")
    x_term, y_term = (Term.variable(item) for item in (x, y))
    return _from_components(
        name,
        (x, y),
        Equation(
            _apply(product, _apply(product, x_term, y_term), x_term),
            _apply(product, x_term, _apply(product, y_term, x_term)),
        ),
        operations=(product,),
        partial_semantics=partial_semantics,
    )


def jacobi_law(
    bracket: OperationSymbol,
    addition: OperationSymbol,
    zero: OperationSymbol,
    *,
    name: str = "jacobi",
    partial_semantics: str = "strong",
) -> Law:
    """Return the explicit equation ``[x,[y,z]]+[y,[z,x]]+[z,[x,y]]=0``."""
    checked_bracket = _require_closed_binary(bracket, field="bracket")
    checked_addition = _require_closed_binary(addition, field="addition")
    if checked_addition.output != checked_bracket.output:
        raise LawFactoryError(
            field="addition", reason="must have the bracket sort as its closed sort"
        )
    checked_zero = _require_zero(zero, sort=checked_bracket.output)
    x, y, z = _variables(checked_bracket.output, "x", "y", "z")
    x_term, y_term, z_term = (Term.variable(item) for item in (x, y, z))
    first = _apply(checked_bracket, x_term, _apply(checked_bracket, y_term, z_term))
    second = _apply(checked_bracket, y_term, _apply(checked_bracket, z_term, x_term))
    third = _apply(checked_bracket, z_term, _apply(checked_bracket, x_term, y_term))
    return _from_components(
        name,
        (x, y, z),
        Equation(
            _apply(checked_addition, _apply(checked_addition, first, second), third),
            _apply(checked_zero),
        ),
        operations=(checked_bracket, checked_addition, checked_zero),
        partial_semantics=partial_semantics,
    )


def jordan_law(
    operation: OperationSymbol,
    *,
    name: str = "jordan",
    partial_semantics: str = "strong",
) -> Law:
    """Return the Jordan identity ``((x*x)*y)*x = (x*x)*(y*x)``.

    Commutativity is intentionally a separate law: treating it as a premise
    would only make this equation conditional, not assert it globally.
    """
    product = _require_closed_binary(operation)
    x, y = _variables(product.output, "x", "y")
    x_term, y_term = (Term.variable(item) for item in (x, y))
    square = _apply(product, x_term, x_term)
    return _from_components(
        name,
        (x, y),
        Equation(
            _apply(product, _apply(product, square, y_term), x_term),
            _apply(product, square, _apply(product, y_term, x_term)),
        ),
        operations=(product,),
        partial_semantics=partial_semantics,
    )


def custom_law(
    name: str,
    variables: Iterable[Variable],
    conclusion: Equation,
    *,
    operations: Iterable[OperationSymbol],
    premises: Iterable[Equation] = (),
    partial_semantics: str = "strong",
) -> Law:
    """Construct a custom law from explicit symbols and ordered premises only."""
    return _from_components(
        name,
        variables,
        conclusion,
        operations=operations,
        premises=premises,
        partial_semantics=partial_semantics,
    )


# Early v0.0 spelling aliases retain compatibility while the documented names
# above remain the canonical callable API.
associativity_law = associative_law
commutativity_law = commutative_law
flexibility_law = flexible_law
