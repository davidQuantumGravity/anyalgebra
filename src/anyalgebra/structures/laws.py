"""Immutable quantified equation laws without evaluation or relation atoms.

This syntax layer represents only equations over typed operation terms.  A
relation conclusion or hypothesis will need a dedicated relation-atom AST; it
must not be represented as a boolean-valued :class:`~anyalgebra.structures.terms.Term`.
The declared partial-semantics value is retained for a later validator and has
no evaluator behavior here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import Sort
from anyalgebra.structures.terms import Term, Variable


class EquationDefinitionError(AnyAlgebraError, ValueError):
    """An equation did not join two exact same-sort terms."""

    def __init__(
        self,
        *,
        field: str,
        reason: str,
        expected_sort: Sort | None = None,
        actual_sort: Sort | None = None,
    ) -> None:
        """Record an address-free malformed-equation diagnostic."""
        self.field = field
        self.reason = reason
        self.expected_sort = expected_sort
        self.actual_sort = actual_sort
        detail = reason
        if expected_sort is not None and actual_sort is not None:
            detail = (
                f"expected sort {expected_sort.name!r}, got sort {actual_sort.name!r}"
            )
        super().__init__(f"invalid equation {field}: {detail}")


class LawDefinitionError(AnyAlgebraError, ValueError):
    """A quantified equation-law declaration was malformed."""

    def __init__(
        self,
        *,
        field: str,
        reason: str,
        index: int | None = None,
        conflicting_index: int | None = None,
        side: str | None = None,
        term_path: tuple[int, ...] | None = None,
    ) -> None:
        """Record a structured, deterministic law-definition failure."""
        self.field = field
        self.reason = reason
        self.index = index
        self.conflicting_index = conflicting_index
        self.side = side
        self.term_path = term_path
        location = field if index is None else f"{field}[{index}]"
        if side is not None:
            location = f"{location}.{side}"
        if term_path is not None:
            path = ".".join(str(part) for part in term_path)
            location = f"{location}.term" if not path else f"{location}.term[{path}]"
        detail = f"invalid law {location}: {reason}"
        if conflicting_index is not None:
            assert index is not None
            detail = (
                f"invalid law {field}: {reason} at declared indices "
                f"{conflicting_index} and {index}"
            )
        super().__init__(detail)


class LawSetDefinitionError(AnyAlgebraError, ValueError):
    """A named immutable law collection was malformed."""

    def __init__(
        self,
        *,
        field: str,
        reason: str,
        index: int | None = None,
        conflicting_index: int | None = None,
    ) -> None:
        """Record an address-free law-set definition diagnostic."""
        self.field = field
        self.reason = reason
        self.index = index
        self.conflicting_index = conflicting_index
        location = field if index is None else f"{field}[{index}]"
        detail = f"invalid law set {location}: {reason}"
        if conflicting_index is not None:
            assert index is not None
            detail = (
                f"invalid law set {field}: {reason} at declared indices "
                f"{conflicting_index} and {index}"
            )
        super().__init__(detail)


def _require_name(
    value: object, *, error_type: type[LawDefinitionError] | type[LawSetDefinitionError]
) -> str:
    """Validate an exact, nonempty, unpadded declared name."""
    if type(value) is not str or not value or value != value.strip():
        raise error_type(
            field="name", reason="must be a non-empty trimmed built-in str"
        )
    return value


def _require_partial_semantics(value: object) -> str:
    """Validate the two v0.0 syntax-level partial-law policies."""
    if type(value) is not str or value not in (
        "strong",
        "definedness_conditional",
    ):
        raise LawDefinitionError(
            field="partial_semantics",
            reason="must be one of 'strong' or 'definedness_conditional'",
        )
    return value


def _snapshot_variables(values: Iterable[Variable]) -> tuple[Variable, ...]:
    """Take one immutable ordered snapshot of exact quantified variables."""
    if isinstance(values, str | bytes):
        raise LawDefinitionError(
            field="variables", reason="must be an iterable of exact Variable values"
        )
    try:
        snapshot = tuple(values)
    except (TypeError, RuntimeError) as error:
        raise LawDefinitionError(
            field="variables",
            reason="could not be snapshotted as an iterable of exact Variable values",
        ) from error
    for index, value in enumerate(snapshot):
        if type(value) is not Variable:
            raise LawDefinitionError(
                field="variables",
                reason="must contain exact Variable values",
                index=index,
            )
    return snapshot


def _snapshot_equations(values: Iterable[Equation]) -> tuple[Equation, ...]:
    """Take one immutable ordered snapshot of exact hypothesis equations."""
    if isinstance(values, str | bytes):
        raise LawDefinitionError(
            field="hypotheses", reason="must be an iterable of exact Equation values"
        )
    try:
        snapshot = tuple(values)
    except (TypeError, RuntimeError) as error:
        raise LawDefinitionError(
            field="hypotheses",
            reason="could not be snapshotted as an iterable of exact Equation values",
        ) from error
    for index, value in enumerate(snapshot):
        if type(value) is not Equation:
            raise LawDefinitionError(
                field="hypotheses",
                reason="must contain exact Equation values",
                index=index,
            )
    return snapshot


def _snapshot_laws(values: Iterable[Law]) -> tuple[Law, ...]:
    """Take one immutable ordered snapshot of exact law values."""
    if isinstance(values, str | bytes):
        raise LawSetDefinitionError(
            field="laws", reason="must be an iterable of exact Law values"
        )
    try:
        snapshot = tuple(values)
    except (TypeError, RuntimeError) as error:
        raise LawSetDefinitionError(
            field="laws",
            reason="could not be snapshotted as an iterable of exact Law values",
        ) from error
    for index, value in enumerate(snapshot):
        if type(value) is not Law:
            raise LawSetDefinitionError(
                field="laws", reason="must contain exact Law values", index=index
            )
    return snapshot


def _reject_duplicate_variable_names(variables: tuple[Variable, ...]) -> None:
    """Reject the first repeated quantifier name regardless of variable sort."""
    declared_indices: dict[str, int] = {}
    for index, variable in enumerate(variables):
        prior_index = declared_indices.get(variable.name)
        if prior_index is not None:
            raise LawDefinitionError(
                field="variables",
                reason="duplicate quantified variable name",
                index=index,
                conflicting_index=prior_index,
            )
        declared_indices[variable.name] = index


def _find_undeclared_variable(
    term: Term, declared: frozenset[Variable], path: tuple[int, ...] = ()
) -> tuple[int, ...] | None:
    """Return the first depth-first path to an undeclared variable leaf."""
    if term.variable_value is not None:
        return None if term.variable_value in declared else path
    for index, argument in enumerate(term.arguments):
        found = _find_undeclared_variable(argument, declared, (*path, index))
        if found is not None:
            return found
    return None


def _validate_equation_variables(
    equation: Equation,
    declared: frozenset[Variable],
    *,
    field: str,
    index: int | None,
) -> None:
    """Ensure both equation sides use only structurally declared variables."""
    for side, term in (("left", equation.left), ("right", equation.right)):
        path = _find_undeclared_variable(term, declared)
        if path is not None:
            raise LawDefinitionError(
                field=field,
                reason=(
                    "contains a variable not structurally declared by the "
                    "quantifier list"
                ),
                index=index,
                side=side,
                term_path=path,
            )


@dataclass(frozen=True, slots=True, init=False)
class Equation:
    """An immutable equality between two exact terms of one structural sort."""

    left: Term
    right: Term

    def __init__(self, left: Term, right: Term) -> None:
        """Validate and retain the literal same-sort term instances supplied."""
        if type(left) is not Term:
            raise EquationDefinitionError(field="left", reason="must be an exact Term")
        if type(right) is not Term:
            raise EquationDefinitionError(field="right", reason="must be an exact Term")
        if left.sort != right.sort:
            raise EquationDefinitionError(
                field="right",
                reason="term sort does not match the left term sort",
                expected_sort=left.sort,
                actual_sort=right.sort,
            )
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)


@dataclass(frozen=True, slots=True, init=False)
class Law:
    """A quantified equation with ordered equation hypotheses and a policy tag."""

    name: str
    variables: tuple[Variable, ...]
    conclusion: Equation
    hypotheses: tuple[Equation, ...]
    partial_semantics: str

    def __init__(
        self,
        name: str,
        variables: Iterable[Variable],
        conclusion: Equation,
        *,
        hypotheses: Iterable[Equation] = (),
        partial_semantics: str = "strong",
    ) -> None:
        """Validate a quantified equation without interpreting or evaluating it."""
        validated_name = _require_name(name, error_type=LawDefinitionError)
        variable_snapshot = _snapshot_variables(variables)
        _reject_duplicate_variable_names(variable_snapshot)
        if type(conclusion) is not Equation:
            raise LawDefinitionError(
                field="conclusion", reason="must be an exact Equation"
            )
        hypothesis_snapshot = _snapshot_equations(hypotheses)
        validated_semantics = _require_partial_semantics(partial_semantics)

        declared_variables = frozenset(variable_snapshot)
        _validate_equation_variables(
            conclusion,
            declared_variables,
            field="conclusion",
            index=None,
        )
        for index, hypothesis in enumerate(hypothesis_snapshot):
            _validate_equation_variables(
                hypothesis,
                declared_variables,
                field="hypotheses",
                index=index,
            )

        object.__setattr__(self, "name", validated_name)
        object.__setattr__(self, "variables", variable_snapshot)
        object.__setattr__(self, "conclusion", conclusion)
        object.__setattr__(self, "hypotheses", hypothesis_snapshot)
        object.__setattr__(self, "partial_semantics", validated_semantics)


@dataclass(frozen=True, slots=True, init=False)
class LawSet:
    """A named immutable ordered collection of named quantified equation laws."""

    name: str
    laws: tuple[Law, ...]

    def __init__(self, name: str, laws: Iterable[Law]) -> None:
        """Validate and freeze a law collection without adding evaluation behavior."""
        validated_name = _require_name(name, error_type=LawSetDefinitionError)
        law_snapshot = _snapshot_laws(laws)
        declared_indices: dict[str, int] = {}
        for index, law in enumerate(law_snapshot):
            prior_index = declared_indices.get(law.name)
            if prior_index is not None:
                raise LawSetDefinitionError(
                    field="laws",
                    reason="duplicate law name",
                    index=index,
                    conflicting_index=prior_index,
                )
            declared_indices[law.name] = index
        object.__setattr__(self, "name", validated_name)
        object.__setattr__(self, "laws", law_snapshot)
