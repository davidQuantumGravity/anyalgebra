"""Bounded deterministic syntactic term rewriting without confluence claims."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.structures.terms import Term, Variable


class RewriteDefinitionError(AnyAlgebraError, ValueError):
    """A rewrite definition, system, bound, or result record was malformed."""

    def __init__(self, *, field: str, reason: str) -> None:
        """Keep failures deterministic and free of caller term reprs."""
        self.field = field
        self.reason = reason
        super().__init__(f"invalid rewrite {field}: {reason}")


def _variables(term: Term) -> tuple[Variable, ...]:
    """Return distinct variables in deterministic preorder occurrence order."""
    found: list[Variable] = []

    def visit(current: Term) -> None:
        if current.variable_value is not None:
            if current.variable_value not in found:
                found.append(current.variable_value)
            return
        for child in current.arguments:
            visit(child)

    visit(term)
    return tuple(found)


@dataclass(frozen=True, slots=True, init=False)
class RewriteRule:
    """One sort-preserving structural pattern and replacement template."""

    pattern: Term
    replacement: Term
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, pattern: Term, replacement: Term) -> None:
        """Validate terms, sort preservation, and replacement bindings eagerly."""
        if type(pattern) is not Term:
            raise RewriteDefinitionError(
                field="pattern", reason="pattern must be an exact Term"
            )
        if type(replacement) is not Term:
            raise RewriteDefinitionError(
                field="replacement", reason="replacement must be an exact Term"
            )
        if pattern.sort != replacement.sort:
            raise RewriteDefinitionError(
                field="replacement",
                reason="pattern and replacement must have the same sort",
            )
        pattern_variables = _variables(pattern)
        for variable in _variables(replacement):
            if variable not in pattern_variables:
                raise RewriteDefinitionError(
                    field="replacement",
                    reason="replacement variable is not bound by pattern",
                )
        object.__setattr__(self, "pattern", pattern)
        object.__setattr__(self, "replacement", replacement)


@dataclass(frozen=True, slots=True, init=False)
class RewriteSystem:
    """An ordered, deterministic first-applicable list of rewrite rules."""

    rules: tuple[RewriteRule, ...]
    strategy: str
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, rules: Iterable[RewriteRule]) -> None:
        """Snapshot one exact ordered rule collection without accepting strings."""
        if isinstance(rules, str | bytes):
            raise RewriteDefinitionError(
                field="rules",
                reason="rules must be an iterable of exact RewriteRule values",
            )
        try:
            snapshot = tuple(rules)
        except (TypeError, RuntimeError) as error:
            raise RewriteDefinitionError(
                field="rules",
                reason="rules must be an iterable of exact RewriteRule values",
            ) from error
        if any(type(rule) is not RewriteRule for rule in snapshot):
            raise RewriteDefinitionError(
                field="rules",
                reason="rules must be an iterable of exact RewriteRule values",
            )
        object.__setattr__(self, "rules", snapshot)
        object.__setattr__(self, "strategy", "preorder-first-applicable")


def _validate_trace(trace: tuple[Term, ...], *, field: str) -> None:
    """Require a nonempty exact immutable term trace."""
    if (
        type(trace) is not tuple
        or not trace
        or any(type(term) is not Term for term in trace)
    ):
        raise RewriteDefinitionError(
            field=field, reason="trace must be a nonempty tuple of exact Term values"
        )


@dataclass(frozen=True, slots=True, init=False)
class NormalForm:
    """A term at which the configured strategy finds no rewrite."""

    term: Term
    trace: tuple[Term, ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, term: Term, trace: tuple[Term, ...]) -> None:
        """Validate the completed deterministic normal-form trace."""
        if type(term) is not Term:
            raise RewriteDefinitionError(
                field="term", reason="term must be an exact Term"
            )
        _validate_trace(trace, field="trace")
        if trace[-1] != term:
            raise RewriteDefinitionError(field="trace", reason="trace must end at term")
        object.__setattr__(self, "term", term)
        object.__setattr__(self, "trace", trace)


@dataclass(frozen=True, slots=True, init=False)
class BoundExhausted:
    """A rewrite exists but the declared step limit was reached first."""

    last_term: Term
    trace: tuple[Term, ...]
    max_steps: int
    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self, last_term: Term, trace: tuple[Term, ...], max_steps: int
    ) -> None:
        """Validate an explicit non-normal bounded execution record."""
        if type(last_term) is not Term:
            raise RewriteDefinitionError(
                field="last_term", reason="last_term must be an exact Term"
            )
        _validate_trace(trace, field="trace")
        if trace[-1] != last_term:
            raise RewriteDefinitionError(
                field="trace", reason="trace must end at last_term"
            )
        _validate_bound(max_steps)
        if len(trace) != max_steps + 1:
            raise RewriteDefinitionError(
                field="trace", reason="trace length must equal max_steps plus one"
            )
        object.__setattr__(self, "last_term", last_term)
        object.__setattr__(self, "trace", trace)
        object.__setattr__(self, "max_steps", max_steps)


@dataclass(frozen=True, slots=True, init=False)
class CycleDetected:
    """An exact repeated term and the deterministic cycle path that reached it."""

    entry_index: int
    cycle_path: tuple[Term, ...]
    trace: tuple[Term, ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self, entry_index: int, cycle_path: tuple[Term, ...], trace: tuple[Term, ...]
    ) -> None:
        """Validate a closed cycle path embedded at one trace entry index."""
        if type(entry_index) is not int or entry_index < 0:
            raise RewriteDefinitionError(
                field="entry_index", reason="must be a non-negative built-in int"
            )
        _validate_trace(trace, field="trace")
        _validate_trace(cycle_path, field="cycle_path")
        if entry_index >= len(trace) or cycle_path[0] != trace[entry_index]:
            raise RewriteDefinitionError(
                field="cycle_path", reason="must start at the indexed trace term"
            )
        if len(cycle_path) < 2 or cycle_path[0] != cycle_path[-1]:
            raise RewriteDefinitionError(field="cycle_path", reason="must be closed")
        if cycle_path != trace[entry_index:]:
            raise RewriteDefinitionError(
                field="cycle_path",
                reason="must equal the trace suffix from entry_index",
            )
        object.__setattr__(self, "entry_index", entry_index)
        object.__setattr__(self, "cycle_path", cycle_path)
        object.__setattr__(self, "trace", trace)


def _validate_bound(max_steps: object) -> int:
    """Require a concrete nonnegative rewrite count, excluding Boolean values."""
    if type(max_steps) is not int or max_steps < 0:
        raise RewriteDefinitionError(
            field="max_steps", reason="max_steps must be a non-negative built-in int"
        )
    return max_steps


def _match(pattern: Term, target: Term, bindings: dict[Variable, Term]) -> bool:
    """Match structural syntax with repeated-variable consistency."""
    if pattern.sort != target.sort:
        return False
    if pattern.variable_value is not None:
        prior = bindings.get(pattern.variable_value)
        if prior is None:
            bindings[pattern.variable_value] = target
            return True
        return prior == target
    if target.variable_value is not None or pattern.symbol != target.symbol:
        return False
    return all(
        _match(left, right, bindings)
        for left, right in zip(pattern.arguments, target.arguments, strict=True)
    )


def _substitute(template: Term, bindings: dict[Variable, Term]) -> Term:
    """Instantiate a checked template from one complete structural match."""
    if template.variable_value is not None:
        return bindings[template.variable_value]
    assert template.symbol is not None
    return Term.apply(
        template.symbol,
        *(_substitute(child, bindings) for child in template.arguments),
    )


def _rewrite_once(term: Term, system: RewriteSystem) -> Term | None:
    """Apply the first matching rule at the first preorder tree position."""
    for rule in system.rules:
        bindings: dict[Variable, Term] = {}
        if _match(rule.pattern, term, bindings):
            return _substitute(rule.replacement, bindings)
    if term.variable_value is not None:
        return None
    for index, child in enumerate(term.arguments):
        rewritten = _rewrite_once(child, system)
        if rewritten is not None:
            assert term.symbol is not None
            arguments = list(term.arguments)
            arguments[index] = rewritten
            return Term.apply(term.symbol, *arguments)
    return None


def normalize(
    term: Term, system: RewriteSystem, *, max_steps: int
) -> NormalForm | BoundExhausted | CycleDetected:
    """Run one bounded strategy trace without termination assumptions.

    This is deliberately not relation-wide ``reachable`` or ``normal_forms``
    search: those later APIs must explore all one-step alternatives and report
    frontier exhaustion separately.  This function makes only its ordered
    preorder first-applicable strategy claim.
    """
    if type(term) is not Term:
        raise RewriteDefinitionError(field="term", reason="term must be an exact Term")
    if type(system) is not RewriteSystem:
        raise RewriteDefinitionError(
            field="system", reason="system must be an exact RewriteSystem"
        )
    bound = _validate_bound(max_steps)
    trace: list[Term] = [term]
    seen: dict[Term, int] = {term: 0}
    current = term
    steps = 0
    while True:
        next_term = _rewrite_once(current, system)
        if next_term is None:
            return NormalForm(current, tuple(trace))
        if steps >= bound:
            return BoundExhausted(current, tuple(trace), bound)
        steps += 1
        trace.append(next_term)
        entry_index = seen.get(next_term)
        if entry_index is not None:
            return CycleDetected(entry_index, tuple(trace[entry_index:]), tuple(trace))
        seen[next_term] = len(trace) - 1
        current = next_term
