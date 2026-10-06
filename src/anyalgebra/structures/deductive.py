"""Finite ground deductive closure with explicit bounded-round evidence."""

from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.structures.terms import Term


class DeductiveDefinitionError(AnyAlgebraError, ValueError):
    def __init__(self, *, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid deductive {field}: {reason}")


def _terms(values: Iterable[Term], field: str) -> tuple[Term, ...]:
    if isinstance(values, str | bytes):
        raise DeductiveDefinitionError(
            field=field, reason="must be an iterable of exact Term values"
        )
    try:
        result = tuple(values)
    except Exception as error:
        raise DeductiveDefinitionError(
            field=field, reason="must be an iterable of exact Term values"
        ) from error
    if any(type(value) is not Term for value in result):
        raise DeductiveDefinitionError(
            field=field, reason="must be an iterable of exact Term values"
        )
    if any(not _is_ground(value) for value in result):
        raise DeductiveDefinitionError(
            field=field, reason="must contain only ground Term values"
        )
    return result


def _is_ground(term: Term) -> bool:
    """Return whether an exact term contains no variable leaf."""
    return term.variable_value is None and all(
        _is_ground(child) for child in term.arguments
    )


@dataclass(frozen=True, slots=True, init=False)
class InferenceRule:
    premises: tuple[Term, ...]
    conclusion: Term
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, premises: Iterable[Term], conclusion: Term) -> None:
        if type(conclusion) is not Term:
            raise DeductiveDefinitionError(
                field="conclusion", reason="must be an exact Term"
            )
        if not _is_ground(conclusion):
            raise DeductiveDefinitionError(
                field="conclusion", reason="must be a ground Term"
            )
        object.__setattr__(self, "premises", _terms(premises, "premises"))
        object.__setattr__(self, "conclusion", conclusion)


@dataclass(frozen=True, slots=True, init=False)
class DeductiveSystem:
    rules: tuple[InferenceRule, ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, rules: Iterable[InferenceRule]) -> None:
        if isinstance(rules, str | bytes):
            raise DeductiveDefinitionError(
                field="rules",
                reason="must be an iterable of exact InferenceRule values",
            )
        try:
            snapshot = tuple(rules)
        except Exception as error:
            raise DeductiveDefinitionError(
                field="rules",
                reason="must be an iterable of exact InferenceRule values",
            ) from error
        if any(type(rule) is not InferenceRule for rule in snapshot):
            raise DeductiveDefinitionError(
                field="rules",
                reason="must be an iterable of exact InferenceRule values",
            )
        object.__setattr__(self, "rules", snapshot)


@dataclass(frozen=True, slots=True, init=False)
class Derivation:
    conclusion: Term
    rule_index: int
    premises: tuple[Term, ...]
    premise_indices: tuple[int, ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise DeductiveDefinitionError(
            field="derivation", reason="created only by closure"
        )

    @classmethod
    def _make(
        cls,
        conclusion: Term,
        rule_index: int,
        premises: tuple[Term, ...],
        premise_indices: tuple[int, ...],
    ) -> Derivation:
        if (
            type(conclusion) is not Term
            or type(premises) is not tuple
            or type(premise_indices) is not tuple
            or any(type(x) is not Term for x in premises)
            or any(not _is_ground(x) for x in premises)
            or not _is_ground(conclusion)
            or len(premises) != len(premise_indices)
            or type(rule_index) is not int
            or rule_index < 0
            or any(type(i) is not int or i < 0 for i in premise_indices)
        ):
            raise DeductiveDefinitionError(
                field="derivation", reason="invalid derivation evidence"
            )
        value = object.__new__(cls)
        object.__setattr__(value, "conclusion", conclusion)
        object.__setattr__(value, "rule_index", rule_index)
        object.__setattr__(value, "premises", premises)
        object.__setattr__(value, "premise_indices", premise_indices)
        return value


@dataclass(frozen=True, slots=True, init=False)
class CompleteClosure:
    conclusions: tuple[Term, ...]
    derivations: tuple[Derivation, ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise DeductiveDefinitionError(
            field="closure", reason="created only by closure"
        )

    @classmethod
    def _make(
        cls, conclusions: tuple[Term, ...], derivations: tuple[Derivation, ...]
    ) -> CompleteClosure:
        if type(conclusions) is not tuple or type(derivations) is not tuple:
            raise DeductiveDefinitionError(
                field="closure", reason="invalid closure evidence"
            )
        if any(type(term) is not Term or not _is_ground(term) for term in conclusions):
            raise DeductiveDefinitionError(
                field="closure", reason="conclusions must be exact ground Terms"
            )
        if any(type(derivation) is not Derivation for derivation in derivations):
            raise DeductiveDefinitionError(
                field="closure", reason="derivations must be exact Derivation values"
            )
        for index, conclusion in enumerate(conclusions):
            if _index(list(conclusions[:index]), conclusion) is not None:
                raise DeductiveDefinitionError(
                    field="closure", reason="conclusions must be unique"
                )

        initial_count = len(conclusions) - len(derivations)
        if initial_count < 0:
            raise DeductiveDefinitionError(
                field="closure", reason="more derivations than conclusions"
            )
        for offset, derivation in enumerate(derivations):
            conclusion_index = initial_count + offset
            if not _same_term(derivation.conclusion, conclusions[conclusion_index]):
                raise DeductiveDefinitionError(
                    field="closure",
                    reason="derivation conclusion suffix does not replay",
                )
            for premise, premise_index in zip(
                derivation.premises, derivation.premise_indices, strict=True
            ):
                if premise_index >= conclusion_index or not _same_term(
                    premise, conclusions[premise_index]
                ):
                    raise DeductiveDefinitionError(
                        field="closure", reason="derivation premise does not replay"
                    )
        value = object.__new__(cls)
        object.__setattr__(value, "conclusions", conclusions)
        object.__setattr__(value, "derivations", derivations)
        return value


@dataclass(frozen=True, slots=True, init=False)
class BoundExhausted:
    conclusions: tuple[Term, ...]
    derivations: tuple[Derivation, ...]
    max_rounds: int
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise DeductiveDefinitionError(
            field="closure", reason="created only by closure"
        )

    @classmethod
    def _make(
        cls,
        conclusions: tuple[Term, ...],
        derivations: tuple[Derivation, ...],
        max_rounds: int,
    ) -> BoundExhausted:
        base = CompleteClosure._make(conclusions, derivations)
        if type(max_rounds) is not int or max_rounds < 0:
            raise DeductiveDefinitionError(
                field="max_rounds", reason="must be a non-negative built-in int"
            )
        value = object.__new__(cls)
        object.__setattr__(value, "conclusions", base.conclusions)
        object.__setattr__(value, "derivations", base.derivations)
        object.__setattr__(value, "max_rounds", max_rounds)
        return value


def _same_term(left: Term, right: Term) -> bool:
    """Compare terms structurally without requiring their hash implementation."""
    if left is right:
        return True
    try:
        return bool(left == right)
    except Exception as error:
        raise DeductiveDefinitionError(
            field="terms", reason="structural equality comparison failed"
        ) from error


def _index(values: list[Term], term: Term) -> int | None:
    for index, value in enumerate(values):
        if _same_term(value, term):
            return index
    return None


def _applicable(rule: InferenceRule, values: list[Term]) -> tuple[int, ...] | None:
    indices = []
    for premise in rule.premises:
        index = _index(values, premise)
        if index is None:
            return None
        indices.append(index)
    return tuple(indices)


def closure(
    premises: Iterable[Term], system: DeductiveSystem, *, max_rounds: int
) -> CompleteClosure | BoundExhausted:
    """Close finite ground rules; variables and unification are intentionally absent."""
    if type(system) is not DeductiveSystem:
        raise DeductiveDefinitionError(
            field="system", reason="must be an exact DeductiveSystem"
        )
    if type(max_rounds) is not int or max_rounds < 0:
        raise DeductiveDefinitionError(
            field="max_rounds", reason="must be a non-negative built-in int"
        )
    values: list[Term] = []
    for premise in _terms(premises, "premises"):
        if _index(values, premise) is None:
            values.append(premise)
    derivations: list[Derivation] = []
    rounds = 0
    while True:
        pending: list[tuple[Term, Derivation]] = []
        visible = values[:]
        for rule_index, rule in enumerate(system.rules):
            indices = _applicable(rule, visible)
            pending_terms = [conclusion for conclusion, _ in pending]
            if (
                indices is not None
                and _index(visible, rule.conclusion) is None
                and _index(pending_terms, rule.conclusion) is None
            ):
                pending.append(
                    (
                        rule.conclusion,
                        Derivation._make(
                            rule.conclusion, rule_index, rule.premises, indices
                        ),
                    )
                )
        if not pending:
            return CompleteClosure._make(tuple(values), tuple(derivations))
        if rounds >= max_rounds:
            return BoundExhausted._make(tuple(values), tuple(derivations), max_rounds)
        for conclusion, derivation in pending:
            values.append(conclusion)
            derivations.append(derivation)
        rounds += 1
