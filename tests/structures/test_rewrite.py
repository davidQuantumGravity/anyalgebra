"""Contract tests for bounded deterministic syntactic rewrite systems."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from anyalgebra.core.parents import Sort
from anyalgebra.structures.rewrite import (
    BoundExhausted,
    CycleDetected,
    NormalForm,
    RewriteDefinitionError,
    RewriteRule,
    RewriteSystem,
    normalize,
)
from anyalgebra.structures.signatures import OperationSymbol
from anyalgebra.structures.terms import Term, Variable


def test_root_nested_ordered_and_repeated_variable_matching() -> None:
    scalar = Sort("scalar")
    zero = OperationSymbol("zero", (), scalar)
    pair = OperationSymbol("pair", (scalar, scalar), scalar)
    wrap = OperationSymbol("wrap", (scalar,), scalar)
    x_variable = Variable("x", scalar)
    x = Term.variable(x_variable)
    zero_term = Term.apply(zero)
    collapse = RewriteRule(Term.apply(pair, x, x), x)
    unwrap = RewriteRule(Term.apply(wrap, x), x)
    system = RewriteSystem((collapse, unwrap))

    nested = Term.apply(wrap, Term.apply(pair, zero_term, zero_term))
    normal = normalize(nested, system, max_steps=4)
    collapsed = Term.apply(pair, zero_term, zero_term)
    assert normal == NormalForm(zero_term, (nested, collapsed, zero_term))
    nested_right = Term.apply(pair, zero_term, Term.apply(wrap, zero_term))
    assert normalize(nested_right, system, max_steps=4) == NormalForm(
        zero_term,
        (nested_right, collapsed, zero_term),
    )


def test_ordered_first_applicable_strategy_is_explicit_not_confluent() -> None:
    scalar = Sort("scalar")
    a = OperationSymbol("a", (), scalar)
    b = OperationSymbol("b", (), scalar)
    c = OperationSymbol("c", (), scalar)
    source = Term.apply(a)
    first = RewriteRule(source, Term.apply(b))
    second = RewriteRule(source, Term.apply(c))

    result = normalize(source, RewriteSystem((first, second)), max_steps=1)
    assert result == NormalForm(Term.apply(b), (source, Term.apply(b)))


def test_cycle_and_bound_exhaustion_are_distinct_with_exact_witnesses() -> None:
    scalar = Sort("scalar")
    a = OperationSymbol("a", (), scalar)
    b = OperationSymbol("b", (), scalar)
    source = Term.apply(a)
    middle = Term.apply(b)
    system = RewriteSystem((RewriteRule(source, middle), RewriteRule(middle, source)))

    cycle = normalize(source, system, max_steps=4)
    assert isinstance(cycle, CycleDetected)
    assert cycle.entry_index == 0
    assert cycle.cycle_path == (source, middle, source)
    assert cycle.trace == (source, middle, source)
    bounded = normalize(source, system, max_steps=1)
    assert bounded == BoundExhausted(middle, (source, middle), 1)


def test_zero_bound_normal_form_and_high_arity_parenthesized_boundaries() -> None:
    scalar = Sort("scalar")
    a = OperationSymbol("a", (), scalar)
    f = OperationSymbol("f", (scalar,), scalar)
    many = OperationSymbol("many", (scalar,) * 257, scalar)
    source = Term.apply(a)
    irreducible = Term.apply(f, source)
    system = RewriteSystem((RewriteRule(source, Term.apply(f, source)),))

    assert normalize(source, system, max_steps=0) == BoundExhausted(
        source, (source,), 0
    )
    assert normalize(irreducible, RewriteSystem(()), max_steps=0) == NormalForm(
        irreducible, (irreducible,)
    )
    high = Term.apply(many, *(source,) * 257)
    assert normalize(high, RewriteSystem(()), max_steps=3) == NormalForm(high, (high,))


def test_malformed_rules_sorts_bounds_and_factory_records_are_typed() -> None:
    scalar = Sort("scalar")
    other = Sort("other")
    a = Term.apply(OperationSymbol("a", (), scalar))
    b = Term.apply(OperationSymbol("b", (), other))
    with pytest.raises(RewriteDefinitionError) as sort_mismatch:
        RewriteRule(a, b)
    assert (
        sort_mismatch.value.reason == "pattern and replacement must have the same sort"
    )
    with pytest.raises(RewriteDefinitionError) as bad_rule:
        RewriteRule(object(), a)  # type: ignore[arg-type]
    assert bad_rule.value.reason == "pattern must be an exact Term"
    with pytest.raises(RewriteDefinitionError) as bad_replacement:
        RewriteRule(a, object())  # type: ignore[arg-type]
    assert bad_replacement.value.reason == "replacement must be an exact Term"
    free = Term.variable(Variable("free", scalar))
    with pytest.raises(RewriteDefinitionError) as unbound:
        RewriteRule(a, free)
    assert unbound.value.reason == "replacement variable is not bound by pattern"
    with pytest.raises(RewriteDefinitionError) as bad_system:
        RewriteSystem("not rules")  # type: ignore[arg-type]
    assert (
        bad_system.value.reason
        == "rules must be an iterable of exact RewriteRule values"
    )
    with pytest.raises(RewriteDefinitionError) as non_rule:
        RewriteSystem((object(),))  # type: ignore[arg-type]
    assert (
        non_rule.value.reason == "rules must be an iterable of exact RewriteRule values"
    )
    system = RewriteSystem(())
    for bound in (True, -1, "1"):
        with pytest.raises(RewriteDefinitionError) as bad_bound:
            normalize(a, system, max_steps=bound)  # type: ignore[arg-type]
        assert bad_bound.value.reason == "max_steps must be a non-negative built-in int"
    with pytest.raises(RewriteDefinitionError) as bad_term:
        normalize(cast(Term, object()), system, max_steps=0)
    assert bad_term.value.reason == "term must be an exact Term"
    with pytest.raises(RewriteDefinitionError) as bad_normalizer_system:
        normalize(a, cast(RewriteSystem, object()), max_steps=0)
    assert bad_normalizer_system.value.reason == "system must be an exact RewriteSystem"
    with pytest.raises(RewriteDefinitionError):
        NormalForm(a, ())

    result = NormalForm(a, (a,))
    assert not hasattr(result, "__dict__")
    with pytest.raises(TypeError, match="unhashable"):
        hash(result)
    with pytest.raises(FrozenInstanceError):
        result.term = b  # type: ignore[misc]
    with pytest.raises(RewriteDefinitionError) as bad_bounded_trace:
        BoundExhausted(a, (a,), 1)
    assert (
        bad_bounded_trace.value.reason == "trace length must equal max_steps plus one"
    )
    with pytest.raises(RewriteDefinitionError) as bad_cycle_path:
        CycleDetected(0, (a, a), (a, b, a))
    assert bad_cycle_path.value.reason == "must equal the trace suffix from entry_index"


class _BrokenIterator:
    def __iter__(self) -> _BrokenIterator:
        return self

    def __next__(self) -> RewriteRule:
        raise RuntimeError("fixture")


def test_system_sources_are_snapshotted_and_iteration_failures_are_typed() -> None:
    scalar = Sort("scalar")
    a = Term.apply(OperationSymbol("a", (), scalar))
    b = Term.apply(OperationSymbol("b", (), scalar))
    rules = [RewriteRule(a, b)]
    system = RewriteSystem(rules)
    rules.clear()
    assert normalize(a, system, max_steps=1) == NormalForm(b, (a, b))
    with pytest.raises(RewriteDefinitionError) as failed_iterator:
        RewriteSystem(_BrokenIterator())
    assert (
        failed_iterator.value.reason
        == "rules must be an iterable of exact RewriteRule values"
    )


def test_nested_substitution_deep_terms_and_trace_replay_are_deterministic() -> None:
    scalar = Sort("scalar")
    a = OperationSymbol("a", (), scalar)
    f = OperationSymbol("f", (scalar,), scalar)
    g = OperationSymbol("g", (scalar,), scalar)
    h = OperationSymbol("h", (scalar,), scalar)
    x_variable = Variable("x", scalar)
    x = Term.variable(x_variable)
    source = Term.apply(f, Term.apply(a))
    rule = RewriteRule(Term.apply(f, x), Term.apply(g, Term.apply(h, x)))
    system = RewriteSystem((rule,))
    result = normalize(source, system, max_steps=3)

    expected = Term.apply(g, Term.apply(h, Term.apply(a)))
    assert result == NormalForm(expected, (source, expected))
    replay = normalize(source, system, max_steps=1)
    assert replay == result
    deep = Term.apply(a)
    for _ in range(257):
        deep = Term.apply(f, deep)
    assert normalize(deep, RewriteSystem(()), max_steps=1) == NormalForm(deep, (deep,))


def test_self_loop_prefixed_cycle_bound_before_cycle_and_result_records() -> None:
    scalar = Sort("scalar")
    start_symbol = OperationSymbol("start", (), scalar)
    a_symbol = OperationSymbol("a", (), scalar)
    b_symbol = OperationSymbol("b", (), scalar)
    start = Term.apply(start_symbol)
    a = Term.apply(a_symbol)
    b = Term.apply(b_symbol)
    system = RewriteSystem(
        (RewriteRule(start, a), RewriteRule(a, b), RewriteRule(b, a))
    )
    prefixed = normalize(start, system, max_steps=4)
    assert prefixed == CycleDetected(1, (a, b, a), (start, a, b, a))
    assert normalize(start, system, max_steps=2) == BoundExhausted(b, (start, a, b), 2)
    self_loop = RewriteSystem((RewriteRule(a, a),))
    loop = normalize(a, self_loop, max_steps=1)
    assert loop == CycleDetected(0, (a, a), (a, a))
    for result in (prefixed, loop, BoundExhausted(a, (start, a), 1)):
        assert not hasattr(result, "__dict__")
        with pytest.raises(TypeError, match="unhashable"):
            hash(result)
