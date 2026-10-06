"""Contract tests for bounded finite ground deductive closure."""

from __future__ import annotations
from collections.abc import Iterator
from dataclasses import FrozenInstanceError
import pytest
from anyalgebra.core.parents import Sort
from anyalgebra.structures.deductive import (
    BoundExhausted,
    CompleteClosure,
    DeductiveDefinitionError,
    Derivation,
    DeductiveSystem,
    InferenceRule,
    closure,
)
from anyalgebra.structures.signatures import OperationSymbol
from anyalgebra.structures.terms import Term, Variable


def _atom(name: str, sort: Sort) -> Term:
    return Term.apply(OperationSymbol(name, (), sort))


def test_ground_axioms_joins_order_and_replayable_derivations() -> None:
    s = Sort("s")
    a, b, c, d = (_atom(n, s) for n in "abcd")
    system = DeductiveSystem(
        (
            InferenceRule((), a),
            InferenceRule((a,), b),
            InferenceRule((a, b), c),
            InferenceRule((c,), d),
        )
    )
    result = closure((), system, max_rounds=4)
    assert isinstance(result, CompleteClosure)
    assert result.conclusions == (a, b, c, d)
    assert tuple(item.conclusion for item in result.derivations) == (a, b, c, d)
    assert result.derivations[2].premise_indices == (0, 1)


def test_bound_zero_distinguishes_closed_from_remaining_frontier_and_cycles() -> None:
    s = Sort("s")
    a, b = (_atom(n, s) for n in "ab")
    system = DeductiveSystem((InferenceRule((a,), b), InferenceRule((b,), a)))
    zero = closure((a,), system, max_rounds=0)
    assert isinstance(zero, BoundExhausted)
    assert (zero.conclusions, zero.derivations, zero.max_rounds) == ((a,), (), 0)
    assert isinstance(closure((a, b), system, max_rounds=0), CompleteClosure)
    one_round = closure((a,), system, max_rounds=1)
    assert isinstance(one_round, CompleteClosure)
    assert one_round.conclusions == (a, b)


def test_synchronous_rounds_bound_before_chain_completion() -> None:
    s = Sort("s")
    a, b, c = (_atom(name, s) for name in "abc")
    system = DeductiveSystem((InferenceRule((a,), b), InferenceRule((b,), c)))
    exhausted = closure((a,), system, max_rounds=1)
    assert isinstance(exhausted, BoundExhausted)
    assert exhausted.conclusions == (a, b)
    complete = closure((a,), system, max_rounds=2)
    assert isinstance(complete, CompleteClosure)
    assert complete.conclusions == (a, b, c)


def test_nullary_high_premise_snapshots_validation_and_records() -> None:
    s = Sort("s")
    atoms = tuple(_atom(f"a{i}", s) for i in range(257))
    goal = _atom("goal", s)
    rule = InferenceRule(atoms, goal)
    rules = [rule]
    system = DeductiveSystem(rules)
    rules.clear()
    assert closure(atoms, system, max_rounds=1).conclusions[-1] == goal
    with pytest.raises(DeductiveDefinitionError):
        InferenceRule((object(),), goal)  # type: ignore[arg-type]
    with pytest.raises(DeductiveDefinitionError):
        closure(atoms, system, max_rounds=True)
    with pytest.raises(DeductiveDefinitionError):
        DeductiveSystem("bad")  # type: ignore[arg-type]
    result = closure((), DeductiveSystem(()), max_rounds=0)
    assert not hasattr(result, "__dict__")
    with pytest.raises(TypeError, match="unhashable"):
        hash(result)
    with pytest.raises(FrozenInstanceError):
        result.conclusions = ()  # type: ignore[misc]
    with pytest.raises(DeductiveDefinitionError):
        BoundExhausted((), (), 0)


def test_closure_uses_snapshot_order_and_one_round_visibility() -> None:
    s = Sort("s")
    a, b, c, d = (_atom(name, s) for name in "abcd")
    first = InferenceRule((a,), c)
    second = InferenceRule((a,), b)
    delayed = InferenceRule((b,), d)
    rules = [first, second, delayed]
    system = DeductiveSystem(iter(rules))
    rules[:] = []

    result = closure((a, a), system, max_rounds=2)

    assert isinstance(result, CompleteClosure)
    assert result.conclusions == (a, c, b, d)
    assert [trace.rule_index for trace in result.derivations] == [0, 1, 2]
    assert result.derivations[2].premise_indices == (2,)


def test_replayable_traces_keep_repeated_premises_and_join_indices() -> None:
    s = Sort("s")
    a, b, c = (_atom(name, s) for name in "abc")
    system = DeductiveSystem(
        (
            InferenceRule((a, a), b),
            InferenceRule((a, b), c),
        )
    )

    result = closure((a,), system, max_rounds=2)

    assert isinstance(result, CompleteClosure)
    first, second = result.derivations
    assert first.premises == (a, a)
    assert first.premise_indices == (0, 0)
    assert second.premise_indices == (0, 1)
    for offset, trace in enumerate(result.derivations, start=1):
        assert trace.conclusion == result.conclusions[offset]
        assert all(index < offset for index in trace.premise_indices)
        assert tuple(result.conclusions[index] for index in trace.premise_indices) == (
            trace.premises
        )


def test_empty_system_and_axioms_distinguish_complete_from_bound_exhausted() -> None:
    s = Sort("s")
    axiom = _atom("axiom", s)
    empty = DeductiveSystem(())

    assert closure((), empty, max_rounds=0) == CompleteClosure._make((), ())
    exhausted = closure((), DeductiveSystem((InferenceRule((), axiom),)), max_rounds=0)
    assert isinstance(exhausted, BoundExhausted)
    assert exhausted.conclusions == ()
    assert exhausted.derivations == ()
    completed = closure((), DeductiveSystem((InferenceRule((), axiom),)), max_rounds=1)
    assert isinstance(completed, CompleteClosure)
    assert completed.conclusions == (axiom,)


def test_cycles_and_duplicate_rules_do_not_repeat_conclusions_or_traces() -> None:
    s = Sort("s")
    a, b = (_atom(name, s) for name in "ab")
    rule = InferenceRule((a,), b)
    system = DeductiveSystem((rule, rule, InferenceRule((b,), a)))

    result = closure((a,), system, max_rounds=8)

    assert isinstance(result, CompleteClosure)
    assert result.conclusions == (a, b)
    assert len(result.derivations) == 1
    assert result.derivations[0].rule_index == 0


def test_rejects_variables_and_non_terms_in_every_ground_input_position() -> None:
    s = Sort("s")
    a = _atom("a", s)
    variable = Term.variable(Variable("x", s))
    with pytest.raises(DeductiveDefinitionError, match="premises"):
        InferenceRule((variable,), a)
    with pytest.raises(DeductiveDefinitionError, match="conclusion"):
        InferenceRule((), variable)
    with pytest.raises(DeductiveDefinitionError, match="premises"):
        closure((variable,), DeductiveSystem(()), max_rounds=0)
    with pytest.raises(DeductiveDefinitionError, match="conclusion"):
        InferenceRule((), object())  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_bound", (-1, 1.0, True, "1", None))
def test_rejects_malformed_bounds_before_search(bad_bound: object) -> None:
    s = Sort("s")
    a = _atom("a", s)
    with pytest.raises(DeductiveDefinitionError, match="max_rounds"):
        closure((a,), DeductiveSystem(()), max_rounds=bad_bound)  # type: ignore[arg-type]


def test_rejects_malformed_iterables_systems_and_factory_bypass() -> None:
    s = Sort("s")
    a = _atom("a", s)
    rule = InferenceRule((), a)
    with pytest.raises(DeductiveDefinitionError, match="rules"):
        DeductiveSystem((rule, object()))  # type: ignore[arg-type]
    with pytest.raises(DeductiveDefinitionError, match="premises"):
        InferenceRule("not terms", a)  # type: ignore[arg-type]
    with pytest.raises(DeductiveDefinitionError, match="system"):
        closure((), object(), max_rounds=0)  # type: ignore[arg-type]
    for record, arguments in (
        (Derivation, ()),
        (CompleteClosure, ()),
        (BoundExhausted, ()),
    ):
        with pytest.raises(DeductiveDefinitionError):
            record(*arguments)


def test_normalizes_broken_one_pass_iterators_and_noniterable_premises() -> None:
    s = Sort("s")
    a = _atom("a", s)
    rule = InferenceRule((), a)

    def broken_terms() -> Iterator[Term]:
        yield a
        raise RuntimeError("iterator failed")

    def broken_rules() -> Iterator[InferenceRule]:
        yield rule
        raise RuntimeError("iterator failed")

    with pytest.raises(DeductiveDefinitionError, match="premises"):
        InferenceRule(broken_terms(), a)
    with pytest.raises(DeductiveDefinitionError, match="rules"):
        DeductiveSystem(broken_rules())
    with pytest.raises(DeductiveDefinitionError, match="premises"):
        closure(object(), DeductiveSystem(()), max_rounds=0)  # type: ignore[arg-type]


def test_evidence_records_are_slots_frozen_and_unhashable() -> None:
    s = Sort("s")
    a, b = (_atom(name, s) for name in "ab")
    result = closure((a,), DeductiveSystem((InferenceRule((a,), b),)), max_rounds=1)
    assert isinstance(result, CompleteClosure)
    trace = result.derivations[0]
    for value in (trace, result):
        assert not hasattr(value, "__dict__")
        with pytest.raises(TypeError, match="unhashable"):
            hash(value)
    with pytest.raises(FrozenInstanceError):
        trace.rule_index = 3  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.derivations = ()  # type: ignore[misc]


def test_internal_evidence_validation_rejects_duplicate_and_nonreplayable_data() -> (
    None
):
    s = Sort("s")
    a, b = (_atom(name, s) for name in "ab")
    with pytest.raises(DeductiveDefinitionError, match="unique"):
        CompleteClosure._make((a, a), ())
    with pytest.raises(DeductiveDefinitionError, match="suffix"):
        CompleteClosure._make((a, b), (Derivation._make(a, 0, (a,), (0,)),))
    with pytest.raises(DeductiveDefinitionError, match="premise"):
        CompleteClosure._make((a, b), (Derivation._make(b, 0, (b,), (1,)),))
    with pytest.raises(DeductiveDefinitionError, match="max_rounds"):
        BoundExhausted._make((a,), (), True)
