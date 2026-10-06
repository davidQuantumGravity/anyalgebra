"""Exercise defensive finite-structure, term, and rewrite boundaries."""

from __future__ import annotations

from collections.abc import ItemsView, Iterator, Mapping
from typing import cast

import pytest

import anyalgebra.structures.deductive as deductive
import anyalgebra.structures.evaluate as evaluation
import anyalgebra.structures.operations as operations
import anyalgebra.structures.partiality as partiality
import anyalgebra.structures.relations as relations
import anyalgebra.structures.rewrite as rewrite
from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.structures.deductive import (
    CompleteClosure,
    DeductiveDefinitionError,
    Derivation,
)
from anyalgebra.structures.evaluate import (
    EvaluationDefinitionError,
    EvaluationResult,
    evaluate_term,
)
from anyalgebra.structures.laws import Equation, Law, LawDefinitionError
from anyalgebra.structures.operations import (
    Operation,
    OperationApplicationError,
    OperationDefinitionError,
    PartialOperation,
    PartialOperationApplicationError,
    PartialOperationDefinitionError,
)
from anyalgebra.structures.outcomes import Defined, UnsupportedCapability
from anyalgebra.structures.partiality import TotalizationDefinitionError
from anyalgebra.structures.relations import Relation, RelationDefinitionError
from anyalgebra.structures.rewrite import (
    BoundExhausted,
    CycleDetected,
    NormalForm,
    RewriteDefinitionError,
    RewriteSystem,
)
from anyalgebra.structures.signatures import OperationSymbol, RelationSymbol, Signature
from anyalgebra.structures.structure import (
    Structure,
    StructureBuilder,
    StructureDefinitionError,
)
from anyalgebra.structures.terms import Term, Variable


def _atom(name: str, sort: Sort) -> Term:
    return Term.apply(OperationSymbol(name, (), sort))


def test_deductive_evidence_factories_reject_each_malformed_shape() -> None:
    sort = Sort("s")
    atom = _atom("a", sort)
    variable = Term.variable(Variable("x", sort))
    with pytest.raises(DeductiveDefinitionError, match="derivation"):
        Derivation._make(cast(Term, object()), 0, (), ())
    with pytest.raises(DeductiveDefinitionError, match="closure evidence"):
        CompleteClosure._make(cast(tuple[Term, ...], []), ())
    with pytest.raises(DeductiveDefinitionError, match="ground Terms"):
        CompleteClosure._make((variable,), ())
    with pytest.raises(DeductiveDefinitionError, match="Derivation"):
        CompleteClosure._make((atom,), cast(tuple[Derivation, ...], (object(),)))
    derivation = Derivation._make(atom, 0, (), ())
    with pytest.raises(DeductiveDefinitionError, match="more derivations"):
        CompleteClosure._make((), (derivation,))


def test_deductive_term_comparison_failure_is_typed() -> None:
    class Exploding:
        def __eq__(self, other: object) -> bool:
            del other
            raise RuntimeError("private")

    with pytest.raises(DeductiveDefinitionError, match="comparison failed"):
        deductive._same_term(cast(Term, Exploding()), cast(Term, object()))


class _BadEnvironment(Mapping[Variable, object]):
    def __init__(self, entries: tuple[object, ...], *, raises: bool = False) -> None:
        self.entries = entries
        self.raises = raises

    def __getitem__(self, key: Variable) -> object:
        raise KeyError(key)

    def __iter__(self) -> Iterator[Variable]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self) -> ItemsView[Variable, object]:
        if self.raises:
            raise RuntimeError("private")
        return cast(ItemsView[Variable, object], self.entries)


def _empty_structure(sort: Sort) -> Structure:
    return (
        StructureBuilder(Signature((sort,)))
        .with_carrier(sort, FiniteCarrier((0,), sort=sort))
        .freeze()
    )


def test_evaluation_result_and_environment_snapshot_guards() -> None:
    with pytest.raises(TypeError, match="exact tuple"):
        EvaluationResult(Defined(0), cast(tuple[int, ...], []))
    sort = Sort("s")
    variable = Variable("x", sort)
    term = Term.variable(variable)
    structure = _empty_structure(sort)
    with pytest.raises(EvaluationDefinitionError, match="snapshotted"):
        evaluate_term(structure, term, _BadEnvironment((), raises=True))
    with pytest.raises(EvaluationDefinitionError, match="pairs"):
        evaluate_term(structure, term, _BadEnvironment(((variable, 0, 1),)))
    with pytest.raises(EvaluationDefinitionError, match="duplicate"):
        evaluate_term(
            structure,
            term,
            _BadEnvironment(((variable, 0), (Variable("x", sort), 0))),
        )


def test_evaluator_converts_a_corrupt_operation_outcome_to_failed() -> None:
    sort = Sort("s")
    symbol = OperationSymbol("unit", (), sort)
    carrier = FiniteCarrier((0,), sort=sort)
    operation = Operation.from_table(symbol, ((sort, carrier),), (((), 0),))
    structure = (
        StructureBuilder(Signature((sort,), operations=(symbol,)))
        .with_carrier(sort, carrier)
        .with_operation(symbol, operation)
        .freeze()
    )

    class InvalidOperation:
        def apply(self, *_arguments: object) -> object:
            return object()

    object.__setattr__(structure, "operations", (InvalidOperation(),))
    result = evaluation._evaluate(structure, Term.apply(symbol), ())
    assert type(result.outcome).__name__ == "Failed"


def test_law_hypothesis_snapshot_normalizes_iterator_failure() -> None:
    sort = Sort("s")
    atom = _atom("a", sort)

    def broken() -> Iterator[Equation]:
        raise RuntimeError("private")
        yield

    with pytest.raises(LawDefinitionError, match="iterable"):
        Law("law", (), Equation(atom, atom), hypotheses=broken())


class _ExplodingEquality:
    def __eq__(self, other: object) -> bool:
        del other
        raise RuntimeError("private")


def test_operation_snapshot_binding_lookup_and_equality_guards() -> None:
    sort = Sort("s")
    symbol = OperationSymbol("op", (sort,), sort)
    carrier = FiniteCarrier((0,), sort=sort)
    with pytest.raises(OperationDefinitionError, match="iterable"):
        Operation.from_table(symbol, ((sort, carrier),), "bad")  # type: ignore[arg-type]
    with pytest.raises(OperationDefinitionError, match="snapshotted"):
        Operation.from_table(symbol, ((sort, carrier),), object())  # type: ignore[arg-type]
    with pytest.raises(OperationDefinitionError, match="exact Sort"):
        Operation.from_table(symbol, ((object(), carrier),), ())  # type: ignore[arg-type]
    with pytest.raises(AssertionError, match="missing"):
        operations._carrier_for_sort(((sort, carrier),), Sort("other"))

    hostile = FiniteCarrier((_ExplodingEquality(),), sort=sort)
    with pytest.raises(OperationDefinitionError, match="equality comparison"):
        Operation.from_table(
            symbol, ((sort, hostile),), (((object(),), hostile.items[0]),)
        )
    with pytest.raises(OperationDefinitionError, match="exact OperationSymbol"):
        Operation.from_table(cast(OperationSymbol, object()), (), ())
    with pytest.raises(PartialOperationDefinitionError, match="exact OperationSymbol"):
        PartialOperation.from_table(
            cast(OperationSymbol, object()), (), (), undefined_marker=object()
        )


def test_operation_internal_and_callable_guard_paths() -> None:
    sort = Sort("s")
    symbol = OperationSymbol("op", (sort,), sort)
    carrier = FiniteCarrier((0,), sort=sort)
    operation = Operation.from_table(symbol, ((sort, carrier),), (((0,), 0),))
    object.__setattr__(operation, "table", ())
    with pytest.raises(AssertionError, match="lost"):
        operation.apply(0)
    with pytest.raises(OperationDefinitionError, match="from_callable"):
        operations._CallableOperation()
    with pytest.raises(PartialOperationDefinitionError, match="from_callable"):
        operations._CallablePartialOperation()

    hostile = FiniteCarrier((_ExplodingEquality(),), sort=sort)
    callable_operation = Operation.from_callable(
        symbol, ((sort, hostile),), lambda x: x
    )
    with pytest.raises(OperationApplicationError, match="equality comparison"):
        callable_operation.apply(object())
    foreign_result = Operation.from_callable(
        symbol, ((sort, carrier),), lambda _x: Defined(1)
    )
    assert type(foreign_result.apply(0)).__name__ == "Failed"

    partial_callable = PartialOperation.from_callable(
        symbol, ((sort, hostile),), lambda x: Defined(x)
    )
    assert partial_callable == partial_callable
    with pytest.raises(PartialOperationApplicationError, match="equality comparison"):
        partial_callable.apply(object())


def test_relation_snapshot_binding_lookup_and_symbol_guards() -> None:
    sort = Sort("s")
    symbol = RelationSymbol("r", (sort,))
    carrier = FiniteCarrier((0,), sort=sort)
    with pytest.raises(RelationDefinitionError, match="iterable"):
        Relation.from_tuples(symbol, ((sort, carrier),), "bad")  # type: ignore[arg-type]
    with pytest.raises(RelationDefinitionError, match="snapshotted"):
        Relation.from_tuples(symbol, ((sort, carrier),), object())  # type: ignore[arg-type]
    with pytest.raises(RelationDefinitionError, match="entry must be a pair"):
        Relation.from_tuples(symbol, ((sort, carrier, "extra"),), ())  # type: ignore[arg-type]
    with pytest.raises(RelationDefinitionError, match="exact Sort"):
        Relation.from_tuples(symbol, ((object(), carrier),), ())  # type: ignore[arg-type]
    with pytest.raises(RelationDefinitionError, match="FiniteCarrier"):
        Relation.from_tuples(symbol, ((sort, object()),), ())  # type: ignore[arg-type]
    with pytest.raises(AssertionError, match="missing"):
        relations._carrier_for_sort(((sort, carrier),), Sort("other"))
    with pytest.raises(RelationDefinitionError, match="exact RelationSymbol"):
        Relation.from_tuples(cast(RelationSymbol, object()), (), ())
    with pytest.raises(RelationDefinitionError, match="exact RelationSymbol"):
        Relation.from_predicate(cast(RelationSymbol, object()), (), lambda: True)
    with pytest.raises(RelationDefinitionError, match="from_predicate"):
        relations._CallableRelation()


def test_totalization_embedding_and_lookup_guards() -> None:
    sort = Sort("s")
    hostile = FiniteCarrier((_ExplodingEquality(),), sort=sort)
    target = FiniteCarrier((0,), sort=sort)
    embedding = partiality.TotalizationEmbedding._from_validated(
        hostile, target, target.items[-1]
    )
    with pytest.raises(TotalizationDefinitionError, match="equality comparison"):
        embedding.embed(object())
    with pytest.raises(AssertionError, match="missing"):
        partiality._carrier_for_sort(((sort, hostile),), Sort("other"))
    with pytest.raises(TotalizationDefinitionError, match="use totalize"):
        partiality.TotalizationEmbedding()


def test_rewrite_result_guards_and_private_match_paths() -> None:
    sort = Sort("s")
    other = Sort("other")
    atom = _atom("a", sort)
    second = _atom("b", sort)
    foreign = _atom("a", other)
    with pytest.raises(RewriteDefinitionError, match="exact Term"):
        NormalForm(cast(Term, object()), (atom,))
    with pytest.raises(RewriteDefinitionError, match="end at term"):
        NormalForm(atom, (second,))
    with pytest.raises(RewriteDefinitionError, match="exact Term"):
        BoundExhausted(cast(Term, object()), (atom,), 0)
    with pytest.raises(RewriteDefinitionError, match="end at last_term"):
        BoundExhausted(atom, (second,), 0)
    with pytest.raises(RewriteDefinitionError, match="entry_index"):
        CycleDetected(True, (atom, atom), (atom, atom))
    with pytest.raises(RewriteDefinitionError, match="indexed trace"):
        CycleDetected(1, (atom, atom), (atom, second))
    with pytest.raises(RewriteDefinitionError, match="closed"):
        CycleDetected(0, (atom, second), (atom, second))
    assert not rewrite._match(atom, foreign, {})
    variable = Term.variable(Variable("x", sort))
    assert rewrite._rewrite_once(variable, RewriteSystem(())) is None


def test_structure_rejects_relation_symbol_and_extra_binding_sort() -> None:
    sort = Sort("s")
    other = Sort("other")
    symbol = RelationSymbol("r", (sort,))
    wrong = RelationSymbol("wrong", (sort,))
    carrier = FiniteCarrier((0,), sort=sort)
    wrong_relation = Relation.from_tuples(wrong, ((sort, carrier),), ())
    builder = StructureBuilder(Signature((sort,), relations=(symbol,))).with_carrier(
        sort, carrier
    )
    with pytest.raises(StructureDefinitionError, match="symbol does not match"):
        builder.with_relation(symbol, wrong_relation)

    operation_symbol = OperationSymbol("op", (sort,), sort)
    other_carrier = FiniteCarrier((0,), sort=other)
    operation = Operation.from_table(
        operation_symbol,
        ((sort, carrier), (other, other_carrier)),
        (((0,), 0),),
    )
    builder = StructureBuilder(Signature((sort,), operations=(operation_symbol,)))
    with pytest.raises(StructureDefinitionError, match="sort is not declared"):
        builder.with_operation(operation_symbol, operation)


def test_unsupported_capability_accepts_absent_context() -> None:
    error = UnsupportedCapability("feature")
    assert error.context is None
