"""A finite two-sorted structure with a ternary operation and relation."""

from __future__ import annotations

import json
from typing import TypedDict

from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.structures.evaluate import EvaluationResult, evaluate_term
from anyalgebra.structures.operations import Operation
from anyalgebra.structures.outcomes import Defined
from anyalgebra.structures.relations import Relation, RelationResult
from anyalgebra.structures.signatures import OperationSymbol, RelationSymbol, Signature
from anyalgebra.structures.structure import Structure, StructureBuilder
from anyalgebra.structures.terms import Term, Variable


class TwoSortedExample(TypedDict):
    """Publicly constructed components and one evaluated term fixture."""

    evaluation: EvaluationResult
    operation: Operation
    relation: Relation
    structure: Structure


def build_example() -> TwoSortedExample:
    """Assemble a two-sorted structure through its public builder API."""
    point = Sort("point")
    color = Sort("color")
    points = FiniteCarrier(("left", "right"), sort=point)
    colors = FiniteCarrier(("red", "blue"), sort=color)
    choose = OperationSymbol("choose", (point, color, point), point)
    has_color = RelationSymbol("has_color", (point, color))
    signature = Signature((point, color), (choose,), (has_color,))
    operation = Operation.from_table(
        choose,
        ((point, points), (color, colors)),
        tuple(
            ((first, shade, second), second if shade == "blue" else first)
            for first in points
            for shade in colors
            for second in points
        ),
    )
    relation = Relation.from_tuples(
        has_color,
        ((point, points), (color, colors)),
        (("left", "red"), ("right", "blue")),
    )
    structure = (
        StructureBuilder(signature)
        .with_carrier(point, points)
        .with_carrier(color, colors)
        .with_operation(choose, operation)
        .with_relation(has_color, relation)
        .freeze()
    )
    first = Variable("first", point)
    shade = Variable("shade", color)
    second = Variable("second", point)
    term = Term.apply(
        choose,
        Term.variable(first),
        Term.variable(shade),
        Term.variable(second),
    )
    evaluation = evaluate_term(
        structure,
        term,
        {first: "left", shade: "blue", second: "right"},
    )
    return {
        "evaluation": evaluation,
        "operation": operation,
        "relation": relation,
        "structure": structure,
    }


def run_example() -> dict[str, bool | int | str]:
    """Evaluate the ternary term and query true and false relation tuples."""
    example = build_example()
    evaluation = example["evaluation"]
    relation = example["relation"]
    relation_true = relation.apply("right", "blue")
    relation_false = relation.apply("left", "blue")
    if type(evaluation.outcome) is not Defined:
        raise RuntimeError("two-sorted ternary evaluation did not produce a value")
    if type(evaluation.outcome.value) is not str:
        raise RuntimeError("two-sorted ternary evaluation has an unexpected value")
    if (
        type(relation_true) is not RelationResult
        or type(relation_false) is not RelationResult
    ):
        raise RuntimeError("two-sorted relation did not produce a Boolean result")
    return {
        "evaluation": evaluation.outcome.value,
        "relation_false": relation_false.holds,
        "relation_true": relation_true.holds,
        "ternary_arity": example["operation"].symbol.arity,
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), sort_keys=True, separators=(",", ":")))
