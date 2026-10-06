"""A ground finite modus-ponens fixture with replayable closure evidence."""

from __future__ import annotations

import json
from typing import TypedDict

from anyalgebra.core.parents import Sort
from anyalgebra.structures.deductive import (
    CompleteClosure,
    DeductiveSystem,
    InferenceRule,
    closure,
)
from anyalgebra.structures.signatures import OperationSymbol
from anyalgebra.structures.terms import Term


class DeductiveExample(TypedDict):
    """A finite rule system, its ground premises, and closure record."""

    closure: CompleteClosure
    premises: tuple[Term, ...]
    system: DeductiveSystem


def _atom(name: str, sort: Sort) -> Term:
    """Construct a ground judgment as a nullary operation term."""
    return Term.apply(OperationSymbol(name, (), sort))


def build_example() -> DeductiveExample:
    """Build and close a ground two-premise modus-ponens-style rule."""
    judgment = Sort("judgment")
    rain = _atom("rain", judgment)
    rain_implies_wet = _atom("rain_implies_wet", judgment)
    wet = _atom("wet", judgment)
    system = DeductiveSystem((InferenceRule((rain, rain_implies_wet), wet),))
    premises = (rain, rain_implies_wet)
    result = closure(premises, system, max_rounds=1)
    if type(result) is not CompleteClosure:
        raise RuntimeError("ground modus ponens unexpectedly exhausted its bound")
    return {"closure": result, "premises": premises, "system": system}


def run_example() -> dict[str, list[int] | list[str] | str]:
    """Verify the trace replay before exposing a compact stable summary."""
    result = build_example()["closure"]
    trace = result.derivations[0]
    replayed_premises = tuple(
        result.conclusions[index] for index in trace.premise_indices
    )
    if replayed_premises != trace.premises:
        raise RuntimeError("modus ponens derivation does not replay")
    conclusion_names = [
        term.symbol.name for term in result.conclusions if term.symbol is not None
    ]
    return {
        "conclusions": conclusion_names,
        "premise_indices": list(trace.premise_indices),
        "status": "complete",
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), sort_keys=True, separators=(",", ":")))
