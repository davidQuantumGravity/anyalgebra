"""Closed-substructure, congruence, and capability-boundary controls."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from itertools import product
import json

import pytest

from anyalgebra.census.spec import CensusSpecCore
from anyalgebra.census.subobjects import (
    FiniteSubobjectAnalysisError,
    FiniteSubobjectBounds,
    analyze_finite_subobjects,
    finite_carrier_derivations,
    finite_carrier_ideals,
    finite_subobject_canonical_bytes,
)
from anyalgebra.structures.outcomes import UnsupportedCapability


def _core(size: int, arity: int = 2) -> CensusSpecCore:
    return CensusSpecCore.create(
        carrier_size=size,
        arity=arity,
        corpus_name=f"subobjects-{size}-{arity}",
    )


def _apply(outputs: tuple[int, ...], size: int, arguments: tuple[int, ...]) -> int:
    index = 0
    for argument in arguments:
        index = index * size + argument
    return outputs[index]


def _block_index(partition: tuple[tuple[int, ...], ...]) -> dict[int, int]:
    return {
        element: block_index
        for block_index, block in enumerate(partition)
        for element in block
    }


def test_cyclic_group_two_has_expected_substructures_and_congruences() -> None:
    core = _core(2)
    result = analyze_finite_subobjects(core, (0, 1, 1, 0))

    assert result.substructures == ((0,), (0, 1))
    assert result.congruences == (((0,), (1,)), ((0, 1),))
    assert result.substructure_status == result.congruence_status == "complete"
    assert result.ideal_status == result.derivation_status == "unsupported_no_module"
    assert result.subset_candidates_examined == 3
    assert result.partition_candidates_examined == 2
    assert result.complete is True
    assert result.hypotheses[-1] == "no additive or scalar module structure is present"


def test_every_returned_object_satisfies_all_defining_cells() -> None:
    core = _core(3)
    outputs = tuple(left for left in range(3) for _right in range(3))
    result = analyze_finite_subobjects(core, outputs)

    assert len(result.substructures) == 7
    assert len(result.congruences) == 5
    carrier_tuples = tuple(product(range(3), repeat=2))
    for subset in result.substructures:
        for arguments in product(subset, repeat=2):
            assert _apply(outputs, 3, arguments) in subset
    for partition in result.congruences:
        classes = _block_index(partition)
        for left in carrier_tuples:
            for right in carrier_tuples:
                if all(
                    classes[x] == classes[y] for x, y in zip(left, right, strict=True)
                ):
                    assert (
                        classes[_apply(outputs, 3, left)]
                        == classes[_apply(outputs, 3, right)]
                    )


def test_nullary_and_unary_closure_are_not_mistaken_for_binary_closure() -> None:
    nullary = analyze_finite_subobjects(_core(2, 0), (1,))
    unary = analyze_finite_subobjects(_core(2, 1), (1, 0))

    assert nullary.substructures == ((1,), (0, 1))
    assert nullary.congruences == (((0,), (1,)), ((0, 1),))
    assert unary.substructures == ((0, 1),)
    assert unary.congruences == (((0,), (1,)), ((0, 1),))


def test_bare_carrier_ideal_and_derivation_requests_are_unsupported() -> None:
    core = _core(2)
    outputs = (0, 1, 1, 0)
    with pytest.raises(UnsupportedCapability) as ideal:
        finite_carrier_ideals(core, outputs)
    with pytest.raises(UnsupportedCapability) as derivation:
        finite_carrier_derivations(core, outputs)

    assert ideal.value.capability == "linear-ideals"
    assert derivation.value.capability == "linear-derivations"
    assert ideal.value.context == derivation.value.context == "bare finite carrier"


def test_empty_carrier_has_no_nonempty_substructure_and_one_congruence() -> None:
    result = analyze_finite_subobjects(_core(0), ())

    assert result.substructures == ()
    assert result.congruences == ((),)
    assert result.subset_candidates_examined == 0
    assert result.partition_candidates_examined == 1


def test_bounds_fail_before_enumeration_and_reject_malformed_values() -> None:
    core = _core(3)
    outputs = tuple(0 for _ in range(9))
    with pytest.raises(FiniteSubobjectAnalysisError) as caught:
        analyze_finite_subobjects(
            core,
            outputs,
            options=FiniteSubobjectBounds(max_subset_candidates=6),
        )
    assert caught.value.field == "bounds"
    with pytest.raises(FiniteSubobjectAnalysisError) as caught:
        FiniteSubobjectBounds(max_partition_candidates=True)
    assert caught.value.field == "max_partition_candidates"


def test_invalid_core_and_table_fail_closed() -> None:
    with pytest.raises(FiniteSubobjectAnalysisError) as caught:
        analyze_finite_subobjects(None, ())  # type: ignore[arg-type]
    assert caught.value.field == "core"
    with pytest.raises(FiniteSubobjectAnalysisError) as caught:
        analyze_finite_subobjects(_core(2), (0, 0, 0))
    assert caught.value.field == "outputs"


def test_result_is_sealed_canonical_and_replayed_before_serialization() -> None:
    core = _core(2)
    outputs = (0, 1, 1, 0)
    first_result = analyze_finite_subobjects(core, outputs)
    second_result = analyze_finite_subobjects(core, outputs)
    first = finite_subobject_canonical_bytes(first_result)

    assert first_result == second_result
    assert first == finite_subobject_canonical_bytes(second_result)
    assert (
        json.loads(first)["contentHash"]["digest"] == first_result.semantic_hash.digest
    )
    with pytest.raises(FiniteSubobjectAnalysisError, match="factory-owned"):
        type(first_result)()
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("Derived", (type(first_result),), {})
    with pytest.raises(FrozenInstanceError):
        first_result.complete = False  # type: ignore[misc]
    object.__setattr__(first_result, "substructures", ())
    with pytest.raises(FiniteSubobjectAnalysisError, match="content drift"):
        finite_subobject_canonical_bytes(first_result)
