"""Contracts for exact conjugation transport of finite operation tables."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from itertools import product
from typing import cast

import pytest

from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.permutations import (
    CarrierPermutation,
    generate_allowed_permutations,
)
from anyalgebra.census.spec import CensusSpecCore
from anyalgebra.census.table_codes import unrank_operation_table
from anyalgebra.census.transport import (
    TableTransportError,
    transport_operation_table,
)


def _core(size: int, arity: int = 2) -> CensusSpecCore:
    return CensusSpecCore.create(
        carrier_size=size,
        arity=arity,
        corpus_name=f"transport-{size}-{arity}",
    )


def _operation(outputs: tuple[int, ...], size: int, arguments: tuple[int, ...]) -> int:
    index = 0
    for argument in arguments:
        index = index * size + argument
    return outputs[index]


def test_transport_equation_is_verified_for_every_cell_exhaustively() -> None:
    core = _core(2)
    policy = EquivalencePolicy.relabeling(core)

    for table_index in range(core.candidate_count):
        source = unrank_operation_table(core, table_index)
        for permutation in generate_allowed_permutations(policy):
            certificate = transport_operation_table(core, source, permutation)
            assert certificate.verify_cells() is True
            for old_inputs in product(range(2), repeat=2):
                new_inputs = tuple(permutation(item) for item in old_inputs)
                assert _operation(certificate.target_outputs, 2, new_inputs) == (
                    permutation(_operation(source, 2, old_inputs))
                )


def test_transport_followed_by_inverse_reproduces_every_source_cell() -> None:
    core = _core(3, 1)
    policy = EquivalencePolicy.relabeling(core)

    for table_index in range(core.candidate_count):
        source = unrank_operation_table(core, table_index)
        for permutation in generate_allowed_permutations(policy):
            forward = transport_operation_table(core, source, permutation)
            backward = transport_operation_table(
                core, forward.target_outputs, permutation.inverse()
            )
            assert backward.target_outputs == source
            assert backward.verify_cells() is True


def test_composition_order_agrees_with_direct_function_conjugation() -> None:
    core = _core(3, 1)
    policy = EquivalencePolicy.relabeling(core)
    source = (1, 2, 0)
    group = generate_allowed_permutations(policy)

    for first, second in product(group, repeat=2):
        step_one = transport_operation_table(core, source, first)
        step_two = transport_operation_table(core, step_one.target_outputs, second)
        direct = transport_operation_table(core, source, first.then(second))
        assert step_two.target_outputs == direct.target_outputs


def test_identity_transport_is_literal_and_certificate_direction_is_explicit() -> None:
    core = _core(2)
    policy = EquivalencePolicy.relabeling(core)
    source = (0, 1, 1, 0)
    certificate = transport_operation_table(
        core, source, CarrierPermutation.identity(policy)
    )

    assert certificate.source_outputs is source
    assert certificate.target_outputs == source
    assert certificate.direction == "old_index_to_new_index_conjugation"
    assert certificate.checked_cell_count == 4


def test_nullary_and_empty_carrier_boundaries() -> None:
    nullary = _core(2, 0)
    swap = CarrierPermutation.create(EquivalencePolicy.relabeling(nullary), (1, 0))
    assert transport_operation_table(nullary, (0,), swap).target_outputs == (1,)

    empty = _core(0, 1)
    identity = CarrierPermutation.identity(EquivalencePolicy.relabeling(empty))
    assert transport_operation_table(empty, (), identity).target_outputs == ()

    impossible = _core(0, 0)
    impossible_identity = CarrierPermutation.identity(
        EquivalencePolicy.relabeling(impossible)
    )
    with pytest.raises(TableTransportError, match="no operation tables"):
        transport_operation_table(impossible, (), impossible_identity)


def test_policy_core_must_be_the_literal_transport_core() -> None:
    core = _core(2)
    other = _core(2)
    permutation = CarrierPermutation.identity(EquivalencePolicy.relabeling(other))

    with pytest.raises(TableTransportError) as caught:
        transport_operation_table(core, (0, 0, 0, 0), permutation)
    assert caught.value.field == "permutation"


@pytest.mark.parametrize(
    "outputs",
    ((0, 0, 0), (0, 0, 0, 0, 0), (0, 0, True, 0), (0, 0, 2, 0)),
)
def test_malformed_source_table_fails_closed(outputs: tuple[int, ...]) -> None:
    core = _core(2)
    identity = CarrierPermutation.identity(EquivalencePolicy.relabeling(core))
    with pytest.raises(TableTransportError) as caught:
        transport_operation_table(core, outputs, identity)
    assert caught.value.field == "source_outputs"


def test_certificate_is_factory_owned_sealed_immutable_and_safely_represented() -> None:
    core = _core(2)
    identity = CarrierPermutation.identity(EquivalencePolicy.relabeling(core))
    certificate = transport_operation_table(core, (0, 0, 0, 0), identity)

    with pytest.raises(TableTransportError, match="factory-owned"):
        type(certificate)()
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("Derived", (type(certificate),), {})
    with pytest.raises(FrozenInstanceError):
        certificate.target_outputs = ()  # type: ignore[misc]
    with pytest.raises(TypeError, match="unhashable"):
        hash(certificate)
    assert repr(certificate) == (
        "TableTransportCertificate(size=2, arity=2, checked_cell_count=4)"
    )


def test_nonexact_inputs_fail_before_table_consumption() -> None:
    with pytest.raises(TableTransportError) as caught:
        transport_operation_table(
            cast(CensusSpecCore, None), (), cast(CarrierPermutation, None)
        )
    assert caught.value.field == "core"
