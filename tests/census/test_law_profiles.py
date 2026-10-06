"""Complete finite-grid law profiles with typed unsupported results."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from typing import cast

import pytest

from anyalgebra.census.analysis import (
    DisprovedLaw,
    FiniteLawAnalysisError,
    FiniteLawProfile,
    LawResult,
    ProvedLaw,
    UnsupportedLaw,
    analyze_finite_laws,
    law_profile_canonical_bytes,
)
from anyalgebra.census.constraints import CensusConstraint, CensusTerm
from anyalgebra.census.spec import CensusSpecCore


def _core(size: int, arity: int = 2) -> CensusSpecCore:
    return CensusSpecCore.create(
        carrier_size=size,
        arity=arity,
        corpus_name=f"law-profile-{size}-{arity}",
    )


def _by_name(profile: FiniteLawProfile) -> dict[str, LawResult]:
    return {result.name: result for result in profile.results}


def test_cyclic_group_two_has_complete_expected_binary_law_profile() -> None:
    core = _core(2)
    profile = analyze_finite_laws(core, (0, 1, 1, 0))
    laws = _by_name(profile)

    for name in (
        "totality",
        "commutativity",
        "associativity",
        "flexibility",
        "left_alternativity",
        "right_alternativity",
        "alternativity",
    ):
        assert type(laws[name]) is ProvedLaw
    idempotence = laws["idempotence"]
    assert type(idempotence) is DisprovedLaw
    assert idempotence.witness.assignment == (1,)
    assert laws["associativity"].assignment_count == 8
    assert laws["flexibility"].assignment_count == 4
    assert profile.complete is True
    assert profile.total_evaluations == sum(
        result.assignment_count
        for result in profile.results
        if not isinstance(result, UnsupportedLaw)
    )


def test_disproved_laws_retain_lexicographically_smallest_assignment() -> None:
    core = _core(2)
    profile = analyze_finite_laws(core, (0, 0, 1, 0))
    laws = _by_name(profile)
    associativity = laws["associativity"]

    assert type(associativity) is DisprovedLaw
    assert associativity.witness.assignment == (1, 0, 1)
    assert associativity.witness.left_value != associativity.witness.right_value
    assert associativity.assignment_count == 8
    assert associativity.evaluated_count == 8


def test_nonbinary_laws_are_typed_unsupported_not_false() -> None:
    core = _core(3, 1)
    profile = analyze_finite_laws(core, (1, 2, 0))
    laws = _by_name(profile)

    assert type(laws["totality"]) is ProvedLaw
    assert type(laws["idempotence"]) is DisprovedLaw
    for name in (
        "commutativity",
        "associativity",
        "flexibility",
        "left_alternativity",
        "right_alternativity",
        "alternativity",
    ):
        unsupported = laws[name]
        assert type(unsupported) is UnsupportedLaw
        assert unsupported.reason == "requires_binary_operation"


def test_user_equation_uses_full_grid_and_matches_builtin_idempotence() -> None:
    core = _core(2)
    variable = CensusTerm.variable(0)
    equation = CensusConstraint.equation(
        CensusTerm.apply(variable, variable), variable, variable_count=1
    )
    profile = analyze_finite_laws(core, (0, 1, 1, 0), equations=(equation,))
    laws = _by_name(profile)
    custom = laws["equation_000"]

    assert type(custom) is DisprovedLaw
    assert custom.assignment_count == 2
    assert custom.evaluated_count == 2
    builtin = laws["idempotence"]
    assert type(builtin) is DisprovedLaw
    builtin_witness = builtin.witness
    assert custom.witness.assignment == builtin_witness.assignment
    assert custom.witness.left_value == builtin_witness.left_value
    assert custom.witness.right_value == builtin_witness.right_value


def test_complete_counts_cover_full_grids_even_when_failure_is_early() -> None:
    core = _core(2)
    profile = analyze_finite_laws(core, (0, 0, 1, 0))
    laws = _by_name(profile)

    assert laws["associativity"].assignment_count == core.carrier_size**3
    assert laws["commutativity"].assignment_count == core.carrier_size**2
    assert (
        laws["associativity"].evaluated_count == laws["associativity"].assignment_count
    )
    assert profile.declared_grid_count == sum(
        result.assignment_count
        for result in profile.results
        if not isinstance(result, UnsupportedLaw)
    )


def test_nullary_and_empty_carrier_boundaries_are_explicit() -> None:
    nullary = analyze_finite_laws(_core(2, 0), (1,))
    laws = _by_name(nullary)
    assert type(laws["totality"]) is ProvedLaw
    assert laws["totality"].assignment_count == 1
    assert type(laws["idempotence"]) is UnsupportedLaw

    empty = analyze_finite_laws(_core(0, 1), ())
    empty_laws = _by_name(empty)
    assert type(empty_laws["totality"]) is ProvedLaw
    assert type(empty_laws["idempotence"]) is ProvedLaw
    assert empty_laws["idempotence"].assignment_count == 0


def test_invalid_table_non_equation_and_nonexact_core_fail_closed() -> None:
    core = _core(2)
    with pytest.raises(FiniteLawAnalysisError) as caught:
        analyze_finite_laws(core, (0, 0, 0))
    assert caught.value.field == "outputs"
    with pytest.raises(FiniteLawAnalysisError) as caught:
        analyze_finite_laws(
            core,
            (0, 0, 0, 0),
            equations=(CensusConstraint.commutative(),),
        )
    assert caught.value.field == "equations"
    with pytest.raises(FiniteLawAnalysisError) as caught:
        analyze_finite_laws(cast(CensusSpecCore, None), ())
    assert caught.value.field == "core"


def test_profile_is_sealed_canonical_deterministic_and_drift_checked() -> None:
    core = _core(2)
    first_result = analyze_finite_laws(core, (0, 1, 1, 0))
    second_result = analyze_finite_laws(core, (0, 1, 1, 0))
    first = law_profile_canonical_bytes(first_result)
    second = law_profile_canonical_bytes(second_result)
    record = json.loads(first)

    assert first_result == second_result
    assert first == second
    assert record["contentHash"]["digest"] == first_result.semantic_hash.digest
    with pytest.raises(FiniteLawAnalysisError, match="factory-owned"):
        type(first_result)()
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("Derived", (type(first_result),), {})
    with pytest.raises(FrozenInstanceError):
        first_result.complete = False  # type: ignore[misc]

    object.__setattr__(first_result, "total_evaluations", 0)
    with pytest.raises(FiniteLawAnalysisError):
        law_profile_canonical_bytes(first_result)
