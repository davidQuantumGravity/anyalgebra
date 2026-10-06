"""Contract tests for immutable evaluation outcomes and unwrapping boundaries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError
from typing import get_args, get_origin

import pytest

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.structures.outcomes import (
    CapabilityDefinitionError,
    Defined,
    EvaluationOutcome,
    Failed,
    FailedOutcomeError,
    Indeterminate,
    IndeterminateOutcomeError,
    InvalidEvaluationOutcomeError,
    Undefined,
    UndefinedOutcomeError,
    UnsupportedCapability,
    unwrap_defined,
)


class _StringSubclass(str):
    """A string subtype used to prove exact built-in-string validation."""


def test_defined_preserves_none_and_is_a_distinct_frozen_slot_backed_branch() -> None:
    defined = Defined(None)

    assert defined.value is None
    assert type(defined) is Defined
    assert type(Undefined("outside domain")) is Undefined
    assert type(Indeterminate("search limit")) is Indeterminate
    assert type(Failed("validation-error")) is Failed
    assert defined == Defined(None)
    assert hash(defined) == hash(Defined(None))
    assert not hasattr(defined, "__dict__")
    with pytest.raises(FrozenInstanceError):
        defined.value = 1  # type: ignore[assignment, misc]


def test_outcomes_preserve_literal_payloads_and_support_explicit_matching() -> None:
    witness = ["left", "right"]
    bounds = {"steps": 0}
    undefined = Undefined("outside domain", witness)
    indeterminate = Indeterminate("search limit", bounds)
    failed = Failed("validation-error", stage="decode")

    match undefined:
        case Undefined(reason=reason, witness=matched_witness):
            assert reason == "outside domain"
            assert matched_witness is witness
        case _:
            pytest.fail("Undefined did not match its explicit branch")
    match indeterminate:
        case Indeterminate(reason=reason, bounds=matched_bounds):
            assert reason == "search limit"
            assert matched_bounds is bounds
        case _:
            pytest.fail("Indeterminate did not match its explicit branch")
    match failed:
        case Failed(error=error, stage=stage):
            assert (error, stage) == ("validation-error", "decode")
        case _:
            pytest.fail("Failed did not match its explicit branch")

    assert not hasattr(undefined, "__dict__")
    assert not hasattr(indeterminate, "__dict__")
    assert not hasattr(failed, "__dict__")
    with pytest.raises(FrozenInstanceError):
        undefined.reason = "other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        indeterminate.reason = "other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        failed.error = "other"  # type: ignore[misc]


def test_nondefined_branches_are_structural_when_payloads_support_hashing() -> None:
    undefined = Undefined("outside domain", witness=("left", "right"))
    indeterminate = Indeterminate("search limit", bounds=("steps", 0))
    failed = Failed("validation-error", stage="decode")

    assert undefined == Undefined("outside domain", witness=("left", "right"))
    assert indeterminate == Indeterminate("search limit", bounds=("steps", 0))
    assert failed == Failed("validation-error", stage="decode")
    assert hash(undefined) == hash(Undefined("outside domain", ("left", "right")))
    assert hash(indeterminate) == hash(Indeterminate("search limit", ("steps", 0)))
    assert hash(failed) == hash(Failed("validation-error", stage="decode"))
    assert Undefined("outside domain") == Undefined("outside domain", witness=None)
    assert Indeterminate("search limit") == Indeterminate("search limit", bounds=None)
    assert Failed("validation-error") == Failed("validation-error", stage=None)
    indeterminate_as_object: object = indeterminate
    failed_as_object: object = failed
    defined_as_object: object = Defined("validation-error")
    assert undefined != indeterminate_as_object
    assert indeterminate != failed_as_object
    assert failed != defined_as_object


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (lambda: Undefined(" reason"), "reason"),
        (lambda: Undefined(""), "reason"),
        (lambda: Undefined(_StringSubclass("reason")), "reason"),
        (lambda: Indeterminate("reason "), "reason"),
        (lambda: Indeterminate(""), "reason"),
        (lambda: Failed(" error"), "error"),
        (lambda: Failed(""), "error"),
        (lambda: Failed(_StringSubclass("error")), "error"),
        (lambda: Failed("error", stage=" stage"), "stage"),
        (lambda: Failed("error", stage=""), "stage"),
        (lambda: Failed("error", stage=_StringSubclass("stage")), "stage"),
    ],
)
def test_outcome_metadata_rejects_unstable_strings(
    factory: Callable[[], object], field: str
) -> None:
    with pytest.raises(InvalidEvaluationOutcomeError) as raised:
        factory()

    assert raised.value.field == field
    assert "0x" not in str(raised.value)


def test_failed_rejects_live_exception_payloads_and_does_not_raise_them() -> None:
    with pytest.raises(InvalidEvaluationOutcomeError) as raised:
        Failed(ValueError("boom"))  # type: ignore[arg-type]

    assert raised.value.field == "error"
    failure = Failed("fixture fault", stage="evaluate")
    with pytest.raises(FailedOutcomeError) as unwrapped:
        unwrap_defined(failure)
    assert unwrapped.value.outcome is failure
    assert str(unwrapped.value) == "cannot unwrap a Failed evaluation outcome"


def test_unwrap_defined_returns_only_defined_values_and_preserves_other_outcomes() -> (
    None
):
    assert unwrap_defined(Defined(None)) is None

    outcomes_and_errors = (
        (Undefined("outside domain"), UndefinedOutcomeError),
        (Indeterminate("search limit", bounds=0), IndeterminateOutcomeError),
        (Failed("validation-error"), FailedOutcomeError),
    )
    for outcome, error_type in outcomes_and_errors:
        with pytest.raises(error_type) as raised:
            unwrap_defined(outcome)
        assert isinstance(raised.value, AnyAlgebraError)
        assert raised.value.outcome is outcome

    with pytest.raises(InvalidEvaluationOutcomeError):
        UndefinedOutcomeError(Failed("validation-error"))  # type: ignore[arg-type]
    with pytest.raises(InvalidEvaluationOutcomeError):
        IndeterminateOutcomeError(Undefined("outside domain"))  # type: ignore[arg-type]
    with pytest.raises(InvalidEvaluationOutcomeError):
        FailedOutcomeError(Indeterminate("search limit"))  # type: ignore[arg-type]


def test_evaluation_outcome_alias_is_closed_over_exactly_four_branches() -> None:
    branch_types = tuple(
        origin if isinstance(origin := get_origin(branch), type) else branch
        for branch in get_args(EvaluationOutcome)
    )

    assert branch_types == (Defined, Undefined, Indeterminate, Failed)


def test_unwrap_rejects_nonoutcomes_and_unsupported_capability_is_separate() -> None:
    capability = UnsupportedCapability("gpu-kernel", context="reference backend")

    assert isinstance(capability, AnyAlgebraError)
    assert (capability.capability, capability.context) == (
        "gpu-kernel",
        "reference backend",
    )
    assert str(capability) == "unsupported capability: gpu-kernel"
    assert not isinstance(capability, Undefined)
    with pytest.raises(InvalidEvaluationOutcomeError) as unsupported:
        unwrap_defined(capability)  # type: ignore[arg-type]
    with pytest.raises(InvalidEvaluationOutcomeError) as invalid:
        unwrap_defined(object())  # type: ignore[arg-type]

    assert unsupported.value.field == invalid.value.field == "outcome"
    assert "0x" not in str(unsupported.value)


@pytest.mark.parametrize(
    ("capability", "context", "field"),
    [
        (" gpu", None, "capability"),
        ("", None, "capability"),
        (_StringSubclass("gpu"), None, "capability"),
        ("gpu", " context", "context"),
        ("gpu", "", "context"),
        ("gpu", _StringSubclass("context"), "context"),
    ],
)
def test_unsupported_capability_rejects_unstable_metadata(
    capability: object, context: object, field: str
) -> None:
    with pytest.raises(CapabilityDefinitionError) as raised:
        UnsupportedCapability(capability, context=context)  # type: ignore[arg-type]

    assert raised.value.field == field
    assert "evaluation outcome" not in str(raised.value)
