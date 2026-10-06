"""Immutable evaluation outcomes and their strict convenience boundary.

The neutral evaluation API represents partiality with one of four outcomes.
``Failed.error`` is deliberately an inert, stable string in v0.0: live
exceptions, tracebacks, and arbitrary error payloads are not accepted here and
no serialization codec is supplied by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeAlias, TypeVar

from anyalgebra.core.errors import AnyAlgebraError


T = TypeVar("T")


class InvalidEvaluationOutcomeError(AnyAlgebraError, ValueError):
    """A public evaluation-outcome boundary received invalid metadata."""

    def __init__(self, *, field: str, reason: str) -> None:
        """Record a deterministic, address-free boundary diagnostic."""
        self.field = field
        self.reason = reason
        super().__init__(f"invalid evaluation outcome {field}: {reason}")


class CapabilityDefinitionError(AnyAlgebraError, ValueError):
    """An unsupported-capability request received malformed safe metadata."""

    def __init__(self, *, field: str, reason: str) -> None:
        """Record a deterministic, address-free capability-boundary diagnostic."""
        self.field = field
        self.reason = reason
        super().__init__(f"invalid unsupported capability {field}: {reason}")


def _require_stable_string(
    value: object,
    *,
    field_name: str,
    error_type: type[InvalidEvaluationOutcomeError]
    | type[CapabilityDefinitionError] = InvalidEvaluationOutcomeError,
) -> str:
    """Require exact, nonempty, trimmed string metadata before any comparison."""
    if type(value) is not str or not value or value != value.strip():
        raise error_type(
            field=field_name,
            reason="must be a non-empty trimmed built-in str",
        )
    return value


@dataclass(frozen=True, slots=True)
class Defined(Generic[T]):
    """A requested mathematical value was computed, including a literal ``None``."""

    value: T


@dataclass(frozen=True, slots=True, init=False)
class Undefined:
    """A mathematically meaningful operation is not defined on these inputs."""

    reason: str
    witness: object | None = field(default=None)

    def __init__(self, reason: str, witness: object | None = None) -> None:
        """Validate stable explanatory metadata and preserve the literal witness."""
        object.__setattr__(
            self, "reason", _require_stable_string(reason, field_name="reason")
        )
        object.__setattr__(self, "witness", witness)


@dataclass(frozen=True, slots=True, init=False)
class Indeterminate:
    """A bounded or assumption-dependent method could not decide."""

    reason: str
    bounds: object | None = field(default=None)

    def __init__(self, reason: str, bounds: object | None = None) -> None:
        """Validate stable explanatory metadata and preserve declared bounds."""
        object.__setattr__(
            self, "reason", _require_stable_string(reason, field_name="reason")
        )
        object.__setattr__(self, "bounds", bounds)


@dataclass(frozen=True, slots=True, init=False)
class Failed:
    """A meaningful evaluation request failed without becoming undefinedness.

    ``error`` is restricted to an inert stable string until a later schema-owned
    error-record type exists.  This branch never stores or re-raises a caller's
    exception object.
    """

    error: str
    stage: str | None = field(default=None)

    def __init__(self, error: str, stage: str | None = None) -> None:
        """Validate the v0.0 inert error representation and optional stage."""
        object.__setattr__(
            self, "error", _require_stable_string(error, field_name="error")
        )
        if stage is not None:
            stage = _require_stable_string(stage, field_name="stage")
        object.__setattr__(self, "stage", stage)


EvaluationOutcome: TypeAlias = Defined[T] | Undefined | Indeterminate | Failed
"""The complete v0.0 evaluation-outcome union; no capability is a fifth branch."""


class OutcomeUnwrapError(AnyAlgebraError):
    """Base class for strict unwrapping of a non-defined exact outcome branch."""

    def __init__(self, outcome: Undefined | Indeterminate | Failed) -> None:
        """Retain the exact neutral outcome without rendering its payload."""
        self.outcome = outcome
        super().__init__(f"cannot unwrap a {type(outcome).__name__} evaluation outcome")


class UndefinedOutcomeError(OutcomeUnwrapError):
    """Strict unwrapping encountered mathematical undefinedness."""

    def __init__(self, outcome: Undefined) -> None:
        """Retain only an exact undefined outcome in this typed exception."""
        if type(outcome) is not Undefined:
            raise InvalidEvaluationOutcomeError(
                field="outcome",
                reason="must be an exact Undefined branch",
            )
        super().__init__(outcome)


class IndeterminateOutcomeError(OutcomeUnwrapError):
    """Strict unwrapping encountered a bounded inability to decide."""

    def __init__(self, outcome: Indeterminate) -> None:
        """Retain only an exact indeterminate outcome in this typed exception."""
        if type(outcome) is not Indeterminate:
            raise InvalidEvaluationOutcomeError(
                field="outcome",
                reason="must be an exact Indeterminate branch",
            )
        super().__init__(outcome)


class FailedOutcomeError(OutcomeUnwrapError):
    """Strict unwrapping encountered a failed meaningful evaluation request."""

    def __init__(self, outcome: Failed) -> None:
        """Retain only an exact failed outcome in this typed exception."""
        if type(outcome) is not Failed:
            raise InvalidEvaluationOutcomeError(
                field="outcome",
                reason="must be an exact Failed branch",
            )
        super().__init__(outcome)


class UnsupportedCapability(AnyAlgebraError):
    """A requested optional capability is unavailable outside evaluation outcomes."""

    def __init__(self, capability: str, *, context: str | None = None) -> None:
        """Record safe, deterministic request metadata without arbitrary payloads."""
        self.capability = _require_stable_string(
            capability,
            field_name="capability",
            error_type=CapabilityDefinitionError,
        )
        if context is not None:
            context = _require_stable_string(
                context,
                field_name="context",
                error_type=CapabilityDefinitionError,
            )
        self.context = context
        super().__init__(f"unsupported capability: {self.capability}")


def unwrap_defined(outcome: EvaluationOutcome[T]) -> T:
    """Return a defined value or raise a typed error for the exact branch.

    Only the four exact branch classes are accepted.  In particular, a stored
    ``Failed`` string is diagnostic data, not an exception to execute or raise.
    """
    if type(outcome) is Defined:
        return outcome.value
    if type(outcome) is Undefined:
        raise UndefinedOutcomeError(outcome)
    if type(outcome) is Indeterminate:
        raise IndeterminateOutcomeError(outcome)
    if type(outcome) is Failed:
        raise FailedOutcomeError(outcome)
    raise InvalidEvaluationOutcomeError(
        field="outcome",
        reason="must be an exact EvaluationOutcome branch",
    )
