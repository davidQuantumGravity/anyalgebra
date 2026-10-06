"""Exact, neutral differential comparison receipts (V00-081).

This module deliberately has no registry or adapter discovery.  Callers supply
the already-enabled adapters and an immutable calculation contract.  It records
only sealed backend metadata and semantic hashes: backend-native values and
exception details never enter the report.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
from inspect import (
    CO_ASYNC_GENERATOR,
    CO_COROUTINE,
    CO_GENERATOR,
    CO_VARARGS,
    CO_VARKEYWORDS,
)
from types import FunctionType
from typing import cast

from anyalgebra.backends.base import (
    BackendCapabilities,
    BackendOptions,
    BackendRequest,
    BackendResult,
    BackendSupported,
    BackendUnsupported,
    CapabilitySpec,
    UnsupportedReason,
    negotiate,
)
from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import SemanticHash
from anyalgebra.evidence.contracts import (
    CalculationContract,
    calculation_contract_record,
    contract_record_registry,
)
from anyalgebra.evidence.run import (
    CalculationResult,
    ResultReceipt,
    result_receipt_record,
    run_calculation,
)
from anyalgebra.structures.outcomes import Defined, Failed, Indeterminate, Undefined


_MAX_ADAPTERS = 32
_MAX_DIAGNOSTIC = 96
_MAX_METADATA_TEXT = 256


class DifferentialError(AnyAlgebraError, ValueError):
    """A differential input failed the closed neutral comparison boundary."""

    def __init__(self, field: str, reason: str) -> None:
        self.field, self.reason = field, reason
        super().__init__(f"invalid differential comparison {field}: {reason}")


class DifferentialStatus(StrEnum):
    EXACT_MATCH = "exact_match"
    DISAGREEMENT = "disagreement"
    UNAVAILABLE = "unavailable"
    EXECUTION_FAILURE = "execution_failure"


def _hash(value: object, field: str) -> SemanticHash:
    if type(value) is not SemanticHash:
        raise DifferentialError(field, "must be an exact SemanticHash")
    try:
        return SemanticHash(value.algorithm, value.digest)
    except (TypeError, ValueError, AttributeError) as error:
        raise DifferentialError(field, "contains an invalid SemanticHash") from error


def _digest(record: dict[str, object]) -> SemanticHash:
    try:
        payload = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise DifferentialError("record", "canonical encoding rejected") from error
    return SemanticHash("sha256", hashlib.sha256(payload).hexdigest())


def _snapshot(value: object, field: str) -> tuple[object, ...]:
    if type(value) is str:
        raise DifferentialError(field, "must be an iterable, not str")
    try:
        iterator = iter(cast(Iterable[object], value))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise DifferentialError(field, "must be an iterable") from error
    items: list[object] = []
    try:
        for _ in range(_MAX_ADAPTERS + 1):
            items.append(next(iterator))
    except StopIteration:
        return tuple(items)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise DifferentialError(field, "iterable snapshot failed") from error
    raise DifferentialError(field, "adapter limit exceeded")


def _request(value: object) -> BackendRequest:
    if type(value) is not BackendRequest:
        raise DifferentialError("request", "must be an exact BackendRequest")
    try:
        value._assert()
        return BackendRequest.create(
            operation=value.operation,
            domain=value.domain,
            algorithm=value.algorithm,
            requires_exact=value.requires_exact,
            precondition_hashes=value.precondition_hashes,
            convention_hashes=value.convention_hashes,
            input_hashes=value.input_hashes,
            limits=value.limits,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise DifferentialError(
            "request", "content address integrity rejected"
        ) from error


def _capabilities(value: object) -> BackendCapabilities:
    if type(value) is not BackendCapabilities:
        raise DifferentialError("capabilities", "must be exact neutral metadata")
    try:
        value._assert()
        specifications = tuple(
            CapabilitySpec.create(
                operation=item.operation,
                domain=item.domain,
                algorithm=item.algorithm,
                exact=item.exact,
                precondition_hashes=item.precondition_hashes,
                convention_hashes=item.convention_hashes,
            )
            for item in value.specifications
        )
        return BackendCapabilities.create(
            name=value.name,
            version=value.version,
            specifications=specifications,
            convention_hashes=value.convention_hashes,
            limits=value.limits,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise DifferentialError(
            "capabilities", "content address integrity rejected"
        ) from error


def _contract(
    value: object,
    request: BackendRequest,
    capabilities: tuple[BackendCapabilities, ...],
) -> CalculationContract:
    if type(value) is not CalculationContract:
        raise DifferentialError("contract", "must be an exact CalculationContract")
    try:
        parsed = contract_record_registry().from_record(
            calculation_contract_record(value)
        )
        if type(parsed) is not CalculationContract:
            raise DifferentialError("contract", "round trip did not return a contract")
        if tuple(item[1] for item in parsed.input_hashes) != request.input_hashes:
            raise DifferentialError("contract", "input hashes do not bind the request")
        expected_algorithms = tuple(
            sorted(
                {
                    ("differential", "v00_081"),
                    *((f"backend.{item.name}", item.version) for item in capabilities),
                }
            )
        )
        if (
            parsed.backend_request != str(request.semantic_hash)
            or parsed.backend_name != "anyalgebra.differential"
            or parsed.backend_version != "v00_081"
            or parsed.convention_manifests != request.convention_hashes
            or parsed.algorithms != expected_algorithms
        ):
            raise DifferentialError("contract", "does not bind differential metadata")
        return parsed
    except (KeyboardInterrupt, SystemExit):
        raise
    except DifferentialError:
        raise
    except Exception as error:
        raise DifferentialError(
            "contract", "content address integrity rejected"
        ) from error


def _method(
    adapter: object, name: str, positional: int, keyword_only: tuple[str, ...]
) -> Callable[..., object] | None:
    try:
        adapter_type = type(adapter)
        mro = type.__getattribute__(adapter_type, "__mro__")
        for cls in mro:
            candidate = type.__getattribute__(cls, "__dict__").get(name)
            if candidate is None:
                continue
            if type(candidate) is not FunctionType:
                return None
            code = candidate.__code__
            if (
                code.co_flags
                & (
                    CO_VARARGS
                    | CO_VARKEYWORDS
                    | CO_GENERATOR
                    | CO_COROUTINE
                    | CO_ASYNC_GENERATOR
                )
                or code.co_posonlyargcount != 0
                or code.co_argcount != positional
                or code.co_kwonlyargcount != len(keyword_only)
                or code.co_varnames[
                    code.co_argcount : code.co_argcount + code.co_kwonlyargcount
                ]
                != keyword_only
                or candidate.__defaults__ is not None
                or candidate.__kwdefaults__ is not None
            ):
                return None
            return cast(Callable[..., object], candidate)
        return None
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise DifferentialError("adapter", f"{name} method is unavailable") from error


def _call_capabilities(adapter: object) -> BackendCapabilities:
    try:
        method = _method(adapter, "capabilities", 1, ())
        if method is None:
            raise DifferentialError("adapter", "capabilities method is unavailable")
        raw = method(adapter)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise DifferentialError(
            "capabilities", "adapter capability call failed"
        ) from error
    return _capabilities(raw)


def _copy_options(request: BackendRequest) -> BackendOptions:
    return BackendOptions.create(
        convention_hashes=request.convention_hashes, limits=request.limits
    )


def _plain_diagnostic(value: object) -> str:
    if (
        type(value) is str
        and value
        and value == value.strip()
        and len(value) <= _MAX_DIAGNOSTIC
        and not any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        return value
    raise DifferentialError("outcome", "diagnostic is not bounded plain text")


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class DifferentialObservation:
    """One backend-to-outcome pairing without an adapter-native value."""

    name: str
    version: str
    operation: str
    domain: str
    algorithm: str
    request_hash: SemanticHash
    specification_hash: SemanticHash
    capabilities_hash: SemanticHash
    outcome: str
    outcome_hash: SemanticHash
    comparison_hash: SemanticHash
    value_hash: SemanticHash | None
    provenance_hash: SemanticHash | None
    options_hash: SemanticHash | None
    payload_hash: SemanticHash | None
    diagnostic: str | None
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise DifferentialError("observation", "is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("DifferentialObservation cannot be subclassed")

    @classmethod
    def _create(
        cls,
        *,
        capabilities: BackendCapabilities,
        request: BackendRequest,
        specification: CapabilitySpec,
        outcome: str,
        outcome_hash: SemanticHash,
        comparison_hash: SemanticHash,
        value_hash: SemanticHash | None = None,
        provenance_hash: SemanticHash | None = None,
        options_hash: SemanticHash | None = None,
        payload_hash: SemanticHash | None = None,
        diagnostic: str | None = None,
    ) -> DifferentialObservation:
        value = object.__new__(DifferentialObservation)
        for name, item in (
            ("name", capabilities.name),
            ("version", capabilities.version),
            ("operation", specification.operation),
            ("domain", specification.domain),
            ("algorithm", specification.algorithm),
            ("request_hash", _hash(request.semantic_hash, "request_hash")),
            (
                "specification_hash",
                _hash(specification.semantic_hash, "specification_hash"),
            ),
            (
                "capabilities_hash",
                _hash(capabilities.semantic_hash, "capabilities_hash"),
            ),
            ("outcome", outcome),
            ("outcome_hash", _hash(outcome_hash, "outcome_hash")),
            ("comparison_hash", _hash(comparison_hash, "comparison_hash")),
            (
                "value_hash",
                None if value_hash is None else _hash(value_hash, "value_hash"),
            ),
            (
                "provenance_hash",
                None
                if provenance_hash is None
                else _hash(provenance_hash, "provenance_hash"),
            ),
            (
                "options_hash",
                None if options_hash is None else _hash(options_hash, "options_hash"),
            ),
            (
                "payload_hash",
                None if payload_hash is None else _hash(payload_hash, "payload_hash"),
            ),
            ("diagnostic", diagnostic),
        ):
            object.__setattr__(value, name, item)
        object.__setattr__(value, "semantic_hash", _digest(value._record()))
        return value

    def _record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "operation": self.operation,
            "domain": self.domain,
            "algorithm": self.algorithm,
            "request": str(self.request_hash),
            "specification": str(self.specification_hash),
            "capabilities": str(self.capabilities_hash),
            "outcome": self.outcome,
            "outcomeHash": str(self.outcome_hash),
            "comparisonHash": str(self.comparison_hash),
            "value": None if self.value_hash is None else str(self.value_hash),
            "provenance": None
            if self.provenance_hash is None
            else str(self.provenance_hash),
            "options": None if self.options_hash is None else str(self.options_hash),
            "payload": None if self.payload_hash is None else str(self.payload_hash),
            "diagnostic": self.diagnostic,
        }

    def _assert(self) -> None:
        try:
            for item in (
                self.request_hash,
                self.specification_hash,
                self.capabilities_hash,
                self.outcome_hash,
                self.comparison_hash,
            ):
                _hash(item, "observation_hash")
            if self.value_hash is not None:
                _hash(self.value_hash, "value_hash")
            if self.provenance_hash is not None:
                _hash(self.provenance_hash, "provenance_hash")
            if self.options_hash is not None:
                _hash(self.options_hash, "options_hash")
            if self.payload_hash is not None:
                _hash(self.payload_hash, "payload_hash")
            if self.diagnostic is not None:
                _plain_diagnostic(self.diagnostic)
            operational = {"not_executed", "execution_failed", "invalid_outcome"}
            metadata = (
                self.name,
                self.version,
                self.operation,
                self.domain,
                self.algorithm,
            )
            if any(
                type(item) is not str
                or not item
                or item != item.strip()
                or len(item) > _MAX_METADATA_TEXT
                or any(ord(char) < 32 or ord(char) == 127 for char in item)
                for item in metadata
            ):
                raise DifferentialError("observation", "metadata text rejected")
            expected_hash: SemanticHash
            if self.outcome == "defined":
                expected_hash = _digest(
                    {
                        "request": str(self.request_hash),
                        "provenance": str(self.provenance_hash),
                        "options": str(self.options_hash),
                        "value": str(self.value_hash),
                    }
                )
                expected_comparison = _digest(
                    {"outcome": "defined", "value": str(self.value_hash)}
                )
            elif self.outcome in {"undefined", "indeterminate"}:
                assert self.diagnostic is not None
                expected_hash = _digest(
                    {
                        "outcome": self.outcome,
                        "label": self.diagnostic,
                        "payload": None
                        if self.payload_hash is None
                        else str(self.payload_hash),
                    }
                )
                expected_comparison = expected_hash
            elif self.outcome == "failed":
                expected_hash = _digest(
                    {
                        "outcome": "failed",
                        "payload": str(self.payload_hash),
                    }
                )
                expected_comparison = expected_hash
            elif self.outcome == "unsupported":
                assert self.diagnostic is not None
                expected_hash = _digest(
                    {"outcome": "unsupported", "reason": self.diagnostic}
                )
                expected_comparison = expected_hash
            else:
                expected_hash = _digest({"outcome": self.outcome})
                expected_comparison = expected_hash
            if (
                type(self.outcome) is not str
                or not self.outcome
                or self.outcome != self.outcome.strip()
                or len(self.outcome) > _MAX_DIAGNOSTIC
                or any(ord(char) < 32 or ord(char) == 127 for char in self.outcome)
                or self.outcome
                not in {
                    "defined",
                    "undefined",
                    "indeterminate",
                    "failed",
                    "unsupported",
                    "not_executed",
                    "execution_failed",
                    "invalid_outcome",
                }
                or (self.outcome == "defined") != (self.value_hash is not None)
                or (self.outcome == "defined" and self.diagnostic is not None)
                or (self.outcome == "defined" and self.payload_hash is not None)
                or (
                    self.outcome == "defined"
                    and (self.provenance_hash is None or self.options_hash is None)
                )
                or (
                    self.outcome in {"undefined", "indeterminate"}
                    and self.diagnostic is None
                )
                or (
                    self.outcome not in {"undefined", "indeterminate", "failed"}
                    and self.payload_hash is not None
                )
                or (self.outcome == "failed" and self.payload_hash is None)
                or (
                    self.outcome != "defined"
                    and (
                        self.provenance_hash is not None
                        or self.options_hash is not None
                    )
                )
                or (self.outcome == "failed" and self.diagnostic != "backend_failed")
                or (
                    self.outcome == "unsupported"
                    and self.diagnostic
                    not in {item.value for item in UnsupportedReason}
                )
                or (self.outcome in operational and self.diagnostic != self.outcome)
                or self.outcome_hash != expected_hash
                or self.comparison_hash != expected_comparison
                or _digest(self._record()) != _hash(self.semantic_hash, "semantic_hash")
            ):
                raise DifferentialError(
                    "observation", "content address integrity rejected"
                )
        except (KeyboardInterrupt, SystemExit):
            raise
        except DifferentialError:
            raise
        except Exception as error:
            raise DifferentialError(
                "observation", "content address integrity rejected"
            ) from error

    def __repr__(self) -> str:
        return f"DifferentialObservation(semantic_hash={str(self.semantic_hash)!r})"


def _specification(
    capabilities: BackendCapabilities, request: BackendRequest
) -> CapabilitySpec:
    return next(
        item
        for item in capabilities.specifications
        if (item.operation, item.domain, item.algorithm)
        == (request.operation, request.domain, request.algorithm)
    )


def _observation(
    capabilities: BackendCapabilities,
    request: BackendRequest,
    specification: CapabilitySpec,
    outcome: str,
    payload: object = None,
) -> DifferentialObservation:
    if outcome == "unsupported":
        assert type(payload) is UnsupportedReason
        digest = _digest({"outcome": outcome, "reason": payload.value})
        return DifferentialObservation._create(
            capabilities=capabilities,
            request=request,
            specification=specification,
            outcome=outcome,
            outcome_hash=digest,
            comparison_hash=digest,
            diagnostic=payload.value,
        )
    if outcome in {"not_executed", "execution_failed", "invalid_outcome"}:
        digest = _digest({"outcome": outcome})
        return DifferentialObservation._create(
            capabilities=capabilities,
            request=request,
            specification=specification,
            outcome=outcome,
            outcome_hash=digest,
            comparison_hash=digest,
            diagnostic=outcome,
        )
    if type(payload) is BackendResult:
        try:
            payload._assert()
            return DifferentialObservation._create(
                capabilities=capabilities,
                request=request,
                specification=specification,
                outcome="defined",
                outcome_hash=payload.semantic_hash,
                comparison_hash=_digest(
                    {"outcome": "defined", "value": str(payload.value_hash)}
                ),
                value_hash=payload.value_hash,
                provenance_hash=payload.provenance.semantic_hash,
                options_hash=payload.options_hash,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            return _observation(capabilities, request, specification, "invalid_outcome")
    auxiliary_hash: SemanticHash | None = None
    if type(payload) is Undefined:
        auxiliary = payload.witness
        label = payload.reason
        kind = "undefined"
    elif type(payload) is Indeterminate:
        auxiliary = payload.bounds
        label = payload.reason
        kind = "indeterminate"
    elif type(payload) is Failed:
        try:
            checked = Failed(payload.error, payload.stage)
            error = _plain_diagnostic(checked.error)
            stage = None if checked.stage is None else _plain_diagnostic(checked.stage)
        except Exception:
            return _observation(capabilities, request, specification, "invalid_outcome")
        auxiliary_hash = _digest({"error": error, "stage": stage})
        kind = "failed"
    else:
        return _observation(capabilities, request, specification, "invalid_outcome")
    if kind != "failed" and auxiliary is not None:
        try:
            auxiliary_hash = _hash(auxiliary, "outcome_payload")
        except DifferentialError:
            return _observation(capabilities, request, specification, "invalid_outcome")
    elif kind != "failed":
        auxiliary_hash = None
    if kind == "failed":
        plain_label = "backend_failed"
    else:
        try:
            plain_label = _plain_diagnostic(label)
        except DifferentialError:
            return _observation(capabilities, request, specification, "invalid_outcome")
    neutral_hash = (
        _digest({"outcome": "failed", "payload": str(auxiliary_hash)})
        if kind == "failed"
        else _digest(
            {
                "outcome": kind,
                "label": plain_label,
                "payload": None if auxiliary_hash is None else str(auxiliary_hash),
            }
        )
    )
    return DifferentialObservation._create(
        capabilities=capabilities,
        request=request,
        specification=specification,
        outcome=kind,
        outcome_hash=neutral_hash,
        comparison_hash=neutral_hash,
        payload_hash=auxiliary_hash,
        diagnostic="backend_failed" if kind == "failed" else plain_label,
    )


def _execute(
    adapter: object,
    capabilities: BackendCapabilities,
    request: BackendRequest,
    specification: CapabilitySpec,
    support: BackendSupported,
) -> DifferentialObservation:
    local_request = _request(request)
    options = _copy_options(local_request)
    try:
        method = _method(adapter, "execute", 2, ("options",))
        if method is None:
            return _observation(
                capabilities, request, specification, "execution_failed"
            )
        raw = method(adapter, local_request, options=options)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return _observation(capabilities, request, specification, "execution_failed")
    observed = _observation(
        capabilities,
        request,
        specification,
        "result",
        raw.value if type(raw) is Defined else raw,
    )
    if observed.outcome != "defined":
        return observed
    assert type(raw) is Defined and type(raw.value) is BackendResult
    result = raw.value
    try:
        if (
            result.request_hash != request.semantic_hash
            or result.provenance != support.provenance
            or result.options_hash != options.semantic_hash
        ):
            return _observation(capabilities, request, specification, "invalid_outcome")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return _observation(capabilities, request, specification, "invalid_outcome")
    return observed


class _ReceiptFailure(Exception):
    pass


def _derive_status(
    observations: tuple[DifferentialObservation, ...],
) -> DifferentialStatus:
    """Classify a complete paired neutral observation set without promotion."""
    kinds = {item.outcome for item in observations}
    if kinds & {"execution_failed", "invalid_outcome", "failed"}:
        return DifferentialStatus.EXECUTION_FAILURE
    if "unsupported" in kinds or "not_executed" in kinds:
        if kinds <= {"unsupported", "not_executed"} and "unsupported" in kinds:
            return DifferentialStatus.UNAVAILABLE
        raise DifferentialError("observations", "has an invalid availability pattern")
    if not kinds <= {"defined", "undefined", "indeterminate"}:
        raise DifferentialError("observations", "has an unsupported outcome kind")
    return (
        DifferentialStatus.EXACT_MATCH
        if len({item.comparison_hash for item in observations}) == 1
        else DifferentialStatus.DISAGREEMENT
    )


def _validate_receipt_mapping(
    *,
    receipt: object,
    request_hash: SemanticHash,
    contract_hash: SemanticHash,
    observations: tuple[DifferentialObservation, ...],
    status: DifferentialStatus,
) -> None:
    """Bind a runner-owned E1 receipt to one sealed differential conclusion."""
    if type(receipt) is not ResultReceipt:
        raise DifferentialError("receipt", "must be an exact ResultReceipt")
    try:
        receipt.canonical_bytes()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise DifferentialError(
            "receipt", "content address integrity rejected"
        ) from error
    expected_algorithms = tuple(
        sorted(
            {("differential", "v00_081")}
            | {(f"backend.{item.name}", item.version) for item in observations}
        )
    )
    if (
        receipt.contract_hash != contract_hash
        or receipt.contract.backend_request != str(request_hash)
        or receipt.contract.backend_name != "anyalgebra.differential"
        or receipt.contract.backend_version != "v00_081"
        or receipt.contract.algorithms != expected_algorithms
        or receipt.evidence_tier != "E1-executed"
    ):
        raise DifferentialError("receipt", "does not bind differential metadata")
    if status is DifferentialStatus.EXECUTION_FAILURE:
        if (
            receipt.execution_status != "failed"
            or receipt.mathematical_outcome != "implementation_error"
        ):
            raise DifferentialError("receipt", "failure mapping rejected")
        return
    expected_outcome = (
        "counterexample"
        if status is DifferentialStatus.DISAGREEMENT
        else "not_applicable"
        if status is DifferentialStatus.UNAVAILABLE
        or observations[0].outcome == "undefined"
        else "inconclusive"
        if observations[0].outcome == "indeterminate"
        else "constructed"
    )
    if (
        receipt.execution_status != "completed"
        or receipt.mathematical_outcome != expected_outcome
        or set(receipt.artifacts) != {item.semantic_hash for item in observations}
    ):
        raise DifferentialError("receipt", "completed mapping rejected")


def _receipt(
    contract: CalculationContract,
    status: DifferentialStatus,
    observations: tuple[DifferentialObservation, ...],
) -> ResultReceipt:
    observation_hashes = tuple(item.semantic_hash for item in observations)

    def packet(_: CalculationContract) -> CalculationResult:
        if (
            status is DifferentialStatus.EXACT_MATCH
            and observations[0].outcome == "defined"
        ):
            return CalculationResult.create(
                outcome="constructed",
                summary="exact backend comparison completed",
                artifacts=observation_hashes,
            )
        if (
            status is DifferentialStatus.EXACT_MATCH
            and observations[0].outcome == "undefined"
        ):
            return CalculationResult.create(
                outcome="not_applicable",
                summary="exact backend comparison found undefined outcomes",
                artifacts=observation_hashes,
            )
        if (
            status is DifferentialStatus.EXACT_MATCH
            and observations[0].outcome == "indeterminate"
        ):
            return CalculationResult.create(
                outcome="inconclusive",
                summary="exact backend comparison found bounded indeterminate outcomes",
                artifacts=observation_hashes,
                case_counts=(("casesRun", len(observations)),),
                remaining_branches=("backend outcome unresolved",),
            )
        if status is DifferentialStatus.DISAGREEMENT:
            return CalculationResult.create(
                outcome="counterexample",
                summary="neutral backend outcomes disagree",
                witnesses=tuple(
                    sorted(
                        {
                            item.value_hash
                            if item.value_hash is not None
                            else item.semantic_hash
                            for item in observations
                        },
                        key=str,
                    )
                ),
                artifacts=observation_hashes,
            )
        if status is DifferentialStatus.UNAVAILABLE:
            return CalculationResult.create(
                outcome="not_applicable",
                summary="declared exact backend capability is unavailable",
                artifacts=observation_hashes,
            )
        raise _ReceiptFailure()

    return run_calculation(contract, packet)


def _receipt_projection(receipt: object) -> SemanticHash:
    """Return the deterministic receipt content bound into a report hash.

    The registered receipt record is the complete sealed receipt surface.  The
    runner's environment and UTC interval are operational provenance only, so
    they remain protected by ``receipt_hash`` but are excluded from neutral
    differential identity.  Every other canonical receipt field remains bound.
    """
    try:
        record = result_receipt_record(receipt)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise DifferentialError(
            "receipt", "content address integrity rejected"
        ) from error
    if type(record) is not dict:
        raise DifferentialError("receipt", "content address integrity rejected")
    return _digest(
        {
            name: value
            for name, value in record.items()
            if name not in {"environment", "startedAt", "finishedAt"}
        }
    )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class DifferentialReport:
    """Immutable report tied to one request, one contract, and one receipt."""

    request_hash: SemanticHash
    contract_hash: SemanticHash
    common_exact_domain: tuple[str, str, str]
    observations: tuple[DifferentialObservation, ...]
    status: DifferentialStatus
    receipt: ResultReceipt
    receipt_hash: SemanticHash
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise DifferentialError("report", "is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("DifferentialReport cannot be subclassed")

    @classmethod
    def _create(
        cls,
        *,
        request: BackendRequest,
        contract: CalculationContract,
        common_exact_domain: tuple[str, str, str],
        observations: tuple[DifferentialObservation, ...],
        status: DifferentialStatus,
        receipt: ResultReceipt,
    ) -> DifferentialReport:
        if type(receipt) is not ResultReceipt:
            raise DifferentialError("receipt", "must be an exact ResultReceipt")
        try:
            receipt.canonical_bytes()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            raise DifferentialError(
                "receipt", "content address integrity rejected"
            ) from error
        if receipt.contract_hash != contract.semantic_hash:
            raise DifferentialError("receipt", "does not bind the calculation contract")
        ordered = tuple(sorted(observations, key=lambda item: item.name))
        if len({item.name for item in ordered}) != len(ordered):
            raise DifferentialError("observations", "contains duplicate backend names")
        for observation in ordered:
            observation._assert()
            if (
                observation.request_hash != request.semantic_hash
                or (observation.operation, observation.domain, observation.algorithm)
                != common_exact_domain
            ):
                raise DifferentialError("observations", "does not bind the request")
        if len(ordered) < 2 or _derive_status(ordered) is not status:
            raise DifferentialError("report", "status does not match observations")
        if receipt.execution_status == "completed" and set(receipt.artifacts) != {
            item.semantic_hash for item in ordered
        }:
            raise DifferentialError(
                "receipt", "artifacts do not bind every observation"
            )
        _validate_receipt_mapping(
            receipt=receipt,
            request_hash=request.semantic_hash,
            contract_hash=contract.semantic_hash,
            observations=ordered,
            status=status,
        )
        value = object.__new__(DifferentialReport)
        for name, assigned in (
            ("request_hash", _hash(request.semantic_hash, "request_hash")),
            ("contract_hash", _hash(contract.semantic_hash, "contract_hash")),
            ("common_exact_domain", common_exact_domain),
            ("observations", ordered),
            ("status", status),
            ("receipt", receipt),
            ("receipt_hash", _hash(receipt.semantic_hash, "receipt_hash")),
        ):
            object.__setattr__(value, name, assigned)
        object.__setattr__(value, "semantic_hash", _digest(value._record()))
        return value

    def _record(self) -> dict[str, object]:
        return {
            "request": str(self.request_hash),
            "contract": str(self.contract_hash),
            "domain": self.common_exact_domain,
            "observations": tuple(
                str(item.semantic_hash) for item in self.observations
            ),
            "status": self.status.value,
            "receipt": str(_receipt_projection(self.receipt)),
        }

    def _assert(self) -> None:
        try:
            if (
                type(self.status) is not DifferentialStatus
                or type(self.receipt) is not ResultReceipt
            ):
                raise DifferentialError("report", "content address integrity rejected")
            for observation in self.observations:
                observation._assert()
                if (
                    observation.request_hash != self.request_hash
                    or (
                        observation.operation,
                        observation.domain,
                        observation.algorithm,
                    )
                    != self.common_exact_domain
                ):
                    raise DifferentialError(
                        "report", "observation does not bind report domain"
                    )
            if (
                type(self.observations) is not tuple
                or len(self.observations) < 2
                or len(self.observations) > _MAX_ADAPTERS
                or tuple(sorted(self.observations, key=lambda item: item.name))
                != self.observations
                or len({item.name for item in self.observations})
                != len(self.observations)
                or _derive_status(self.observations) is not self.status
            ):
                raise DifferentialError("report", "observation structure rejected")
            _validate_receipt_mapping(
                receipt=self.receipt,
                request_hash=self.request_hash,
                contract_hash=self.contract_hash,
                observations=self.observations,
                status=self.status,
            )
            if (
                self.receipt.semantic_hash != self.receipt_hash
                or self.receipt.contract_hash != self.contract_hash
                or (
                    self.receipt.execution_status == "completed"
                    and set(self.receipt.artifacts)
                    != {item.semantic_hash for item in self.observations}
                )
                or _digest(self._record()) != _hash(self.semantic_hash, "semantic_hash")
            ):
                raise DifferentialError("report", "content address integrity rejected")
        except (KeyboardInterrupt, SystemExit):
            raise
        except DifferentialError:
            raise DifferentialError(
                "report", "content address integrity rejected"
            ) from None
        except Exception as error:
            raise DifferentialError(
                "report", "content address integrity rejected"
            ) from error

    def __repr__(self) -> str:
        return (
            f"DifferentialReport(semantic_hash={str(self.semantic_hash)!r}, "
            f"status={self.status.value!r})"
        )


def compare_enabled_backends(
    request: object, adapters: object, *, contract: object
) -> DifferentialReport:
    """Compare supplied adapters once on their exact shared declared domain.

    The function deliberately does not select, load, or retry adapters.  Every
    eligible adapter receives one fresh request/options snapshot and is executed
    at most once.  Adapter order is normalized by declared backend name.
    """
    checked_request = _request(request)
    if not checked_request.requires_exact:
        raise DifferentialError("request", "differential comparison requires exactness")
    source_adapters = _snapshot(adapters, "adapters")
    if len(source_adapters) < 2:
        raise DifferentialError("adapters", "requires at least two enabled adapters")
    declared = tuple(
        (_call_capabilities(adapter), adapter) for adapter in source_adapters
    )
    if len({capabilities.name for capabilities, _ in declared}) != len(declared):
        raise DifferentialError("adapters", "contains duplicate declared backend names")
    declared = tuple(sorted(declared, key=lambda item: item[0].name))
    checked_contract = _contract(
        contract, checked_request, tuple(capabilities for capabilities, _ in declared)
    )
    common = set(declared[0][0].exactness)
    for capabilities, _ in declared[1:]:
        common.intersection_update(capabilities.exactness)
    triple = (
        checked_request.operation,
        checked_request.domain,
        checked_request.algorithm,
    )
    if triple not in common:
        raise DifferentialError(
            "request", "is outside the declared common exact domain"
        )
    eligible: list[
        tuple[BackendCapabilities, object, CapabilitySpec, BackendSupported]
    ] = []
    observations: list[DifferentialObservation] = []
    for capabilities, adapter in declared:
        specification = _specification(capabilities, checked_request)
        expected = negotiate(capabilities, checked_request)
        try:
            method = _method(adapter, "supports", 2, ())
            if method is None:
                raise DifferentialError("adapter", "supports method is unavailable")
            actual = method(adapter, _request(checked_request))
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            observations.append(
                _observation(
                    capabilities, checked_request, specification, "execution_failed"
                )
            )
            continue
        if type(actual) is not type(expected):
            observations.append(
                _observation(
                    capabilities, checked_request, specification, "execution_failed"
                )
            )
            continue
        actual_result = cast(BackendSupported, actual)
        try:
            actual_result._assert()
            if actual_result.semantic_hash != expected.semantic_hash:
                raise DifferentialError(
                    "support", "adapter support result does not match declared metadata"
                )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            observations.append(
                _observation(
                    capabilities, checked_request, specification, "execution_failed"
                )
            )
            continue
        if type(expected) is BackendUnsupported:
            observations.append(
                _observation(
                    capabilities,
                    checked_request,
                    specification,
                    "unsupported",
                    expected.reason,
                )
            )
        else:
            assert type(actual) is BackendSupported
            assert type(expected) is BackendSupported
            eligible.append((capabilities, adapter, specification, expected))
    if len(eligible) != len(declared):
        observations.extend(
            _observation(capabilities, checked_request, specification, "not_executed")
            for capabilities, _, specification, _ in eligible
        )
    else:
        observations = [
            _execute(adapter, capabilities, checked_request, specification, support)
            for capabilities, adapter, specification, support in eligible
        ]
    frozen_observations = tuple(sorted(observations, key=lambda item: item.name))
    status = _derive_status(frozen_observations)
    receipt = _receipt(checked_contract, status, frozen_observations)
    return DifferentialReport._create(
        request=checked_request,
        contract=checked_contract,
        common_exact_domain=triple,
        observations=frozen_observations,
        status=status,
        receipt=receipt,
    )
