"""Trusted synchronous execution and content-addressed evidence receipts.

Only :func:`run_calculation` is public execution authority and it always emits
E1.  The sealed private builder already understands promotion evidence so the
next task can add promotion without changing this persistence boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, cast

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import SemanticHash
from anyalgebra.evidence.contracts import (
    CalculationContract,
    ContractError,
    calculation_contract_canonical_bytes,
    calculation_contract_record,
    contract_record_registry,
)
from anyalgebra.persistence.canonical import canonical_json
from anyalgebra.persistence.registry import JSONValue, SchemaError, SerializerRegistry


_TAG, _VERSION, _MAX_ITEMS, _MAX_TEXT = (
    "anyalgebra.evidence.result_receipt",
    1,
    256,
    4_096,
)
_TIERS = (
    "E1-executed",
    "E2-regression",
    "E3-bounded-exact",
    "E4-cross-checked",
    "E5-proof-linked",
    "E6-reproduced",
)
_OUTCOMES = frozenset(
    (
        "constructed",
        "verified_within_domain",
        "counterexample",
        "no_go_within_assumptions",
        "inconclusive",
        "not_applicable",
        "reproduction_only",
        "implementation_error",
    )
)


class ExecutionStatus(StrEnum):
    """The normative lifecycle vocabulary for calculation execution."""

    NOT_RUN = "not_run"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


TerminalExecutionStatus = Literal["completed", "failed"]
"""Statuses permitted on immutable, terminal :class:`ResultReceipt` values."""


_TERMINAL_EXECUTION_STATUSES = frozenset(
    (ExecutionStatus.COMPLETED.value, ExecutionStatus.FAILED.value)
)
_DISALLOWED_E1 = re.compile(
    r"\b(proved|confirmed|complete|ruled(?:[^\w]|_)+out)\b", re.I
)


class ReceiptError(AnyAlgebraError, ValueError):
    """Closed receipt-boundary diagnostic; it never renders hostile input."""

    def __init__(self, field: str, reason: str) -> None:
        self.field, self.reason = field, reason
        super().__init__(f"invalid result receipt {field}: {reason}")


def _text(value: object, field: str, *, identifier: bool = False) -> str:
    if type(value) is not str or not value:
        raise ReceiptError(field, "must be a nonempty exact built-in str")
    if len(value) > _MAX_TEXT or any("\ud800" <= char <= "\udfff" for char in value):
        raise ReceiptError(field, "text limit or Unicode scalar violation")
    if identifier and value != value.strip():
        raise ReceiptError(field, "must not have leading or trailing whitespace")
    return value


def _items(value: object, field: str) -> tuple[object, ...]:
    if type(value) is str:
        raise ReceiptError(field, "must be an iterable")
    try:
        iterator = iter(cast(Iterable[object], value))
    except Exception as error:
        raise ReceiptError(field, "must be an iterable") from error
    result: list[object] = []
    try:
        for _ in range(_MAX_ITEMS + 1):
            result.append(next(iterator))
    except StopIteration:
        return tuple(result)
    except Exception as error:
        raise ReceiptError(field, "iterable snapshot failed") from error
    raise ReceiptError(field, "item limit exceeded")


def _hash(value: object, field: str) -> SemanticHash:
    if type(value) is not SemanticHash:
        raise ReceiptError(field, "must be an exact SemanticHash")
    return SemanticHash(value.algorithm, value.digest)


def _hashes(value: object, field: str) -> tuple[SemanticHash, ...]:
    result = tuple(_hash(item, field) for item in _items(value, field))
    if len(set(result)) != len(result):
        raise ReceiptError(field, "contains duplicate content addresses")
    return tuple(sorted(result, key=str))


def _strings(value: object, field: str) -> tuple[str, ...]:
    result = tuple(_text(item, field) for item in _items(value, field))
    if len(set(result)) != len(result):
        raise ReceiptError(field, "contains duplicate declarations")
    return tuple(sorted(result))


def _counts(value: object, field: str) -> tuple[tuple[str, int], ...]:
    pairs: list[tuple[str, int]] = []
    for item in _items(value, field):
        if type(item) not in (tuple, list):
            raise ReceiptError(field, "each count must be an exact pair")
        pair = cast(tuple[object, object] | list[object], item)
        if len(pair) != 2:
            raise ReceiptError(field, "each count must be an exact pair")
        key = _text(pair[0], field, identifier=True)
        if type(pair[1]) is not int or pair[1] < 0:
            raise ReceiptError(field, "counts must be nonnegative exact ints")
        pairs.append((key, pair[1]))
    if len({key for key, _ in pairs}) != len(pairs):
        raise ReceiptError(field, "contains duplicate count names")
    return tuple(sorted(pairs))


def _pairs(value: object, field: str) -> tuple[tuple[str, str | int], ...]:
    pairs: list[tuple[str, str | int]] = []
    for item in _items(value, field):
        if type(item) not in (tuple, list):
            raise ReceiptError(field, "each declaration must be an exact pair")
        pair = cast(tuple[object, object] | list[object], item)
        if len(pair) != 2:
            raise ReceiptError(field, "each declaration must be an exact pair")
        key = _text(pair[0], field, identifier=True)
        if type(pair[1]) is int and pair[1] >= 0:
            item_value: str | int = pair[1]
        elif type(pair[1]) is str:
            item_value = _text(pair[1], field)
        else:
            raise ReceiptError(field, "has an invalid declaration value")
        pairs.append((key, item_value))
    if len({key for key, _ in pairs}) != len(pairs):
        raise ReceiptError(field, "contains duplicate declaration names")
    return tuple(sorted(pairs))


def _hash_record(value: SemanticHash) -> dict[str, str]:
    return {"algorithm": value.algorithm, "digest": value.digest}


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _timestamp(value: object, field: str) -> str:
    text = _text(value, field, identifier=True)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", text):
        raise ReceiptError(field, "must be an exact UTC timestamp")
    try:
        datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ReceiptError(field, "must be an exact UTC timestamp") from error
    return text


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ExecutionEnvironment:
    platform: str
    python_version: str
    software_revision: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ReceiptError("environment", "is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("ExecutionEnvironment cannot be subclassed")

    @classmethod
    def create(
        cls,
        *,
        platform: object = "unavailable",
        python_version: object = "unavailable",
        software_revision: object = "unavailable",
    ) -> ExecutionEnvironment:
        if cls is not ExecutionEnvironment:
            raise ReceiptError(
                "environment", "factory requires exact ExecutionEnvironment"
            )
        value = object.__new__(ExecutionEnvironment)
        for name, item in (
            ("platform", platform),
            ("python_version", python_version),
            ("software_revision", software_revision),
        ):
            object.__setattr__(value, name, _text(item, name, identifier=True))
        return value

    def __repr__(self) -> str:
        return "ExecutionEnvironment(<metadata>)"

    def __eq__(self, other: object) -> bool:
        return type(other) is ExecutionEnvironment and (
            self.platform,
            self.python_version,
            self.software_revision,
        ) == (other.platform, other.python_version, other.software_revision)

    def __hash__(self) -> int:
        return hash((self.platform, self.python_version, self.software_revision))


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class IndependentCheck:
    """A named implementation/result pair used to establish E4 independence."""

    method_id: str
    implementation_hash: SemanticHash
    result_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ReceiptError("independent_check", "is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("IndependentCheck cannot be subclassed")

    @classmethod
    def create(
        cls, *, method_id: object, implementation_hash: object, result_hash: object
    ) -> IndependentCheck:
        if cls is not IndependentCheck:
            raise ReceiptError(
                "independent_check", "factory requires exact IndependentCheck"
            )
        value = object.__new__(IndependentCheck)
        object.__setattr__(
            value, "method_id", _text(method_id, "method_id", identifier=True)
        )
        object.__setattr__(
            value,
            "implementation_hash",
            _hash(implementation_hash, "implementation_hash"),
        )
        object.__setattr__(value, "result_hash", _hash(result_hash, "result_hash"))
        return value

    def __repr__(self) -> str:
        return "IndependentCheck(<hashes>)"

    def __eq__(self, other: object) -> bool:
        return type(other) is IndependentCheck and (
            self.method_id,
            self.implementation_hash,
            self.result_hash,
        ) == (other.method_id, other.implementation_hash, other.result_hash)

    def __hash__(self) -> int:
        return hash((self.method_id, self.implementation_hash, self.result_hash))


def _checks(value: object) -> tuple[IndependentCheck, ...]:
    checks = tuple(item for item in _items(value, "independent_checks"))
    if any(type(item) is not IndependentCheck for item in checks):
        raise ReceiptError(
            "independent_checks", "must contain exact IndependentCheck values"
        )
    copied = tuple(
        IndependentCheck.create(
            method_id=item.method_id,
            implementation_hash=item.implementation_hash,
            result_hash=item.result_hash,
        )
        for item in cast(tuple[IndependentCheck, ...], checks)
    )
    if len({item.method_id for item in copied}) != len(copied):
        raise ReceiptError("independent_checks", "contains duplicate method IDs")
    return tuple(sorted(copied, key=lambda item: item.method_id))


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class CalculationResult:
    outcome: str
    summary: str
    case_counts: tuple[tuple[str, int], ...]
    warnings: tuple[str, ...]
    witnesses: tuple[SemanticHash, ...]
    artifacts: tuple[SemanticHash, ...]
    logs: tuple[SemanticHash, ...]
    checks_performed: tuple[str, ...]
    dependency_receipts: tuple[SemanticHash, ...]
    remaining_branches: tuple[str, ...]
    exact_certificate: SemanticHash | None

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ReceiptError("calculation_result", "is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("CalculationResult cannot be subclassed")

    @classmethod
    def create(
        cls,
        *,
        outcome: object,
        summary: object,
        case_counts: object = (),
        warnings: object = (),
        witnesses: object = (),
        artifacts: object = (),
        logs: object = (),
        checks_performed: object = (),
        dependency_receipts: object = (),
        remaining_branches: object = (),
        exact_certificate: object | None = None,
    ) -> CalculationResult:
        if cls is not CalculationResult:
            raise ReceiptError(
                "calculation_result", "factory requires exact CalculationResult"
            )
        value = object.__new__(CalculationResult)
        checked_summary = _text(summary, "summary")
        if _DISALLOWED_E1.search(checked_summary):
            raise ReceiptError("summary", "uses unsupported E1 claim language")
        checked_outcome = _text(outcome, "outcome", identifier=True)
        if checked_outcome not in _OUTCOMES - {"implementation_error"}:
            raise ReceiptError("outcome", "is not callable-owned")
        for name, item in (
            ("outcome", checked_outcome),
            ("summary", checked_summary),
            ("case_counts", _counts(case_counts, "case_counts")),
            ("warnings", _strings(warnings, "warnings")),
            ("witnesses", _hashes(witnesses, "witnesses")),
            ("artifacts", _hashes(artifacts, "artifacts")),
            ("logs", _hashes(logs, "logs")),
            ("checks_performed", _strings(checks_performed, "checks_performed")),
            (
                "dependency_receipts",
                _hashes(dependency_receipts, "dependency_receipts"),
            ),
            ("remaining_branches", _strings(remaining_branches, "remaining_branches")),
            (
                "exact_certificate",
                None
                if exact_certificate is None
                else _hash(exact_certificate, "exact_certificate"),
            ),
        ):
            object.__setattr__(value, name, item)
        return value

    def __repr__(self) -> str:
        return "CalculationResult(<packet>)"


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ResultReceipt:
    contract: CalculationContract
    contract_hash: SemanticHash
    contract_id: str
    input_hashes: tuple[tuple[str, SemanticHash], ...]
    source_anchors: tuple[SemanticHash, ...]
    convention_manifests: tuple[SemanticHash, ...]
    algorithms: tuple[tuple[str, str], ...]
    backend_request: str
    backend_name: str
    backend_version: str
    bounds: tuple[tuple[str, str | int], ...]
    primary_implementation_hash: SemanticHash
    primary_result_hash: SemanticHash
    environment: ExecutionEnvironment
    started_at: str
    finished_at: str
    execution_status: TerminalExecutionStatus
    mathematical_outcome: str
    evidence_tier: str
    summary: str
    case_counts: tuple[tuple[str, int], ...]
    warnings: tuple[str, ...]
    witnesses: tuple[SemanticHash, ...]
    artifacts: tuple[SemanticHash, ...]
    logs: tuple[SemanticHash, ...]
    checks_performed: tuple[str, ...]
    dependency_receipts: tuple[SemanticHash, ...]
    remaining_branches: tuple[str, ...]
    fixtures: tuple[SemanticHash, ...]
    test_artifacts: tuple[SemanticHash, ...]
    exact_certificate: SemanticHash | None
    independent_checks: tuple[IndependentCheck, ...]
    proof_obligations: tuple[SemanticHash, ...]
    proof_certificate: SemanticHash | None
    reviewer_attestations: tuple[SemanticHash, ...]
    reproduction_attestations: tuple[SemanticHash, ...]
    supersedes: SemanticHash | None
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ReceiptError("result_receipt", "is runner-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("ResultReceipt cannot be subclassed")

    @property
    def receipt_id(self) -> str:
        _ensure_receipt(self)
        return f"receipt:sha256:{self.semantic_hash.digest}"

    def to_record(self) -> dict[str, object]:
        _ensure_receipt(self)
        return result_receipt_record(self)

    def canonical_bytes(self) -> bytes:
        _ensure_receipt(self)
        return result_receipt_canonical_bytes(self)

    def __repr__(self) -> str:
        return f"ResultReceipt(receipt_id={self.receipt_id!r})"

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is ResultReceipt
            and self.canonical_bytes() == other.canonical_bytes()
        )

    def __hash__(self) -> int:
        _ensure_receipt(self)
        return int.from_bytes(
            bytes.fromhex(self.semantic_hash.digest[:16]), "big", signed=True
        )


def _contract_snapshot(value: object) -> CalculationContract:
    if type(value) is not CalculationContract:
        raise ReceiptError("contract", "must be an exact CalculationContract")
    try:
        parsed = contract_record_registry().from_record(
            calculation_contract_record(value)
        )
        if type(
            parsed
        ) is not CalculationContract or calculation_contract_canonical_bytes(
            parsed
        ) != calculation_contract_canonical_bytes(value):
            raise ReceiptError("contract", "content address integrity rejected")
        return parsed
    except (ContractError, SchemaError, TypeError, ValueError, AttributeError) as error:
        raise ReceiptError("contract", "content address integrity rejected") from error


def _primary(contract: CalculationContract) -> SemanticHash:
    body = {
        "algorithms": [{"key": k, "value": v} for k, v in contract.algorithms],
        "backendRequest": contract.backend_request,
        "backendName": contract.backend_name,
        "backendVersion": contract.backend_version,
    }
    encoded = json.dumps(
        body, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return SemanticHash("sha256", hashlib.sha256(encoded).hexdigest())


def _count_value(counts: tuple[tuple[str, int], ...], key: str) -> int | None:
    return next((value for name, value in counts if name == key), None)


def _declared_cases(bounds: tuple[tuple[str, str | int], ...]) -> int | None:
    return next(
        (value for name, value in bounds if name == "cases" and type(value) is int),
        None,
    )


def _complete_counts(
    counts: tuple[tuple[str, int], ...],
    bounds: tuple[tuple[str, str | int], ...],
    certificate: SemanticHash | None,
) -> bool:
    expected, run = (
        _count_value(counts, "casesExpected"),
        _count_value(counts, "casesRun"),
    )
    if expected is None or run is None or expected != run:
        return False
    declared = _declared_cases(bounds)
    return expected == declared if declared is not None else certificate is not None


def _primary_result(
    contract_hash: SemanticHash,
    outcome: str,
    counts: tuple[tuple[str, int], ...],
    witnesses: tuple[SemanticHash, ...],
    artifacts: tuple[SemanticHash, ...],
    remaining_branches: tuple[str, ...],
) -> SemanticHash:
    body = {
        "contractHash": _hash_record(contract_hash),
        "outcome": outcome,
        "caseCounts": [{"key": key, "value": value} for key, value in counts],
        "witnesses": [_hash_record(value) for value in witnesses],
        "artifacts": [_hash_record(value) for value in artifacts],
        "remainingBranches": list(remaining_branches),
    }
    encoded = json.dumps(
        body, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return SemanticHash("sha256", hashlib.sha256(encoded).hexdigest())


def _tier_obligations(
    *,
    tier: str,
    execution_status: str,
    outcome: str,
    case_counts: tuple[tuple[str, int], ...],
    contract_bounds: tuple[tuple[str, str | int], ...],
    backend_request: str,
    remaining_branches: tuple[str, ...],
    has_assumptions: bool,
    witnesses: tuple[SemanticHash, ...],
    fixtures: tuple[SemanticHash, ...],
    test_artifacts: tuple[SemanticHash, ...],
    exact_certificate: SemanticHash | None,
    independent_checks: tuple[IndependentCheck, ...],
    primary: SemanticHash,
    primary_result: SemanticHash,
    proof_obligations: tuple[SemanticHash, ...],
    proof_certificate: SemanticHash | None,
    reproduction_attestations: tuple[SemanticHash, ...],
    supersedes: SemanticHash | None,
) -> None:
    if tier not in _TIERS:
        raise ReceiptError("evidence_tier", "is unsupported")
    if execution_status == "failed" and tier != "E1-executed":
        raise ReceiptError("evidence_tier", "failed receipts are E1 only")
    exactness = exact_certificate is not None or backend_request == "exact"
    if outcome == "verified_within_domain" and (
        not _complete_counts(case_counts, contract_bounds, exact_certificate)
        or not exactness
    ):
        raise ReceiptError(
            "case_counts",
            "verified outcome needs exhaustive counts and exactness evidence",
        )
    if outcome == "counterexample" and not witnesses:
        raise ReceiptError("witnesses", "counterexample needs a witness")
    if outcome == "no_go_within_assumptions" and not has_assumptions:
        raise ReceiptError(
            "assumptions", "no-go outcome needs a declared contract assumption"
        )
    if outcome == "inconclusive" and (
        not remaining_branches
        or (not contract_bounds and _count_value(case_counts, "casesRun") is None)
    ):
        raise ReceiptError(
            "remaining_branches",
            "inconclusive outcome needs remaining branches and explored bounds",
        )
    if tier == "E1-executed":
        if any(
            (
                fixtures,
                test_artifacts,
                independent_checks,
                proof_obligations,
                proof_certificate is not None,
                reproduction_attestations,
            )
        ):
            raise ReceiptError(
                "promotion_support", "E1 cannot carry promotion evidence"
            )
        return
    if supersedes is None or not fixtures or not test_artifacts:
        raise ReceiptError(
            "promotion_support", "E2 and above need supersession, fixtures, and tests"
        )
    if tier in _TIERS[2:] and (
        not _complete_counts(case_counts, contract_bounds, exact_certificate)
        or not exactness
    ):
        raise ReceiptError(
            "promotion_support",
            "E3 and above need exhaustive counts and exactness evidence",
        )
    if tier in _TIERS[3:] and not any(
        check.implementation_hash != primary and check.result_hash == primary_result
        for check in independent_checks
    ):
        raise ReceiptError(
            "independent_checks", "E4 and above need a distinct implementation"
        )
    if tier in _TIERS[4:] and not proof_obligations and proof_certificate is None:
        raise ReceiptError(
            "proof_obligations", "E5 and above need proof obligations or certificate"
        )
    if tier == "E6-reproduced" and not reproduction_attestations:
        raise ReceiptError(
            "reproduction_attestations", "E6 needs reproduction attestation hashes"
        )


def _build_receipt(
    *,
    contract: object,
    environment: object,
    started_at: object,
    finished_at: object,
    execution_status: object,
    outcome: object,
    summary: object,
    case_counts: object = (),
    warnings: object = (),
    witnesses: object = (),
    artifacts: object = (),
    logs: object = (),
    checks_performed: object = (),
    dependency_receipts: object = (),
    remaining_branches: object = (),
    fixtures: object = (),
    test_artifacts: object = (),
    exact_certificate: object | None = None,
    independent_checks: object = (),
    proof_obligations: object = (),
    proof_certificate: object | None = None,
    reviewer_attestations: object = (),
    reproduction_attestations: object = (),
    supersedes: object | None = None,
    evidence_tier: object = "E1-executed",
) -> ResultReceipt:
    """Private sealed construction surface for runner and future promotion."""
    checked_contract = _contract_snapshot(contract)
    checked_environment = _environment(environment)
    checked_start, checked_finish = (
        _timestamp(started_at, "started_at"),
        _timestamp(finished_at, "finished_at"),
    )
    status, checked_outcome, checked_tier = (
        _text(execution_status, "execution_status", identifier=True),
        _text(outcome, "mathematical_outcome", identifier=True),
        _text(evidence_tier, "evidence_tier", identifier=True),
    )
    checked_summary = _text(summary, "summary")
    if (
        status not in _TERMINAL_EXECUTION_STATUSES
        or checked_outcome not in _OUTCOMES
        or checked_finish < checked_start
    ):
        raise ReceiptError(
            "classification", "has invalid status, outcome, or timestamps"
        )
    terminal_status = cast(TerminalExecutionStatus, status)
    if status == "completed" and checked_outcome == "implementation_error":
        raise ReceiptError(
            "outcome", "completed receipt cannot be implementation_error"
        )
    if status == "failed" and checked_outcome != "implementation_error":
        raise ReceiptError("outcome", "failed receipt must be implementation_error")
    if checked_tier in _TIERS[:4] and _DISALLOWED_E1.search(checked_summary):
        raise ReceiptError("summary", "uses unsupported evidence-tier claim language")
    checked_counts, checked_warnings = (
        _counts(case_counts, "case_counts"),
        _strings(warnings, "warnings"),
    )
    checked_witnesses, checked_artifacts, checked_logs = (
        _hashes(witnesses, "witnesses"),
        _hashes(artifacts, "artifacts"),
        _hashes(logs, "logs"),
    )
    checked_checks, checked_dependencies = (
        _strings(checks_performed, "checks_performed"),
        _hashes(dependency_receipts, "dependency_receipts"),
    )
    checked_remaining = _strings(remaining_branches, "remaining_branches")
    checked_fixtures, checked_tests = (
        _hashes(fixtures, "fixtures"),
        _hashes(test_artifacts, "test_artifacts"),
    )
    checked_exact = (
        None
        if exact_certificate is None
        else _hash(exact_certificate, "exact_certificate")
    )
    checked_independent, checked_obligations = (
        _checks(independent_checks),
        _hashes(proof_obligations, "proof_obligations"),
    )
    checked_proof = (
        None
        if proof_certificate is None
        else _hash(proof_certificate, "proof_certificate")
    )
    checked_reviewers, checked_reproductions = (
        _hashes(reviewer_attestations, "reviewer_attestations"),
        _hashes(reproduction_attestations, "reproduction_attestations"),
    )
    checked_supersedes = None if supersedes is None else _hash(supersedes, "supersedes")
    primary = _primary(checked_contract)
    primary_result = _primary_result(
        checked_contract.semantic_hash,
        checked_outcome,
        checked_counts,
        checked_witnesses,
        checked_artifacts,
        checked_remaining,
    )
    _tier_obligations(
        tier=checked_tier,
        execution_status=terminal_status,
        outcome=checked_outcome,
        case_counts=checked_counts,
        contract_bounds=checked_contract.bounds,
        backend_request=checked_contract.backend_request,
        remaining_branches=checked_remaining,
        has_assumptions=bool(checked_contract.assumptions),
        witnesses=checked_witnesses,
        fixtures=checked_fixtures,
        test_artifacts=checked_tests,
        exact_certificate=checked_exact,
        independent_checks=checked_independent,
        primary=primary,
        primary_result=primary_result,
        proof_obligations=checked_obligations,
        proof_certificate=checked_proof,
        reproduction_attestations=checked_reproductions,
        supersedes=checked_supersedes,
    )
    value = object.__new__(ResultReceipt)
    fields = (
        ("contract", checked_contract),
        ("contract_hash", checked_contract.semantic_hash),
        ("contract_id", checked_contract.contract_id),
        ("input_hashes", checked_contract.input_hashes),
        ("source_anchors", checked_contract.source_anchors),
        ("convention_manifests", checked_contract.convention_manifests),
        ("algorithms", checked_contract.algorithms),
        ("backend_request", checked_contract.backend_request),
        ("backend_name", checked_contract.backend_name),
        ("backend_version", checked_contract.backend_version),
        ("bounds", checked_contract.bounds),
        ("primary_implementation_hash", primary),
        ("primary_result_hash", primary_result),
        ("environment", checked_environment),
        ("started_at", checked_start),
        ("finished_at", checked_finish),
        ("execution_status", status),
        ("mathematical_outcome", checked_outcome),
        ("evidence_tier", checked_tier),
        ("summary", checked_summary),
        ("case_counts", checked_counts),
        ("warnings", checked_warnings),
        ("witnesses", checked_witnesses),
        ("artifacts", checked_artifacts),
        ("logs", checked_logs),
        ("checks_performed", checked_checks),
        ("dependency_receipts", checked_dependencies),
        ("remaining_branches", checked_remaining),
        ("fixtures", checked_fixtures),
        ("test_artifacts", checked_tests),
        ("exact_certificate", checked_exact),
        ("independent_checks", checked_independent),
        ("proof_obligations", checked_obligations),
        ("proof_certificate", checked_proof),
        ("reviewer_attestations", checked_reviewers),
        ("reproduction_attestations", checked_reproductions),
        ("supersedes", checked_supersedes),
        ("semantic_hash", SemanticHash("sha256", "0" * 64)),
    )
    for name, item in fields:
        object.__setattr__(value, name, item)
    object.__setattr__(value, "semantic_hash", _record_hash(value))
    return value


def _packet_valid(
    packet: object, contract: CalculationContract
) -> tuple[
    str,
    str,
    tuple[tuple[str, int], ...],
    tuple[str, ...],
    tuple[SemanticHash, ...],
    tuple[SemanticHash, ...],
    tuple[SemanticHash, ...],
    tuple[str, ...],
    tuple[SemanticHash, ...],
    tuple[str, ...],
    SemanticHash | None,
]:
    if type(packet) is not CalculationResult:
        raise ReceiptError("calculation_result", "callable returned an invalid packet")
    outcome, summary = (
        _text(packet.outcome, "outcome", identifier=True),
        _text(packet.summary, "summary"),
    )
    if outcome not in contract.acceptable_outcomes:
        raise ReceiptError("outcome", "is not acceptable under the contract")
    return (
        outcome,
        summary,
        _counts(packet.case_counts, "case_counts"),
        _strings(packet.warnings, "warnings"),
        _hashes(packet.witnesses, "witnesses"),
        _hashes(packet.artifacts, "artifacts"),
        _hashes(packet.logs, "logs"),
        _strings(packet.checks_performed, "checks_performed"),
        _hashes(packet.dependency_receipts, "dependency_receipts"),
        _strings(packet.remaining_branches, "remaining_branches"),
        None
        if packet.exact_certificate is None
        else _hash(packet.exact_certificate, "exact_certificate"),
    )


def _failure(
    contract: CalculationContract,
    environment: ExecutionEnvironment,
    started: str,
    code: str,
) -> ResultReceipt:
    finished = _timestamp(_utc_now(), "finished_at")
    return _build_receipt(
        contract=contract,
        environment=environment,
        started_at=started,
        finished_at=max(started, finished),
        execution_status="failed",
        outcome="implementation_error",
        summary="calculation did not produce an accepted result packet",
        warnings=(code,),
    )


def run_calculation(
    contract: object,
    calculation: Callable[[CalculationContract], object],
    *,
    environment: object | None = None,
) -> ResultReceipt:
    """Run explicit trusted code once; runner authority is always E1-executed."""
    snapshot, receipt_contract = (
        _contract_snapshot(contract),
        _contract_snapshot(contract),
    )
    if not callable(calculation):
        raise ReceiptError("calculation", "must be a trusted callable")
    env = (
        ExecutionEnvironment.create()
        if environment is None
        else _environment(environment)
    )
    started = _timestamp(_utc_now(), "started_at")
    try:
        raw = calculation(snapshot)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return _failure(receipt_contract, env, started, "calculation_failed")
    try:
        (
            outcome,
            summary,
            counts,
            warnings,
            witnesses,
            artifacts,
            logs,
            checks,
            dependencies,
            remaining_branches,
            certificate,
        ) = _packet_valid(raw, receipt_contract)
        finished = _timestamp(_utc_now(), "finished_at")
        return _build_receipt(
            contract=receipt_contract,
            environment=env,
            started_at=started,
            finished_at=max(started, finished),
            execution_status="completed",
            outcome=outcome,
            summary=summary,
            case_counts=counts,
            warnings=warnings,
            witnesses=witnesses,
            artifacts=artifacts,
            logs=logs,
            checks_performed=checks,
            dependency_receipts=dependencies,
            remaining_branches=remaining_branches,
            exact_certificate=certificate,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return _failure(receipt_contract, env, started, "result_packet_rejected")


def _environment(value: object) -> ExecutionEnvironment:
    if type(value) is not ExecutionEnvironment:
        raise ReceiptError("environment", "must be an exact ExecutionEnvironment")
    return ExecutionEnvironment.create(
        platform=value.platform,
        python_version=value.python_version,
        software_revision=value.software_revision,
    )


def _mutable(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _mutable(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_mutable(item) for item in value]
    return value


def _check_record(value: IndependentCheck) -> dict[str, object]:
    return {
        "methodId": value.method_id,
        "implementationHash": _hash_record(value.implementation_hash),
        "resultHash": _hash_record(value.result_hash),
    }


def _pair_record(value: tuple[tuple[str, object], ...]) -> list[dict[str, object]]:
    return [{"key": key, "value": item} for key, item in value]


def _body(value: ResultReceipt) -> dict[str, object]:
    return {
        "contractRecord": calculation_contract_record(value.contract),
        "contractHash": _hash_record(value.contract_hash),
        "contractId": value.contract_id,
        "inputHashes": [
            {"role": k, "hash": _hash_record(h)} for k, h in value.input_hashes
        ],
        "sourceAnchors": [_hash_record(h) for h in value.source_anchors],
        "conventionManifests": [_hash_record(h) for h in value.convention_manifests],
        "algorithms": _pair_record(value.algorithms),
        "backendRequest": value.backend_request,
        "backendName": value.backend_name,
        "backendVersion": value.backend_version,
        "bounds": _pair_record(value.bounds),
        "primaryImplementationHash": _hash_record(value.primary_implementation_hash),
        "primaryResultHash": _hash_record(value.primary_result_hash),
        "environment": {
            "platform": value.environment.platform,
            "pythonVersion": value.environment.python_version,
            "softwareRevision": value.environment.software_revision,
        },
        "startedAt": value.started_at,
        "finishedAt": value.finished_at,
        "executionStatus": value.execution_status,
        "mathematicalOutcome": value.mathematical_outcome,
        "evidenceTier": value.evidence_tier,
        "summary": value.summary,
        "caseCounts": _pair_record(value.case_counts),
        "warnings": list(value.warnings),
        "witnesses": [_hash_record(h) for h in value.witnesses],
        "artifacts": [_hash_record(h) for h in value.artifacts],
        "logs": [_hash_record(h) for h in value.logs],
        "checksPerformed": list(value.checks_performed),
        "dependencyReceipts": [_hash_record(h) for h in value.dependency_receipts],
        "remainingBranches": list(value.remaining_branches),
        "fixtures": [_hash_record(h) for h in value.fixtures],
        "testArtifacts": [_hash_record(h) for h in value.test_artifacts],
        "exactCertificate": None
        if value.exact_certificate is None
        else _hash_record(value.exact_certificate),
        "independentChecks": [_check_record(item) for item in value.independent_checks],
        "proofObligations": [_hash_record(h) for h in value.proof_obligations],
        "proofCertificate": None
        if value.proof_certificate is None
        else _hash_record(value.proof_certificate),
        "reviewerAttestations": [_hash_record(h) for h in value.reviewer_attestations],
        "reproductionAttestations": [
            _hash_record(h) for h in value.reproduction_attestations
        ],
        "supersedes": None
        if value.supersedes is None
        else _hash_record(value.supersedes),
    }


def _encode(value: ResultReceipt) -> dict[str, JSONValue]:
    return cast(dict[str, JSONValue], _body(value))


RECEIPT_RECORD_REGISTRY = SerializerRegistry().with_codec(
    _TAG, _VERSION, ResultReceipt, _encode, lambda record: _parse(record)
)


def _record_hash(value: ResultReceipt) -> SemanticHash:
    try:
        return SemanticHash(
            "sha256",
            hashlib.sha256(
                canonical_json(value, registry=RECEIPT_RECORD_REGISTRY)
            ).hexdigest(),
        )
    except SchemaError as error:
        raise ReceiptError(
            "result_receipt", "canonical record resource limit"
        ) from error


def _ensure_receipt(value: object) -> None:
    if type(value) is not ResultReceipt:
        raise ReceiptError("result_receipt", "must be an exact ResultReceipt")
    try:
        if _hash(value.semantic_hash, "semantic_hash") != _record_hash(value):
            raise ReceiptError("result_receipt", "content address integrity rejected")
    except (AttributeError, TypeError, ValueError, SchemaError) as error:
        raise ReceiptError(
            "result_receipt", "content address integrity rejected"
        ) from error


def result_receipt_record(value: object) -> dict[str, object]:
    _ensure_receipt(value)
    return cast(dict[str, object], RECEIPT_RECORD_REGISTRY.to_record(value))


def result_receipt_canonical_bytes(value: object) -> bytes:
    _ensure_receipt(value)
    return canonical_json(value, registry=RECEIPT_RECORD_REGISTRY)


def result_receipt_registry() -> SerializerRegistry:
    return RECEIPT_RECORD_REGISTRY


def _mapping(value: object, keys: frozenset[str], field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReceiptError(field, "has an invalid record shape")
    return value


def _parse_hash(value: object, field: str) -> SemanticHash:
    fields = _mapping(value, frozenset(("algorithm", "digest")), field)
    if type(fields["algorithm"]) is not str or type(fields["digest"]) is not str:
        raise ReceiptError(field, "has an invalid semantic hash")
    try:
        return SemanticHash(fields["algorithm"], fields["digest"])
    except Exception as error:
        raise ReceiptError(field, "has an invalid semantic hash") from error


def _tuple(value: object, field: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise ReceiptError(field, "has an invalid record shape")
    return _items(value, field)


def _parse_hashes(value: object, field: str) -> tuple[SemanticHash, ...]:
    return _hashes(
        tuple(_parse_hash(item, field) for item in _tuple(value, field)), field
    )


def _parse_pairs(
    value: object, field: str, *, counts: bool = False
) -> tuple[tuple[str, object], ...]:
    pairs = tuple(
        (
            _mapping(item, frozenset(("key", "value")), field)["key"],
            _mapping(item, frozenset(("key", "value")), field)["value"],
        )
        for item in _tuple(value, field)
    )
    return cast(
        tuple[tuple[str, object], ...],
        _counts(pairs, field) if counts else _pairs(pairs, field),
    )


def _parse_inputs(value: object) -> tuple[tuple[str, SemanticHash], ...]:
    pairs: list[tuple[str, SemanticHash]] = []
    for item in _tuple(value, "input_hashes"):
        fields = _mapping(item, frozenset(("role", "hash")), "input_hashes")
        pairs.append(
            (
                _text(fields["role"], "input_hashes", identifier=True),
                _parse_hash(fields["hash"], "input_hashes"),
            )
        )
    if len({name for name, _ in pairs}) != len(pairs):
        raise ReceiptError("input_hashes", "contains duplicate roles")
    return tuple(sorted(pairs))


def _parse_environment(value: object) -> ExecutionEnvironment:
    fields = _mapping(
        value,
        frozenset(("platform", "pythonVersion", "softwareRevision")),
        "environment",
    )
    return ExecutionEnvironment.create(
        platform=fields["platform"],
        python_version=fields["pythonVersion"],
        software_revision=fields["softwareRevision"],
    )


def _parse_checks(value: object) -> tuple[IndependentCheck, ...]:
    return _checks(
        tuple(
            IndependentCheck.create(
                method_id=_mapping(
                    item,
                    frozenset(("methodId", "implementationHash", "resultHash")),
                    "independent_checks",
                )["methodId"],
                implementation_hash=_parse_hash(
                    _mapping(
                        item,
                        frozenset(("methodId", "implementationHash", "resultHash")),
                        "independent_checks",
                    )["implementationHash"],
                    "independent_checks",
                ),
                result_hash=_parse_hash(
                    _mapping(
                        item,
                        frozenset(("methodId", "implementationHash", "resultHash")),
                        "independent_checks",
                    )["resultHash"],
                    "independent_checks",
                ),
            )
            for item in _tuple(value, "independent_checks")
        )
    )


def _parse_contract(value: object) -> CalculationContract:
    try:
        parsed = contract_record_registry().from_record(
            cast(dict[str, JSONValue], _mutable(value))
        )
    except (SchemaError, TypeError, ValueError) as error:
        raise ReceiptError(
            "contract_record", "is not an exact registered contract"
        ) from error
    if type(parsed) is not CalculationContract:
        raise ReceiptError("contract_record", "is not an exact registered contract")
    return _contract_snapshot(parsed)


def _keys() -> frozenset[str]:
    return frozenset(("schemaType", "schemaVersion", *(_body_keys())))


def _body_keys() -> tuple[str, ...]:
    return (
        "contractRecord",
        "contractHash",
        "contractId",
        "inputHashes",
        "sourceAnchors",
        "conventionManifests",
        "algorithms",
        "backendRequest",
        "backendName",
        "backendVersion",
        "bounds",
        "primaryImplementationHash",
        "primaryResultHash",
        "environment",
        "startedAt",
        "finishedAt",
        "executionStatus",
        "mathematicalOutcome",
        "evidenceTier",
        "summary",
        "caseCounts",
        "warnings",
        "witnesses",
        "artifacts",
        "logs",
        "checksPerformed",
        "dependencyReceipts",
        "remainingBranches",
        "fixtures",
        "testArtifacts",
        "exactCertificate",
        "independentChecks",
        "proofObligations",
        "proofCertificate",
        "reviewerAttestations",
        "reproductionAttestations",
        "supersedes",
    )


def _parse(record: object) -> ResultReceipt:
    fields = _mapping(record, _keys(), "result_receipt")
    if fields["schemaType"] != _TAG or fields["schemaVersion"] != _VERSION:
        raise ReceiptError("result_receipt", "has an invalid schema identity")
    contract = _parse_contract(fields["contractRecord"])
    expected = {
        "contractHash": _hash_record(contract.semantic_hash),
        "contractId": contract.contract_id,
        "inputHashes": [
            {"role": k, "hash": _hash_record(h)} for k, h in contract.input_hashes
        ],
        "sourceAnchors": [_hash_record(h) for h in contract.source_anchors],
        "conventionManifests": [_hash_record(h) for h in contract.convention_manifests],
        "algorithms": _pair_record(contract.algorithms),
        "backendRequest": contract.backend_request,
        "backendName": contract.backend_name,
        "backendVersion": contract.backend_version,
        "bounds": _pair_record(contract.bounds),
        "primaryImplementationHash": _hash_record(_primary(contract)),
        "primaryResultHash": _hash_record(
            _primary_result(
                contract.semantic_hash,
                _text(fields["mathematicalOutcome"], "mathematical_outcome"),
                cast(
                    tuple[tuple[str, int], ...],
                    _parse_pairs(fields["caseCounts"], "case_counts", counts=True),
                ),
                _parse_hashes(fields["witnesses"], "witnesses"),
                _parse_hashes(fields["artifacts"], "artifacts"),
                _strings(
                    _tuple(fields["remainingBranches"], "remaining_branches"),
                    "remaining_branches",
                ),
            )
        ),
    }
    for name, value in expected.items():
        if _mutable(fields[name]) != value:
            raise ReceiptError(
                "contract_dependency", "does not match embedded contract"
            )
    return _build_receipt(
        contract=contract,
        environment=_parse_environment(fields["environment"]),
        started_at=fields["startedAt"],
        finished_at=fields["finishedAt"],
        execution_status=fields["executionStatus"],
        outcome=fields["mathematicalOutcome"],
        summary=fields["summary"],
        case_counts=_parse_pairs(fields["caseCounts"], "case_counts", counts=True),
        warnings=_tuple(fields["warnings"], "warnings"),
        witnesses=_parse_hashes(fields["witnesses"], "witnesses"),
        artifacts=_parse_hashes(fields["artifacts"], "artifacts"),
        logs=_parse_hashes(fields["logs"], "logs"),
        checks_performed=_tuple(fields["checksPerformed"], "checks_performed"),
        dependency_receipts=_parse_hashes(
            fields["dependencyReceipts"], "dependency_receipts"
        ),
        remaining_branches=_tuple(fields["remainingBranches"], "remaining_branches"),
        fixtures=_parse_hashes(fields["fixtures"], "fixtures"),
        test_artifacts=_parse_hashes(fields["testArtifacts"], "test_artifacts"),
        exact_certificate=None
        if fields["exactCertificate"] is None
        else _parse_hash(fields["exactCertificate"], "exact_certificate"),
        independent_checks=_parse_checks(fields["independentChecks"]),
        proof_obligations=_parse_hashes(
            fields["proofObligations"], "proof_obligations"
        ),
        proof_certificate=None
        if fields["proofCertificate"] is None
        else _parse_hash(fields["proofCertificate"], "proof_certificate"),
        reviewer_attestations=_parse_hashes(
            fields["reviewerAttestations"], "reviewer_attestations"
        ),
        reproduction_attestations=_parse_hashes(
            fields["reproductionAttestations"], "reproduction_attestations"
        ),
        supersedes=None
        if fields["supersedes"] is None
        else _parse_hash(fields["supersedes"], "supersedes"),
        evidence_tier=fields["evidenceTier"],
    )
