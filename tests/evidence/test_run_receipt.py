"""V00-075 tests: runner-issued E1 receipts never become scientific claims."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
from itertools import cycle
from typing import cast

import pytest

from anyalgebra.core.parents import SemanticHash
from anyalgebra.evidence.contracts import (
    CalculationContract,
    calculation_contract_canonical_bytes,
)
from anyalgebra.evidence.run import (
    CalculationResult,
    ExecutionEnvironment,
    IndependentCheck,
    ReceiptError,
    ResultReceipt,
    result_receipt_record,
    result_receipt_registry,
    run_calculation,
)
import anyalgebra.evidence.run as run_module
from anyalgebra.persistence.registry import SchemaError


def _hash(character: str = "a") -> SemanticHash:
    return SemanticHash("sha256", character * 64)


def _contract(
    *,
    outcomes: tuple[str, ...] = ("constructed",),
    assumptions: object = (),
    backend_request: str = "exact",
    bounds: object = (("cases", 8),),
) -> CalculationContract:
    return CalculationContract.create(
        contract_name="receipt-test",
        project_id="test.project",
        question="test question",
        acceptable_outcomes=outcomes,
        input_hashes=(("input", _hash("a")),),
        source_anchors=(),
        convention_manifests=(),
        algorithms=(("reference", "1"),),
        backend_request=backend_request,
        backend_name="reference",
        backend_version="1",
        assumptions=assumptions,
        theorem_hypotheses=(),
        bounds=bounds,
        required_cross_checks=(),
        expected_artifacts=(),
        acceptance_predicates=("stored",),
    )


def _packet(outcome: str = "constructed") -> CalculationResult:
    return CalculationResult.create(
        outcome=outcome,
        summary="constructed object on declared fixture",
        case_counts=(
            (("casesExpected", 8), ("casesRun", 8))
            if outcome == "verified_within_domain"
            else (("casesRun", 8),)
        ),
        remaining_branches=("unexplored-case",) if outcome == "inconclusive" else (),
        warnings=("small fixture",),
        witnesses=(_hash("b"),),
        artifacts=(_hash("c"),),
        logs=(_hash("d"),),
        checks_performed=("direct multiplication",),
    )


@pytest.fixture(autouse=True)
def _clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run_module, "_test_original_utc_now", run_module._utc_now, raising=False
    )
    monkeypatch.setattr(run_module, "_utc_now", lambda: "2026-07-24T12:00:00Z")


def test_positive_e1_receipt_snapshots_contract_and_environment() -> None:
    contract = _contract()
    environment = ExecutionEnvironment.create(
        platform="test", python_version="3.x", software_revision="abc"
    )
    receipt = run_calculation(
        contract,
        lambda given: _packet() if given == contract else None,
        environment=environment,
    )
    assert receipt.execution_status == "completed"
    assert receipt.mathematical_outcome == "constructed"
    assert receipt.evidence_tier == "E1-executed"
    assert receipt.contract_hash == contract.semantic_hash
    assert receipt.contract_id == contract.contract_id
    assert receipt.environment == environment
    assert receipt.started_at == "2026-07-24T12:00:00Z"
    assert receipt.finished_at == "2026-07-24T12:00:00Z"
    assert receipt.receipt_id == f"receipt:sha256:{receipt.semantic_hash.digest}"


@pytest.mark.parametrize(
    "outcome",
    (
        "counterexample",
        "inconclusive",
        "not_applicable",
        "reproduction_only",
        "verified_within_domain",
    ),
)
def test_negative_and_inconclusive_outcomes_are_durable_data(outcome: str) -> None:
    receipt = run_calculation(
        _contract(outcomes=(outcome,)), lambda _: _packet(outcome)
    )
    assert receipt.execution_status == "completed"
    assert receipt.mathematical_outcome == outcome
    assert receipt.evidence_tier == "E1-executed"


def test_failure_and_invalid_packet_return_sanitized_implementation_error() -> None:
    def fails(_: CalculationContract) -> object:
        raise RuntimeError("secret filesystem path")

    for calculation in (fails, lambda _: object(), lambda _: _packet("constructed")):
        receipt = run_calculation(_contract(outcomes=("inconclusive",)), calculation)
        assert receipt.execution_status == "failed"
        assert receipt.mathematical_outcome == "implementation_error"
        assert receipt.evidence_tier == "E1-executed"
        assert "secret" not in receipt.summary
        assert "path" not in receipt.summary
    assert run_calculation(_contract(outcomes=("inconclusive",)), fails).warnings == (
        "calculation_failed",
    )


def test_interrupts_are_not_converted_to_receipts() -> None:
    with pytest.raises(KeyboardInterrupt):
        run_calculation(
            _contract(), lambda _: (_ for _ in ()).throw(KeyboardInterrupt())
        )
    with pytest.raises(SystemExit):
        run_calculation(_contract(), lambda _: (_ for _ in ()).throw(SystemExit()))


def test_packet_is_narrow_and_e1_language_is_restrained() -> None:
    with pytest.raises(ReceiptError, match="callable-owned"):
        CalculationResult.create(outcome="implementation_error", summary="bad")
    for word in ("proved", "CONFIRMED", "complete", "ruled out"):
        with pytest.raises(ReceiptError, match="unsupported E1"):
            CalculationResult.create(outcome="constructed", summary=f"it is {word}")
    with pytest.raises(TypeError):
        CalculationResult.create(
            outcome="constructed", summary="ok", evidence_tier="E6"
        )  # type: ignore[call-arg]


def test_contract_snapshot_survives_later_callable_mutation() -> None:
    contract = _contract()

    def mutates(given: CalculationContract) -> CalculationResult:
        object.__setattr__(given, "bounds", (("cases", 999),))
        return _packet()

    receipt = run_calculation(contract, mutates)
    assert receipt.bounds == (("cases", 8),)
    assert contract.bounds == (("cases", 8),)

    def changes_acceptance(given: CalculationContract) -> CalculationResult:
        object.__setattr__(given, "acceptable_outcomes", ("inconclusive",))
        return CalculationResult.create(outcome="inconclusive", summary="bounded")

    receipt = run_calculation(contract, changes_acceptance)
    assert receipt.mathematical_outcome == "implementation_error"
    assert receipt.warnings == ("result_packet_rejected",)


def test_default_environment_is_explicit_and_not_guessed() -> None:
    receipt = run_calculation(_contract(), lambda _: _packet())
    assert receipt.environment == ExecutionEnvironment.create()
    with pytest.raises(ReceiptError):
        run_calculation(_contract(), lambda _: _packet(), environment=object())


def test_wall_clock_regression_is_clamped_without_losing_a_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = cycle(("2026-07-24T12:00:01Z", "2026-07-24T12:00:00Z"))
    monkeypatch.setattr(run_module, "_utc_now", lambda: next(times))
    assert (
        run_calculation(_contract(), lambda _: _packet()).finished_at
        == "2026-07-24T12:00:01Z"
    )
    receipt = run_calculation(
        _contract(), lambda _: (_ for _ in ()).throw(RuntimeError("failure"))
    )
    assert receipt.finished_at == "2026-07-24T12:00:01Z"


def test_round_trip_golden_bytes_and_derived_id() -> None:
    receipt = run_calculation(_contract(), lambda _: _packet())
    assert (
        receipt.canonical_bytes()
        == b'{"algorithms":[{"key":"reference","value":"1"}],"artifacts":[{"algorithm":"sha256","digest":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"}],"backendName":"reference","backendRequest":"exact","backendVersion":"1","bounds":[{"key":"cases","value":8}],"caseCounts":[{"key":"casesRun","value":8}],"checksPerformed":["direct multiplication"],"contractHash":{"algorithm":"sha256","digest":"6545eb9e323336a58a5e8ba1094fedaccba58dbd8b9ce5f51cf3ca61a9d8310e"},"contractId":"contract:sha256:6545eb9e323336a58a5e8ba1094fedaccba58dbd8b9ce5f51cf3ca61a9d8310e","contractRecord":{"acceptableOutcomes":["constructed"],"acceptancePredicates":["stored"],"algorithms":[{"key":"reference","value":"1"}],"assumptions":[],"backend":{"name":"reference","request":"exact","version":"1"},"bounds":[{"key":"cases","value":8}],"contractName":"receipt-test","conventionManifests":[],"expectedArtifacts":[],"inputHashes":[{"hash":{"algorithm":"sha256","digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},"role":"input"}],"projectId":"test.project","question":"test question","requiredCrossChecks":[],"schemaType":"anyalgebra.evidence.calculation_contract","schemaVersion":1,"sourceAnchors":[],"theoremHypotheses":[]},"conventionManifests":[],"dependencyReceipts":[],"environment":{"platform":"unavailable","pythonVersion":"unavailable","softwareRevision":"unavailable"},"evidenceTier":"E1-executed","exactCertificate":null,"executionStatus":"completed","finishedAt":"2026-07-24T12:00:00Z","fixtures":[],"independentChecks":[],"inputHashes":[{"hash":{"algorithm":"sha256","digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},"role":"input"}],"logs":[{"algorithm":"sha256","digest":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"}],"mathematicalOutcome":"constructed","primaryImplementationHash":{"algorithm":"sha256","digest":"9033022c36a7065f9db832f19f040708e7eac38fe0c935e5a91646d4e33921bc"},"primaryResultHash":{"algorithm":"sha256","digest":"b92d73296189a34abfd525dc6e9aea95bde3a786bb881ec4adc20a782ec20dae"},"proofCertificate":null,"proofObligations":[],"remainingBranches":[],"reproductionAttestations":[],"reviewerAttestations":[],"schemaType":"anyalgebra.evidence.result_receipt","schemaVersion":1,"sourceAnchors":[],"startedAt":"2026-07-24T12:00:00Z","summary":"constructed object on declared fixture","supersedes":null,"testArtifacts":[],"warnings":["small fixture"],"witnesses":[{"algorithm":"sha256","digest":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}]}'
    )
    assert (
        receipt.semantic_hash.digest
        == "afc842aa5c38cf6ba8c7085d0e84beb42af3230a1120a1e27b7135855b98e3da"
    )


def test_parser_round_trip_is_strict_and_no_functions_execute() -> None:
    receipt = run_calculation(_contract(), lambda _: _packet())
    record = result_receipt_record(receipt)
    parsed = result_receipt_registry().from_record(record)
    assert type(parsed) is ResultReceipt
    assert parsed == receipt
    assert parsed.receipt_id == receipt.receipt_id
    for malformed in (
        {**record, "extra": 1},
        {key: value for key, value in record.items() if key != "summary"},
        {**record, "schemaType": "os.system"},
        {**record, "schemaVersion": 2},
        {**record, "evidenceTier": "E6-reproduced"},
    ):
        with pytest.raises(SchemaError, match=r"parser_failed|unknown_schema"):
            result_receipt_registry().from_record(malformed)


def test_all_semantic_result_fields_change_hash() -> None:
    first = run_calculation(_contract(), lambda _: _packet())
    changed = run_calculation(
        _contract(),
        lambda _: CalculationResult.create(
            outcome="constructed",
            summary="another constructed fixture",
            case_counts=(("casesRun", 7),),
            warnings=(),
            witnesses=(),
            artifacts=(),
            logs=(),
            checks_performed=(),
        ),
    )
    assert first.semantic_hash != changed.semantic_hash


def test_hash_preimage_integrity_tamper_and_safe_repr() -> None:
    receipt = run_calculation(_contract(), lambda _: _packet())
    assert (
        hashlib.sha256(receipt.canonical_bytes()).hexdigest()
        == receipt.semantic_hash.digest
    )
    assert "constructed object" not in repr(receipt)
    with pytest.raises(AttributeError):
        receipt.summary = "x"  # type: ignore[misc]
    object.__setattr__(receipt, "summary", "tampered")
    with pytest.raises(ReceiptError, match="integrity"):
        _ = receipt.receipt_id


def test_future_promotion_fields_are_present_but_v1_parser_rejects_them() -> None:
    receipt = run_calculation(_contract(), lambda _: _packet())
    record = result_receipt_record(receipt)
    for field, value in (
        ("fixtures", [{"algorithm": "sha256", "digest": "e" * 64}]),
        ("independentMethods", ["same backend"]),
        ("proofObligations", [{"algorithm": "sha256", "digest": "f" * 64}]),
        ("reviewerAttestations", ["review"]),
        ("reproductionAttestations", ["attestation"]),
    ):
        altered = dict(record)
        altered[field] = value
        with pytest.raises(SchemaError, match="parser_failed"):
            result_receipt_registry().from_record(altered)


def test_256_257_and_text_boundaries() -> None:
    warnings = tuple(f"w{index}" for index in range(256))
    result = CalculationResult.create(
        outcome="constructed", summary="ok", warnings=warnings
    )
    assert len(result.warnings) == 256
    with pytest.raises(ReceiptError, match="limit"):
        CalculationResult.create(
            outcome="constructed", summary="ok", warnings=(*warnings, "too-many")
        )
    assert (
        CalculationResult.create(outcome="constructed", summary="x" * 4096).summary
        == "x" * 4096
    )
    with pytest.raises(ReceiptError, match="text"):
        CalculationResult.create(outcome="constructed", summary="x" * 4097)


def test_receipt_constructor_is_closed_and_environment_is_immutable() -> None:
    with pytest.raises(ReceiptError, match="runner-owned"):
        ResultReceipt()
    with pytest.raises(ReceiptError, match="factory-owned"):
        ExecutionEnvironment()
    environment = ExecutionEnvironment.create()
    with pytest.raises(AttributeError):
        environment.platform = "x"  # type: ignore[misc]
    for sealed in (ExecutionEnvironment, CalculationResult, ResultReceipt):
        with pytest.raises(TypeError):
            type("Child", (sealed,), {})


def test_packet_and_private_receipt_validation_boundaries() -> None:
    class BrokenIterable:
        def __iter__(self) -> object:
            raise RuntimeError("secret")

    for value in (None, "", "x" * 4097, "bad\ud800"):
        with pytest.raises(ReceiptError):
            run_module._text(value, "field")
    for iterable in ("not-items", BrokenIterable()):
        with pytest.raises(ReceiptError):
            run_module._items(iterable, "field")
    with pytest.raises(ReceiptError):
        run_module._counts((("x", -1),), "counts")
    with pytest.raises(ReceiptError):
        run_module._counts((("x", 1), ("x", 2)), "counts")
    with pytest.raises(ReceiptError):
        run_module._hash("not-a-hash", "field")
    with pytest.raises(ReceiptError):
        run_module._timestamp("yesterday", "field")
    contract = _contract()
    with pytest.raises(ReceiptError, match="classification"):
        run_module._build_receipt(
            contract=contract,
            environment=ExecutionEnvironment.create(),
            started_at="2026-07-24T12:00:01Z",
            finished_at="2026-07-24T12:00:00Z",
            execution_status="completed",
            outcome="constructed",
            summary="constructed fixture",
        )
    with pytest.raises(ReceiptError, match="promotion"):
        run_module._build_receipt(
            contract=contract,
            environment=ExecutionEnvironment.create(),
            started_at="2026-07-24T12:00:00Z",
            finished_at="2026-07-24T12:00:00Z",
            execution_status="completed",
            outcome="constructed",
            summary="constructed fixture",
            fixtures=(_hash("e"),),
        )


def test_exact_packet_collection_boundaries_and_one_mebibyte_limit() -> None:
    hashes = tuple(SemanticHash("sha256", f"{index:064x}") for index in range(256))
    strings = tuple(f"item-{index}" for index in range(256))
    packet = CalculationResult.create(
        outcome="constructed",
        summary="constructed fixture",
        case_counts=tuple((f"count-{index}", index) for index in range(256)),
        warnings=strings,
        witnesses=hashes,
        artifacts=hashes,
        logs=hashes,
        checks_performed=strings,
    )
    assert (
        len(packet.case_counts) == len(packet.warnings) == len(packet.witnesses) == 256
    )
    with pytest.raises(ReceiptError, match="limit"):
        CalculationResult.create(
            outcome="constructed", summary="ok", artifacts=(*hashes, _hash("f"))
        )
    large = tuple("x" * 4096 for _ in range(255))
    # Duplicate strings are intentionally rejected before serialization.
    with pytest.raises(ReceiptError, match="duplicate"):
        CalculationResult.create(outcome="constructed", summary="ok", warnings=large)
    unique_large = tuple(f"{index:04d}" + "x" * 4092 for index in range(255))
    receipt = run_module._build_receipt(
        contract=_contract(),
        environment=ExecutionEnvironment.create(),
        started_at="2026-07-24T12:00:00Z",
        finished_at="2026-07-24T12:00:00Z",
        execution_status="completed",
        outcome="constructed",
        summary="constructed fixture",
        warnings=unique_large,
    )
    assert len(receipt.canonical_bytes()) < 1_048_576
    too_large = tuple(f"{index:04d}" + "x" * 4092 for index in range(256))
    with pytest.raises(ReceiptError, match="resource"):
        run_module._build_receipt(
            contract=_contract(),
            environment=ExecutionEnvironment.create(),
            started_at="2026-07-24T12:00:00Z",
            finished_at="2026-07-24T12:00:00Z",
            execution_status="completed",
            outcome="constructed",
            summary="constructed fixture",
            warnings=too_large,
        )


def test_parser_helpers_and_contract_identity_are_closed() -> None:
    receipt = run_calculation(_contract(), lambda _: _packet())
    record = result_receipt_record(receipt)
    with pytest.raises(ReceiptError):
        run_module._mapping([], frozenset(), "field")
    with pytest.raises(ReceiptError):
        run_module._parse_hash({"algorithm": "sha256", "digest": 1}, "hash")
    with pytest.raises(ReceiptError):
        run_module._tuple([], "field")
    altered = dict(record)
    altered["contractId"] = "contract:sha256:" + "0" * 64
    with pytest.raises(SchemaError, match="parser_failed"):
        result_receipt_registry().from_record(altered)
    with pytest.raises(ReceiptError):
        run_calculation(object(), lambda _: _packet())
    with pytest.raises(ReceiptError):
        run_calculation(_contract(), object())  # type: ignore[arg-type]


def test_remaining_closed_branches_and_integrity_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def late_failure() -> object:
        yield "first"
        raise RuntimeError("late")

    with pytest.raises(ReceiptError, match="whitespace"):
        run_module._text(" padded ", "field", identifier=True)
    with pytest.raises(ReceiptError, match="snapshot"):
        run_module._items(late_failure(), "field")
    with pytest.raises(ReceiptError, match="timestamp"):
        run_module._timestamp("2026-13-24T12:00:00Z", "time")
    with pytest.raises(ReceiptError, match="duplicate"):
        run_module._hashes((_hash("a"), _hash("a")), "hashes")
    with pytest.raises(ReceiptError):
        run_module._counts(("not-a-pair",), "counts")
    with pytest.raises(ReceiptError):
        run_module._counts((("x",),), "counts")
    assert run_module._test_original_utc_now().endswith("Z")  # type: ignore[attr-defined]
    environment = ExecutionEnvironment.create()
    assert repr(environment) == "ExecutionEnvironment(<metadata>)"
    assert environment == ExecutionEnvironment.create()
    assert hash(environment) == hash(ExecutionEnvironment.create())
    assert repr(_packet()) == "CalculationResult(<packet>)"
    with pytest.raises(ReceiptError, match="factory-owned"):
        CalculationResult()
    with pytest.raises(ReceiptError, match="factory requires"):
        CalculationResult.create.__func__(  # type: ignore[attr-defined]
            object, outcome="constructed", summary="x"
        )
    with pytest.raises(ReceiptError, match="factory requires"):
        ExecutionEnvironment.create.__func__(object)  # type: ignore[attr-defined]
    receipt = run_calculation(_contract(), lambda _: _packet())
    assert receipt.to_record() == result_receipt_record(receipt)
    assert hash(receipt) == hash(
        result_receipt_registry().from_record(receipt.to_record())
    )
    with pytest.raises(ReceiptError):
        result_receipt_record(object())
    with pytest.raises(ReceiptError):
        run_module.result_receipt_canonical_bytes(object())
    with pytest.raises(ReceiptError):
        run_module._ensure_receipt(object())
    stale_contract = _contract()
    object.__setattr__(stale_contract, "question", "changed")
    with pytest.raises(ReceiptError, match="integrity"):
        run_calculation(stale_contract, lambda _: _packet())
    snapshot_contract = _contract()

    class WrongRegistry:
        def from_record(self, record: object) -> object:
            del record
            return object()

    monkeypatch.setattr(run_module, "contract_record_registry", lambda: WrongRegistry())
    with pytest.raises(ReceiptError, match="integrity"):
        run_module._contract_snapshot(snapshot_contract)
    monkeypatch.setattr(
        run_module,
        "contract_record_registry",
        lambda: type("Registry", (), {"from_record": lambda _, __: _contract()})(),
    )
    original_bytes = calculation_contract_canonical_bytes
    monkeypatch.setattr(
        run_module,
        "calculation_contract_canonical_bytes",
        lambda value: b"a" if value is snapshot_contract else b"b",
    )
    with pytest.raises(ReceiptError, match="content address"):
        run_module._contract_snapshot(snapshot_contract)
    monkeypatch.setattr(
        run_module, "calculation_contract_canonical_bytes", original_bytes
    )
    contract = _contract()
    for status, outcome in (
        ("completed", "implementation_error"),
        ("failed", "constructed"),
    ):
        with pytest.raises(ReceiptError, match="outcome"):
            run_module._build_receipt(
                contract=contract,
                environment=environment,
                started_at="2026-07-24T12:00:00Z",
                finished_at="2026-07-24T12:00:00Z",
                execution_status=status,
                outcome=outcome,
                summary="constructed fixture",
            )
    with pytest.raises(ReceiptError, match="claim"):
        run_module._build_receipt(
            contract=contract,
            environment=environment,
            started_at="2026-07-24T12:00:00Z",
            finished_at="2026-07-24T12:00:00Z",
            execution_status="completed",
            outcome="constructed",
            summary="proved fixture",
        )
    bad_hash = run_calculation(_contract(), lambda _: _packet())
    object.__setattr__(bad_hash, "semantic_hash", "bad")
    with pytest.raises(ReceiptError, match="integrity"):
        hash(bad_hash)
    with pytest.raises(ReceiptError):
        run_module._parse_hash({"algorithm": "sha256", "digest": "bad"}, "hash")
    with pytest.raises(ReceiptError):
        run_module._pairs((("key", object()),), "pairs")
    with pytest.raises(ReceiptError):
        run_module._pairs(("not-a-pair",), "pairs")
    with pytest.raises(ReceiptError):
        run_module._pairs((("key",),), "pairs")
    with pytest.raises(ReceiptError):
        run_module._pairs((("key", "x"), ("key", "y")), "pairs")
    with pytest.raises(ReceiptError):
        run_module._parse_inputs(
            (
                {"role": "same", "hash": {"algorithm": "sha256", "digest": "a" * 64}},
                {"role": "same", "hash": {"algorithm": "sha256", "digest": "b" * 64}},
            )
        )
    direct = dict(result_receipt_record(receipt))
    direct["schemaType"] = "wrong"
    with pytest.raises(ReceiptError, match="schema identity"):
        run_module._parse(direct)


def test_contract_record_pins_every_duplicate_dependency() -> None:
    receipt = run_calculation(_contract(), lambda _: _packet())
    record = result_receipt_record(receipt)
    mutations = {
        "contractHash": {"algorithm": "sha256", "digest": "0" * 64},
        "contractId": "contract:sha256:" + "0" * 64,
        "inputHashes": [],
        "sourceAnchors": [{"algorithm": "sha256", "digest": "a" * 64}],
        "conventionManifests": [{"algorithm": "sha256", "digest": "b" * 64}],
        "algorithms": [{"key": "other", "value": "2"}],
        "backendRequest": "other",
        "backendName": "other",
        "backendVersion": "2",
        "bounds": [{"key": "other", "value": 1}],
        "primaryImplementationHash": {"algorithm": "sha256", "digest": "c" * 64},
    }
    for field, replacement in mutations.items():
        altered = dict(record)
        altered[field] = replacement
        with pytest.raises(SchemaError, match="parser_failed"):
            result_receipt_registry().from_record(altered)
    altered_contract = dict(record)
    embedded = dict(cast(dict[str, object], altered_contract["contractRecord"]))
    embedded["backendName"] = "other"
    altered_contract["contractRecord"] = embedded
    with pytest.raises(SchemaError, match="parser_failed"):
        result_receipt_registry().from_record(altered_contract)


def test_tampered_packet_is_revalidated_and_becomes_fixed_failure() -> None:
    packet = _packet()
    object.__setattr__(packet, "case_counts", (("bad", -1),))
    receipt = run_calculation(_contract(), lambda _: packet)
    assert (receipt.execution_status, receipt.mathematical_outcome) == (
        "failed",
        "implementation_error",
    )
    assert receipt.warnings == ("result_packet_rejected",)
    packet = _packet()
    object.__setattr__(packet, "artifacts", ("not-a-hash",))
    assert run_calculation(_contract(), lambda _: packet).warnings == (
        "result_packet_rejected",
    )
    packet = _packet()

    def interrupting_warnings() -> object:
        raise KeyboardInterrupt()
        yield "unreachable"

    object.__setattr__(packet, "warnings", interrupting_warnings())
    with pytest.raises(KeyboardInterrupt):
        run_calculation(_contract(), lambda _: packet)
    environment = ExecutionEnvironment.create()
    object.__setattr__(environment, "platform", " ")
    with pytest.raises(ReceiptError):
        run_module._build_receipt(
            contract=_contract(),
            environment=environment,
            started_at="2026-07-24T12:00:00Z",
            finished_at="2026-07-24T12:00:00Z",
            execution_status="completed",
            outcome="constructed",
            summary="constructed fixture",
        )


def test_acceptance_and_promotion_tier_obligations_roundtrip() -> None:
    assert (
        run_calculation(
            _contract(outcomes=("verified_within_domain",)),
            lambda _: CalculationResult.create(
                outcome="verified_within_domain", summary="bounded calculation"
            ),
        ).mathematical_outcome
        == "implementation_error"
    )
    assert (
        run_calculation(
            _contract(outcomes=("verified_within_domain",)),
            lambda _: CalculationResult.create(
                outcome="verified_within_domain",
                summary="bounded calculation",
                case_counts=(("casesExpected", 8), ("casesRun", 8)),
                exact_certificate=_hash("e"),
            ),
        ).execution_status
        == "completed"
    )
    assert (
        run_calculation(
            _contract(outcomes=("counterexample",)),
            lambda _: CalculationResult.create(
                outcome="counterexample", summary="candidate failed"
            ),
        ).mathematical_outcome
        == "implementation_error"
    )
    contract = _contract()
    base = run_calculation(contract, lambda _: _packet())
    common = {
        "contract": contract,
        "environment": ExecutionEnvironment.create(),
        "started_at": "2026-07-24T12:00:00Z",
        "finished_at": "2026-07-24T12:00:00Z",
        "execution_status": "completed",
        "outcome": "constructed",
        "summary": "constructed fixture",
        "fixtures": (_hash("e"),),
        "test_artifacts": (_hash("f"),),
        "supersedes": base.semantic_hash,
    }
    e2 = run_module._build_receipt(**common, evidence_tier="E2-regression")
    assert result_receipt_registry().from_record(e2.to_record()) == e2
    e3 = run_module._build_receipt(
        **common,
        evidence_tier="E3-bounded-exact",
        case_counts=(("casesExpected", 8), ("casesRun", 8)),
    )
    check = IndependentCheck.create(
        method_id="independent",
        implementation_hash=_hash("1"),
        result_hash=run_module._primary_result(
            contract.semantic_hash,
            "constructed",
            (("casesExpected", 8), ("casesRun", 8)),
            (),
            (),
            (),
        ),
    )
    e4 = run_module._build_receipt(
        **common,
        evidence_tier="E4-cross-checked",
        case_counts=(("casesExpected", 8), ("casesRun", 8)),
        independent_checks=(check,),
    )
    e5 = run_module._build_receipt(
        **common,
        evidence_tier="E5-proof-linked",
        case_counts=(("casesExpected", 8), ("casesRun", 8)),
        independent_checks=(check,),
        proof_obligations=(_hash("3"),),
        proof_certificate=_hash("4"),
    )
    e6 = run_module._build_receipt(
        **common,
        evidence_tier="E6-reproduced",
        case_counts=(("casesExpected", 8), ("casesRun", 8)),
        independent_checks=(check,),
        proof_obligations=(_hash("3"),),
        proof_certificate=_hash("4"),
        reproduction_attestations=(_hash("5"),),
    )
    assert [e3.evidence_tier, e4.evidence_tier, e5.evidence_tier, e6.evidence_tier] == [
        "E3-bounded-exact",
        "E4-cross-checked",
        "E5-proof-linked",
        "E6-reproduced",
    ]
    same = IndependentCheck.create(
        method_id="wrapper",
        implementation_hash=e3.primary_implementation_hash,
        result_hash=_hash("6"),
    )
    with pytest.raises(ReceiptError, match="distinct"):
        run_module._build_receipt(
            **common,
            evidence_tier="E4-cross-checked",
            case_counts=(("casesExpected", 8), ("casesRun", 8)),
            independent_checks=(same,),
        )
    for receipt in (e3, e4, e5, e6):
        assert result_receipt_registry().from_record(receipt.to_record()) == receipt


def test_independent_check_and_remaining_tier_validation_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    check = IndependentCheck.create(
        method_id="method",
        implementation_hash=_hash("a"),
        result_hash=run_module._primary_result(
            contract.semantic_hash,
            "constructed",
            (("casesExpected", 8), ("casesRun", 8)),
            (),
            (),
            (),
        ),
    )
    assert repr(check) == "IndependentCheck(<hashes>)"
    assert check == IndependentCheck.create(
        method_id="method",
        implementation_hash=_hash("a"),
        result_hash=run_module._primary_result(
            contract.semantic_hash,
            "constructed",
            (("casesExpected", 8), ("casesRun", 8)),
            (),
            (),
            (),
        ),
    )
    assert hash(check) == hash(check)
    with pytest.raises(ReceiptError):
        IndependentCheck()
    with pytest.raises(TypeError):
        type("BadCheck", (IndependentCheck,), {})
    with pytest.raises(ReceiptError):
        IndependentCheck.create.__func__(  # type: ignore[attr-defined]
            object, method_id="m", implementation_hash=_hash(), result_hash=_hash("b")
        )
    with pytest.raises(ReceiptError):
        run_module._checks((object(),))
    with pytest.raises(ReceiptError, match="duplicate"):
        run_module._checks((check, check))
    assert run_module._pairs((("limit", 2),), "pairs") == (("limit", 2),)
    common = {
        "contract": contract,
        "environment": ExecutionEnvironment.create(),
        "started_at": "2026-07-24T12:00:00Z",
        "finished_at": "2026-07-24T12:00:00Z",
        "execution_status": "completed",
        "outcome": "constructed",
        "summary": "constructed fixture",
        "fixtures": (_hash("c"),),
        "test_artifacts": (_hash("d"),),
        "supersedes": _hash("e"),
    }
    for kwargs, match in (
        ({"evidence_tier": "unknown"}, "unsupported"),
        ({"evidence_tier": "E3-bounded-exact"}, "exhaustive"),
        (
            {
                "evidence_tier": "E5-proof-linked",
                "case_counts": (("casesExpected", 8), ("casesRun", 8)),
                "independent_checks": (check,),
            },
            "proof",
        ),
        (
            {
                "evidence_tier": "E6-reproduced",
                "case_counts": (("casesExpected", 8), ("casesRun", 8)),
                "independent_checks": (check,),
                "proof_obligations": (_hash("f"),),
                "proof_certificate": _hash("1"),
            },
            "reproduction",
        ),
    ):
        with pytest.raises(ReceiptError, match=match):
            run_module._build_receipt(**common, **kwargs)
    with pytest.raises(ReceiptError, match="failed"):
        run_module._build_receipt(
            **{
                **common,
                "execution_status": "failed",
                "outcome": "implementation_error",
                "evidence_tier": "E2-regression",
            }
        )
    assert run_module._parse_inputs(
        ({"role": "x", "hash": {"algorithm": "sha256", "digest": "a" * 64}},)
    ) == (("x", _hash("a")),)

    class WrongRegistry:
        def from_record(self, record: object) -> object:
            del record
            return object()

    monkeypatch.setattr(run_module, "contract_record_registry", lambda: WrongRegistry())
    with pytest.raises(ReceiptError, match="contract_record"):
        run_module._parse_contract({})


def test_claim_language_is_bounded_through_e4_but_allowed_at_e5() -> None:
    contract = _contract()
    base = run_calculation(contract, lambda _: _packet())
    common = {
        "contract": contract,
        "environment": ExecutionEnvironment.create(),
        "started_at": "2026-07-24T12:00:00Z",
        "finished_at": "2026-07-24T12:00:00Z",
        "execution_status": "completed",
        "outcome": "constructed",
        "fixtures": (_hash("e"),),
        "test_artifacts": (_hash("f"),),
        "supersedes": base.semantic_hash,
        "case_counts": (("casesExpected", 8), ("casesRun", 8)),
        "independent_checks": (
            IndependentCheck.create(
                method_id="independent",
                implementation_hash=_hash("1"),
                result_hash=run_module._primary_result(
                    contract.semantic_hash,
                    "constructed",
                    (("casesExpected", 8), ("casesRun", 8)),
                    (),
                    (),
                    (),
                ),
            ),
        ),
    }
    for tier in (
        "E1-executed",
        "E2-regression",
        "E3-bounded-exact",
        "E4-cross-checked",
    ):
        for word in (
            "Proved",
            "confirmed",
            "COMPLETE",
            "ruled out",
            "RULED-OUT",
            "ruled_out",
            "ruled/out",
            "ruled — out",
            "ruled\u2013__ /out",
        ):
            values = {**common, "evidence_tier": tier, "summary": f"result {word}"}
            if tier == "E1-executed":
                values = {
                    **values,
                    "fixtures": (),
                    "test_artifacts": (),
                    "supersedes": None,
                    "independent_checks": (),
                }
            with pytest.raises(ReceiptError, match="claim language"):
                run_module._build_receipt(**values)
    for phrase in ("ruled-out", "ruled_out", "ruled/out", "ruled—out"):
        with pytest.raises(ReceiptError, match="claim language"):
            CalculationResult.create(outcome="constructed", summary=phrase)
    for permitted in ("completed fixture", "incomplete search", "rule d out"):
        assert (
            CalculationResult.create(outcome="constructed", summary=permitted).summary
            == permitted
        )
    receipt = run_module._build_receipt(
        **common,
        evidence_tier="E5-proof-linked",
        summary="proved under stated proof obligations",
        proof_obligations=(_hash("3"),),
        proof_certificate=_hash("4"),
    )
    assert receipt.evidence_tier == "E5-proof-linked"


def test_e1_supersedes_and_runner_dependency_receipts_are_semantic() -> None:
    base = run_calculation(_contract(), lambda _: _packet())
    corrected = run_module._build_receipt(
        contract=_contract(),
        environment=ExecutionEnvironment.create(),
        started_at="2026-07-24T12:00:00Z",
        finished_at="2026-07-24T12:00:00Z",
        execution_status="completed",
        outcome="constructed",
        summary="corrected fixture",
        supersedes=base.semantic_hash,
    )
    parsed = cast(
        ResultReceipt, result_receipt_registry().from_record(corrected.to_record())
    )
    assert parsed.supersedes == base.semantic_hash
    altered = corrected.to_record()
    altered["supersedes"] = {"algorithm": "sha256", "digest": "e" * 64}
    assert (
        cast(
            ResultReceipt, result_receipt_registry().from_record(altered)
        ).semantic_hash
        != corrected.semantic_hash
    )
    dependencies = (_hash("e"), _hash("f"))
    receipt = run_calculation(
        _contract(),
        lambda _: CalculationResult.create(
            outcome="constructed",
            summary="constructed with dependencies",
            dependency_receipts=dependencies,
        ),
    )
    assert receipt.dependency_receipts == dependencies
    assert result_receipt_registry().from_record(receipt.to_record()) == receipt
    packet = CalculationResult.create(outcome="constructed", summary="x")
    object.__setattr__(packet, "dependency_receipts", ("bad",))
    assert run_calculation(_contract(), lambda _: packet).warnings == (
        "result_packet_rejected",
    )
    many = tuple(SemanticHash("sha256", f"{index:064x}") for index in range(256))
    assert (
        len(
            CalculationResult.create(
                outcome="constructed", summary="x", dependency_receipts=many
            ).dependency_receipts
        )
        == 256
    )
    with pytest.raises(ReceiptError, match="limit"):
        CalculationResult.create(
            outcome="constructed",
            summary="x",
            dependency_receipts=(*many, _hash("a")),
        )


def test_no_go_assumptions_and_inconclusive_continuations() -> None:
    no_assumption = run_calculation(
        _contract(outcomes=("no_go_within_assumptions",)),
        lambda _: CalculationResult.create(
            outcome="no_go_within_assumptions", summary="obstruction found"
        ),
    )
    assert no_assumption.mathematical_outcome == "implementation_error"
    assumed = run_calculation(
        _contract(
            outcomes=("no_go_within_assumptions",),
            assumptions=(("signature", "fixed"),),
        ),
        lambda _: CalculationResult.create(
            outcome="no_go_within_assumptions", summary="obstruction found"
        ),
    )
    assert assumed.mathematical_outcome == "no_go_within_assumptions"
    missing_branches = run_calculation(
        _contract(outcomes=("inconclusive",)),
        lambda _: CalculationResult.create(
            outcome="inconclusive",
            summary="bounded search",
            case_counts=(("casesRun", 1),),
        ),
    )
    assert missing_branches.mathematical_outcome == "implementation_error"
    receipt = run_calculation(
        _contract(outcomes=("inconclusive",)),
        lambda _: CalculationResult.create(
            outcome="inconclusive",
            summary="bounded search",
            remaining_branches=("next",),
            case_counts=(("casesRun", 1),),
        ),
    )
    assert receipt.remaining_branches == ("next",)


def test_verified_and_e3_exactness_requirements_are_explicit() -> None:
    approximate = _contract(
        outcomes=("verified_within_domain",), backend_request="approximate"
    )
    missing_exactness = run_calculation(
        approximate,
        lambda _: CalculationResult.create(
            outcome="verified_within_domain",
            summary="finite domain checked",
            case_counts=(("casesExpected", 8), ("casesRun", 8)),
        ),
    )
    assert missing_exactness.mathematical_outcome == "implementation_error"
    certified = run_calculation(
        approximate,
        lambda _: CalculationResult.create(
            outcome="verified_within_domain",
            summary="finite domain checked",
            case_counts=(("casesExpected", 8), ("casesRun", 8)),
            exact_certificate=_hash("e"),
        ),
    )
    assert certified.mathematical_outcome == "verified_within_domain"
    bad_counts: tuple[tuple[tuple[str, int], ...], ...] = (
        (),
        (("casesExpected", 8), ("casesRun", 7)),
    )
    for counts in bad_counts:

        def certified_packet(
            _: CalculationContract,
            case_counts: tuple[tuple[str, int], ...] = counts,
        ) -> CalculationResult:
            return CalculationResult.create(
                outcome="verified_within_domain",
                summary="certificate provided",
                case_counts=case_counts,
                exact_certificate=_hash("e"),
            )

        receipt = run_calculation(
            approximate,
            certified_packet,
        )
        assert receipt.mathematical_outcome == "implementation_error"

    undeclared = _contract(
        outcomes=("verified_within_domain",), bounds=(("label", "finite"),)
    )
    missing_certificate = run_calculation(
        undeclared,
        lambda _: CalculationResult.create(
            outcome="verified_within_domain",
            summary="finite domain checked",
            case_counts=(("casesExpected", 8), ("casesRun", 8)),
        ),
    )
    assert missing_certificate.mathematical_outcome == "implementation_error"
    assert (
        run_calculation(
            undeclared,
            lambda _: CalculationResult.create(
                outcome="verified_within_domain",
                summary="finite domain checked",
                case_counts=(("casesExpected", 8), ("casesRun", 8)),
                exact_certificate=_hash("e"),
            ),
        ).mathematical_outcome
        == "verified_within_domain"
    )


def test_e4_agreement_and_e5_proof_alternatives_round_trip() -> None:
    contract = _contract()
    base = run_calculation(contract, lambda _: _packet())
    common = {
        "contract": contract,
        "environment": ExecutionEnvironment.create(),
        "started_at": "2026-07-24T12:00:00Z",
        "finished_at": "2026-07-24T12:00:00Z",
        "execution_status": "completed",
        "outcome": "constructed",
        "summary": "constructed fixture",
        "case_counts": (("casesExpected", 8), ("casesRun", 8)),
        "fixtures": (_hash("e"),),
        "test_artifacts": (_hash("f"),),
        "supersedes": base.semantic_hash,
    }
    matching = run_module._primary_result(
        contract.semantic_hash,
        "constructed",
        (("casesExpected", 8), ("casesRun", 8)),
        (),
        (),
        (),
    )
    check = IndependentCheck.create(
        method_id="independent", implementation_hash=_hash("1"), result_hash=matching
    )
    disagreement = IndependentCheck.create(
        method_id="disagrees", implementation_hash=_hash("2"), result_hash=_hash("3")
    )
    with pytest.raises(ReceiptError, match="distinct implementation"):
        run_module._build_receipt(
            **common,
            evidence_tier="E4-cross-checked",
            independent_checks=(disagreement,),
        )
    obligations = run_module._build_receipt(
        **common,
        evidence_tier="E5-proof-linked",
        independent_checks=(check,),
        proof_obligations=(_hash("4"),),
    )
    certificate = run_module._build_receipt(
        **common,
        evidence_tier="E5-proof-linked",
        independent_checks=(check,),
        proof_certificate=_hash("5"),
    )
    for receipt in (obligations, certificate):
        assert result_receipt_registry().from_record(receipt.to_record()) == receipt
    with pytest.raises(ReceiptError, match="proof obligations or certificate"):
        run_module._build_receipt(
            **common,
            evidence_tier="E5-proof-linked",
            independent_checks=(check,),
        )

    record = obligations.to_record()
    record["primaryResultHash"] = {"algorithm": "sha256", "digest": "9" * 64}
    with pytest.raises(SchemaError, match="parser_failed"):
        result_receipt_registry().from_record(record)
    altered = run_module._build_receipt(
        **{**common, "artifacts": (_hash("a"),)},
        evidence_tier="E2-regression",
    )
    assert altered.primary_result_hash != obligations.primary_result_hash


def test_e4_primary_result_is_bound_to_contract_and_remaining_branches() -> None:
    contract = _contract()
    other_contract = _contract(bounds=(("cases", 9),))
    payload = ("constructed", (("casesExpected", 8), ("casesRun", 8)), (), (), ())
    original_target = run_module._primary_result(contract.semantic_hash, *payload)
    assert original_target != run_module._primary_result(
        other_contract.semantic_hash, *payload
    )
    other_base = run_calculation(other_contract, lambda _: _packet())
    other_common = {
        "contract": other_contract,
        "environment": ExecutionEnvironment.create(),
        "started_at": "2026-07-24T12:00:00Z",
        "finished_at": "2026-07-24T12:00:00Z",
        "execution_status": "completed",
        "outcome": "constructed",
        "summary": "constructed fixture",
        "case_counts": (("casesExpected", 9), ("casesRun", 9)),
        "fixtures": (_hash("e"),),
        "test_artifacts": (_hash("f"),),
        "supersedes": other_base.semantic_hash,
        "evidence_tier": "E4-cross-checked",
    }
    foreign_contract_check = IndependentCheck.create(
        method_id="foreign-contract",
        implementation_hash=_hash("1"),
        result_hash=original_target,
    )
    with pytest.raises(ReceiptError, match="distinct implementation"):
        run_module._build_receipt(
            **other_common, independent_checks=(foreign_contract_check,)
        )

    inconclusive_contract = _contract(outcomes=("inconclusive",))
    branches_a = ("branch-a",)
    branches_b = ("branch-b",)
    branch_target = run_module._primary_result(
        inconclusive_contract.semantic_hash,
        "inconclusive",
        (("casesExpected", 8), ("casesRun", 8)),
        (),
        (),
        branches_a,
    )
    assert branch_target != run_module._primary_result(
        inconclusive_contract.semantic_hash,
        "inconclusive",
        (("casesExpected", 8), ("casesRun", 8)),
        (),
        (),
        branches_b,
    )
    branch_base = run_calculation(
        inconclusive_contract, lambda _: _packet("inconclusive")
    )
    branch_common = {
        "contract": inconclusive_contract,
        "environment": ExecutionEnvironment.create(),
        "started_at": "2026-07-24T12:00:00Z",
        "finished_at": "2026-07-24T12:00:00Z",
        "execution_status": "completed",
        "outcome": "inconclusive",
        "summary": "bounded search",
        "case_counts": (("casesExpected", 8), ("casesRun", 8)),
        "remaining_branches": branches_a,
        "fixtures": (_hash("e"),),
        "test_artifacts": (_hash("f"),),
        "supersedes": branch_base.semantic_hash,
        "evidence_tier": "E4-cross-checked",
    }
    agreed = IndependentCheck.create(
        method_id="same-branches",
        implementation_hash=_hash("2"),
        result_hash=branch_target,
    )
    assert (
        run_module._build_receipt(
            **branch_common, independent_checks=(agreed,)
        ).evidence_tier
        == "E4-cross-checked"
    )
    foreign_branch_check = IndependentCheck.create(
        method_id="foreign-branches",
        implementation_hash=_hash("3"),
        result_hash=run_module._primary_result(
            inconclusive_contract.semantic_hash,
            "inconclusive",
            (("casesExpected", 8), ("casesRun", 8)),
            (),
            (),
            branches_b,
        ),
    )
    with pytest.raises(ReceiptError, match="distinct implementation"):
        run_module._build_receipt(
            **branch_common, independent_checks=(foreign_branch_check,)
        )


def test_inconclusive_needs_bounds_or_recognized_case_count() -> None:
    unbounded = _contract(outcomes=("inconclusive",), bounds=())
    arbitrary = run_calculation(
        unbounded,
        lambda _: CalculationResult.create(
            outcome="inconclusive",
            summary="search paused",
            remaining_branches=("next",),
            case_counts=(("bananas", 0),),
        ),
    )
    assert arbitrary.mathematical_outcome == "implementation_error"
    recognized = run_calculation(
        unbounded,
        lambda _: CalculationResult.create(
            outcome="inconclusive",
            summary="search paused",
            remaining_branches=("next",),
            case_counts=(("casesRun", 0),),
        ),
    )
    assert recognized.mathematical_outcome == "inconclusive"
