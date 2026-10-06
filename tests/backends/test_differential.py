"""Observable V00-081 contracts for exact backend differential receipts."""

from __future__ import annotations

import hashlib
import runpy
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from anyalgebra.backends import base
from anyalgebra.backends.base import (
    BackendCapabilities,
    BackendOptions,
    BackendRequest,
    BackendResult,
    BackendSupported,
    CapabilitySpec,
    UnsupportedReason,
    negotiate,
)
from anyalgebra.backends.differential import (
    DifferentialError,
    DifferentialStatus,
    compare_enabled_backends,
)
import anyalgebra.backends.differential as differential
from anyalgebra.core.parents import SemanticHash
from anyalgebra.evidence.contracts import CalculationContract
from anyalgebra.evidence.run import ResultReceipt
import anyalgebra.evidence.run as evidence_run
from anyalgebra.structures.outcomes import Defined, Failed, Indeterminate, Undefined


def _hash(text: str) -> SemanticHash:
    return SemanticHash("sha256", hashlib.sha256(text.encode("utf-8")).hexdigest())


def _request() -> BackendRequest:
    return BackendRequest.create(
        operation="hash_identity",
        domain="semantic_hash",
        algorithm="identity",
        input_hashes=(_hash("input"),),
        limits=(("inputs", 1),),
    )


def _contract(request: BackendRequest, *names: str) -> CalculationContract:
    return CalculationContract.create(
        contract_name="exact-differential",
        project_id="test.backends",
        question="compare exact neutral backend results",
        acceptable_outcomes=(
            "constructed",
            "counterexample",
            "not_applicable",
            "inconclusive",
        ),
        input_hashes=(("input0", request.input_hashes[0]),),
        source_anchors=(),
        convention_manifests=request.convention_hashes,
        algorithms=tuple(
            sorted(
                (
                    ("differential", "v00_081"),
                    *((f"backend.{name}", "1") for name in names),
                )
            )
        ),
        backend_request=str(request.semantic_hash),
        backend_name="anyalgebra.differential",
        backend_version="v00_081",
        assumptions=(),
        theorem_hypotheses=(),
        bounds=(("cases", 2),),
        required_cross_checks=(),
        expected_artifacts=(),
        acceptance_predicates=("capture",),
    )


class Adapter:
    def __init__(self, name: str, value: SemanticHash | None = None) -> None:
        self.name = name
        self.value = _hash(name) if value is None else value
        self.execute_calls = 0

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities.create(
            name=self.name,
            version="1",
            specifications=(
                CapabilitySpec.create(
                    operation="hash_identity",
                    domain="semantic_hash",
                    algorithm="identity",
                    exact=True,
                ),
            ),
            limits=(("inputs", 1),),
        )

    def supports(self, request: BackendRequest) -> object:
        return negotiate(self.capabilities(), request)

    def execute(self, request: BackendRequest, *, options: BackendOptions) -> object:
        self.execute_calls += 1
        supported = negotiate(self.capabilities(), request)
        assert hasattr(supported, "provenance")
        return Defined(
            BackendResult.create(
                request=request,
                provenance=supported.provenance,
                options=options,
                value_hash=self.value,
            )
        )


def test_exact_match_is_e1_receipt_bound_and_backend_order_is_deterministic() -> None:
    request = _request()
    contract = _contract(request, "left", "right")
    value = _hash("same")
    left, right = Adapter("left", value), Adapter("right", value)
    report = compare_enabled_backends(request, (right, left), contract=contract)
    assert report.status is DifferentialStatus.EXACT_MATCH
    assert report.receipt.evidence_tier == "E1-executed"
    assert report.receipt.mathematical_outcome == "constructed"
    assert tuple(item.name for item in report.observations) == ("left", "right")
    assert all(item.value_hash == value for item in report.observations)
    assert left.execute_calls == right.execute_calls == 1
    assert not hasattr(report, "evidence_tier")


def test_disagreement_keeps_backend_observation_pairing_and_real_receipt() -> None:
    request = _request()
    report = compare_enabled_backends(
        request,
        (Adapter("left", _hash("left")), Adapter("right", _hash("right"))),
        contract=_contract(request, "left", "right"),
    )
    assert report.status is DifferentialStatus.DISAGREEMENT
    assert report.receipt.mathematical_outcome == "counterexample"
    assert [(item.name, item.value_hash) for item in report.observations] == [
        ("left", _hash("left")),
        ("right", _hash("right")),
    ]
    assert report.receipt_hash == report.receipt.semantic_hash


def test_unsupported_and_non_neutral_outcomes_are_first_class_and_do_not_execute() -> (
    None
):
    request = _request()

    class Unsupported(Adapter):
        def capabilities(self) -> BackendCapabilities:
            return BackendCapabilities.create(
                name="unsupported",
                version="1",
                specifications=(
                    CapabilitySpec.create(
                        operation="hash_identity",
                        domain="semantic_hash",
                        algorithm="identity",
                        exact=True,
                        precondition_hashes=(_hash("different-precondition"),),
                    ),
                ),
                limits=(("inputs", 1),),
            )

    unavailable = Unsupported("unsupported")
    report = compare_enabled_backends(
        request,
        (Adapter("ready"), unavailable),
        contract=_contract(request, "ready", "unsupported"),
    )
    assert report.status is DifferentialStatus.UNAVAILABLE
    assert report.receipt.mathematical_outcome == "not_applicable"
    assert unavailable.execute_calls == 0

    class NativePayload(Adapter):
        def execute(
            self, request: BackendRequest, *, options: BackendOptions
        ) -> object:
            self.execute_calls += 1
            return Indeterminate("bound", bounds=object())

    report = compare_enabled_backends(
        request,
        (Adapter("ready"), NativePayload("native")),
        contract=_contract(request, "ready", "native"),
    )
    assert report.status is DifferentialStatus.EXECUTION_FAILURE
    assert report.receipt.execution_status == "failed"
    assert {item.outcome for item in report.observations} == {
        "defined",
        "invalid_outcome",
    }


def test_contract_binding_and_hostile_adapter_errors_are_closed() -> None:
    request = _request()
    wrong_contract = CalculationContract.create(
        contract_name="wrong",
        project_id="test.backends",
        question="wrong input binding",
        acceptable_outcomes=("constructed",),
        input_hashes=(("wrong", _hash("wrong")),),
        source_anchors=(),
        convention_manifests=(),
        algorithms=(("differential", "v00_081"),),
        backend_request="differential",
        backend_name="differential",
        backend_version="v00_081",
        assumptions=(),
        theorem_hypotheses=(),
        bounds=(),
        required_cross_checks=(),
        expected_artifacts=(),
        acceptance_predicates=("capture",),
    )
    with pytest.raises(DifferentialError, match="contract"):
        compare_enabled_backends(
            request, (Adapter("one"), Adapter("two")), contract=wrong_contract
        )

    class Crashes(Adapter):
        def capabilities(self) -> BackendCapabilities:
            raise RuntimeError("native detail")

    with pytest.raises(DifferentialError, match="capabilities") as raised:
        compare_enabled_backends(
            request,
            (Crashes("bad"), Adapter("good")),
            contract=_contract(request, "bad", "good"),
        )
    assert "detail" not in str(raised.value)


def test_neutral_undefined_and_indeterminate_outcomes_are_durable() -> None:
    request = _request()

    class Neutral(Adapter):
        def __init__(self, name: str, outcome: object) -> None:
            super().__init__(name)
            self.outcome = outcome

        def execute(
            self, request: BackendRequest, *, options: BackendOptions
        ) -> object:
            self.execute_calls += 1
            return self.outcome

    undefined = compare_enabled_backends(
        request,
        (
            Neutral("left", Undefined("outside domain")),
            Neutral("right", Undefined("outside domain")),
        ),
        contract=_contract(request, "left", "right"),
    )
    assert undefined.status is DifferentialStatus.EXACT_MATCH
    assert undefined.receipt.mathematical_outcome == "not_applicable"
    indeterminate = compare_enabled_backends(
        request,
        (
            Neutral("left", Indeterminate("bounded", bounds=None)),
            Neutral("right", Indeterminate("bounded", bounds=None)),
        ),
        contract=_contract(request, "left", "right"),
    )
    assert indeterminate.status is DifferentialStatus.EXACT_MATCH
    assert indeterminate.receipt.mathematical_outcome == "inconclusive"


def test_receipt_replay_example_executes_without_import_side_effects() -> None:
    namespace = runpy.run_path("examples/evidence_replay.py")
    result = cast(Callable[[], dict[str, object]], namespace["run_example"])()
    assert result["round_trip"] is True
    assert result["exact_replay"] == "exact_match"
    assert result["stale_replay"] == "stale"


def test_mutated_observations_reports_and_receipts_fail_structural_validation() -> None:
    request = _request()

    def fresh() -> differential.DifferentialReport:
        return compare_enabled_backends(
            request,
            (Adapter("left", _hash("same")), Adapter("right", _hash("same"))),
            contract=_contract(request, "left", "right"),
        )

    def rehash_observation(report: differential.DifferentialReport) -> None:
        observation = report.observations[0]
        object.__setattr__(
            observation, "semantic_hash", differential._digest(observation._record())
        )
        with suppress(differential.DifferentialError):
            object.__setattr__(
                report, "semantic_hash", differential._digest(report._record())
            )

    def rehash_report(report: differential.DifferentialReport) -> None:
        with suppress(differential.DifferentialError):
            object.__setattr__(
                report, "semantic_hash", differential._digest(report._record())
            )

    mutations: tuple[Callable[[differential.DifferentialReport], None], ...] = (
        lambda report: object.__setattr__(
            report.observations[0], "outcome", "undefined"
        ),
        lambda report: object.__setattr__(report.observations[0], "value_hash", None),
        lambda report: object.__setattr__(report.observations[0], "diagnostic", "bad"),
        lambda report: object.__setattr__(report.observations[0], "operation", "wrong"),
        lambda report: object.__setattr__(report.observations[0], "domain", "wrong"),
        lambda report: object.__setattr__(
            report.observations[0], "request_hash", _hash("wrong")
        ),
        lambda report: object.__setattr__(
            report.observations[0], "specification_hash", _hash("wrong")
        ),
        lambda report: object.__setattr__(
            report.observations[0], "capabilities_hash", _hash("wrong")
        ),
        lambda report: object.__setattr__(
            report.observations[0], "outcome_hash", _hash("wrong")
        ),
        lambda report: object.__setattr__(
            report.observations[0], "comparison_hash", _hash("wrong")
        ),
        lambda report: object.__setattr__(
            report, "observations", tuple(reversed(report.observations))
        ),
        lambda report: object.__setattr__(
            report, "observations", (report.observations[0],) * 2
        ),
        lambda report: object.__setattr__(
            report, "status", DifferentialStatus.DISAGREEMENT
        ),
        lambda report: object.__setattr__(
            report, "common_exact_domain", ("wrong", "domain", "algorithm")
        ),
        lambda report: object.__setattr__(report, "request_hash", _hash("wrong")),
        lambda report: object.__setattr__(report, "contract_hash", _hash("wrong")),
        lambda report: object.__setattr__(report, "receipt_hash", _hash("wrong")),
        lambda report: object.__setattr__(report.receipt, "execution_status", "failed"),
        lambda report: object.__setattr__(
            report.receipt, "mathematical_outcome", "counterexample"
        ),
        lambda report: object.__setattr__(
            report.receipt, "evidence_tier", "E4-cross-checked"
        ),
        lambda report: object.__setattr__(report.receipt, "artifacts", ()),
        lambda report: object.__setattr__(
            report.receipt.contract, "backend_request", "wrong"
        ),
        lambda report: object.__setattr__(
            report.receipt.contract, "backend_name", "wrong"
        ),
        lambda report: object.__setattr__(
            report.receipt.contract, "backend_version", "wrong"
        ),
        lambda report: object.__setattr__(report.receipt.contract, "algorithms", ()),
    )
    for mutate in mutations:
        report = fresh()
        mutate(report)
        rehash_observation(report)
        rehash_report(report)
        with pytest.raises(DifferentialError):
            report._assert()


def test_adapter_cardinality_duplicates_and_capability_failures_do_not_execute() -> (
    None
):
    request = _request()
    for adapters in (
        (),
        (Adapter("one"),),
        tuple(Adapter(f"n{index}") for index in range(33)),
    ):
        with pytest.raises(DifferentialError):
            compare_enabled_backends(request, adapters, contract=_contract(request))
        assert all(adapter.execute_calls == 0 for adapter in adapters)
    first, second = Adapter("same"), Adapter("same")
    with pytest.raises(DifferentialError, match="duplicate"):
        compare_enabled_backends(
            request, (first, second), contract=_contract(request, "same")
        )
    assert first.execute_calls == second.execute_calls == 0

    class Broken(Adapter):
        def capabilities(self) -> BackendCapabilities:
            raise DifferentialError("secret", "adapter detail")

    good, broken = Adapter("good"), Broken("broken")
    with pytest.raises(DifferentialError) as raised:
        compare_enabled_backends(
            request, (good, broken), contract=_contract(request, "good", "broken")
        )
    assert "secret" not in str(raised.value)
    assert good.execute_calls == broken.execute_calls == 0


def test_support_failure_prevents_partial_execution_and_controls_propagate() -> None:
    request = _request()

    class BadSupport(Adapter):
        def supports(self, request: BackendRequest) -> object:
            raise RuntimeError("native")

    good, bad = Adapter("good"), BadSupport("bad")
    report = compare_enabled_backends(
        request, (good, bad), contract=_contract(request, "good", "bad")
    )
    assert report.status is DifferentialStatus.EXECUTION_FAILURE
    assert good.execute_calls == bad.execute_calls == 0

    class Interrupted(Adapter):
        def capabilities(self) -> BackendCapabilities:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        compare_enabled_backends(
            request,
            (Adapter("good"), Interrupted("interrupted")),
            contract=_contract(request, "good", "interrupted"),
        )


def test_neutral_outcomes_failures_and_three_backend_disagreement() -> None:
    request = _request()

    class Outcome(Adapter):
        def __init__(self, name: str, result: object) -> None:
            super().__init__(name)
            self.result = result

        def execute(
            self, request: BackendRequest, *, options: BackendOptions
        ) -> object:
            self.execute_calls += 1
            return self.result

    for left, right, expected in (
        (
            Undefined("outside", None),
            Undefined("outside", None),
            DifferentialStatus.EXACT_MATCH,
        ),
        (
            Undefined("outside", _hash("w")),
            Undefined("other", _hash("w")),
            DifferentialStatus.DISAGREEMENT,
        ),
        (
            Indeterminate("bound", None),
            Indeterminate("bound", None),
            DifferentialStatus.EXACT_MATCH,
        ),
        (
            Indeterminate("bound", _hash("b")),
            Indeterminate("other", _hash("b")),
            DifferentialStatus.DISAGREEMENT,
        ),
    ):
        report = compare_enabled_backends(
            request,
            (Outcome("left", left), Outcome("right", right)),
            contract=_contract(request, "left", "right"),
        )
        assert report.status is expected
    failed = compare_enabled_backends(
        request,
        (
            Outcome("left", Failed("failure", "stage")),
            Outcome("right", Failed("failure", "other")),
        ),
        contract=_contract(request, "left", "right"),
    )
    assert failed.status is DifferentialStatus.EXECUTION_FAILURE
    assert failed.receipt.execution_status == "failed"
    a, b, c = (
        Adapter("a", _hash("same")),
        Adapter("b", _hash("same")),
        Adapter("c", _hash("other")),
    )
    report = compare_enabled_backends(
        request, (a, b, c), contract=_contract(request, "a", "b", "c")
    )
    assert report.status is DifferentialStatus.DISAGREEMENT
    assert report.receipt.mathematical_outcome == "counterexample"
    assert len(report.receipt.witnesses) == 2
    assert (a.execute_calls, b.execute_calls, c.execute_calls) == (1, 1, 1)


def test_method_surface_matrix_closes_without_backend_lookup() -> None:
    request = _request()

    def no_self() -> object:
        return object()

    def extra(self: object, extra: object) -> object:
        del self, extra
        return object()

    def default(self: object = object()) -> object:
        del self
        return object()

    def variadic(self: object, *values: object) -> object:
        del self, values
        return object()

    def keyword_variadic(self: object, **values: object) -> object:
        del self, values
        return object()

    bad_capabilities: tuple[object, ...] = (
        no_self,
        extra,
        default,
        variadic,
        keyword_variadic,
        staticmethod(Adapter.capabilities),
        classmethod(cast(Any, Adapter.capabilities)),
        property(lambda self: Adapter.capabilities),
    )
    for method in bad_capabilities:
        adapter = type("BadCapabilities", (Adapter,), {"capabilities": method})("bad")
        with pytest.raises(DifferentialError) as raised:
            compare_enabled_backends(
                request,
                (adapter, Adapter("good")),
                contract=_contract(request, "bad", "good"),
            )
        assert "native" not in str(raised.value)
        assert adapter.execute_calls == 0

    def supports_extra(self: object, value: object, extra: object) -> object:
        del self, value, extra
        return object()

    def supports_default(self: object, value: object = object()) -> object:
        del self, value
        return object()

    def supports_variadic(self: object, *values: object) -> object:
        del self, values
        return object()

    bad_supports: tuple[object, ...] = (
        no_self,
        supports_extra,
        supports_default,
        supports_variadic,
        keyword_variadic,
        staticmethod(Adapter.supports),
        classmethod(cast(Any, Adapter.supports)),
        property(lambda self: Adapter.supports),
    )
    for method in bad_supports:
        adapter = type("BadSupports", (Adapter,), {"supports": method})("bad")
        report = compare_enabled_backends(
            request,
            (adapter, Adapter("good")),
            contract=_contract(request, "bad", "good"),
        )
        assert report.status is DifferentialStatus.EXECUTION_FAILURE
        assert adapter.execute_calls == 0

    def execute_positional(self: object, value: object, options: object) -> object:
        del self, value, options
        return object()

    def execute_wrong_keyword(self: object, value: object, *, wrong: object) -> object:
        del self, value, wrong
        return object()

    def execute_default_keyword(
        self: object, value: object, *, options: object = object()
    ) -> object:
        del self, value, options
        return object()

    async def execute_async(self: object, value: object, *, options: object) -> object:
        del self, value, options
        return object()

    def execute_generator(self: object, value: object, *, options: object) -> object:
        del self, value, options
        yield object()

    bad_execute: tuple[object, ...] = (
        no_self,
        execute_positional,
        execute_wrong_keyword,
        execute_default_keyword,
        execute_async,
        execute_generator,
        variadic,
        keyword_variadic,
        staticmethod(Adapter.execute),
        classmethod(cast(Any, Adapter.execute)),
        property(lambda self: Adapter.execute),
    )
    for method in bad_execute:
        adapter = type("BadExecute", (Adapter,), {"execute": method})("bad")
        report = compare_enabled_backends(
            request,
            (adapter, Adapter("good")),
            contract=_contract(request, "bad", "good"),
        )
        assert report.status is DifferentialStatus.EXECUTION_FAILURE
        assert adapter.execute_calls == 0


def test_inherited_and_hostile_metaclass_methods_are_safe() -> None:
    class Inherited(Adapter):
        pass

    class HostileMeta(type):
        def __getattribute__(cls, name: str) -> object:
            if name in {"__mro__", "__dict__"}:
                raise RuntimeError("native metaclass detail")
            return super().__getattribute__(name)

    class Hostile(Inherited, metaclass=HostileMeta):
        pass

    request = _request()
    left, right = (
        Hostile("hostile", _hash("same")),
        Inherited("inherited", _hash("same")),
    )
    report = compare_enabled_backends(
        request, (left, right), contract=_contract(request, "hostile", "inherited")
    )
    assert report.status is DifferentialStatus.EXACT_MATCH
    assert left.execute_calls == right.execute_calls == 1


def test_support_forgery_and_toctou_never_start_execution() -> None:
    request = _request()

    class Forged(Adapter):
        def __init__(self, name: str, forged: object) -> None:
            super().__init__(name)
            self.forged = forged

        def supports(self, value: BackendRequest) -> object:
            del value
            return self.forged

    original = Adapter("original")
    expected = negotiate(original.capabilities(), request)
    assert type(expected) is BackendSupported
    other_request = BackendRequest.create(
        operation="hash_identity",
        domain="semantic_hash",
        algorithm="identity",
        input_hashes=(_hash("other"),),
        limits=(("inputs", 1),),
    )
    cross_request = negotiate(original.capabilities(), other_request)
    assert type(cross_request) is BackendSupported
    wrong_support = negotiate(
        BackendCapabilities.create(
            name="other",
            version="1",
            specifications=original.capabilities().specifications,
            limits=(("inputs", 1),),
        ),
        request,
    )
    assert type(wrong_support) is BackendSupported
    corrupted = negotiate(original.capabilities(), request)
    assert type(corrupted) is BackendSupported
    object.__setattr__(corrupted, "semantic_hash", _hash("corrupt"))
    for forged in (object(), cross_request, wrong_support, corrupted):
        bad = Forged("bad", forged)
        good = Adapter("good")
        report = compare_enabled_backends(
            request, (bad, good), contract=_contract(request, "bad", "good")
        )
        assert report.status is DifferentialStatus.EXECUTION_FAILURE
        assert bad.execute_calls == good.execute_calls == 0

    class ChangedCapability(Adapter):
        def __init__(self) -> None:
            super().__init__("changed")
            self.calls = 0

        def capabilities(self) -> BackendCapabilities:
            self.calls += 1
            if self.calls == 1:
                return super().capabilities()
            return BackendCapabilities.create(
                name="changed",
                version="1",
                specifications=(),
                limits=(("inputs", 1),),
            )

    changed, good = ChangedCapability(), Adapter("good")
    report = compare_enabled_backends(
        request, (changed, good), contract=_contract(request, "changed", "good")
    )
    assert report.status is DifferentialStatus.EXECUTION_FAILURE
    assert changed.execute_calls == good.execute_calls == 0

    class Interrupted(Adapter):
        def supports(self, value: BackendRequest) -> object:
            del value
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        compare_enabled_backends(
            request,
            (Interrupted("interrupt"), Adapter("good")),
            contract=_contract(request, "interrupt", "good"),
        )


def test_execute_result_forgery_is_closed_and_runs_each_eligible_once() -> None:
    request = _request()

    def result_for(
        adapter: Adapter,
        value: BackendRequest,
        options: BackendOptions,
    ) -> BackendResult:
        support = negotiate(adapter.capabilities(), value)
        assert type(support) is BackendSupported
        return BackendResult.create(
            request=value,
            provenance=support.provenance,
            options=options,
            value_hash=_hash("value"),
        )

    class ForgedResult(Adapter):
        def __init__(self, name: str, mode: str) -> None:
            super().__init__(name)
            self.mode = mode

        def execute(self, value: BackendRequest, *, options: BackendOptions) -> object:
            self.execute_calls += 1
            if self.mode == "raw":
                return object()
            if self.mode == "defined_object":
                return Defined(object())
            result = result_for(self, value, options)
            if self.mode == "cross_request":
                other = BackendRequest.create(
                    operation="hash_identity",
                    domain="semantic_hash",
                    algorithm="identity",
                    input_hashes=(_hash("other-result"),),
                    limits=(("inputs", 1),),
                )
                result = result_for(self, other, options)
            elif self.mode == "wrong_provenance":
                other_adapter = Adapter("other")
                alternate = negotiate(other_adapter.capabilities(), value)
                assert type(alternate) is BackendSupported
                result = BackendResult.create(
                    request=value,
                    provenance=alternate.provenance,
                    options=options,
                    value_hash=_hash("value"),
                )
            elif self.mode == "wrong_options":
                object.__setattr__(result, "options_hash", _hash("wrong-options"))
                object.__setattr__(
                    result, "semantic_hash", base._digest(result._record())
                )
            elif self.mode == "request_hash":
                object.__setattr__(result, "request_hash", _hash("wrong-request"))
                object.__setattr__(
                    result, "semantic_hash", base._digest(result._record())
                )
            elif self.mode == "provenance":
                other_adapter = Adapter("other")
                alternate = negotiate(other_adapter.capabilities(), value)
                assert type(alternate) is BackendSupported
                object.__setattr__(result, "provenance", alternate.provenance)
                object.__setattr__(
                    result, "semantic_hash", base._digest(result._record())
                )
            else:
                field = self.mode
                object.__setattr__(result, field, _hash("corrupt"))
            return Defined(result)

    modes = (
        "raw",
        "defined_object",
        "cross_request",
        "wrong_provenance",
        "wrong_options",
        "request_hash",
        "provenance",
        "options_hash",
        "value_hash",
        "semantic_hash",
    )
    for mode in modes:
        bad, good = ForgedResult("bad", mode), Adapter("good")
        report = compare_enabled_backends(
            request, (bad, good), contract=_contract(request, "bad", "good")
        )
        assert report.status is DifferentialStatus.EXECUTION_FAILURE
        assert bad.execute_calls == good.execute_calls == 1

    class Exit(Adapter):
        def execute(self, value: BackendRequest, *, options: BackendOptions) -> object:
            del value, options
            raise SystemExit(17)

    with pytest.raises(SystemExit, match="17"):
        compare_enabled_backends(
            request,
            (Exit("exit"), Adapter("good")),
            contract=_contract(request, "exit", "good"),
        )


def test_input_contract_and_common_domain_boundaries_are_closed() -> None:
    request = _request()

    class HostileIterator:
        def __iter__(self) -> HostileIterator:
            return self

        def __next__(self) -> object:
            raise RuntimeError("native iterator detail")

    class InterruptedIterator:
        def __iter__(self) -> InterruptedIterator:
            return self

        def __next__(self) -> object:
            raise KeyboardInterrupt

    for adapters in ("bad", object(), HostileIterator()):
        with pytest.raises(DifferentialError):
            compare_enabled_backends(request, adapters, contract=_contract(request))
    with pytest.raises(KeyboardInterrupt):
        compare_enabled_backends(
            request, InterruptedIterator(), contract=_contract(request)
        )
    with pytest.raises(DifferentialError):
        compare_enabled_backends(object(), (), contract=object())

    two_inputs = BackendRequest.create(
        operation="hash_identity",
        domain="semantic_hash",
        algorithm="identity",
        input_hashes=(_hash("first"), _hash("second")),
        limits=(("inputs", 1),),
    )
    with pytest.raises(DifferentialError, match="input hashes"):
        compare_enabled_backends(
            two_inputs,
            (Adapter("left"), Adapter("right")),
            contract=_contract(two_inputs, "left", "right"),
        )

    def corrupt(field: str, replacement: object) -> None:
        contract = _contract(request, "left", "right")
        object.__setattr__(contract, field, replacement)
        with pytest.raises(DifferentialError):
            compare_enabled_backends(
                request,
                (Adapter("left"), Adapter("right")),
                contract=contract,
            )

    for field, replacement in (
        ("backend_request", "wrong"),
        ("backend_name", "wrong"),
        ("backend_version", "wrong"),
        ("convention_manifests", (_hash("wrong"),)),
        ("algorithms", (("differential", "wrong"),)),
        ("semantic_hash", _hash("wrong")),
    ):
        corrupt(field, replacement)

    class MissingCommon(Adapter):
        def capabilities(self) -> BackendCapabilities:
            return BackendCapabilities.create(
                name="missing",
                version="1",
                specifications=(),
                limits=(("inputs", 1),),
            )

    class Inexact(Adapter):
        def capabilities(self) -> BackendCapabilities:
            return BackendCapabilities.create(
                name="inexact",
                version="1",
                specifications=(
                    CapabilitySpec.create(
                        operation="hash_identity",
                        domain="semantic_hash",
                        algorithm="identity",
                        exact=False,
                    ),
                ),
                limits=(("inputs", 1),),
            )

    for adapter in (MissingCommon("missing"), Inexact("inexact")):
        with pytest.raises(DifferentialError, match="common exact"):
            compare_enabled_backends(
                request,
                (adapter, Adapter("good")),
                contract=_contract(request, adapter.name, "good"),
            )


def test_observation_report_receipt_and_factory_boundaries_are_tamper_evident() -> None:
    request = _request()
    report = compare_enabled_backends(
        request,
        (Adapter("left", _hash("same")), Adapter("right", _hash("same"))),
        contract=_contract(request, "left", "right"),
    )
    observation = report.observations[0]
    with pytest.raises(DifferentialError):
        differential.DifferentialObservation()
    with pytest.raises(TypeError):
        type("BadObservation", (differential.DifferentialObservation,), {})
    with pytest.raises(DifferentialError):
        differential.DifferentialReport()
    with pytest.raises(TypeError):
        type("BadReport", (differential.DifferentialReport,), {})
    assert "left" not in repr(observation)
    assert "left" not in repr(report)

    mutations = (
        ("diagnostic", "bad"),
        ("payload_hash", _hash("payload")),
        ("outcome_hash", _hash("outcome")),
        ("comparison_hash", _hash("comparison")),
        ("name", "x" * 257),
        ("version", "bad\nversion"),
    )
    for field, replacement in mutations:
        fresh = compare_enabled_backends(
            request,
            (Adapter("left", _hash("same")), Adapter("right", _hash("same"))),
            contract=_contract(request, "left", "right"),
        )
        target = fresh.observations[0]
        object.__setattr__(target, field, replacement)
        object.__setattr__(
            target, "semantic_hash", differential._digest(target._record())
        )
        object.__setattr__(
            fresh, "semantic_hash", differential._digest(fresh._record())
        )
        with pytest.raises(DifferentialError):
            fresh._assert()

    class Neutral(Adapter):
        def __init__(self, name: str, outcome: object) -> None:
            super().__init__(name)
            self.outcome = outcome

        def execute(self, value: BackendRequest, *, options: BackendOptions) -> object:
            del value, options
            self.execute_calls += 1
            return self.outcome

    for outcome in (
        Undefined("missing", None),
        Indeterminate("bounded", _hash("bound")),
        Failed("ordinary error", "stage\nleak"),
        Failed("ordinary error", "x" * 97),
    ):
        result = compare_enabled_backends(
            request,
            (Neutral("left", outcome), Neutral("right", outcome)),
            contract=_contract(request, "left", "right"),
        )
        assert (
            result.status is DifferentialStatus.EXECUTION_FAILURE
            if type(outcome) is Failed
            else DifferentialStatus.EXACT_MATCH
        )

    neutral = compare_enabled_backends(
        request,
        (
            Neutral("left", Undefined("missing", None)),
            Neutral("right", Undefined("missing", None)),
        ),
        contract=_contract(request, "left", "right"),
    )
    undefined = neutral.observations[0]
    object.__setattr__(undefined, "diagnostic", None)
    object.__setattr__(
        undefined, "semantic_hash", differential._digest(undefined._record())
    )
    object.__setattr__(
        neutral, "semantic_hash", differential._digest(neutral._record())
    )
    with pytest.raises(DifferentialError):
        neutral._assert()


def test_private_boundary_normalizers_close_malformed_and_control_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    capabilities = Adapter("normalizer").capabilities()
    broken_hash = _hash("broken")
    object.__setattr__(broken_hash, "digest", "not-a-digest")
    with pytest.raises(DifferentialError):
        differential._hash(broken_hash, "hash")
    with pytest.raises(DifferentialError):
        differential._digest({"not_json": {"set"}})

    class FailingIter:
        def __iter__(self) -> FailingIter:
            raise RuntimeError("native iterator detail")

    class ControlIter:
        def __iter__(self) -> ControlIter:
            raise KeyboardInterrupt

    class StepFailure:
        def __iter__(self) -> StepFailure:
            return self

        def __next__(self) -> object:
            raise RuntimeError("native step detail")

    for value in (FailingIter(), StepFailure()):
        with pytest.raises(DifferentialError):
            differential._snapshot(value, "items")
    with pytest.raises(KeyboardInterrupt):
        differential._snapshot(ControlIter(), "items")
    with pytest.raises(DifferentialError):
        differential._request(object())
    with pytest.raises(DifferentialError):
        differential._capabilities(object())
    object.__setattr__(request, "semantic_hash", _hash("bad-request"))
    with pytest.raises(DifferentialError):
        differential._request(request)
    object.__setattr__(capabilities, "semantic_hash", _hash("bad-capabilities"))
    with pytest.raises(DifferentialError):
        differential._capabilities(capabilities)

    def interrupt_request(_: BackendRequest) -> None:
        raise KeyboardInterrupt

    def interrupt_capabilities(_: BackendCapabilities) -> None:
        raise SystemExit(12)

    with monkeypatch.context() as scoped:
        scoped.setattr(BackendRequest, "_assert", interrupt_request)
        with pytest.raises(KeyboardInterrupt):
            differential._request(_request())
    with monkeypatch.context() as scoped:
        scoped.setattr(BackendCapabilities, "_assert", interrupt_capabilities)
        with pytest.raises(SystemExit, match="12"):
            differential._capabilities(Adapter("control").capabilities())


def test_private_contract_method_outcome_and_status_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    capabilities = Adapter("left").capabilities()
    contract = _contract(request, "left")
    with pytest.raises(DifferentialError):
        differential._contract(object(), request, (capabilities,))
    assert differential._method(object(), "missing", 1, ()) is None

    class WrongRegistry:
        def from_record(self, value: object) -> object:
            del value
            return object()

    with monkeypatch.context() as scoped:
        scoped.setattr(
            differential, "contract_record_registry", lambda: WrongRegistry()
        )
        with pytest.raises(DifferentialError, match="round trip"):
            differential._contract(contract, request, (capabilities,))

    def interrupt_record(_: object) -> dict[str, object]:
        raise KeyboardInterrupt

    with monkeypatch.context() as scoped:
        scoped.setattr(differential, "calculation_contract_record", interrupt_record)
        with pytest.raises(KeyboardInterrupt):
            differential._contract(contract, request, (capabilities,))

    specification = capabilities.specifications[0]
    support = negotiate(capabilities, request)
    assert type(support) is BackendSupported
    options = BackendOptions.create(limits=request.limits)
    result = BackendResult.create(
        request=request,
        provenance=support.provenance,
        options=options,
        value_hash=_hash("value"),
    )

    def interrupt_result(_: BackendResult) -> None:
        raise KeyboardInterrupt

    with monkeypatch.context() as scoped:
        scoped.setattr(BackendResult, "_assert", interrupt_result)
        with pytest.raises(KeyboardInterrupt):
            differential._observation(
                capabilities, request, specification, "result", result
            )

    bad_undefined = Undefined("valid", None)
    object.__setattr__(bad_undefined, "reason", "bad\nlabel")
    assert (
        differential._observation(
            capabilities, request, specification, "result", bad_undefined
        ).outcome
        == "invalid_outcome"
    )

    base = compare_enabled_backends(
        request,
        (Adapter("left"), Adapter("right")),
        contract=_contract(request, "left", "right"),
    )
    first, second = base.observations
    object.__setattr__(first, "outcome", "unsupported")
    object.__setattr__(second, "outcome", "defined")
    with pytest.raises(DifferentialError, match="availability"):
        differential._derive_status((first, second))
    object.__setattr__(first, "outcome", "unknown")
    with pytest.raises(DifferentialError, match="unsupported outcome"):
        differential._derive_status((first, second))


def test_execution_receipt_report_and_runner_control_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()

    class Raises(Adapter):
        def execute(self, value: BackendRequest, *, options: BackendOptions) -> object:
            del value, options
            self.execute_calls += 1
            raise RuntimeError("native execute detail")

    report = compare_enabled_backends(
        request,
        (Raises("raises"), Adapter("good")),
        contract=_contract(request, "raises", "good"),
    )
    assert report.status is DifferentialStatus.EXECUTION_FAILURE
    with pytest.raises(DifferentialError):
        differential._validate_receipt_mapping(
            receipt=object(),
            request_hash=request.semantic_hash,
            contract_hash=_contract(request, "raises", "good").semantic_hash,
            observations=report.observations,
            status=report.status,
        )

    def interrupt_receipt(_: object) -> bytes:
        raise KeyboardInterrupt

    with monkeypatch.context() as scoped:
        scoped.setattr(ResultReceipt, "canonical_bytes", interrupt_receipt)
        with pytest.raises(KeyboardInterrupt):
            differential._validate_receipt_mapping(
                receipt=report.receipt,
                request_hash=request.semantic_hash,
                contract_hash=report.contract_hash,
                observations=report.observations,
                status=report.status,
            )

    with pytest.raises(DifferentialError):
        differential.DifferentialReport._create(
            request=request,
            contract=_contract(request, "raises", "good"),
            common_exact_domain=("hash_identity", "semantic_hash", "identity"),
            observations=report.observations,
            status=report.status,
            receipt=cast(ResultReceipt, object()),
        )
    object.__setattr__(report, "receipt", object())
    with pytest.raises(DifferentialError):
        report._assert()

    def interrupt_runner(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise SystemExit(23)

    with monkeypatch.context() as scoped:
        scoped.setattr(differential, "run_calculation", interrupt_runner)
        with pytest.raises(SystemExit, match="23"):
            differential._receipt(
                _contract(request, "raises", "good"),
                DifferentialStatus.EXECUTION_FAILURE,
                (
                    differential._observation(
                        Adapter("left").capabilities(),
                        request,
                        Adapter("left").capabilities().specifications[0],
                        "execution_failed",
                    ),
                    differential._observation(
                        Adapter("right").capabilities(),
                        request,
                        Adapter("right").capabilities().specifications[0],
                        "execution_failed",
                    ),
                ),
            )

    inexact = BackendRequest.create(
        operation="hash_identity",
        domain="semantic_hash",
        algorithm="identity",
        input_hashes=(_hash("input"),),
        limits=(("inputs", 1),),
        requires_exact=False,
    )
    with pytest.raises(DifferentialError, match="requires exactness"):
        compare_enabled_backends(
            inexact,
            (Adapter("left"), Adapter("right")),
            contract=_contract(inexact, "left", "right"),
        )


def test_contract_metadata_and_missing_method_are_closed() -> None:
    request = _request()
    capabilities = Adapter("left").capabilities()
    wrong_runner = CalculationContract.create(
        contract_name="exact-differential",
        project_id="test.backends",
        question="compare exact neutral backend results",
        acceptable_outcomes=(
            "constructed",
            "counterexample",
            "not_applicable",
            "inconclusive",
        ),
        input_hashes=(("input0", request.input_hashes[0]),),
        source_anchors=(),
        convention_manifests=request.convention_hashes,
        algorithms=(("backend.left", "1"), ("differential", "v00_081")),
        backend_request=str(request.semantic_hash),
        backend_name="wrong-runner",
        backend_version="v00_081",
        assumptions=(),
        theorem_hypotheses=(),
        bounds=(("cases", 2),),
        required_cross_checks=(),
        expected_artifacts=(),
        acceptance_predicates=("capture",),
    )
    with pytest.raises(DifferentialError, match="differential metadata"):
        differential._contract(wrong_runner, request, (capabilities,))

    assert differential._method(object(), "missing", 1, ()) is None


def test_execution_failure_observation_sets_remain_content_addressed() -> None:
    request = _request()

    class Fails(Adapter):
        def __init__(self, name: str, outcome: object) -> None:
            super().__init__(name)
            self.outcome = outcome

        def execute(self, value: BackendRequest, *, options: BackendOptions) -> object:
            del value, options
            self.execute_calls += 1
            return self.outcome

    failed = compare_enabled_backends(
        request,
        (
            Fails("left", Failed("ordinary failure", "left-stage")),
            Fails("right", Failed("ordinary failure", "right-stage")),
        ),
        contract=_contract(request, "left", "right"),
    )
    invalid = compare_enabled_backends(
        request,
        (Fails("left", object()), Fails("right", object())),
        contract=_contract(request, "left", "right"),
    )
    assert failed.status is invalid.status is DifferentialStatus.EXECUTION_FAILURE
    assert failed.receipt.artifacts == invalid.receipt.artifacts == ()
    assert failed.receipt.semantic_hash == invalid.receipt.semantic_hash
    assert failed.semantic_hash != invalid.semantic_hash


def test_failed_outcomes_bind_opaque_payloads_without_exposing_diagnostics() -> None:
    request = _request()

    class Fails(Adapter):
        def __init__(self, name: str, outcome: Failed) -> None:
            super().__init__(name)
            self.outcome = outcome

        def execute(self, value: BackendRequest, *, options: BackendOptions) -> object:
            del value, options
            self.execute_calls += 1
            return self.outcome

    def compare(outcome: Failed) -> differential.DifferentialReport:
        return compare_enabled_backends(
            request,
            (Fails("left", outcome), Fails("right", outcome)),
            contract=_contract(request, "left", "right"),
        )

    without_stage = compare(Failed("ordinary failure"))
    same_payload = compare(Failed("ordinary failure"))
    distinct_payload = compare(Failed("other failure", "stage"))
    assert without_stage.status is DifferentialStatus.EXECUTION_FAILURE
    assert without_stage.semantic_hash == same_payload.semantic_hash
    assert without_stage.semantic_hash != distinct_payload.semantic_hash
    assert all(item.outcome == "failed" for item in without_stage.observations)
    assert all(
        item.diagnostic == "backend_failed" for item in without_stage.observations
    )
    assert all(item.payload_hash is not None for item in without_stage.observations)
    assert "ordinary failure" not in repr(without_stage.observations[0])

    for error, stage in (
        ("x" * 97, None),
        ("bad\nerror", None),
        ("error", "x" * 97),
        ("error", "bad\nstage"),
    ):
        hostile = Failed("valid")
        object.__setattr__(hostile, "error", error)
        object.__setattr__(hostile, "stage", stage)
        report = compare(hostile)
        assert report.status is DifferentialStatus.EXECUTION_FAILURE
        assert {item.outcome for item in report.observations} == {"invalid_outcome"}


def test_failure_report_hash_ignores_receipt_clock_boundaries_and_binds_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()

    class Fails(Adapter):
        def execute(self, value: BackendRequest, *, options: BackendOptions) -> object:
            del value, options
            self.execute_calls += 1
            return Failed("ordinary failure")

    origin = datetime(2026, 1, 1, tzinfo=UTC)
    calls = 0

    def clock() -> str:
        nonlocal calls
        value = origin + timedelta(seconds=calls // 2)
        calls += 1
        return value.isoformat().replace("+00:00", "Z")

    monkeypatch.setattr(evidence_run, "_utc_now", clock)

    reports = tuple(
        compare_enabled_backends(
            request,
            (Fails("left"), Fails("right")),
            contract=_contract(request, "left", "right"),
        )
        for _ in range(128)
    )
    assert calls == 256
    assert reports[0].receipt.semantic_hash != reports[-1].receipt.semantic_hash
    assert {report.semantic_hash for report in reports} == {reports[0].semantic_hash}
    assert all(
        report.receipt_hash == report.receipt.semantic_hash for report in reports
    )
    for report in reports:
        report._assert()

    operational = reports[-1]
    original_hash = operational.semantic_hash
    original_receipt_hash = operational.receipt.semantic_hash
    object.__setattr__(
        operational.receipt,
        "environment",
        evidence_run.ExecutionEnvironment.create(platform="alternate"),
    )
    object.__setattr__(
        operational.receipt,
        "semantic_hash",
        evidence_run._record_hash(operational.receipt),
    )
    object.__setattr__(operational, "receipt_hash", operational.receipt.semantic_hash)
    object.__setattr__(
        operational, "semantic_hash", differential._digest(operational._record())
    )
    assert operational.receipt.semantic_hash != original_receipt_hash
    assert operational.semantic_hash == original_hash
    operational._assert()


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("contract_id", "contract:sha256:" + "0" * 64),
        ("input_hashes", (("input0", _hash("forged-input")),)),
        ("source_anchors", (_hash("forged-anchor"),)),
        ("convention_manifests", (_hash("forged-convention"),)),
        (
            "algorithms",
            (
                ("backend.left", "forged"),
                ("backend.right", "1"),
                ("differential", "v00_081"),
            ),
        ),
        ("backend_request", "forged-request"),
        ("backend_name", "forged.backend"),
        ("backend_version", "forged"),
        ("bounds", (("cases", 3),)),
        ("warnings", ("forged",)),
    ),
)
def test_receipt_projection_binds_all_nonoperational_receipt_fields(
    field: str,
    replacement: object,
) -> None:
    request = _request()

    class Fails(Adapter):
        def execute(self, value: BackendRequest, *, options: BackendOptions) -> object:
            del value, options
            self.execute_calls += 1
            return Failed("ordinary failure")

    report = compare_enabled_backends(
        request,
        (Fails("left"), Fails("right")),
        contract=_contract(request, "left", "right"),
    )
    original_hash = report.semantic_hash
    object.__setattr__(report.receipt, field, replacement)
    with pytest.raises(DifferentialError):
        report._assert()
    object.__setattr__(
        report.receipt,
        "semantic_hash",
        evidence_run._record_hash(report.receipt),
    )
    object.__setattr__(report, "receipt_hash", report.receipt.semantic_hash)
    with pytest.raises(DifferentialError):
        report._assert()
    object.__setattr__(report, "semantic_hash", differential._digest(report._record()))
    assert report.semantic_hash != original_hash


def test_adapter_owned_support_cannot_be_mutated_to_forge_execute_provenance() -> None:
    request = _request()

    class MutatingSupport(Adapter):
        def __init__(self) -> None:
            super().__init__("left")
            self.saved: BackendSupported | None = None

        def supports(self, value: BackendRequest) -> object:
            actual = negotiate(self.capabilities(), value)
            assert type(actual) is BackendSupported
            self.saved = actual
            return actual

        def execute(self, value: BackendRequest, *, options: BackendOptions) -> object:
            self.execute_calls += 1
            assert self.saved is not None
            forged_capabilities = BackendCapabilities.create(
                name="forged",
                version="1",
                specifications=self.capabilities().specifications,
                limits=(("inputs", 1),),
            )
            forged = negotiate(forged_capabilities, value)
            assert type(forged) is BackendSupported
            object.__setattr__(self.saved, "provenance", forged.provenance)
            return Defined(
                BackendResult.create(
                    request=value,
                    provenance=self.saved.provenance,
                    options=options,
                    value_hash=_hash("same"),
                )
            )

    left, right = MutatingSupport(), Adapter("right", _hash("same"))
    report = compare_enabled_backends(
        request, (left, right), contract=_contract(request, "left", "right")
    )
    assert report.status is DifferentialStatus.EXECUTION_FAILURE
    assert left.execute_calls == right.execute_calls == 1
    assert report.observations[0].name == "left"
    assert report.observations[0].outcome == "invalid_outcome"


def test_adapter_owned_unsupported_reason_cannot_change_unavailability_receipt() -> (
    None
):
    request = _request()

    class SavedUnsupported(Adapter):
        def __init__(self) -> None:
            super().__init__("left")
            self.saved: object | None = None

        def capabilities(self) -> BackendCapabilities:
            return BackendCapabilities.create(
                name="left",
                version="1",
                specifications=(
                    CapabilitySpec.create(
                        operation="hash_identity",
                        domain="semantic_hash",
                        algorithm="identity",
                        exact=True,
                        precondition_hashes=(_hash("required"),),
                    ),
                ),
                limits=(("inputs", 1),),
            )

        def supports(self, value: BackendRequest) -> object:
            self.saved = negotiate(self.capabilities(), value)
            return self.saved

    left = SavedUnsupported()

    class MutatesSaved(Adapter):
        def supports(self, value: BackendRequest) -> object:
            assert left.saved is not None
            object.__setattr__(left.saved, "reason", UnsupportedReason.MISSING_DOMAIN)
            return super().supports(value)

    report = compare_enabled_backends(
        request,
        (left, MutatesSaved("right")),
        contract=_contract(request, "left", "right"),
    )
    assert report.status is DifferentialStatus.UNAVAILABLE
    assert report.observations[0].name == "left"
    assert report.observations[0].outcome == "unsupported"
    assert report.observations[0].diagnostic == "precondition"


def test_remaining_control_and_corrupt_result_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()

    class FakeType:
        def __call__(self, value: object) -> type[object]:
            return type(value)

        def __getattribute__(self, name: str) -> object:
            if name == "__getattribute__":

                def interrupt(value: object, attribute: str) -> object:
                    del value, attribute
                    raise KeyboardInterrupt

                return interrupt
            return object.__getattribute__(self, name)

    with monkeypatch.context() as scoped:
        scoped.setattr(differential, "type", FakeType(), raising=False)
        with pytest.raises(KeyboardInterrupt):
            differential._method(Adapter("left"), "capabilities", 1, ())

    class BrokenType(FakeType):
        def __getattribute__(self, name: str) -> object:
            if name == "__getattribute__":

                def fail(value: object, attribute: str) -> object:
                    del value, attribute
                    raise RuntimeError("private")

                return fail
            return object.__getattribute__(self, name)

    with monkeypatch.context() as scoped:
        scoped.setattr(differential, "type", BrokenType(), raising=False)
        with pytest.raises(DifferentialError, match="unavailable"):
            differential._method(Adapter("left"), "capabilities", 1, ())

    base_report = compare_enabled_backends(
        request,
        (Adapter("left", _hash("same")), Adapter("right", _hash("same"))),
        contract=_contract(request, "left", "right"),
    )
    observation = base_report.observations[0]

    def interrupt_record(self: object) -> object:
        del self
        raise KeyboardInterrupt

    with monkeypatch.context() as scoped:
        scoped.setattr(type(observation), "_record", interrupt_record)
        with pytest.raises(KeyboardInterrupt):
            observation._assert()

    with monkeypatch.context() as scoped:
        scoped.setattr(
            differential,
            "result_receipt_record",
            lambda _receipt: (_ for _ in ()).throw(SystemExit(9)),
        )
        with pytest.raises(SystemExit, match="9"):
            differential._receipt_projection(base_report.receipt)
    with monkeypatch.context() as scoped:
        scoped.setattr(differential, "result_receipt_record", lambda _receipt: ())
        with pytest.raises(DifferentialError, match="content address"):
            differential._receipt_projection(base_report.receipt)


def test_remaining_receipt_and_report_creation_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    contract = _contract(request, "left", "right")

    def fresh() -> differential.DifferentialReport:
        return compare_enabled_backends(
            request,
            (Adapter("left", _hash("same")), Adapter("right", _hash("same"))),
            contract=contract,
        )

    report = fresh()
    failed = compare_enabled_backends(
        request,
        (
            type(
                "FailsLeft",
                (Adapter,),
                {"execute": lambda self, value, *, options: Failed("failed")},
            )("left"),
            type(
                "FailsRight",
                (Adapter,),
                {"execute": lambda self, value, *, options: Failed("failed")},
            )("right"),
        ),
        contract=contract,
    )
    object.__setattr__(failed.receipt, "execution_status", "completed")
    object.__setattr__(
        failed.receipt, "semantic_hash", evidence_run._record_hash(failed.receipt)
    )
    with pytest.raises(DifferentialError, match="failure mapping"):
        differential._validate_receipt_mapping(
            receipt=failed.receipt,
            request_hash=request.semantic_hash,
            contract_hash=contract.semantic_hash,
            observations=failed.observations,
            status=DifferentialStatus.EXECUTION_FAILURE,
        )

    with monkeypatch.context() as scoped:
        scoped.setattr(
            ResultReceipt,
            "canonical_bytes",
            lambda self: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        with pytest.raises(KeyboardInterrupt):
            differential.DifferentialReport._create(
                request=request,
                contract=contract,
                common_exact_domain=report.common_exact_domain,
                observations=report.observations,
                status=report.status,
                receipt=report.receipt,
            )
    with monkeypatch.context() as scoped:
        scoped.setattr(
            ResultReceipt,
            "canonical_bytes",
            lambda self: (_ for _ in ()).throw(RuntimeError("private")),
        )
        with pytest.raises(DifferentialError, match="content address"):
            differential.DifferentialReport._create(
                request=request,
                contract=contract,
                common_exact_domain=report.common_exact_domain,
                observations=report.observations,
                status=report.status,
                receipt=report.receipt,
            )

    wrong_contract_receipt = fresh().receipt
    object.__setattr__(wrong_contract_receipt, "contract_hash", _hash("wrong"))
    object.__setattr__(
        wrong_contract_receipt,
        "semantic_hash",
        evidence_run._record_hash(wrong_contract_receipt),
    )
    with pytest.raises(DifferentialError, match="calculation contract"):
        differential.DifferentialReport._create(
            request=request,
            contract=contract,
            common_exact_domain=report.common_exact_domain,
            observations=report.observations,
            status=report.status,
            receipt=wrong_contract_receipt,
        )

    with pytest.raises(DifferentialError, match="duplicate"):
        differential.DifferentialReport._create(
            request=request,
            contract=contract,
            common_exact_domain=report.common_exact_domain,
            observations=(report.observations[0], report.observations[0]),
            status=report.status,
            receipt=report.receipt,
        )
    with pytest.raises(DifferentialError, match="does not bind"):
        differential.DifferentialReport._create(
            request=request,
            contract=contract,
            common_exact_domain=("wrong", "domain", "algorithm"),
            observations=report.observations,
            status=report.status,
            receipt=report.receipt,
        )
    with pytest.raises(DifferentialError, match="status"):
        differential.DifferentialReport._create(
            request=request,
            contract=contract,
            common_exact_domain=report.common_exact_domain,
            observations=report.observations,
            status=DifferentialStatus.DISAGREEMENT,
            receipt=report.receipt,
        )

    artifacts_receipt = fresh().receipt
    object.__setattr__(artifacts_receipt, "artifacts", ())
    object.__setattr__(
        artifacts_receipt,
        "semantic_hash",
        evidence_run._record_hash(artifacts_receipt),
    )
    with pytest.raises(DifferentialError, match="artifacts"):
        differential.DifferentialReport._create(
            request=request,
            contract=contract,
            common_exact_domain=report.common_exact_domain,
            observations=report.observations,
            status=report.status,
            receipt=artifacts_receipt,
        )

    with monkeypatch.context() as scoped:
        scoped.setattr(
            differential,
            "_validate_receipt_mapping",
            lambda **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        with pytest.raises(KeyboardInterrupt):
            report._assert()
    with monkeypatch.context() as scoped:
        scoped.setattr(
            differential,
            "_validate_receipt_mapping",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("private")),
        )
        with pytest.raises(DifferentialError, match="content address"):
            report._assert()


def test_corrupt_backend_result_and_support_controls_are_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()

    class ExplodingComparison:
        def __eq__(self, other: object) -> bool:
            del other
            raise RuntimeError("private")

        def __ne__(self, other: object) -> bool:
            del other
            raise RuntimeError("private")

    class CorruptResult(Adapter):
        def execute(self, value: BackendRequest, *, options: BackendOptions) -> object:
            supported = negotiate(self.capabilities(), value)
            assert type(supported) is BackendSupported
            result = BackendResult.create(
                request=value,
                provenance=supported.provenance,
                options=options,
                value_hash=_hash("same"),
            )
            object.__setattr__(result, "request_hash", ExplodingComparison())
            return Defined(result)

    report = compare_enabled_backends(
        request,
        (CorruptResult("left"), Adapter("right", _hash("same"))),
        contract=_contract(request, "left", "right"),
    )
    assert report.status is DifferentialStatus.EXECUTION_FAILURE

    corrupt = CorruptResult("left")
    capabilities = corrupt.capabilities()
    specification = capabilities.specifications[0]
    support = negotiate(capabilities, request)
    assert type(support) is BackendSupported
    with monkeypatch.context() as scoped:
        scoped.setattr(BackendResult, "_assert", lambda self: None)
        observation = differential._execute(
            corrupt, capabilities, request, specification, support
        )
    assert observation.outcome == "invalid_outcome"

    class ControlComparison:
        def __ne__(self, other: object) -> bool:
            del other
            raise KeyboardInterrupt

    class ControlResult(CorruptResult):
        def execute(self, value: BackendRequest, *, options: BackendOptions) -> object:
            supported = negotiate(self.capabilities(), value)
            assert type(supported) is BackendSupported
            result = BackendResult.create(
                request=value,
                provenance=supported.provenance,
                options=options,
                value_hash=_hash("same"),
            )
            object.__setattr__(result, "request_hash", ControlComparison())
            return Defined(result)

    control = ControlResult("left")
    control_capabilities = control.capabilities()
    control_support = negotiate(control_capabilities, request)
    assert type(control_support) is BackendSupported
    with monkeypatch.context() as scoped:
        scoped.setattr(BackendResult, "_assert", lambda self: None)
        with pytest.raises(KeyboardInterrupt):
            differential._execute(
                control,
                control_capabilities,
                request,
                control_capabilities.specifications[0],
                control_support,
            )

    with monkeypatch.context() as scoped:
        scoped.setattr(
            BackendSupported,
            "_assert",
            lambda self: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        with pytest.raises(KeyboardInterrupt):
            compare_enabled_backends(
                request,
                (Adapter("left"), Adapter("right")),
                contract=_contract(request, "left", "right"),
            )
