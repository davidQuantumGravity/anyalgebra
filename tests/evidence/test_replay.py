"""V00-077: replay reports are durable, read-only evidence comparisons."""

from __future__ import annotations

import pytest

from anyalgebra.core.parents import SemanticHash
from anyalgebra.evidence.contracts import CalculationContract
from anyalgebra.evidence.models import ConventionManifest, SourceAnchor
from anyalgebra.evidence.replay import (
    ReplayError,
    ReplayReport,
    ReplayResolver,
    replay_report_record,
    replay_report_registry,
    verify_replay,
)
from anyalgebra.evidence.run import (
    CalculationResult,
    ExecutionEnvironment,
    ResultReceipt,
    run_calculation,
)
import anyalgebra.evidence.replay as replay_module
import anyalgebra.evidence.run as run_module
from anyalgebra.persistence.registry import SchemaError


@pytest.fixture(autouse=True)
def _clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_module, "_utc_now", lambda: "2026-07-24T12:00:00Z")


def _hash(character: str = "a") -> SemanticHash:
    return SemanticHash("sha256", character * 64)


def _contract(
    *,
    outcomes: tuple[str, ...] = ("constructed",),
    input_character: str = "a",
    bounds: tuple[tuple[str, int], ...] = (("cases", 8),),
    sources: tuple[SourceAnchor, ...] = (),
    manifests: tuple[ConventionManifest, ...] = (),
) -> CalculationContract:
    return CalculationContract.create(
        contract_name="replay-test",
        project_id="test.project",
        question="test question",
        acceptable_outcomes=outcomes,
        input_hashes=(("input", _hash(input_character)),),
        source_anchors=sources,
        convention_manifests=manifests,
        algorithms=(("reference", "1"),),
        backend_request="exact",
        backend_name="reference",
        backend_version="1",
        assumptions=(),
        theorem_hypotheses=(),
        bounds=bounds,
        required_cross_checks=(),
        expected_artifacts=(),
        acceptance_predicates=("stored",),
    )


def _source(character: str = "a") -> SourceAnchor:
    return SourceAnchor.create(
        location="doi:10.1000/example",
        content_hash=_hash(character),
        edition_or_commit="rev-7",
        locator="p. 13",
        excerpt_hash=_hash("b"),
    )


def _manifest(character: str = "a") -> ConventionManifest:
    source = _source(character)
    return ConventionManifest.create(
        manifest_id=f"test.{character}",
        basis_order=("1",),
        coefficient_domain="QQ",
        multiplication_fixture=_hash("c"),
        sources=(source,),
        status="accepted",
        signs=(),
        involutions=(),
        normalization=(),
        indexing="zero-based",
        parenthesization="explicit",
        conversions=(),
    )


def _receipt(
    *,
    contract: CalculationContract | None = None,
    outcome: str = "constructed",
    summary: str = "constructed fixture",
    artifacts: tuple[SemanticHash, ...] = (),
    witnesses: tuple[SemanticHash, ...] = (),
    dependencies: tuple[SemanticHash, ...] = (),
    environment: ExecutionEnvironment | None = None,
) -> ResultReceipt:
    selected = _contract(outcomes=(outcome,)) if contract is None else contract
    return run_calculation(
        selected,
        lambda _: CalculationResult.create(
            outcome=outcome,
            summary=summary,
            artifacts=artifacts,
            witnesses=witnesses,
            dependency_receipts=dependencies,
            remaining_branches=("unexplored",) if outcome == "inconclusive" else (),
        ),
        environment=environment,
    )


def test_exact_replay_reports_a_durable_exact_match() -> None:
    original = _receipt()
    rerun = _receipt(environment=ExecutionEnvironment.create(platform="rerun"))
    resolver = ReplayResolver.create(
        receipts=(original, rerun),
        reruns=((original.semantic_hash, rerun.semantic_hash),),
    )
    report = verify_replay(original, resolver=resolver)
    assert report.status == "exact_match"
    assert report.original_receipt_hash == original.semantic_hash
    assert report.rerun_receipt_hash == rerun.semantic_hash
    assert report.stale_edges == ()


@pytest.mark.parametrize(
    "outcome", ("counterexample", "inconclusive", "reproduction_only")
)
def test_matching_negative_and_inconclusive_outcomes_are_exact(outcome: str) -> None:
    witnesses = (_hash("b"),) if outcome == "counterexample" else ()
    contract = _contract(outcomes=(outcome,))
    original = _receipt(contract=contract, outcome=outcome, witnesses=witnesses)
    rerun = _receipt(
        contract=contract,
        outcome=outcome,
        witnesses=witnesses,
        environment=ExecutionEnvironment.create(platform="elsewhere"),
    )
    resolver = ReplayResolver.create(
        receipts=(original, rerun),
        reruns=((original.semantic_hash, rerun.semantic_hash),),
    )
    report = verify_replay(original, resolver=resolver)
    assert report.status == "exact_match"
    assert report.rerun_receipt_hash == rerun.semantic_hash


def test_exact_match_ignores_timestamp_environment_and_display_metadata() -> None:
    contract = _contract()
    original = _receipt(
        contract=contract,
        summary="first display",
        environment=ExecutionEnvironment.create(platform="one"),
    )
    rerun = _receipt(
        contract=contract,
        summary="second display",
        environment=ExecutionEnvironment.create(platform="two"),
    )
    report = verify_replay(
        original,
        resolver=ReplayResolver.create(
            receipts=(original, rerun),
            reruns=((original.semantic_hash, rerun.semantic_hash),),
        ),
    )
    assert report.status == "exact_match"


def test_result_difference_is_a_mismatch_only_when_contract_is_fresh() -> None:
    contract = _contract()
    original, rerun = (
        _receipt(contract=contract, artifacts=(_hash("b"),)),
        _receipt(contract=contract, artifacts=(_hash("c"),)),
    )
    report = verify_replay(
        original,
        resolver=ReplayResolver.create(
            receipts=(original, rerun),
            reruns=((original.semantic_hash, rerun.semantic_hash),),
        ),
    )
    assert report.status == "mismatch"
    assert report.mismatch_fields == ("primary_result_hash",)
    assert report.matched_fields == (
        "contract",
        "direct_dependencies",
        "mathematical_outcome",
    )


def test_direct_changes_are_stale_before_result_comparison() -> None:
    original = _receipt(contract=_contract(input_character="a"))
    rerun = _receipt(contract=_contract(input_character="b"), artifacts=(_hash("b"),))
    report = verify_replay(
        original,
        resolver=ReplayResolver.create(
            receipts=(original, rerun),
            reruns=((original.semantic_hash, rerun.semantic_hash),),
        ),
    )
    assert report.status == "stale"
    assert report.stale_edges == ("contract.inputs",)

    failed_stale = run_calculation(
        _contract(input_character="c"),
        lambda _: (_ for _ in ()).throw(RuntimeError("secret")),
    )
    report = verify_replay(
        original,
        resolver=ReplayResolver.create(
            receipts=(original, failed_stale),
            reruns=((original.semantic_hash, failed_stale.semantic_hash),),
        ),
    )
    assert report.status == "stale"
    assert report.stale_edges == ("contract.inputs",)

    dependency_a, dependency_b = _receipt(), _receipt(artifacts=(_hash("d"),))
    original = _receipt(dependencies=(dependency_a.semantic_hash,))
    rerun = _receipt(dependencies=(dependency_b.semantic_hash,))
    report = verify_replay(
        original,
        resolver=ReplayResolver.create(
            receipts=(original, rerun, dependency_a, dependency_b),
            reruns=((original.semantic_hash, rerun.semantic_hash),),
        ),
    )
    assert report.status == "stale"
    assert len(report.stale_edges) == 2
    assert all(
        edge.startswith("dependency_receipt:sha256:") for edge in report.stale_edges
    )


def test_missing_resolutions_and_failed_reruns_are_unavailable() -> None:
    original = _receipt()
    resolver = ReplayResolver.create(receipts=(original,))
    report = verify_replay(original, resolver=resolver)
    assert report.status == "unavailable"
    assert report.unavailable_references == (original.semantic_hash,)

    rerun = _receipt(dependencies=(_hash("e"),))
    report = verify_replay(
        original,
        resolver=ReplayResolver.create(
            receipts=(original, rerun),
            reruns=((original.semantic_hash, rerun.semantic_hash),),
        ),
    )
    assert report.status == "unavailable"
    assert report.unavailable_references == (_hash("e"),)

    shared_missing = _hash("f")
    original = _receipt(dependencies=(shared_missing,))
    rerun = _receipt(
        dependencies=(shared_missing,),
        environment=ExecutionEnvironment.create(platform="rerun"),
    )
    report = verify_replay(
        original,
        resolver=ReplayResolver.create(
            receipts=(original, rerun),
            reruns=((original.semantic_hash, rerun.semantic_hash),),
        ),
    )
    assert report.status == "unavailable"
    assert report.unavailable_references == (shared_missing,)

    original = _receipt()
    failed = run_calculation(
        _contract(), lambda _: (_ for _ in ()).throw(RuntimeError("secret"))
    )
    report = verify_replay(
        original,
        resolver=ReplayResolver.create(
            receipts=(original, failed),
            reruns=((original.semantic_hash, failed.semantic_hash),),
        ),
    )
    assert report.status == "unavailable"
    assert report.unavailable_references == (failed.semantic_hash,)


def test_resolver_and_report_serialization_are_closed_and_canonical() -> None:
    original = _receipt()
    rerun = _receipt(environment=ExecutionEnvironment.create(platform="rerun"))
    report = verify_replay(
        original,
        resolver=ReplayResolver.create(
            receipts=(original, rerun),
            reruns=((original.semantic_hash, rerun.semantic_hash),),
        ),
    )
    record = replay_report_record(report)
    assert replay_report_registry().from_record(record) == report
    assert report.to_record() == record
    assert report.canonical_bytes() == replay_module.replay_report_canonical_bytes(
        report
    )
    assert (
        report.canonical_bytes()
        == b'{"matchedFields":["contract","direct_dependencies","mathematical_outcome","primary_result_hash"],"mismatchFields":[],"originalReceiptHash":{"algorithm":"sha256","digest":"71d71c712f2cf6086a797f4125e3598356fc40166d4d404c7b4129f67f766e8b"},"rerunReceiptHash":{"algorithm":"sha256","digest":"fa87d6545f252adcb953baae7f8472d85c3b808992d3ab2fd64a8cba97e4397c"},"schemaType":"anyalgebra.evidence.replay_report","schemaVersion":1,"staleEdges":[],"status":"exact_match","unavailableReferences":[]}'  # noqa: E501
    )
    assert (
        report.semantic_hash.digest
        == "d56a44bc5591971e5825800be364efc72449d1de42b18e035b54fd0d8f326d18"
    )
    assert report.report_id == f"replay:sha256:{report.semantic_hash.digest}"
    for malformed in (
        {**record, "extra": 1},
        {key: value for key, value in record.items() if key != "status"},
        {**record, "schemaType": "os.system"},
        {**record, "schemaVersion": 2},
        {**record, "status": "promoted"},
        {**record, "matchedFields": ["proved"]},
        {**record, "staleEdges": ["dependency_receipt:made-up"]},
        {**record, "rerunReceiptHash": record["originalReceiptHash"]},
    ):
        with pytest.raises(SchemaError):
            replay_report_registry().from_record(malformed)
    with pytest.raises(ReplayError):
        ReplayReport()
    with pytest.raises(ReplayError):
        ReplayResolver()
    with pytest.raises(TypeError):
        type("Child", (ReplayReport,), {})
    unavailable = replay_module._build(
        status="unavailable",
        original_receipt_hash=report.original_receipt_hash,
        rerun_receipt_hash=None,
        unavailable_references=(_hash("b"),),
    )
    assert unavailable.semantic_hash != report.semantic_hash


def test_boundary_validation_tampering_and_no_callable_resolver() -> None:
    original = _receipt()
    with pytest.raises(ReplayError):
        ReplayResolver.create(receipts=(original, original))
    with pytest.raises(ReplayError):
        ReplayResolver.create(
            receipts=(original,), reruns=((_hash("b"), original.semantic_hash),)
        )
    with pytest.raises(ReplayError):
        ReplayResolver.create(receipts="not-items")
    with pytest.raises(ReplayError):
        verify_replay(original, resolver=lambda _: original)
    with pytest.raises(ReplayError):
        verify_replay(
            original,
            resolver=ReplayResolver.create(receipts=(original,)),
            environment=object(),
        )
    object.__setattr__(original, "summary", "tampered")
    with pytest.raises(ReplayError, match="integrity"):
        verify_replay(original, resolver=ReplayResolver.create(receipts=()))


def test_private_build_validation_and_parser_helpers_cover_failure_routes() -> None:
    original = _receipt()
    with pytest.raises(ReplayError):
        replay_module._build(
            status="exact_match",
            original_receipt_hash=original.semantic_hash,
            rerun_receipt_hash=None,
        )
    with pytest.raises(ReplayError):
        replay_module._build(
            status="mismatch",
            original_receipt_hash=original.semantic_hash,
            rerun_receipt_hash=original.semantic_hash,
        )
    with pytest.raises(ReplayError):
        replay_module._build(
            status="unavailable",
            original_receipt_hash=original.semantic_hash,
            rerun_receipt_hash=original.semantic_hash,
        )
    with pytest.raises(ReplayError):
        replay_module._build(
            status="bad",
            original_receipt_hash=original.semantic_hash,
            rerun_receipt_hash=None,
        )
    with pytest.raises(ReplayError):
        replay_module._items("bad", "items")
    with pytest.raises(ReplayError):
        replay_module._hash("bad", "hash")
    with pytest.raises(ReplayError):
        replay_module._strings(("x", "x"), "text")
    rerun = _receipt(environment=ExecutionEnvironment.create(platform="rerun"))
    report = verify_replay(
        original,
        resolver=ReplayResolver.create(
            receipts=(original, rerun),
            reruns=((original.semantic_hash, rerun.semantic_hash),),
        ),
    )
    object.__setattr__(report, "status", "tampered")
    with pytest.raises(ReplayError, match="integrity"):
        report.to_record()


def test_predecessor_is_resolved_directly_and_a_change_is_stale() -> None:
    predecessor = _receipt(artifacts=(_hash("b"),))
    contract = _contract()
    original = run_module._build_receipt(
        contract=contract,
        environment=ExecutionEnvironment.create(),
        started_at="2026-07-24T12:00:00Z",
        finished_at="2026-07-24T12:00:00Z",
        execution_status="completed",
        outcome="constructed",
        summary="constructed fixture",
        supersedes=predecessor.semantic_hash,
    )
    rerun = _receipt(contract=contract)
    report = verify_replay(
        original,
        resolver=ReplayResolver.create(
            receipts=(original, rerun, predecessor),
            reruns=((original.semantic_hash, rerun.semantic_hash),),
        ),
    )
    assert report.status == "stale"
    assert report.stale_edges == (
        f"supersedes:sha256:{predecessor.semantic_hash.digest}",
    )
    report = verify_replay(
        original,
        resolver=ReplayResolver.create(
            receipts=(original, rerun),
            reruns=((original.semantic_hash, rerun.semantic_hash),),
        ),
    )
    assert report.status == "unavailable"
    assert report.unavailable_references == (predecessor.semantic_hash,)


def test_direct_dependency_updates_make_original_evidence_stale() -> None:
    old_dependency = _receipt(artifacts=(_hash("b"),))
    new_dependency = _receipt(artifacts=(_hash("c"),))
    original = _receipt(dependencies=(old_dependency.semantic_hash,))
    correct_rerun = _receipt(
        dependencies=(new_dependency.semantic_hash,),
        environment=ExecutionEnvironment.create(platform="correct"),
    )
    resolver = ReplayResolver.create(
        receipts=(original, correct_rerun, old_dependency, new_dependency),
        reruns=(
            (original.semantic_hash, correct_rerun.semantic_hash),
            (old_dependency.semantic_hash, new_dependency.semantic_hash),
        ),
    )
    report = verify_replay(original, resolver=resolver)
    assert report.status == "stale"
    assert (
        f"dependency_receipt:{old_dependency.semantic_hash}->{new_dependency.semantic_hash}"
        in report.stale_edges
    )

    stale_rerun = _receipt(
        dependencies=(old_dependency.semantic_hash,),
        environment=ExecutionEnvironment.create(platform="stale"),
    )
    resolver = ReplayResolver.create(
        receipts=(original, stale_rerun, old_dependency, new_dependency),
        reruns=(
            (original.semantic_hash, stale_rerun.semantic_hash),
            (old_dependency.semantic_hash, new_dependency.semantic_hash),
        ),
    )
    report = verify_replay(original, resolver=resolver)
    assert report.status == "stale"
    assert f"dependency_receipt:{new_dependency.semantic_hash}" in report.stale_edges


def test_all_direct_contract_edges_have_precise_names() -> None:
    original = _receipt(
        contract=_contract(sources=(_source("a"),), manifests=(_manifest("a"),))
    )
    altered = CalculationContract.create(
        contract_name="other",
        project_id="other.project",
        question="other question",
        acceptable_outcomes=("constructed", "inconclusive"),
        input_hashes=(("other", _hash("b")),),
        source_anchors=(_source("d"),),
        convention_manifests=(_manifest("d"),),
        algorithms=(("other", "2"),),
        backend_request="other",
        backend_name="other",
        backend_version="2",
        assumptions=(("assumption", "value"),),
        theorem_hypotheses=(("hypothesis", "value"),),
        bounds=(("other", 1),),
        required_cross_checks=("check",),
        expected_artifacts=("artifact",),
        acceptance_predicates=("other",),
    )
    rerun = _receipt(contract=altered)
    report = verify_replay(
        original,
        resolver=ReplayResolver.create(
            receipts=(original, rerun),
            reruns=((original.semantic_hash, rerun.semantic_hash),),
        ),
    )
    assert report.status == "stale"
    assert set(report.stale_edges) == {
        "contract.contract_name",
        "contract.project_id",
        "contract.question",
        "contract.acceptable_outcomes",
        "contract.inputs",
        "contract.sources",
        "contract.conventions",
        "contract.algorithms",
        "contract.backend_request",
        "contract.backend_name",
        "contract.backend_version",
        "contract.assumptions",
        "contract.theorem_hypotheses",
        "contract.bounds",
        "contract.required_cross_checks",
        "contract.expected_artifacts",
        "contract.acceptance_predicates",
    }


def test_internal_resource_limits_exact_parsers_and_safe_repr() -> None:
    original = _receipt()

    class BrokenIterable:
        def __iter__(self) -> object:
            raise RuntimeError("secret")

    class BrokenNext:
        def __iter__(self) -> BrokenNext:
            return self

        def __next__(self) -> object:
            raise RuntimeError("secret")

    class Interrupted:
        def __init__(self, error: BaseException) -> None:
            self.error = error

        def __iter__(self) -> Interrupted:
            return self

        def __next__(self) -> object:
            raise self.error

    for value in (None, "", "x" * 4097, "bad\ud800"):
        with pytest.raises(ReplayError):
            replay_module._text(value, "field")
    for bad_item in ("text", BrokenIterable(), BrokenNext(), tuple(range(257))):
        with pytest.raises(ReplayError):
            replay_module._items(bad_item, "items")
    with pytest.raises(ReplayError):
        replay_module._hashes((_hash(), _hash()), "hashes")
    assert replay_module._hashes((_hash("b"), _hash("a")), "hashes") == (
        _hash("a"),
        _hash("b"),
    )
    with pytest.raises(ReplayError):
        replay_module._pairs(((original.semantic_hash,),), "pairs")
    with pytest.raises(ReplayError):
        replay_module._pairs((object(),), "pairs")
    with pytest.raises(ReplayError):
        replay_module._pairs(
            ((original.semantic_hash, original.semantic_hash),) * 2, "pairs"
        )
    with pytest.raises(ReplayError):
        replay_module._receipt(object(), "receipt")
    with pytest.raises(ReplayError):
        replay_module._resolver_items(BrokenIterable(), "resolver")
    with pytest.raises(ReplayError):
        replay_module._resolver_items(BrokenNext(), "resolver")
    assert len(replay_module._resolver_items(tuple(range(1024)), "resolver")) == 1024
    with pytest.raises(ReplayError, match="resolver item limit"):
        replay_module._resolver_items(tuple(range(1025)), "resolver")
    assert len(replay_module._report_items(tuple(range(1024)), "report")) == 1024
    with pytest.raises(ReplayError, match="report item limit"):
        replay_module._report_items(tuple(range(1025)), "report")
    for bad_item in ("text", BrokenIterable(), BrokenNext()):
        with pytest.raises(ReplayError):
            replay_module._report_items(bad_item, "report")
    with pytest.raises(ReplayError):
        replay_module._report_hashes((_hash(), _hash()), "report")
    with pytest.raises(ReplayError):
        replay_module._report_strings(("x", "x"), "report")
    with pytest.raises(KeyboardInterrupt):
        replay_module._items(Interrupted(KeyboardInterrupt()), "items")
    with pytest.raises(SystemExit):
        replay_module._resolver_items(Interrupted(SystemExit()), "resolver")
    with pytest.raises(KeyboardInterrupt):
        replay_module._report_items(Interrupted(KeyboardInterrupt()), "report")
    rerun = _receipt(environment=ExecutionEnvironment.create(platform="rerun"))
    resolver = ReplayResolver.create(
        receipts=(original, rerun),
        reruns=((original.semantic_hash, rerun.semantic_hash),),
    )
    assert "receipt_count=2" in repr(resolver)
    report = verify_replay(original, resolver=resolver)
    assert "exact_match" in repr(report)
    assert hash(report)
    assert report == report
    assert report != object()


def test_parser_shape_hash_and_factory_paths_are_strict() -> None:
    original = _receipt()
    rerun = _receipt(environment=ExecutionEnvironment.create(platform="rerun"))
    report = verify_replay(
        original,
        resolver=ReplayResolver.create(
            receipts=(original, rerun),
            reruns=((original.semantic_hash, rerun.semantic_hash),),
        ),
    )
    record = replay_report_record(report)
    with pytest.raises(ReplayError):
        replay_module._mapping([], frozenset(), "mapping")
    with pytest.raises(ReplayError):
        replay_module._parse_hash({"algorithm": 1, "digest": "a" * 64}, "hash")
    with pytest.raises(ReplayError):
        replay_module._parse_hash({"algorithm": "sha256"}, "hash")
    with pytest.raises(ReplayError):
        replay_module._tuple([], "tuple")
    with pytest.raises(ReplayError, match="too many pages"):
        replay_module._pages(((), (), (), (), ()), "pages")
    with pytest.raises(SchemaError):
        replay_report_registry().from_record(
            {**record, "originalReceiptHash": {"algorithm": "bad", "digest": "x"}}
        )
    assert replay_module._mutable({"a": (1,)}) == {"a": [1]}
    with pytest.raises(ReplayError):
        replay_module._ensure(object())
    with pytest.raises(ReplayError):
        replay_module._record_hash(object())  # type: ignore[arg-type]
    with pytest.raises(ReplayError):
        replay_module._build(
            status="stale",
            original_receipt_hash=original.semantic_hash,
            rerun_receipt_hash=original.semantic_hash,
            stale_edges=("x",),
            unavailable_references=(_hash("b"),),
        )


def test_resolver_rejects_self_cycles_convergence_and_mutation() -> None:
    first = _receipt(environment=ExecutionEnvironment.create(platform="first"))
    second = _receipt(environment=ExecutionEnvironment.create(platform="second"))
    third = _receipt(environment=ExecutionEnvironment.create(platform="third"))
    with pytest.raises(ReplayError, match="distinct"):
        ReplayResolver.create(
            receipts=(first,), reruns=((first.semantic_hash, first.semantic_hash),)
        )
    with pytest.raises(ReplayError, match="converging"):
        ReplayResolver.create(
            receipts=(first, second, third),
            reruns=(
                (first.semantic_hash, third.semantic_hash),
                (second.semantic_hash, third.semantic_hash),
            ),
        )
    with pytest.raises(ReplayError, match="cycle"):
        ReplayResolver.create(
            receipts=(first, second),
            reruns=(
                (first.semantic_hash, second.semantic_hash),
                (second.semantic_hash, first.semantic_hash),
            ),
        )
    resolver = ReplayResolver.create(
        receipts=(first, second), reruns=((first.semantic_hash, second.semantic_hash),)
    )
    object.__setattr__(
        resolver, "reruns", ((first.semantic_hash, first.semantic_hash),)
    )
    with pytest.raises(ReplayError, match="distinct"):
        verify_replay(first, resolver=resolver)
    with pytest.raises(ReplayError):
        ReplayResolver.create.__func__(object, receipts=())  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        type("Child", (ReplayResolver,), {})


def test_report_diagnostic_vocabulary_and_status_shapes_are_closed() -> None:
    original, rerun = (
        _receipt(),
        _receipt(environment=ExecutionEnvironment.create(platform="rerun")),
    )
    with pytest.raises(ReplayError, match="unsupported"):
        replay_module._build(
            status="mismatch",
            original_receipt_hash=original.semantic_hash,
            rerun_receipt_hash=rerun.semantic_hash,
            matched_fields=("contract", "direct_dependencies", "mathematical_outcome"),
            mismatch_fields=("proved",),
        )
    with pytest.raises(ReplayError, match="unsupported"):
        replay_module._build(
            status="stale",
            original_receipt_hash=original.semantic_hash,
            rerun_receipt_hash=rerun.semantic_hash,
            stale_edges=("dependency_receipt:bad",),
        )
    with pytest.raises(ReplayError, match="mismatch requires"):
        replay_module._build(
            status="mismatch",
            original_receipt_hash=original.semantic_hash,
            rerun_receipt_hash=rerun.semantic_hash,
            matched_fields=("contract", "mathematical_outcome"),
            mismatch_fields=("primary_result_hash",),
        )
    with pytest.raises(ReplayError, match="stale requires"):
        replay_module._build(
            status="stale",
            original_receipt_hash=original.semantic_hash,
            rerun_receipt_hash=rerun.semantic_hash,
            stale_edges=("contract.inputs",),
            matched_fields=("contract",),
        )
    with pytest.raises(ReplayError, match="unavailable requires"):
        replay_module._build(
            status="unavailable",
            original_receipt_hash=original.semantic_hash,
            rerun_receipt_hash=None,
            unavailable_references=(),
        )
    with pytest.raises(ReplayError, match="distinct"):
        replay_module._build(
            status="unavailable",
            original_receipt_hash=original.semantic_hash,
            rerun_receipt_hash=original.semantic_hash,
            unavailable_references=(_hash("b"),),
        )
    assert replay_module._hash_text(str(original.semantic_hash))
    assert not replay_module._hash_text("not-a-hash")
    assert not replay_module._hash_text("wrong:" + "a" * 64)
    assert not replay_module._stale_edge("unknown")
    assert not replay_module._stale_edge(
        f"dependency_receipt:{original.semantic_hash}->{original.semantic_hash}"
    )


def test_original_not_registered_and_outcome_change_have_separate_paths() -> None:
    original = _receipt()
    other = _receipt(environment=ExecutionEnvironment.create(platform="other"))
    report = verify_replay(other, resolver=ReplayResolver.create(receipts=(original,)))
    assert report.status == "unavailable"
    assert report.rerun_receipt_hash is None

    contract = _contract(outcomes=("constructed", "inconclusive"))
    original = _receipt(contract=contract)
    rerun = _receipt(
        contract=contract,
        outcome="inconclusive",
        environment=ExecutionEnvironment.create(platform="outcome"),
    )
    report = verify_replay(
        original,
        resolver=ReplayResolver.create(
            receipts=(original, rerun),
            reruns=((original.semantic_hash, rerun.semantic_hash),),
        ),
    )
    assert report.status == "mismatch"
    assert report.mismatch_fields == ("mathematical_outcome", "primary_result_hash")


def test_receipt_parser_mismatch_and_predecessor_update_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _receipt()

    class BadRegistry:
        def from_record(self, record: object) -> object:
            del record
            return object()

    monkeypatch.setattr(replay_module, "result_receipt_registry", lambda: BadRegistry())
    with pytest.raises(ReplayError, match="integrity"):
        replay_module._receipt(original, "receipt")

    monkeypatch.undo()
    old, new = _receipt(artifacts=(_hash("d"),)), _receipt(artifacts=(_hash("e"),))
    contract = _contract()
    original = run_module._build_receipt(
        contract=contract,
        environment=ExecutionEnvironment.create(),
        started_at="2026-07-24T12:00:00Z",
        finished_at="2026-07-24T12:00:00Z",
        execution_status="completed",
        outcome="constructed",
        summary="constructed fixture",
        supersedes=old.semantic_hash,
    )
    rerun = run_module._build_receipt(
        contract=contract,
        environment=ExecutionEnvironment.create(platform="new"),
        started_at="2026-07-24T12:00:00Z",
        finished_at="2026-07-24T12:00:00Z",
        execution_status="completed",
        outcome="constructed",
        summary="constructed fixture",
        supersedes=new.semantic_hash,
    )
    report = verify_replay(
        original,
        resolver=ReplayResolver.create(
            receipts=(original, rerun, old, new),
            reruns=(
                (original.semantic_hash, rerun.semantic_hash),
                (old.semantic_hash, new.semantic_hash),
            ),
        ),
    )
    assert report.status == "stale"
    assert f"supersedes:{old.semantic_hash}->{new.semantic_hash}" in report.stale_edges

    record = replay_report_record(report)
    altered = {**record, "schemaType": "wrong"}
    with pytest.raises(ReplayError):
        replay_module._parse(altered)


def test_predecessor_mismatch_and_acyclic_resolver_chain() -> None:
    first = _receipt(environment=ExecutionEnvironment.create(platform="first"))
    second = _receipt(environment=ExecutionEnvironment.create(platform="second"))
    third = _receipt(environment=ExecutionEnvironment.create(platform="third"))
    ReplayResolver.create(
        receipts=(first, second, third),
        reruns=(
            (first.semantic_hash, second.semantic_hash),
            (second.semantic_hash, third.semantic_hash),
        ),
    )
    old, wrong = _receipt(artifacts=(_hash("a"),)), _receipt(artifacts=(_hash("b"),))
    contract = _contract()
    original = run_module._build_receipt(
        contract=contract,
        environment=ExecutionEnvironment.create(),
        started_at="2026-07-24T12:00:00Z",
        finished_at="2026-07-24T12:00:00Z",
        execution_status="completed",
        outcome="constructed",
        summary="constructed fixture",
        supersedes=old.semantic_hash,
    )
    rerun = run_module._build_receipt(
        contract=contract,
        environment=ExecutionEnvironment.create(platform="wrong"),
        started_at="2026-07-24T12:00:00Z",
        finished_at="2026-07-24T12:00:00Z",
        execution_status="completed",
        outcome="constructed",
        summary="constructed fixture",
        supersedes=wrong.semantic_hash,
    )
    report = verify_replay(
        original,
        resolver=ReplayResolver.create(
            receipts=(original, rerun, old, wrong),
            reruns=((original.semantic_hash, rerun.semantic_hash),),
        ),
    )
    assert report.status == "stale"
    assert f"supersedes:{old.semantic_hash}" in report.stale_edges
    assert f"supersedes:{wrong.semantic_hash}" in report.stale_edges

    no_predecessor = _receipt(contract=contract)
    extra_predecessor = _receipt(artifacts=(_hash("c"),))
    rerun_with_extra = run_module._build_receipt(
        contract=contract,
        environment=ExecutionEnvironment.create(platform="extra"),
        started_at="2026-07-24T12:00:00Z",
        finished_at="2026-07-24T12:00:00Z",
        execution_status="completed",
        outcome="constructed",
        summary="constructed fixture",
        supersedes=extra_predecessor.semantic_hash,
    )
    report = verify_replay(
        no_predecessor,
        resolver=ReplayResolver.create(
            receipts=(no_predecessor, rerun_with_extra, extra_predecessor),
            reruns=((no_predecessor.semantic_hash, rerun_with_extra.semantic_hash),),
        ),
    )
    assert report.status == "stale"
    assert report.stale_edges == (f"supersedes:{extra_predecessor.semantic_hash}",)


def test_maximum_direct_report_and_resolver_collections_are_not_truncated() -> None:
    def numbered(index: int) -> SemanticHash:
        return SemanticHash("sha256", f"{index:064x}")

    pairs = tuple((numbered(index), numbered(index + 257)) for index in range(257))
    assert len(replay_module._pairs(pairs, "reruns")) == 257
    assert (
        len(
            replay_module._report_hashes(
                tuple(numbered(index) for index in range(512)), "unavailable"
            )
        )
        == 512
    )
    original, rerun = (
        _receipt(),
        _receipt(environment=ExecutionEnvironment.create(platform="limit")),
    )
    report = replay_module._build(
        status="stale",
        original_receipt_hash=original.semantic_hash,
        rerun_receipt_hash=rerun.semantic_hash,
        stale_edges=tuple(
            f"dependency_receipt:{numbered(index)}" for index in range(1024)
        ),
    )
    assert len(report.stale_edges) == 1024


@pytest.mark.parametrize(
    ("attribute", "value"),
    (("digest", "INVALID"), ("algorithm", "invalid-algorithm")),
)
def test_mutated_semantic_hashes_fail_through_sanitized_replay_boundary(
    attribute: str, value: str
) -> None:
    malformed = _hash()
    object.__setattr__(malformed, attribute, value)
    for action in (
        lambda: ReplayResolver.create(receipts=(), reruns=((malformed, malformed),)),
        lambda: replay_module._build(
            status="unavailable",
            original_receipt_hash=malformed,
            rerun_receipt_hash=None,
            unavailable_references=(_hash("b"),),
        ),
    ):
        with pytest.raises(ReplayError) as raised:
            action()
        assert value not in str(raised.value)
