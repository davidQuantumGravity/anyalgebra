"""Adversarial contract tests for V00-076 claim promotion."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import cast

import pytest

import anyalgebra.evidence.claims as claims
import anyalgebra.evidence.run as run
from anyalgebra.core.parents import SemanticHash
from anyalgebra.evidence.claims import (
    ClaimError,
    ClaimRecord,
    claim_record_registry,
    promote_claim,
)
from anyalgebra.evidence.contracts import CalculationContract
from anyalgebra.evidence.models import SourceAnchor
from anyalgebra.evidence.run import (
    ExecutionEnvironment,
    IndependentCheck,
    ReceiptError,
    ResultReceipt,
)
from anyalgebra.persistence.registry import SchemaError


def _hash(character: str = "a") -> SemanticHash:
    return SemanticHash("sha256", character * 64)


def _anchor() -> SourceAnchor:
    return SourceAnchor.create("draft.tex", _hash("b"), "v1", "line:1")


def _contract(
    *,
    assumptions: object = (),
    source_anchors: object = (),
    bounds: object = (("cases", 8),),
    outcomes: object = ("constructed", "counterexample", "inconclusive"),
) -> CalculationContract:
    return CalculationContract.create(
        contract_name="claim-test",
        project_id="test.project",
        question="Does the declared finite object satisfy its operation?",
        acceptable_outcomes=outcomes,
        input_hashes=(("input", _hash("c")),),
        source_anchors=source_anchors,
        convention_manifests=(),
        algorithms=(("reference", "1"),),
        backend_request="exact",
        backend_name="reference",
        backend_version="1",
        assumptions=assumptions,
        theorem_hypotheses=(),
        bounds=bounds,
        required_cross_checks=(),
        expected_artifacts=(),
        acceptance_predicates=("stored",),
    )


def _receipt(
    tier: str,
    *,
    outcome: str = "constructed",
    supersedes: SemanticHash | None = None,
    contract: CalculationContract | None = None,
) -> ResultReceipt:
    contract = _contract() if contract is None else contract
    common: dict[str, object] = {
        "contract": contract,
        "environment": ExecutionEnvironment.create(),
        "started_at": "2026-07-24T00:00:00Z",
        "finished_at": "2026-07-24T00:00:00Z",
        "execution_status": "completed",
        "outcome": outcome,
        "summary": "bounded calculation completed",
        "case_counts": (("casesExpected", 8), ("casesRun", 8)),
        "witnesses": (_hash("d"),) if outcome == "counterexample" else (),
        "remaining_branches": ("later",) if outcome == "inconclusive" else (),
    }
    if tier == "E1-executed":
        return run._build_receipt(**common)
    common.update(
        fixtures=(_hash("e"),),
        test_artifacts=(_hash("f"),),
        supersedes=_hash("1") if supersedes is None else supersedes,
    )
    if tier == "E2-regression":
        return run._build_receipt(**common, evidence_tier=tier)
    check = IndependentCheck.create(
        method_id="independent",
        implementation_hash=_hash("2"),
        result_hash=run._primary_result(
            contract.semantic_hash,
            outcome,
            (("casesExpected", 8), ("casesRun", 8)),
            cast(tuple[SemanticHash, ...], common["witnesses"]),
            (),
            cast(tuple[str, ...], common["remaining_branches"]),
        ),
    )
    if tier == "E3-bounded-exact":
        return run._build_receipt(**common, evidence_tier=tier)
    common["independent_checks"] = (check,)
    if tier == "E4-cross-checked":
        return run._build_receipt(**common, evidence_tier=tier)
    common["proof_obligations"] = (_hash("3"),)
    if tier == "E5-proof-linked":
        return run._build_receipt(**common, evidence_tier=tier)
    return run._build_receipt(
        **common, evidence_tier=tier, reproduction_attestations=(_hash("4"),)
    )


def _base() -> ClaimRecord:
    return ClaimRecord.create(
        project_id="test.project",
        claim_key="finite-operation",
        claim_text="The declared finite operation has the requested property.",
        source_anchors=(_anchor(),),
        contract=_contract(),
    )


def _e1_claim(*, outcome: str = "constructed") -> ClaimRecord:
    scoped = {
        "counterexample": "counterexample found",
        "reproduction_only": "reproduction only",
        "not_applicable": "not applicable",
        "inconclusive": "inconclusive to bounds",
        "no_go_within_assumptions": "no-go under assumptions",
    }
    contract = (
        _contract()
        if outcome == "constructed"
        else _contract(outcomes=("constructed", outcome))
    )
    base = (
        _base()
        if outcome == "constructed"
        else ClaimRecord.create(
            project_id="test.project",
            claim_key="finite-operation",
            claim_text="The declared finite operation has the requested property.",
            source_anchors=(_anchor(),),
            contract=contract,
        )
    )
    return promote_claim(
        base,
        target_state="E1-executed",
        supporting_receipts=(
            _receipt("E1-executed", outcome=outcome, contract=contract),
        ),
        allowed_public_wording=(
            "implemented"
            if outcome not in scoped
            else f"implemented; {scoped[outcome]}"
        ),
    )


def test_e0_has_literal_canonical_bytes_and_stable_golden_hash() -> None:
    claim = _base()
    expected_sha256 = "4e297fcf895e1653d877f86c567c1c29b3e9b7dd7163315d627263d911d9db9f"
    expected_bytes = (
        b'{"allowedPublicWording":"unimplemented","claimKey":"finite-operation",'
        b'"claimText":"The declared finite operation has the requested property.",'
        b'"competingHypothesisIds":[],"contractHash":{"algorithm":"sha256",'
        b'"digest":"15cd33e09139c6a9bb146bd8064aa6f8cefb3afb1a9a41626b658833584f3cea"},'
        b'"contractRecord":{"acceptableOutcomes":["constructed","counterexample",'
        b'"inconclusive"],"acceptancePredicates":["stored"],"algorithms":['
        b'{"key":"reference","value":"1"}],"assumptions":[],"backend":'
        b'{"name":"reference","request":"exact","version":"1"},"bounds":'
        b'[{"key":"cases","value":8}],"contractName":"claim-test",'
        b'"conventionManifests":[],"expectedArtifacts":[],"inputHashes":['
        b'{"hash":{"algorithm":"sha256","digest":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"},'
        b'"role":"input"}],"projectId":"test.project","question":'
        b'"Does the declared finite object satisfy its operation?",'
        b'"requiredCrossChecks":[],"schemaType":'
        b'"anyalgebra.evidence.calculation_contract","schemaVersion":1,'
        b'"sourceAnchors":[],"theoremHypotheses":[]},"dependencyClaimHashes":[],'
        b'"dependencyReceiptHashes":[],"evidenceState":"E0-unimplemented",'
        b'"priorSupportingReceiptHashes":[],"projectId":"test.project",'
        b'"schemaType":"anyalgebra.evidence.claim_record","schemaVersion":1,'
        b'"sourceAnchors":[{"contentHash":{"algorithm":"sha256","digest":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},'
        b'"editionOrCommit":"v1","excerptHash":null,"location":"draft.tex",'
        b'"locator":"line:1","schemaType":"anyalgebra.evidence.source_anchor",'
        b'"schemaVersion":1}],"supersedesClaimHash":null,'
        b'"supersedesEvidenceState":null,"supportingReceiptHashes":[],'
        b'"supportingReceipts":[],"unresolvedObligations":[]}'
    )
    assert claim.canonical_bytes() == expected_bytes
    assert hashlib.sha256(claim.canonical_bytes()).hexdigest() == expected_sha256
    assert claim.semantic_hash == SemanticHash("sha256", expected_sha256)
    assert repr(claim) == f"ClaimRecord(claim_id={claim.claim_id!r})"
    assert isinstance(hash(claim), int)
    assert claim_record_registry().from_record(claim.to_record()) == claim
    with pytest.raises(ClaimError, match="factory-owned"):
        ClaimRecord()
    with pytest.raises(TypeError):

        class Bad(ClaimRecord):
            pass


def test_each_promotion_is_append_only_next_state_and_preserves_prior() -> None:
    claim = _base()
    states = (
        "E1-executed",
        "E2-regression",
        "E3-bounded-exact",
        "E4-cross-checked",
        "E5-proof-linked",
        "E6-reproduced",
    )
    for state in states:
        prior = claim
        claim = promote_claim(
            prior,
            target_state=state,
            supporting_receipts=(
                _receipt(
                    state,
                    supersedes=None
                    if state == "E1-executed"
                    else prior.supporting_receipts[0].semantic_hash,
                ),
            ),
            allowed_public_wording=claims._badge(state, _contract()),
        )
        assert claim.supersedes_claim_hash == prior.semantic_hash
        assert prior.evidence_state != claim.evidence_state
        assert claim.supporting_receipts[0].semantic_hash in {
            item.semantic_hash for item in claim.supporting_receipts
        }
    assert claim.evidence_state == "E6-reproduced"
    with pytest.raises(ClaimError, match="next evidence"):
        promote_claim(
            claim,
            target_state="E6-reproduced",
            supporting_receipts=(_receipt("E6-reproduced"),),
            allowed_public_wording="independently reproduced",
        )


@pytest.mark.parametrize(
    ("target", "receipt"),
    (
        ("E2-regression", "E1-executed"),
        ("E3-bounded-exact", "E2-regression"),
        ("E4-cross-checked", "E3-bounded-exact"),
        ("E5-proof-linked", "E4-cross-checked"),
        ("E6-reproduced", "E5-proof-linked"),
    ),
)
def test_promotion_rejects_weak_receipt_and_skips(target: str, receipt: str) -> None:
    claim = _base()
    if target != "E1-executed":
        claim = promote_claim(
            claim,
            target_state="E1-executed",
            supporting_receipts=(_receipt("E1-executed"),),
            allowed_public_wording="implemented",
        )
    with pytest.raises(ClaimError):
        promote_claim(
            claim,
            target_state=target,
            supporting_receipts=(_receipt(receipt),),
            allowed_public_wording=claims._badge(target, _contract()),
        )


def test_contract_linkage_and_hash_only_fabrication_are_rejected() -> None:
    claim = _base()
    with pytest.raises(ClaimError, match="exact linked"):
        promote_claim(
            claim,
            target_state="E1-executed",
            supporting_receipts=(),
            allowed_public_wording="implemented",
        )
    with pytest.raises(ClaimError, match="exact ResultReceipt"):
        promote_claim(
            claim,
            target_state="E1-executed",
            supporting_receipts=(_hash(),),
            allowed_public_wording="implemented",
        )
    foreign = run._build_receipt(
        contract=CalculationContract.create(
            contract_name="other",
            project_id="test.project",
            question="other",
            acceptable_outcomes=("constructed",),
            input_hashes=(("x", _hash()),),
            source_anchors=(),
            convention_manifests=(),
            algorithms=(("a", "1"),),
            backend_request="exact",
            backend_name="a",
            backend_version="1",
            assumptions=(),
            theorem_hypotheses=(),
            bounds=(),
            required_cross_checks=(),
            expected_artifacts=(),
            acceptance_predicates=("stored",),
        ),
        environment=ExecutionEnvironment.create(),
        started_at="2026-07-24T00:00:00Z",
        finished_at="2026-07-24T00:00:00Z",
        execution_status="completed",
        outcome="constructed",
        summary="ok",
    )
    with pytest.raises(ClaimError, match="does not match"):
        promote_claim(
            claim,
            target_state="E1-executed",
            supporting_receipts=(foreign,),
            allowed_public_wording="implemented",
        )


def test_wording_badges_negative_scope_and_hypotheses_are_semantic() -> None:
    base = _base()
    with pytest.raises(ClaimError, match="badge"):
        promote_claim(
            base,
            target_state="E1-executed",
            supporting_receipts=(_receipt("E1-executed"),),
            allowed_public_wording="available",
        )
    with pytest.raises(ClaimError, match="unsupported"):
        promote_claim(
            base,
            target_state="E1-executed",
            supporting_receipts=(_receipt("E1-executed"),),
            allowed_public_wording="implemented: proved",
        )
    negative = promote_claim(
        base,
        target_state="E1-executed",
        supporting_receipts=(_receipt("E1-executed", outcome="counterexample"),),
        allowed_public_wording="implemented; counterexample found",
        competing_hypothesis_ids=("alternative-a",),
    )
    assert negative.competing_hypothesis_ids == ("alternative-a",)
    with pytest.raises(ClaimError, match="negative outcome"):
        promote_claim(
            base,
            target_state="E1-executed",
            supporting_receipts=(_receipt("E1-executed", outcome="counterexample"),),
            allowed_public_wording="implemented",
        )


def test_parser_strictness_tamper_and_collection_boundaries() -> None:
    claim = promote_claim(
        _base(),
        target_state="E1-executed",
        supporting_receipts=(_receipt("E1-executed"),),
        allowed_public_wording="implemented",
    )
    record = claim.to_record()
    record["claimText"] = "other"
    assert (
        cast(ClaimRecord, claim_record_registry().from_record(record)).semantic_hash
        != claim.semantic_hash
    )
    record = claim.to_record()
    record["supportingReceiptHashes"] = []
    with pytest.raises(SchemaError, match="parser_failed"):
        claim_record_registry().from_record(record)
    record = claim.to_record()
    record["unexpected"] = None
    with pytest.raises(SchemaError, match="parser_failed"):
        claim_record_registry().from_record(record)
    hashes = tuple(SemanticHash("sha256", f"{item:064x}") for item in range(256))
    assert (
        len(
            ClaimRecord.create(
                project_id="test.project",
                claim_key="k",
                claim_text="t",
                source_anchors=(_anchor(),),
                contract=_contract(),
                dependency_claim_hashes=hashes,
            ).dependency_claim_hashes
        )
        == 256
    )
    with pytest.raises(ClaimError, match="limit"):
        ClaimRecord.create(
            project_id="test.project",
            claim_key="k",
            claim_text="t",
            source_anchors=(_anchor(),),
            contract=_contract(),
            dependency_claim_hashes=(*hashes, _hash("f")),
        )
    object.__setattr__(claim, "claim_text", "tampered")
    with pytest.raises(ClaimError, match="integrity"):
        claim.to_record()


def test_bad_iterators_interrupts_and_invalid_registry_paths() -> None:
    def bad() -> Iterator[object]:
        yield _anchor()
        raise RuntimeError("hidden")

    with pytest.raises(ClaimError, match="snapshot"):
        ClaimRecord.create(
            project_id="p",
            claim_key="k",
            claim_text="t",
            source_anchors=bad(),
            contract=_contract(),
        )

    def halted() -> Iterator[object]:
        raise KeyboardInterrupt
        yield _anchor()

    with pytest.raises(KeyboardInterrupt):
        ClaimRecord.create(
            project_id="p",
            claim_key="k",
            claim_text="t",
            source_anchors=halted(),
            contract=_contract(),
        )
    with pytest.raises(ClaimError):
        claims._parse_hash({"bad": "shape"}, "x")


def test_all_small_validation_and_parser_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for value in (None, 1, "", " x ", " " * 4097, "\ud800"):
        with pytest.raises(ClaimError):
            claims._text(value, "x", identifier=True)
    for item_value in ("x", object()):
        with pytest.raises(ClaimError):
            claims._items(item_value, "x")
    with pytest.raises(ClaimError, match="exact SemanticHash"):
        claims._hash("no", "x")
    with pytest.raises(ClaimError, match="duplicate"):
        claims._hashes((_hash(), _hash()), "x")
    with pytest.raises(ClaimError, match="duplicate"):
        claims._strings(("x", "x"), "x")
    for invalid_value in (object(),):
        with pytest.raises(ClaimError):
            claims._contract(invalid_value)
        with pytest.raises(ClaimError):
            claims._anchor(invalid_value)
        with pytest.raises(ClaimError):
            claims._receipt(invalid_value)
    bad_contract = _contract()
    object.__setattr__(bad_contract, "semantic_hash", _hash("8"))
    with pytest.raises(ClaimError, match="integrity"):
        claims._contract(bad_contract)
    bad_anchor = _anchor()
    object.__setattr__(bad_anchor, "semantic_hash", _hash("8"))
    with pytest.raises(ClaimError, match="integrity"):
        claims._anchor(bad_anchor)
    bad_receipt = _receipt("E1-executed")
    object.__setattr__(bad_receipt, "semantic_hash", _hash("8"))
    with pytest.raises(ClaimError, match="integrity"):
        claims._receipt(bad_receipt)

    class WrongContractRegistry:
        def from_record(self, record: object) -> object:
            del record
            return object()

    monkeypatch.setattr(
        claims, "contract_record_registry", lambda: WrongContractRegistry()
    )
    with pytest.raises(ClaimError, match="integrity"):
        claims._contract(_contract())
    monkeypatch.undo()
    monkeypatch.setattr(
        claims,
        "result_receipt_record",
        lambda value: (_ for _ in ()).throw(ReceiptError("x", "x")),
    )
    with pytest.raises(ClaimError, match="integrity"):
        claims._receipt(_receipt("E1-executed"))
    monkeypatch.undo()

    class WrongReceiptRegistry:
        def from_record(self, record: object) -> object:
            del record
            return object()

    monkeypatch.setattr(
        claims, "result_receipt_registry", lambda: WrongReceiptRegistry()
    )
    with pytest.raises(ClaimError, match="integrity"):
        claims._receipt(_receipt("E1-executed"))
    monkeypatch.undo()
    with pytest.raises(ClaimError, match="must not be empty"):
        claims._anchors(())
    with pytest.raises(ClaimError, match="duplicate"):
        claims._anchors((_anchor(), _anchor()))
    with pytest.raises(ClaimError, match="duplicate"):
        claims._receipts((_receipt("E1-executed"), _receipt("E1-executed")))
    with pytest.raises(ClaimError, match="closed"):
        claims._claim_state("E7")
    assert claims._wording("proof-linked: proved", "E5-proof-linked", (), _contract())
    with pytest.raises(ClaimError, match="scoped"):
        claims._wording("implemented", "E1-executed", ("counterexample",), _contract())
    incomplete = _receipt("E1-executed")
    object.__setattr__(incomplete, "case_counts", (("casesExpected", 8),))
    assert not claims._complete(incomplete)
    receipt = _receipt("E1-executed")
    object.__setattr__(receipt, "execution_status", "failed")
    with pytest.raises(ClaimError, match="failed"):
        claims._qualifies((receipt,), "E1-executed", _contract())
    with pytest.raises(ClaimError, match="E0 cannot"):
        claims._build(
            project_id="test.project",
            claim_key="k",
            claim_text="t",
            source_anchors=(_anchor(),),
            contract=_contract(),
            evidence_state="E0-unimplemented",
            allowed_public_wording="unimplemented",
            supporting_receipts=(),
            dependency_claim_hashes=(),
            dependency_receipt_hashes=(),
            prior_supporting_receipt_hashes=(),
            competing_hypothesis_ids=(),
            unresolved_obligations=(),
            supersedes_claim_hash=_hash(),
            supersedes_evidence_state=None,
            promotion=False,
        )
    with pytest.raises(ClaimError, match="requires promotion"):
        claims._build(
            project_id="test.project",
            claim_key="k",
            claim_text="t",
            source_anchors=(_anchor(),),
            contract=_contract(),
            evidence_state="E1-executed",
            allowed_public_wording="implemented",
            supporting_receipts=(),
            dependency_claim_hashes=(),
            dependency_receipt_hashes=(),
            prior_supporting_receipt_hashes=(),
            competing_hypothesis_ids=(),
            unresolved_obligations=(),
            supersedes_claim_hash=None,
            supersedes_evidence_state=None,
            promotion=False,
        )
    with pytest.raises(ClaimError, match="E1 cannot"):
        claims._build(
            project_id="test.project",
            claim_key="k",
            claim_text="t",
            source_anchors=(_anchor(),),
            contract=_contract(),
            evidence_state="E1-executed",
            allowed_public_wording="implemented",
            supporting_receipts=(_receipt("E1-executed"),),
            dependency_claim_hashes=(),
            dependency_receipt_hashes=(),
            prior_supporting_receipt_hashes=(_hash("a"),),
            competing_hypothesis_ids=(),
            unresolved_obligations=(),
            supersedes_claim_hash=_hash("b"),
            supersedes_evidence_state="E0-unimplemented",
            promotion=True,
        )
    with pytest.raises(ClaimError):
        claims._mapping((), frozenset(), "x")
    with pytest.raises(ClaimError):
        claims._tuple([], "x")
    with pytest.raises(ClaimError):
        claims._parse_hash({"algorithm": 1, "digest": "a" * 64}, "x")
    with pytest.raises(ClaimError):
        claims._parse_hash({"algorithm": "bad", "digest": "x"}, "x")
    with pytest.raises(ClaimError):
        claims._parse_contract({})
    with pytest.raises(ClaimError):
        claims._parse_anchor({})
    with pytest.raises(ClaimError):
        claims._parse_receipt({})
    e1_claim = promote_claim(
        _base(),
        target_state="E1-executed",
        supporting_receipts=(_receipt("E1-executed"),),
        allowed_public_wording="implemented",
    )
    invalid_schema = e1_claim.to_record()
    invalid_schema["schemaVersion"] = 2
    with pytest.raises(ClaimError):
        claims._parse(invalid_schema)
    mismatch = e1_claim.to_record()
    mismatch["contractHash"] = {"algorithm": "sha256", "digest": "9" * 64}
    with pytest.raises(ClaimError):
        claims._parse(mismatch)
    with pytest.raises(ClaimError):
        claims.claim_record(object())
    with pytest.raises(ClaimError):
        claims.claim_canonical_bytes(object())
    assert claims.claim_record_registry() is claims.CLAIM_RECORD_REGISTRY
    with pytest.raises(ClaimError):
        ClaimRecord.create.__func__(  # type: ignore[attr-defined]
            object,
            project_id="p",
            claim_key="k",
            claim_text="t",
            source_anchors=(_anchor(),),
            contract=_contract(),
        )

    class BrokenRegistry:
        def to_record(self, value: object) -> dict[str, object]:
            del value
            raise SchemaError("x")

    monkeypatch.setattr(claims, "CLAIM_RECORD_REGISTRY", BrokenRegistry())
    raw = object.__new__(ClaimRecord)
    object.__setattr__(raw, "semantic_hash", _hash())
    with pytest.raises(ClaimError, match="canonical"):
        claims._ensure(raw)
    monkeypatch.setattr(
        claims, "_record_hash", lambda _: (_ for _ in ()).throw(AttributeError())
    )
    with pytest.raises(ClaimError, match="integrity"):
        claims._ensure(raw)


def test_qualifying_artifact_rules_and_stale_receipts_are_checked() -> None:
    contract = _contract()
    e1 = _receipt("E1-executed")
    with pytest.raises(ClaimError, match="tier is too weak"):
        claims._qualifies((e1,), "E2-regression", contract)
    for state, text in (
        ("E2-regression", "fixtures"),
        ("E3-bounded-exact", "exact complete"),
        ("E4-cross-checked", "distinct agreeing"),
        ("E5-proof-linked", "proof"),
        ("E6-reproduced", "reproduction"),
    ):
        weak = _receipt(state)
        if state == "E2-regression":
            object.__setattr__(weak, "fixtures", ())
        elif state == "E3-bounded-exact":
            object.__setattr__(weak, "case_counts", (("casesRun", 8),))
        elif state == "E4-cross-checked":
            object.__setattr__(weak, "independent_checks", ())
        elif state == "E5-proof-linked":
            object.__setattr__(weak, "proof_obligations", ())
        else:
            object.__setattr__(weak, "reproduction_attestations", ())
        with pytest.raises(ClaimError, match=text):
            claims._qualifies((weak,), state, contract)
    object.__setattr__(e1, "contract_hash", _hash("9"))
    with pytest.raises(ClaimError, match="does not match"):
        claims._qualifies((e1,), "E1-executed", contract)


def test_badge_and_negative_wording_are_structured_not_substrings() -> None:
    for state in claims._STATES:
        badge = claims._badge(state, _contract())
        assert claims._wording(f"{badge.upper()}: scoped text", state, (), _contract())
        for bad in (f"not {badge}", f"{badge}ness", f"x {badge}"):
            with pytest.raises(ClaimError, match="badge"):
                claims._wording(bad, state, (), _contract())
    for bad in (
        "implemented; no counterexample found",
        "implemented; not counterexample found",
        "implemented; counterexample foundry",
    ):
        with pytest.raises(ClaimError, match="scoped"):
            claims._wording(bad, "E1-executed", ("counterexample",), _contract())
    assert claims._wording(
        "IMPLEMENTED; COUNTEREXAMPLE FOUND.",
        "E1-executed",
        ("counterexample",),
        _contract(),
    )
    with pytest.raises(ClaimError, match="scoped"):
        claims._wording(
            "implemented; counterexample found",
            "E1-executed",
            ("counterexample", "inconclusive"),
            _contract(),
        )
    assert claims._wording(
        "implemented; counterexample found; inconclusive to bounds",
        "E1-executed",
        ("counterexample", "inconclusive"),
        _contract(),
    )


def test_claim_and_receipt_predecessor_links_are_explicit_and_strict() -> None:
    e1 = _e1_claim()
    unrelated = _receipt("E2-regression")
    with pytest.raises(ClaimError, match="supersession edge"):
        promote_claim(
            e1,
            target_state="E2-regression",
            supporting_receipts=(unrelated,),
            allowed_public_wording="regression-tested",
        )
    e2_receipt = _receipt(
        "E2-regression", supersedes=e1.supporting_receipts[0].semantic_hash
    )
    e2 = promote_claim(
        e1,
        target_state="E2-regression",
        supporting_receipts=(e2_receipt,),
        allowed_public_wording="regression-tested",
    )
    assert e1.supporting_receipts[0].semantic_hash in e2.dependency_receipt_hashes
    assert e2.supersedes_evidence_state == "E1-executed"
    missing = e2.to_record()
    missing["supersedesClaimHash"] = None
    with pytest.raises(SchemaError, match="parser_failed"):
        claim_record_registry().from_record(missing)
    wrong_state = e2.to_record()
    wrong_state["supersedesEvidenceState"] = "E0-unimplemented"
    with pytest.raises(SchemaError, match="parser_failed"):
        claim_record_registry().from_record(wrong_state)
    e3_receipt = _receipt("E3-bounded-exact", supersedes=e2_receipt.semantic_hash)
    e3 = promote_claim(
        e2,
        target_state="E3-bounded-exact",
        supporting_receipts=(e3_receipt,),
        allowed_public_wording="verified on cases=8",
        dependency_receipt_hashes=(_hash("9"),),
    )
    assert _hash("9") in e3.dependency_receipt_hashes
    skip = e3.to_record()
    skip["supersedesEvidenceState"] = "E0-unimplemented"
    with pytest.raises(SchemaError, match="parser_failed"):
        claim_record_registry().from_record(skip)


def test_prior_support_is_not_laundered_through_general_dependencies() -> None:
    e1 = _e1_claim()
    unrelated = _receipt("E2-regression")
    with pytest.raises(ClaimError, match="supersession edge"):
        promote_claim(
            e1,
            target_state="E2-regression",
            supporting_receipts=(unrelated,),
            allowed_public_wording="regression-tested",
            dependency_receipt_hashes=(unrelated.supersedes,),
        )
    valid = _receipt(
        "E2-regression", supersedes=e1.supporting_receipts[0].semantic_hash
    )
    promoted = promote_claim(
        e1,
        target_state="E2-regression",
        supporting_receipts=(valid,),
        allowed_public_wording="regression-tested",
        dependency_receipt_hashes=(_hash("a"),),
    )
    assert promoted.prior_supporting_receipt_hashes == (
        e1.supporting_receipts[0].semantic_hash,
    )
    assert _hash("a") in promoted.dependency_receipt_hashes
    record = promoted.to_record()
    record["priorSupportingReceiptHashes"] = []
    with pytest.raises(SchemaError, match="parser_failed"):
        claim_record_registry().from_record(record)
    record = promoted.to_record()
    record["priorSupportingReceiptHashes"] = [
        {"algorithm": "sha256", "digest": "f" * 64}
    ]
    with pytest.raises(SchemaError, match="parser_failed"):
        claim_record_registry().from_record(record)


def test_claim_dependencies_accumulate_without_caller_drop() -> None:
    base = ClaimRecord.create(
        project_id="test.project",
        claim_key="k",
        claim_text="t",
        source_anchors=(_anchor(),),
        contract=_contract(),
        dependency_claim_hashes=(_hash("a"),),
    )
    e1 = promote_claim(
        base,
        target_state="E1-executed",
        supporting_receipts=(_receipt("E1-executed"),),
        allowed_public_wording="implemented",
        dependency_claim_hashes=(_hash("b"),),
    )
    assert e1.dependency_claim_hashes == (_hash("a"), _hash("b"))
    e2_receipt = _receipt(
        "E2-regression", supersedes=e1.supporting_receipts[0].semantic_hash
    )
    e2 = promote_claim(
        e1,
        target_state="E2-regression",
        supporting_receipts=(e2_receipt,),
        allowed_public_wording="regression-tested",
        dependency_claim_hashes=(),
    )
    assert e2.dependency_claim_hashes == (_hash("a"), _hash("b"))


def test_public_badges_are_contract_bound_and_all_scoped_outcomes_are_named() -> None:
    contract = _contract()
    assert claims._badge("E3-bounded-exact", contract) == "verified on cases=8"
    assert (
        claims._badge("E4-cross-checked", contract)
        == "independently cross-checked on cases=8"
    )
    for state, valid in (
        ("E3-bounded-exact", "verified on cases=8"),
        ("E4-cross-checked", "independently cross-checked on cases=8"),
    ):
        assert claims._wording(valid, state, (), contract)
        for wrong in (
            valid.replace("=8", "=9"),
            "verified on explicit domain",
            "independently cross-checked on domain",
        ):
            with pytest.raises(ClaimError, match="badge"):
                claims._wording(wrong, state, (), contract)
    no_cases = _contract(bounds=(("label", "finite"),))
    fallback = f"verified on contract={no_cases.contract_id}"
    assert claims._wording(fallback, "E3-bounded-exact", (), no_cases)
    with pytest.raises(ClaimError, match="badge"):
        claims._wording(
            "verified on contract=contract:sha256:" + "0" * 64,
            "E3-bounded-exact",
            (),
            no_cases,
        )
    for outcome, phrase in (
        ("reproduction_only", "reproduction only"),
        ("not_applicable", "not applicable"),
    ):
        with pytest.raises(ClaimError, match="scoped"):
            claims._wording("implemented", "E1-executed", (outcome,), contract)
        with pytest.raises(ClaimError, match="scoped"):
            claims._wording(
                f"implemented; no {phrase}", "E1-executed", (outcome,), contract
            )
        assert claims._wording(
            f"implemented; {phrase}", "E1-executed", (outcome,), contract
        )
    with pytest.raises(ClaimError, match="scoped"):
        claims._wording(
            "implemented; reproduction only",
            "E1-executed",
            ("reproduction_only", "not_applicable"),
            contract,
        )
    for wording in (
        "unimplemented; independently reproduced",
        "unimplemented; counterexample found",
        "implemented; independently reproduced",
        "implemented; independently reproduced by Alice",
        "implemented; status is independently reproduced",
        "implemented; proof-linked",
        "implemented; proof-linked to appendix",
        "implemented; this is proof-linked",
        "implemented; verified on cases=999",
        "implemented; independently cross-checked on arbitrary-domain",
        "implemented; counterexample found",
        "implemented; counterexample found in case 3",
        "implemented; result says counterexample found",
        "implemented; no-go under assumptions X",
    ):
        state = (
            "E0-unimplemented" if wording.startswith("unimplemented") else "E1-executed"
        )
        with pytest.raises(ClaimError):
            claims._wording(wording, state, (), contract)


def test_contract_rejects_a_sealed_receipt_with_an_unaccepted_outcome() -> None:
    with pytest.raises(ClaimError, match="outcome is not accepted"):
        promote_claim(
            _base(),
            target_state="E1-executed",
            supporting_receipts=(_receipt("E1-executed", outcome="reproduction_only"),),
            allowed_public_wording="implemented; reproduction only",
        )


def test_project_source_and_outcome_boundaries_are_enforced() -> None:
    anchor, other = (
        _anchor(),
        SourceAnchor.create("other.tex", _hash("7"), "v1", "line:2"),
    )
    contract = _contract(source_anchors=(anchor,))
    with pytest.raises(ClaimError, match="project"):
        ClaimRecord.create(
            project_id="wrong.project",
            claim_key="k",
            claim_text="t",
            source_anchors=(anchor,),
            contract=contract,
        )
    with pytest.raises(ClaimError, match="include every"):
        ClaimRecord.create(
            project_id="test.project",
            claim_key="k",
            claim_text="t",
            source_anchors=(other,),
            contract=contract,
        )
    seeded = ClaimRecord.create(
        project_id="test.project",
        claim_key="k",
        claim_text="t",
        source_anchors=(anchor, other),
        contract=contract,
    )
    record = seeded.to_record()
    record["projectId"] = "wrong.project"
    with pytest.raises(SchemaError, match="parser_failed"):
        claim_record_registry().from_record(record)
    record = seeded.to_record()
    record["sourceAnchors"] = [other.to_record()]
    with pytest.raises(SchemaError, match="parser_failed"):
        claim_record_registry().from_record(record)
    e1 = _e1_claim(outcome="reproduction_only")
    reproduction_e2 = _receipt(
        "E2-regression",
        outcome="reproduction_only",
        supersedes=e1.supporting_receipts[0].semantic_hash,
        contract=e1.contract,
    )
    e2 = promote_claim(
        e1,
        target_state="E2-regression",
        supporting_receipts=(reproduction_e2,),
        allowed_public_wording="regression-tested; reproduction only",
    )
    reproduction_e3 = _receipt(
        "E3-bounded-exact",
        outcome="reproduction_only",
        supersedes=reproduction_e2.semantic_hash,
        contract=e1.contract,
    )
    with pytest.raises(ClaimError, match="correctness-bearing"):
        promote_claim(
            e2,
            target_state="E3-bounded-exact",
            supporting_receipts=(reproduction_e3,),
            allowed_public_wording="verified on cases=8",
        )
    # ``not_applicable`` follows the same E2 cap: it records a failed premise,
    # not a correctness conclusion for the requested broad claim.
    inapplicable_contract = _contract(outcomes=("not_applicable",))
    inapplicable = _receipt(
        "E3-bounded-exact",
        outcome="not_applicable",
        contract=inapplicable_contract,
    )
    with pytest.raises(ClaimError, match="correctness-bearing"):
        claims._qualifies((inapplicable,), "E3-bounded-exact", inapplicable_contract)


def test_inconclusive_e3_support_cannot_be_described_as_positive_only() -> None:
    e1 = _e1_claim()
    e2_receipt = _receipt(
        "E2-regression", supersedes=e1.supporting_receipts[0].semantic_hash
    )
    e2 = promote_claim(
        e1,
        target_state="E2-regression",
        supporting_receipts=(e2_receipt,),
        allowed_public_wording="regression-tested",
    )
    supports = (
        _receipt("E3-bounded-exact", supersedes=e2_receipt.semantic_hash),
        _receipt(
            "E3-bounded-exact",
            outcome="inconclusive",
            supersedes=e2_receipt.semantic_hash,
        ),
    )
    with pytest.raises(ClaimError, match="scoped"):
        promote_claim(
            e2,
            target_state="E3-bounded-exact",
            supporting_receipts=supports,
            allowed_public_wording="verified on cases=8",
        )
    record = promote_claim(
        e2,
        target_state="E3-bounded-exact",
        supporting_receipts=supports,
        allowed_public_wording=("verified on cases=8; inconclusive to bounds"),
    )
    assert record.evidence_state == "E3-bounded-exact"
