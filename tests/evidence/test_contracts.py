"""Tests for immutable, pre-execution calculation contracts."""

from __future__ import annotations

import hashlib

import pytest

from anyalgebra.core.parents import SemanticHash
from anyalgebra.evidence import contracts as contracts_module
from anyalgebra.evidence.contracts import (
    CalculationContract,
    ContractError,
    calculation_contract_canonical_bytes,
    calculation_contract_record,
    contract_record_registry,
)
from anyalgebra.evidence.models import (
    ConventionManifest,
    EVIDENCE_RECORD_REGISTRY,
    SourceAnchor,
    convention_manifest_record,
    source_anchor_canonical_bytes,
    source_anchor_record,
)
from anyalgebra.persistence.registry import SchemaError


def _hash(character: str) -> SemanticHash:
    return SemanticHash("sha256", character * 64)


def _source(*, locator: str = "p. 13") -> SourceAnchor:
    return SourceAnchor.create("doi:10.1000/example", _hash("a"), "rev-7", locator)


def _manifest(*, manifest_id: str = "algmul.O.v1") -> ConventionManifest:
    return ConventionManifest.create(
        manifest_id,
        ("1", "e1"),
        coefficient_domain="QQ",
        multiplication_fixture=_hash("b"),
        sources=(_source(),),
        status="accepted",
        signs=(("orientation", "positive"),),
        involutions=(("conjugation", "standard"),),
        normalization=(("unit", "one"),),
        indexing="zero",
        parenthesization="explicit",
        conversions=(("legacy", "identity"),),
    )


def _contract(**changes: object) -> CalculationContract:
    values: dict[str, object] = {
        "contract_name": "octonion-associator-search.v1",
        "project_id": "exceptional-physics",
        "question": "Find the first nonzero octonion associator in the declared basis.",
        "acceptable_outcomes": ("counterexample", "inconclusive"),
        "input_hashes": (("left", _hash("c")), ("right", _hash("d"))),
        "source_anchors": (_source(),),
        "convention_manifests": (_manifest(),),
        "algorithms": (("direct-table-enumeration", "1.0"),),
        "backend_request": "exact-reference-backend",
        "backend_name": "reference",
        "backend_version": "0.0",
        "assumptions": (("basis", "algmul.O.v1"),),
        "theorem_hypotheses": (("carrier", "finite"),),
        "bounds": (("cases", 512), ("memoryBytes", 1048576)),
        "required_cross_checks": ("direct basis product",),
        "expected_artifacts": ("witness-json", "run-log"),
        "acceptance_predicates": (
            "Record the lexicographically first nonzero witness.",
            "Check every declared basis triple.",
        ),
    }
    values.update(changes)
    return CalculationContract.create(**values)


def _large_text(index: int) -> str:
    return f"{index:06d}" + "x" * 4090


def _alias_spellings(words: tuple[str, ...]) -> tuple[str, ...]:
    joined = "".join(words)
    return (
        "_".join(words),
        " ".join(words),
        ".".join(words),
        "/".join(words),
        "-".join(words),
        words[0] + "".join(word.title() for word in words[1:]),
        "".join(word.title() for word in words),
        joined,
    )


_AUDIT_ALIAS_SPELLINGS = tuple(
    spelling
    for words in (
        ("result", "state"),
        ("receipt", "tier"),
        ("scientific", "conclusion"),
        ("proof", "level"),
        ("promotion", "state"),
        ("is", "verified"),
        ("evidence", "class"),
        ("execution", "phase"),
        ("claim", "outcome"),
        ("receipt", "status"),
        ("run", "phase"),
    )
    for spelling in _alias_spellings(words)
)
_STRUCTURAL_SUFFIX_SPELLINGS = tuple(
    spelling
    for words in (
        ("status", "count"),
        ("tier", "limit"),
        ("verified", "cases"),
        ("promotion", "attempts"),
        ("conclusion", "length"),
        ("claim", "count"),
        ("evidence", "size"),
        ("receipt", "count"),
    )
    for spelling in _alias_spellings(words)
)
_COLLAPSED_MULTIWORD_STATUS_KEYS = (
    "executionstatuscount",
    "resultstatelimit",
    "evidencetierlimit",
    "receiptstatuscount",
    "prooflevelcount",
    "runphasecount",
)


def test_failing_contract_creation_contract_is_content_addressed() -> None:
    contract = _contract()
    assert contract.contract_name == "octonion-associator-search.v1"
    assert contract.contract_id == f"contract:sha256:{contract.semantic_hash.digest}"
    assert contract.semantic_hash.algorithm == "sha256"
    assert calculation_contract_canonical_bytes(contract)


def test_canonical_golden_and_round_trip_are_literal_and_exact() -> None:
    contract = _contract()
    expected = b'{"acceptableOutcomes":["counterexample","inconclusive"],"acceptancePredicates":["Check every declared basis triple.","Record the lexicographically first nonzero witness."],"algorithms":[{"key":"direct-table-enumeration","value":"1.0"}],"assumptions":[{"key":"basis","value":"algmul.O.v1"}],"backend":{"name":"reference","request":"exact-reference-backend","version":"0.0"},"bounds":[{"key":"cases","value":512},{"key":"memoryBytes","value":1048576}],"contractName":"octonion-associator-search.v1","conventionManifests":[{"algorithm":"sha256","digest":"98ba7d26f1958bf1713a940559ca2c642d07cbc33d7c5294f6ba3804043c9571"}],"expectedArtifacts":["run-log","witness-json"],"inputHashes":[{"hash":{"algorithm":"sha256","digest":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"},"role":"left"},{"hash":{"algorithm":"sha256","digest":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"},"role":"right"}],"projectId":"exceptional-physics","question":"Find the first nonzero octonion associator in the declared basis.","requiredCrossChecks":["direct basis product"],"schemaType":"anyalgebra.evidence.calculation_contract","schemaVersion":1,"sourceAnchors":[{"algorithm":"sha256","digest":"a4d61642b0b04d812ed96161b6761fb1f835aa67a1303a0276abc21531eb1b19"}],"theoremHypotheses":[{"key":"carrier","value":"finite"}]}'  # noqa: E501
    expected_digest = "f7217ac9a36b34c79b9ca8a9b4b8e4a76bf7998cb310d06131075f1a9e87f5a7"
    assert contract.canonical_bytes() == expected
    assert contract.semantic_hash.digest == expected_digest
    assert contract.contract_id == f"contract:sha256:{expected_digest}"
    assert contract.contract_id.encode("ascii") not in expected
    record = calculation_contract_record(contract)
    assert record["contractName"] == contract.contract_name
    assert "id" not in record
    parsed = contract_record_registry().from_record(record)
    assert type(parsed) is CalculationContract
    assert parsed == contract
    assert parsed.contract_id == contract.contract_id
    assert parsed.canonical_bytes() == expected


def test_every_semantic_field_changes_content_address() -> None:
    baseline = _contract()
    variants = (
        _contract(contract_name="other.v1"),
        _contract(project_id="other-project"),
        _contract(question="A different exact question."),
        _contract(acceptable_outcomes=("constructed",)),
        _contract(input_hashes=(("left", _hash("e")), ("right", _hash("d")))),
        _contract(input_hashes=(("first", _hash("c")), ("right", _hash("d")))),
        _contract(source_anchors=(_source(locator="p. 14"),)),
        _contract(convention_manifests=(_manifest(manifest_id="algmul.H.v1"),)),
        _contract(algorithms=(("other", "1.0"),)),
        _contract(backend_request="other-request"),
        _contract(backend_name="other-backend"),
        _contract(backend_version="0.1"),
        _contract(assumptions=(("basis", "other"),)),
        _contract(theorem_hypotheses=(("carrier", "infinite"),)),
        _contract(bounds=(("cases", 513),)),
        _contract(required_cross_checks=("independent implementation",)),
        _contract(expected_artifacts=("certificate",)),
        _contract(acceptance_predicates=("A different predicate.",)),
    )
    assert all(item.semantic_hash != baseline.semantic_hash for item in variants)
    assert all(item.contract_id != baseline.contract_id for item in variants)


def test_unordered_collections_canonicalize_including_predicate_conjunctions() -> None:
    first = _contract()
    permuted = _contract(
        acceptable_outcomes=("inconclusive", "counterexample"),
        input_hashes=(("right", _hash("d")), ("left", _hash("c"))),
        algorithms=(("zeta", "9"), ("direct-table-enumeration", "1.0")),
        assumptions=(("z", "last"), ("basis", "algmul.O.v1")),
        theorem_hypotheses=(("z", "last"), ("carrier", "finite")),
        bounds=(("memoryBytes", 1048576), ("cases", 512)),
        required_cross_checks=("z-check", "direct basis product"),
        expected_artifacts=("run-log", "witness-json"),
    )
    reordered_equivalent = _contract(
        algorithms=(("direct-table-enumeration", "1.0"), ("zeta", "9")),
        assumptions=(("basis", "algmul.O.v1"), ("z", "last")),
        theorem_hypotheses=(("carrier", "finite"), ("z", "last")),
        bounds=(("cases", 512), ("memoryBytes", 1048576)),
        required_cross_checks=("direct basis product", "z-check"),
        expected_artifacts=("witness-json", "run-log"),
    )
    assert permuted == reordered_equivalent
    assert permuted.semantic_hash == reordered_equivalent.semantic_hash
    assert first != permuted
    assert (
        _contract(
            acceptance_predicates=(
                "Check every declared basis triple.",
                "Record the lexicographically first nonzero witness.",
            )
        )
        == first
    )
    assert (
        _contract(input_hashes=(("left", _hash("d")), ("right", _hash("c")))) != first
    )
    repeated = _contract(input_hashes=(("left", _hash("c")), ("right", _hash("c"))))
    assert repeated.input_hashes == (("left", _hash("c")), ("right", _hash("c")))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("acceptable_outcomes", ("completed",)),
        ("input_hashes", (("left", _hash("a")), ("left", _hash("b")))),
        ("algorithms", (("a", "1"), ("a", "2"))),
        ("assumptions", (("status", "completed"),)),
        ("bounds", (("cases", -1),)),
        ("required_cross_checks", ("x", "x")),
        ("expected_artifacts", ("x", "x")),
        ("acceptance_predicates", ("x", "x")),
    ),
)
def test_closed_pre_execution_schema_rejects_status_promotion_and_duplicates(
    field: str, value: object
) -> None:
    with pytest.raises(ContractError):
        _contract(**{field: value})


@pytest.mark.parametrize(
    "key",
    (
        "evidence tier",
        "evidence.tier",
        "evidence/tier",
        "evidence-tier",
        "evidence_tier",
        "evidenceTier",
        "execution status",
        "executionStatus",
        "result/status",
        "mathematicalOutcome",
        "claim status",
        "claim_result",
        "proof_status",
        "verification_status",
        "execution_state",
        "evidence_level",
        "verification",
        "tier",
    ),
)
def test_promotion_key_normalization_closes_separator_and_camel_bypasses(
    key: str,
) -> None:
    with pytest.raises(ContractError, match="must not encode"):
        _contract(assumptions=((key, "descriptive setup"),))


@pytest.mark.parametrize("key", _AUDIT_ALIAS_SPELLINGS)
def test_token_aware_promotion_guard_rejects_composed_audit_aliases(key: str) -> None:
    with pytest.raises(ContractError, match="must not encode"):
        _contract(theorem_hypotheses=((key, "descriptive setup"),))


@pytest.mark.parametrize("key", _STRUCTURAL_SUFFIX_SPELLINGS)
def test_structural_axis_tokens_remain_reserved_with_neutral_suffixes(key: str) -> None:
    with pytest.raises(ContractError, match="must not encode"):
        _contract(bounds=((key, 1),))


@pytest.mark.parametrize("key", _COLLAPSED_MULTIWORD_STATUS_KEYS)
def test_collapsed_multiword_status_aliases_match_separated_and_camel_forms(
    key: str,
) -> None:
    with pytest.raises(ContractError, match="must not encode"):
        _contract(bounds=((key, 1),))
    words = contracts_module._semantic_tokens(key)
    if len(words) == 1:
        # The input was deliberately collapsed; construct known equivalent forms
        # from the expected audit partitions for an independent public check.
        equivalents = {
            "executionstatuscount": ("execution_status_count", "executionStatusCount"),
            "resultstatelimit": ("result_state_limit", "resultStateLimit"),
            "evidencetierlimit": ("evidence_tier_limit", "evidenceTierLimit"),
            "receiptstatuscount": ("receipt_status_count", "receiptStatusCount"),
            "prooflevelcount": ("proof_level_count", "proofLevelCount"),
            "runphasecount": ("run_phase_count", "runPhaseCount"),
        }[key]
        for equivalent in equivalents:
            with pytest.raises(ContractError, match="must not encode"):
                _contract(bounds=((equivalent, 1),))


def test_collapsed_segmentation_keeps_neutral_and_near_miss_keys_distinct() -> None:
    assert contracts_module._segment_known_remainder("statuscount") == (
        "status",
        "count",
    )
    assert contracts_module._segment_known_remainder("statuscountx") is None
    assert contracts_module._segment_known_remainder("") is None
    assert contracts_module._is_reserved_clause_key("resultants") is False
    assert contracts_module._is_reserved_clause_key("claimant") is False
    contract = _contract(
        bounds=(
            ("executiontimeseconds", 1),
            ("resultsizebytes", 1),
            ("outcomecountlimit", 1),
            ("verificationsamplecount", 1),
        )
    )
    assert len(contract.bounds) == 4


@pytest.mark.parametrize(
    "token",
    (
        "not_run",
        "running",
        "completed",
        "failed",
        "blocked",
        "E0",
        "E1-executed",
        "E2-regression",
        "E3-bounded-exact",
        "E4-cross-checked",
        "E5-proof-linked",
        "E6-reproduced",
        "constructed",
        "verified_within_domain",
        "counterexample",
        "no_go_within_assumptions",
        "inconclusive",
        "not_applicable",
        "reproduction_only",
        "implementation_error",
    ),
)
def test_promotion_status_values_cannot_be_smuggled_into_generic_clauses(
    token: str,
) -> None:
    with pytest.raises(ContractError, match="must not encode"):
        _contract(assumptions=(("context", token),))
    assert _contract(acceptable_outcomes=("inconclusive",)).acceptable_outcomes == (
        "inconclusive",
    )


def test_descriptive_clause_prose_and_nonstatus_algorithm_names_remain_valid() -> None:
    neutral_bound_spellings = tuple(
        spelling
        for words in (
            ("execution", "time", "seconds"),
            ("execution", "memory", "bytes"),
            ("result", "size", "bytes"),
            ("outcome", "count", "limit"),
            ("verification", "sample", "count"),
        )
        for spelling in _alias_spellings(words)
    )
    contract = _contract(
        assumptions=(("context", "calculation completed under declared conditions"),),
        algorithms=(
            ("execution-checker", "completed"),
            ("evidence-level-analyzer", "E6"),
        ),
        theorem_hypotheses=(
            ("proof_method", "hand-auditable derivation"),
            ("resultants", "elimination polynomials"),
            ("claimant", "research group"),
        ),
        bounds=tuple((key, 1) for key in neutral_bound_spellings),
    )
    assert contract.algorithms == (
        ("evidence-level-analyzer", "E6"),
        ("execution-checker", "completed"),
    )
    assert {key for key, _ in contract.bounds} == set(neutral_bound_spellings)


def test_optional_collections_pin_explicit_empty_lists_and_required_ones_do_not() -> (
    None
):
    contract = _contract(
        input_hashes=(),
        source_anchors=(),
        convention_manifests=(),
        assumptions=(),
        theorem_hypotheses=(),
        bounds=(),
        required_cross_checks=(),
        expected_artifacts=(),
    )
    record = calculation_contract_record(contract)
    for field in (
        "inputHashes",
        "sourceAnchors",
        "conventionManifests",
        "assumptions",
        "theoremHypotheses",
        "bounds",
        "requiredCrossChecks",
        "expectedArtifacts",
    ):
        assert record[field] == []
    assert contract_record_registry().from_record(record) == contract
    for field in ("acceptable_outcomes", "algorithms", "acceptance_predicates"):
        with pytest.raises(ContractError, match="must not be empty"):
            _contract(**{field: ()})
    for field in ("acceptableOutcomes", "algorithms", "acceptancePredicates"):
        malformed = dict(record)
        malformed[field] = []
        with pytest.raises(SchemaError, match="parser_failed"):
            contract_record_registry().from_record(malformed)


def test_exact_256_257_boundaries_for_every_iterable_family() -> None:
    outcomes = ("constructed",) * 256
    with pytest.raises(ContractError, match="duplicate"):
        _contract(acceptable_outcomes=outcomes)
    cases = {
        "input_hashes": tuple(
            (f"input-{index}", _hash(f"{index:064x}"[0])) for index in range(257)
        ),
        "source_anchors": tuple(_source(locator=f"p. {index}") for index in range(257)),
        "convention_manifests": tuple(
            _manifest(manifest_id=f"m.{index}") for index in range(257)
        ),
        "algorithms": tuple((f"a{index}", "1") for index in range(257)),
        "assumptions": tuple((f"a{index}", "v") for index in range(257)),
        "theorem_hypotheses": tuple((f"h{index}", "v") for index in range(257)),
        "bounds": tuple((f"b{index}", index) for index in range(257)),
        "required_cross_checks": tuple(f"c{index}" for index in range(257)),
        "expected_artifacts": tuple(f"a{index}" for index in range(257)),
        "acceptance_predicates": tuple(f"p{index}" for index in range(257)),
    }
    for field, value in cases.items():
        with pytest.raises(ContractError, match="limit"):
            _contract(**{field: value})


def test_source_and_manifest_references_are_snapshotted_before_later_iterables() -> (
    None
):
    source = _source()
    manifest = _manifest()
    source_reference = source.semantic_hash
    manifest_reference = manifest.semantic_hash

    def late_algorithms() -> object:
        object.__setattr__(source, "locator", "tampered")
        object.__setattr__(manifest, "indexing", "tampered")
        yield ("direct-table-enumeration", "1.0")

    contract = _contract(
        source_anchors=(source,),
        convention_manifests=(manifest,),
        input_hashes=(("input", _hash("c")),),
        algorithms=late_algorithms(),
    )
    assert contract.source_anchors == (source_reference,)
    assert contract.convention_manifests == (manifest_reference,)


def test_strict_parser_rejects_extra_missing_and_unsafe_records() -> None:
    record = calculation_contract_record(_contract())
    cases = (
        {**record, "extra": "no"},
        {key: value for key, value in record.items() if key != "projectId"},
        {**record, "schemaType": "os.system"},
        {**record, "schemaVersion": 2},
        {
            **record,
            "backend": {
                "request": "x",
                "name": "y",
                "version": "z",
                "callable": "os.system",
            },
        },
    )
    for bad in cases:
        with pytest.raises(SchemaError, match=r"parser_failed|unknown_schema"):
            contract_record_registry().from_record(bad)


def test_factory_immutability_structural_hash_and_safe_repr() -> None:
    contract = _contract()
    assert contract == _contract()
    assert hash(contract) == hash(_contract())
    assert "octonion" not in repr(contract)
    with pytest.raises(ContractError, match="factory-owned"):
        CalculationContract()
    with pytest.raises(AttributeError):
        contract.question = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        type("Child", (CalculationContract,), {})


def test_record_bytes_are_the_hash_preimage() -> None:
    contract = _contract()
    assert (
        hashlib.sha256(contract.canonical_bytes()).hexdigest()
        == contract.semantic_hash.digest
    )
    assert (
        calculation_contract_record(contract)["schemaType"]
        == "anyalgebra.evidence.calculation_contract"
    )
    assert calculation_contract_record(contract)["schemaVersion"] == 1


def test_typed_diagnostics_cover_text_iterable_hash_and_pair_failures() -> None:
    class BadIterable:
        def __iter__(self) -> object:
            raise RuntimeError("secret")

    def late_failure() -> object:
        yield "first"
        raise RuntimeError("secret")

    for value in (None, "", "x" * 4097, "bad\ud800", " padded "):
        with pytest.raises(ContractError):
            contracts_module._text(value, "field", identifier=True)
    with pytest.raises(ContractError, match="iterable"):
        contracts_module._items("not-an-iterable-declaration", "field")
    with pytest.raises(ContractError, match="iterable"):
        contracts_module._items(BadIterable(), "field")
    with pytest.raises(ContractError, match="snapshot"):
        contracts_module._items(late_failure(), "field")
    with pytest.raises(ContractError, match="empty"):
        contracts_module._nonempty_items((), "field")
    with pytest.raises(ContractError, match="SemanticHash"):
        contracts_module._hash("sha256:bad", "field")
    with pytest.raises(ContractError, match="duplicate"):
        contracts_module._hashes((_hash("a"), _hash("a")), "field")
    with pytest.raises(ContractError, match="duplicate input roles"):
        contracts_module._input_hashes((("x", _hash("a")), ("x", _hash("b"))))
    for inputs in (("not-a-pair",), (("role",),)):
        with pytest.raises(ContractError, match="exact pair"):
            contracts_module._input_hashes(inputs)
    for declarations in (("not-a-pair",), (("a",),), (("a", object()),)):
        with pytest.raises(ContractError):
            contracts_module._pairs(declarations, "field")
    assert contracts_module._concept("_") == ""


def test_reference_integrity_and_exact_type_guards_reject_adversarial_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_canonicalizer(_: object) -> bytes:
        return "not-bytes"  # type: ignore[return-value]

    with pytest.raises(ContractError, match="exact SourceAnchor"):
        contracts_module._reference(
            object(),
            expected_type=SourceAnchor,
            canonicalizer=source_anchor_canonical_bytes,
            field="source_anchors",
        )
    with pytest.raises(ContractError, match="integrity"):
        contracts_module._reference(
            _source(),
            expected_type=SourceAnchor,
            canonicalizer=invalid_canonicalizer,
            field="source_anchors",
        )
    source = _source()
    object.__setattr__(source, "locator", "tampered")
    with pytest.raises(ContractError, match="integrity"):
        _contract(source_anchors=(source,))
    source = _source()
    object.__setattr__(source, "semantic_hash", "not-a-hash")
    with pytest.raises(ContractError, match="integrity"):
        _contract(source_anchors=(source,))
    with pytest.raises(ContractError, match="duplicate"):
        _contract(source_anchors=(_source(), _source()))
    with pytest.raises(ContractError, match="duplicate"):
        _contract(convention_manifests=(_manifest(), _manifest()))

    class SyntheticReference:
        semantic_hash = SemanticHash("sha256", hashlib.sha256(b"synthetic").hexdigest())

    with pytest.raises(ContractError, match="integrity"):
        contracts_module._reference(
            SyntheticReference(),
            expected_type=SyntheticReference,
            canonicalizer=lambda _: b"synthetic",
            field="synthetic",
        )

    class WrongRegistry:
        def from_record(self, record: object) -> ConventionManifest:
            del record
            return _manifest()

    monkeypatch.setattr(contracts_module, "EVIDENCE_RECORD_REGISTRY", WrongRegistry())
    with pytest.raises(ContractError, match="integrity"):
        contracts_module._reference(
            _source(),
            expected_type=SourceAnchor,
            canonicalizer=source_anchor_canonical_bytes,
            field="source_anchors",
        )

    class DriftRegistry:
        def from_record(self, record: object) -> SourceAnchor:
            del record
            return _source(locator="p. 14")

    monkeypatch.setattr(contracts_module, "EVIDENCE_RECORD_REGISTRY", DriftRegistry())
    with pytest.raises(ContractError, match="integrity"):
        contracts_module._reference(
            _source(),
            expected_type=SourceAnchor,
            canonicalizer=source_anchor_canonical_bytes,
            field="source_anchors",
        )


def test_references_must_survive_the_authoritative_evidence_registry() -> None:
    source = _source()
    parsed_source = EVIDENCE_RECORD_REGISTRY.from_record(source_anchor_record(source))
    assert type(parsed_source) is SourceAnchor
    assert parsed_source == source
    assert parsed_source.canonical_bytes() == source.canonical_bytes()
    manifest = _manifest()
    parsed_manifest = EVIDENCE_RECORD_REGISTRY.from_record(
        convention_manifest_record(manifest)
    )
    assert type(parsed_manifest) is ConventionManifest
    assert parsed_manifest == manifest
    assert parsed_manifest.canonical_bytes() == manifest.canonical_bytes()

    object.__setattr__(source.content_hash, "algorithm", "sha1")
    object.__setattr__(
        source,
        "semantic_hash",
        SemanticHash(
            "sha256", hashlib.sha256(source_anchor_canonical_bytes(source)).hexdigest()
        ),
    )
    with pytest.raises(ContractError, match="integrity"):
        _contract(source_anchors=(source,))

    object.__setattr__(manifest, "status", "verified")
    object.__setattr__(
        manifest,
        "semantic_hash",
        SemanticHash("sha256", hashlib.sha256(manifest.canonical_bytes()).hexdigest()),
    )
    with pytest.raises(ContractError, match="integrity"):
        _contract(convention_manifests=(manifest,))

    malformed_source = _source()
    object.__setattr__(malformed_source, "content_hash", object())
    with pytest.raises(ContractError, match="integrity"):
        _contract(source_anchors=(malformed_source,))
    malformed_manifest = _manifest()
    object.__setattr__(malformed_manifest, "sources", (object(),))
    with pytest.raises(ContractError, match="integrity"):
        _contract(convention_manifests=(malformed_manifest,))


def test_exact_256_is_accepted_for_each_unbounded_iterable_family() -> None:
    hashes = tuple(
        (f"input-{index}", SemanticHash("sha256", f"{index:064x}"))
        for index in range(256)
    )
    sources = tuple(_source(locator=f"p.{index}") for index in range(256))
    manifests = tuple(_manifest(manifest_id=f"m.{index}") for index in range(256))
    cases = {
        "input_hashes": hashes,
        "source_anchors": sources,
        "convention_manifests": manifests,
        "algorithms": tuple((f"algorithm-{index}", "1") for index in range(256)),
        "assumptions": tuple((f"assumption-{index}", "x") for index in range(256)),
        "theorem_hypotheses": tuple(
            (f"hypothesis-{index}", "x") for index in range(256)
        ),
        "bounds": tuple((f"bound-{index}", index) for index in range(256)),
        "required_cross_checks": tuple(f"check-{index}" for index in range(256)),
        "expected_artifacts": tuple(f"artifact-{index}" for index in range(256)),
        "acceptance_predicates": tuple(f"predicate-{index}" for index in range(256)),
    }
    for field, value in cases.items():
        contract = _contract(**{field: value})
        assert len(getattr(contract, field)) == 256
    with pytest.raises(ContractError, match="resource limit"):
        _contract(
            input_hashes=hashes,
            source_anchors=sources,
            convention_manifests=manifests,
            algorithms=cases["algorithms"],
            assumptions=cases["assumptions"],
            theorem_hypotheses=cases["theorem_hypotheses"],
            bounds=cases["bounds"],
            required_cross_checks=cases["required_cross_checks"],
            expected_artifacts=cases["expected_artifacts"],
            acceptance_predicates=cases["acceptance_predicates"],
        )


@pytest.mark.parametrize(
    ("field", "values"),
    (
        ("expected_artifacts", tuple(_large_text(index) for index in range(256))),
        ("acceptance_predicates", tuple(_large_text(index) for index in range(256))),
        (
            "algorithms",
            tuple((f"algorithm-{index}", _large_text(index)) for index in range(256)),
        ),
    ),
)
def test_large_canonical_records_raise_only_sanitized_contract_errors(
    field: str, values: object
) -> None:
    with pytest.raises(ContractError, match="canonical record resource limit"):
        _contract(**{field: values})
    # The exact 255-item boundary is below canonical_json's one-MiB output cap.
    if field == "expected_artifacts":
        assert (
            len(
                _contract(
                    **{field: tuple(_large_text(index) for index in range(255))}
                ).canonical_bytes()
            )
            < 1_048_576
        )


def test_parser_and_factory_share_the_contract_aggregate_budget() -> None:
    def hash_record(index: int) -> dict[str, str]:
        return {"algorithm": "sha256", "digest": f"{index:064x}"}

    record = calculation_contract_record(_contract())
    record["inputHashes"] = [
        {"role": f"input-{index}", "hash": hash_record(index)} for index in range(128)
    ]
    record["sourceAnchors"] = [hash_record(index) for index in range(256)]
    record["conventionManifests"] = [hash_record(index) for index in range(256)]
    record["algorithms"] = [
        {"key": f"algorithm-{index}", "value": "1"} for index in range(256)
    ]
    record["assumptions"] = [
        {"key": f"assumption-{index}", "value": "x"} for index in range(256)
    ]
    record["theoremHypotheses"] = [
        {"key": f"hypothesis-{index}", "value": "x"} for index in range(32)
    ]
    with pytest.raises(SchemaError, match="parser_failed"):
        contract_record_registry().from_record(record)
    with pytest.raises(ContractError, match="combined record resource limit"):
        _contract(
            input_hashes=tuple(
                (f"input-{index}", _hash(f"{index:064x}"[0])) for index in range(128)
            ),
            source_anchors=tuple(_source(locator=f"p.{index}") for index in range(256)),
            convention_manifests=tuple(
                _manifest(manifest_id=f"m.{index}") for index in range(256)
            ),
            algorithms=tuple((f"algorithm-{index}", "1") for index in range(256)),
            assumptions=tuple((f"assumption-{index}", "x") for index in range(256)),
            theorem_hypotheses=tuple(
                (f"hypothesis-{index}", "x") for index in range(32)
            ),
        )


def test_parser_private_boundaries_and_mutation_detection_are_closed() -> None:
    with pytest.raises(ContractError):
        contracts_module._parse_hash({"algorithm": "sha256", "digest": 1}, "hash")
    with pytest.raises(ContractError):
        contracts_module._parse_hash({"algorithm": "sha256", "digest": "bad"}, "hash")
    assert contracts_module._parse_hashes((), "hashes") == ()
    assert contracts_module._parse_input_hashes(()) == ()
    assert contracts_module._parse_pairs((), "pairs") == ()
    assert contracts_module._parse_strings((), "strings", ordered=False) == ()
    with pytest.raises(ContractError, match="record shape"):
        contracts_module._parse_hashes([], "hashes")
    with pytest.raises(ContractError, match="record shape"):
        contracts_module._parse_input_hashes([])
    with pytest.raises(ContractError, match="record shape"):
        contracts_module._parse_pairs([], "pairs")
    with pytest.raises(ContractError, match="record shape"):
        contracts_module._parse_strings([], "strings", ordered=False)
    with pytest.raises(ContractError):
        calculation_contract_record(object())
    with pytest.raises(ContractError):
        calculation_contract_canonical_bytes(object())
    with pytest.raises(ContractError, match="factory requires"):
        CalculationContract.create.__func__(  # type: ignore[attr-defined]
            object,
            **{
                "contract_name": "x",
                "project_id": "x",
                "question": "x",
                "acceptable_outcomes": ("constructed",),
                "input_hashes": (("input", _hash("a")),),
                "source_anchors": (_source(),),
                "convention_manifests": (_manifest(),),
                "algorithms": (("a", "1"),),
                "backend_request": "x",
                "backend_name": "x",
                "backend_version": "x",
                "assumptions": (("a", "x"),),
                "theorem_hypotheses": (("a", "x"),),
                "bounds": (("a", 0),),
                "required_cross_checks": ("x",),
                "expected_artifacts": ("x",),
                "acceptance_predicates": ("x",),
            },
        )
    valid = _contract()
    assert valid.__eq__(object()) is False
    stale = _contract()
    object.__setattr__(stale, "question", "tampered")
    assert stale != valid
    with pytest.raises(ContractError, match="content address"):
        hash(stale)
    with pytest.raises(ContractError, match="content address"):
        stale.to_record()
    with pytest.raises(ContractError, match="content address"):
        stale.canonical_bytes()
    with pytest.raises(ContractError, match="content address"):
        _ = stale.contract_id
    fabricated_hash = _contract()
    object.__setattr__(
        fabricated_hash, "semantic_hash", SemanticHash("sha256", "f" * 64)
    )
    with pytest.raises(ContractError, match="content address"):
        _ = fabricated_hash.contract_id
    missing_hash = _contract()
    object.__setattr__(missing_hash, "semantic_hash", "bad")
    with pytest.raises(ContractError, match="integrity"):
        hash(missing_hash)
    with pytest.raises(ContractError, match="integrity"):
        _ = missing_hash.contract_id
    deleted_hash = _contract()
    object.__delattr__(deleted_hash, "semantic_hash")
    with pytest.raises(ContractError, match="integrity"):
        hash(deleted_hash)
    with pytest.raises(ContractError, match="integrity"):
        _ = deleted_hash.contract_id
