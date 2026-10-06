"""Append-only, evidence-gated public claim records.

This module deliberately does *not* run calculations or alter a receipt.  Its
only authority is to turn already sealed receipts into the next, narrowly
justified claim state.  Replay and dependency invalidation are V00-077 work;
the links captured here are intentionally sufficient for that later graph.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import SemanticHash
from anyalgebra.evidence.contracts import (
    CalculationContract,
    ContractError,
    calculation_contract_canonical_bytes,
    calculation_contract_record,
    contract_record_registry,
)
from anyalgebra.evidence.models import (
    EVIDENCE_RECORD_REGISTRY,
    EvidenceModelError,
    SourceAnchor,
    source_anchor_canonical_bytes,
    source_anchor_record,
)
from anyalgebra.evidence.run import (
    ReceiptError,
    ResultReceipt,
    result_receipt_canonical_bytes,
    result_receipt_record,
    result_receipt_registry,
)
from anyalgebra.persistence.canonical import canonical_json
from anyalgebra.persistence.registry import JSONValue, SchemaError, SerializerRegistry

_TAG, _VERSION, _MAX_ITEMS, _MAX_TEXT = "anyalgebra.evidence.claim_record", 1, 256, 4096
_STATES = (
    "E0-unimplemented",
    "E1-executed",
    "E2-regression",
    "E3-bounded-exact",
    "E4-cross-checked",
    "E5-proof-linked",
    "E6-reproduced",
)
_BADGES = {
    "E0-unimplemented": "unimplemented",
    "E1-executed": "implemented",
    "E2-regression": "regression-tested",
    "E5-proof-linked": "proof-linked",
    "E6-reproduced": "independently reproduced",
}
_SCOPED_OUTCOMES = {
    "counterexample": "counterexample found",
    "no_go_within_assumptions": "no-go under assumptions",
    "inconclusive": "inconclusive to bounds",
    "reproduction_only": "reproduction only",
    "not_applicable": "not applicable",
}
_FORBIDDEN = re.compile(r"\b(proved|confirmed|complete|ruled(?:[^\w]|_)+out)\b", re.I)


class ClaimError(AnyAlgebraError, ValueError):
    """Sanitized claim-boundary error that never displays attacker input."""

    def __init__(self, field: str, reason: str) -> None:
        self.field, self.reason = field, reason
        super().__init__(f"invalid claim record {field}: {reason}")


def _text(value: object, field: str, *, identifier: bool = False) -> str:
    if type(value) is not str or not value:
        raise ClaimError(field, "must be a nonempty exact built-in str")
    if len(value) > _MAX_TEXT or any("\ud800" <= char <= "\udfff" for char in value):
        raise ClaimError(field, "text limit or Unicode scalar violation")
    if identifier and value != value.strip():
        raise ClaimError(field, "must not have leading or trailing whitespace")
    return value


def _items(value: object, field: str) -> tuple[object, ...]:
    if type(value) is str:
        raise ClaimError(field, "must be an iterable")
    try:
        iterator = iter(cast(Iterable[object], value))
    except Exception as error:
        raise ClaimError(field, "must be an iterable") from error
    result: list[object] = []
    try:
        for _ in range(_MAX_ITEMS + 1):
            result.append(next(iterator))
    except StopIteration:
        return tuple(result)
    except Exception as error:
        raise ClaimError(field, "iterable snapshot failed") from error
    raise ClaimError(field, "item limit exceeded")


def _hash(value: object, field: str) -> SemanticHash:
    if type(value) is not SemanticHash:
        raise ClaimError(field, "must be an exact SemanticHash")
    return SemanticHash(value.algorithm, value.digest)


def _hashes(value: object, field: str) -> tuple[SemanticHash, ...]:
    result = tuple(_hash(item, field) for item in _items(value, field))
    if len(set(result)) != len(result):
        raise ClaimError(field, "contains duplicate content addresses")
    return tuple(sorted(result, key=str))


def _strings(value: object, field: str) -> tuple[str, ...]:
    result = tuple(_text(item, field, identifier=True) for item in _items(value, field))
    if len(set(result)) != len(result):
        raise ClaimError(field, "contains duplicate declarations")
    return tuple(sorted(result))


def _hash_record(value: SemanticHash) -> dict[str, str]:
    return {"algorithm": value.algorithm, "digest": value.digest}


def _contract(value: object) -> CalculationContract:
    if type(value) is not CalculationContract:
        raise ClaimError("contract", "must be an exact CalculationContract")
    try:
        parsed = contract_record_registry().from_record(
            calculation_contract_record(value)
        )
        if type(parsed) is not CalculationContract or (
            calculation_contract_canonical_bytes(parsed)
            != calculation_contract_canonical_bytes(value)
        ):
            raise ClaimError("contract", "content address integrity rejected")
        return parsed
    except (ContractError, SchemaError, TypeError, ValueError, AttributeError) as error:
        raise ClaimError("contract", "content address integrity rejected") from error


def _anchor(value: object) -> SourceAnchor:
    if type(value) is not SourceAnchor:
        raise ClaimError("source_anchors", "must contain exact SourceAnchor values")
    try:
        parsed = EVIDENCE_RECORD_REGISTRY.from_record(source_anchor_record(value))
        if (
            type(parsed) is not SourceAnchor
            or (
                source_anchor_canonical_bytes(parsed)
                != source_anchor_canonical_bytes(value)
            )
            or parsed.semantic_hash != value.semantic_hash
        ):
            raise ClaimError("source_anchors", "content address integrity rejected")
        return parsed
    except (
        EvidenceModelError,
        SchemaError,
        TypeError,
        ValueError,
        AttributeError,
    ) as error:
        raise ClaimError(
            "source_anchors", "content address integrity rejected"
        ) from error


def _anchors(value: object) -> tuple[SourceAnchor, ...]:
    result = tuple(_anchor(item) for item in _items(value, "source_anchors"))
    if not result:
        raise ClaimError("source_anchors", "must not be empty")
    if len({item.semantic_hash for item in result}) != len(result):
        raise ClaimError("source_anchors", "contains duplicate content addresses")
    return tuple(sorted(result, key=lambda item: str(item.semantic_hash)))


def _receipt(value: object) -> ResultReceipt:
    if type(value) is not ResultReceipt:
        raise ClaimError(
            "supporting_receipts", "must contain exact ResultReceipt values"
        )
    try:
        parsed = result_receipt_registry().from_record(result_receipt_record(value))
        if type(parsed) is not ResultReceipt or (
            result_receipt_canonical_bytes(parsed)
            != result_receipt_canonical_bytes(value)
        ):
            raise ClaimError(
                "supporting_receipts", "content address integrity rejected"
            )
        return parsed
    except (ReceiptError, SchemaError, TypeError, ValueError, AttributeError) as error:
        raise ClaimError(
            "supporting_receipts", "content address integrity rejected"
        ) from error


def _receipts(value: object) -> tuple[ResultReceipt, ...]:
    result = tuple(_receipt(item) for item in _items(value, "supporting_receipts"))
    if len({item.semantic_hash for item in result}) != len(result):
        raise ClaimError("supporting_receipts", "contains duplicate content addresses")
    return tuple(sorted(result, key=lambda item: str(item.semantic_hash)))


def _claim_state(value: object) -> str:
    result = _text(value, "evidence_state", identifier=True)
    if result not in _STATES:
        raise ClaimError("evidence_state", "is not a closed claim evidence state")
    return result


def _state_index(value: str) -> int:
    return _STATES.index(value)


def _badge(state: str, contract: CalculationContract) -> str:
    """Derive E3/E4's exact public domain label from immutable contract data."""
    if state == "E3-bounded-exact":
        prefix = "verified on "
    elif state == "E4-cross-checked":
        prefix = "independently cross-checked on "
    else:
        return _BADGES[state]
    cases = dict(contract.bounds).get("cases")
    return prefix + (
        f"cases={cases}" if type(cases) is int else f"contract={contract.contract_id}"
    )


def _wording(
    value: object,
    state: str,
    outcomes: tuple[str, ...],
    contract: CalculationContract,
) -> str:
    wording = _text(value, "allowed_public_wording")
    if not _badge_prefix(wording, _badge(state, contract)):
        raise ClaimError("allowed_public_wording", "omits the exact state badge")
    if _state_index(state) <= 4 and _FORBIDDEN.search(wording):
        raise ClaimError("allowed_public_wording", "uses unsupported claim language")
    current_index = _state_index(state)
    if any(
        _bounded_phrase(
            wording,
            "verified on"
            if candidate == "E3-bounded-exact"
            else (
                "independently cross-checked on"
                if candidate == "E4-cross-checked"
                else _badge(candidate, contract)
            ),
        )
        for candidate in _STATES[current_index + 1 :]
    ):
        raise ClaimError(
            "allowed_public_wording", "uses a stronger evidence-tier badge"
        )
    scoped = tuple(
        _SCOPED_OUTCOMES[item] for item in outcomes if item in _SCOPED_OUTCOMES
    )
    if any(
        _bounded_phrase(wording, phrase)
        for outcome, phrase in _SCOPED_OUTCOMES.items()
        if outcome not in outcomes
    ):
        raise ClaimError(
            "allowed_public_wording", "uses an outcome phrase absent from receipts"
        )
    if scoped and not all(_scoped_phrase(wording, item) for item in scoped):
        raise ClaimError(
            "allowed_public_wording", "negative outcome needs scoped wording"
        )
    return wording


def _badge_prefix(wording: str, badge: str) -> bool:
    """Accept one case-insensitive badge only as the leading structured label."""
    checked, required = wording.casefold(), badge.casefold()
    if checked == required:
        return True
    if not checked.startswith(required):
        return False
    remainder = checked[len(required) :]
    return bool(remainder) and remainder[0] in ":;,.—"


def _scoped_phrase(wording: str, phrase: str) -> bool:
    """Find an outcome phrase as a separate clause, never under a negation."""
    pattern = rf"(?:^|[;:,.—]\s*){re.escape(phrase.casefold())}(?=$|[;:,.—])"
    return re.search(pattern, wording.casefold()) is not None


def _bounded_phrase(wording: str, phrase: str) -> bool:
    """Find one known claim phrase anywhere without matching word extensions."""
    pattern = rf"(?<!\w){re.escape(phrase.casefold())}(?!\w)"
    return re.search(pattern, wording.casefold()) is not None


def _complete(receipt: ResultReceipt) -> bool:
    values = dict(receipt.case_counts)
    declared = dict(receipt.bounds).get("cases")
    return (
        values.get("casesExpected") is not None
        and values.get("casesExpected") == values.get("casesRun")
        and (
            values["casesExpected"] == declared
            if type(declared) is int
            else receipt.exact_certificate is not None
        )
    )


def _qualifies(
    receipts: tuple[ResultReceipt, ...], state: str, contract: CalculationContract
) -> None:
    if not receipts:
        raise ClaimError(
            "supporting_receipts", "promotion requires exact linked receipts"
        )
    if any(item.execution_status != "completed" for item in receipts):
        raise ClaimError(
            "supporting_receipts", "failed receipts cannot promote a claim"
        )
    if any(item.contract_hash != contract.semantic_hash for item in receipts):
        raise ClaimError("supporting_receipts", "receipt contract does not match claim")
    if any(
        item.mathematical_outcome not in contract.acceptable_outcomes
        for item in receipts
    ):
        raise ClaimError(
            "supporting_receipts", "receipt outcome is not accepted by claim contract"
        )
    target = _state_index(state)
    if any(_state_index(item.evidence_tier) < target for item in receipts):
        raise ClaimError("supporting_receipts", "receipt evidence tier is too weak")
    if target >= _state_index("E3-bounded-exact") and not any(
        item.mathematical_outcome
        in {
            "constructed",
            "verified_within_domain",
            "counterexample",
            "no_go_within_assumptions",
        }
        for item in receipts
    ):
        raise ClaimError(
            "supporting_receipts",
            "E3 and above need an independently correctness-bearing outcome",
        )
    if state == "E1-executed":
        return
    if state == "E2-regression" and not any(
        item.fixtures and item.test_artifacts for item in receipts
    ):
        raise ClaimError("supporting_receipts", "E2 requires fixtures and tests")
    if state == "E3-bounded-exact" and not any(
        _complete(item)
        and (item.exact_certificate is not None or item.backend_request == "exact")
        for item in receipts
    ):
        raise ClaimError(
            "supporting_receipts", "E3 requires exact complete domain counts"
        )
    if state == "E4-cross-checked" and not any(
        any(
            check.implementation_hash != item.primary_implementation_hash
            and check.result_hash == item.primary_result_hash
            for check in item.independent_checks
        )
        for item in receipts
    ):
        raise ClaimError(
            "supporting_receipts", "E4 requires a distinct agreeing implementation"
        )
    if state == "E5-proof-linked" and not any(
        item.proof_obligations or item.proof_certificate is not None
        for item in receipts
    ):
        raise ClaimError(
            "supporting_receipts", "E5 requires proof obligations or certificate"
        )
    if state == "E6-reproduced" and not any(
        item.reproduction_attestations for item in receipts
    ):
        raise ClaimError("supporting_receipts", "E6 requires independent reproduction")


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ClaimRecord:
    """A sealed project claim and the exact evidence currently linked to it."""

    project_id: str
    claim_key: str
    claim_text: str
    source_anchors: tuple[SourceAnchor, ...]
    contract: CalculationContract
    evidence_state: str
    allowed_public_wording: str
    supporting_receipts: tuple[ResultReceipt, ...]
    dependency_claim_hashes: tuple[SemanticHash, ...]
    dependency_receipt_hashes: tuple[SemanticHash, ...]
    prior_supporting_receipt_hashes: tuple[SemanticHash, ...]
    competing_hypothesis_ids: tuple[str, ...]
    unresolved_obligations: tuple[SemanticHash, ...]
    supersedes_claim_hash: SemanticHash | None
    supersedes_evidence_state: str | None
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ClaimError("claim_record", "is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("ClaimRecord cannot be subclassed")

    @property
    def claim_id(self) -> str:
        _ensure(self)
        return f"claim:sha256:{self.semantic_hash.digest}"

    @classmethod
    def create(
        cls,
        *,
        project_id: object,
        claim_key: object,
        claim_text: object,
        source_anchors: object,
        contract: object,
        allowed_public_wording: object = "unimplemented",
        dependency_claim_hashes: object = (),
        dependency_receipt_hashes: object = (),
        competing_hypothesis_ids: object = (),
        unresolved_obligations: object = (),
    ) -> ClaimRecord:
        """Create an E0 claim; later states are only created by promotion."""
        if cls is not ClaimRecord:
            raise ClaimError("claim_record", "factory requires exact ClaimRecord")
        return _build(
            project_id=project_id,
            claim_key=claim_key,
            claim_text=claim_text,
            source_anchors=source_anchors,
            contract=contract,
            evidence_state="E0-unimplemented",
            allowed_public_wording=allowed_public_wording,
            supporting_receipts=(),
            dependency_claim_hashes=dependency_claim_hashes,
            dependency_receipt_hashes=dependency_receipt_hashes,
            prior_supporting_receipt_hashes=(),
            competing_hypothesis_ids=competing_hypothesis_ids,
            unresolved_obligations=unresolved_obligations,
            supersedes_claim_hash=None,
            supersedes_evidence_state=None,
            promotion=False,
        )

    def to_record(self) -> dict[str, object]:
        _ensure(self)
        return claim_record(self)

    def canonical_bytes(self) -> bytes:
        _ensure(self)
        return claim_canonical_bytes(self)

    def __repr__(self) -> str:
        return f"ClaimRecord(claim_id={self.claim_id!r})"

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is ClaimRecord
            and self.canonical_bytes() == other.canonical_bytes()
        )

    def __hash__(self) -> int:
        _ensure(self)
        return int.from_bytes(
            bytes.fromhex(self.semantic_hash.digest[:16]), "big", signed=True
        )


def _build(
    *,
    project_id: object,
    claim_key: object,
    claim_text: object,
    source_anchors: object,
    contract: object,
    evidence_state: object,
    allowed_public_wording: object,
    supporting_receipts: object,
    dependency_claim_hashes: object,
    dependency_receipt_hashes: object,
    prior_supporting_receipt_hashes: object,
    competing_hypothesis_ids: object,
    unresolved_obligations: object,
    supersedes_claim_hash: object | None,
    supersedes_evidence_state: object | None,
    promotion: bool,
) -> ClaimRecord:
    checked_contract, checked_state = _contract(contract), _claim_state(evidence_state)
    checked_receipts = _receipts(supporting_receipts)
    checked_dependency_receipts = _hashes(
        dependency_receipt_hashes, "dependency_receipt_hashes"
    )
    checked_prior_support = _hashes(
        prior_supporting_receipt_hashes, "prior_supporting_receipt_hashes"
    )
    checked_anchors = _anchors(source_anchors)
    checked_project = _text(project_id, "project_id", identifier=True)
    if checked_project != checked_contract.project_id:
        raise ClaimError("project_id", "must match the embedded contract project")
    if not set(checked_contract.source_anchors) <= {
        item.semantic_hash for item in checked_anchors
    }:
        raise ClaimError(
            "source_anchors", "must include every embedded contract source anchor"
        )
    if checked_state == "E0-unimplemented":
        if (
            promotion
            or checked_receipts
            or supersedes_claim_hash is not None
            or supersedes_evidence_state is not None
            or checked_prior_support
        ):
            raise ClaimError("evidence_state", "E0 cannot carry promotion evidence")
    elif not promotion:
        raise ClaimError("evidence_state", "non-E0 state requires promotion")
    else:
        if supersedes_claim_hash is None or supersedes_evidence_state is None:
            raise ClaimError(
                "supersedes_claim_hash",
                "non-E0 claims require a superseded claim state",
            )
        checked_previous = _claim_state(supersedes_evidence_state)
        if _state_index(checked_previous) != _state_index(checked_state) - 1:
            raise ClaimError(
                "supersedes_evidence_state",
                "must be the immediately preceding evidence state",
            )
        if checked_state == "E1-executed" and checked_prior_support:
            raise ClaimError(
                "prior_supporting_receipt_hashes", "E1 cannot have prior support"
            )
        if checked_state != "E1-executed" and not checked_prior_support:
            raise ClaimError(
                "prior_supporting_receipt_hashes", "E2 and above require prior support"
            )
        if checked_state != "E1-executed" and not any(
            item.supersedes in checked_prior_support for item in checked_receipts
        ):
            raise ClaimError(
                "supporting_receipts",
                "E2 and above need a receipt supersession edge to prior support",
            )
        _qualifies(checked_receipts, checked_state, checked_contract)
    outcomes = tuple(sorted({item.mathematical_outcome for item in checked_receipts}))
    value = object.__new__(ClaimRecord)
    fields = (
        ("project_id", checked_project),
        ("claim_key", _text(claim_key, "claim_key", identifier=True)),
        ("claim_text", _text(claim_text, "claim_text")),
        ("source_anchors", checked_anchors),
        ("contract", checked_contract),
        ("evidence_state", checked_state),
        (
            "allowed_public_wording",
            _wording(allowed_public_wording, checked_state, outcomes, checked_contract),
        ),
        ("supporting_receipts", checked_receipts),
        (
            "dependency_claim_hashes",
            _hashes(dependency_claim_hashes, "dependency_claim_hashes"),
        ),
        (
            "dependency_receipt_hashes",
            checked_dependency_receipts,
        ),
        ("prior_supporting_receipt_hashes", checked_prior_support),
        (
            "competing_hypothesis_ids",
            _strings(competing_hypothesis_ids, "competing_hypothesis_ids"),
        ),
        (
            "unresolved_obligations",
            _hashes(unresolved_obligations, "unresolved_obligations"),
        ),
        (
            "supersedes_claim_hash",
            None
            if supersedes_claim_hash is None
            else _hash(supersedes_claim_hash, "supersedes_claim_hash"),
        ),
        (
            "supersedes_evidence_state",
            None
            if supersedes_evidence_state is None
            else _claim_state(supersedes_evidence_state),
        ),
        ("semantic_hash", SemanticHash("sha256", "0" * 64)),
    )
    for name, item in fields:
        object.__setattr__(value, name, item)
    object.__setattr__(value, "semantic_hash", _record_hash(value))
    return value


def promote_claim(
    claim: object,
    *,
    target_state: object,
    supporting_receipts: object,
    allowed_public_wording: object,
    dependency_claim_hashes: object | None = None,
    dependency_receipt_hashes: object | None = None,
    competing_hypothesis_ids: object | None = None,
    unresolved_obligations: object | None = None,
) -> ClaimRecord:
    """Append exactly one evidence state when sealed linked receipts justify it."""
    _ensure(claim)
    prior = cast(ClaimRecord, claim)
    target = _claim_state(target_state)
    if _state_index(target) != _state_index(prior.evidence_state) + 1:
        raise ClaimError("target_state", "must be the next evidence state")
    carried_receipts = set(prior.dependency_receipt_hashes) | {
        item.semantic_hash for item in prior.supporting_receipts
    }
    if dependency_receipt_hashes is not None:
        carried_receipts |= set(
            _hashes(dependency_receipt_hashes, "dependency_receipt_hashes")
        )
    carried_claims = set(prior.dependency_claim_hashes)
    if dependency_claim_hashes is not None:
        carried_claims |= set(
            _hashes(dependency_claim_hashes, "dependency_claim_hashes")
        )
    return _build(
        project_id=prior.project_id,
        claim_key=prior.claim_key,
        claim_text=prior.claim_text,
        source_anchors=prior.source_anchors,
        contract=prior.contract,
        evidence_state=target,
        allowed_public_wording=allowed_public_wording,
        supporting_receipts=supporting_receipts,
        dependency_claim_hashes=tuple(sorted(carried_claims, key=str)),
        dependency_receipt_hashes=tuple(sorted(carried_receipts, key=str)),
        prior_supporting_receipt_hashes=tuple(
            sorted((item.semantic_hash for item in prior.supporting_receipts), key=str)
        ),
        competing_hypothesis_ids=prior.competing_hypothesis_ids
        if competing_hypothesis_ids is None
        else competing_hypothesis_ids,
        unresolved_obligations=prior.unresolved_obligations
        if unresolved_obligations is None
        else unresolved_obligations,
        supersedes_claim_hash=prior.semantic_hash,
        supersedes_evidence_state=prior.evidence_state,
        promotion=True,
    )


def _anchor_record(value: SourceAnchor) -> dict[str, object]:
    return source_anchor_record(value)


def _body(value: ClaimRecord) -> dict[str, object]:
    return {
        "projectId": value.project_id,
        "claimKey": value.claim_key,
        "claimText": value.claim_text,
        "sourceAnchors": [_anchor_record(item) for item in value.source_anchors],
        "contractRecord": calculation_contract_record(value.contract),
        "contractHash": _hash_record(value.contract.semantic_hash),
        "evidenceState": value.evidence_state,
        "allowedPublicWording": value.allowed_public_wording,
        "supportingReceipts": [
            result_receipt_record(item) for item in value.supporting_receipts
        ],
        "supportingReceiptHashes": [
            _hash_record(item.semantic_hash) for item in value.supporting_receipts
        ],
        "dependencyClaimHashes": [
            _hash_record(item) for item in value.dependency_claim_hashes
        ],
        "dependencyReceiptHashes": [
            _hash_record(item) for item in value.dependency_receipt_hashes
        ],
        "priorSupportingReceiptHashes": [
            _hash_record(item) for item in value.prior_supporting_receipt_hashes
        ],
        "competingHypothesisIds": list(value.competing_hypothesis_ids),
        "unresolvedObligations": [
            _hash_record(item) for item in value.unresolved_obligations
        ],
        "supersedesClaimHash": None
        if value.supersedes_claim_hash is None
        else _hash_record(value.supersedes_claim_hash),
        "supersedesEvidenceState": value.supersedes_evidence_state,
    }


def _encode(value: ClaimRecord) -> dict[str, JSONValue]:
    return cast(dict[str, JSONValue], _body(value))


CLAIM_RECORD_REGISTRY = SerializerRegistry().with_codec(
    _TAG, _VERSION, ClaimRecord, _encode, lambda record: _parse(record)
)


def _record_hash(value: ClaimRecord) -> SemanticHash:
    try:
        return SemanticHash(
            "sha256",
            hashlib.sha256(
                canonical_json(value, registry=CLAIM_RECORD_REGISTRY)
            ).hexdigest(),
        )
    except (SchemaError, TypeError, ValueError, AttributeError) as error:
        raise ClaimError(
            "claim_record", "cannot derive canonical content address"
        ) from error


def _ensure(value: object) -> None:
    if type(value) is not ClaimRecord:
        raise ClaimError("claim_record", "must be an exact ClaimRecord")
    try:
        if _hash(value.semantic_hash, "semantic_hash") != _record_hash(value):
            raise ClaimError("claim_record", "content address integrity rejected")
    except (ClaimError, SchemaError, TypeError, ValueError, AttributeError) as error:
        if type(error) is ClaimError:
            raise
        raise ClaimError(
            "claim_record", "content address integrity rejected"
        ) from error


def claim_record(value: object) -> dict[str, object]:
    _ensure(value)
    return cast(dict[str, object], CLAIM_RECORD_REGISTRY.to_record(value))


def claim_canonical_bytes(value: object) -> bytes:
    _ensure(value)
    return canonical_json(value, registry=CLAIM_RECORD_REGISTRY)


def claim_record_registry() -> SerializerRegistry:
    return CLAIM_RECORD_REGISTRY


def _mapping(value: object, keys: frozenset[str], field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ClaimError(field, "has an invalid record shape")
    return value


def _mutable(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _mutable(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_mutable(item) for item in value]
    return value


def _parse_hash(value: object, field: str) -> SemanticHash:
    fields = _mapping(value, frozenset(("algorithm", "digest")), field)
    if type(fields["algorithm"]) is not str or type(fields["digest"]) is not str:
        raise ClaimError(field, "has an invalid semantic hash")
    try:
        return SemanticHash(fields["algorithm"], fields["digest"])
    except Exception as error:
        raise ClaimError(field, "has an invalid semantic hash") from error


def _tuple(value: object, field: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise ClaimError(field, "has an invalid record shape")
    return _items(value, field)


def _parse_contract(value: object) -> CalculationContract:
    try:
        parsed = contract_record_registry().from_record(
            cast(dict[str, JSONValue], _mutable(value))
        )
    except (SchemaError, TypeError, ValueError) as error:
        raise ClaimError(
            "contract_record", "is not an exact registered contract"
        ) from error
    return _contract(parsed)


def _parse_anchor(value: object) -> SourceAnchor:
    try:
        parsed = EVIDENCE_RECORD_REGISTRY.from_record(
            cast(dict[str, JSONValue], _mutable(value))
        )
    except (SchemaError, TypeError, ValueError) as error:
        raise ClaimError(
            "source_anchors", "is not an exact registered source"
        ) from error
    return _anchor(parsed)


def _parse_receipt(value: object) -> ResultReceipt:
    try:
        parsed = result_receipt_registry().from_record(
            cast(dict[str, JSONValue], _mutable(value))
        )
    except (SchemaError, TypeError, ValueError) as error:
        raise ClaimError(
            "supporting_receipts", "is not an exact registered receipt"
        ) from error
    return _receipt(parsed)


def _keys() -> frozenset[str]:
    return frozenset(("schemaType", "schemaVersion", *(_body_keys())))


def _body_keys() -> tuple[str, ...]:
    return (
        "projectId",
        "claimKey",
        "claimText",
        "sourceAnchors",
        "contractRecord",
        "contractHash",
        "evidenceState",
        "allowedPublicWording",
        "supportingReceipts",
        "supportingReceiptHashes",
        "dependencyClaimHashes",
        "dependencyReceiptHashes",
        "priorSupportingReceiptHashes",
        "competingHypothesisIds",
        "unresolvedObligations",
        "supersedesClaimHash",
        "supersedesEvidenceState",
    )


def _parse(record: object) -> ClaimRecord:
    fields = _mapping(record, _keys(), "claim_record")
    if fields["schemaType"] != _TAG or fields["schemaVersion"] != _VERSION:
        raise ClaimError("claim_record", "has an invalid schema identity")
    contract = _parse_contract(fields["contractRecord"])
    if _mutable(fields["contractHash"]) != _hash_record(contract.semantic_hash):
        raise ClaimError("contract_hash", "does not match embedded contract")
    anchors = tuple(
        _parse_anchor(item)
        for item in _tuple(fields["sourceAnchors"], "source_anchors")
    )
    receipts = tuple(
        _parse_receipt(item)
        for item in _tuple(fields["supportingReceipts"], "supporting_receipts")
    )
    if _mutable(fields["supportingReceiptHashes"]) != [
        _hash_record(item.semantic_hash) for item in receipts
    ]:
        raise ClaimError("supporting_receipt_hashes", "do not match embedded receipts")
    return _build(
        project_id=fields["projectId"],
        claim_key=fields["claimKey"],
        claim_text=fields["claimText"],
        source_anchors=anchors,
        contract=contract,
        evidence_state=fields["evidenceState"],
        allowed_public_wording=fields["allowedPublicWording"],
        supporting_receipts=receipts,
        dependency_claim_hashes=tuple(
            _parse_hash(item, "dependency_claim_hashes")
            for item in _tuple(
                fields["dependencyClaimHashes"], "dependency_claim_hashes"
            )
        ),
        dependency_receipt_hashes=tuple(
            _parse_hash(item, "dependency_receipt_hashes")
            for item in _tuple(
                fields["dependencyReceiptHashes"], "dependency_receipt_hashes"
            )
        ),
        prior_supporting_receipt_hashes=tuple(
            _parse_hash(item, "prior_supporting_receipt_hashes")
            for item in _tuple(
                fields["priorSupportingReceiptHashes"],
                "prior_supporting_receipt_hashes",
            )
        ),
        competing_hypothesis_ids=_tuple(
            fields["competingHypothesisIds"], "competing_hypothesis_ids"
        ),
        unresolved_obligations=tuple(
            _parse_hash(item, "unresolved_obligations")
            for item in _tuple(
                fields["unresolvedObligations"], "unresolved_obligations"
            )
        ),
        supersedes_claim_hash=None
        if fields["supersedesClaimHash"] is None
        else _parse_hash(fields["supersedesClaimHash"], "supersedes_claim_hash"),
        supersedes_evidence_state=fields["supersedesEvidenceState"],
        promotion=fields["evidenceState"] != "E0-unimplemented",
    )
