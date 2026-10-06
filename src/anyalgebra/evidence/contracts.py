"""Immutable, content-addressed pre-execution calculation contracts.

Only the question to be asked and the conditions under which a future run may
be accepted live here.  Execution status, receipts, claim language, evidence
tiers, replay, and promotion deliberately have no representation in this v1
schema.  This keeps a contract from asserting that its own future calculation
has succeeded.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import cast

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import SemanticHash
from anyalgebra.evidence.models import (
    ConventionManifest,
    EVIDENCE_RECORD_REGISTRY,
    EvidenceModelError,
    SourceAnchor,
    convention_manifest_record,
    convention_manifest_canonical_bytes,
    source_anchor_record,
    source_anchor_canonical_bytes,
)
from anyalgebra.persistence.canonical import canonical_json
from anyalgebra.persistence.registry import JSONValue, SchemaError, SerializerRegistry


_TAG = "anyalgebra.evidence.calculation_contract"
_VERSION = 1
_MAX_ITEMS = 256
_MAX_TEXT_CODE_POINTS = 4_096
# This conservative contract-local budget stays below SerializerRegistry's
# 4,096-node hard boundary, so factories and parsers fail through the same
# typed contract diagnostic rather than leaking a lower-level resource error.
_MAX_RECORD_NODES = 3_800
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
_PROHIBITED_CLAUSE_CONCEPTS = frozenset(
    (
        "claim",
        "claim_status",
        "claim_result",
        "evidence",
        "evidence_tier",
        "evidence_level",
        "execution",
        "execution_status",
        "execution_state",
        "receipt",
        "result",
        "result_status",
        "status",
        "outcome",
        "mathematical_outcome",
        "verified",
        "verification_status",
        "promotion",
        "proof_status",
        "tier",
    )
)
_PROHIBITED_STATUS_TOKENS = frozenset(
    (
        "not_run",
        "running",
        "completed",
        "failed",
        "blocked",
        "e0",
        "e1",
        "e2",
        "e3",
        "e4",
        "e5",
        "e6",
        "e0_unimplemented",
        "e1_executed",
        "e2_regression",
        "e3_bounded_exact",
        "e4_cross_checked",
        "e5_proof_linked",
        "e6_reproduced",
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
_ALWAYS_RESERVED_CLAUSE_TOKENS = frozenset(
    (
        "claim",
        "evidence",
        "receipt",
        "status",
        "verified",
        "promotion",
        "tier",
        "conclusion",
    )
)
_STATUS_SUBJECT_TOKENS = frozenset(
    (
        "execution",
        "result",
        "outcome",
        "verification",
        "proof",
        "run",
        "scientific",
        "mathematical",
        "is",
    )
)
_CONTEXTUAL_RESOURCE_SUBJECT_TOKENS = frozenset(
    ("execution", "result", "outcome", "verification")
)
_NEUTRAL_RESOURCE_QUALIFIER_TOKENS = frozenset(
    (
        "time",
        "memory",
        "size",
        "count",
        "limit",
        "sample",
        "samples",
        "bytes",
        "cases",
        "attempts",
        "length",
    )
)
_STATUS_QUALIFIER_TOKENS = frozenset(
    (
        "status",
        "state",
        "tier",
        "level",
        "grade",
        "class",
        "phase",
        "outcome",
        "result",
        "conclusion",
        "verified",
        "linked",
    )
)


class ContractError(AnyAlgebraError, ValueError):
    """A calculation contract cannot be safely frozen into the v1 schema."""

    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid calculation contract {field}: {reason}")


def _text(value: object, field: str, *, identifier: bool = False) -> str:
    """Accept a bounded exact Unicode string without rendering hostile input."""
    if type(value) is not str:
        raise ContractError(field, "must be an exact built-in str")
    if not value:
        raise ContractError(field, "must not be empty")
    if len(value) > _MAX_TEXT_CODE_POINTS:
        raise ContractError(field, "text limit exceeded")
    if any("\ud800" <= character <= "\udfff" for character in value):
        raise ContractError(field, "contains an invalid Unicode scalar")
    if identifier and value != value.strip():
        raise ContractError(field, "must not have leading or trailing whitespace")
    return value


def _semantic_tokens(value: str) -> tuple[str, ...]:
    """Split identifier-like text at separators and camel/Pascal boundaries."""
    tokens: list[str] = []
    current: list[str] = []
    for index, character in enumerate(value):
        if "A" <= character <= "Z":
            next_is_lower = index + 1 < len(value) and value[index + 1].islower()
            if current and (
                current[-1].islower() or (current[-1].isupper() and next_is_lower)
            ):
                tokens.append("".join(current))
                current = []
            current.append(character.lower())
        elif "a" <= character <= "z" or "0" <= character <= "9":
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tuple(tokens)


def _concept(value: str) -> str:
    """Return a collapsed fallback for legacy unseparated identifier spellings."""
    return "".join(_semantic_tokens(value))


_PROHIBITED_CLAUSE_KEY_CONCEPTS = frozenset(
    _concept(item) for item in _PROHIBITED_CLAUSE_CONCEPTS
)
_PROHIBITED_STATUS_CONCEPTS = frozenset(
    _concept(item) for item in _PROHIBITED_STATUS_TOKENS
)
_KNOWN_REMAINDER_TOKENS = _STATUS_QUALIFIER_TOKENS | _NEUTRAL_RESOURCE_QUALIFIER_TOKENS
_SEGMENT_SUBJECTS = tuple(
    sorted(
        _ALWAYS_RESERVED_CLAUSE_TOKENS | _STATUS_SUBJECT_TOKENS,
        key=lambda item: (-len(item), item),
    )
)


def _segment_known_remainder(value: str) -> tuple[str, ...] | None:
    """Segment a bounded collapsed suffix using only the fixed v1 vocabulary."""
    if not value or len(value) > _MAX_TEXT_CODE_POINTS:
        return None
    options = tuple(
        sorted(_KNOWN_REMAINDER_TOKENS, key=lambda item: (-len(item), item))
    )
    paths: list[tuple[str, ...] | None] = [None] * (len(value) + 1)
    paths[0] = ()
    for start, path in enumerate(paths):
        if path is None:
            continue
        for token in options:
            end = start + len(token)
            if value[start:end] == token and paths[end] is None:
                paths[end] = (*path, token)
    return paths[-1]


def _is_reserved_collapsed_concept(concept: str) -> bool:
    """Classify legacy unseparated status/resource aliases without recursion."""
    if concept in _PROHIBITED_CLAUSE_KEY_CONCEPTS:
        return True
    for subject in _SEGMENT_SUBJECTS:
        if not concept.startswith(subject):
            continue
        segments = _segment_known_remainder(concept[len(subject) :])
        if segments is None:
            continue
        if subject in _ALWAYS_RESERVED_CLAUSE_TOKENS:
            return True
        if any(segment in _STATUS_QUALIFIER_TOKENS for segment in segments):
            return True
    return False


def _is_reserved_clause_key(key: str) -> bool:
    """Reject semantic status/evidence aliases without matching near-miss words."""
    tokens = frozenset(_semantic_tokens(key))
    if tokens & _ALWAYS_RESERVED_CLAUSE_TOKENS:
        return True
    concept = _concept(key)
    if _is_reserved_collapsed_concept(concept):
        return True
    if len(tokens) == 1 and tokens & _CONTEXTUAL_RESOURCE_SUBJECT_TOKENS:
        return True
    subjects = tokens & _STATUS_SUBJECT_TOKENS
    qualifiers = tokens & _STATUS_QUALIFIER_TOKENS
    return any(subject != qualifier for subject in subjects for qualifier in qualifiers)


def _items(value: object, field: str) -> tuple[object, ...]:
    """Snapshot one iterable exactly once, including a deterministic 257th check."""
    if type(value) is str:
        raise ContractError(field, "must be an iterable")
    try:
        iterator = iter(cast(Iterable[object], value))
    except Exception as error:
        raise ContractError(field, "must be an iterable") from error
    snapshot: list[object] = []
    try:
        for _ in range(_MAX_ITEMS + 1):
            snapshot.append(next(iterator))
    except StopIteration:
        return tuple(snapshot)
    except Exception as error:
        raise ContractError(field, "iterable snapshot failed") from error
    raise ContractError(field, "declaration limit exceeded")


def _nonempty_items(value: object, field: str) -> tuple[object, ...]:
    """Require one bounded nonempty semantic collection."""
    result = _items(value, field)
    if not result:
        raise ContractError(field, "must not be empty")
    return result


def _hash(value: object, field: str) -> SemanticHash:
    """Require exactly the package's canonical semantic hash value object."""
    if type(value) is not SemanticHash:
        raise ContractError(field, "must be an exact SemanticHash")
    return SemanticHash(value.algorithm, value.digest)


def _hash_record(value: SemanticHash) -> dict[str, str]:
    """Encode one already frozen content address without display aliases."""
    return {"algorithm": value.algorithm, "digest": value.digest}


def _hashes(
    value: object, field: str, *, required: bool = False
) -> tuple[SemanticHash, ...]:
    """Canonicalize an unordered, duplicate-free nonempty hash-reference set."""
    items = _nonempty_items(value, field) if required else _items(value, field)
    hashes = tuple(_hash(item, field) for item in items)
    if len(set(hashes)) != len(hashes):
        raise ContractError(field, "contains duplicate content addresses")
    return tuple(sorted(hashes, key=str))


def _input_hashes(value: object) -> tuple[tuple[str, SemanticHash], ...]:
    """Freeze unordered named input bindings while retaining semantic roles.

    A hash by itself cannot distinguish a left operand from a right operand.
    Roles are therefore part of the content address and, unlike a bare set of
    hashes, permit one immutable input artifact to fill multiple named roles.
    """
    bindings: list[tuple[str, SemanticHash]] = []
    roles: set[str] = set()
    for item in _items(value, "input_hashes"):
        if type(item) not in (tuple, list):
            raise ContractError("input_hashes", "each input must be an exact pair")
        pair = cast(tuple[object, ...] | list[object], item)
        if len(pair) != 2:
            raise ContractError("input_hashes", "each input must be an exact pair")
        role = _text(pair[0], "input_hashes", identifier=True)
        if role in roles:
            raise ContractError("input_hashes", "contains duplicate input roles")
        roles.add(role)
        bindings.append((role, _hash(pair[1], "input_hashes")))
    return tuple(sorted(bindings, key=lambda binding: binding[0]))


def _reference(
    item: object,
    *,
    expected_type: type[object],
    canonicalizer: Callable[[object], bytes],
    field: str,
) -> SemanticHash:
    """Snapshot only an exact, registry-valid source or manifest identity."""
    if type(item) is not expected_type:
        raise ContractError(
            field, f"must contain exact {expected_type.__name__} values"
        )
    try:
        declared = item.semantic_hash  # type: ignore[attr-defined]
        if type(declared) is not SemanticHash:
            raise ContractError(field, "reference has no exact semantic hash")
        bytes_for_item = canonicalizer(item)
        if type(bytes_for_item) is not bytes:
            raise ContractError(field, "reference canonical bytes are invalid")
        calculated = SemanticHash("sha256", hashlib.sha256(bytes_for_item).hexdigest())
        if calculated != declared:
            raise ContractError(
                field, "reference content address does not match record"
            )
        # The evidence registry is the authoritative schema boundary for V00-073
        # records.  Do not admit a merely hashable object whose mutated fields
        # would be rejected as an invalid source or convention elsewhere.
        if expected_type is SourceAnchor:
            record = source_anchor_record(item)
        elif expected_type is ConventionManifest:
            record = convention_manifest_record(item)
        else:
            raise ContractError(field, "unsupported reference type")
        parsed = EVIDENCE_RECORD_REGISTRY.from_record(record)
        if type(parsed) is not expected_type:
            raise ContractError(field, "reference registry type mismatch")
        parsed_bytes = canonicalizer(parsed)
        if parsed != item or parsed_bytes != bytes_for_item:
            raise ContractError(field, "reference registry round trip is not exact")
    except (
        AttributeError,
        EvidenceModelError,
        SchemaError,
        TypeError,
        ValueError,
    ) as error:
        raise ContractError(field, "reference integrity rejected") from error
    return SemanticHash(calculated.algorithm, calculated.digest)


def _references(
    value: object,
    *,
    expected_type: type[object],
    canonicalizer: Callable[[object], bytes],
    field: str,
) -> tuple[SemanticHash, ...]:
    """Freeze a set of independently revalidated source or convention IDs."""
    references = tuple(
        _reference(
            item,
            expected_type=expected_type,
            canonicalizer=canonicalizer,
            field=field,
        )
        for item in _items(value, field)
    )
    if len(set(references)) != len(references):
        raise ContractError(field, "contains duplicate content addresses")
    return tuple(sorted(references, key=str))


def _strings(
    value: object,
    field: str,
    *,
    ordered: bool,
    required: bool = False,
    allowed: frozenset[str] | None = None,
) -> tuple[str, ...]:
    """Freeze one string collection, retaining order only when it is semantic."""
    items = _nonempty_items(value, field) if required else _items(value, field)
    strings = tuple(_text(item, field) for item in items)
    if allowed is not None and any(item not in allowed for item in strings):
        raise ContractError(field, "contains an unsupported outcome kind")
    if len(set(strings)) != len(strings):
        raise ContractError(field, "contains duplicate declarations")
    return strings if ordered else tuple(sorted(strings))


def _pairs(
    value: object,
    field: str,
    *,
    bound_values: bool = False,
    required: bool = False,
    guard_promotion: bool = True,
) -> tuple[tuple[str, str | int], ...]:
    """Freeze unordered named clauses with exact keys and values.

    The only non-string payload permitted in v1 is a nonnegative exact integer
    bound.  This is enough for enumeration/resource limits without admitting a
    general untyped configuration language into the evidence boundary.
    """
    pairs: list[tuple[str, str | int]] = []
    keys: set[str] = set()
    items = _nonempty_items(value, field) if required else _items(value, field)
    for item in items:
        if type(item) not in (tuple, list):
            raise ContractError(field, "each declaration must be an exact pair")
        pair = cast(tuple[object, ...] | list[object], item)
        if len(pair) != 2:
            raise ContractError(field, "each declaration must be an exact pair")
        key = _text(pair[0], field, identifier=True)
        if guard_promotion and _is_reserved_clause_key(key):
            raise ContractError(field, "must not encode execution or evidence status")
        if key in keys:
            raise ContractError(field, "contains duplicate keys")
        raw = pair[1]
        if bound_values and type(raw) is int and raw >= 0:
            checked: str | int = raw
        elif type(raw) is str:
            checked = _text(raw, field)
            if guard_promotion and _concept(checked) in _PROHIBITED_STATUS_CONCEPTS:
                raise ContractError(
                    field, "must not encode execution, evidence, or result status"
                )
        else:
            raise ContractError(field, "has an invalid declaration value")
        keys.add(key)
        pairs.append((key, checked))
    return tuple(sorted(pairs, key=lambda pair: pair[0]))


def _algorithms(value: object) -> tuple[tuple[str, str], ...]:
    """Freeze unordered named algorithm/version requirements by algorithm name."""
    values = _pairs(value, "algorithms", required=True, guard_promotion=False)
    return tuple((key, cast(str, item)) for key, item in values)


def _clauses_record(
    value: tuple[tuple[str, str | int], ...],
) -> list[dict[str, str | int]]:
    """Encode sorted clauses as explicit JSON key/value records."""
    return [{"key": key, "value": item} for key, item in value]


def _check_record_budget(
    input_hashes: tuple[tuple[str, SemanticHash], ...],
    source_anchors: tuple[SemanticHash, ...],
    convention_manifests: tuple[SemanticHash, ...],
    algorithms: tuple[tuple[str, str], ...],
    assumptions: tuple[tuple[str, str | int], ...],
    theorem_hypotheses: tuple[tuple[str, str | int], ...],
    bounds: tuple[tuple[str, str | int], ...],
    required_cross_checks: tuple[str, ...],
    expected_artifacts: tuple[str, ...],
    acceptance_predicates: tuple[str, ...],
) -> None:
    """Keep a combined v1 record below the established 4,096-node codec cap."""
    three_node_values = (
        len(source_anchors)
        + len(convention_manifests)
        + len(algorithms)
        + len(assumptions)
        + len(theorem_hypotheses)
        + len(bounds)
    )
    estimated_nodes = (
        64
        + 5 * len(input_hashes)
        + 3 * three_node_values
        + len(required_cross_checks)
        + len(expected_artifacts)
        + len(acceptance_predicates)
    )
    if estimated_nodes > _MAX_RECORD_NODES:
        raise ContractError(
            "calculation_contract", "combined record resource limit exceeded"
        )


def _contract_body(
    contract_name: str,
    project_id: str,
    question: str,
    acceptable_outcomes: tuple[str, ...],
    input_hashes: tuple[tuple[str, SemanticHash], ...],
    source_anchors: tuple[SemanticHash, ...],
    convention_manifests: tuple[SemanticHash, ...],
    algorithms: tuple[tuple[str, str], ...],
    backend_request: str,
    backend_name: str,
    backend_version: str,
    assumptions: tuple[tuple[str, str | int], ...],
    theorem_hypotheses: tuple[tuple[str, str | int], ...],
    bounds: tuple[tuple[str, str | int], ...],
    required_cross_checks: tuple[str, ...],
    expected_artifacts: tuple[str, ...],
    acceptance_predicates: tuple[str, ...],
) -> dict[str, object]:
    """Build every v1 semantic field; collection order is never semantic here."""
    return {
        "acceptableOutcomes": list(acceptable_outcomes),
        "acceptancePredicates": list(acceptance_predicates),
        "algorithms": _clauses_record(algorithms),
        "assumptions": _clauses_record(assumptions),
        "backend": {
            "name": backend_name,
            "request": backend_request,
            "version": backend_version,
        },
        "bounds": _clauses_record(bounds),
        "conventionManifests": [_hash_record(item) for item in convention_manifests],
        "expectedArtifacts": list(expected_artifacts),
        "contractName": contract_name,
        "inputHashes": [
            {"hash": _hash_record(content_hash), "role": role}
            for role, content_hash in input_hashes
        ],
        "projectId": project_id,
        "question": question,
        "requiredCrossChecks": list(required_cross_checks),
        "sourceAnchors": [_hash_record(item) for item in source_anchors],
        "theoremHypotheses": _clauses_record(theorem_hypotheses),
    }


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class CalculationContract:
    """A sealed, content-addressed specification that exists before execution."""

    contract_name: str
    project_id: str
    question: str
    acceptable_outcomes: tuple[str, ...]
    input_hashes: tuple[tuple[str, SemanticHash], ...]
    source_anchors: tuple[SemanticHash, ...]
    convention_manifests: tuple[SemanticHash, ...]
    algorithms: tuple[tuple[str, str], ...]
    backend_request: str
    backend_name: str
    backend_version: str
    assumptions: tuple[tuple[str, str | int], ...]
    theorem_hypotheses: tuple[tuple[str, str | int], ...]
    bounds: tuple[tuple[str, str | int], ...]
    required_cross_checks: tuple[str, ...]
    expected_artifacts: tuple[str, ...]
    acceptance_predicates: tuple[str, ...]
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ContractError("calculation_contract", "is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("CalculationContract cannot be subclassed")

    @classmethod
    def create(
        cls,
        *,
        contract_name: object,
        project_id: object,
        question: object,
        acceptable_outcomes: object,
        input_hashes: object,
        source_anchors: object,
        convention_manifests: object,
        algorithms: object,
        backend_request: object,
        backend_name: object,
        backend_version: object,
        assumptions: object,
        theorem_hypotheses: object,
        bounds: object,
        required_cross_checks: object,
        expected_artifacts: object,
        acceptance_predicates: object,
    ) -> CalculationContract:
        """Validate and freeze all semantic inputs before deriving one digest."""
        if cls is not CalculationContract:
            raise ContractError(
                "calculation_contract", "factory requires exact CalculationContract"
            )
        # Snapshot externally owned evidence references before any later iterable
        # can mutate them; store only their independently revalidated hashes.
        checked_sources = _references(
            source_anchors,
            expected_type=SourceAnchor,
            canonicalizer=source_anchor_canonical_bytes,
            field="source_anchors",
        )
        checked_manifests = _references(
            convention_manifests,
            expected_type=ConventionManifest,
            canonicalizer=convention_manifest_canonical_bytes,
            field="convention_manifests",
        )
        checked_inputs = _input_hashes(input_hashes)
        checked_algorithms = _algorithms(algorithms)
        checked_assumptions = _pairs(assumptions, "assumptions")
        checked_hypotheses = _pairs(theorem_hypotheses, "theorem_hypotheses")
        checked_bounds = _pairs(bounds, "bounds", bound_values=True)
        checked_checks = _strings(
            required_cross_checks, "required_cross_checks", ordered=False
        )
        checked_artifacts = _strings(
            expected_artifacts, "expected_artifacts", ordered=False
        )
        checked_predicates = _strings(
            acceptance_predicates,
            "acceptance_predicates",
            ordered=False,
            required=True,
        )
        return _instance(
            _text(contract_name, "contract_name", identifier=True),
            _text(project_id, "project_id", identifier=True),
            _text(question, "question"),
            _strings(
                acceptable_outcomes,
                "acceptable_outcomes",
                ordered=False,
                required=True,
                allowed=_OUTCOMES,
            ),
            checked_inputs,
            checked_sources,
            checked_manifests,
            checked_algorithms,
            _text(backend_request, "backend_request", identifier=True),
            _text(backend_name, "backend_name", identifier=True),
            _text(backend_version, "backend_version", identifier=True),
            checked_assumptions,
            checked_hypotheses,
            checked_bounds,
            checked_checks,
            checked_artifacts,
            checked_predicates,
        )

    def to_record(self) -> dict[str, object]:
        """Return a fresh exact v1 record for safe registry serialization."""
        return calculation_contract_record(self)

    def canonical_bytes(self) -> bytes:
        """Return the sole canonical byte preimage of ``semantic_hash``."""
        return calculation_contract_canonical_bytes(self)

    @property
    def contract_id(self) -> str:
        """Return the immutable content address used by receipts and dependencies."""
        _assert_integrity(self)
        return f"contract:sha256:{self.semantic_hash.digest}"

    def __repr__(self) -> str:
        """Avoid rendering question, assumptions, source locations, or artifacts."""
        return f"CalculationContract(semantic_hash={str(self.semantic_hash)!r})"

    def __eq__(self, other: object) -> bool:
        """Use complete structural content equality, never object identity."""
        if type(other) is not CalculationContract:
            return False
        try:
            _assert_integrity(self)
            _assert_integrity(other)
        except ContractError:
            return False
        return self._content() == other._content()

    def __hash__(self) -> int:
        """Return a deterministic signed 64-bit prefix of canonical SHA-256."""
        _assert_integrity(self)
        return int.from_bytes(
            bytes.fromhex(self.semantic_hash.digest[:16]), "big", signed=True
        )

    def _content(self) -> tuple[object, ...]:
        """Expose complete immutable defining fields internally for equality only."""
        return (
            self.contract_name,
            self.project_id,
            self.question,
            self.acceptable_outcomes,
            self.input_hashes,
            self.source_anchors,
            self.convention_manifests,
            self.algorithms,
            self.backend_request,
            self.backend_name,
            self.backend_version,
            self.assumptions,
            self.theorem_hypotheses,
            self.bounds,
            self.required_cross_checks,
            self.expected_artifacts,
            self.acceptance_predicates,
            self.semantic_hash,
        )


def _instance(
    contract_name: str,
    project_id: str,
    question: str,
    acceptable_outcomes: tuple[str, ...],
    input_hashes: tuple[tuple[str, SemanticHash], ...],
    source_anchors: tuple[SemanticHash, ...],
    convention_manifests: tuple[SemanticHash, ...],
    algorithms: tuple[tuple[str, str], ...],
    backend_request: str,
    backend_name: str,
    backend_version: str,
    assumptions: tuple[tuple[str, str | int], ...],
    theorem_hypotheses: tuple[tuple[str, str | int], ...],
    bounds: tuple[tuple[str, str | int], ...],
    required_cross_checks: tuple[str, ...],
    expected_artifacts: tuple[str, ...],
    acceptance_predicates: tuple[str, ...],
) -> CalculationContract:
    """Construct an already validated factory/parser-owned sealed record."""
    _check_record_budget(
        input_hashes,
        source_anchors,
        convention_manifests,
        algorithms,
        assumptions,
        theorem_hypotheses,
        bounds,
        required_cross_checks,
        expected_artifacts,
        acceptance_predicates,
    )
    value = object.__new__(CalculationContract)
    for name, item in (
        ("contract_name", contract_name),
        ("project_id", project_id),
        ("question", question),
        ("acceptable_outcomes", acceptable_outcomes),
        ("input_hashes", input_hashes),
        ("source_anchors", source_anchors),
        ("convention_manifests", convention_manifests),
        ("algorithms", algorithms),
        ("backend_request", backend_request),
        ("backend_name", backend_name),
        ("backend_version", backend_version),
        ("assumptions", assumptions),
        ("theorem_hypotheses", theorem_hypotheses),
        ("bounds", bounds),
        ("required_cross_checks", required_cross_checks),
        ("expected_artifacts", expected_artifacts),
        ("acceptance_predicates", acceptance_predicates),
        ("semantic_hash", SemanticHash("sha256", "0" * 64)),
    ):
        object.__setattr__(value, name, item)
    try:
        bytes_for_value = _canonical_bytes_unchecked(value)
    except SchemaError as error:
        raise ContractError(
            "calculation_contract", "canonical record resource limit exceeded"
        ) from error
    object.__setattr__(
        value,
        "semantic_hash",
        SemanticHash("sha256", hashlib.sha256(bytes_for_value).hexdigest()),
    )
    return value


def _record_unchecked(value: CalculationContract) -> dict[str, object]:
    """Encode factory-owned fields while a new value's hash is still unset."""
    return {
        "schemaType": _TAG,
        "schemaVersion": _VERSION,
        **_contract_body(
            value.contract_name,
            value.project_id,
            value.question,
            value.acceptable_outcomes,
            value.input_hashes,
            value.source_anchors,
            value.convention_manifests,
            value.algorithms,
            value.backend_request,
            value.backend_name,
            value.backend_version,
            value.assumptions,
            value.theorem_hypotheses,
            value.bounds,
            value.required_cross_checks,
            value.expected_artifacts,
            value.acceptance_predicates,
        ),
    }


def _assert_integrity(value: CalculationContract) -> None:
    """Reject illicit ``object.__setattr__`` changes before exposing a record."""
    try:
        declared = value.semantic_hash
        if type(declared) is not SemanticHash:
            raise ContractError("calculation_contract", "has no exact semantic hash")
        calculated = SemanticHash(
            "sha256", hashlib.sha256(_canonical_bytes_unchecked(value)).hexdigest()
        )
    except (AttributeError, SchemaError, TypeError, ValueError) as error:
        raise ContractError("calculation_contract", "integrity rejected") from error
    if calculated != declared:
        raise ContractError(
            "calculation_contract", "content address does not match record"
        )


def calculation_contract_record(value: object) -> dict[str, object]:
    """Encode an exact contract in the closed, no-result/no-status v1 schema."""
    if type(value) is not CalculationContract:
        raise ContractError(
            "calculation_contract", "must be an exact CalculationContract"
        )
    _assert_integrity(value)
    return _record_unchecked(value)


def _body(value: CalculationContract) -> dict[str, JSONValue]:
    """Remove registry-owned tag/version fields from the trusted direct encoder."""
    return {
        key: cast(JSONValue, item)
        for key, item in _record_unchecked(value).items()
        if key not in {"schemaType", "schemaVersion"}
    }


def _encode(value: CalculationContract) -> dict[str, JSONValue]:
    """Expose only the trusted CalculationContract encoder to this registry."""
    return _body(value)


CONTRACT_RECORD_REGISTRY = SerializerRegistry().with_codec(
    _TAG, _VERSION, CalculationContract, _encode, lambda record: _parse(record)
)


def _canonical_bytes_unchecked(value: CalculationContract) -> bytes:
    """Encode factory/parser-owned fields while deriving or checking a digest."""
    return canonical_json(value, registry=CONTRACT_RECORD_REGISTRY)


def calculation_contract_canonical_bytes(value: object) -> bytes:
    """Encode one exact contract through this closed allow-listed registry."""
    if type(value) is not CalculationContract:
        raise ContractError(
            "calculation_contract", "must be an exact CalculationContract"
        )
    _assert_integrity(value)
    return _canonical_bytes_unchecked(value)


def _mapping(value: object, fields: frozenset[str], field: str) -> Mapping[str, object]:
    """Require exactly the closed record shape after registry deep snapshotting."""
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ContractError(field, "has an invalid record shape")
    return value


def _parse_hash(value: object, field: str) -> SemanticHash:
    """Parse one exact semantic hash object without accepting display aliases."""
    fields = _mapping(value, frozenset(("algorithm", "digest")), field)
    if type(fields["algorithm"]) is not str or type(fields["digest"]) is not str:
        raise ContractError(field, "has an invalid semantic hash")
    try:
        return SemanticHash(fields["algorithm"], fields["digest"])
    except Exception as error:
        raise ContractError(field, "has an invalid semantic hash") from error


def _parse_hashes(value: object, field: str) -> tuple[SemanticHash, ...]:
    """Parse registry-snapshotted JSON arrays into canonical hash tuples."""
    if type(value) is not tuple:
        raise ContractError(field, "has an invalid record shape")
    return _hashes(tuple(_parse_hash(item, field) for item in value), field)


def _parse_input_hashes(value: object) -> tuple[tuple[str, SemanticHash], ...]:
    """Parse named input bindings without collapsing distinct operand roles."""
    if type(value) is not tuple:
        raise ContractError("input_hashes", "has an invalid record shape")
    bindings: list[tuple[object, object]] = []
    for item in value:
        fields = _mapping(item, frozenset(("role", "hash")), "input_hashes")
        bindings.append((fields["role"], _parse_hash(fields["hash"], "input_hashes")))
    return _input_hashes(tuple(bindings))


def _parse_pairs(
    value: object,
    field: str,
    *,
    bound_values: bool = False,
    required: bool = False,
) -> tuple[tuple[str, str | int], ...]:
    """Parse explicit JSON clauses with strict key/value shapes."""
    if type(value) is not tuple:
        raise ContractError(field, "has an invalid record shape")
    pairs: list[tuple[object, object]] = []
    for item in value:
        fields = _mapping(item, frozenset(("key", "value")), field)
        pairs.append((fields["key"], fields["value"]))
    return _pairs(tuple(pairs), field, bound_values=bound_values, required=required)


def _parse_strings(
    value: object,
    field: str,
    *,
    ordered: bool,
    required: bool = False,
    allowed: frozenset[str] | None = None,
) -> tuple[str, ...]:
    """Parse one exact JSON string tuple through normal factory validators."""
    if type(value) is not tuple:
        raise ContractError(field, "has an invalid record shape")
    return _strings(value, field, ordered=ordered, required=required, allowed=allowed)


def _parse(record: object) -> CalculationContract:
    """Strictly rebuild a contract with no imports, callback lookup, or execution."""
    fields = _mapping(
        record,
        frozenset(
            (
                "schemaType",
                "schemaVersion",
                "contractName",
                "projectId",
                "question",
                "acceptableOutcomes",
                "inputHashes",
                "sourceAnchors",
                "conventionManifests",
                "algorithms",
                "backend",
                "assumptions",
                "theoremHypotheses",
                "bounds",
                "requiredCrossChecks",
                "expectedArtifacts",
                "acceptancePredicates",
            )
        ),
        "calculation_contract",
    )
    backend = _mapping(
        fields["backend"], frozenset(("request", "name", "version")), "backend"
    )
    return _instance(
        _text(fields["contractName"], "contract_name", identifier=True),
        _text(fields["projectId"], "project_id", identifier=True),
        _text(fields["question"], "question"),
        _parse_strings(
            fields["acceptableOutcomes"],
            "acceptable_outcomes",
            ordered=False,
            required=True,
            allowed=_OUTCOMES,
        ),
        _parse_input_hashes(fields["inputHashes"]),
        _parse_hashes(fields["sourceAnchors"], "source_anchors"),
        _parse_hashes(fields["conventionManifests"], "convention_manifests"),
        cast(
            tuple[tuple[str, str], ...],
            _parse_pairs(fields["algorithms"], "algorithms", required=True),
        ),
        _text(backend["request"], "backend_request", identifier=True),
        _text(backend["name"], "backend_name", identifier=True),
        _text(backend["version"], "backend_version", identifier=True),
        _parse_pairs(fields["assumptions"], "assumptions"),
        _parse_pairs(fields["theoremHypotheses"], "theorem_hypotheses"),
        _parse_pairs(fields["bounds"], "bounds", bound_values=True),
        _parse_strings(
            fields["requiredCrossChecks"], "required_cross_checks", ordered=False
        ),
        _parse_strings(
            fields["expectedArtifacts"], "expected_artifacts", ordered=False
        ),
        _parse_strings(
            fields["acceptancePredicates"],
            "acceptance_predicates",
            ordered=False,
            required=True,
        ),
    )


def contract_record_registry() -> SerializerRegistry:
    """Return the immutable registry containing only CalculationContract v1."""
    return CONTRACT_RECORD_REGISTRY
