"""Read-only, direct replay comparison for sealed result receipts.

Replay does not execute code and it does not change a receipt or a claim.  A
caller supplies a closed, factory-built resolver which identifies the original
receipt, its rerun, and the directly referenced predecessor/dependency
receipts.  Transitive invalidation is deliberately deferred to V00-078.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import SemanticHash
from anyalgebra.evidence.run import (
    ReceiptError,
    ResultReceipt,
    result_receipt_canonical_bytes,
    result_receipt_record,
    result_receipt_registry,
)
from anyalgebra.persistence.canonical import canonical_json
from anyalgebra.persistence.registry import JSONValue, SchemaError, SerializerRegistry


_TAG, _VERSION, _MAX_ITEMS, _MAX_RESOLVER_ITEMS, _MAX_REPORT_ITEMS, _MAX_TEXT = (
    "anyalgebra.evidence.replay_report",
    1,
    256,
    1024,
    1024,
    4096,
)
_STATUSES = frozenset(("exact_match", "mismatch", "unavailable", "stale"))
_MATCHED_FIELDS = frozenset(
    ("contract", "primary_result_hash", "mathematical_outcome", "direct_dependencies")
)
_MISMATCH_FIELDS = frozenset(("primary_result_hash", "mathematical_outcome"))
_CONTRACT_STALE_FIELDS = frozenset(
    (
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
    )
)


class ReplayError(AnyAlgebraError, ValueError):
    """Sanitized replay-boundary diagnostic; no hostile value is rendered."""

    def __init__(self, field: str, reason: str) -> None:
        self.field, self.reason = field, reason
        super().__init__(f"invalid replay {field}: {reason}")


def _items(value: object, field: str) -> tuple[object, ...]:
    if type(value) is str:
        raise ReplayError(field, "must be an iterable")
    try:
        iterator = iter(cast(Iterable[object], value))
    except Exception as error:
        raise ReplayError(field, "must be an iterable") from error
    result: list[object] = []
    try:
        for _ in range(_MAX_ITEMS + 1):
            result.append(next(iterator))
    except StopIteration:
        return tuple(result)
    except Exception as error:
        raise ReplayError(field, "iterable snapshot failed") from error
    raise ReplayError(field, "item limit exceeded")


def _resolver_items(value: object, field: str) -> tuple[object, ...]:
    """Snapshot a resolver graph large enough for both 256-edge receipts."""
    if type(value) is str:
        raise ReplayError(field, "must be an iterable")
    try:
        iterator = iter(cast(Iterable[object], value))
    except Exception as error:
        raise ReplayError(field, "must be an iterable") from error
    result: list[object] = []
    try:
        for _ in range(_MAX_RESOLVER_ITEMS + 1):
            result.append(next(iterator))
    except StopIteration:
        return tuple(result)
    except Exception as error:
        raise ReplayError(field, "iterable snapshot failed") from error
    raise ReplayError(field, "resolver item limit exceeded")


def _text(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise ReplayError(field, "must be a nonempty exact built-in str")
    if len(value) > _MAX_TEXT or any("\ud800" <= char <= "\udfff" for char in value):
        raise ReplayError(field, "text limit or Unicode scalar violation")
    return value


def _hash(value: object, field: str) -> SemanticHash:
    if type(value) is not SemanticHash:
        raise ReplayError(field, "must be an exact SemanticHash")
    try:
        return SemanticHash(value.algorithm, value.digest)
    except Exception as error:
        raise ReplayError(field, "has an invalid semantic hash") from error


def _hashes(value: object, field: str) -> tuple[SemanticHash, ...]:
    result = tuple(_hash(item, field) for item in _items(value, field))
    if len(set(result)) != len(result):
        raise ReplayError(field, "contains duplicate content addresses")
    return tuple(sorted(result, key=str))


def _report_items(value: object, field: str) -> tuple[object, ...]:
    """Snapshot report unions up to their 1,024 direct-edge worst case."""
    if type(value) is str:
        raise ReplayError(field, "must be an iterable")
    try:
        iterator = iter(cast(Iterable[object], value))
    except Exception as error:
        raise ReplayError(field, "must be an iterable") from error
    result: list[object] = []
    try:
        for _ in range(_MAX_REPORT_ITEMS + 1):
            result.append(next(iterator))
    except StopIteration:
        return tuple(result)
    except Exception as error:
        raise ReplayError(field, "iterable snapshot failed") from error
    raise ReplayError(field, "report item limit exceeded")


def _report_hashes(value: object, field: str) -> tuple[SemanticHash, ...]:
    result = tuple(_hash(item, field) for item in _report_items(value, field))
    if len(set(result)) != len(result):
        raise ReplayError(field, "contains duplicate content addresses")
    return tuple(sorted(result, key=str))


def _report_strings(value: object, field: str) -> tuple[str, ...]:
    result = tuple(_text(item, field) for item in _report_items(value, field))
    if len(set(result)) != len(result):
        raise ReplayError(field, "contains duplicate declarations")
    return tuple(sorted(result))


def _hash_text(value: str) -> bool:
    algorithm, separator, digest = value.partition(":")
    if not separator:
        return False
    try:
        SemanticHash(algorithm, digest)
    except (TypeError, ValueError):
        return False
    return True


def _stale_edge(value: str) -> bool:
    if value in _CONTRACT_STALE_FIELDS:
        return True
    prefix, separator, remainder = value.partition(":")
    if prefix not in {"dependency_receipt", "supersedes"} or not separator:
        return False
    parts = remainder.split("->")
    return (
        len(parts) in {1, 2}
        and all(_hash_text(item) for item in parts)
        and (len(parts) == 1 or parts[0] != parts[1])
    )


def _strings(value: object, field: str) -> tuple[str, ...]:
    result = tuple(_text(item, field) for item in _items(value, field))
    if len(set(result)) != len(result):
        raise ReplayError(field, "contains duplicate declarations")
    return tuple(sorted(result))


def _receipt(value: object, field: str) -> ResultReceipt:
    if type(value) is not ResultReceipt:
        raise ReplayError(field, "must contain exact ResultReceipt values")
    try:
        record = result_receipt_record(value)
        parsed = result_receipt_registry().from_record(record)
        if type(parsed) is not ResultReceipt or result_receipt_canonical_bytes(
            parsed
        ) != result_receipt_canonical_bytes(value):
            raise ReplayError(field, "content address integrity rejected")
    except (ReceiptError, SchemaError, TypeError, ValueError, AttributeError) as error:
        raise ReplayError(field, "content address integrity rejected") from error
    return parsed


def _pairs(value: object, field: str) -> tuple[tuple[SemanticHash, SemanticHash], ...]:
    pairs: list[tuple[SemanticHash, SemanticHash]] = []
    for item in _resolver_items(value, field):
        if type(item) not in (tuple, list):
            raise ReplayError(field, "each entry must be an exact pair")
        pair = cast(tuple[object, object] | list[object], item)
        if len(pair) != 2:
            raise ReplayError(field, "each entry must be an exact pair")
        pairs.append((_hash(pair[0], field), _hash(pair[1], field)))
    if len({left for left, _ in pairs}) != len(pairs):
        raise ReplayError(field, "contains duplicate original references")
    return tuple(sorted(pairs, key=lambda pair: str(pair[0])))


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ReplayResolver:
    """Closed in-memory lookup used by :func:`verify_replay`.

    It accepts records only from trusted caller code.  It never imports a
    module, invokes an input-selected callable, or looks at ambient storage.
    """

    receipts: tuple[ResultReceipt, ...]
    reruns: tuple[tuple[SemanticHash, SemanticHash], ...]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ReplayError("resolver", "is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("ReplayResolver cannot be subclassed")

    @classmethod
    def create(cls, *, receipts: object, reruns: object = ()) -> ReplayResolver:
        if cls is not ReplayResolver:
            raise ReplayError("resolver", "factory requires exact ReplayResolver")
        checked = tuple(
            _receipt(item, "receipts") for item in _resolver_items(receipts, "receipts")
        )
        if len({item.semantic_hash for item in checked}) != len(checked):
            raise ReplayError("receipts", "contains duplicate content addresses")
        checked = tuple(sorted(checked, key=lambda item: str(item.semantic_hash)))
        hashes = {item.semantic_hash for item in checked}
        edges = _pairs(reruns, "reruns")
        if any(left == right for left, right in edges):
            raise ReplayError("reruns", "must identify a distinct rerun receipt")
        if any(left not in hashes or right not in hashes for left, right in edges):
            raise ReplayError("reruns", "must resolve through supplied receipts")
        if len({right for _, right in edges}) != len(edges):
            raise ReplayError("reruns", "contains converging rerun references")
        edge_map = dict(edges)
        for start in edge_map:
            cursor = start
            for _ in range(len(edge_map) + 1):  # pragma: no branch
                next_hash = edge_map.get(cursor)
                if next_hash is None:
                    break
                if next_hash == start:
                    raise ReplayError("reruns", "contains a rerun reference cycle")
                cursor = next_hash
        value = object.__new__(ReplayResolver)
        object.__setattr__(value, "receipts", checked)
        object.__setattr__(value, "reruns", edges)
        return value

    def __repr__(self) -> str:
        return (
            f"ReplayResolver(receipt_count={len(self.receipts)}, "
            f"rerun_count={len(self.reruns)})"
        )

    def _lookup(self, reference: SemanticHash) -> ResultReceipt | None:
        return next(
            (item for item in self.receipts if item.semantic_hash == reference), None
        )

    def _rerun_for(self, reference: SemanticHash) -> SemanticHash | None:
        return next((right for left, right in self.reruns if left == reference), None)


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ReplayReport:
    """Immutable content-addressed direct replay conclusion.

    ``status`` is operational evidence only.  It has no claim/evidence-tier
    field and cannot promote a :class:`ClaimRecord`.
    """

    status: str
    original_receipt_hash: SemanticHash
    rerun_receipt_hash: SemanticHash | None
    matched_fields: tuple[str, ...]
    mismatch_fields: tuple[str, ...]
    stale_edges: tuple[str, ...]
    unavailable_references: tuple[SemanticHash, ...]
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ReplayError("replay_report", "is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("ReplayReport cannot be subclassed")

    @property
    def report_id(self) -> str:
        _ensure(self)
        return f"replay:sha256:{self.semantic_hash.digest}"

    def to_record(self) -> dict[str, object]:
        _ensure(self)
        return replay_report_record(self)

    def canonical_bytes(self) -> bytes:
        _ensure(self)
        return replay_report_canonical_bytes(self)

    def __repr__(self) -> str:
        return f"ReplayReport(report_id={self.report_id!r}, status={self.status!r})"

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is ReplayReport
            and self.canonical_bytes() == other.canonical_bytes()
        )

    def __hash__(self) -> int:
        _ensure(self)
        return int.from_bytes(
            bytes.fromhex(self.semantic_hash.digest[:16]), "big", signed=True
        )


def _build(
    *,
    status: object,
    original_receipt_hash: object,
    rerun_receipt_hash: object | None,
    matched_fields: object = (),
    mismatch_fields: object = (),
    stale_edges: object = (),
    unavailable_references: object = (),
) -> ReplayReport:
    checked_status = _text(status, "status")
    if checked_status not in _STATUSES:
        raise ReplayError("status", "is not a supported replay status")
    original = _hash(original_receipt_hash, "original_receipt_hash")
    rerun = (
        None
        if rerun_receipt_hash is None
        else _hash(rerun_receipt_hash, "rerun_receipt_hash")
    )
    if rerun == original:
        raise ReplayError(
            "rerun_receipt_hash", "must identify a distinct rerun receipt"
        )
    matched, mismatches, stale = (
        _strings(matched_fields, "matched_fields"),
        _strings(mismatch_fields, "mismatch_fields"),
        _report_strings(stale_edges, "stale_edges"),
    )
    unavailable = _report_hashes(unavailable_references, "unavailable_references")
    if set(matched) - _MATCHED_FIELDS:
        raise ReplayError("matched_fields", "contains an unsupported diagnostic")
    if set(mismatches) - _MISMATCH_FIELDS:
        raise ReplayError("mismatch_fields", "contains an unsupported diagnostic")
    if any(not _stale_edge(item) for item in stale):
        raise ReplayError("stale_edges", "contains an unsupported diagnostic")
    if checked_status == "exact_match":
        if (
            rerun is None
            or set(matched) != _MATCHED_FIELDS
            or mismatches
            or stale
            or unavailable
        ):
            raise ReplayError(
                "status", "exact_match requires only matched resolved data"
            )
    elif checked_status == "mismatch":
        if (
            rerun is None
            or not {"contract", "direct_dependencies"} <= set(matched)
            or not mismatches
            or set(matched) & set(mismatches)
            or set(matched) | set(mismatches) != _MATCHED_FIELDS
            or stale
            or unavailable
        ):
            raise ReplayError(
                "status", "mismatch requires a resolved result difference"
            )
    elif checked_status == "stale":
        if rerun is None or matched or mismatches or not stale or unavailable:
            raise ReplayError("status", "stale requires resolved direct stale edges")
    elif matched or mismatches or stale or not unavailable:
        raise ReplayError(
            "status", "unavailable requires an absent reference or execution"
        )
    value = object.__new__(ReplayReport)
    for name, item in (
        ("status", checked_status),
        ("original_receipt_hash", original),
        ("rerun_receipt_hash", rerun),
        ("matched_fields", matched),
        ("mismatch_fields", mismatches),
        ("stale_edges", stale),
        ("unavailable_references", unavailable),
        ("semantic_hash", SemanticHash("sha256", "0" * 64)),
    ):
        object.__setattr__(value, name, item)
    object.__setattr__(value, "semantic_hash", _record_hash(value))
    return value


def _contract_edges(original: ResultReceipt, rerun: ResultReceipt) -> tuple[str, ...]:
    contract = original.contract
    candidate = rerun.contract
    fields = (
        ("contract_name", contract.contract_name, candidate.contract_name),
        ("project_id", contract.project_id, candidate.project_id),
        ("question", contract.question, candidate.question),
        (
            "acceptable_outcomes",
            contract.acceptable_outcomes,
            candidate.acceptable_outcomes,
        ),
        ("inputs", contract.input_hashes, candidate.input_hashes),
        ("sources", contract.source_anchors, candidate.source_anchors),
        ("conventions", contract.convention_manifests, candidate.convention_manifests),
        ("algorithms", contract.algorithms, candidate.algorithms),
        ("backend_request", contract.backend_request, candidate.backend_request),
        ("backend_name", contract.backend_name, candidate.backend_name),
        ("backend_version", contract.backend_version, candidate.backend_version),
        ("assumptions", contract.assumptions, candidate.assumptions),
        (
            "theorem_hypotheses",
            contract.theorem_hypotheses,
            candidate.theorem_hypotheses,
        ),
        ("bounds", contract.bounds, candidate.bounds),
        (
            "required_cross_checks",
            contract.required_cross_checks,
            candidate.required_cross_checks,
        ),
        (
            "expected_artifacts",
            contract.expected_artifacts,
            candidate.expected_artifacts,
        ),
        (
            "acceptance_predicates",
            contract.acceptance_predicates,
            candidate.acceptance_predicates,
        ),
    )
    return tuple(f"contract.{name}" for name, left, right in fields if left != right)


def _missing_direct_references(
    receipt: ResultReceipt, resolver: ReplayResolver
) -> tuple[SemanticHash, ...]:
    references = list(receipt.dependency_receipts)
    if receipt.supersedes is not None:
        references.append(receipt.supersedes)
    return tuple(
        reference for reference in references if resolver._lookup(reference) is None
    )


def verify_replay(
    receipt: object, *, resolver: object, environment: object | None = None
) -> ReplayReport:
    """Compare a resolved rerun with one sealed receipt without executing code.

    ``environment`` is reserved by the v0.0 API for a future executor.  Direct
    replay comparison never reads it, so accepting only ``None`` prevents a
    caller from believing that an ignored environment changed the result.
    """
    if environment is not None:
        raise ReplayError(
            "environment", "direct replay does not execute in an environment"
        )
    original_input = _receipt(receipt, "receipt")
    if type(resolver) is not ReplayResolver:
        raise ReplayError("resolver", "must be an exact ReplayResolver")
    supplied_resolver = resolver
    # Resolver values may have been altered through an adversarial
    # ``object.__setattr__`` after construction.  Rebuild from canonical
    # receipts before trusting either lookup table; no caller callable runs.
    checked_resolver = ReplayResolver.create(
        receipts=supplied_resolver.receipts, reruns=supplied_resolver.reruns
    )
    original = checked_resolver._lookup(original_input.semantic_hash)
    if original is None:
        return _build(
            status="unavailable",
            original_receipt_hash=original_input.semantic_hash,
            rerun_receipt_hash=None,
            unavailable_references=(original_input.semantic_hash,),
        )
    rerun_hash = checked_resolver._rerun_for(original.semantic_hash)
    if rerun_hash is None:
        return _build(
            status="unavailable",
            original_receipt_hash=original.semantic_hash,
            rerun_receipt_hash=None,
            unavailable_references=(original.semantic_hash,),
        )
    rerun = checked_resolver._lookup(rerun_hash)
    assert rerun is not None  # ReplayResolver.create checked both edge endpoints.
    missing = _report_hashes(
        tuple(
            sorted(
                set(
                    _missing_direct_references(original, checked_resolver)
                    + _missing_direct_references(rerun, checked_resolver)
                ),
                key=str,
            )
        ),
        "unavailable_references",
    )
    if missing:
        return _build(
            status="unavailable",
            original_receipt_hash=original.semantic_hash,
            rerun_receipt_hash=rerun.semantic_hash,
            unavailable_references=missing,
        )
    stale = list(_contract_edges(original, rerun))
    dependency_updates = {
        reference: checked_resolver._rerun_for(reference)
        for reference in original.dependency_receipts
    }
    expected_dependencies = {
        replacement or reference
        for reference, replacement in dependency_updates.items()
    }
    for reference, replacement in dependency_updates.items():
        if replacement is not None:
            stale.append(f"dependency_receipt:{reference}->{replacement}")
    for reference in sorted(
        expected_dependencies ^ set(rerun.dependency_receipts), key=str
    ):
        stale.append(f"dependency_receipt:{reference}")
    predecessor_replacement = (
        None
        if original.supersedes is None
        else checked_resolver._rerun_for(original.supersedes)
    )
    expected_predecessor = (
        None
        if original.supersedes is None
        else predecessor_replacement or original.supersedes
    )
    if predecessor_replacement is not None:
        stale.append(f"supersedes:{original.supersedes}->{predecessor_replacement}")
    if expected_predecessor != rerun.supersedes:
        if expected_predecessor is not None:
            stale.append(f"supersedes:{expected_predecessor}")
        if rerun.supersedes is not None:
            stale.append(f"supersedes:{rerun.supersedes}")
    if stale:
        return _build(
            status="stale",
            original_receipt_hash=original.semantic_hash,
            rerun_receipt_hash=rerun.semantic_hash,
            stale_edges=stale,
        )
    if (
        original.execution_status != "completed"
        or rerun.execution_status != "completed"
    ):
        return _build(
            status="unavailable",
            original_receipt_hash=original.semantic_hash,
            rerun_receipt_hash=rerun.semantic_hash,
            unavailable_references=(
                original.semantic_hash
                if original.execution_status != "completed"
                else rerun.semantic_hash,
            ),
        )
    mismatches: list[str] = []
    matched = ["contract", "direct_dependencies"]
    if original.primary_result_hash != rerun.primary_result_hash:
        mismatches.append("primary_result_hash")
    else:
        matched.append("primary_result_hash")
    if original.mathematical_outcome != rerun.mathematical_outcome:
        mismatches.append("mathematical_outcome")
    else:
        matched.append("mathematical_outcome")
    if mismatches:
        return _build(
            status="mismatch",
            original_receipt_hash=original.semantic_hash,
            rerun_receipt_hash=rerun.semantic_hash,
            matched_fields=matched,
            mismatch_fields=mismatches,
        )
    return _build(
        status="exact_match",
        original_receipt_hash=original.semantic_hash,
        rerun_receipt_hash=rerun.semantic_hash,
        matched_fields=(
            "contract",
            "primary_result_hash",
            "mathematical_outcome",
            "direct_dependencies",
        ),
    )


def _hash_record(value: SemanticHash) -> dict[str, str]:
    return {"algorithm": value.algorithm, "digest": value.digest}


def _body(value: ReplayReport) -> dict[str, object]:
    return {
        "status": value.status,
        "originalReceiptHash": _hash_record(value.original_receipt_hash),
        "rerunReceiptHash": None
        if value.rerun_receipt_hash is None
        else _hash_record(value.rerun_receipt_hash),
        "matchedFields": list(value.matched_fields),
        "mismatchFields": list(value.mismatch_fields),
        # Registry records cap every JSON container at 256 entries.  Preserve
        # every direct edge by paging the bounded 1,024-item report fields.
        "staleEdges": [
            list(value.stale_edges[index : index + _MAX_ITEMS])
            for index in range(0, len(value.stale_edges), _MAX_ITEMS)
        ],
        "unavailableReferences": [
            [
                _hash_record(item)
                for item in value.unavailable_references[index : index + _MAX_ITEMS]
            ]
            for index in range(0, len(value.unavailable_references), _MAX_ITEMS)
        ],
    }


def _encode(value: ReplayReport) -> dict[str, JSONValue]:
    return cast(dict[str, JSONValue], _body(value))


REPLAY_REPORT_REGISTRY = SerializerRegistry().with_codec(
    _TAG, _VERSION, ReplayReport, _encode, lambda record: _parse(record)
)


def _record_hash(value: ReplayReport) -> SemanticHash:
    try:
        return SemanticHash(
            "sha256",
            hashlib.sha256(
                canonical_json(value, registry=REPLAY_REPORT_REGISTRY)
            ).hexdigest(),
        )
    except (SchemaError, TypeError, ValueError, AttributeError) as error:
        raise ReplayError(
            "replay_report", "cannot derive canonical content address"
        ) from error


def _ensure(value: object) -> None:
    if type(value) is not ReplayReport:
        raise ReplayError("replay_report", "must be an exact ReplayReport")
    try:
        if _hash(value.semantic_hash, "semantic_hash") != _record_hash(value):
            raise ReplayError("replay_report", "content address integrity rejected")
    except (SchemaError, TypeError, ValueError, AttributeError) as error:
        raise ReplayError(
            "replay_report", "content address integrity rejected"
        ) from error


def replay_report_record(value: object) -> dict[str, object]:
    _ensure(value)
    return cast(dict[str, object], REPLAY_REPORT_REGISTRY.to_record(value))


def replay_report_canonical_bytes(value: object) -> bytes:
    _ensure(value)
    return canonical_json(value, registry=REPLAY_REPORT_REGISTRY)


def replay_report_registry() -> SerializerRegistry:
    return REPLAY_REPORT_REGISTRY


def _mapping(value: object, keys: frozenset[str], field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReplayError(field, "has an invalid record shape")
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
        raise ReplayError(field, "has an invalid semantic hash")
    try:
        return SemanticHash(fields["algorithm"], fields["digest"])
    except Exception as error:
        raise ReplayError(field, "has an invalid semantic hash") from error


def _tuple(value: object, field: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise ReplayError(field, "has an invalid record shape")
    return _items(value, field)


def _pages(value: object, field: str) -> tuple[object, ...]:
    pages = _tuple(value, field)
    if len(pages) > _MAX_REPORT_ITEMS // _MAX_ITEMS:
        raise ReplayError(field, "has too many pages")
    flattened: list[object] = []
    for page in pages:
        flattened.extend(_tuple(page, field))
    return tuple(flattened)


def _parse(record: object) -> ReplayReport:
    fields = _mapping(
        record,
        frozenset(
            (
                "schemaType",
                "schemaVersion",
                "status",
                "originalReceiptHash",
                "rerunReceiptHash",
                "matchedFields",
                "mismatchFields",
                "staleEdges",
                "unavailableReferences",
            )
        ),
        "replay_report",
    )
    if fields["schemaType"] != _TAG or fields["schemaVersion"] != _VERSION:
        raise ReplayError("replay_report", "has an invalid schema identity")
    return _build(
        status=fields["status"],
        original_receipt_hash=_parse_hash(
            fields["originalReceiptHash"], "original_receipt_hash"
        ),
        rerun_receipt_hash=None
        if fields["rerunReceiptHash"] is None
        else _parse_hash(fields["rerunReceiptHash"], "rerun_receipt_hash"),
        matched_fields=_tuple(fields["matchedFields"], "matched_fields"),
        mismatch_fields=_tuple(fields["mismatchFields"], "mismatch_fields"),
        stale_edges=_pages(fields["staleEdges"], "stale_edges"),
        unavailable_references=tuple(
            _parse_hash(item, "unavailable_references")
            for item in _pages(
                fields["unavailableReferences"], "unavailable_references"
            )
        ),
    )
