"""Fail-closed atlas loading and deterministic non-executing queries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import cast

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import SemanticHash

from .models import (
    AtlasRepresentative,
    FiniteAlgebraAtlas,
    atlas_record,
    create_atlas_corpus,
)


_MAX_FILE_BYTES = 16 * 1024 * 1024
_MAX_QUERY_ITEMS = 256
_ATLAS_KEYS = frozenset(
    ("contentHash", "corpora", "header", "schemaType", "schemaVersion")
)
_HEADER_KEYS = frozenset(
    (
        "corpusCount",
        "evidenceIds",
        "representativeCount",
        "schemaVersion",
        "status",
    )
)
_CORPUS_KEYS = frozenset(
    (
        "acceptedLabeledCount",
        "classificationId",
        "complete",
        "corpusId",
        "enumerationId",
        "examinedCandidateCount",
        "orbitCount",
        "receiptId",
        "rejectedLabeledCount",
        "representatives",
        "specId",
        "totalCandidateCount",
    )
)
_REPRESENTATIVE_KEYS = frozenset(
    (
        "acceptedMemberCount",
        "analysisIds",
        "canonicalId",
        "certificateIds",
        "fullOrbitSize",
        "representativeOutputs",
    )
)
_HASH_KEYS = frozenset(("algorithm", "digest"))
_OBJECT_TYPES = frozenset(
    (
        "anyalgebra.atlas.build_receipt",
        "anyalgebra.atlas.canonical_certificate_evidence",
        "anyalgebra.census.capability_analysis_record",
        "anyalgebra.census.orbit_classification",
        "anyalgebra.census.report.complete",
        "anyalgebra.census.report.incomplete",
        "anyalgebra.census.spec",
    )
)
_LAWS = frozenset(
    (
        "alternativity",
        "associativity",
        "commutativity",
        "flexibility",
        "idempotence",
        "left_alternativity",
        "right_alternativity",
        "totality",
    )
)
_INVARIANTS = frozenset(
    (
        "center_size",
        "commutant_size",
        "congruence_count",
        "idempotent_count",
        "identity_status",
        "nucleus_size",
        "substructure_count",
        "zero_status",
    )
)


class AtlasQueryError(AnyAlgebraError, ValueError):
    """An atlas directory, record, reference, or query failed validation."""

    def __init__(self, *, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid atlas query {field}: {reason}")


@dataclass(frozen=True, slots=True)
class LoadedAtlas:
    """A fully verified atlas plus immutable canonical object bytes."""

    atlas: FiniteAlgebraAtlas
    root: Path
    object_bytes: tuple[tuple[SemanticHash, bytes], ...]
    verified_object_count: int


@dataclass(frozen=True, slots=True)
class AtlasMatch:
    """One deterministic representative query view."""

    corpus_id: str
    canonical_id: SemanticHash
    representative_outputs: tuple[int, ...]
    full_orbit_size: int
    accepted_member_count: int
    carrier_size: int
    arity: int
    law_statuses: Mapping[str, str]
    invariants: Mapping[str, str | int | bool]


@dataclass(frozen=True, slots=True)
class AtlasQueryResult:
    """Immutable ordered matches and the exact examined representative count."""

    matches: tuple[AtlasMatch, ...]
    examined_representative_count: int


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise AtlasQueryError(field="record", reason="duplicate JSON key")
        result[key] = value
    return result


def _encoded(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _read_canonical(path: Path, field: str) -> tuple[dict[str, object], bytes]:
    try:
        size = path.stat().st_size
    except OSError:
        unavailable = True
    else:
        unavailable = False
    if unavailable:
        raise AtlasQueryError(field=field, reason="file is unavailable")
    if size > _MAX_FILE_BYTES:
        raise AtlasQueryError(field=field, reason="file size limit exceeded")
    try:
        encoded = path.read_bytes()
        parsed = json.loads(encoded, object_pairs_hook=_pairs_no_duplicates)
    except AtlasQueryError:
        raise
    except Exception:
        invalid = True
    else:
        invalid = False
    if invalid:
        raise AtlasQueryError(field=field, reason="invalid JSON")
    if type(parsed) is not dict:
        raise AtlasQueryError(field=field, reason="record must be an object")
    record = cast(dict[str, object], parsed)
    if _encoded(record) != encoded:
        raise AtlasQueryError(field=field, reason="must use canonical JSON bytes")
    return record, encoded


def _mapping(value: object, keys: frozenset[str], field: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise AtlasQueryError(field=field, reason="record shape is invalid")
    return cast(dict[str, object], value)


def _list(value: object, field: str, maximum: int = 65_536) -> list[object]:
    if type(value) is not list or len(value) > maximum:
        raise AtlasQueryError(field=field, reason="must be a bounded array")
    return cast(list[object], value)


def _count(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise AtlasQueryError(field=field, reason="must be a nonnegative exact int")
    return value


def _text(value: object, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise AtlasQueryError(field=field, reason="must be nonempty stable text")
    return value


def _hash(value: object, field: str) -> SemanticHash:
    record = _mapping(value, _HASH_KEYS, field)
    try:
        result = SemanticHash(record["algorithm"], record["digest"])  # type: ignore[arg-type]
    except Exception:
        invalid = True
    else:
        invalid = False
    if invalid:
        raise AtlasQueryError(field=field, reason="invalid content address")
    return result


def _hashes(value: object, field: str) -> tuple[SemanticHash, ...]:
    return tuple(_hash(item, field) for item in _list(value, field, 4_096))


def _parse_representative(value: object) -> AtlasRepresentative:
    record = _mapping(value, _REPRESENTATIVE_KEYS, "representative")
    return AtlasRepresentative.create(
        canonical_id=_hash(record["canonicalId"], "canonical_id"),
        representative_outputs=tuple(
            _count(item, "representative_outputs")
            for item in _list(record["representativeOutputs"], "representative_outputs")
        ),
        full_orbit_size=_count(record["fullOrbitSize"], "full_orbit_size"),
        accepted_member_count=_count(
            record["acceptedMemberCount"], "accepted_member_count"
        ),
        certificate_ids=_hashes(record["certificateIds"], "certificate_ids"),
        analysis_ids=_hashes(record["analysisIds"], "analysis_ids"),
    )


def _parse_atlas(record: dict[str, object]) -> FiniteAlgebraAtlas:
    if set(record) != _ATLAS_KEYS:
        raise AtlasQueryError(field="atlas", reason="atlas record shape is invalid")
    if record["schemaType"] != "anyalgebra.atlas.finite_algebra":
        raise AtlasQueryError(field="atlas", reason="unknown atlas schema")
    if record["schemaVersion"] != 1:
        raise AtlasQueryError(field="atlas", reason="unknown atlas version")
    _mapping(record["header"], _HEADER_KEYS, "header")
    corpora = []
    for raw in _list(record["corpora"], "corpora", 256):
        corpus = _mapping(raw, _CORPUS_KEYS, "corpus")
        classification_raw = corpus["classificationId"]
        classification = (
            None
            if classification_raw is None
            else _hash(classification_raw, "classification_id")
        )
        representatives = tuple(
            _parse_representative(item)
            for item in _list(corpus["representatives"], "representatives", 4_096)
        )
        corpora.append(
            create_atlas_corpus(
                corpus_id=_text(corpus["corpusId"], "corpus_id"),
                spec_id=_hash(corpus["specId"], "spec_id"),
                enumeration_id=_hash(corpus["enumerationId"], "enumeration_id"),
                classification_id=classification,
                receipt_id=_hash(corpus["receiptId"], "receipt_id"),
                total_candidate_count=_count(
                    corpus["totalCandidateCount"], "total_candidate_count"
                ),
                examined_candidate_count=_count(
                    corpus["examinedCandidateCount"], "examined_candidate_count"
                ),
                accepted_labeled_count=_count(
                    corpus["acceptedLabeledCount"], "accepted_labeled_count"
                ),
                rejected_labeled_count=_count(
                    corpus["rejectedLabeledCount"], "rejected_labeled_count"
                ),
                representatives=representatives,
                complete=corpus["complete"],
            )
        )
    try:
        atlas = FiniteAlgebraAtlas.create(tuple(corpora))
        if atlas_record(atlas) != record:
            raise AtlasQueryError(field="atlas", reason="atlas content drift")
    except AtlasQueryError:
        raise
    except Exception:
        invalid = True
    else:
        invalid = False
    if invalid:
        raise AtlasQueryError(field="atlas", reason="atlas content drift")
    return atlas


def _object_record(encoded: bytes, identifier: SemanticHash) -> dict[str, object]:
    try:
        parsed = json.loads(encoded, object_pairs_hook=_pairs_no_duplicates)
    except Exception:
        invalid = True
    else:
        invalid = False
    if invalid:
        raise AtlasQueryError(field="object", reason="object content is invalid")
    if type(parsed) is not dict or _encoded(parsed) != encoded:
        raise AtlasQueryError(field="object", reason="object content is invalid")
    record = cast(dict[str, object], parsed)
    content = record.get("contentHash")
    try:
        content_identifier = _hash(content, "object_content_hash")
    except AtlasQueryError:
        invalid = True
    else:
        invalid = False
    if invalid:
        raise AtlasQueryError(field="object", reason="object content is invalid")
    if content_identifier != identifier:
        raise AtlasQueryError(field="object", reason="object content address mismatch")
    body = {key: value for key, value in record.items() if key != "contentHash"}
    actual = SemanticHash("sha256", hashlib.sha256(_encoded(body)).hexdigest())
    if actual != identifier:
        raise AtlasQueryError(field="object", reason="object content address mismatch")
    if (
        record.get("schemaType") not in _OBJECT_TYPES
        or record.get("schemaVersion") != 1
    ):
        raise AtlasQueryError(field="object", reason="unknown object schema")
    return record


def _record_for(
    records: Mapping[SemanticHash, dict[str, object]],
    identifier: SemanticHash,
    schema_type: str,
) -> dict[str, object]:
    record = records[identifier]
    if record.get("schemaType") != schema_type:
        raise AtlasQueryError(field="reference", reason="object role mismatch")
    return record


def _hex_or_int(value: object, field: str) -> int:
    if type(value) is int:
        return _count(value, field)
    record = _mapping(value, frozenset(("digits", "radix")), field)
    if record["radix"] != 16 or type(record["digits"]) is not str:
        raise AtlasQueryError(field=field, reason="invalid exact integer")
    try:
        result = int(record["digits"], 16)
    except ValueError:
        invalid = True
    else:
        invalid = False
    if invalid:
        raise AtlasQueryError(field=field, reason="invalid exact integer")
    return result


def _verify_corpus_links(
    atlas: FiniteAlgebraAtlas,
    records: Mapping[SemanticHash, dict[str, object]],
) -> None:
    for corpus in atlas.corpora:
        spec = _record_for(records, corpus.spec_id, "anyalgebra.census.spec")
        if spec.get("corpusName") != corpus.corpus_id:
            raise AtlasQueryError(field="reference", reason="spec corpus mismatch")
        report_type = (
            "anyalgebra.census.report.complete"
            if corpus.complete
            else "anyalgebra.census.report.incomplete"
        )
        report = _record_for(records, corpus.enumeration_id, report_type)
        if (
            _hash(report.get("specHash"), "report_spec") != corpus.spec_id
            or _hex_or_int(report.get("totalCandidateCount"), "total")
            != corpus.total_candidate_count
            or _hex_or_int(report.get("examinedCandidateCount"), "examined")
            != corpus.examined_candidate_count
            or _hex_or_int(report.get("emittedCandidateCount"), "emitted")
            != corpus.accepted_labeled_count
            or _hex_or_int(report.get("rejectedCandidateCount"), "rejected")
            != corpus.rejected_labeled_count
        ):
            raise AtlasQueryError(field="reference", reason="report corpus mismatch")
        linked = {corpus.spec_id, corpus.enumeration_id}
        if corpus.classification_id is not None:
            classification = _record_for(
                records,
                corpus.classification_id,
                "anyalgebra.census.orbit_classification",
            )
            if (
                _hash(classification.get("specHash"), "classification_spec")
                != corpus.spec_id
                or classification.get("acceptedLabeledCount")
                != corpus.accepted_labeled_count
                or classification.get("rejectedLabeledCount")
                != corpus.rejected_labeled_count
                or classification.get("orbitCount") != corpus.orbit_count
            ):
                raise AtlasQueryError(
                    field="reference", reason="classification corpus mismatch"
                )
            orbit_records = _list(classification.get("orbits"), "orbits", 4_096)
            actual_orbits = tuple(
                sorted(
                    [
                        (
                            _hash(
                                cast(dict[str, object], item).get("canonicalId"),
                                "orbit_canonical_id",
                            ),
                            tuple(
                                _count(output, "orbit_outputs")
                                for output in _list(
                                    cast(dict[str, object], item).get(
                                        "representativeOutputs"
                                    ),
                                    "orbit_outputs",
                                )
                            ),
                            _count(
                                cast(dict[str, object], item).get("fullOrbitSize"),
                                "full_orbit_size",
                            ),
                            _count(
                                cast(dict[str, object], item).get(
                                    "acceptedMemberCount"
                                ),
                                "accepted_member_count",
                            ),
                        )
                        for item in orbit_records
                        if type(item) is dict
                    ],
                    key=lambda item: str(item[0]),
                )
            )
            expected_orbits = tuple(
                (
                    item.canonical_id,
                    item.representative_outputs,
                    item.full_orbit_size,
                    item.accepted_member_count,
                )
                for item in corpus.representatives
            )
            if (
                len(actual_orbits) != len(orbit_records)
                or actual_orbits != expected_orbits
            ):
                raise AtlasQueryError(
                    field="reference", reason="classification orbit mismatch"
                )
            linked.add(corpus.classification_id)
        for representative in corpus.representatives:
            for certificate_id in representative.certificate_ids:
                certificate = _record_for(
                    records,
                    certificate_id,
                    "anyalgebra.atlas.canonical_certificate_evidence",
                )
                if (
                    _hash(certificate.get("canonicalId"), "certificate_canonical")
                    != representative.canonical_id
                    or certificate.get("targetOutputs")
                    != list(representative.representative_outputs)
                    or certificate.get("transportVerified") is not True
                    or certificate.get("identifierVerified") is not True
                ):
                    raise AtlasQueryError(
                        field="reference", reason="certificate target mismatch"
                    )
                linked.add(certificate_id)
            for analysis_id in representative.analysis_ids:
                analysis = _record_for(
                    records,
                    analysis_id,
                    "anyalgebra.census.capability_analysis_record",
                )
                subject = cast(dict[str, object], analysis.get("subject"))
                if (
                    type(subject) is not dict
                    or subject.get("corpusName") != corpus.corpus_id
                    or subject.get("carrierSize") != spec.get("carrierSize")
                    or subject.get("arity") != spec.get("arity")
                    or subject.get("outputs")
                    != list(representative.representative_outputs)
                ):
                    raise AtlasQueryError(
                        field="reference", reason="analysis subject mismatch"
                    )
                linked.add(analysis_id)
        receipt = _record_for(
            records, corpus.receipt_id, "anyalgebra.atlas.build_receipt"
        )
        receipt_links = {
            _hash(item, "receipt_link")
            for item in _list(receipt.get("linkedObjects"), "receipt_links", 4_096)
        }
        if (
            receipt.get("corpusId") != corpus.corpus_id
            or receipt.get("complete") is not corpus.complete
            or receipt_links != linked
        ):
            raise AtlasQueryError(field="reference", reason="receipt link mismatch")


def _root(value: object) -> Path:
    if type(value) is not str and not isinstance(value, Path):
        raise AtlasQueryError(field="directory", reason="must be an exact str or Path")
    raw = Path(value)
    if not raw.is_absolute():
        raise AtlasQueryError(field="directory", reason="must be absolute")
    if raw.is_symlink():
        raise AtlasQueryError(field="directory", reason="symlink root is not allowed")
    root = raw.resolve(strict=True)
    if not root.is_dir():
        raise AtlasQueryError(field="directory", reason="must be a directory")
    return root


def load_finite_algebra_atlas(directory: object) -> LoadedAtlas:
    """Verify canonical atlas bytes, exact file set, hashes, and role links."""
    try:
        root = _root(directory)
        atlas_path = root / "atlas.json"
        if atlas_path.is_symlink():
            raise AtlasQueryError(field="atlas", reason="symlink file is not allowed")
        atlas_record_value, _ = _read_canonical(atlas_path, "atlas")
        atlas = _parse_atlas(atlas_record_value)
        expected_paths = {
            root / "objects" / item.algorithm / f"{item.digest}.json"
            for item in atlas.header.evidence_ids
        }
        actual_paths = {item for item in root.rglob("*") if item.is_file()}
        if actual_paths != expected_paths | {atlas_path}:
            raise AtlasQueryError(field="directory", reason="object file set mismatch")
        records: dict[SemanticHash, dict[str, object]] = {}
        encoded_objects: list[tuple[SemanticHash, bytes]] = []
        for identifier in atlas.header.evidence_ids:
            path = root / "objects" / identifier.algorithm / f"{identifier.digest}.json"
            if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
                raise AtlasQueryError(field="object", reason="path escapes atlas root")
            record, encoded = _read_canonical(path, "object")
            checked = _object_record(encoded, identifier)
            assert checked == record
            records[identifier] = checked
            encoded_objects.append((identifier, encoded))
        _verify_corpus_links(atlas, records)
        return LoadedAtlas(
            atlas=atlas,
            root=root,
            object_bytes=tuple(encoded_objects),
            verified_object_count=len(records),
        )
    except AtlasQueryError:
        raise
    except Exception:
        pass
    raise AtlasQueryError(field="atlas", reason="atlas content drift")


def _items(value: object, field: str) -> tuple[object, ...]:
    if isinstance(value, str | bytes):
        raise AtlasQueryError(field=field, reason="must be an iterable")
    try:
        iterator = iter(cast(Iterable[object], value))
    except Exception:
        failed = True
    else:
        failed = False
    if failed:
        raise AtlasQueryError(field=field, reason="must be an iterable")
    result: list[object] = []
    try:
        for _ in range(_MAX_QUERY_ITEMS + 1):
            result.append(next(iterator))
    except StopIteration:
        failed = False
    except Exception:
        failed = True
    else:
        failed = False
    if failed:
        raise AtlasQueryError(field=field, reason="iterable snapshot failed")
    if len(result) > _MAX_QUERY_ITEMS:
        raise AtlasQueryError(field=field, reason="item limit exceeded")
    return tuple(result)


def _analysis_view(
    record: dict[str, object],
) -> tuple[int, int, dict[str, str], dict[str, str | int | bool]]:
    subject = cast(dict[str, object], record["subject"])
    payloads = cast(dict[str, object], record["payloads"])
    law_profile = cast(dict[str, object], payloads["lawProfile"])
    element = cast(dict[str, object], payloads["elementAnalysis"])
    magma = cast(dict[str, object], payloads["magmaAnalysis"])
    subobjects = cast(dict[str, object], payloads["subobjectAnalysis"])
    law_statuses = {
        cast(str, item["name"]): cast(str, item["status"])
        for raw in cast(list[object], law_profile["results"])
        if type(raw) is dict
        for item in (cast(dict[str, object], raw),)
    }
    invariants: dict[str, str | int | bool] = {
        "center_size": len(cast(list[object], magma["center"])),
        "commutant_size": len(cast(list[object], magma["commutant"])),
        "congruence_count": len(cast(list[object], subobjects["congruences"])),
        "idempotent_count": len(cast(list[object], element["idempotents"])),
        "identity_status": cast(str, element["identityStatus"]),
        "nucleus_size": len(cast(list[object], magma["nucleus"])),
        "substructure_count": len(cast(list[object], subobjects["substructures"])),
        "zero_status": cast(str, element["zeroStatus"]),
    }
    return (
        cast(int, subject["carrierSize"]),
        cast(int, subject["arity"]),
        law_statuses,
        invariants,
    )


def query_atlas(
    loaded: object,
    *,
    carrier_size: int | None = None,
    arity: int | None = None,
    canonical_id: SemanticHash | None = None,
    laws: object = (),
    invariant_fields: object = (),
) -> AtlasQueryResult:
    """Filter verified representative records without executing stored payloads."""
    if type(loaded) is not LoadedAtlas:
        raise AtlasQueryError(field="atlas", reason="must be an exact LoadedAtlas")
    if carrier_size is not None:
        _count(carrier_size, "carrier_size")
    if arity is not None:
        _count(arity, "arity")
    if canonical_id is not None and type(canonical_id) is not SemanticHash:
        raise AtlasQueryError(field="canonical_id", reason="must be a SemanticHash")
    requested_laws = tuple(_text(item, "laws") for item in _items(laws, "laws"))
    if len(set(requested_laws)) != len(requested_laws):
        raise AtlasQueryError(field="laws", reason="contains duplicates")
    if any(item not in _LAWS for item in requested_laws):
        raise AtlasQueryError(field="laws", reason="contains unknown law")
    requested_invariants: list[tuple[str, str | int | bool]] = []
    for raw in _items(invariant_fields, "invariant_fields"):
        if type(raw) is not tuple or len(raw) != 2:
            raise AtlasQueryError(field="invariant_fields", reason="must contain pairs")
        name, expected = raw
        checked_name = _text(name, "invariant_fields")
        if checked_name not in _INVARIANTS:
            raise AtlasQueryError(
                field="invariant_fields", reason="contains unknown invariant"
            )
        if type(expected) not in (str, int, bool):
            raise AtlasQueryError(
                field="invariant_fields", reason="has invalid expected value"
            )
        requested_invariants.append((checked_name, cast(str | int | bool, expected)))
    if len({name for name, _ in requested_invariants}) != len(requested_invariants):
        raise AtlasQueryError(field="invariant_fields", reason="contains duplicates")
    records = {
        identifier: _object_record(encoded, identifier)
        for identifier, encoded in loaded.object_bytes
    }
    if set(records) != set(
        loaded.atlas.header.evidence_ids
    ) or loaded.verified_object_count != len(records):
        raise AtlasQueryError(field="atlas", reason="loaded object set drift")
    _verify_corpus_links(loaded.atlas, records)
    matches: list[AtlasMatch] = []
    examined = 0
    for corpus in loaded.atlas.corpora:
        for representative in corpus.representatives:
            examined += 1
            analysis = _record_for(
                records,
                representative.analysis_ids[0],
                "anyalgebra.census.capability_analysis_record",
            )
            size, operation_arity, statuses, invariants = _analysis_view(analysis)
            if carrier_size is not None and size != carrier_size:
                continue
            if arity is not None and operation_arity != arity:
                continue
            if canonical_id is not None and representative.canonical_id != canonical_id:
                continue
            if any(
                statuses.get(law) != "proved_on_complete_grid" for law in requested_laws
            ):
                continue
            if any(
                invariants[name] != expected for name, expected in requested_invariants
            ):
                continue
            matches.append(
                AtlasMatch(
                    corpus_id=corpus.corpus_id,
                    canonical_id=representative.canonical_id,
                    representative_outputs=representative.representative_outputs,
                    full_orbit_size=representative.full_orbit_size,
                    accepted_member_count=representative.accepted_member_count,
                    carrier_size=size,
                    arity=operation_arity,
                    law_statuses=MappingProxyType(dict(sorted(statuses.items()))),
                    invariants=MappingProxyType(dict(sorted(invariants.items()))),
                )
            )
    return AtlasQueryResult(tuple(matches), examined)


__all__ = (
    "AtlasMatch",
    "AtlasQueryError",
    "AtlasQueryResult",
    "LoadedAtlas",
    "load_finite_algebra_atlas",
    "query_atlas",
)
