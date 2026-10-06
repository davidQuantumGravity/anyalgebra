"""Deterministic finite-census composition and atomic atlas publication."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from anyalgebra.census.analysis_records import (
    analysis_record_canonical_bytes,
    assemble_finite_carrier_analysis,
)
from anyalgebra.census.classify import (
    CanonicalOrbit,
    OrbitMember,
    classify_reference_orbits,
    orbit_classification_canonical_bytes,
)
from anyalgebra.census.reference import enumerate_reference
from anyalgebra.census.reports import (
    CompleteEnumerationReport,
    enumeration_report_canonical_bytes,
    report_reference_enumeration,
)
from anyalgebra.census.serialization import census_spec_canonical_bytes
from anyalgebra.census.spec import CensusSpec
from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import SemanticHash

from .models import (
    AtlasCorpus,
    AtlasRepresentative,
    FiniteAlgebraAtlas,
    atlas_canonical_bytes,
    create_atlas_corpus,
)


_MAX_CORPORA = 256


class AtlasBuildError(AnyAlgebraError, ValueError):
    """An atlas build input, computation, or publication step failed closed."""

    def __init__(self, *, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid finite atlas build {field}: {reason}")


@dataclass(frozen=True, slots=True)
class AtlasBuildResult:
    """Published atlas value and its explicit immutable output locations."""

    atlas: FiniteAlgebraAtlas
    output_directory: Path
    atlas_path: Path
    object_count: int


def _specs(value: object) -> tuple[CensusSpec, ...]:
    if isinstance(value, str | bytes):
        raise AtlasBuildError(field="specs", reason="must be an iterable")
    try:
        iterator = iter(cast(Iterable[object], value))
    except Exception:
        failed = True
    else:
        failed = False
    if failed:
        raise AtlasBuildError(field="specs", reason="must be an iterable")
    result: list[object] = []
    try:
        for _ in range(_MAX_CORPORA + 1):
            result.append(next(iterator))
    except StopIteration:
        pass
    except Exception:
        failed = True
    else:
        failed = False
    if failed:
        raise AtlasBuildError(field="specs", reason="iterable snapshot failed")
    if len(result) > _MAX_CORPORA:
        raise AtlasBuildError(field="specs", reason="item limit exceeded")
    if not result:
        raise AtlasBuildError(field="specs", reason="must not be empty")
    if any(type(item) is not CensusSpec for item in result):
        raise AtlasBuildError(
            field="specs", reason="must contain exact CensusSpec records"
        )
    specs = tuple(
        sorted(
            cast(tuple[CensusSpec, ...], tuple(result)),
            key=lambda item: str(item.semantic_hash),
        )
    )
    if len({item.semantic_hash for item in specs}) != len(specs):
        raise AtlasBuildError(field="specs", reason="contains duplicate specifications")
    if len({item.core.corpus_name for item in specs}) != len(specs):
        raise AtlasBuildError(field="specs", reason="contains duplicate corpus names")
    return specs


def _output_path(value: object) -> Path:
    if type(value) is not str and not isinstance(value, Path):
        raise AtlasBuildError(
            field="output_directory", reason="must be an exact str or Path"
        )
    raw = Path(value)
    if not raw.is_absolute():
        raise AtlasBuildError(field="output_directory", reason="must be absolute")
    target = raw.resolve(strict=False)
    if target.exists():
        raise AtlasBuildError(field="output_directory", reason="must not already exist")
    if not target.parent.is_dir():
        raise AtlasBuildError(
            field="output_directory", reason="parent directory must already exist"
        )
    if target == Path(target.anchor) or target == target.parent:
        raise AtlasBuildError(field="output_directory", reason="unsafe broad target")
    return target


def _encoded(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _semantic_bytes(record: dict[str, object]) -> tuple[SemanticHash, bytes]:
    identifier = SemanticHash("sha256", hashlib.sha256(_encoded(record)).hexdigest())
    return identifier, _encoded(
        {
            **record,
            "contentHash": {
                "algorithm": identifier.algorithm,
                "digest": identifier.digest,
            },
        }
    )


def _certificate_artifact(member: OrbitMember) -> tuple[SemanticHash, bytes]:
    verification = member.verification
    return _semantic_bytes(
        {
            "canonicalId": {
                "algorithm": member.certificate.canonical_id.algorithm,
                "digest": member.certificate.canonical_id.digest,
            },
            "checkedCellCount": verification.checked_cell_count,
            "identifierVerified": verification.identifier_verified,
            "permutationImages": list(member.certificate.permutation_images),
            "schemaType": "anyalgebra.atlas.canonical_certificate_evidence",
            "schemaVersion": 1,
            "sourceOutputs": list(member.certificate.source_outputs),
            "targetOutputs": list(member.certificate.canonical_outputs),
            "transportVerified": verification.transport_verified,
        }
    )


def _representative(
    spec: CensusSpec,
    orbit: CanonicalOrbit,
    objects: dict[SemanticHash, bytes],
) -> AtlasRepresentative:
    certificate_ids: list[SemanticHash] = []
    for member in orbit.members:
        identifier, encoded = _certificate_artifact(member)
        _retain(objects, identifier, encoded)
        certificate_ids.append(identifier)
    analysis = assemble_finite_carrier_analysis(
        spec.core, orbit.representative_outputs, spec.equivalence
    )
    analysis_bytes = analysis_record_canonical_bytes(analysis)
    _retain(objects, analysis.semantic_hash, analysis_bytes)
    return AtlasRepresentative.create(
        canonical_id=orbit.canonical_id,
        representative_outputs=orbit.representative_outputs,
        full_orbit_size=orbit.full_orbit_size,
        accepted_member_count=orbit.accepted_member_count,
        certificate_ids=tuple(certificate_ids),
        analysis_ids=(analysis.semantic_hash,),
    )


def _retain(
    objects: dict[SemanticHash, bytes], identifier: SemanticHash, encoded: bytes
) -> None:
    prior = objects.get(identifier)
    if prior is not None and prior != encoded:
        raise AtlasBuildError(field="object_store", reason="content-address collision")
    objects[identifier] = encoded


def _receipt_artifact(
    *,
    corpus_id: str,
    complete: bool,
    links: tuple[SemanticHash, ...],
) -> tuple[SemanticHash, bytes]:
    return _semantic_bytes(
        {
            "algorithms": [
                "anyalgebra.census.reference",
                "anyalgebra.census.canonical_bruteforce",
                "anyalgebra.capability_aware_analysis_assembly",
            ],
            "complete": complete,
            "corpusId": corpus_id,
            "linkedObjects": [
                {"algorithm": item.algorithm, "digest": item.digest}
                for item in sorted(links, key=str)
            ],
            "schemaType": "anyalgebra.atlas.build_receipt",
            "schemaVersion": 1,
        }
    )


def _build_corpus(spec: CensusSpec, objects: dict[SemanticHash, bytes]) -> AtlasCorpus:
    spec_bytes = census_spec_canonical_bytes(spec)
    _retain(objects, spec.semantic_hash, spec_bytes)
    enumeration = enumerate_reference(spec)
    report = report_reference_enumeration(enumeration)
    report_bytes = enumeration_report_canonical_bytes(report)
    _retain(objects, report.semantic_hash, report_bytes)
    linked = [spec.semantic_hash, report.semantic_hash]
    if type(report) is CompleteEnumerationReport:
        classification = classify_reference_orbits(enumeration)
        classification_bytes = orbit_classification_canonical_bytes(classification)
        _retain(objects, classification.semantic_hash, classification_bytes)
        linked.append(classification.semantic_hash)
        representatives = tuple(
            _representative(spec, orbit, objects) for orbit in classification.orbits
        )
        for representative in representatives:
            linked.extend(representative.certificate_ids)
            linked.extend(representative.analysis_ids)
        classification_id: SemanticHash | None = classification.semantic_hash
        accepted = classification.accepted_labeled_count
        rejected = classification.rejected_labeled_count
        complete = True
    else:
        representatives = ()
        classification_id = None
        accepted = report.emitted_candidate_count
        rejected = report.rejected_candidate_count
        complete = False
    receipt_id, receipt_bytes = _receipt_artifact(
        corpus_id=spec.core.corpus_name,
        complete=complete,
        links=tuple(linked),
    )
    _retain(objects, receipt_id, receipt_bytes)
    return create_atlas_corpus(
        corpus_id=spec.core.corpus_name,
        spec_id=spec.semantic_hash,
        enumeration_id=report.semantic_hash,
        classification_id=classification_id,
        receipt_id=receipt_id,
        total_candidate_count=report.total_candidate_count,
        examined_candidate_count=report.examined_candidate_count,
        accepted_labeled_count=accepted,
        rejected_labeled_count=rejected,
        representatives=representatives,
        complete=complete,
    )


def _write_object(stage: Path, identifier: SemanticHash, encoded: bytes) -> None:
    directory = stage / "objects" / identifier.algorithm
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{identifier.digest}.json").write_bytes(encoded)


def build_finite_algebra_atlas(
    specs: object, *, output_directory: object
) -> AtlasBuildResult:
    """Build into a private sibling directory and publish with one rename."""
    checked_specs = _specs(specs)
    target = _output_path(output_directory)
    try:
        with TemporaryDirectory(prefix=".anyalgebra-atlas-", dir=target.parent) as raw:
            stage = Path(raw)
            objects: dict[SemanticHash, bytes] = {}
            corpora = tuple(_build_corpus(spec, objects) for spec in checked_specs)
            atlas = FiniteAlgebraAtlas.create(corpora)
            if set(atlas.header.evidence_ids) != set(objects):
                raise AtlasBuildError(
                    field="object_store", reason="atlas links do not match objects"
                )
            for identifier in sorted(objects, key=str):
                _write_object(stage, identifier, objects[identifier])
            (stage / "atlas.json").write_bytes(atlas_canonical_bytes(atlas))
            os.replace(stage, target)
    except AtlasBuildError:
        raise
    except Exception:
        failed = True
    else:
        failed = False
    if failed:
        raise AtlasBuildError(
            field="publication", reason="build failed before publication"
        )
    return AtlasBuildResult(
        atlas=atlas,
        output_directory=target,
        atlas_path=target / "atlas.json",
        object_count=len(objects),
    )


__all__ = (
    "AtlasBuildError",
    "AtlasBuildResult",
    "build_finite_algebra_atlas",
)
