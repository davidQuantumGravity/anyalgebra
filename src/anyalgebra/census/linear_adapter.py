"""Strict evidence adapter for existing module-backed QQ algebra analyses."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, cast

from anyalgebra.algebra.multilinear import FiniteMultilinearStructure
from anyalgebra.analysis import elementary, fingerprint
from anyalgebra.analysis.elementary import AnalysisBounds, SubspaceReport
from anyalgebra.analysis.fingerprint import (
    AlgebraFingerprint,
    DerivationReport,
    IdealSearchResult,
    UnitReport,
)
from anyalgebra.core.domains import DomainElement, QQ
from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import SemanticHash
from anyalgebra.core.rational import Rational


_SCHEMA_VERSION = 1


class LinearAnalysisAdapterError(AnyAlgebraError, ValueError):
    """An adapter input or retained evidence envelope is invalid."""

    def __init__(self, *, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid linear analysis adapter {field}: {reason}")


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class LinearAnalysisAdapterReport:
    """Existing exact reports retained under one capability-aware envelope."""

    algebra: FiniteMultilinearStructure
    bounds: AnalysisBounds
    left_nucleus: SubspaceReport
    middle_nucleus: SubspaceReport
    right_nucleus: SubspaceReport
    nucleus: SubspaceReport
    center: SubspaceReport
    ideals: IdealSearchResult
    derivations: DerivationReport
    fingerprint: AlgebraFingerprint
    unit: UnitReport
    product_span_dimension: int
    capability_statuses: tuple[tuple[str, str], ...]
    hypotheses: tuple[str, ...]
    complete: bool
    exact: bool
    algorithm: str
    algorithm_version: int
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise LinearAnalysisAdapterError(field="report", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("LinearAnalysisAdapterReport cannot be subclassed")

    def __eq__(self, other: object) -> bool:
        return type(other) is LinearAnalysisAdapterReport and (
            self.algebra is other.algebra
            and self.bounds.max_dimension == other.bounds.max_dimension
            and self.bounds.max_constraints == other.bounds.max_constraints
            and self.bounds.max_elimination_work == other.bounds.max_elimination_work
            and self.semantic_hash == other.semantic_hash
        )


def _encoded(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _semantic_hash(body: dict[str, object]) -> SemanticHash:
    return SemanticHash("sha256", hashlib.sha256(_encoded(body)).hexdigest())


def _snapshot_bounds(options: object) -> AnalysisBounds:
    if options is None:
        return AnalysisBounds()
    if type(options) is not AnalysisBounds:
        raise LinearAnalysisAdapterError(
            field="options", reason="must be exact AnalysisBounds or None"
        )
    for field in ("max_dimension", "max_constraints", "max_elimination_work"):
        value = getattr(options, field)
        if type(value) is not int or value <= 0:
            raise LinearAnalysisAdapterError(
                field=field, reason="must be a positive exact int"
            )
    return AnalysisBounds(
        max_dimension=options.max_dimension,
        max_constraints=options.max_constraints,
        max_elimination_work=options.max_elimination_work,
    )


def _checked_algebra(value: object) -> FiniteMultilinearStructure:
    if type(value) is not FiniteMultilinearStructure:
        raise LinearAnalysisAdapterError(
            field="algebra",
            reason="must be an exact module-backed FiniteMultilinearStructure",
        )
    if value.module.domain is not QQ():
        raise LinearAnalysisAdapterError(
            field="coefficient_domain", reason="must be the literal QQ parent"
        )
    if value.arity != 2:
        raise LinearAnalysisAdapterError(
            field="arity", reason="must be exactly two for v0.1 linear analysis"
        )
    return value


def analyze_module_backed_algebra(
    algebra: FiniteMultilinearStructure,
    *,
    options: AnalysisBounds | None = None,
) -> LinearAnalysisAdapterReport:
    """Route one literal QQ binary algebra to all existing exact analyzers."""
    checked = _checked_algebra(algebra)
    bounds = _snapshot_bounds(options)
    left = elementary.left_nucleus(checked, options=bounds)
    middle = elementary.middle_nucleus(checked, options=bounds)
    right = elementary.right_nucleus(checked, options=bounds)
    full = elementary.nucleus(checked, options=bounds)
    central = elementary.center(checked, options=bounds)
    ideal_report = fingerprint.ideals(checked, options=bounds)
    derivation_report = fingerprint.derivation_algebra(checked, options=bounds)
    algebra_report = fingerprint.algebra_fingerprint(checked, options=bounds)
    value = object.__new__(LinearAnalysisAdapterReport)
    for field, item in (
        ("algebra", checked),
        ("bounds", bounds),
        ("left_nucleus", left),
        ("middle_nucleus", middle),
        ("right_nucleus", right),
        ("nucleus", full),
        ("center", central),
        ("ideals", ideal_report),
        ("derivations", derivation_report),
        ("fingerprint", algebra_report),
        ("unit", algebra_report.unit_report),
        ("product_span_dimension", algebra_report.product_span_dimension),
        (
            "capability_statuses",
            (
                ("nuclei", "complete_exact"),
                ("center", "complete_exact"),
                ("ideals", ideal_report.status),
                ("derivations", "complete_exact"),
                ("unit", "complete_exact"),
                ("product_span", "complete_exact"),
                ("fingerprint", "rejection_only"),
            ),
        ),
        (
            "hypotheses",
            (
                "literal QQ finite free module",
                "one total binary bilinear operation on the declared basis",
                "ideal discovery is canonical but not a complete QQ-subspace census",
                "fingerprint equality is not proof of isomorphism",
            ),
        ),
        ("complete", False),
        ("exact", True),
        ("algorithm", "anyalgebra.existing_linear_analysis_adapter"),
        ("algorithm_version", 1),
        ("semantic_hash", SemanticHash("sha256", "0" * 64)),
    ):
        object.__setattr__(value, field, item)
    object.__setattr__(value, "semantic_hash", _semantic_hash(_body(value)))
    return value


def _rational(value: Rational) -> list[int]:
    return [value.numerator, value.denominator]


def _bounds_body(value: AnalysisBounds) -> dict[str, int]:
    return {
        "maxConstraints": value.max_constraints,
        "maxDimension": value.max_dimension,
        "maxEliminationWork": value.max_elimination_work,
    }


def _sparse_basis(report: SubspaceReport) -> list[list[list[int]]]:
    return [
        [
            [index, *_rational(cast(Any, coefficient).value)]
            for index, coefficient in element.coordinates().items()
        ]
        for element in report.basis
    ]


def _subspace_body(report: SubspaceReport) -> dict[str, object]:
    return {
        "algorithm": report.algorithm,
        "algorithmVersion": report.algorithm_version,
        "ambientDimension": report.ambient_dimension,
        "basis": _sparse_basis(report),
        "checkedConstraints": report.checked_constraints,
        "dimension": report.dimension,
        "eliminationWork": report.elimination_work,
        "expectedConstraints": report.expected_constraints,
        "kind": report.kind,
    }


def _vector_body(vector: tuple[Rational, ...]) -> list[list[int]]:
    return [_rational(value) for value in vector]


def _ideal_body(report: IdealSearchResult) -> dict[str, object]:
    return {
        "algorithm": report.algorithm,
        "algorithmVersion": report.algorithm_version,
        "complete": report.complete,
        "evaluations": report.evaluations,
        "ideals": [
            {
                "ambientDimension": item.ambient_dimension,
                "basis": [_vector_body(vector) for vector in item.basis],
                "closed": item.closed,
                "dimension": item.dimension,
                "origins": list(item.origins),
            }
            for item in report.ideals
        ],
        "status": report.status,
    }


def _derivation_body(report: DerivationReport) -> dict[str, object]:
    return {
        "algorithm": report.algorithm,
        "algorithmVersion": report.algorithm_version,
        "basis": [
            [
                [
                    _rational(cast(DomainElement[Rational], coefficient).value)
                    for coefficient in row
                ]
                for row in matrix.entries
            ]
            for matrix in report.basis
        ],
        "constraints": report.constraints,
        "convention": report.convention,
        "dimension": report.dimension,
        "eliminationWork": report.elimination_work,
    }


def _law_body(report: Any) -> dict[str, object]:
    witness_value = report.witness_value
    return {
        "checkedAssignments": report.checked_assignments,
        "expectedAssignments": report.expected_assignments,
        "kind": report.kind,
        "reductionComplete": report.reduction_complete,
        "reductionHypothesis": report.reduction_hypothesis,
        "status": report.status,
        "witnessIndices": (
            None if report.witness_indices is None else list(report.witness_indices)
        ),
        "witnessValue": (
            None if witness_value is None else _vector_body(witness_value)
        ),
    }


def _unit_body(report: UnitReport) -> dict[str, object]:
    return {
        "algorithm": report.algorithm,
        "algorithmVersion": report.algorithm_version,
        "scalarEquations": report.scalar_equations,
        "status": report.status,
        "unitVector": (
            None if report.unit_vector is None else _vector_body(report.unit_vector)
        ),
        "witnessRow": (
            None if report.witness_row is None else _vector_body(report.witness_row)
        ),
    }


def _fingerprint_body(report: AlgebraFingerprint) -> dict[str, object]:
    fields = (
        "dimension",
        "unit_status",
        "commutative",
        "associative",
        "commutativity_pairs_checked",
        "associativity_triples_checked",
        "unit_scalar_equations_checked",
        "left_nucleus_dimension",
        "middle_nucleus_dimension",
        "right_nucleus_dimension",
        "nucleus_dimension",
        "center_dimension",
        "left_annihilator_dimension",
        "right_annihilator_dimension",
        "two_sided_annihilator_dimension",
        "product_span_dimension",
        "product_span_ideal_dimension",
        "derivation_dimension",
        "ideal_classification_status",
        "isomorphism_rejection_only",
        "algorithm",
        "algorithm_version",
    )
    return {
        **{field: getattr(report, field) for field in fields},
        "associativityReport": _law_body(report.associativity_report),
        "commutativityReport": _law_body(report.commutativity_report),
        "conventions": list(report.conventions),
        "unitReport": _unit_body(report.unit_report),
    }


def _algebra_body(algebra: FiniteMultilinearStructure) -> dict[str, object]:
    return {
        "arity": algebra.arity,
        "basisLabels": list(algebra.module.basis.labels),
        "coefficientDomain": "QQ",
        "entries": [
            [*key, *_rational(cast(Any, coefficient).value)]
            for key, coefficient in algebra.operation.constants.entries
        ],
        "rank": algebra.module.rank,
    }


def _body(value: LinearAnalysisAdapterReport) -> dict[str, object]:
    return {
        "algebra": _algebra_body(value.algebra),
        "algorithm": value.algorithm,
        "algorithmVersion": value.algorithm_version,
        "bounds": _bounds_body(value.bounds),
        "capabilityStatuses": [list(item) for item in value.capability_statuses],
        "center": _subspace_body(value.center),
        "complete": value.complete,
        "derivations": _derivation_body(value.derivations),
        "exact": value.exact,
        "fingerprint": _fingerprint_body(value.fingerprint),
        "hypotheses": list(value.hypotheses),
        "ideals": _ideal_body(value.ideals),
        "leftNucleus": _subspace_body(value.left_nucleus),
        "middleNucleus": _subspace_body(value.middle_nucleus),
        "nucleus": _subspace_body(value.nucleus),
        "productSpanDimension": value.product_span_dimension,
        "rightNucleus": _subspace_body(value.right_nucleus),
        "schemaType": "anyalgebra.census.linear_analysis_adapter",
        "schemaVersion": _SCHEMA_VERSION,
        "unit": _unit_body(value.unit),
    }


def linear_analysis_adapter_record(
    value: LinearAnalysisAdapterReport,
) -> dict[str, object]:
    if type(value) is not LinearAnalysisAdapterReport:
        raise LinearAnalysisAdapterError(
            field="report", reason="must be exact LinearAnalysisAdapterReport"
        )
    expected = analyze_module_backed_algebra(value.algebra, options=value.bounds)
    if value != expected:
        raise LinearAnalysisAdapterError(field="report", reason="content drift")
    body = _body(value)
    semantic_hash = _semantic_hash(body)
    if semantic_hash != value.semantic_hash:
        raise LinearAnalysisAdapterError(field="semantic_hash", reason="content drift")
    return {
        **body,
        "contentHash": {
            "algorithm": semantic_hash.algorithm,
            "digest": semantic_hash.digest,
        },
    }


def linear_analysis_adapter_canonical_bytes(
    value: LinearAnalysisAdapterReport,
) -> bytes:
    return _encoded(linear_analysis_adapter_record(value))


__all__ = (
    "LinearAnalysisAdapterError",
    "LinearAnalysisAdapterReport",
    "analyze_module_backed_algebra",
    "linear_analysis_adapter_canonical_bytes",
    "linear_analysis_adapter_record",
)
