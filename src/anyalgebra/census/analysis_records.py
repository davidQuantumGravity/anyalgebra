"""Capability-aware aggregate records for finite and module-backed analyses."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from anyalgebra.algebra.multilinear import FiniteMultilinearStructure
from anyalgebra.analysis.elementary import AnalysisBounds
from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import SemanticHash

from .analysis import (
    DisprovedLaw,
    FiniteElementAnalysis,
    FiniteLawProfile,
    FiniteMagmaSubobjectAnalysis,
    analyze_finite_elements,
    analyze_finite_laws,
    analyze_finite_magma_subobjects,
    element_profile_record,
    finite_magma_subobject_record,
    law_profile_record,
)
from .equivalence import EquivalencePolicy
from .linear_adapter import (
    LinearAnalysisAdapterReport,
    analyze_module_backed_algebra,
    linear_analysis_adapter_canonical_bytes,
)
from .spec import CensusSpecCore
from .subobjects import (
    FiniteSubobjectAnalysis,
    FiniteSubobjectBounds,
    analyze_finite_subobjects,
    finite_subobject_record,
)


_SCHEMA_VERSION = 1
_STATUSES = frozenset(("complete", "bounded", "rejection_only", "unsupported"))


class AnalysisRecordError(AnyAlgebraError, ValueError):
    """An aggregate analysis field, subject, or retained record is invalid."""

    def __init__(self, *, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid capability-aware analysis {field}: {reason}")


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class AnalysisField:
    """One analysis capability with explicit evidence semantics."""

    name: str
    status: str
    algorithm: str
    hypotheses: tuple[str, ...]
    bounds: tuple[tuple[str, int], ...]
    work: tuple[tuple[str, int], ...]
    witnesses: tuple[str, ...]
    payload_hash: SemanticHash | None

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AnalysisRecordError(field="analysis_field", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("AnalysisField cannot be subclassed")

    @classmethod
    def _create(
        cls,
        *,
        name: str,
        status: str,
        algorithm: str,
        hypotheses: tuple[str, ...],
        bounds: tuple[tuple[str, int], ...] = (),
        work: tuple[tuple[str, int], ...] = (),
        witnesses: tuple[str, ...] = (),
        payload_hash: SemanticHash | None = None,
    ) -> AnalysisField:
        if status not in _STATUSES:
            raise AnalysisRecordError(field="status", reason="is not allow-listed")
        if not name or not algorithm or not hypotheses:
            raise AnalysisRecordError(
                field="analysis_field", reason="requires name, algorithm, hypotheses"
            )
        if status == "unsupported" and payload_hash is not None:
            raise AnalysisRecordError(
                field="payload_hash", reason="unsupported field cannot carry payload"
            )
        if status != "unsupported" and type(payload_hash) is not SemanticHash:
            raise AnalysisRecordError(
                field="payload_hash", reason="supported field requires exact hash"
            )
        value = object.__new__(AnalysisField)
        for field, item in (
            ("name", name),
            ("status", status),
            ("algorithm", algorithm),
            ("hypotheses", hypotheses),
            ("bounds", bounds),
            ("work", work),
            ("witnesses", witnesses),
            ("payload_hash", payload_hash),
        ):
            object.__setattr__(value, field, item)
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is AnalysisField and _field_identity(
            self
        ) == _field_identity(other)


def _field_identity(value: AnalysisField) -> tuple[object, ...]:
    return (
        value.name,
        value.status,
        value.algorithm,
        value.hypotheses,
        value.bounds,
        value.work,
        value.witnesses,
        value.payload_hash,
    )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class AnalysisRecord:
    """One schema with disjoint finite-carrier and module-backed payloads."""

    subject_kind: str
    finite_core: CensusSpecCore | None
    finite_outputs: tuple[int, ...] | None
    equivalence: EquivalencePolicy | None
    linear_algebra: FiniteMultilinearStructure | None
    law_profile: FiniteLawProfile | None
    element_analysis: FiniteElementAnalysis | None
    magma_analysis: FiniteMagmaSubobjectAnalysis | None
    subobject_analysis: FiniteSubobjectAnalysis | None
    linear_analysis: LinearAnalysisAdapterReport | None
    fields: tuple[AnalysisField, ...]
    complete: bool
    algorithm: str
    algorithm_version: int
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AnalysisRecordError(field="analysis_record", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("AnalysisRecord cannot be subclassed")

    def __eq__(self, other: object) -> bool:
        if type(other) is not AnalysisRecord:
            return False
        same_subject = (
            self.finite_core is other.finite_core
            and self.finite_outputs == other.finite_outputs
            and self.equivalence == other.equivalence
            and self.linear_algebra is other.linear_algebra
        )
        return same_subject and (
            self.subject_kind,
            self.fields,
            self.complete,
            self.semantic_hash,
        ) == (
            other.subject_kind,
            other.fields,
            other.complete,
            other.semantic_hash,
        )


def _unsupported(name: str, reason: str) -> AnalysisField:
    return AnalysisField._create(
        name=name,
        status="unsupported",
        algorithm="not_applicable",
        hypotheses=(reason,),
    )


def _field(
    name: str,
    status: str,
    algorithm: str,
    hypotheses: tuple[str, ...],
    payload_hash: SemanticHash,
    *,
    bounds: tuple[tuple[str, int], ...] = (),
    work: tuple[tuple[str, int], ...] = (),
    witnesses: tuple[str, ...] = (),
) -> AnalysisField:
    return AnalysisField._create(
        name=name,
        status=status,
        algorithm=algorithm,
        hypotheses=hypotheses,
        bounds=bounds,
        work=work,
        witnesses=witnesses,
        payload_hash=payload_hash,
    )


def _make_record(
    *,
    subject_kind: str,
    finite_core: CensusSpecCore | None = None,
    finite_outputs: tuple[int, ...] | None = None,
    equivalence: EquivalencePolicy | None = None,
    linear_algebra: FiniteMultilinearStructure | None = None,
    law_profile: FiniteLawProfile | None = None,
    element_analysis: FiniteElementAnalysis | None = None,
    magma_analysis: FiniteMagmaSubobjectAnalysis | None = None,
    subobject_analysis: FiniteSubobjectAnalysis | None = None,
    linear_analysis: LinearAnalysisAdapterReport | None = None,
    fields: tuple[AnalysisField, ...],
) -> AnalysisRecord:
    value = object.__new__(AnalysisRecord)
    for field, item in (
        ("subject_kind", subject_kind),
        ("finite_core", finite_core),
        ("finite_outputs", finite_outputs),
        ("equivalence", equivalence),
        ("linear_algebra", linear_algebra),
        ("law_profile", law_profile),
        ("element_analysis", element_analysis),
        ("magma_analysis", magma_analysis),
        ("subobject_analysis", subobject_analysis),
        ("linear_analysis", linear_analysis),
        ("fields", fields),
        ("complete", all(field.status == "complete" for field in fields)),
        ("algorithm", "anyalgebra.capability_aware_analysis_assembly"),
        ("algorithm_version", 1),
        ("semantic_hash", SemanticHash("sha256", "0" * 64)),
    ):
        object.__setattr__(value, field, item)
    object.__setattr__(value, "semantic_hash", _semantic_hash(_body(value)))
    return value


def assemble_finite_carrier_analysis(
    core: CensusSpecCore,
    outputs: tuple[int, ...],
    equivalence: EquivalencePolicy,
    *,
    nilpotence_zero: int | None = None,
    max_nilpotence_index: int | None = None,
    subobject_options: FiniteSubobjectBounds | None = None,
) -> AnalysisRecord:
    """Assemble exact carrier results without importing module capabilities."""
    # Subobject growth is the strictest configurable finite-carrier ceiling.
    # Run its validated preflight path before any other exhaustive analysis so
    # an out-of-domain carrier cannot consume a law or partition traversal.
    subobjects = analyze_finite_subobjects(core, outputs, options=subobject_options)
    laws = analyze_finite_laws(core, outputs)
    elements = analyze_finite_elements(
        core,
        outputs,
        equivalence,
        nilpotence_zero=nilpotence_zero,
        max_nilpotence_index=max_nilpotence_index,
    )
    magma = analyze_finite_magma_subobjects(core, outputs)
    law_witnesses = tuple(
        f"{result.name}:{','.join(map(str, result.witness.assignment))}"
        for result in laws.results
        if type(result) is DisprovedLaw
    )
    finite_hypothesis = ("one total operation on the declared finite carrier",)
    subobject_bounds = (
        ("max_carrier_size", subobjects.bounds.max_carrier_size),
        ("max_subset_candidates", subobjects.bounds.max_subset_candidates),
        ("max_partition_candidates", subobjects.bounds.max_partition_candidates),
        ("max_closure_checks", subobjects.bounds.max_closure_checks),
        ("max_compatibility_checks", subobjects.bounds.max_compatibility_checks),
    )
    magma_field = (
        _field(
            "magma_nuclei_center",
            "complete",
            "anyalgebra.finite_magma_complete_grid",
            ("the declared operation is binary",),
            magma.semantic_hash,
            work=(
                ("associator_assignments", magma.associator_assignments_checked),
                ("commutator_pairs", magma.commutator_pairs_checked),
            ),
        )
        if magma.complete
        else _unsupported(
            "magma_nuclei_center", "magma nuclei and commutant require arity two"
        )
    )
    nilpotence_field = (
        _field(
            "nilpotence",
            "bounded",
            "anyalgebra.left_associated_power_search",
            ("caller declared an absorbing zero and finite maximum index",),
            elements.semantic_hash,
            bounds=(("max_index", max_nilpotence_index),),
            work=(("carrier_elements", core.carrier_size),),
        )
        if max_nilpotence_index is not None
        else _unsupported("nilpotence", "no zero and finite power bound declared")
    )
    fields = (
        _field(
            "finite_carrier_laws",
            "complete",
            "anyalgebra.finite_complete_grid_laws",
            finite_hypothesis,
            laws.semantic_hash,
            work=(("assignments", laws.total_evaluations),),
            witnesses=law_witnesses,
        ),
        _field(
            "element_profiles",
            "complete",
            "anyalgebra.finite_element_profiles",
            finite_hypothesis,
            elements.semantic_hash,
            work=(("carrier_elements", elements.examined_element_count),),
        ),
        nilpotence_field,
        magma_field,
        _field(
            "substructures",
            "complete",
            subobjects.algorithm,
            subobjects.hypotheses,
            subobjects.semantic_hash,
            bounds=subobject_bounds,
            work=(
                ("subset_candidates", subobjects.subset_candidates_examined),
                ("closure_assignments", subobjects.closure_assignments_checked),
            ),
        ),
        _field(
            "congruences",
            "complete",
            subobjects.algorithm,
            subobjects.hypotheses,
            subobjects.semantic_hash,
            bounds=subobject_bounds,
            work=(
                ("partition_candidates", subobjects.partition_candidates_examined),
                ("compatibility_pairs", subobjects.compatibility_pairs_checked),
            ),
        ),
        _field(
            "invariant_colors",
            "rejection_only",
            "anyalgebra.incomplete_element_color_partition",
            ("color agreement is not proof of isomorphism",),
            elements.semantic_hash,
            work=(("carrier_elements", elements.examined_element_count),),
        ),
        _unsupported("linear_laws", "finite carrier has no scalar module"),
        _unsupported("linear_nuclei_center", "finite carrier has no scalar module"),
        _unsupported("linear_ideals", "finite carrier has no scalar module"),
        _unsupported("linear_derivations", "finite carrier has no scalar module"),
        _unsupported("unit", "finite carrier has no additive linear unit solve"),
        _unsupported("product_span", "finite carrier has no linear span"),
        _unsupported("linear_fingerprint", "finite carrier has no scalar module"),
    )
    return _make_record(
        subject_kind="finite_carrier",
        finite_core=core,
        finite_outputs=outputs,
        equivalence=equivalence,
        law_profile=laws,
        element_analysis=elements,
        magma_analysis=magma,
        subobject_analysis=subobjects,
        fields=fields,
    )


def assemble_module_backed_analysis(
    algebra: FiniteMultilinearStructure,
    *,
    options: AnalysisBounds | None = None,
) -> AnalysisRecord:
    """Assemble existing exact QQ linear reports without finite-carrier claims."""
    linear = analyze_module_backed_algebra(algebra, options=options)
    bounds = (
        ("max_dimension", linear.bounds.max_dimension),
        ("max_constraints", linear.bounds.max_constraints),
        ("max_elimination_work", linear.bounds.max_elimination_work),
    )
    payload = linear.semantic_hash
    no_finite = "QQ module algebra has no finite carrier enumeration"
    fields = (
        _unsupported("finite_carrier_laws", no_finite),
        _unsupported("element_profiles", no_finite),
        _unsupported("nilpotence", "no finite carrier power bound declared"),
        _unsupported("magma_nuclei_center", no_finite),
        _unsupported("substructures", no_finite),
        _unsupported("congruences", no_finite),
        _unsupported("invariant_colors", no_finite),
        _field(
            "linear_laws",
            "complete",
            linear.fingerprint.commutativity_report.algorithm,
            (linear.fingerprint.commutativity_report.reduction_hypothesis,),
            payload,
            bounds=bounds,
            work=(
                (
                    "commutativity_assignments",
                    linear.fingerprint.commutativity_pairs_checked,
                ),
                (
                    "associativity_assignments",
                    linear.fingerprint.associativity_triples_checked,
                ),
            ),
        ),
        _field(
            "linear_nuclei_center",
            "complete",
            linear.nucleus.algorithm,
            linear.hypotheses,
            payload,
            bounds=bounds,
            work=(
                (
                    "checked_constraints",
                    sum(
                        report.checked_constraints
                        for report in (
                            linear.left_nucleus,
                            linear.middle_nucleus,
                            linear.right_nucleus,
                            linear.nucleus,
                            linear.center,
                        )
                    ),
                ),
            ),
        ),
        _field(
            "linear_ideals",
            "bounded",
            linear.ideals.algorithm,
            linear.hypotheses,
            payload,
            bounds=bounds,
            work=(("product_evaluations", linear.ideals.evaluations),),
        ),
        _field(
            "linear_derivations",
            "complete",
            linear.derivations.algorithm,
            linear.hypotheses,
            payload,
            bounds=bounds,
            work=(
                ("constraints", linear.derivations.constraints),
                ("elimination_work", linear.derivations.elimination_work),
            ),
        ),
        _field(
            "unit",
            "complete",
            linear.unit.algorithm,
            linear.hypotheses,
            payload,
            bounds=bounds,
            work=(("scalar_equations", linear.unit.scalar_equations),),
        ),
        _field(
            "product_span",
            "complete",
            "anyalgebra.exact_product_span",
            linear.hypotheses,
            payload,
            bounds=bounds,
            work=(("dimension", linear.product_span_dimension),),
        ),
        _field(
            "linear_fingerprint",
            "rejection_only",
            linear.fingerprint.algorithm,
            linear.hypotheses,
            payload,
            bounds=bounds,
            work=(
                ("commutativity_pairs", linear.fingerprint.commutativity_pairs_checked),
                (
                    "associativity_triples",
                    linear.fingerprint.associativity_triples_checked,
                ),
            ),
        ),
    )
    return _make_record(
        subject_kind="module_backed_algebra",
        linear_algebra=algebra,
        linear_analysis=linear,
        fields=fields,
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


def _hash_body(value: SemanticHash | None) -> dict[str, str] | None:
    if value is None:
        return None
    return {"algorithm": value.algorithm, "digest": value.digest}


def _field_body(value: AnalysisField) -> dict[str, object]:
    return {
        "algorithm": value.algorithm,
        "bounds": [[name, bound] for name, bound in value.bounds],
        "hypotheses": list(value.hypotheses),
        "name": value.name,
        "payloadHash": _hash_body(value.payload_hash),
        "status": value.status,
        "witnesses": list(value.witnesses),
        "work": [[name, count] for name, count in value.work],
    }


def _finite_subject_body(value: AnalysisRecord) -> dict[str, object]:
    assert value.finite_core is not None
    assert value.finite_outputs is not None
    assert value.equivalence is not None
    core = value.finite_core
    policy = value.equivalence
    return {
        "arity": core.arity,
        "carrierSize": core.carrier_size,
        "corpusName": core.corpus_name,
        "distinguishedElements": [list(item) for item in core.distinguished_elements],
        "equivalence": {
            "fixedElements": list(policy.fixed_elements),
            "kind": policy.kind,
            "permutationCount": policy.permutation_count,
            "sortBlocks": [
                {"elements": list(block.elements), "name": block.name}
                for block in policy.sort_blocks
            ],
        },
        "outputs": list(value.finite_outputs),
    }


def _body(value: AnalysisRecord) -> dict[str, object]:
    if value.subject_kind == "finite_carrier":
        assert value.law_profile is not None
        assert value.element_analysis is not None
        assert value.magma_analysis is not None
        assert value.subobject_analysis is not None
        subject = _finite_subject_body(value)
        payloads: dict[str, object] = {
            "elementAnalysis": element_profile_record(value.element_analysis),
            "lawProfile": law_profile_record(value.law_profile),
            "magmaAnalysis": finite_magma_subobject_record(value.magma_analysis),
            "subobjectAnalysis": finite_subobject_record(value.subobject_analysis),
        }
    else:
        assert value.linear_analysis is not None
        subject = {
            "linearAnalysisHash": _hash_body(value.linear_analysis.semantic_hash)
        }
        payloads = {
            "linearAnalysis": json.loads(
                linear_analysis_adapter_canonical_bytes(value.linear_analysis)
            )
        }
    return {
        "algorithm": value.algorithm,
        "algorithmVersion": value.algorithm_version,
        "complete": value.complete,
        "fields": [_field_body(field) for field in value.fields],
        "payloads": payloads,
        "schemaType": "anyalgebra.census.capability_analysis_record",
        "schemaVersion": _SCHEMA_VERSION,
        "subject": subject,
        "subjectKind": value.subject_kind,
    }


def _replay(value: AnalysisRecord) -> AnalysisRecord:
    if value.subject_kind == "finite_carrier":
        if (
            value.finite_core is None
            or value.finite_outputs is None
            or value.equivalence is None
            or value.element_analysis is None
            or value.subobject_analysis is None
        ):
            raise AnalysisRecordError(field="analysis_record", reason="content drift")
        return assemble_finite_carrier_analysis(
            value.finite_core,
            value.finite_outputs,
            value.equivalence,
            nilpotence_zero=value.element_analysis.nilpotence_zero,
            max_nilpotence_index=value.element_analysis.max_nilpotence_index,
            subobject_options=value.subobject_analysis.bounds,
        )
    if value.subject_kind == "module_backed_algebra":
        if value.linear_algebra is None or value.linear_analysis is None:
            raise AnalysisRecordError(field="analysis_record", reason="content drift")
        return assemble_module_backed_analysis(
            value.linear_algebra, options=value.linear_analysis.bounds
        )
    raise AnalysisRecordError(field="subject_kind", reason="content drift")


def analysis_record(value: AnalysisRecord) -> dict[str, object]:
    if type(value) is not AnalysisRecord:
        raise AnalysisRecordError(
            field="analysis_record", reason="must be exact AnalysisRecord"
        )
    expected = _replay(value)
    if value != expected:
        raise AnalysisRecordError(field="analysis_record", reason="content drift")
    body = _body(value)
    semantic_hash = _semantic_hash(body)
    if semantic_hash != value.semantic_hash:
        raise AnalysisRecordError(field="semantic_hash", reason="content drift")
    return {**body, "contentHash": _hash_body(semantic_hash)}


def analysis_record_canonical_bytes(value: AnalysisRecord) -> bytes:
    return _encoded(analysis_record(value))


__all__ = (
    "AnalysisField",
    "AnalysisRecord",
    "AnalysisRecordError",
    "analysis_record",
    "analysis_record_canonical_bytes",
    "assemble_finite_carrier_analysis",
    "assemble_module_backed_analysis",
)
