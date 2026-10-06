"""Capability-aware finite-carrier and module-backed analysis records."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from anyalgebra.algebra.multilinear import BasisProductTable, FiniteMultilinearStructure
from anyalgebra.census.analysis_records import (
    AnalysisField,
    AnalysisRecord,
    AnalysisRecordError,
    analysis_record_canonical_bytes,
    assemble_finite_carrier_analysis,
    assemble_module_backed_analysis,
)
from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.spec import CensusSpecCore
from anyalgebra.core.domains import QQ
from anyalgebra.core.modules import Basis, FreeModule


def _core(size: int, arity: int = 2) -> CensusSpecCore:
    return CensusSpecCore.create(
        carrier_size=size,
        arity=arity,
        corpus_name=f"analysis-record-{size}-{arity}",
    )


def _dual() -> FiniteMultilinearStructure:
    domain = QQ()
    module = FreeModule(domain, Basis(("1", "e"), coefficient_domain=domain))
    return FiniteMultilinearStructure.from_tables(
        module,
        BasisProductTable.from_cells(
            module,
            2,
            (
                {0: domain.element(1)},
                {1: domain.element(1)},
                {1: domain.element(1)},
                {},
            ),
        ),
    )


def _fields(record: AnalysisRecord) -> dict[str, AnalysisField]:
    return {field.name: field for field in record.fields}


def test_finite_carrier_record_populates_only_supported_capabilities() -> None:
    core = _core(2)
    result = assemble_finite_carrier_analysis(
        core, (0, 1, 1, 0), EquivalencePolicy.relabeling(core)
    )
    fields = _fields(result)

    assert result.subject_kind == "finite_carrier"
    for name in (
        "finite_carrier_laws",
        "element_profiles",
        "magma_nuclei_center",
        "substructures",
        "congruences",
    ):
        assert fields[name].status == "complete"
        assert fields[name].payload_hash is not None
    assert fields["invariant_colors"].status == "rejection_only"
    for name in ("linear_ideals", "linear_derivations", "linear_fingerprint"):
        assert fields[name].status == "unsupported"
        assert fields[name].payload_hash is None
    assert result.linear_analysis is None
    assert result.law_profile is not None
    assert result.subobject_analysis is not None


def test_declared_nilpotence_is_bounded_and_undeclared_is_unsupported() -> None:
    core = _core(2)
    policy = EquivalencePolicy.relabeling(core)
    undeclared = _fields(assemble_finite_carrier_analysis(core, (0, 0, 0, 0), policy))
    declared = _fields(
        assemble_finite_carrier_analysis(
            core,
            (0, 0, 0, 0),
            policy,
            nilpotence_zero=0,
            max_nilpotence_index=3,
        )
    )

    assert undeclared["nilpotence"].status == "unsupported"
    assert declared["nilpotence"].status == "bounded"
    assert declared["nilpotence"].bounds == (("max_index", 3),)


def test_nonbinary_magma_fields_are_unsupported_but_other_fields_complete() -> None:
    core = _core(2, 1)
    result = assemble_finite_carrier_analysis(
        core, (1, 0), EquivalencePolicy.relabeling(core)
    )
    fields = _fields(result)

    assert fields["magma_nuclei_center"].status == "unsupported"
    assert fields["finite_carrier_laws"].status == "complete"
    assert fields["substructures"].status == "complete"
    assert fields["congruences"].status == "complete"


def test_module_record_keeps_exact_bounded_and_rejection_roles_distinct() -> None:
    result = assemble_module_backed_analysis(_dual())
    fields = _fields(result)

    assert result.subject_kind == "module_backed_algebra"
    assert fields["linear_nuclei_center"].status == "complete"
    assert fields["linear_ideals"].status == "bounded"
    assert fields["linear_derivations"].status == "complete"
    assert fields["unit"].status == "complete"
    assert fields["product_span"].status == "complete"
    assert fields["linear_fingerprint"].status == "rejection_only"
    assert fields["finite_carrier_laws"].status == "unsupported"
    assert result.linear_analysis is not None
    assert result.law_profile is None


def test_every_field_has_algorithm_hypotheses_bounds_work_and_status() -> None:
    core = _core(2)
    result = assemble_finite_carrier_analysis(
        core, (0, 0, 0, 1), EquivalencePolicy.relabeling(core)
    )
    for field in result.fields:
        assert field.algorithm
        assert field.hypotheses
        assert type(field.bounds) is tuple
        assert type(field.work) is tuple
        assert field.status in {"complete", "bounded", "rejection_only", "unsupported"}


def test_analysis_fields_and_records_are_factory_owned() -> None:
    with pytest.raises(AnalysisRecordError, match="factory-owned"):
        AnalysisField()
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("Derived", (AnalysisField,), {})


def test_record_is_sealed_canonical_and_replayed_before_serialization() -> None:
    core = _core(2)
    policy = EquivalencePolicy.relabeling(core)
    first_result = assemble_finite_carrier_analysis(core, (0, 1, 1, 0), policy)
    second_result = assemble_finite_carrier_analysis(core, (0, 1, 1, 0), policy)
    first = analysis_record_canonical_bytes(first_result)

    assert first_result == second_result
    assert first == analysis_record_canonical_bytes(second_result)
    assert (
        json.loads(first)["contentHash"]["digest"] == first_result.semantic_hash.digest
    )
    with pytest.raises(AnalysisRecordError, match="factory-owned"):
        type(first_result)()
    with pytest.raises(FrozenInstanceError):
        first_result.subject_kind = "other"  # type: ignore[misc]
    object.__setattr__(first_result, "fields", ())
    with pytest.raises(AnalysisRecordError, match="content drift"):
        analysis_record_canonical_bytes(first_result)
