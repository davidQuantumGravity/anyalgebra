"""Strict routing from module-backed QQ algebras to exact linear analyses."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from anyalgebra.algebra.multilinear import BasisProductTable, FiniteMultilinearStructure
from anyalgebra.analysis import elementary, fingerprint
from anyalgebra.analysis.elementary import AnalysisBounds
from anyalgebra.census.linear_adapter import (
    LinearAnalysisAdapterError,
    analyze_module_backed_algebra,
    linear_analysis_adapter_canonical_bytes,
)
from anyalgebra.census.spec import CensusSpecCore
from anyalgebra.core.domains import QQ, ZZ
from anyalgebra.core.modules import Basis, FreeModule


def _algebra(
    cells: tuple[dict[int, int], ...], *, arity: int = 2
) -> FiniteMultilinearStructure:
    domain = QQ()
    module = FreeModule(domain, Basis(("e0", "e1"), coefficient_domain=domain))
    return FiniteMultilinearStructure.from_tables(
        module,
        BasisProductTable.from_cells(
            module,
            arity,
            tuple(
                {index: domain.element(value) for index, value in cell.items()}
                for cell in cells
            ),
        ),
    )


def _dual() -> FiniteMultilinearStructure:
    return _algebra(({0: 1}, {1: 1}, {1: 1}, {}))


def _zero(rank: int) -> FiniteMultilinearStructure:
    domain = QQ()
    module = FreeModule(
        domain, Basis(tuple(f"e{i}" for i in range(rank)), coefficient_domain=domain)
    )
    return FiniteMultilinearStructure.from_tables(
        module, BasisProductTable.from_cells(module, 2, ({} for _ in range(rank**2)))
    )


def test_qq_adapter_reproduces_every_existing_report_family() -> None:
    algebra = _dual()
    result = analyze_module_backed_algebra(algebra)

    assert result.algebra is algebra
    assert result.left_nucleus.basis == elementary.left_nucleus(algebra).basis
    assert result.middle_nucleus.basis == elementary.middle_nucleus(algebra).basis
    assert result.right_nucleus.basis == elementary.right_nucleus(algebra).basis
    assert result.nucleus.basis == elementary.nucleus(algebra).basis
    assert result.center.basis == elementary.center(algebra).basis
    direct_ideals = fingerprint.ideals(algebra)
    assert tuple(item.basis for item in result.ideals.ideals) == tuple(
        item.basis for item in direct_ideals.ideals
    )
    assert tuple(matrix.entries for matrix in result.derivations.basis) == tuple(
        matrix.entries for matrix in fingerprint.derivation_algebra(algebra).basis
    )
    assert result.fingerprint == fingerprint.algebra_fingerprint(algebra)
    assert result.unit is result.fingerprint.unit_report
    assert result.product_span_dimension == result.fingerprint.product_span_dimension


def test_adapter_preserves_complete_and_incomplete_evidence_roles() -> None:
    result = analyze_module_backed_algebra(_dual())
    statuses = dict(result.capability_statuses)

    assert statuses["nuclei"] == statuses["center"] == "complete_exact"
    assert statuses["derivations"] == statuses["unit"] == "complete_exact"
    assert statuses["ideals"] == "incomplete_canonical_discovery"
    assert statuses["fingerprint"] == "rejection_only"
    assert result.complete is False
    assert result.exact is True
    assert result.hypotheses[0] == "literal QQ finite free module"


def test_bare_finite_carrier_cannot_enter_linear_adapter() -> None:
    core = CensusSpecCore.create(carrier_size=2, arity=2, corpus_name="bare")
    with pytest.raises(LinearAnalysisAdapterError) as caught:
        analyze_module_backed_algebra(core)  # type: ignore[arg-type]

    assert caught.value.field == "algebra"
    assert "module-backed" in caught.value.reason


def test_non_qq_and_nonbinary_module_structures_fail_closed() -> None:
    domain = ZZ()
    module = FreeModule(domain, Basis(("e",), coefficient_domain=domain))
    integer_algebra = FiniteMultilinearStructure.from_tables(
        module, BasisProductTable.from_cells(module, 2, ({0: domain.element(1)},))
    )
    with pytest.raises(LinearAnalysisAdapterError) as caught:
        analyze_module_backed_algebra(integer_algebra)
    assert caught.value.field == "coefficient_domain"

    unary = _algebra(({0: 1}, {1: 1}), arity=1)
    with pytest.raises(LinearAnalysisAdapterError) as caught:
        analyze_module_backed_algebra(unary)
    assert caught.value.field == "arity"


def test_declared_bounds_are_snapshotted_and_route_to_all_reports() -> None:
    supplied = AnalysisBounds(max_constraints=200, max_elimination_work=2000)
    result = analyze_module_backed_algebra(_dual(), options=supplied)

    assert result.bounds is not supplied
    assert result.left_nucleus.bounds.max_constraints == 200
    assert result.ideals.bounds.max_elimination_work == 2000
    assert result.derivations.bounds.max_constraints == 200
    assert result.fingerprint.bounds.max_elimination_work == 2000


def test_zero_dimensional_module_is_an_exact_boundary() -> None:
    result = analyze_module_backed_algebra(_zero(0))

    assert result.nucleus.dimension == result.center.dimension == 0
    assert result.derivations.dimension == 0
    assert result.unit.unit_vector == ()
    assert result.product_span_dimension == 0


def test_result_is_sealed_canonical_and_replayed_before_serialization() -> None:
    algebra = _dual()
    first_result = analyze_module_backed_algebra(algebra)
    second_result = analyze_module_backed_algebra(algebra)
    first = linear_analysis_adapter_canonical_bytes(first_result)

    assert first_result == second_result
    assert first == linear_analysis_adapter_canonical_bytes(second_result)
    assert (
        json.loads(first)["contentHash"]["digest"] == first_result.semantic_hash.digest
    )
    with pytest.raises(LinearAnalysisAdapterError, match="factory-owned"):
        type(first_result)()
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("Derived", (type(first_result),), {})
    with pytest.raises(FrozenInstanceError):
        first_result.complete = True  # type: ignore[misc]
    object.__setattr__(first_result, "product_span_dimension", 0)
    with pytest.raises(LinearAnalysisAdapterError, match="content drift"):
        linear_analysis_adapter_canonical_bytes(first_result)
