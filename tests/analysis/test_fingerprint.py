"""Contracts for bounded exact ideals, derivations, and fingerprints."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from anyalgebra.algebra.multilinear import (
    BasisProductTable,
    FiniteMultilinearStructure,
    StructureConstants,
)
from anyalgebra.analysis import elementary, fingerprint
from anyalgebra.analysis.elementary import AnalysisBounds
from anyalgebra.analysis.fingerprint import (
    AlgebraFingerprint,
    DerivationReport,
    FingerprintAnalysisError,
    IdealData,
    IdealSearchResult,
    algebra_fingerprint,
    derivation_algebra,
    ideals,
)
from anyalgebra.core.domains import DomainElement, QQ, ZZ
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.core.rational import Rational
from anyalgebra.presentations.basis_change import ExactBasisIsomorphism, change_basis
from anyalgebra.fixtures.composition import octonion_fixture, quaternion_fixture
from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.structures.operations import PartialOperation
from anyalgebra.structures.signatures import OperationSymbol, Signature
from anyalgebra.structures.structure import StructureBuilder


def _algebra(cells: list[dict[int, object]]) -> FiniteMultilinearStructure:
    domain = QQ()
    module = FreeModule(domain, Basis(("e0", "e1"), coefficient_domain=domain))
    return FiniteMultilinearStructure.from_tables(
        module,
        BasisProductTable.from_cells(
            module,
            2,
            [
                {index: domain.element(value) for index, value in cell.items()}
                for cell in cells
            ],
        ),
    )


def _dual() -> FiniteMultilinearStructure:
    return _algebra([{0: 1}, {1: 1}, {1: 1}, {}])


def _zero(rank: int) -> FiniteMultilinearStructure:
    domain = QQ()
    module = FreeModule(
        domain, Basis(tuple(f"e{i}" for i in range(rank)), coefficient_domain=domain)
    )
    return FiniteMultilinearStructure.from_tables(
        module, BasisProductTable.from_cells(module, 2, [{} for _ in range(rank**2)])
    )


def _qq_lift(value: FiniteMultilinearStructure) -> FiniteMultilinearStructure:
    domain = QQ()
    module = FreeModule(
        domain, Basis(value.module.basis.labels, coefficient_domain=domain)
    )
    constants = StructureConstants.from_sparse(
        (module.basis, module.basis),
        module.basis,
        tuple(
            (key, domain.element(coefficient.value))
            for key, coefficient in value.operation.constants.entries
        ),
    )
    return FiniteMultilinearStructure.from_structure_constants(module, constants)


def _apply(
    matrix: tuple[tuple[Rational, ...], ...], vector: tuple[Rational, ...]
) -> tuple[Rational, ...]:
    return tuple(
        sum_rational(
            tuple(
                matrix[row][column].multiply(vector[column])
                for column in range(len(vector))
            )
        )
        for row in range(len(matrix))
    )


def sum_rational(values: tuple[Rational, ...]) -> Rational:
    total = Rational(0)
    for value in values:
        total = total.add(value)
    return total


def _sheared_dual() -> FiniteMultilinearStructure:
    """Transport dual numbers through e0'=e0 and e1'=e0+2e1."""
    source = _dual()
    transform = ((Rational(1), Rational(1)), (Rational(0), Rational(2)))
    inverse = ((Rational(1), Rational(-1, 2)), (Rational(0), Rational(1, 2)))
    vectors = (
        (Rational(1), Rational(0)),
        (Rational(0), Rational(1)),
    )
    cells: list[dict[int, object]] = []
    for left in vectors:
        for right in vectors:
            old_left, old_right = _apply(transform, left), _apply(transform, right)
            product = elementary._product(source, old_left, old_right)
            new = _apply(inverse, product)
            cells.append(
                {index: value for index, value in enumerate(new) if value.numerator}
            )
    return _algebra(cells)


def _verify_leibniz(
    algebra: FiniteMultilinearStructure, report: DerivationReport
) -> None:
    rank = algebra.module.rank
    for matrix in report.basis:
        assert matrix.parent is report.matrix_space
        for left in range(rank):
            for right in range(rank):
                product = elementary._product(
                    algebra,
                    fingerprint._basis_vector(rank, left),
                    fingerprint._basis_vector(rank, right),
                )
                image_product = tuple(
                    sum_rational(
                        tuple(
                            product[column].multiply(
                                cast(
                                    DomainElement[Rational],
                                    matrix.entries[row][column],
                                ).value
                            )
                            for column in range(rank)
                        )
                    )
                    for row in range(rank)
                )
                image_left = tuple(
                    cast(DomainElement[Rational], matrix.entries[row][left]).value
                    for row in range(rank)
                )
                image_right = tuple(
                    cast(DomainElement[Rational], matrix.entries[row][right]).value
                    for row in range(rank)
                )
                expected = tuple(
                    elementary._product(
                        algebra, image_left, fingerprint._basis_vector(rank, right)
                    )[row].add(
                        elementary._product(
                            algebra, fingerprint._basis_vector(rank, left), image_right
                        )[row]
                    )
                    for row in range(rank)
                )
                assert image_product == expected


def test_derivation_controls_and_direct_leibniz_check() -> None:
    dual = derivation_algebra(_dual())
    assert (dual.dimension, dual.constraints, dual.elimination_work) == (1, 8, 128)
    assert dual.matrix_space.entry_parent is QQ()
    assert "a*n+j" in dual.convention
    _verify_leibniz(_dual(), dual)

    idempotent = _algebra([{0: 1}, {}, {}, {}])
    assert derivation_algebra(idempotent).dimension == 1
    zero = derivation_algebra(_zero(2))
    assert zero.dimension == 4
    _verify_leibniz(_zero(2), zero)


def test_composition_derivation_controls_over_qq() -> None:
    assert derivation_algebra(_qq_lift(quaternion_fixture())).dimension == 3
    assert (
        derivation_algebra(
            _qq_lift(octonion_fixture()),
            options=AnalysisBounds(max_elimination_work=3_000_000),
        ).dimension
        == 14
    )


def test_ideals_are_deduplicated_actual_subspaces_and_explicitly_incomplete() -> None:
    result = ideals(_zero(2))
    assert result.complete is False
    assert result.status == "incomplete_canonical_discovery"
    assert [
        (value.origins, value.dimension, value.ambient_dimension)
        for value in result.ideals
    ] == [
        (("zero", "product_span_two_sided"), 0, 2),
        (("two_sided_annihilator", "whole"), 2, 2),
    ]
    assert result.evaluations == 12
    assert all(value.closed for value in result.ideals)
    assert ideals(_dual()).ideals[0].origins == ("zero", "two_sided_annihilator")


def test_fingerprint_records_exact_data_not_an_isomorphism_claim() -> None:
    value = algebra_fingerprint(_dual())
    assert (
        value.dimension,
        value.unit_status,
        value.commutative,
        value.associative,
        value.product_span_dimension,
        value.derivation_dimension,
    ) == (2, "present", True, True, 2, 1)
    assert (
        value.left_annihilator_dimension,
        value.right_annihilator_dimension,
        value.two_sided_annihilator_dimension,
        value.commutativity_pairs_checked,
        value.associativity_triples_checked,
        value.unit_scalar_equations_checked,
    ) == (0, 0, 0, 4, 8, 8)
    assert value.isomorphism_rejection_only is True
    assert "incomplete" in value.ideal_classification_status
    assert "incomplete" in value.conventions[-1]
    assert value.unit_report.unit_vector == (Rational(1), Rational(0))
    with pytest.raises(TypeError):
        hash(value)


def test_fingerprint_is_transport_invariant_under_rational_shear_and_bounds() -> None:
    original = algebra_fingerprint(_dual())
    transported = algebra_fingerprint(_sheared_dual())
    assert original == transported
    assert original == algebra_fingerprint(
        _dual(), options=AnalysisBounds(max_constraints=200, max_elimination_work=2000)
    )


def test_zero_dimensional_conventions_and_sealed_reports() -> None:
    value = algebra_fingerprint(_zero(0))
    assert (value.unit_status, value.derivation_dimension, value.center_dimension) == (
        "present",
        0,
        0,
    )
    report = derivation_algebra(_zero(0))
    assert report.matrix_space.rows == report.matrix_space.columns == 0
    assert value.unit_report.unit_vector == ()
    with pytest.raises(FrozenInstanceError):
        setattr(report, "dimension", 5)  # noqa: B010
    for factory in (IdealData, IdealSearchResult, DerivationReport, AlgebraFingerprint):
        with pytest.raises(FingerprintAnalysisError, match="analysis-owned"):
            factory()


def test_bounds_and_typed_rejections_preflight_before_products(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(FingerprintAnalysisError, match="constraint"):
        derivation_algebra(_dual(), options=AnalysisBounds(max_constraints=7))
    with pytest.raises(FingerprintAnalysisError, match="elimination-work"):
        derivation_algebra(_dual(), options=AnalysisBounds(max_elimination_work=127))
    monkeypatch.setattr(
        fingerprint,
        "_product",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not multiply")),
    )
    with pytest.raises(FingerprintAnalysisError, match="constraint"):
        ideals(_dual(), options=AnalysisBounds(max_constraints=23))

    integer = ZZ()
    module = FreeModule(integer, Basis(("x",), coefficient_domain=integer))
    bad_domain = FiniteMultilinearStructure.from_tables(
        module,
        BasisProductTable.from_cells(module, 2, [{0: integer.element(1)}]),
    )
    with pytest.raises(FingerprintAnalysisError, match="literal QQ"):
        algebra_fingerprint(bad_domain)
    with pytest.raises(FingerprintAnalysisError, match="exact FiniteMultilinear"):
        algebra_fingerprint(cast(FiniteMultilinearStructure, object()))


def test_malformed_tables_and_tampered_bounds_are_translated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bounds = AnalysisBounds()
    object.__setattr__(bounds, "max_constraints", True)
    with pytest.raises(FingerprintAnalysisError, match="positive"):
        ideals(_dual(), options=bounds)
    bad = _dual()
    monkeypatch.setattr(
        type(bad.operation),
        "coefficients_for_basis",
        lambda *_args: ((3, QQ().element(1)),),
    )
    with pytest.raises(FingerprintAnalysisError, match="in-range"):
        derivation_algebra(bad)


def test_repr_identity_boundaries_and_nonbinary_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = derivation_algebra(_dual())
    found = ideals(_dual())
    value = algebra_fingerprint(_dual())
    for item in (*found.ideals, found, report, value):
        assert "0x" not in repr(item)
    assert value != object()
    assert found.bounds is not AnalysisBounds()

    dual = _dual()
    unary = FiniteMultilinearStructure.from_tables(
        dual.module,
        BasisProductTable.from_cells(
            dual.module,
            1,
            [{0: QQ().element(1)}, {1: QQ().element(1)}],
        ),
    )
    with pytest.raises(FingerprintAnalysisError, match="arity"):
        algebra_fingerprint(unary)

    class BadIdeal(IdealData):
        pass

    with pytest.raises(FingerprintAnalysisError, match="factory"):
        BadIdeal._create((), (), 0)
    monkeypatch.setattr(
        fingerprint,
        "_derivation_rows",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("injected")),
    )
    with pytest.raises(FingerprintAnalysisError, match="derivation kernel"):
        derivation_algebra(_dual())


def test_certified_basis_change_transport_rebuilds_equal_fingerprint() -> None:
    source = _dual()
    target = FreeModule(QQ(), Basis(("f0", "f1"), coefficient_domain=QQ()))
    change = ExactBasisIsomorphism.from_images(
        source.module,
        target,
        (target.element({0: 1}), target.element({0: 1, 1: 2})),
        (source.module.element({0: 1}), source.module.element({0: (-1, 2), 1: (1, 2)})),
    )
    constants = change_basis(source.operation.constants, change)
    transported = FiniteMultilinearStructure.from_structure_constants(
        target, cast(StructureConstants, constants)
    )
    assert algebra_fingerprint(source) == algebra_fingerprint(transported)


def test_negative_laws_private_edges_and_translation_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    noncommutative_nonassociative = _algebra([{1: 1}, {}, {1: 1}, {}])
    value = algebra_fingerprint(noncommutative_nonassociative)
    assert value.commutative is False
    assert value.associative is False
    assert value.commutativity_report.status == "Disproved"
    assert value.commutativity_report.witness_indices == (0, 1)
    assert value.commutativity_report.witness_value is not None
    assert value.associativity_report.status == "Disproved"
    assert value.associativity_report.witness_indices == (0, 0, 0)
    assert fingerprint._rref(((Rational(1),),), 1)[1] == (0,)

    class BadSearch(IdealSearchResult):
        pass

    class BadDerivation(DerivationReport):
        pass

    class BadFingerprint(AlgebraFingerprint):
        pass

    bounds = AnalysisBounds()
    with pytest.raises(FingerprintAnalysisError, match="factory"):
        BadSearch._create((), bounds, 0)
    with pytest.raises(FingerprintAnalysisError, match="factory"):
        BadDerivation._create(
            (), derivation_algebra(_zero(0)).matrix_space, 0, 0, bounds
        )
    with pytest.raises(FingerprintAnalysisError, match="factory"):
        BadFingerprint._create(
            algebra=_dual(),
            bounds=bounds,
            unit=algebra_fingerprint(_dual()).unit_report,
            commutative=algebra_fingerprint(_dual()).commutativity_report,
            associative=algebra_fingerprint(_dual()).associativity_report,
            annihilators=(0, 0, 0),
            ideal_data=ideals(_zero(0)),
            derivations=derivation_algebra(_zero(0)),
        )

    monkeypatch.setattr(
        fingerprint,
        "_product",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("injected")),
    )
    with pytest.raises(FingerprintAnalysisError, match="canonical ideal"):
        ideals(_dual())
    with pytest.raises(FingerprintAnalysisError, match="fingerprint evaluation"):
        algebra_fingerprint(_dual())


def test_partial_many_sorted_and_malformed_table_routes_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, flag = Sort("value"), Sort("flag")
    values, flags = (
        FiniteCarrier((object(),), sort=value),
        FiniteCarrier(("yes",), sort=flag),
    )
    symbol = OperationSymbol("p", (value, value), value)
    partial = PartialOperation.from_table(
        symbol, ((value, values), (flag, flags)), (), undefined_marker=object()
    )
    structure = (
        StructureBuilder(Signature((value, flag), (symbol,)))
        .with_carrier(value, values)
        .with_carrier(flag, flags)
        .with_operation(symbol, partial)
        .freeze()
    )
    with pytest.raises(FingerprintAnalysisError, match="exact FiniteMultilinear"):
        algebra_fingerprint(cast(FiniteMultilinearStructure, structure))

    algebra = _dual()
    for entries, reason in (
        (object(), "entries are malformed"),
        (((True, QQ().element(1)),), "exact in-range"),
        (((2, QQ().element(1)),), "exact in-range"),
        (((0, QQ().element(1)), (0, QQ().element(1))), "duplicated"),
        (((0, object()),), "payload is malformed"),
    ):
        monkeypatch.setattr(
            type(algebra.operation),
            "coefficients_for_basis",
            lambda *_args, e=entries: e,
        )
        with pytest.raises(FingerprintAnalysisError, match=reason):
            derivation_algebra(algebra)


def test_unit_contradiction_and_subspace_normalized_ideal_dedup() -> None:
    zero = algebra_fingerprint(_zero(2))
    assert zero.unit_status == "absent"
    assert zero.unit_report.witness_row is not None
    # Every basis product is -e0+e1.  A^2 and the annihilator are the same
    # line, initially found with opposite normalizations.
    algebra = _algebra([{0: -1, 1: 1} for _ in range(4)])
    found = ideals(algebra)
    assert len(found.ideals) == 3
    assert found.ideals[0].origins == ("zero",)
    assert found.ideals[1].origins == (
        "product_span_two_sided",
        "two_sided_annihilator",
    )
    assert found.ideals[2].origins == ("whole",)


@pytest.mark.parametrize(
    ("field", "payload"),
    [("denominator", 0), ("numerator", True), ("denominator", "x")],
)
def test_tampered_rational_payloads_are_typed(field: str, payload: object) -> None:
    algebra = _dual()
    coefficient = algebra.operation.constants.entries[0][1]
    object.__setattr__(coefficient.value, field, payload)
    with pytest.raises(FingerprintAnalysisError, match="rational payload"):
        algebra_fingerprint(algebra)


def test_evidence_factories_reject_fabricated_records() -> None:
    for value in (
        algebra_fingerprint(_dual()).commutativity_report,
        algebra_fingerprint(_dual()).unit_report,
    ):
        assert "0x" not in repr(value)
        with pytest.raises(TypeError):
            hash(value)
    with pytest.raises(FingerprintAnalysisError):
        fingerprint.LawReport()
    with pytest.raises(FingerprintAnalysisError):
        fingerprint.UnitReport()
    for law_args in (
        ("bad", 1, 1, None, None),
        ("commutativity", 1, 0, None, None),
        ("commutativity", 1, 1, (0, 0), (Rational(0),)),
        ("commutativity", 4, 2, (0, 0), (Rational(1), Rational(0))),
    ):
        with pytest.raises(FingerprintAnalysisError):
            fingerprint.LawReport._create(*law_args)
    for unit_args in (
        ("bogus", 0, (), None),
        ("present", 2, None, None),
        ("absent", 0, None, (Rational(1),)),
        ("absent", 2, (), (Rational(0), Rational(0))),
    ):
        with pytest.raises(FingerprintAnalysisError):
            fingerprint.UnitReport._create(*unit_args)


def test_full_table_metadata_and_duck_module_rejections_are_typed() -> None:
    cases: tuple[tuple[object, str], ...] = (
        ([], "constants"),
        (((0, 0, 0, QQ().element(1)),), "entries"),
        ((((0, 0), QQ().element(1)),), "key"),
        ((((2, 0, 0), QQ().element(1)),), "range"),
        (
            (((0, 0, 0), QQ().element(1)), ((0, 0, 0), QQ().element(1))),
            "strictly sorted",
        ),
        ((((0, 0, 0), QQ().element(0)),), "payload"),
    )
    for entries, reason in cases:
        fresh = _dual()
        object.__setattr__(fresh.operation.constants, "entries", entries)
        with pytest.raises(FingerprintAnalysisError, match=reason):
            algebra_fingerprint(fresh)

    class DuckModule:
        domain = QQ()
        rank = 1

    broken = _dual()
    object.__setattr__(broken, "module", DuckModule())
    with pytest.raises(FingerprintAnalysisError, match="module"):
        algebra_fingerprint(broken)


def test_remaining_internal_error_translations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        elementary,
        "_algebra",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("hidden")),
    )
    with pytest.raises(FingerprintAnalysisError, match="metadata"):
        fingerprint._check_algebra(_dual(), AnalysisBounds())
    monkeypatch.undo()
    labels = _dual()
    object.__setattr__(labels.module.basis, "labels", ("ok", object()))
    with pytest.raises(FingerprintAnalysisError, match="labels"):
        ideals(labels)
    operation = _dual()
    object.__setattr__(operation, "operation", object())
    with pytest.raises(FingerprintAnalysisError, match="metadata"):
        ideals(operation)
    with pytest.raises(FingerprintAnalysisError):
        fingerprint.LawReport._create("associativity", 2, 2, None, None)

    class BadLaw(fingerprint.LawReport):
        pass

    class BadUnit(fingerprint.UnitReport):
        pass

    with pytest.raises(FingerprintAnalysisError, match="factory"):
        BadLaw._create("commutativity", 1, 1, None, None)
    with pytest.raises(FingerprintAnalysisError, match="factory"):
        BadUnit._create("present", 0, (), None)
    with pytest.raises(FingerprintAnalysisError):
        fingerprint.UnitReport._create("present", True, (), None)
    monkeypatch.setattr(
        fingerprint,
        "_product",
        lambda *_args: (_ for _ in ()).throw(
            FingerprintAnalysisError(field="x", reason="injected")
        ),
    )
    with pytest.raises(FingerprintAnalysisError, match="injected"):
        ideals(_dual())


def test_defensive_internal_branches_remain_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed_operation = _dual()
    object.__setattr__(malformed_operation, "operation", object())
    with pytest.raises(FingerprintAnalysisError, match="operation is malformed"):
        fingerprint._check_constants(malformed_operation)

    source = _dual()
    monkeypatch.setattr(
        StructureConstants,
        "from_sparse",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected")),
    )
    with pytest.raises(FingerprintAnalysisError, match="could not be snapshotted"):
        fingerprint._check_algebra(source, AnalysisBounds())
    monkeypatch.undo()

    monkeypatch.setattr(fingerprint, "_rref", lambda *_args: ((), ()))
    with pytest.raises(FingerprintAnalysisError, match="unexpectedly nonunique"):
        fingerprint._unit_report(_dual())
    monkeypatch.undo()

    injected = FingerprintAnalysisError(field="law", reason="injected")
    monkeypatch.setattr(
        fingerprint,
        "_law_data",
        lambda *_args: (_ for _ in ()).throw(injected),
    )
    with pytest.raises(FingerprintAnalysisError, match="injected"):
        algebra_fingerprint(_dual())
    monkeypatch.undo()

    elementary_error = elementary.ElementaryAnalysisError(
        field="center", reason="injected"
    )
    monkeypatch.setattr(
        elementary,
        "left_nucleus",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(elementary_error),
    )
    with pytest.raises(FingerprintAnalysisError, match="injected"):
        algebra_fingerprint(_dual())
