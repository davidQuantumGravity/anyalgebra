from __future__ import annotations
from dataclasses import FrozenInstanceError
from typing import Any, cast
import pytest
from anyalgebra.core.domains import QQ
from anyalgebra.linear.matrix import Matrix, MatrixSpace
from anyalgebra.linear.solve import ExactSolveError
from anyalgebra.linear.recovery import (
    ReconstructionCheck,
    RecoveryError,
    RecoveryFailure,
    RecoverySuccess,
    recover_structure_constants,
)
from anyalgebra.linear.span import Closed, DeclaredBilinearOperation, PairCoordinates
import anyalgebra.linear.recovery as recovery_module
from anyalgebra.linear.span import closure


def assert_failure_provenance(
    report: RecoveryFailure, declaration: DeclaredBilinearOperation
) -> None:
    assert report.declaration is declaration
    if report.closure_evidence is not None:
        assert report.closure_evidence.declaration is declaration


def test_abelian_and_nonabelian_recovery() -> None:
    space = MatrixSpace(1, 2, QQ())
    e0 = space.element(((1, 0),))
    e1 = space.element(((0, 1),))
    calls = []

    def bracket(left: Matrix, right: Matrix) -> Matrix:
        calls.append((left, right))
        if left is e0 and right is e1:
            return e1
        if left is e1 and right is e0:
            return space.element(((0, -1),))
        return space.zero()

    declaration = DeclaredBilinearOperation.create(space, bracket, bilinear=True)
    report = recover_structure_constants((e0, e1), declaration)
    assert type(report) is RecoverySuccess and report.rank == 2
    assert (
        report.declaration is declaration and report.closure.declaration is declaration
    )
    assert report.closure.declaration.operation is bracket
    assert report.constants.entries == (
        ((0, 1, 1), QQ().element(1)),
        ((1, 0, 1), QQ().element(-1)),
    )
    assert len(calls) == 4


def test_dependency_and_nonclosure_do_not_return_constants() -> None:
    space = MatrixSpace(1, 2, QQ())
    e0 = space.element(((1, 0),))
    e1 = space.element(((0, 1),))
    dependent_declaration = DeclaredBilinearOperation.create(
        space, lambda a, b: space.zero(), bilinear=True
    )
    dep = recover_structure_constants((e0, e0), dependent_declaration)
    assert type(dep) is RecoveryFailure and dep.dependencies
    assert_failure_provenance(dep, dependent_declaration)
    nonclosed_declaration = DeclaredBilinearOperation.create(
        space, lambda a, b: e1, bilinear=True
    )
    bad = recover_structure_constants((e0,), nonclosed_declaration)
    assert type(bad) is RecoveryFailure and bad.closure_evidence is not None
    assert_failure_provenance(bad, nonclosed_declaration)


def test_malformed_and_sealed() -> None:
    space = MatrixSpace(1, 1, QQ())
    one = space.element(((1,),))
    report = recover_structure_constants((one,), object())
    assert type(report) is RecoveryFailure
    with pytest.raises(RecoveryError):
        RecoverySuccess()


def test_empty_abelian_reconstruction_and_false_declarations() -> None:
    space = MatrixSpace(1, 1, QQ())
    calls = 0

    def bracket(left: Matrix, right: Matrix) -> Matrix:
        nonlocal calls
        calls += 1
        return space.zero()

    empty = recover_structure_constants(
        (), DeclaredBilinearOperation.create(space, bracket, bilinear=True)
    )
    assert (
        type(empty) is RecoverySuccess
        and empty.rank == 0
        and empty.constants.entries == ()
        and calls == 0
    )
    one = space.element(((1,),))
    abelian = recover_structure_constants(
        (one,), DeclaredBilinearOperation.create(space, bracket, bilinear=True)
    )
    assert type(abelian) is RecoverySuccess and len(abelian.reconstruction_checks) == 1
    check = abelian.reconstruction_checks[0]
    assert (
        type(check) is ReconstructionCheck
        and check.exact
        and check.captured_product == check.reconstructed_product
    )
    calls = 0
    false_declaration = DeclaredBilinearOperation.create(space, bracket, bilinear=False)
    false = recover_structure_constants((one,), false_declaration)
    assert type(false) is RecoveryFailure and calls == 0
    assert_failure_provenance(false, false_declaration)


@pytest.mark.parametrize("operators", ["bad", {0: object()}, object(), (object(),)])
def test_bad_operator_inputs_are_rejected(operators: object) -> None:
    space = MatrixSpace(1, 1, QQ())
    with pytest.raises(RecoveryError):
        recover_structure_constants(
            operators,
            DeclaredBilinearOperation.create(
                space, lambda a, b: space.zero(), bilinear=True
            ),
        )


def test_bracket_failures_and_declaration_parent_do_not_claim_success() -> None:
    space = MatrixSpace(1, 1, QQ())
    foreign = MatrixSpace(1, 1, QQ())
    one = space.element(((1,),))
    calls = 0

    def counted(left: Matrix, right: Matrix) -> Matrix:
        nonlocal calls
        calls += 1
        return space.zero()

    mismatch_declaration = DeclaredBilinearOperation.create(
        foreign, counted, bilinear=True
    )
    mismatch = recover_structure_constants((one,), mismatch_declaration)
    assert type(mismatch) is RecoveryFailure and calls == 0
    assert_failure_provenance(mismatch, mismatch_declaration)
    for bracket in (
        lambda a, b: object(),
        lambda a, b: foreign.element(((0,),)),
        lambda a, b: (_ for _ in ()).throw(RuntimeError("secret")),
    ):
        declaration = DeclaredBilinearOperation.create(space, bracket, bilinear=True)
        result = recover_structure_constants((one,), declaration)
        assert type(result) is RecoveryFailure
        assert_failure_provenance(result, declaration)


def test_records_are_safe_and_factories_sealed() -> None:
    space = MatrixSpace(1, 1, QQ())
    one = space.element(((1,),))
    result = recover_structure_constants(
        (one,),
        DeclaredBilinearOperation.create(
            space, lambda a, b: space.zero(), bilinear=True
        ),
    )
    assert type(result) is RecoverySuccess
    dependent = recover_structure_constants(
        (one, one),
        DeclaredBilinearOperation.create(
            space, lambda a, b: space.zero(), bilinear=True
        ),
    )
    assert type(dependent) is RecoveryFailure
    assert result.dependencies == () and dependent.dependencies
    relation = dependent.dependencies[0]
    assert cast(Any, relation.entry(0, 0)).add(relation.entry(1, 0)) == QQ().element(0)
    for record in (result, dependent, result.reconstruction_checks[0]):
        assert not hasattr(record, "__dict__") and "0x" not in repr(record)
        with pytest.raises(TypeError):
            hash(record)
        assert record == record
    with pytest.raises(FrozenInstanceError):
        result.rank = 3  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        dependent.reason = "bad"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.reconstruction_checks[0].exact = False  # type: ignore[misc]
    with pytest.raises(RecoveryError):
        ReconstructionCheck()


def test_hostile_and_bounded_operator_iterables() -> None:
    space = MatrixSpace(1, 1, QQ())
    one = space.element(((1,),))
    declaration = DeclaredBilinearOperation.create(
        space, lambda a, b: space.zero(), bilinear=True
    )

    class BadIter:
        def __iter__(self) -> object:
            raise RuntimeError("secret iter")

    class BadNext:
        def __iter__(self) -> BadNext:
            return self

        def __next__(self) -> object:
            raise RuntimeError("secret next")

    for bad in (BadIter(), BadNext()):
        with pytest.raises(RecoveryError) as caught:
            recover_structure_constants(bad, declaration)
        assert "secret" not in str(caught.value)
    exact = recover_structure_constants(tuple(one for _ in range(128)), declaration)
    assert type(exact) is RecoveryFailure
    with pytest.raises(RecoveryError) as overflow:
        recover_structure_constants((one for _ in range(513)), declaration)
    assert (overflow.value.kind, overflow.value.observed, overflow.value.maximum) == (
        "operators",
        513,
        512,
    )


def test_all_recovery_record_factories_are_sealed() -> None:
    class CheckSubclass(ReconstructionCheck):
        pass

    class SuccessSubclass(RecoverySuccess):
        pass

    class FailureSubclass(RecoveryFailure):
        pass

    for action in (
        lambda: RecoveryFailure(),
        lambda: CheckSubclass._create(
            0, 0, cast(Any, object()), cast(Any, object()), True
        ),
        lambda: SuccessSubclass._create(
            (), (), cast(Any, object()), cast(Any, object()), cast(Any, object()), ()
        ),
        lambda: FailureSubclass._create("x", ()),
    ):
        with pytest.raises(RecoveryError):
            action()


def test_monkeypatched_missing_capture_has_pair_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space = MatrixSpace(1, 1, QQ())
    one = space.element(((1,),))
    declaration = DeclaredBilinearOperation.create(
        space, lambda a, b: space.zero(), bilinear=True
    )
    evidence = closure((one,), declaration)
    monkeypatch.setattr(recovery_module, "closure", lambda basis, operation: evidence)
    result = recovery_module.recover_structure_constants((one,), declaration)
    assert type(result) is RecoveryFailure and result.kind == "invariant"
    assert (result.left_index, result.right_index) == (0, 0)
    assert_failure_provenance(result, declaration)


def test_structured_solve_bounds_are_recovery_failures() -> None:
    space = MatrixSpace(1, 1, QQ())
    one = space.element(((1,),))
    calls = 0

    def bracket(left: Matrix, right: Matrix) -> Matrix:
        nonlocal calls
        calls += 1
        return space.zero()

    declaration = DeclaredBilinearOperation.create(space, bracket, bilinear=True)
    result = recover_structure_constants(tuple(one for _ in range(512)), declaration)
    assert type(result) is RecoveryFailure
    assert (result.kind, result.observed, result.maximum, calls) == (
        "work",
        262144,
        65536,
        0,
    )
    assert_failure_provenance(result, declaration)


def test_inherited_span_bounds_are_no_touch_and_structured() -> None:
    pair_space = MatrixSpace(1, 9, QQ())
    pair_basis = tuple(
        pair_space.element((tuple(1 if i == chosen else 0 for i in range(9)),))
        for chosen in range(9)
    )
    calls = 0

    def bracket(left: Matrix, right: Matrix) -> Matrix:
        nonlocal calls
        calls += 1
        return left

    pair_declaration = DeclaredBilinearOperation.create(
        pair_space, bracket, bilinear=True
    )
    pair = recover_structure_constants(pair_basis, pair_declaration)
    assert type(pair) is RecoveryFailure
    assert (pair.kind, pair.observed, pair.maximum, calls) == ("pairs", 81, 64, 0)
    assert_failure_provenance(pair, pair_declaration)

    work_space = MatrixSpace(1, 13, QQ())
    work_basis = tuple(
        work_space.element((tuple(1 if i == chosen else 0 for i in range(13)),))
        for chosen in range(8)
    )
    work_declaration = DeclaredBilinearOperation.create(
        work_space, bracket, bilinear=True
    )
    work = recover_structure_constants(work_basis, work_declaration)
    assert type(work) is RecoveryFailure
    assert (work.kind, work.observed, work.maximum, calls) == ("work", 97344, 65536, 0)
    assert_failure_provenance(work, work_declaration)

    flat_space = MatrixSpace(3, 171, QQ())
    flat_declaration = DeclaredBilinearOperation.create(
        flat_space, bracket, bilinear=True
    )
    flat = recover_structure_constants((flat_space.zero(),), flat_declaration)
    assert type(flat) is RecoveryFailure
    assert (flat.kind, flat.observed, flat.maximum, calls) == ("dimension", 513, 512, 0)
    assert_failure_provenance(flat, flat_declaration)


def test_recovery_defensive_evidence_seams(monkeypatch: pytest.MonkeyPatch) -> None:
    space = MatrixSpace(1, 1, QQ())
    one = space.element(((1,),))
    declaration = DeclaredBilinearOperation.create(
        space, lambda a, b: space.zero(), bilinear=True
    )
    original_decompose = cast(Any, recovery_module).span_decompose

    def solve_error(target: object, generators: object) -> object:
        del target, generators
        raise ExactSolveError("forced", kind="work", observed=17, maximum=16)

    monkeypatch.setattr(recovery_module, "span_decompose", solve_error)
    dependency = recovery_module.recover_structure_constants((one,), declaration)
    assert type(dependency) is RecoveryFailure
    assert (dependency.kind, dependency.observed, dependency.maximum) == (
        "work",
        17,
        16,
    )
    assert_failure_provenance(dependency, declaration)

    monkeypatch.setattr(recovery_module, "span_decompose", original_decompose)
    monkeypatch.setattr(recovery_module, "closure", lambda basis, operation: object())
    malformed = recovery_module.recover_structure_constants((one,), declaration)
    assert type(malformed) is RecoveryFailure
    assert_failure_provenance(malformed, declaration)

    valid = closure((one,), declaration)
    monkeypatch.setattr(recovery_module, "closure", lambda basis, operation: valid)
    monkeypatch.setattr(
        recovery_module, "_public_evidence", lambda evidence, declared: object()
    )
    unknown = recovery_module.recover_structure_constants((one,), declaration)
    assert type(unknown) is RecoveryFailure
    assert_failure_provenance(unknown, declaration)


def test_reconstruction_mismatch_is_single_capture_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space = MatrixSpace(1, 1, QQ())
    one = space.element(((1,),))
    calls = 0

    def bracket(left: Matrix, right: Matrix) -> Matrix:
        nonlocal calls
        calls += 1
        return one

    def fake_closed(
        basis: tuple[Matrix, ...], operation: DeclaredBilinearOperation
    ) -> Closed:
        operation.operation(basis[0], basis[0])
        coordinates = MatrixSpace(1, 1, QQ()).zero()
        pair = PairCoordinates._create(0, 0, coordinates)
        return Closed._create(basis, operation, coordinates.parent, (pair,), 1)

    monkeypatch.setattr(recovery_module, "closure", fake_closed)
    declaration = DeclaredBilinearOperation.create(space, bracket, bilinear=True)
    result = recovery_module.recover_structure_constants((one,), declaration)
    assert type(result) is RecoveryFailure
    assert (result.kind, result.left_index, result.right_index, calls) == (
        "reconstruction",
        0,
        0,
        1,
    )
    assert_failure_provenance(result, declaration)


def test_independent_two_dimensional_structure_table_oracle() -> None:
    tables: tuple[dict[tuple[int, int], tuple[int, object]], ...] = (
        {},
        {(0, 1): (0, 1), (1, 0): (0, -1)},
        {(0, 0): (1, (1, 2)), (0, 1): (0, 2), (1, 1): (1, -1)},
    )
    for table in tables:
        space = MatrixSpace(1, 2, QQ())
        basis = (space.element(((1, 0),)), space.element(((0, 1),)))
        calls = 0

        def bracket(
            left: Matrix,
            right: Matrix,
            basis: tuple[Matrix, Matrix] = basis,
            table: dict[tuple[int, int], tuple[int, object]] = table,
            space: MatrixSpace = space,
        ) -> Matrix:
            nonlocal calls
            calls += 1
            left_index = 0 if left is basis[0] else 1
            right_index = 0 if right is basis[0] else 1
            output, coefficient = table.get((left_index, right_index), (0, 0))
            values: list[object] = [0, 0]
            values[output] = coefficient
            return space.element((tuple(values),))

        report = recover_structure_constants(
            basis, DeclaredBilinearOperation.create(space, bracket, bilinear=True)
        )
        assert type(report) is RecoverySuccess
        expected = tuple(
            ((left, right, output), QQ().element(coefficient))
            for (left, right), (output, coefficient) in sorted(table.items())
            if QQ().element(coefficient).value.numerator != 0
        )
        assert report.constants.entries == expected
        assert report.rank == 2 and report.constants.output_basis.labels == ("O0", "O1")
        assert len(report.reconstruction_checks) == 4 and calls == 4
        assert all(
            check.exact and check.captured_product == check.reconstructed_product
            for check in report.reconstruction_checks
        )
        assert all(
            pair.coordinates.parent is report.closure.coordinate_space
            for pair in report.closure.decompositions
        )


def test_stateful_bracket_is_captured_once() -> None:
    space = MatrixSpace(1, 1, QQ())
    one = space.element(((1,),))
    calls = 0

    def bracket(left: Matrix, right: Matrix) -> Matrix:
        nonlocal calls
        calls += 1
        return space.element(((calls,),))

    report = recover_structure_constants(
        (one,), DeclaredBilinearOperation.create(space, bracket, bilinear=True)
    )
    assert type(report) is RecoverySuccess and calls == 1
    check = report.reconstruction_checks[0]
    assert check.exact and check.captured_product == space.element(((1,),))
