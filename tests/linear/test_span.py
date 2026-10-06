from __future__ import annotations

from dataclasses import FrozenInstanceError
from itertools import repeat
from typing import cast

import pytest

from anyalgebra.core.domains import QQ, ZZ, _RationalElement
from anyalgebra.linear.matrix import Matrix, MatrixSpace
import anyalgebra.linear.span as span_module
from anyalgebra.linear.span import (
    Closed,
    ClosureFailure,
    DeclaredBilinearOperation,
    DependentSpan,
    NonClosed,
    OutsideSpan,
    PairCoordinates,
    SpanError,
    UniqueSpan,
    closure,
    span_basis,
    span_decompose,
)


def _matrix(space: MatrixSpace, rows: tuple[tuple[object, ...], ...]) -> Matrix:
    return space.element(rows)


def _entry(matrix: Matrix, row: int, column: int) -> _RationalElement:
    return cast(_RationalElement, matrix.entry(row, column))


def _column_entries(matrix: Matrix) -> tuple[_RationalElement, ...]:
    return tuple(_entry(matrix, index, 0) for index in range(matrix.parent.rows))


def _linear_combination(generators: tuple[Matrix, ...], coordinates: Matrix) -> Matrix:
    result = generators[0].parent.zero()
    for generator, coordinate in zip(
        generators, _column_entries(coordinates), strict=True
    ):
        scaled = generator.parent.element(
            tuple(
                tuple(
                    cast(_RationalElement, entry).multiply(coordinate) for entry in row
                )
                for row in generator.entries
            )
        )
        result = result.add(scaled)
    return result


def _pair(left: Matrix, right: Matrix) -> _RationalElement:
    total = QQ().element(0)
    for left_row, right_row in zip(left.entries, right.entries, strict=True):
        for left_entry, right_entry in zip(left_row, right_row, strict=True):
            total = total.add(
                cast(_RationalElement, left_entry).multiply(
                    cast(_RationalElement, right_entry)
                )
            )
    return total


def test_unique_dependent_outside_and_flattened_operator_coordinates() -> None:
    space = MatrixSpace(2, 2, QQ())
    first = _matrix(space, ((1, 0), (0, 0)))
    second = _matrix(space, ((0, 0), (0, 1)))
    target = _matrix(space, ((2, 0), (0, 3)))
    unique = span_decompose(target, (first, second))
    assert type(unique) is UniqueSpan
    assert _column_entries(unique.coordinates) == (QQ().element(2), QQ().element(3))
    assert _linear_combination((first, second), unique.coordinates) == target

    dependent = span_decompose(target, (first, first, second))
    assert type(dependent) is DependentSpan
    assert len(dependent.dependency_basis) == 1
    assert (
        dependent.particular_coordinates.parent is dependent.dependency_basis[0].parent
    )
    assert (
        _linear_combination((first, first, second), dependent.particular_coordinates)
        == target
    )
    assert (
        _linear_combination((first, first, second), dependent.dependency_basis[0])
        == space.zero()
    )

    outside_target = _matrix(space, ((0, 1), (0, 0)))
    outside = span_decompose(outside_target, (first, second))
    assert type(outside) is OutsideSpan
    assert outside.target_pairing.value.numerator != 0
    assert outside.target_pairing == _pair(outside.functional, outside_target)
    assert all(
        _pair(outside.functional, generator).value.numerator == 0
        for generator in (first, second)
    )


def test_span_basis_declaration_order_duplicates_zero_and_empty_cases() -> None:
    space = MatrixSpace(1, 2, QQ())
    first, duplicate, second, zero = (
        _matrix(space, ((1, 0),)),
        _matrix(space, ((1, 0),)),
        _matrix(space, ((0, 1),)),
        _matrix(space, ((0, 0),)),
    )
    assert span_basis((first, duplicate, second, zero)) == (first, second)
    zero_target = space.zero()
    empty_unique = span_decompose(zero_target, ())
    assert (
        type(empty_unique) is UniqueSpan and empty_unique.coordinates.parent.rows == 0
    )
    nonzero = span_decompose(first, ())
    assert type(nonzero) is OutsideSpan and nonzero.target_pairing.value.numerator != 0
    assert span_basis(()) == ()


def test_zero_shapes_rectangles_and_parent_rules() -> None:
    zero_shape = MatrixSpace(0, 3, QQ())
    generator = zero_shape.zero()
    result = span_decompose(generator, (generator,))
    assert type(result) is DependentSpan
    assert len(result.dependency_basis) == 1
    assert span_basis((generator,)) == ()
    rectangular = MatrixSpace(2, 3, QQ())
    first = _matrix(rectangular, ((1, 0, 0), (0, 0, 0)))
    assert type(span_decompose(first, (first,))) is UniqueSpan
    foreign = MatrixSpace(2, 3, QQ()).element(((1, 0, 0), (0, 0, 0)))
    with pytest.raises(SpanError, match="literal same"):
        span_decompose(first, (foreign,))
    with pytest.raises(SpanError, match="literal QQ"):
        span_decompose(MatrixSpace(1, 1, ZZ()).element(((1,),)), ())


def test_closure_order_noncommutativity_failure_and_no_touch_preflight() -> None:
    space = MatrixSpace(1, 1, QQ())
    one = _matrix(space, ((1,),))
    calls: list[tuple[int, int]] = []

    def multiply(left: Matrix, right: Matrix) -> Matrix:
        calls.append(
            (_entry(left, 0, 0).value.numerator, _entry(right, 0, 0).value.numerator)
        )
        return space.element(((_entry(left, 0, 0).multiply(_entry(right, 0, 0)),),))

    declared = DeclaredBilinearOperation.create(space, multiply, bilinear=True)
    closed = closure((one,), declared)
    assert type(closed) is Closed
    assert calls == [(1, 1)]
    assert closed.decompositions[0].coordinates.parent is closed.coordinate_space
    assert closed.declaration is declared and closed.pair_count == 1

    nonclosed_space = MatrixSpace(1, 2, QQ())
    basis_element = _matrix(nonclosed_space, ((1, 0),))
    outside_element = _matrix(nonclosed_space, ((0, 1),))
    nonclosed = closure(
        (basis_element,),
        DeclaredBilinearOperation.create(
            nonclosed_space, lambda left, right: outside_element, bilinear=True
        ),
    )
    assert type(nonclosed) is NonClosed
    assert (nonclosed.left_index, nonclosed.right_index) == (0, 0)
    assert nonclosed.pair_count == 1

    order_calls: list[tuple[int, int]] = []

    def ordered(left: Matrix, right: Matrix) -> Matrix:
        order_calls.append(
            (_entry(left, 0, 0).value.numerator, _entry(right, 0, 0).value.numerator)
        )
        return left

    ordered_closed = closure(
        (basis_element, outside_element),
        DeclaredBilinearOperation.create(nonclosed_space, ordered, bilinear=True),
    )
    assert type(ordered_closed) is Closed
    assert order_calls == [(1, 1), (1, 0), (0, 1), (0, 0)]

    bad = DeclaredBilinearOperation.create(space, multiply, bilinear=False)
    before = len(calls)
    failed = closure((one,), bad)
    assert type(failed) is ClosureFailure and len(calls) == before


def test_closure_operation_failures_hostile_iterators_and_bounds() -> None:
    space = MatrixSpace(1, 1, QQ())
    one = _matrix(space, ((1,),))
    for operation in (
        lambda left, right: (_ for _ in ()).throw(RuntimeError("secret")),
        lambda left, right: object(),
        lambda left, right: MatrixSpace(1, 1, QQ()).element(((1,),)),
    ):
        report = closure(
            (one,), DeclaredBilinearOperation.create(space, operation, bilinear=True)
        )
        assert type(report) is ClosureFailure
        assert "secret" not in report.reason
        assert (report.left_index, report.right_index, report.pair_count) == (0, 0, 1)

    with pytest.raises(SpanError) as infinite:
        span_basis(repeat(one))
    assert (infinite.value.kind, infinite.value.observed, infinite.value.maximum) == (
        "generators",
        513,
        512,
    )
    with pytest.raises(SpanError, match="iterable"):
        span_basis({0: one})


def test_records_are_sealed_frozen_unhashable_safe_and_deterministic() -> None:
    space = MatrixSpace(1, 1, QQ())
    one = _matrix(space, ((1,),))
    unique = span_decompose(one, (one,))
    dependent = span_decompose(one, (one, one))
    outside = span_decompose(_matrix(space, ((2,),)), ())
    declared = DeclaredBilinearOperation.create(
        space, lambda left, right: one, bilinear=True
    )
    reports = (unique, dependent, outside, declared, closure((one,), declared))
    for record in reports:
        assert not hasattr(record, "__dict__")
        with pytest.raises(TypeError):
            hash(record)
        assert "0x" not in repr(record)
    with pytest.raises(FrozenInstanceError):
        unique.status = "bad"  # type: ignore[misc]

    class UniqueSubclass(UniqueSpan):
        pass

    with pytest.raises(SpanError):
        UniqueSubclass._create(cast(UniqueSpan, unique).coordinates)
    with pytest.raises(SpanError):
        UniqueSpan()
    first = cast(UniqueSpan, span_decompose(one, (one,))).coordinates
    second = cast(UniqueSpan, span_decompose(one, (one,))).coordinates
    assert _column_entries(first) == _column_entries(second)


def test_all_factories_preflight_and_sanitized_failure_paths() -> None:
    space = MatrixSpace(1, 1, QQ())
    one = _matrix(space, ((1,),))
    outside = cast(OutsideSpan, span_decompose(_matrix(space, ((2,),)), ()))
    coordinate = cast(UniqueSpan, span_decompose(one, (one,))).coordinates
    pair = PairCoordinates._create(0, 0, coordinate)

    for record_type in (
        UniqueSpan,
        DependentSpan,
        OutsideSpan,
        DeclaredBilinearOperation,
        PairCoordinates,
        Closed,
        NonClosed,
        ClosureFailure,
    ):
        with pytest.raises(SpanError):
            record_type()

    class DependentSubclass(DependentSpan):
        pass

    class OutsideSubclass(OutsideSpan):
        pass

    class PairSubclass(PairCoordinates):
        pass

    class ClosedSubclass(Closed):
        pass

    class NonClosedSubclass(NonClosed):
        pass

    class FailureSubclass(ClosureFailure):
        pass

    class DeclarationSubclass(DeclaredBilinearOperation):
        pass

    with pytest.raises(SpanError):
        DependentSubclass._create(coordinate, ())
    with pytest.raises(SpanError):
        OutsideSubclass._create(one, QQ().element(1))
    with pytest.raises(SpanError):
        PairSubclass._create(0, 0, coordinate)
    with pytest.raises(SpanError):
        ClosedSubclass._create(
            (one,),
            cast(DeclaredBilinearOperation, object()),
            coordinate.parent,
            (pair,),
            1,
        )
    with pytest.raises(SpanError):
        NonClosedSubclass._create(
            0,
            0,
            (one,),
            cast(DeclaredBilinearOperation, object()),
            1,
            one,
            outside,
        )
    with pytest.raises(SpanError):
        FailureSubclass._create("x")
    with pytest.raises(SpanError):
        DeclaredBilinearOperation.create(
            MatrixSpace(1, 1, ZZ()), lambda a, b: one, bilinear=True
        )
    with pytest.raises(SpanError):
        DeclaredBilinearOperation.create(space, object(), bilinear=True)
    with pytest.raises(SpanError):
        DeclarationSubclass.create(space, lambda a, b: one, bilinear=True)

    class BadIter:
        def __iter__(self) -> object:
            raise RuntimeError("secret iterator")

    class BadNext:
        def __iter__(self) -> BadNext:
            return self

        def __next__(self) -> object:
            raise RuntimeError("secret next")

    for bad in (BadIter(), BadNext()):
        with pytest.raises(SpanError) as caught:
            span_basis(bad)
        assert "secret" not in str(caught.value)
        assert caught.value.__cause__ is None and caught.value.__context__ is None
    with pytest.raises(SpanError, match="exact Matrix"):
        span_decompose(object(), ())
    with pytest.raises(SpanError, match="exact Matrix"):
        span_decompose(one, (object(),))
    with pytest.raises(SpanError, match="exact Matrix"):
        span_basis((object(),))
    malformed = Matrix._create(space, ((object(),),))
    with pytest.raises(SpanError, match="non-QQ"):
        span_decompose(malformed, ())

    calls = 0

    def count_operation(left: Matrix, right: Matrix) -> Matrix:
        nonlocal calls
        calls += 1
        return one

    mismatch = DeclaredBilinearOperation.create(
        MatrixSpace(1, 1, QQ()), count_operation, bilinear=True
    )
    assert type(closure((one,), mismatch)) is ClosureFailure
    assert calls == 0
    malformed_product = Matrix._create(space, ((object(),),))
    failed = closure(
        (one,),
        DeclaredBilinearOperation.create(
            space, lambda left, right: malformed_product, bilinear=True
        ),
    )
    assert type(failed) is ClosureFailure and "product decomposition" in failed.reason
    assert type(closure((), object())) is ClosureFailure
    preflight = closure(
        (object(),),
        DeclaredBilinearOperation.create(space, count_operation, bilinear=True),
    )
    assert type(preflight) is ClosureFailure and calls == 0


def test_flatten_and_closure_work_bounds_are_preflighted_before_calls() -> None:
    flat_512 = MatrixSpace(1, 512, QQ()).zero()
    assert type(span_decompose(flat_512, ())) is UniqueSpan
    flat_513 = MatrixSpace(3, 171, QQ()).zero()
    with pytest.raises(SpanError) as dimension:
        span_decompose(flat_513, ())
    assert (
        dimension.value.kind,
        dimension.value.observed,
        dimension.value.maximum,
    ) == (
        "dimension",
        513,
        512,
    )

    pair_space = MatrixSpace(1, 9, QQ())
    pair_basis = tuple(
        pair_space.element(
            (tuple(1 if index == selected else 0 for index in range(9)),)
        )
        for selected in range(9)
    )
    calls = 0

    def operation(left: Matrix, right: Matrix) -> Matrix:
        nonlocal calls
        calls += 1
        return left

    pair_report = closure(
        pair_basis,
        DeclaredBilinearOperation.create(pair_space, operation, bilinear=True),
    )
    assert type(pair_report) is ClosureFailure
    assert (pair_report.kind, pair_report.observed, pair_report.maximum) == (
        "pairs",
        81,
        64,
    )
    assert calls == 0

    work_space = MatrixSpace(1, 11, QQ())
    work_basis = tuple(
        work_space.element(
            (tuple(1 if index == selected else 0 for index in range(11)),)
        )
        for selected in range(8)
    )
    report = closure(
        work_basis,
        DeclaredBilinearOperation.create(work_space, operation, bilinear=True),
    )
    assert type(report) is ClosureFailure
    assert (report.kind, report.observed, report.maximum) == ("work", 69696, 65536)
    assert calls == 0


def test_small_exhaustive_coordinate_oracle_for_flattened_matrices() -> None:
    """Finite differential-style oracle against direct coordinate construction."""
    space = MatrixSpace(1, 2, QQ())
    first, second = _matrix(space, ((1, 0),)), _matrix(space, ((0, 1),))
    for left in range(-2, 3):
        for right in range(-2, 3):
            target = _matrix(space, ((left, right),))
            result = span_decompose(target, (first, second))
            assert type(result) is UniqueSpan
            assert _column_entries(result.coordinates) == (
                QQ().element(left),
                QQ().element(right),
            )
            assert _linear_combination((first, second), result.coordinates) == target


def test_small_exhaustive_dependent_and_outside_oracles() -> None:
    space = MatrixSpace(1, 2, QQ())
    first, second = _matrix(space, ((1, 0),)), _matrix(space, ((0, 1),))
    for left in range(-1, 2):
        for right in range(-1, 2):
            target = _matrix(space, ((left, right),))
            dependent = span_decompose(target, (first, first, second))
            assert type(dependent) is DependentSpan
            assert (
                _linear_combination(
                    (first, first, second), dependent.particular_coordinates
                )
                == target
            )
            for relation in dependent.dependency_basis:
                assert (
                    _linear_combination((first, first, second), relation)
                    == space.zero()
                )
    for scalar in (-2, -1, 1, 2):
        outside = span_decompose(_matrix(space, ((0, scalar),)), (first,))
        assert type(outside) is OutsideSpan
        assert _pair(outside.functional, first).value.numerator == 0
        assert outside.target_pairing.value.numerator != 0


def test_declaration_repr_equality_and_closure_invariant_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Hostile:
        def __repr__(self) -> str:
            raise RuntimeError("secret repr")

        def __eq__(self, other: object) -> bool:
            del other
            raise RuntimeError("secret equality")

    space = MatrixSpace(1, 1, QQ())
    one = _matrix(space, ((1,),))
    hostile = DeclaredBilinearOperation.create(
        space, lambda left, right: one, bilinear=Hostile()
    )
    address_bearing = DeclaredBilinearOperation.create(
        space, lambda left, right: one, bilinear=object()
    )
    assert "secret" not in repr(hostile) and "<non-bool>" in repr(hostile)
    assert "0x" not in repr(address_bearing)
    assert hostile == hostile and hostile != address_bearing
    assert closure((one,), hostile) != closure((one,), object())

    dependent = cast(DependentSpan, span_decompose(one, (one, one)))
    monkeypatch.setattr(
        span_module, "span_decompose", lambda target, generators: dependent
    )
    declaration = DeclaredBilinearOperation.create(
        space, lambda left, right: one, bilinear=True
    )
    failure = closure((one,), declaration)
    assert type(failure) is ClosureFailure
    assert (failure.kind, failure.left_index, failure.right_index) == (
        "invariant",
        0,
        0,
    )
