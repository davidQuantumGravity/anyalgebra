from __future__ import annotations
from collections.abc import Callable, Iterator
from dataclasses import FrozenInstanceError
from itertools import repeat
from typing import cast
import pytest
from anyalgebra.core.domains import QQ, ZZ
from anyalgebra.linear.matrix import Matrix, MatrixDefinitionError, MatrixSpace


class Word:
    def __init__(self, parent: WordParent, value: str) -> None:
        self.parent, self.value = parent, value

    def add(self, other: object) -> Word:
        if type(other) is not Word or other.parent is not self.parent:
            raise TypeError("foreign")
        return Word(self.parent, f"({self.value}+{other.value})")

    def multiply(self, other: object) -> Word:
        if type(other) is not Word or other.parent is not self.parent:
            raise TypeError("foreign")
        return Word(self.parent, f"{self.value}{other.value}")

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is Word
            and self.parent is other.parent
            and self.value == other.value
        )


class WordParent:
    def element(self, value: object) -> Word:
        if type(value) is Word and value.parent is self:
            return value
        if type(value) is str:
            return Word(self, value)
        if type(value) is int and value == 0:
            return Word(self, "0")
        raise TypeError("bad word")


def test_matrix_construction_addition_and_rectangles() -> None:
    space = MatrixSpace.from_shape(2, 2, QQ())
    left = space.element(((1, 2), (3, 4)))
    right = space.element(((5, 6), (7, 8)))
    assert left.add(right).entries[0][0] == QQ().element(6)
    assert MatrixSpace.from_shape(0, 0, QQ()).element(()).entries == ()
    assert MatrixSpace.from_shape(0, 2, QQ()).element(()).entries == ()
    assert MatrixSpace.from_shape(2, 0, QQ()).element(((), ())).entries == ((), ())


def test_multiply_order_and_boundaries() -> None:
    left = MatrixSpace.from_shape(2, 3, QQ()).element(((1, 2, 3), (4, 5, 6)))
    right = MatrixSpace.from_shape(3, 1, QQ()).element(((1,), (0,), (1,)))
    product = left.multiply(right)
    assert product.entries[0][0] == QQ().element(4)
    with pytest.raises(MatrixDefinitionError, match="shape"):
        left.multiply(left)
    with pytest.raises(MatrixDefinitionError, match="maximum 512"):
        MatrixSpace.from_shape(513, 0, QQ())
    assert MatrixSpace.from_shape(512, 0, QQ()).rows == 512


def test_noncommutative_order_transpose_and_zero_inner_dimension() -> None:
    parent = WordParent()
    left = MatrixSpace(1, 2, parent).element((("a", "b"),))
    right = MatrixSpace(2, 1, parent).element((("c",), ("d",)))
    assert cast(Word, left.matmul(right).entry(0, 0)).value == "(ac+bd)"
    square = MatrixSpace(1, 1, parent).element((("x",),))
    assert square.transpose().parent is square.parent
    empty_left = MatrixSpace(2, 0, parent).zero()
    empty_right = MatrixSpace(0, 3, parent).zero()
    result = empty_left.matmul(empty_right)
    assert result.parent.rows == 2 and result.parent.columns == 3
    assert all(
        cast(Word, result.entry(i, j)).value == "0" for i in range(2) for j in range(3)
    )


def test_parent_shape_hostile_and_closed_boundaries() -> None:
    space = MatrixSpace.from_shape(1, 1, QQ())
    with pytest.raises(MatrixDefinitionError, match="construction failed"):
        space.element(((ZZ().element(1),),))
    with pytest.raises(MatrixDefinitionError, match="wrong number"):
        space.element(())

    class Broken:
        def __iter__(self) -> Iterator[object]:
            raise RuntimeError("secret")

    with pytest.raises(MatrixDefinitionError) as caught:
        space.element(Broken())
    assert "secret" not in str(caught.value) and caught.value.__context__ is None
    value = space.element(((1,),))
    for record in (space, value):
        assert not hasattr(record, "__dict__")
        with pytest.raises(TypeError):
            hash(record)
    with pytest.raises(FrozenInstanceError):
        value.parent = space  # type: ignore[misc]
    assert MatrixSpace(1, 1, QQ()).rows == 1
    with pytest.raises(MatrixDefinitionError, match=r"MatrixSpace\.element"):
        Matrix(space, ())


def test_complete_preflight_hostile_and_no_touch() -> None:
    calls = {"element": 0}

    class CountingParent(WordParent):
        def element(self, value: object) -> Word:
            calls["element"] += 1
            return super().element(value)

    parent = CountingParent()
    space = MatrixSpace(2, 1, parent)
    with pytest.raises(MatrixDefinitionError, match="row 1"):
        space.element((("a",), ("b", "extra")))
    assert calls["element"] == 0

    class BrokenNext:
        def __iter__(self) -> BrokenNext:
            return self

        def __next__(self) -> object:
            raise RuntimeError("secret next")

    with pytest.raises(MatrixDefinitionError) as caught:
        space.element(BrokenNext())
    assert (
        "secret" not in str(caught.value)
        and caught.value.__cause__ is None
        and caught.value.__context__ is None
    )
    with pytest.raises(MatrixDefinitionError, match="iterable"):
        space.element({0: ("a",)})
    with pytest.raises(MatrixDefinitionError, match="row 0"):
        MatrixSpace(1, 1, parent).element((iter(("a", "b")),))


def test_entry_result_and_parent_mismatch_boundaries() -> None:
    parent = WordParent()
    foreign = WordParent()
    left = MatrixSpace(1, 1, parent).element((("a",),))
    right = MatrixSpace(1, 1, foreign).element((("b",),))
    with pytest.raises(MatrixDefinitionError, match="literal same entry parent"):
        left.matmul(right)
    with pytest.raises(MatrixDefinitionError, match="exact Matrix"):
        left.matmul(object())
    with pytest.raises(MatrixDefinitionError, match="bounds"):
        left.entry(True, 0)
    with pytest.raises(MatrixDefinitionError, match="non-negative"):
        MatrixSpace(True, 1, parent)


def test_limits_snapshots_and_transpose_parent_semantics() -> None:
    parent = WordParent()
    with pytest.raises(MatrixDefinitionError) as axis:
        MatrixSpace(513, 0, parent)
    assert (axis.value.kind, axis.value.observed, axis.value.maximum) == (
        "axis",
        513,
        512,
    )
    assert MatrixSpace(512, 0, parent).rows == 512
    with pytest.raises(MatrixDefinitionError) as cells:
        MatrixSpace(512, 129, parent)
    assert (cells.value.kind, cells.value.observed, cells.value.maximum) == (
        "cells",
        66048,
        65536,
    )
    assert MatrixSpace(256, 256, parent).rows == 256
    rows: list[list[object]] = [[] for _ in range(512)]
    assert MatrixSpace(512, 0, parent).element(rows).parent.rows == 512
    with pytest.raises(MatrixDefinitionError) as outer:
        MatrixSpace(512, 0, parent).element([[] for _ in range(513)])
    assert (outer.value.observed, outer.value.maximum) == (513, 512)
    nested = [["a", "b"], ["c", "d"]]
    matrix = MatrixSpace(2, 2, parent).element(nested)
    nested[0][0] = "changed"
    assert cast(Word, matrix.entry(0, 0)).value == "a"
    rectangular = MatrixSpace(1, 2, parent).element((("a", "b"),)).transpose()
    assert (
        rectangular.parent is not matrix.parent
        and rectangular.parent.rows == 2
        and cast(Word, rectangular.entry(1, 0)).value == "b"
    )
    assert MatrixSpace(1, 1, parent).element((("a",),)) != MatrixSpace(
        1, 1, parent
    ).element((("a",),))


def test_matmul_cell_and_work_preflight() -> None:
    parent = WordParent()
    left = MatrixSpace(512, 0, parent).zero()
    right = MatrixSpace(0, 129, parent).zero()
    with pytest.raises(MatrixDefinitionError) as cells:
        left.matmul(right)
    assert (cells.value.kind, cells.value.observed, cells.value.maximum) == (
        "cells",
        66048,
        65536,
    )
    qq = QQ()
    exact_left = MatrixSpace(1, 512, qq).element((tuple(1 for _ in range(512)),))
    exact_right = MatrixSpace(512, 128, qq).element(
        tuple(tuple(1 for _ in range(128)) for _ in range(512))
    )
    assert exact_left.matmul(exact_right).parent.rows == 1
    over_left = MatrixSpace(2, 257, qq).element(
        tuple(tuple(1 for _ in range(257)) for _ in range(2))
    )
    over_right = MatrixSpace(257, 128, qq).element(
        tuple(tuple(1 for _ in range(128)) for _ in range(257))
    )
    with pytest.raises(MatrixDefinitionError) as work:
        over_left.matmul(over_right)
    assert (work.value.kind, work.value.observed, work.value.maximum) == (
        "work",
        65792,
        65536,
    )


def test_hostile_rows_and_conservative_equality() -> None:
    parent = WordParent()
    space = MatrixSpace(1, 1, parent)

    class BadIter:
        def __iter__(self) -> Iterator[object]:
            raise RuntimeError("secret iter")

    class BadNext:
        def __iter__(self) -> BadNext:
            return self

        def __next__(self) -> object:
            raise RuntimeError("secret next")

    for value in (BadIter(), BadNext(), (BadIter(),), (BadNext(),)):
        with pytest.raises(MatrixDefinitionError) as caught:
            space.element(value)
        assert (
            "secret" not in str(caught.value)
            and caught.value.__cause__ is None
            and caught.value.__context__ is None
        )


def test_probe_parent_and_arithmetic_failure_results_are_rejected() -> None:
    class Value:
        def __init__(self, parent: object) -> None:
            self.parent = parent

        def add(self, other: object) -> object:
            return self.parent.add(self, other)  # type: ignore[attr-defined]

        def multiply(self, other: object) -> object:
            return self.parent.multiply(self, other)  # type: ignore[attr-defined]

    class Parent:
        def __init__(self, mode: str) -> None:
            self.mode = mode

        def element(self, value: object) -> object:
            if self.mode == "element":
                raise RuntimeError("secret element")
            if self.mode == "unparented":
                return object()
            if type(value) is Value and value.parent is self:
                return value
            return Value(self)

        def add(self, left: object, right: object) -> object:
            if self.mode == "addraise":
                raise RuntimeError("secret add")
            return object() if self.mode == "addbad" else Value(self)

        def multiply(self, left: object, right: object) -> object:
            if self.mode == "mulraise":
                raise RuntimeError("secret mul")
            return object() if self.mode == "mulbad" else Value(self)

    class Lookup:
        @property
        def element(self) -> object:
            raise RuntimeError("secret lookup")

    for parent in (object(), Parent, Lookup(), Parent("element"), Parent("unparented")):
        with pytest.raises(MatrixDefinitionError) as caught:
            MatrixSpace(1, 1, parent).element((("x",),))
        assert (
            "secret" not in str(caught.value)
            and caught.value.__cause__ is None
            and caught.value.__context__ is None
        )
    for mode, method in (
        ("addraise", "add"),
        ("addbad", "add"),
        ("mulraise", "matmul"),
        ("mulbad", "matmul"),
    ):
        parent = Parent(mode)
        left = MatrixSpace(1, 1, parent).element((("x",),))
        right = MatrixSpace(1, 1, parent).element((("y",),))
        with pytest.raises(MatrixDefinitionError) as caught:
            getattr(left, method)(right)
        assert caught.value.__cause__ is None and caught.value.__context__ is None


def test_exact_sealing_and_conservative_equality_branches() -> None:
    parent = WordParent()
    space = MatrixSpace(1, 1, parent)

    class SpaceSubclass(MatrixSpace):
        pass

    class MatrixSubclass(Matrix):
        pass

    for action in (
        lambda: SpaceSubclass(1, 1, parent),
        lambda: SpaceSubclass.from_shape(1, 1, parent),
        lambda: MatrixSubclass(space, ()),
        lambda: MatrixSubclass._create(space, ()),
    ):
        with pytest.raises(MatrixDefinitionError):
            action()

    class Raising(Word):
        def __eq__(self, other: object) -> bool:
            raise RuntimeError("no equality")

    class Truthy:
        def __bool__(self) -> bool:
            return True

    class Indeterminate(Word):
        def __eq__(self, other: object) -> object:  # type: ignore[override]
            return Truthy()

    raise_left = Matrix._create(space, ((Raising(parent, "x"),),))
    raise_right = Matrix._create(space, ((Raising(parent, "x"),),))
    ind_left = Matrix._create(space, ((Indeterminate(parent, "x"),),))
    ind_right = Matrix._create(space, ((Indeterminate(parent, "x"),),))
    assert raise_left != raise_right
    assert ind_left != ind_right


def test_remaining_matrix_branches_and_repr() -> None:
    parent = WordParent()
    space = MatrixSpace(1, 1, parent)
    with pytest.raises(MatrixDefinitionError, match="wrong number of columns"):
        space.element(((),))
    assert space == space and space != MatrixSpace(1, 1, parent)
    assert "0x" not in repr(space) and "0x" not in repr(space.element((("a",),)))
    left = MatrixSpace(1, 1, parent).element((("a",),))
    right = MatrixSpace(1, 2, parent).element((("b", "c"),))
    assert left.matmul(right).parent is right.parent
    assert left.entry(0, 0) is left.entry(0, 0)


class _AuditValue:
    """Small owned value whose operations deliberately route through its parent."""

    def __init__(self, parent: object, value: object) -> None:
        self.parent, self.value = parent, value

    def add(self, other: object) -> object:
        return cast(_AuditParent, self.parent).add(self, other)

    def multiply(self, other: object) -> object:
        return cast(_AuditParent, self.parent).multiply(self, other)

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is _AuditValue
            and other.parent is self.parent
            and other.value == self.value
        )


class _AuditParent:
    """Counting parent used to prove validation completes before arithmetic."""

    def __init__(self, mode: str = "ok", foreign: object | None = None) -> None:
        self.mode, self.foreign = mode, foreign
        self.counts = {"element": 0, "add": 0, "multiply": 0}

    def reset(self) -> None:
        for key in self.counts:
            self.counts[key] = 0

    def element(self, value: object) -> object:
        self.counts["element"] += 1
        if self.mode == "element-raise":
            raise RuntimeError("secret element")
        if self.mode == "element-unparented":
            return object()
        if self.mode == "element-foreign":
            assert self.foreign is not None
            return _AuditValue(self.foreign, value)
        if type(value) is _AuditValue and value.parent is self:
            return value
        return _AuditValue(self, value)

    def add(self, left: object, right: object) -> object:
        self.counts["add"] += 1
        if self.mode == "add-raise":
            raise RuntimeError("secret add")
        if self.mode == "add-unparented":
            return object()
        if self.mode == "add-foreign":
            assert self.foreign is not None
            return _AuditValue(self.foreign, "foreign")
        return _AuditValue(self, "sum")

    def multiply(self, left: object, right: object) -> object:
        self.counts["multiply"] += 1
        if self.mode == "multiply-raise":
            raise RuntimeError("secret multiply")
        if self.mode == "multiply-unparented":
            return object()
        if self.mode == "multiply-foreign":
            assert self.foreign is not None
            return _AuditValue(self.foreign, "foreign")
        return _AuditValue(self, "product")


def _assert_sanitized_error(action: Callable[[], object], secret: str | None) -> None:
    with pytest.raises(MatrixDefinitionError) as caught:
        action()
    if secret is not None:
        assert secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_parent_provenance_and_user_failures_are_sanitized() -> None:
    """Element, add, and multiply reject actual foreign and unparented values."""
    foreign = _AuditParent()
    for mode, secret in (
        ("element-raise", "secret element"),
        ("element-unparented", "object"),
        ("element-foreign", None),
    ):
        parent = _AuditParent(mode, foreign)

        def element_action(parent: _AuditParent = parent) -> object:
            return MatrixSpace(1, 1, parent).element((("x",),))

        _assert_sanitized_error(element_action, secret)

    for mode, method, secret in (
        ("add-raise", "add", "secret add"),
        ("add-unparented", "add", "object"),
        ("add-foreign", "add", None),
        ("multiply-raise", "matmul", "secret multiply"),
        ("multiply-unparented", "matmul", "object"),
        ("multiply-foreign", "matmul", None),
    ):
        parent = _AuditParent(mode, foreign)
        space = MatrixSpace(1, 1, parent)
        left, right = space.element((("x",),)), space.element((("y",),))

        def arithmetic_action(
            left: Matrix = left, right: Matrix = right, method: str = method
        ) -> object:
            return left.add(right) if method == "add" else left.matmul(right)

        _assert_sanitized_error(arithmetic_action, secret)


def test_matmul_preflight_never_touches_counting_parent() -> None:
    """All shape/type/resource failures occur before element or arithmetic calls."""
    parent, foreign = _AuditParent(), _AuditParent()
    left = MatrixSpace(1, 1, parent).element((("x",),))
    wrong_shape = MatrixSpace(2, 1, parent).element((("y",), ("z",)))
    foreign_matrix = MatrixSpace(1, 1, foreign).element((("z",),))
    output_left = MatrixSpace(512, 0, parent).zero()
    output_right = MatrixSpace(0, 129, parent).zero()
    work_left = MatrixSpace(2, 257, parent).element(
        tuple(tuple("x" for _ in range(257)) for _ in range(2))
    )
    work_right = MatrixSpace(257, 128, parent).element(
        tuple(tuple("y" for _ in range(128)) for _ in range(257))
    )
    parent.reset()
    foreign.reset()
    actions: tuple[Callable[[], Matrix], ...] = (
        lambda: left.matmul(object()),
        lambda: left.matmul(foreign_matrix),
        lambda: left.matmul(wrong_shape),
        lambda: output_left.matmul(output_right),
        lambda: work_left.matmul(work_right),
    )
    for action in actions:
        with pytest.raises(MatrixDefinitionError):
            action()
        assert parent.counts == {"element": 0, "add": 0, "multiply": 0}
        assert foreign.counts == {"element": 0, "add": 0, "multiply": 0}


def test_matmul_result_parent_reuse_freshness_and_cross_space_addition() -> None:
    parent = WordParent()
    left_space = MatrixSpace(1, 1, parent)
    right_space = MatrixSpace(1, 2, parent)
    assert (
        left_space.element((("a",),)).matmul(left_space.element((("b",),))).parent
        is left_space
    )
    assert (
        left_space.element((("a",),)).matmul(right_space.element((("b", "c"),))).parent
        is right_space
    )
    wide = MatrixSpace(2, 3, parent)
    tall = MatrixSpace(3, 4, parent)
    fresh = wide.element((("a", "b", "c"), ("d", "e", "f"))).matmul(
        tall.element((("g", "h", "i", "j"), ("k", "l", "m", "n"), ("o", "p", "q", "r")))
    )
    assert fresh.parent is not wide and fresh.parent is not tall
    with pytest.raises(MatrixDefinitionError, match="literal same MatrixSpace"):
        left_space.element((("a",),)).add(MatrixSpace(1, 1, parent).element((("a",),)))


def test_exact_work_limit_product_has_exact_first_and_last_entries() -> None:
    parent = QQ()
    left = MatrixSpace(1, 512, parent).element((tuple(1 for _ in range(512)),))
    right = MatrixSpace(512, 128, parent).element(
        tuple(tuple(1 for _ in range(128)) for _ in range(512))
    )
    result = left.matmul(right)
    assert result.entry(0, 0) == parent.element(512)
    assert result.entry(0, 127) == parent.element(512)


def test_iterators_are_bounded_and_exact_axis_boundaries_are_accepted() -> None:
    parent = WordParent()
    outer_space = MatrixSpace(2, 0, parent)
    with pytest.raises(MatrixDefinitionError) as outer:
        outer_space.element(repeat(()))
    assert (outer.value.kind, outer.value.observed, outer.value.maximum) == (
        "rows",
        3,
        2,
    )
    row_space = MatrixSpace(1, 2, parent)
    with pytest.raises(MatrixDefinitionError) as row:
        row_space.element((repeat("x"),))
    assert (row.value.kind, row.value.observed, row.value.maximum) == ("row 0", 3, 2)
    assert MatrixSpace(1, 512, parent).element((tuple("x" for _ in range(512)),))
    with pytest.raises(MatrixDefinitionError, match="iterable"):
        MatrixSpace(1, 1, parent).element(({0: "x"},))


@pytest.mark.parametrize(
    "rows, columns",
    [(-1, 0), (0, -1), (1.0, 0), (0, 1.0), (True, 0), (0, True)],
)
def test_invalid_dimensions_and_entry_indices(rows: object, columns: object) -> None:
    with pytest.raises(MatrixDefinitionError, match="non-negative built-in int"):
        MatrixSpace(rows, columns, WordParent())
    matrix = MatrixSpace(2, 3, WordParent()).element((("a", "b", "c"), ("d", "e", "f")))
    for row, column in ((-1, 0), (0, -1), (True, 0), (0, True), (2, 0), (0, 3)):
        with pytest.raises(MatrixDefinitionError, match="bounds"):
            matrix.entry(row, column)


def test_snapshot_replacement_append_and_safe_repr() -> None:
    parent = WordParent()
    rows: list[list[object]] = [["a", "b"], ["c", "d"]]
    matrix = MatrixSpace(2, 2, parent).element(rows)
    rows[0][0] = "changed"
    rows[1] = ["replacement", "replacement"]
    rows.append(["extra", "extra"])
    assert cast(Word, matrix.entry(0, 0)).value == "a"
    assert cast(Word, matrix.entry(1, 0)).value == "c"
    assert "0x" not in repr(matrix) and "0x" not in repr(matrix.parent)


@pytest.mark.parametrize(
    "mode, secret",
    [
        ("element-raise", "secret element"),
        ("element-unparented", "object"),
        ("element-foreign", None),
    ],
)
def test_zero_inner_product_rejects_bad_zero_results(
    mode: str, secret: str | None
) -> None:
    foreign = _AuditParent()
    parent = _AuditParent(mode, foreign)
    left = MatrixSpace(1, 0, parent).zero()
    right = MatrixSpace(0, 1, parent).zero()
    _assert_sanitized_error(lambda: left.matmul(right), secret)
    assert parent.counts == {"element": 1, "add": 0, "multiply": 0}


def test_conservative_equality_identity_true_false_and_zero_columns() -> None:
    parent = WordParent()
    space = MatrixSpace(1, 1, parent)
    marker = Word(parent, "x")
    identity_left = Matrix._create(space, ((marker,),))
    identity_right = Matrix._create(space, ((marker,),))
    assert identity_left == identity_right
    assert space.element((("x",),)) == space.element((("x",),))
    assert space.element((("x",),)) != space.element((("y",),))
    zero_space = MatrixSpace(2, 0, parent)
    assert zero_space.zero() == zero_space.zero()
