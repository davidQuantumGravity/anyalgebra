"""Neutral exact finite matrices with explicit literal entry-parent ownership.

Entry parents supply ``element(value)``; owned entries supply explicit ``add``
and ``multiply`` methods.  Algebra adapters with operations separate from values
and determinant/inverse/rank semantics are intentionally deferred.
"""

from __future__ import annotations
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast
from anyalgebra.core.errors import AnyAlgebraError

_MAX_AXIS = 512
_MAX_CELLS = 65_536
_MAX_MULTIPLICATION_WORK = 65_536


class MatrixDefinitionError(AnyAlgebraError, ValueError):
    def __init__(
        self,
        reason: str,
        *,
        kind: str | None = None,
        observed: int | None = None,
        maximum: int | None = None,
    ) -> None:
        self.reason, self.kind, self.observed, self.maximum = (
            reason,
            kind,
            observed,
            maximum,
        )
        suffix = (
            ""
            if kind is None
            else f"; kind={kind}; observed={observed}; maximum={maximum}"
        )
        super().__init__(f"invalid matrix definition: {reason}{suffix}")


def _name(error: Exception) -> str:
    name = type(error).__name__
    return name if type(name) is str and name.isidentifier() else "exception"


def _number(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise MatrixDefinitionError(f"{field} must be a non-negative built-in int")
    if value > _MAX_AXIS:
        raise MatrixDefinitionError(
            f"{field} exceeds maximum {_MAX_AXIS}",
            kind="axis",
            observed=value,
            maximum=_MAX_AXIS,
        )
    return value


def _parent(value: object) -> object:
    failure = None
    if isinstance(value, type):
        raise MatrixDefinitionError("entry_parent must supply callable element(value)")
    try:
        capability = getattr(value, "element", None)
    except Exception as error:
        capability = None
        failure = _name(error)
    if failure is not None or not callable(capability):
        raise MatrixDefinitionError("entry_parent must supply callable element(value)")
    return value


def _snapshot(value: object, maximum: int, field: str) -> tuple[object, ...]:
    if isinstance(value, str | bytes | Mapping):
        raise MatrixDefinitionError(f"{field} must be an iterable")
    failure = None
    try:
        iterator = iter(cast(Iterable[object], value))
    except Exception as error:
        iterator = None
        failure = _name(error)
    if failure is not None:
        raise MatrixDefinitionError(f"{field} could not be iterated ({failure})")
    assert iterator is not None
    result = []
    for _ in range(maximum + 1):
        try:
            item = next(iterator)
        except StopIteration:
            break
        except Exception as error:
            failure = _name(error)
            break
        result.append(item)
    if failure is not None:
        raise MatrixDefinitionError(f"{field} could not be iterated ({failure})")
    if len(result) > maximum:
        raise MatrixDefinitionError(
            f"{field} exceeds its declared bound",
            kind=field,
            observed=len(result),
            maximum=maximum,
        )
    return tuple(result)


def _owned(value: object, parent: object) -> bool:
    try:
        return cast(Any, value).parent is parent
    except Exception:
        return False


def _call(
    entry: object, method: str, argument: object, parent: object, stage: str
) -> object:
    failure = None
    try:
        result = getattr(entry, method)(argument)
    except Exception as error:
        result = None
        failure = _name(error)
    if failure is not None or not _owned(result, parent):
        raise MatrixDefinitionError(f"{stage} failed ({failure or 'foreign-result'})")
    return result


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class MatrixSpace:
    rows: int
    columns: int
    entry_parent: object
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, rows: object, columns: object, entry_parent: object) -> None:
        if type(self) is not MatrixSpace:
            raise MatrixDefinitionError("factory requires exact MatrixSpace")
        row, column = _number(rows, "rows"), _number(columns, "columns")
        if row * column > _MAX_CELLS:
            raise MatrixDefinitionError(
                "shape exceeds cell limit",
                kind="cells",
                observed=row * column,
                maximum=_MAX_CELLS,
            )
        object.__setattr__(self, "rows", row)
        object.__setattr__(self, "columns", column)
        object.__setattr__(self, "entry_parent", _parent(entry_parent))

    @classmethod
    def from_shape(
        cls, rows: object, columns: object, entry_parent: object
    ) -> MatrixSpace:
        if cls is not MatrixSpace:
            raise MatrixDefinitionError("factory requires exact MatrixSpace")
        return cls(rows, columns, entry_parent)

    def element(self, entries: object) -> Matrix:
        rows = _snapshot(entries, self.rows, "rows")
        if len(rows) != self.rows:
            raise MatrixDefinitionError("wrong number of rows")
        raw = []
        for i, row in enumerate(rows):
            values = _snapshot(row, self.columns, f"row {i}")
            if len(values) != self.columns:
                raise MatrixDefinitionError(f"row {i} has wrong number of columns")
            raw.append(values)
        # Structural preflight above completes before parent.element can run.
        normalized = []
        for i, row in enumerate(raw):
            output = []
            for j, item in enumerate(row):
                failure = None
                try:
                    entry = cast(Any, self.entry_parent).element(item)
                except Exception as error:
                    entry = None
                    failure = _name(error)
                if failure is not None or not _owned(entry, self.entry_parent):
                    raise MatrixDefinitionError(
                        f"entry ({i}, {j}) construction failed "
                        f"({failure or 'foreign-result'})"
                    )
                output.append(entry)
            normalized.append(tuple(output))
        return Matrix._create(self, tuple(normalized))

    def zero(self) -> Matrix:
        return self.element(
            tuple(tuple(0 for _ in range(self.columns)) for _ in range(self.rows))
        )

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        return f"MatrixSpace(rows={self.rows}, columns={self.columns})"


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class Matrix:
    parent: MatrixSpace
    entries: tuple[tuple[object, ...], ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise MatrixDefinitionError("use MatrixSpace.element")

    @classmethod
    def _create(
        cls, parent: MatrixSpace, entries: tuple[tuple[object, ...], ...]
    ) -> Matrix:
        if cls is not Matrix:
            raise MatrixDefinitionError("factory requires exact Matrix")
        value = object.__new__(cls)
        object.__setattr__(value, "parent", parent)
        object.__setattr__(value, "entries", entries)
        return value

    def entry(self, row: object, column: object) -> object:
        if (
            type(row) is not int
            or type(column) is not int
            or row < 0
            or column < 0
            or row >= self.parent.rows
            or column >= self.parent.columns
        ):
            raise MatrixDefinitionError("entry index is outside matrix bounds")
        return self.entries[row][column]

    def transpose(self) -> Matrix:
        space = (
            self.parent
            if self.parent.rows == self.parent.columns
            else MatrixSpace(
                self.parent.columns, self.parent.rows, self.parent.entry_parent
            )
        )
        return Matrix._create(
            space,
            tuple(
                tuple(self.entries[i][j] for i in range(self.parent.rows))
                for j in range(self.parent.columns)
            ),
        )

    def add(self, other: object) -> Matrix:
        if type(other) is not Matrix or other.parent is not self.parent:
            raise MatrixDefinitionError(
                "addition requires the literal same MatrixSpace"
            )
        return Matrix._create(
            self.parent,
            tuple(
                tuple(
                    _call(
                        entry,
                        "add",
                        other.entries[i][j],
                        self.parent.entry_parent,
                        "addition",
                    )
                    for j, entry in enumerate(row)
                )
                for i, row in enumerate(self.entries)
            ),
        )

    def matmul(self, other: object) -> Matrix:
        if type(other) is not Matrix:
            raise MatrixDefinitionError("matmul requires an exact Matrix")
        if self.parent.entry_parent is not other.parent.entry_parent:
            raise MatrixDefinitionError("matmul requires the literal same entry parent")
        if self.parent.columns != other.parent.rows:
            raise MatrixDefinitionError("matrix multiplication shape mismatch")
        output_cells = self.parent.rows * other.parent.columns
        if output_cells > _MAX_CELLS:
            raise MatrixDefinitionError(
                "matmul output shape exceeds cell limit",
                kind="cells",
                observed=output_cells,
                maximum=_MAX_CELLS,
            )
        work = self.parent.rows * other.parent.columns * self.parent.columns
        if work > _MAX_MULTIPLICATION_WORK:
            raise MatrixDefinitionError(
                "matmul exceeds work limit",
                kind="work",
                observed=work,
                maximum=_MAX_MULTIPLICATION_WORK,
            )
        if (self.parent.rows, self.parent.columns) == (
            self.parent.rows,
            other.parent.columns,
        ):
            space = self.parent
        elif (other.parent.rows, other.parent.columns) == (
            self.parent.rows,
            other.parent.columns,
        ):
            space = other.parent
        else:
            space = MatrixSpace(
                self.parent.rows, other.parent.columns, self.parent.entry_parent
            )
        rows = []
        for i in range(space.rows):
            row = []
            for j in range(space.columns):
                if self.parent.columns == 0:
                    failure = None
                    try:
                        total = cast(Any, space.entry_parent).element(0)
                    except Exception as error:
                        total = None
                        failure = _name(error)
                    if failure is not None or not _owned(total, space.entry_parent):
                        raise MatrixDefinitionError(
                            f"zero construction failed ({failure or 'foreign-result'})"
                        )
                else:
                    total = _call(
                        self.entries[i][0],
                        "multiply",
                        other.entries[0][j],
                        space.entry_parent,
                        "multiplication",
                    )
                    for k in range(1, self.parent.columns):
                        total = _call(
                            total,
                            "add",
                            _call(
                                self.entries[i][k],
                                "multiply",
                                other.entries[k][j],
                                space.entry_parent,
                                "multiplication",
                            ),
                            space.entry_parent,
                            "addition",
                        )
                row.append(total)
            rows.append(tuple(row))
        return Matrix._create(space, tuple(rows))

    multiply = matmul

    def __eq__(self, other: object) -> bool:
        if type(other) is not Matrix or self.parent is not other.parent:
            return False
        for left_row, right_row in zip(self.entries, other.entries, strict=True):
            for left, right in zip(left_row, right_row, strict=True):
                if left is right:
                    continue
                try:
                    equal = left == right
                except Exception:
                    return False
                if type(equal) is not bool or not equal:
                    return False
        return True

    def __repr__(self) -> str:
        return f"Matrix(rows={self.parent.rows}, columns={self.parent.columns})"
