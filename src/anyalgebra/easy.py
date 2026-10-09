"""A convenience layer for everyday calculation.

The kernel is deliberately explicit: elements belong to a module, a product
names its structure, and nothing is printed prettily.  This module wraps
kernel objects, without changing them, so that a calculation reads like
mathematics::

    import anyalgebra.easy as aa

    H = aa.quaternions()
    one, i, j, k = H.basis
    x = (1 + 2 * i) * (j + k) / 2
    print(x)  # -1/2*j + 3/2*k
    print(f"{x:v}")  # [0, 0, -1/2, 3/2]

Every handle uses exact rational coefficients.  ``a * b * c`` means
``(a * b) * c`` because Python evaluates it left to right; write the
parentheses, or use :func:`assoc`, when the algebra is not associative.
``x.raw`` is the kernel element and ``A.structure`` the kernel structure.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from fractions import Fraction
from typing import Final

from anyalgebra.algebra.multilinear import (
    FiniteMultilinearStructure,
    StructureConstants,
    evaluate_multilinear,
)
from anyalgebra.analysis.elementary import AnalysisBounds
from anyalgebra.analysis.fingerprint import algebra_fingerprint
from anyalgebra.core.domains import QQ
from anyalgebra.core.elements import SparseElement
from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.fixtures import composition

Scalar = int | Fraction
Form = Callable[["Element"], str]

_UNIT_LABEL: Final = "1"


class EasyError(AnyAlgebraError, TypeError):
    """A convenience-layer request that has no unambiguous meaning."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


# --- print forms -------------------------------------------------------------

_FORMS: dict[str, Form] = {}
_DISPLAY: list[str] = ["sum"]


def register_form(*names: str) -> Callable[[Form], Form]:
    """Register a print form under one or more names.

    A form is a function from an element to text.  It becomes available as
    ``format(x, name)``, ``f"{x:name}"`` and ``aa.display(name)``.
    """
    if not names or any(type(name) is not str or not name for name in names):
        raise EasyError("form", "a print form needs at least one nonempty name")

    def register(function: Form) -> Form:
        for name in names:
            _FORMS[name] = function
        return function

    return register


def forms() -> tuple[str, ...]:
    """Return the registered form names in sorted order."""
    return tuple(sorted(_FORMS))


def _form(name: str) -> Form:
    try:
        return _FORMS[name]
    except KeyError:
        known = ", ".join(forms())
        raise EasyError(
            "form", f"unknown print form {name!r}; known: {known}"
        ) from None


@contextlib.contextmanager
def display(name: str) -> Iterator[None]:
    """Use one print form as the default inside a ``with`` block."""
    _form(name)
    _DISPLAY.append(name)
    try:
        yield
    finally:
        _DISPLAY.pop()


def set_display(name: str) -> None:
    """Change the default print form for the rest of the session."""
    _form(name)
    _DISPLAY[0] = name


def _number(value: Fraction, *, latex: bool = False) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    if latex:
        return rf"\frac{{{value.numerator}}}{{{value.denominator}}}"
    return f"{value.numerator}/{value.denominator}"


def _latex_label(label: str) -> str:
    head = label.rstrip("0123456789")
    if head and head != label:
        return f"{head}_{{{label[len(head) :]}}}"
    return label


def _terms(x: Element, *, latex: bool) -> str:
    parts: list[tuple[bool, str]] = []
    for label, value in zip(x.algebra.labels, x.vector, strict=True):
        if value == 0:
            continue
        size = abs(value)
        shown = _latex_label(label) if latex else label
        if label == _UNIT_LABEL:
            body = _number(size, latex=latex)
        elif size == 1:
            body = shown
        else:
            body = _number(size, latex=latex) + (r"\," if latex else "*") + shown
        parts.append((value < 0, body))
    if not parts:
        return "0"
    text = ("-" if parts[0][0] else "") + parts[0][1]
    return text + "".join(
        (" - " if negative else " + ") + body for negative, body in parts[1:]
    )


@register_form("sum", "s")
def _as_sum(x: Element) -> str:
    return _terms(x, latex=False)


@register_form("latex", "l")
def _as_latex(x: Element) -> str:
    return _terms(x, latex=True)


@register_form("vector", "v")
def _as_vector(x: Element) -> str:
    return "[" + ", ".join(_number(value) for value in x.vector) + "]"


@register_form("sparse", "d")
def _as_sparse(x: Element) -> str:
    pairs = (
        f"{label}: {_number(value)}"
        for label, value in zip(x.algebra.labels, x.vector, strict=True)
        if value != 0
    )
    return "{" + ", ".join(pairs) + "}"


# --- elements ----------------------------------------------------------------


class Element:
    """An element of one :class:`Algebra`, with operators and print forms."""

    __slots__ = ("algebra", "raw")
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, algebra: Algebra, raw: SparseElement) -> None:
        if raw.parent is not algebra.structure.module:
            raise EasyError("parent", "the element does not belong to this algebra")
        self.algebra = algebra
        self.raw = raw

    @property
    def vector(self) -> tuple[Fraction, ...]:
        """Return the coordinates in basis order as exact fractions."""
        found = {
            index: Fraction(value.value.numerator, value.value.denominator)  # type: ignore[attr-defined]
            for index, value in self.raw.coordinates().items()
        }
        return tuple(
            found.get(index, Fraction(0)) for index in range(self.algebra.rank)
        )

    @property
    def coefficients(self) -> dict[str, Fraction]:
        """Return the nonzero coordinates keyed by basis label."""
        return {
            label: value
            for label, value in zip(self.algebra.labels, self.vector, strict=True)
            if value != 0
        }

    def _operand(self, other: object) -> Element | None:
        if isinstance(other, Element):
            if other.algebra is not self.algebra:
                raise EasyError(
                    "parent",
                    f"cannot combine elements of {self.algebra.name} "
                    f"and {other.algebra.name}",
                )
            return other
        if type(other) is int or isinstance(other, Fraction):
            return self.algebra.scalar(other)
        return None

    def __add__(self, other: object) -> Element:
        right = self._operand(other)
        if right is None:
            return NotImplemented
        return Element(self.algebra, self.raw.add(right.raw))

    __radd__ = __add__

    def __neg__(self) -> Element:
        return Element(self.algebra, self.raw.negate())

    def __pos__(self) -> Element:
        return self

    def __sub__(self, other: object) -> Element:
        right = self._operand(other)
        if right is None:
            return NotImplemented
        return Element(self.algebra, self.raw.subtract(right.raw))

    def __rsub__(self, other: object) -> Element:
        left = self._operand(other)
        if left is None:
            return NotImplemented
        return Element(self.algebra, left.raw.subtract(self.raw))

    def __mul__(self, other: object) -> Element:
        if type(other) is int or isinstance(other, Fraction):
            return Element(
                self.algebra, self.raw.scale(self.algebra.coefficient(other))
            )
        right = self._operand(other)
        if right is None:
            return NotImplemented
        product = evaluate_multilinear(self.algebra.structure, self.raw, right.raw)
        return Element(self.algebra, product)

    def __rmul__(self, other: object) -> Element:
        if type(other) is int or isinstance(other, Fraction):
            return self * other
        return NotImplemented

    def __truediv__(self, other: object) -> Element:
        if not (type(other) is int or isinstance(other, Fraction)):
            return NotImplemented
        if other == 0:
            raise EasyError("division", "division by the scalar zero")
        return self * (Fraction(1) / other)

    def __pow__(self, exponent: object) -> Element:
        if type(exponent) is not int or exponent < 1:
            raise EasyError("power", "a power needs an integer exponent of at least 1")
        result = self
        for _ in range(exponent - 1):
            result = result * self
        return result

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Element):
            return other.algebra is self.algebra and self.raw == other.raw
        if (type(other) is int or isinstance(other, Fraction)) and (
            self.algebra.unit is not None
        ):
            return self.raw == self.algebra.scalar(other).raw
        return False

    def __bool__(self) -> bool:
        return any(value != 0 for value in self.vector)

    def conj(self) -> Element:
        """Return the conjugate under the algebra's declared involution."""
        signs = self.algebra.conjugation_signs
        if signs is None:
            raise EasyError(
                "involution", f"{self.algebra.name} declares no conjugation"
            )
        return self.algebra(
            *(sign * value for sign, value in zip(signs, self.vector, strict=True))
        )

    def norm(self) -> Fraction:
        """Return the quadratic norm ``x * conj(x)`` as a rational number."""
        value = self * self.conj()
        unit = self.algebra.unit_index
        scalar = value.vector[unit] if unit is not None else Fraction(0)
        if unit is None or value != self.algebra.scalar(scalar):
            raise EasyError("norm", "x * conj(x) is not a multiple of the unit")
        return scalar

    def inv(self) -> Element:
        """Return ``conj(x) / norm(x)`` when the norm is not zero."""
        norm = self.norm()
        if norm == 0:
            raise EasyError("division", "the element has norm zero and no inverse")
        return self.conj() / norm

    def __format__(self, spec: str) -> str:
        if spec:
            return _form(spec)(self)
        if len(_DISPLAY) > 1:
            return _form(_DISPLAY[-1])(self)
        return _form(self.algebra.display or _DISPLAY[0])(self)

    def __str__(self) -> str:
        return format(self, "")

    def __repr__(self) -> str:
        return f"{self.algebra.name}({_as_sum(self)})"

    def _repr_latex_(self) -> str:
        return f"${_as_latex(self)}$"


# --- algebras ----------------------------------------------------------------


def over_rationals(structure: FiniteMultilinearStructure) -> FiniteMultilinearStructure:
    """Rebuild a binary integer or rational algebra over the rationals.

    The center, nucleus, derivation and fingerprint solvers require rational
    coefficients, while the named fixtures are integer tables.
    """
    if type(structure) is not FiniteMultilinearStructure or structure.arity != 2:
        raise EasyError("structure", "a binary module-backed algebra is required")
    domain = QQ()
    if structure.module.domain is domain:
        return structure
    module = FreeModule(
        domain, Basis(structure.module.basis.labels, coefficient_domain=domain)
    )
    entries = tuple(
        (key, domain.element((_integer(value.value), 1)))
        for key, value in structure.operation.constants.entries
    )
    constants = StructureConstants.from_sparse(
        (module.basis, module.basis), module.basis, entries
    )
    return FiniteMultilinearStructure.from_structure_constants(
        module, constants, name=structure.name
    )


def _integer(value: object) -> int:
    if type(value) is not int:  # pragma: no cover - no third exact domain exists
        raise EasyError("structure", "the coefficients are not integers")
    return value


class Algebra:
    """A handle on one binary algebra with rational coefficients."""

    def __init__(
        self,
        structure: FiniteMultilinearStructure,
        *,
        name: str | None = None,
        unit: str | None = _UNIT_LABEL,
        conjugation_signs: Sequence[int] | None = None,
        display: str | None = None,
    ) -> None:
        self.structure: Final = over_rationals(structure)
        module = self.structure.module
        self.labels: Final[tuple[str, ...]] = tuple(
            str(label) for label in module.basis.labels
        )
        self.rank: Final[int] = module.rank
        self.name: Final[str] = name or self.structure.name or "A"
        if display is not None:
            _form(display)
        self.display = display
        self.unit_index: Final[int | None] = (
            self.labels.index(unit)
            if unit is not None and unit in self.labels
            else None
        )
        if conjugation_signs is not None and (
            len(conjugation_signs) != self.rank
            or any(sign not in (1, -1) for sign in conjugation_signs)
        ):
            raise EasyError("involution", "one sign, +1 or -1, per basis element")
        self.conjugation_signs: Final[tuple[int, ...] | None] = (
            None if conjugation_signs is None else tuple(conjugation_signs)
        )
        self.basis: Final[tuple[Element, ...]] = tuple(
            Element(self, module.element({index: 1})) for index in range(self.rank)
        )

    # coefficients and scalars

    def coefficient(self, value: Scalar) -> object:
        """Return a Python number as an element of the coefficient domain."""
        exact = Fraction(value)
        return self.structure.module.domain.element(
            (exact.numerator, exact.denominator)
        )

    @property
    def unit(self) -> Element | None:
        """Return the declared unit element, if the algebra has one."""
        return None if self.unit_index is None else self.basis[self.unit_index]

    @property
    def zero(self) -> Element:
        """Return the zero element."""
        return Element(self, self.structure.module.zero())

    def scalar(self, value: Scalar) -> Element:
        """Return a number as a multiple of the declared unit."""
        unit = self.unit
        if unit is None:
            raise EasyError(
                "unit",
                f"{self.name} declares no unit, so the scalar {value} "
                "is not one of its elements",
            )
        return unit * value

    # naming elements

    def __call__(self, *coordinates: Scalar) -> Element:
        """Build an element from its coordinates in basis order."""
        if len(coordinates) != self.rank:
            raise EasyError(
                "coordinates", f"{self.name} needs exactly {self.rank} coordinates"
            )
        module = self.structure.module
        pairs = {
            index: self.coefficient(value)
            for index, value in enumerate(coordinates)
            if value != 0
        }
        return Element(self, module.element(pairs))

    def _index(self, label: str) -> int | None:
        if label == "one" and self.unit_index is not None:
            return self.unit_index
        return self.labels.index(label) if label in self.labels else None

    def __getitem__(self, key: int | str) -> Element:
        if type(key) is int:
            if not -self.rank <= key < self.rank:
                raise EasyError("symbol", f"{self.name} has no basis element {key}")
            return self.basis[key]
        index = self._index(key) if type(key) is str else None
        if index is None:
            raise EasyError("symbol", f"{self.name} has no basis element {key!r}")
        return self.basis[index]

    def __getattr__(self, label: str) -> Element:
        labels = self.__dict__.get("labels")
        if labels is None or label.startswith("_"):
            raise AttributeError(label)
        index = self._index(label)
        if index is None:
            raise AttributeError(f"{self.name} has no basis element {label!r}")
        return self.basis[index]

    def symbols(self) -> dict[str, Element]:
        """Return the basis elements whose labels are valid Python names."""
        named = {
            label: element
            for label, element in zip(self.labels, self.basis, strict=True)
            if label.isidentifier()
        }
        if self.unit is not None and "one" not in named:
            named["one"] = self.unit
        return named

    def inject(self, namespace: dict[str, object] | None = None) -> tuple[str, ...]:
        """Define the basis symbols in a namespace and return their names.

        Without an argument the caller's module namespace is used, which is
        meant for interactive sessions and scripts.  Inside a function, pass
        the dictionary to fill.
        """
        target = sys._getframe(1).f_globals if namespace is None else namespace
        named = self.symbols()
        target.update(named)
        return tuple(named)

    # tables and reports

    def table(self, form: str = "sum") -> str:
        """Return the multiplication table as aligned text in one print form."""
        render = _form(form)
        cells = [[render(a * b) for b in self.basis] for a in self.basis]
        width = max(len(cell) for row in cells for cell in row) if cells else 0
        width = max([width, *(len(label) for label in self.labels)]) + 2
        side = max((len(label) for label in self.labels), default=0) + 2
        lines = [" " * side + "".join(label.rjust(width) for label in self.labels)]
        for label, row in zip(self.labels, cells, strict=True):
            lines.append(label.ljust(side) + "".join(cell.rjust(width) for cell in row))
        return "\n".join(lines)

    def report(self, *, max_work: int = 5_000_000) -> str:
        """Return the exact fingerprint of the algebra as a short text report.

        ``max_work`` bounds the exact elimination; the octonions need about
        three million units.
        """
        found = algebra_fingerprint(
            self.structure, options=AnalysisBounds(max_elimination_work=max_work)
        )
        rows = (
            ("dimension", found.dimension),
            ("commutative", found.commutative),
            ("associative", found.associative),
            ("unit", found.unit_status),
            ("center dimension", found.center_dimension),
            ("nucleus dimension", found.nucleus_dimension),
            ("derivation dimension", found.derivation_dimension),
        )
        width = max(len(title) for title, _ in rows)
        head = f"{self.name}: basis " + ", ".join(self.labels)
        return "\n".join([head, *(f"  {t.ljust(width)}  {v}" for t, v in rows)])

    def __repr__(self) -> str:
        return f"Algebra({self.name!r}, rank={self.rank})"


# --- constructors ------------------------------------------------------------


def wrap(
    structure: FiniteMultilinearStructure,
    *,
    name: str | None = None,
    unit: str | None = _UNIT_LABEL,
    conjugation_signs: Sequence[int] | None = None,
) -> Algebra:
    """Wrap any binary kernel algebra in a convenience handle."""
    return Algebra(structure, name=name, unit=unit, conjugation_signs=conjugation_signs)


def _composition(name: str, source: FiniteMultilinearStructure) -> Algebra:
    signs = (1, *(-1 for _ in range(source.module.rank - 1)))
    return Algebra(source, name=name, conjugation_signs=signs)


def quaternions() -> Algebra:
    """Return the quaternions ``H`` in the pinned ``algmul.H.v1`` convention."""
    return _composition("H", composition.quaternion_fixture())


def split_quaternions() -> Algebra:
    """Return the split quaternions ``Hs`` in the ``algmul.Hs.v1`` convention."""
    return _composition("Hs", composition.split_quaternion_fixture())


def octonions() -> Algebra:
    """Return the octonions ``O`` in the pinned ``algmul.O.v1`` convention."""
    return _composition("O", composition.octonion_fixture())


def split_octonions() -> Algebra:
    """Return the split octonions ``Os`` in the ``algmul.Os.v1`` convention."""
    return _composition("Os", composition.split_octonion_fixture())


Product = Mapping[str, Scalar] | str | int


def _product_entries(
    labels: tuple[str, ...], pair: tuple[str, str], value: Product
) -> dict[str, Fraction]:
    if isinstance(value, Mapping):
        found = {str(label): Fraction(number) for label, number in value.items()}
    elif value == 0 or value == "0":
        found = {}
    elif type(value) is str:
        negative = value.startswith("-")
        found = {value.lstrip("+-"): Fraction(-1 if negative else 1)}
    else:
        raise EasyError("table", f"the product {pair} is not a label or a mapping")
    unknown = sorted(set(found) - set(labels))
    if unknown:
        raise EasyError("table", f"the product {pair} names unknown labels {unknown}")
    return found


def algebra(
    labels: Sequence[str],
    table: Mapping[tuple[str, str], Product],
    *,
    name: str = "A",
    unit: str | None = _UNIT_LABEL,
) -> Algebra:
    """Build an algebra from basis labels and a table of basis products.

    ``table`` maps a pair of labels to their product: a label such as
    ``"z"``, a signed label such as ``"-z"``, ``0``, or a mapping from labels
    to coefficients.  A pair that is left out has product zero.
    """
    names = tuple(labels)
    if (
        not names
        or len(set(names)) != len(names)
        or any(type(label) is not str or not label for label in names)
    ):
        raise EasyError("table", "the basis labels must be distinct nonempty strings")
    domain = QQ()
    module = FreeModule(domain, Basis(names, coefficient_domain=domain))
    entries = []
    for pair, value in table.items():
        if len(pair) != 2 or pair[0] not in names or pair[1] not in names:
            raise EasyError("table", f"{pair!r} is not a pair of basis labels")
        for label, number in _product_entries(names, pair, value).items():
            if number != 0:
                key = (names.index(pair[0]), names.index(pair[1]), names.index(label))
                entries.append(
                    (key, domain.element((number.numerator, number.denominator)))
                )
    constants = StructureConstants.from_sparse(
        (module.basis, module.basis), module.basis, tuple(entries)
    )
    structure = FiniteMultilinearStructure.from_structure_constants(
        module, constants, name=name
    )
    return Algebra(structure, name=name, unit=unit)


# --- named operations --------------------------------------------------------


def comm(x: Element, y: Element) -> Element:
    """Return the commutator ``x*y - y*x``."""
    return x * y - y * x


def anticomm(x: Element, y: Element) -> Element:
    """Return the anticommutator ``x*y + y*x``."""
    return x * y + y * x


def assoc(x: Element, y: Element, z: Element) -> Element:
    """Return the associator ``(x*y)*z - x*(y*z)``."""
    return (x * y) * z - x * (y * z)
