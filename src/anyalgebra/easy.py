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
import functools
import html
import inspect
import itertools
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from fractions import Fraction
from typing import Final, TypeVar

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
from anyalgebra.polynomials import Polynomial
from anyalgebra.structures.evaluate import EvaluationResult
from anyalgebra.structures.outcomes import Defined
from anyalgebra.structures.terms import Term
from anyalgebra.validation.validate import Disproved, Proved, ValidationReport

Scalar = int | Fraction
Form = Callable[["Element"], str]
AnyElement = TypeVar("AnyElement", "Element", "Generic")

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
        if isinstance(other, Polynomial):
            return self.symbolic() + other  # type: ignore[return-value]
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
        if isinstance(other, Polynomial):
            return self.symbolic() - other  # type: ignore[return-value]
        right = self._operand(other)
        if right is None:
            return NotImplemented
        return Element(self.algebra, self.raw.subtract(right.raw))

    def __rsub__(self, other: object) -> Element:
        if isinstance(other, Polynomial):
            return other - self.symbolic()  # type: ignore[return-value]
        left = self._operand(other)
        if left is None:
            return NotImplemented
        return Element(self.algebra, left.raw.subtract(self.raw))

    def __mul__(self, other: object) -> Element:
        if type(other) is int or isinstance(other, Fraction):
            return Element(
                self.algebra, self.raw.scale(self.algebra.coefficient(other))
            )
        if isinstance(other, Polynomial):
            # A symbolic scalar makes the result a generic element.
            return self.symbolic() * other  # type: ignore[return-value]
        right = self._operand(other)
        if right is None:
            return NotImplemented
        product = evaluate_multilinear(self.algebra.structure, self.raw, right.raw)
        return Element(self.algebra, product)

    def __rmul__(self, other: object) -> Element:
        if type(other) is int or isinstance(other, Fraction):
            return self * other
        if isinstance(other, Polynomial):
            return self.symbolic() * other  # type: ignore[return-value]
        return NotImplemented

    def symbolic(self) -> Generic:
        """Return the same element with constant polynomials as coordinates."""
        return Generic(
            self.algebra, [Polynomial.constant(value) for value in self.vector]
        )

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
        if isinstance(other, Generic):
            return NotImplemented
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
        self._constants: tuple[tuple[tuple[Fraction, ...], ...], ...] | None = None

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

    def table(self, form: str = "sum", *, rules: bool = False) -> str:
        """Return the multiplication table as aligned text in one print form."""
        render = _form(form)
        cells = [[render(a * b) for b in self.basis] for a in self.basis]
        return _grid("", self.labels, self.labels, cells, rules=rules)

    def _repr_html_(self) -> str:
        cells = [[_as_html(a * b) for b in self.basis] for a in self.basis]
        heads = [_as_html(element) for element in self.basis]
        return _html_table(f"{self.name}: row times column", "", heads, heads, cells)

    def constants(self) -> tuple[tuple[tuple[Fraction, ...], ...], ...]:
        """Return the coordinates of every product of two basis elements."""
        found = self._constants
        if found is None:
            found = tuple(tuple((a * b).vector for b in self.basis) for a in self.basis)
            self._constants = found
        return found

    def generic(self, prefix: str = "a") -> Generic:
        """Return an element with one symbol per coordinate: ``a0, a1, ...``."""
        return Generic(
            self, [Polynomial.symbol(f"{prefix}{index}") for index in range(self.rank)]
        )

    def identity(self, name: str, law: Callable[..., object]) -> LawResult:
        """Decide an identity for every element; see :func:`prove_identity`."""
        return prove_identity(self, name, law)

    def facts(self, *, max_work: int = 5_000_000) -> tuple[tuple[str, object], ...]:
        """Return the exact fingerprint as pairs of a title and a value."""
        found = algebra_fingerprint(
            self.structure, options=AnalysisBounds(max_elimination_work=max_work)
        )
        return (
            ("dimension", found.dimension),
            ("commutative", found.commutative),
            ("associative", found.associative),
            ("unit", found.unit_status),
            ("center dimension", found.center_dimension),
            ("nucleus dimension", found.nucleus_dimension),
            ("derivation dimension", found.derivation_dimension),
        )

    def report(self, *, max_work: int = 5_000_000) -> str:
        """Return the exact fingerprint of the algebra as a short text report.

        ``max_work`` bounds the exact elimination; the octonions need about
        three million units.
        """
        rows = self.facts(max_work=max_work)
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


def comm(x: AnyElement, y: AnyElement) -> AnyElement:
    """Return the commutator ``x*y - y*x``."""
    return x * y - y * x


def anticomm(x: AnyElement, y: AnyElement) -> AnyElement:
    """Return the anticommutator ``x*y + y*x``."""
    return x * y + y * x


def assoc(x: AnyElement, y: AnyElement, z: AnyElement) -> AnyElement:
    """Return the associator ``(x*y)*z - x*(y*z)``."""
    return (x * y) * z - x * (y * z)


# --- a pretty form -----------------------------------------------------------

_SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_FRACTIONS = {
    Fraction(1, 2): "½",
    Fraction(1, 3): "⅓",
    Fraction(2, 3): "⅔",
    Fraction(1, 4): "¼",
    Fraction(3, 4): "¾",
}


def _fancy(x: Element, *, markup: bool) -> str:
    minus = "&minus;" if markup else "\u2212"
    parts: list[tuple[bool, str]] = []
    for label, value in zip(x.algebra.labels, x.vector, strict=True):
        if value == 0:
            continue
        size = abs(value)
        number = _FRACTIONS.get(size, _number(size))
        head = label.rstrip("0123456789")
        tail = label[len(head) :]
        if not head:
            shown = label
        elif markup:
            shown = html.escape(head) + (f"<sub>{tail}</sub>" if tail else "")
        else:
            shown = head + tail.translate(_SUBSCRIPTS)
        if label == _UNIT_LABEL:
            body = number
        elif size == 1:
            body = shown
        else:
            body = f"{number} {shown}"
        parts.append((value < 0, body))
    if not parts:
        return "0"
    text = (minus if parts[0][0] else "") + parts[0][1]
    return text + "".join(
        (f" {minus} " if negative else " + ") + body for negative, body in parts[1:]
    )


@register_form("pretty", "p")
def _as_pretty(x: Element) -> str:
    """Render with subscripts, vulgar fractions and a true minus sign."""
    return _fancy(x, markup=False)


@register_form("wolfram", "w")
def _as_wolfram(x: Element) -> str:
    """Render as a Wolfram Language sum that can be pasted into Mathematica."""
    terms = [
        _number(value) if label == _UNIT_LABEL else f"{_number(value)} {label}"
        for label, value in zip(x.algebra.labels, x.vector, strict=True)
        if value != 0
    ]
    return " + ".join(terms) if terms else "0"


# --- arbitrary finite structures ---------------------------------------------


class LawResult:
    """The outcome of an exact law check, readable as a sentence."""

    __slots__ = ("holds", "law", "text", "witness")

    def __init__(
        self, law: str, holds: bool, text: str, witness: tuple[object, ...] | None
    ) -> None:
        self.law, self.holds, self.text, self.witness = law, holds, text, witness

    def __bool__(self) -> bool:
        return self.holds

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"LawResult({self.law!r}, holds={self.holds})"


class Magma:
    """A finite set with one binary operation, total or partial.

    The operation is a function of two elements.  Returning ``None`` means
    that the product is undefined, which makes the operation partial.
    """

    def __init__(
        self,
        elements: Sequence[object],
        operation: Callable[[object, object], object],
        *,
        name: str = "M",
        symbol: str = "*",
    ) -> None:
        self.elements: Final[tuple[object, ...]] = tuple(elements)
        if not self.elements or len(set(self.elements)) != len(self.elements):
            raise EasyError("table", "the elements must be distinct and at least one")
        self.name, self.symbol = name, symbol
        self._table: dict[tuple[object, object], object] = {}
        for a in self.elements:
            for b in self.elements:
                value = operation(a, b)
                if value is not None and value not in self.elements:
                    raise EasyError(
                        "table",
                        f"{a}{symbol}{b} = {value!r} is not one of the elements",
                    )
                self._table[(a, b)] = value

    def __call__(self, a: object, b: object) -> object:
        """Return the product, or ``None`` where it is undefined."""
        try:
            return self._table[(a, b)]
        except KeyError:
            raise EasyError("symbol", f"{a!r} or {b!r} is not an element") from None

    @property
    def is_total(self) -> bool:
        """Say whether every product is defined."""
        return all(value is not None for value in self._table.values())

    def _show(self, value: object) -> str:
        return "undefined" if value is None else str(value)

    def _nested(self, a: object, b: object, c: object, *, left: bool) -> object:
        inner = self(a, b) if left else self(b, c)
        if inner is None:
            return None
        return self(inner, c) if left else self(a, inner)

    def check(self, law: str) -> LawResult:
        """Check a law on every tuple and report the first counterexample.

        Known laws: ``commutative``, ``associative``, ``idempotent``.
        """
        s = self.symbol
        if law == "commutative":
            for a in self.elements:
                for b in self.elements:
                    if self(a, b) != self(b, a):
                        text = (
                            f"not commutative: {a}{s}{b} = {self._show(self(a, b))}, "
                            f"but {b}{s}{a} = {self._show(self(b, a))}"
                        )
                        return LawResult(law, False, text, (a, b))
            return LawResult(
                law, True, f"commutative: all {len(self._table)} pairs agree", None
            )
        if law == "associative":
            for a in self.elements:
                for b in self.elements:
                    for c in self.elements:
                        left = self._nested(a, b, c, left=True)
                        right = self._nested(a, b, c, left=False)
                        if left != right:
                            text = (
                                f"not associative: ({a}{s}{b}){s}{c} = "
                                f"{self._show(left)}, "
                                f"but {a}{s}({b}{s}{c}) = {self._show(right)}"
                            )
                            return LawResult(law, False, text, (a, b, c))
            count = len(self.elements) ** 3
            return LawResult(law, True, f"associative: all {count} triples agree", None)
        if law == "idempotent":
            for a in self.elements:
                if self(a, a) != a:
                    text = f"not idempotent: {a}{s}{a} = {self._show(self(a, a))}"
                    return LawResult(law, False, text, (a,))
            return LawResult(law, True, "idempotent: x*x = x for every element", None)
        raise EasyError("law", f"unknown law {law!r}")

    def identity(self) -> object:
        """Return the two-sided identity element, or ``None`` if there is none."""
        for e in self.elements:
            if all(self(e, a) == a and self(a, e) == a for a in self.elements):
                return e
        return None

    def table(self, *, rules: bool = False) -> str:
        """Return the operation table as aligned text; a dot marks an undefined cell."""
        names = [str(element) for element in self.elements]
        cells = [
            ["." if self(a, b) is None else str(self(a, b)) for b in self.elements]
            for a in self.elements
        ]
        return _grid(self.symbol, names, names, cells, rules=rules)

    def _repr_html_(self) -> str:
        names = [html.escape(str(element)) for element in self.elements]
        cells = [
            [
                "&middot;" if self(a, b) is None else html.escape(str(self(a, b)))
                for b in self.elements
            ]
            for a in self.elements
        ]
        return _html_table(self.name, html.escape(self.symbol), names, names, cells)

    def report(self) -> str:
        """Return a short summary: size, totality, identity and the three laws."""
        found = self.identity()
        rows = [
            f"{self.name}: {len(self.elements)} elements, "
            + ("total" if self.is_total else "partial")
            + f" operation {self.symbol}",
            "  identity: " + ("none" if found is None else str(found)),
        ]
        rows += [
            f"  {self.check(law)}"
            for law in ("commutative", "associative", "idempotent")
        ]
        return "\n".join(rows)

    def __repr__(self) -> str:
        return f"Magma({self.name!r}, {len(self.elements)} elements)"


def magma(
    elements: Sequence[object],
    operation: Callable[[object, object], object],
    *,
    name: str = "M",
    symbol: str = "*",
) -> Magma:
    """Build a finite set with a binary operation given as a function."""
    return Magma(elements, operation, name=name, symbol=symbol)


# --- text grids and HTML tables ----------------------------------------------


def _grid(
    corner: str,
    heads: Sequence[str],
    sides: Sequence[str],
    cells: Sequence[Sequence[str]],
    *,
    rules: bool,
) -> str:
    texts = [*heads, *(cell for row in cells for cell in row)]
    width = max((len(text) for text in texts), default=0) + 2
    side = max(len(text) for text in (corner, *sides)) + 2
    if not rules:
        lines = [corner.ljust(side) + "".join(text.rjust(width) for text in heads)]
        for text, row in zip(sides, cells, strict=True):
            lines.append(text.ljust(side) + "".join(cell.rjust(width) for cell in row))
        return "\n".join(lines)
    side -= 2
    lines = [
        f" {corner.ljust(side)} │" + "".join(text.rjust(width) for text in heads),
        "─" * (side + 2) + "┼" + "─" * (width * len(heads) + 1),
    ]
    for text, row in zip(sides, cells, strict=True):
        lines.append(
            f" {text.ljust(side)} │" + "".join(cell.rjust(width) for cell in row)
        )
    return "\n".join(lines)


def _html_table(
    caption: str,
    corner: str,
    heads: Sequence[str],
    sides: Sequence[str],
    cells: Sequence[Sequence[str]],
) -> str:
    """Return an HTML table; the cell texts are already HTML."""
    style = 'style="text-align:center;padding:2px 10px"'
    rows = [
        f"<tr><th {style}>{corner}</th>"
        + "".join(f"<th {style}>{text}</th>" for text in heads)
        + "</tr>"
    ]
    for text, row in zip(sides, cells, strict=True):
        rows.append(
            f"<tr><th {style}>{text}</th>"
            + "".join(f"<td {style}>{cell}</td>" for cell in row)
            + "</tr>"
        )
    return (
        f"<table><caption>{html.escape(caption)}</caption>" + "".join(rows) + "</table>"
    )


@register_form("html", "h")
def _as_html(x: Element) -> str:
    """Render as HTML with subscripts and a true minus sign."""
    return _fancy(x, markup=True)


# --- side-by-side comparison -------------------------------------------------


def compare(*algebras: Algebra, max_work: int = 5_000_000) -> str:
    """Return the exact fingerprints of several algebras side by side."""
    if not algebras:
        raise EasyError("structure", "compare needs at least one algebra")
    columns = [algebra.facts(max_work=max_work) for algebra in algebras]
    titles = [title for title, _ in columns[0]]
    table = [
        ["", *(algebra.name for algebra in algebras)],
        *(
            [title, *(str(column[row][1]) for column in columns)]
            for row, title in enumerate(titles)
        ),
    ]
    widths = [max(len(line[index]) for line in table) for index in range(len(table[0]))]
    return "\n".join(
        "  ".join(
            text.ljust(width) for text, width in zip(line, widths, strict=True)
        ).rstrip()
        for line in table
    )


# --- structures with several sets and operations of any arity ----------------


class _Operation:
    __slots__ = ("inputs", "name", "output", "table")

    def __init__(
        self,
        name: str,
        inputs: tuple[str, ...],
        output: str,
        table: dict[tuple[object, ...], object],
    ) -> None:
        self.name, self.inputs, self.output, self.table = name, inputs, output, table

    @property
    def signature(self) -> str:
        return (" ".join(self.inputs) + " -> " + self.output).strip()

    @property
    def is_total(self) -> bool:
        return all(value is not None for value in self.table.values())


def _parse_signature(name: str, text: str) -> tuple[tuple[str, ...], str]:
    left, arrow, right = text.partition("->")
    output = right.split()
    if not arrow or len(output) != 1:
        raise EasyError(
            "signature",
            f"the signature of {name} must read 'sort sort -> sort', not {text!r}",
        )
    return tuple(left.split()), output[0]


class FiniteStructure:
    """Finite sets with operations of any arity, total or partial.

    Each operation has a signature such as ``"scalar point -> point"`` and a
    rule: a function of that many arguments, or a mapping from argument
    tuples to values.  ``None``, or a missing entry, means undefined.
    """

    def __init__(
        self,
        sorts: Mapping[str, Sequence[object]],
        operations: Mapping[
            str, tuple[str, Callable[..., object] | Mapping[tuple[object, ...], object]]
        ],
        *,
        name: str = "S",
    ) -> None:
        self.name = name
        self.sorts: Final[dict[str, tuple[object, ...]]] = {
            sort: tuple(members) for sort, members in sorts.items()
        }
        for sort, members in self.sorts.items():
            if not members or len(set(members)) != len(members):
                raise EasyError(
                    "table", f"the elements of {sort} must be distinct and at least one"
                )
        self.operations: Final[dict[str, _Operation]] = {}
        for label, (signature, rule) in operations.items():
            inputs, output = _parse_signature(label, signature)
            for sort in (*inputs, output):
                if sort not in self.sorts:
                    raise EasyError(
                        "signature", f"{label} names the unknown sort {sort!r}"
                    )
            table: dict[tuple[object, ...], object] = {}
            for arguments in itertools.product(*(self.sorts[s] for s in inputs)):
                value = rule(*arguments) if callable(rule) else rule.get(arguments)
                if value is not None and value not in self.sorts[output]:
                    shown = ", ".join(str(argument) for argument in arguments)
                    raise EasyError(
                        "table",
                        f"{label}({shown}) = {value!r} is not an element of {output}",
                    )
                table[arguments] = value
            self.operations[label] = _Operation(label, inputs, output, table)

    def apply(self, operation: str, *arguments: object) -> object:
        """Apply an operation; an undefined argument gives an undefined result."""
        found = self.operations.get(operation)
        if found is None:
            raise EasyError("symbol", f"{self.name} has no operation {operation!r}")
        if len(arguments) != len(found.inputs):
            raise EasyError(
                "symbol", f"{operation} takes {len(found.inputs)} arguments"
            )
        if any(argument is None for argument in arguments):
            return None
        try:
            return found.table[arguments]
        except KeyError:
            raise EasyError(
                "symbol",
                f"{operation} expects arguments of sorts {' '.join(found.inputs)}",
            ) from None

    def __getattr__(self, operation: str) -> Callable[..., object]:
        operations = self.__dict__.get("operations")
        if operations is None or operation not in operations:
            raise AttributeError(operation)
        return functools.partial(self.apply, operation)

    def check(
        self,
        name: str,
        law: Callable[..., object],
        over: str | Sequence[str] | None = None,
    ) -> LawResult:
        """Check a law on every assignment and report the first counterexample.

        ``law`` is a function of the variables.  It returns the two sides of
        an equation as a pair, or a truth value.  ``over`` names the sort of
        each variable, as ``"scalar scalar point"``; it may be left out when
        the structure has one sort.
        """
        variables = tuple(inspect.signature(law).parameters)
        if over is None:
            if len(self.sorts) != 1:
                raise EasyError("law", "say which sort each variable ranges over")
            chosen = (next(iter(self.sorts)),) * len(variables)
        else:
            chosen = tuple(over.split()) if isinstance(over, str) else tuple(over)
        if len(chosen) != len(variables) or any(s not in self.sorts for s in chosen):
            raise EasyError(
                "law",
                f"{name} needs one known sort for each of {len(variables)} variables",
            )
        count = 0
        for values in itertools.product(*(self.sorts[sort] for sort in chosen)):
            count += 1
            outcome = law(*values)
            if type(outcome) is tuple and len(outcome) == 2:
                left, right = outcome
                if left == right:
                    continue
                detail = f": left side = {_shown(left)}, right side = {_shown(right)}"
            elif type(outcome) is bool:
                if outcome:
                    continue
                detail = ""
            else:
                raise EasyError("law", "a law returns a pair of sides or True/False")
            where = ", ".join(
                f"{variable} = {value}"
                for variable, value in zip(variables, values, strict=True)
            )
            at = f" at {where}" if where else ""
            return LawResult(name, False, f"{name} fails{at}{detail}", values)
        plural = "assignment" if count == 1 else "assignments"
        return LawResult(name, True, f"{name}: holds for all {count} {plural}", None)

    def table(self, operation: str, *, rules: bool = False) -> str:
        """Return the table of a binary operation; a dot marks an undefined cell."""
        found = self.operations.get(operation)
        if found is None or len(found.inputs) != 2:
            raise EasyError("table", f"{operation!r} is not a binary operation")
        rows, columns = (self.sorts[sort] for sort in found.inputs)
        cells = [
            [_shown(found.table[(a, b)], undefined=".") for b in columns] for a in rows
        ]
        return _grid(
            operation,
            [str(b) for b in columns],
            [str(a) for a in rows],
            cells,
            rules=rules,
        )

    def report(self) -> str:
        """Return the sorts and the operations with their signatures."""

        def count(number: int, noun: str) -> str:
            return f"{number} {noun}" + ("" if number == 1 else "s")

        lines = [
            f"{self.name}: {count(len(self.sorts), 'sort')}, "
            f"{count(len(self.operations), 'operation')}"
        ]
        lines += [
            f"  {sort}: " + ", ".join(str(member) for member in members)
            for sort, members in self.sorts.items()
        ]
        lines += [
            f"  {found.name}: {found.signature}, "
            + ("total" if found.is_total else "partial")
            for found in self.operations.values()
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"FiniteStructure({self.name!r}, {len(self.sorts)} sorts, "
            f"{len(self.operations)} operations)"
        )


def _shown(value: object, *, undefined: str = "undefined") -> str:
    return undefined if value is None else str(value)


def structure(
    sorts: Mapping[str, Sequence[object]],
    operations: Mapping[
        str, tuple[str, Callable[..., object] | Mapping[tuple[object, ...], object]]
    ],
    *,
    name: str = "S",
) -> FiniteStructure:
    """Build finite sets with operations of any arity given by functions or tables."""
    return FiniteStructure(sorts, operations, name=name)


# --- symbolic coefficients ---------------------------------------------------


class Generic:
    """An element whose coordinates are polynomials in named symbols.

    Arithmetic with generic elements is exact polynomial arithmetic, so an
    expression that comes out zero is an identity for every element.
    """

    __slots__ = ("algebra", "coordinates")
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, algebra: Algebra, coordinates: Sequence[Polynomial]) -> None:
        if len(coordinates) != algebra.rank:
            raise EasyError(
                "coordinates",
                f"{algebra.name} needs exactly {algebra.rank} coordinates",
            )
        self.algebra = algebra
        self.coordinates: Final[tuple[Polynomial, ...]] = tuple(coordinates)

    def _lift(self, other: object) -> Generic | None:
        if isinstance(other, Generic | Element):
            if other.algebra is not self.algebra:
                raise EasyError(
                    "parent",
                    f"cannot combine elements of {self.algebra.name} "
                    f"and {other.algebra.name}",
                )
            return other if isinstance(other, Generic) else other.symbolic()
        scalar = Polynomial._coerce(other)
        if scalar is None:
            return None
        unit = self.algebra.unit_index
        if unit is None:
            raise EasyError(
                "unit", f"{self.algebra.name} declares no unit to carry a scalar"
            )
        return Generic(
            self.algebra,
            [
                scalar if index == unit else Polynomial()
                for index in range(len(self.coordinates))
            ],
        )

    def __add__(self, other: object) -> Generic:
        right = self._lift(other)
        if right is None:
            return NotImplemented
        return Generic(
            self.algebra,
            [a + b for a, b in zip(self.coordinates, right.coordinates, strict=True)],
        )

    __radd__ = __add__

    def __neg__(self) -> Generic:
        return Generic(self.algebra, [-value for value in self.coordinates])

    def __sub__(self, other: object) -> Generic:
        right = self._lift(other)
        if right is None:
            return NotImplemented
        return self + (-right)

    def __rsub__(self, other: object) -> Generic:
        left = self._lift(other)
        if left is None:
            return NotImplemented
        return left + (-self)

    def _product(self, right: Generic) -> Generic:
        constants = self.algebra.constants()
        total: list[Polynomial] = [Polynomial() for _ in self.coordinates]
        for i, a in enumerate(self.coordinates):
            if not a:
                continue
            for j, b in enumerate(right.coordinates):
                if not b:
                    continue
                term = a * b
                for k, value in enumerate(constants[i][j]):
                    if value != 0:
                        total[k] = total[k] + term * value
        return Generic(self.algebra, total)

    def __mul__(self, other: object) -> Generic:
        scalar = Polynomial._coerce(other)
        if scalar is not None:
            return Generic(self.algebra, [value * scalar for value in self.coordinates])
        right = self._lift(other)
        if right is None:
            return NotImplemented
        return self._product(right)

    def __rmul__(self, other: object) -> Generic:
        scalar = Polynomial._coerce(other)
        if scalar is not None:
            return self * scalar
        left = self._lift(other)
        if left is None:
            return NotImplemented
        return left._product(self)

    def __truediv__(self, other: object) -> Generic:
        if not (type(other) is int or isinstance(other, Fraction)) or other == 0:
            return NotImplemented
        return self * (Fraction(1) / other)

    def __pow__(self, exponent: object) -> Generic:
        if type(exponent) is not int or exponent < 1:
            raise EasyError("power", "a power needs an integer exponent of at least 1")
        result = self
        for _ in range(exponent - 1):
            result = result * self
        return result

    def __eq__(self, other: object) -> bool:
        if (type(other) is int or isinstance(other, Fraction)) and other == 0:
            return not self
        if isinstance(other, Element | Generic) and other.algebra is not self.algebra:
            return False
        right = (
            self._lift(other)
            if self.algebra.unit_index is not None
            or isinstance(other, Element | Generic)
            else None
        )
        if right is None:
            return NotImplemented
        return self.coordinates == right.coordinates

    def __bool__(self) -> bool:
        return any(self.coordinates)

    def conj(self) -> Generic:
        """Return the conjugate under the algebra's declared involution."""
        signs = self.algebra.conjugation_signs
        if signs is None:
            raise EasyError(
                "involution", f"{self.algebra.name} declares no conjugation"
            )
        return Generic(
            self.algebra,
            [value * sign for sign, value in zip(signs, self.coordinates, strict=True)],
        )

    def norm(self) -> Polynomial:
        """Return the quadratic norm ``x * conj(x)`` as a polynomial."""
        value = self * self.conj()
        unit = self.algebra.unit_index
        if unit is None or any(
            coordinate
            for index, coordinate in enumerate(value.coordinates)
            if index != unit
        ):
            raise EasyError("norm", "x * conj(x) is not a multiple of the unit")
        return value.coordinates[unit]

    @property
    def coefficients(self) -> dict[str, Polynomial]:
        """Return the nonzero coordinates keyed by basis label."""
        return {
            label: value
            for label, value in zip(self.algebra.labels, self.coordinates, strict=True)
            if value
        }

    def subs(
        self, values: Mapping[str, object] | None = None, **named: object
    ) -> Generic:
        """Replace symbols by numbers or polynomials in every coordinate."""
        return Generic(
            self.algebra, [value.subs(values, **named) for value in self.coordinates]
        )

    def element(self) -> Element:
        """Return an ordinary element once no symbol is left."""
        if not all(value.is_constant for value in self.coordinates):
            raise EasyError("symbolic", "the element still contains symbols")
        return self.algebra(*(value.as_fraction() for value in self.coordinates))

    def __str__(self) -> str:
        parts: list[tuple[bool, str]] = []
        for label, value in self.coefficients.items():
            pieces = value._parts()
            if len(pieces) == 1:
                negative, body = pieces[0]
                if label == _UNIT_LABEL:
                    parts.append((negative, body))
                else:
                    parts.append(
                        (negative, label if body == "1" else f"{body}*{label}")
                    )
            elif label == _UNIT_LABEL:
                parts.append((False, str(value) if not parts else f"({value})"))
            else:
                parts.append((False, f"({value})*{label}"))
        if not parts:
            return "0"
        text = ("-" if parts[0][0] else "") + parts[0][1]
        return text + "".join(
            (" - " if negative else " + ") + body for negative, body in parts[1:]
        )

    def __repr__(self) -> str:
        return f"{self.algebra.name}({self})"


def _is_zero(value: object) -> bool:
    if isinstance(value, Element | Generic):
        return not value
    return value == 0


def _difference(outcome: object) -> object:
    if type(outcome) is tuple and len(outcome) == 2:
        return outcome[0] - outcome[1]
    return outcome


def prove_identity(
    algebra: Algebra, name: str, law: Callable[..., object]
) -> LawResult:
    """Decide a polynomial identity for all elements at once.

    ``law`` is a function of elements.  It returns an expression that should
    vanish, or the two sides of an equation as a pair.  It is evaluated once
    on generic elements; if the result is not zero, basis elements are tried
    to find a concrete counterexample.
    """
    variables = tuple(inspect.signature(law).parameters)
    if len(variables) > len(_PREFIXES):
        raise EasyError("law", f"at most {len(_PREFIXES)} variables")
    generic = [algebra.generic(_PREFIXES[index]) for index in range(len(variables))]
    difference = _difference(law(*generic))
    if _is_zero(difference):
        count = algebra.rank * len(variables)
        return LawResult(
            name,
            True,
            f"{name}: an identity in {count} symbols, so it holds for every element",
            None,
        )
    for choice in itertools.product(algebra.basis, repeat=len(variables)):
        value = _difference(law(*choice))
        if not _is_zero(value):
            where = ", ".join(
                f"{variable} = {element}"
                for variable, element in zip(variables, choice, strict=True)
            )
            text = f"{name} fails at {where}: the difference is {value}"
            return LawResult(name, False, text, choice)
    return LawResult(name, False, f"{name} fails: the difference is {difference}", None)


_PREFIXES: Final = "abcdfghk"


# --- kernel validation reports in words --------------------------------------


def _term_text(term: Term) -> str:
    if term.variable_value is not None:
        return str(term.variable_value.name)
    assert term.symbol is not None
    inner = ", ".join(_term_text(argument) for argument in term.arguments)
    return f"{term.symbol.name}({inner})" if term.arguments else str(term.symbol.name)


def _outcome_text(result: EvaluationResult | None) -> str:
    outcome = None if result is None else result.outcome
    if isinstance(outcome, Defined):
        return str(outcome.value)
    return "undefined" if outcome is None else type(outcome).__name__.lower()


def explain(report: ValidationReport) -> str:
    """Say what a kernel law validation found, in the structure's own symbols."""
    law = report.law
    left = _term_text(law.conclusion.left)
    right = _term_text(law.conclusion.right)
    if isinstance(report, Proved):
        return (
            f"{law.name} holds: {left} = {right} "
            f"for all {report.expected_assignments} assignments"
        )
    witness = report.witness
    if not isinstance(report, Disproved) or witness is None:
        return (
            f"{law.name} is undecided: {report.undecidable_assignments} of "
            f"{report.expected_assignments} assignments could not be evaluated"
        )
    where = ", ".join(
        f"{variable.name} = {_member(report, variable.sort, index)}"
        for variable, index in zip(
            law.variables, witness.substitution_indices, strict=True
        )
    )
    return (
        f"{law.name} fails at {where}: {left} = {_outcome_text(witness.left)}, "
        f"but {right} = {_outcome_text(witness.right)}"
    )


def _member(report: ValidationReport, sort: object, index: int) -> object:
    for declared, carrier in zip(
        report.structure.signature.sorts, report.structure.carriers, strict=True
    ):
        if declared == sort:
            return carrier.items[index]
    return f"member {index}"


# --- the catalog -------------------------------------------------------------

_CATALOG: Final = (
    ("aa.quaternions(), aa.octonions()", "H and O in our conventions"),
    ("aa.split_quaternions(), aa.split_octonions()", "their split forms"),
    ("aa.algebra(labels, table)", "any algebra from a table of basis products"),
    ("aa.magma(elements, function)", "any finite set with a binary operation"),
    ("aa.structure(sorts, operations)", "several sets, operations of any arity"),
    ("A.generic(), A.identity(name, law)", "symbolic elements; identities for all x"),
    ("aa.compare(A, B)", "exact fingerprints side by side"),
    ("ca.reals(), ca.complexes(), ca.quaternions()", "the Cayley-Dickson chain"),
    ("ca.octonions(), ca.sedenions()", "... through dimension 16"),
    ("ca.split_complexes(), ca.split_octonions()", "split forms by doubling"),
    ("ca.cayley_dickson(A), ca.tensor(A, B)", "doubling and tensor products"),
    ("am.matrix(A, rows)", "matrices over any algebra"),
    ("al.so(p, q), al.su(p, q), al.sl(n), al.sp(n)", "classical Lie algebras"),
    ("al.root_system(family, rank)", "root systems of types A to G"),
    ("ls.identify(g), ls.root_decomposition(g)", "which Lie algebra is this"),
    ("ac.clifford(p, q, r), ac.grassmann(n)", "Clifford and exterior algebras"),
    ("ac.gamma_matrices(p, q)", "exact gamma matrices in any signature"),
    ("aj.hermitian(A, n)", "Hermitian Jordan algebras; J3(O) is the Albert algebra"),
    ("aj.tensor_jordan(A, n, B)", "the carrier A tensor J_n(B)"),
    ("ml.su(n, A), ml.sl(n, A)", "matrix Lie algebras over R, C, H, O"),
)


def catalog() -> str:
    """Return the built-in structures and constructors as an aligned list.

    The prefixes are the conventional imports: ``aa`` for this module,
    ``ca`` for ``anyalgebra.composition``, ``am`` for ``anyalgebra.matrices``,
    ``al`` for ``anyalgebra.lie``, ``ac`` for ``anyalgebra.clifford``,
    ``aj`` for ``anyalgebra.jordan``, ``ml`` for ``anyalgebra.matrix_lie`` and
    ``ls`` for ``anyalgebra.lie_structure``.
    """
    width = max(len(call) for call, _ in _CATALOG)
    return "\n".join(f"{call.ljust(width)}  {text}" for call, text in _CATALOG)
