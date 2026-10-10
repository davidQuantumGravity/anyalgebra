# The convenience layer: `anyalgebra.easy`

Status: implemented in the v0.1.1 patch. It adds no mathematics. It wraps the
kernel so that an everyday calculation is short, and it changes nothing in the
kernel: `x.raw` is the kernel element and `A.structure` the kernel structure.

```python
import anyalgebra.easy as aa
```

## Algebras

| Call | Result |
|---|---|
| `aa.quaternions()`, `aa.split_quaternions()` | `H`, `Hs` in the pinned `algmul.*.v1` conventions |
| `aa.octonions()`, `aa.split_octonions()` | `O`, `Os` |
| `aa.algebra(labels, table, name="A", unit="1")` | an algebra from a table of basis products |
| `aa.wrap(structure)` | a handle on any binary kernel algebra |
| `aa.over_rationals(structure)` | the same kernel algebra with rational coefficients |

Every handle uses exact rational coefficients, so the center, nucleus,
derivation, and fingerprint solvers apply directly.

A table maps a pair of labels to a product: a label (`"z"`), a signed label
(`"-z"`), `0`, or a mapping from labels to coefficients. A pair that is left
out has product zero.

```python
cross = aa.algebra(
    ("x", "y", "z"),
    {("x", "y"): "z", ("y", "x"): "-z", ("y", "z"): "x",
     ("z", "y"): "-x", ("z", "x"): "y", ("x", "z"): "-y"},
    name="cross", unit=None,
)
print(cross.report())
```

## Symbols

Python has no automatic symbols, so a handle offers four ways to name its
basis elements.

| Way | Example | Use |
|---|---|---|
| unpacking | `one, i, j, k = H.basis` | scripts and functions |
| attribute | `O.e3`, `H.one` | one-off access |
| index | `O[3]`, `O["e3"]` | loops |
| injection | `O.inject()` | interactive sessions |

`O.inject()` defines `e1` to `e7` and `one` in the calling module. Inside a
function, pass the dictionary to fill: `O.inject(namespace)`. The unit label
`"1"` is not a Python name, so the unit is available as `one`.

Symbolic coefficients, as in a generic `a0 + a1*e1 + a2*e2`, are described
under [Symbolic elements](#symbolic-elements) below.

## Operators

| Expression | Meaning |
|---|---|
| `x * y` | the algebra product |
| `x + y`, `x - y`, `-x` | module arithmetic |
| `3 * x`, `x * 3`, `x / 2` | exact scalar action; scalars are `int` or `Fraction` |
| `x ** n` | the left-nested power `((x*x)*x)...`, `n >= 1` |
| `x + 1`, `x == 1` | the scalar as a multiple of the unit, only if the algebra declares one |
| `x == y`, `bool(x)` | exact equality; nonzero test |
| `aa.comm(x, y)`, `aa.anticomm(x, y)` | `x*y - y*x`, `x*y + y*x` |
| `aa.assoc(x, y, z)` | `(x*y)*z - x*(y*z)` |
| `x.conj()`, `x.norm()`, `x.inv()` | conjugate, quadratic norm, inverse, where an involution is declared |

Python evaluates `a * b * c` from left to right, so it means `(a * b) * c`.
In a nonassociative algebra, write the parentheses.

Elements of two different handles never combine, even when the handles were
built the same way. A float is not a scalar.

## Print forms

One element can be printed in several forms without changing it.

| Form | Short | `x = -1/2*j + 3/2*k` in `H` |
|---|---|---|
| `sum` | `s` | `-1/2*j + 3/2*k` |
| `vector` | `v` | `[0, 0, -1/2, 3/2]` |
| `sparse` | `d` | `{j: -1/2, k: 3/2}` |
| `latex` | `l` | `-\frac{1}{2}\,j + \frac{3}{2}\,k` |

A form is chosen in four places, the first that applies winning:

1. per call: `format(x, "v")` or `f"{x:v}"`;
2. per block: `with aa.display("vector"): ...`;
3. per handle: `H.display = "vector"`;
4. per session: `aa.set_display("vector")`. The initial default is `sum`.

`repr(x)` is always `H(-1/2*j + 3/2*k)`, and notebooks render the LaTeX form.

A further form is one function:

```python
@aa.register_form("words")
def words(x):
    return " plus ".join(f"{value} of {label}" for label, value in x.coefficients.items())
```

`x.vector` returns the coordinates as a tuple of `Fraction`, and
`x.coefficients` the nonzero ones keyed by label.

## More forms

| Form | Short | The same `x` |
|---|---|---|
| `pretty` | `p` | subscripts, vulgar fractions and a true minus sign |
| `wolfram` | `w` | `-1/2 j + 3/2 k`, which can be pasted into Mathematica |
| `html` | `h` | subscripts as `<sub>` tags and `&minus;` for the sign |

`H.table(rules=True)` draws the table with rules. In a notebook an element
displays as LaTeX and an algebra or a magma as an HTML table, with no call.

## Any finite structure

`aa.magma(elements, function)` builds a finite set with a binary operation.
The function may return `None` where the operation is undefined.

```python
game = aa.magma(
    ("rock", "paper", "scissors"),
    lambda a, b: a if a == b or (a, b) in {("paper", "rock"), ("scissors", "paper"), ("rock", "scissors")} else b,
    name="RPS",
)
print(game.check("associative"))
```

`game.check(law)` returns a result that is truthy when the law holds and
prints as a sentence, with the first counterexample when it fails. The laws
are `commutative`, `associative` and `idempotent`. `game.table()` prints the
operation table, `game.identity()` finds a two-sided identity, and
`game.report()` gathers all of it.

## Several sets and any arity

`aa.structure(sorts, operations)` builds finite sets with operations of any
number of arguments. Each operation is a pair: a signature such as
`"turn corner -> corner"`, and a rule. The rule is a function, or a mapping
from argument tuples to values. `None`, or a missing entry, means undefined,
and an undefined argument gives an undefined result.

```python
corners = "abc"
triangle = aa.structure(
    {"turn": (0, 1, 2), "corner": tuple(corners)},
    {
        "add": ("turn turn -> turn", lambda m, n: (m + n) % 3),
        "move": ("turn corner -> corner", lambda n, p: corners[(corners.index(p) + n) % 3]),
        "flip": ("corner -> corner", {("a",): "a", ("b",): "c", ("c",): "b"}),
        "zero": ("-> turn", lambda: 0),
    },
    name="triangle",
)
print(triangle.move(1, "a"))
print(triangle.check(
    "action",
    lambda m, n, p: (triangle.move(m, triangle.move(n, p)), triangle.move(triangle.add(m, n), p)),
    "turn turn corner",
))
```

Every operation is a method of the structure. `check(name, law, over)` runs
the law on every assignment. The law is a function of the variables that
returns the two sides of an equation as a pair, or a truth value; `over`
names the sort of each variable and may be left out when there is one sort.
The result is truthy when the law holds and prints as a sentence with the
first counterexample otherwise. `table(operation)` prints a binary operation
and `report()` lists the sorts and the signatures.

## Symbolic elements

`A.generic("a")` is an element whose coordinates are the symbols `a0`, `a1`,
and so on. Arithmetic on such elements is exact polynomial arithmetic, so an
expression that comes out zero is an identity for every element, over any
commutative ring that contains the rationals.

```python
O = aa.octonions()
x, y = O.generic("a"), O.generic("b")
print(aa.assoc(x, x, y) == 0)
print((x * y).norm() == x.norm() * y.norm())
print(O.identity("associative", lambda x, y, z: aa.assoc(x, y, z)))
```

`A.identity(name, law)` wraps this. The law is a function of elements that
returns an expression expected to vanish, or the two sides of an equation.
It is evaluated once on generic elements. If the result is not zero, basis
elements are tried, so that the failure is reported on named elements.

A generic element supports `+`, `-`, `*`, powers, `conj()`, `norm()`,
`coefficients`, `subs(a0=1, ...)`, and `element()` once no symbol is left.
The polynomials are in `anyalgebra.polynomials`, with `symbols("s t")` for
free scalars. They add, multiply, substitute and print; they do not factor.

## Comparison

`aa.compare(A, B, ...)` prints the exact fingerprints side by side, one
column per algebra. `A.facts()` returns the same values as pairs.

## Kernel validation in words

`str()` of a kernel law validation report states the outcome and the
location of a counterexample by index. `aa.explain(report)` names the
elements: `commutative fails at x = 0, y = 1: op(x, y) = 0, but op(y, x) = 1`.

## The catalog

`aa.catalog()` returns every built-in structure and constructor across the
modules, one per line.

## Kernel values

`str()` of a kernel element or scalar is now readable, for example `2*e7`
and `3/2`. `repr()` is unchanged and remains the stable record.

## Tables and reports

`H.table()` returns the multiplication table as aligned text, and
`H.table("vector")` the same table in another form. `H.report()` returns the
exact fingerprint: dimension, commutativity, associativity, unit, and the
dimensions of the center, nucleus, and derivation algebra.

## Errors

A request with no unambiguous meaning raises `aa.EasyError`, a `TypeError`
with a `code`: `parent`, `unit`, `form`, `symbol`, `coordinates`, `division`,
`power`, `involution`, `norm`, `table`, `structure`, `signature`, `law`, or
`symbolic`.

## Coming from AlgMul

Only the legacy names that already have a replacement are listed. The other
families are scheduled as work packages in the
[parity contract](../legacy/algmul-parity.md).

| AlgMul | Here |
|---|---|
| `HMul[x, y]`, `OMul[x, y]`, `HsMul`, `OsMul` | `x * y` |
| `MakeAlg`, `MakeChar` | `aa.algebra(labels, table)` |
| `ListToSym`, `SymToList` | `H(1, 2, 0, -3)` and `x.vector`; `str(x)` |
| `MulTable` | `H.table()` |
| `Comm`, `AntiComm`, `Assoc` | `aa.comm`, `aa.anticomm`, `aa.assoc` |
| `Conj`, `Norm2`, `Inv` | `x.conj()`, `x.norm()`, `x.inv()` for `H`, `Hs`, `O`, `Os` |
| `G2Der` | the derivation dimension in `aa.octonions().report()`; the derivations themselves from `anyalgebra.analysis.fingerprint.derivation_algebra` |
| `AlgebraReport` | `H.report()` |
| generic elements with symbolic coefficients | `H.generic("a")`, `H.identity(name, law)` |
| `MatrixToLieStruct` | `anyalgebra.linear.recovery.recover_structure_constants` |

## Kernel additions in the same patch

`anyalgebra.linear.matrix.Matrix` gained `subtract`, `negate`, and `scale`,
with the operators `+`, `-`, unary `-`, and `@`. A matrix commutator is now
`a @ b - b @ a`.
