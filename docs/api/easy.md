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

Symbolic coefficients, as in a generic `a1*e1 + a2*e2`, need a polynomial
coefficient domain and are not part of this layer.

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
@aa.register_form("wolfram", "w")
def wolfram(x):
    return " + ".join(f"{value} {label}" for label, value in x.coefficients.items())
```

`x.vector` returns the coordinates as a tuple of `Fraction`, and
`x.coefficients` the nonzero ones keyed by label.

## Tables and reports

`H.table()` returns the multiplication table as aligned text, and
`H.table("vector")` the same table in another form. `H.report()` returns the
exact fingerprint: dimension, commutativity, associativity, unit, and the
dimensions of the center, nucleus, and derivation algebra.

## Errors

A request with no unambiguous meaning raises `aa.EasyError`, a `TypeError`
with a `code`: `parent`, `unit`, `form`, `symbol`, `coordinates`, `division`,
`power`, `involution`, `norm`, `table`, or `structure`.

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
| `MatrixToLieStruct` | `anyalgebra.linear.recovery.recover_structure_constants` |

## Kernel additions in the same patch

`anyalgebra.linear.matrix.Matrix` gained `subtract`, `negate`, and `scale`,
with the operators `+`, `-`, unary `-`, and `@`. A matrix commutator is now
`a @ b - b @ a`.
