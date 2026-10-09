# Composition algebras and involutions: `anyalgebra.composition`

Status: implemented in the v0.1.2 patch. It owns the work package
`wp.composition-operations` of the [parity contract](../legacy/algmul-parity.md).

```python
import anyalgebra.composition as ca

O, S = ca.octonions(), ca.sedenions()
print(ca.is_alternative(O), ca.is_composition(O))   # True True
print(ca.is_alternative(S), ca.is_composition(S))   # False False
```

Every function returns or accepts the handles of the
[convenience layer](easy.md), so elements multiply with `*` and print in
every form.

## The doubling chain

`ca.cayley_dickson(A, gamma=-1, name=...)` doubles an algebra with involution:

```text
(a, b)(c, d) = (a c + gamma d* b,  d a + b c*)        (a, b)* = (a*, -b)
```

The new basis is the old one followed by the old one times a new unit `l`
with `l*l = gamma`, labeled `1, e1, e2, ...`.

| Call | Algebra | Dimension | Commutative | Associative | Alternative | Composition |
|---|---|---:|---|---|---|---|
| `ca.reals()` | R | 1 | yes | yes | yes | yes |
| `ca.complexes()`, `ca.split_complexes()` | C, Cs | 2 | yes | yes | yes | yes |
| `ca.quaternions()`, `ca.split_quaternions()` | H, Hs | 4 | no | yes | yes | yes |
| `ca.octonions()`, `ca.split_octonions()` | O, Os | 8 | no | no | yes | yes |
| `ca.sedenions()` | S | 16 | no | no | no | no |

The table is not asserted by the constructors. The tests compute each entry
exactly and compare it with the textbook statement. They also check that the
division forms have a positive norm on every basis element, that each split
form has signature half and half with an isotropic element, that the
derivation algebras of H and O have dimensions 3 and 14, and that the
sedenions contain zero divisors.

These algebras are isomorphic to the pinned `algmul.*.v1` fixtures behind
`aa.quaternions()` and `aa.octonions()`, but the basis and signs differ. Use
one convention within a calculation.

## Involutions, norms, and division

An involution is one sign per basis element. That covers the standard
conjugation of every algebra above and the factor-wise conjugations of
tensor products. A general matrix involution is not part of this patch.

| Call | Meaning |
|---|---|
| `x.conj()`, `x.norm()`, `x.inv()` | conjugate, quadratic norm, inverse |
| `ca.conjugate(x, signs)` | apply any sign involution |
| `ca.inner(x, y)` | the symmetric form `(x y* + y x*) / 2` |
| `ca.left_divide(a, b)`, `ca.right_divide(b, a)` | solve `a x = b` and `x a = b` |
| `ca.is_involution(A, signs=None)` | exact test of `(x y)* = y* x*` |
| `ca.is_automorphism(A, images)` | exact test that basis images define an automorphism |
| `ca.is_commutative`, `ca.is_associative`, `ca.is_alternative`, `ca.is_composition` | exact law tests |

Each test is exact: the law is multilinear or is replaced by its
linearization, and the basis grid then decides it over the rationals.

## Tensor products

`ca.tensor(A, B)` is the tensor product with factor-wise multiplication. The
basis runs with the right factor fastest. Labels read `a.b` with unit factors
left out, and where the factors share a label the right factor's `e1, e2, ...`
become `f1, f2, ...`.

```python
C, H = ca.complexes(), ca.quaternions()
CH = ca.tensor(C, H)
print(CH.labels)      # ('1', 'f1', 'f2', 'f3', 'e1', 'e1.f1', 'e1.f2', 'e1.f3')
first = ca.factor_conjugation(C, H, first=True)
print(ca.is_involution(CH, first), ca.is_involution(CH))   # False True
```

The declared conjugation of a tensor product conjugates both factors.
`ca.factor_conjugation(A, B, first=..., second=...)` gives the one-sided
ones. Conjugating a commutative factor alone is an automorphism, so it is an
anti-automorphism only together with the other factor's conjugation; the test
above shows this for `C tensor H`.

## Coming from AlgMul

| AlgMul | Here |
|---|---|
| `CConj`, `HConj`, `OConj`, `Conj` | `x.conj()` |
| `TConj1`, `TConj2`, `TConjAll`, `CHConj`, `COConj`, `HOConj` | `ca.conjugate(x, ca.factor_conjugation(A, B, ...))` |
| `Norm2`, `Mag` | `x.norm()` (the squared magnitude; no square root is taken) |
| `Inv`, `DvdL`, `DvdR` | `x.inv()`, `ca.left_divide`, `ca.right_divide` |
| `InnerProd` | `ca.inner(x, y)` |
| `TMul`, `MakeTMul` | `ca.tensor(A, B)` and `*` |
| `IsAutomorphism` | `ca.is_automorphism(A, images)` |
| `G2Der` | `O.report()` for the dimension; `derivation_algebra` for the derivations |

Not covered yet: three-factor conjugations such as `CHOConj` as named calls
(nest `ca.tensor`), the left and right conjugate-norm variants
(`LConjNorm2`, `RConjNorm2`, ...), and normalization by a square root, which
is not rational.
