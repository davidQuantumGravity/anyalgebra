# Algebra families: matrices, Lie, Clifford, Jordan

Status: implemented in the v0.1.3 to v0.1.7 patches. These modules are built
on the [convenience layer](easy.md) and on
[composition algebras](composition.md). Every claim below is checked by an
exact computation in the test suite against a textbook statement, never
against legacy output. The interfaces may still change before 1.0.

```python
import anyalgebra.composition as ca
import anyalgebra.matrices as am
import anyalgebra.lie as al
import anyalgebra.clifford as ac
import anyalgebra.jordan as aj
import anyalgebra.matrix_lie as ml
```

## Matrices over an algebra (`anyalgebra.matrices`, v0.1.3)

The entries may be noncommutative or nonassociative, so the order of every
entry product is fixed: `(X @ Y)[i][j]` is the sum over `k` of
`X[i][k] * Y[k][j]`.

```python
H = ca.quaternions()
X = am.matrix(H, [[1, H.e1], [-H.e1, 2]])
print(X.is_hermitian(), X.trace())      # True 3
print(X)                                # [  1  e1]
                                        # [-e1   2]
```

| Call | Meaning |
|---|---|
| `am.matrix`, `am.zeros`, `am.identity`, `am.unit` | constructors |
| `X + Y`, `X - Y`, `X @ Y`, `2 * X`, `a * X`, `X * a` | arithmetic; `a` may be an element |
| `X.transpose()`, `X.conj()`, `X.dagger()`, `X.trace()` | optional `signs` select another involution |
| `X.is_hermitian()`, `X.is_antihermitian()` | exact predicates |
| `am.hermitian_basis(A, n)`, `am.antihermitian_basis(A, n)` | rational bases |
| `am.comm`, `am.anticomm`, `am.jordan`, `am.inner` | named operations |
| `format(X, "v")`, `format(X, "l")`, `aa.display(...)` | the same print forms as elements |

## Lie algebras and root systems (`anyalgebra.lie`, v0.1.4)

```python
g = al.so(3, 1)
print(g.dimension, g.jacobi_holds(), g.killing_signature())   # 6 True (3, 3, 0)
print(al.root_system("F", 4).count)                           # 48
print(al.root_system("G", 2).weyl_dimension((1, 0)))          # 7
```

| Call | Meaning |
|---|---|
| `al.so(p, q)`, `al.su(p, q)`, `al.sl(n)`, `al.sp(n)`, `al.gl(n)` | classical real Lie algebras |
| `al.from_matrices(matrices)` | structure constants from matrices that close; a dependent family or an open commutator is an error naming the pair |
| `al.generated_by(matrices)`, `al.lie_closure(matrices)` | the Lie algebra generated under the commutator |
| `g.jacobi_holds()`, `g.killing_form()`, `g.killing_signature()` | exact checks; the signature is `(noncompact, compact, degenerate)` |
| `g.is_semisimple()`, `g.center_dimension()`, `g.derived_dimension()` | invariants |
| `g.as_algebra()` | the bracket as a convenience-layer algebra |
| `al.cartan_matrix(family, rank)`, `al.root_system(family, rank)` | types A to G in the Bourbaki numbering |
| `system.positive`, `system.count`, `system.weyl_dimension(labels)` | roots and Weyl's dimension formula |

## Clifford algebras and gamma matrices (`anyalgebra.clifford`, v0.1.5)

```python
Cl = ac.clifford(1, 3)
print(Cl.e1 * Cl.e2, Cl.e2 * Cl.e1)                         # e12 -e12
gammas = ac.gamma_matrices(11, 3)                           # 128-by-128, instant
print(ac.satisfies_clifford_relations(gammas, 11, 3))       # True
```

| Call | Meaning |
|---|---|
| `ac.clifford(p, q, r)`, `ac.grassmann(n)` | the algebra as a handle with blade basis, up to 8 generators |
| `ac.grades`, `ac.grade_part`, `ac.pseudoscalar`, `ac.even_subalgebra` | grading |
| `ac.reversion`, `ac.grade_involution`, `ac.clifford_conjugation` | the three involutions as signs; `x.conj()` is reversion |
| `ac.blade_product(a, b, signature)` | one blade product on bit masks, for any number of generators |
| `ac.pauli()`, `ac.gamma_matrices(p, q)` | exact matrices, one nonzero entry per row |
| `ac.satisfies_clifford_relations`, `ac.chirality`, `ac.charge_conjugations` | exact checks and the standard operators |
| `ac.spin_generators`, `ac.spin_algebra(p, q)` | the spin Lie algebra, compared in the tests with `al.so(p, q)` |

## Jordan algebras (`anyalgebra.jordan`, v0.1.6)

```python
J = aj.hermitian(ca.octonions(), 3)      # the 27-dimensional Albert algebra
print(J.rank, aj.is_jordan(J))           # 27 True
print(aj.is_jordan(aj.hermitian(ca.octonions(), 4)))    # False
```

| Call | Meaning |
|---|---|
| `aj.hermitian(A, n)` | `J_n(A)` with the Jordan product; basis `d1..dn`, `xij`, `xij.a` |
| `aj.one(J)`, `aj.trace(x)`, `aj.trace_form(x, y)` | unit, trace, trace form |
| `aj.freudenthal(x, y)`, `aj.determinant(x)` | cross product and cubic norm on `J_3(A)` |
| `aj.is_commutative`, `aj.satisfies_jordan_identity`, `aj.is_jordan` | exact law tests by full linearization |
| `aj.tensor_jordan(A, n, B)`, `aj.tensor_jordan_dimension` | the carrier `A tensor J_n(B)` |

## Matrix Lie algebras over an algebra (`anyalgebra.matrix_lie`, v0.1.7)

Over the octonions, anti-Hermitian matrices do not close under the commutator.
The algebra is the one *generated* by the matrices acting as linear maps.

```python
print(ml.su(2, ca.octonions()).dimension)    # 36
print(ml.sl(2, ca.octonions()).dimension)    # 45
```

| `n`, kind | R | C | H | O |
|---|---|---|---|---|
| `su(2, K)` | `so(2)`, 1 | `su(2)`, 3 | `so(5)`, 10 | `so(9)`, 36 |
| `sl(2, K)` | `so(2,1)`, 3 | `so(3,1)`, 6 | `so(5,1)`, 15 | `so(9,1)`, 45 |
| `su(3, K)` | `so(3)`, 3 | `su(3)`, 8 | `sp(3)`, 21 | `f4`, 52 |
| `sl(3, K)` | `sl(3,R)`, 8 | `sl(3,C)`, 16 | `su*(6)`, 35 | `e6(-26)`, 78 |

The tests reproduce every dimension, and for all but the two largest entries
also the Killing-form signature. For `su(3, O)` and `sl(3, O)` the 27-by-27
closure is run modulo a prime by `ml.dimension_certificate`: the dimension is
at least 52 and 78 exactly, and closure is verified modulo the prime.

Matrices larger than 2-by-2 over a nonassociative algebra other than 3-by-3
over an alternative one have no accepted definition and raise an error.

## Coming from AlgMul

| AlgMul | Here |
|---|---|
| `MMul`, `MTMul`, `MConj`, `HermConj`, `IsHermitian`, `IsAntiHermitian`, `MTrace` | `X @ Y`, `X.conj()`, `X.dagger()`, `X.is_hermitian()`, `X.is_antihermitian()`, `X.trace()` |
| `LieProd`, `MatrixToLieStruct` | `am.comm`, `al.from_matrices` |
| `SOnGens`, `SUnFund`, `MakeSOnFund`, `MakeSUnFund` | `al.so(p, q)`, `al.su(p, q)` |
| `MRootsFromChevalley` | `al.root_system(family, rank)` |
| `GAMul`, `Cliff`, `GrMul` | `ac.clifford(p, q)`, `ac.grassmann(n)` and `*` |
| `PauliMatrices`, `GammaMatrices`, `GammaChiral`, `GammaCharge`, `SpinFund` | `ac.pauli()`, `ac.gamma_matrices`, `ac.chirality`, `ac.charge_conjugations`, `ac.spin_generators` |
| `J2`, `J3`, `J4`, `JordanProd`, `FreudenthalProd` | `aj.hermitian(A, n)` and `*`, `aj.freudenthal` |
| `TJ2`, `TJ3`, `TJ4` | `aj.tensor_jordan(A, n, B)` |
| `SU2A`, `SU3A`, `SL2AGens`, `SUnAGens` | `ml.su(n, A)`, `ml.sl(n, A)` |

## Not covered yet

- Weyl, Dirac and Majorana basis changes and the vector-to-spinor conversions.
- Clifford idempotent searches and Fierz identities.
- Group-level transformations (finite rotations and exponentials).
- Representations and branching rules of the Lie algebras.
- Killing-form signatures for the two 3-by-3 octonionic algebras.
- Print forms for tensor-product elements beyond the element forms.
