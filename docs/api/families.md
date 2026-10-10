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

The tests reproduce every dimension and every Killing-form signature. The
two largest are exact as well: `su(3, O)` is compact of dimension 52, and
`sl(3, O)` has signature `(26, 52, 0)`, the real form `e6(-26)`. Each takes
well under a minute. `ml.dimension_certificate` remains as a quick check
modulo a prime.

| Call | Result |
|---|---|
| `ml.multiplication_algebra(A)` | the Lie algebra generated by `x -> a x` for imaginary `a`; `so(8)` for the octonions |
| `ml.multiplication_algebra(A, right=True)` | with `x -> x a` as well; `so(4)` for the quaternions |
| `ml.so(n, A)` | antisymmetric matrices over a commutative associative `A`; `so(n, C)` has real dimension `n(n-1)` |

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

## Second round (v0.1.9)

What the first versions left open, by module.

**Clifford: idempotents and spinors.** A primitive idempotent is a product of
factors `(1 + b)/2` over commuting blades that square to `+1`.

```python
Cl = ac.clifford(3, 1)
f = ac.primitive_idempotents(Cl)[0]
print(len(ac.left_ideal(f)), ac.division_ring(f))    # 4 R
real_gammas = ac.majorana_matrices(3, 1)             # four real 4-by-4 matrices
```

| Call | Result |
|---|---|
| `ac.commuting_involutions(Cl)` | a maximal set of independent commuting blades with square `+1` |
| `ac.idempotent(blades, signs)`, `ac.is_idempotent(x)` | one idempotent; the test `x*x == x` |
| `ac.primitive_idempotents(Cl)` | mutually annihilating primitive idempotents that sum to 1 |
| `ac.left_ideal(f)`, `ac.division_ring(f)` | a basis of `Cl f`; `R`, `C` or `H` from the dimension of `f Cl f` |
| `ac.spinor_representation(Cl)` | rational matrices of the generators on a minimal left ideal |
| `ac.majorana_matrices(p, q)` | real gamma matrices where the ring is `R`; an error naming the ring otherwise |

The tests compare the ideal dimension and the ring with the classification
table for thirteen signatures. The table form costs `4^n` cells, so seven or
eight generators take from ten seconds to a minute.

**Gamma matrices: bases and vectors.** `ac.gamma_matrices` is a Weyl basis:
its chirality matrix is diagonal. Dense matrices are `am.Mat` over one shared
complex algebra.

| Call | Result |
|---|---|
| `ac.as_matrix(g)` | a monomial matrix as a dense complex matrix |
| `ac.dirac_basis(gammas, index)` | the basis where gamma `index` is diagonal, and the matrix `S` that leads there; `S @ S == 2` |
| `ac.change_basis(gammas, S, S_inverse)` | `S g S^-1`, after checking the inverse |
| `ac.is_real`, `ac.is_imaginary` | Majorana and pseudo-Majorana tests |
| `ac.slash(gammas, v)`, `ac.unslash(gammas, M)` | a vector as `v_a g_a` and back |

**Lie: exact finite transformations.** The exponential is not rational, so
the Cayley transform takes its place.

| Call | Result |
|---|---|
| `al.cayley(X)` | `(1 - X)^-1 (1 + X)`, a group element when `X` is in the Lie algebra of a form |
| `al.rotation(p, q, a, b, t)` | a rotation, or a boost, with half-angle tangent `t`; `t = 1/2` gives the 3-4-5 rotation |
| `al.preserves_form(g, F)`, `al.inverse(M)` | the exact test `g^T F g == F`; the exact inverse |

**Jordan: reports.** `aj.report(J)` prints the exact Jordan facts of an
algebra, and `aj.proof_table([R, C, H, O], (2, 3, 4))` prints which `J_n(A)`
are Jordan: all of them except `J_4(O)`.

**Tensor products with three factors.** `ca.tensor(ca.tensor(C, H), O)` now
works; the third factor's units are named `g1, g2, ...`.
`ca.factors_conjugation([C, H, O], [0, 2])` gives the signs that conjugate
the chosen factors, for elements and, through `X.conj(signs)` and
`X.dagger(signs)`, for matrices.

## Lie structure theory (`anyalgebra.lie_structure`, v0.1.9)

Given structure constants and nothing else, the module says which Lie
algebra it is.

```python
import anyalgebra.lie_structure as ls

print(ls.identify(ml.sl(2, ca.octonions())))     # D5, real form so(9,1)
print(ls.identify(al.so(3, 1)))                  # A1+A1, real form sl(2,C)

found = ls.root_decomposition(al.sl(3))
print(found.type, len(found.roots), found.cartan_matrix())
```

| Call | Result |
|---|---|
| `ls.cartan_subalgebra(g)`, `ls.rank(g)` | a Cartan subalgebra of any Lie algebra, by de Graaf's algorithm |
| `ls.radical(g)` | a basis of the solvable radical |
| `ls.root_decomposition(g)` | a Cartan subalgebra with roots and root vectors, for a reductive algebra |
| `found.roots`, `found.positive`, `found.simple`, `found.vectors` | the roots as rational tuples and one vector per root space |
| `found.cartan_matrix()`, `found.cartan_integer(b, a)` | Cartan integers read off the root strings |
| `found.type`, `found.components()` | the Cartan type of the complexification, such as `A1+A1` or `A2+T1` |
| `found.summands()` | the simple real ideals by name |
| `found.chevalley_generators()` | `(e_i, f_i, h_i)` for each simple root, satisfying the Chevalley relations |
| `ls.classify(cartan_matrix)` | the simple types of any Cartan matrix, in any numbering |
| `ls.real_forms(family, rank, character)` | the real forms of a simple type with a given character |
| `ls.identify(g)` | type and real form in one line |

Roots are found over the Gaussian rationals `Q(i)`. A compact direction of
a real form has imaginary eigenvalues and a noncompact one real eigenvalues,
and `found.kinds` records which is which. The real form of each simple ideal
is read off the Killing signature; two components that complex conjugation
swaps are reported as one complex algebra, as in `sl(2,C)`.

The tests identify twenty-three algebras, among them `su(3, H)` as `sp(3)`,
`sl(3, H)` as `su*(6)`, and, in the slow lane, `su(3, O)` as `f4` and
`sl(3, O)` as `e6(-26)`.

Limits, stated plainly:

- The Cartan subalgebra used for roots is assembled from basis elements and
  sums or differences of two of them. If none of those is diagonalizable
  with eigenvalues in `Q` or `iQ`, the call raises an error with code
  `split`. It does not search further and does not extend the field.
- Two real forms can share a character, such as `so(12,6)` and `so*(18)`.
  Both names are then given.
- There is no Levi decomposition yet, only the radical, and no isomorphism
  test beyond comparing what `identify` returns.

## Not covered yet

- Fierz identities and the bilinear spinor conversions.
- Symplectic matrix algebras over an algebra.
- Representations and branching rules of the Lie algebras.
- Print forms for tensor-product elements beyond the element forms.
