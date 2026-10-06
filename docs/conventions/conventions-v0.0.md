# AnyAlgebra v0.0 conventions

Status: Normative fixture conventions, not universal defaults  
Convention namespace: `anyalgebra.v0_0`  
Last updated: 2026-07-22

## 1. Principle

AnyAlgebra does not declare one convention universally correct. Every
calculation selects a `ConventionManifest`. This document defines canonical
v0.0 fixtures so tests and legacy comparisons are reproducible. Equivalent
conventions are connected by explicit maps and are never silently identified.

## 2. Indexing and coordinates

- Mathematical basis indices are displayed from zero when the unit is included:
  `(e_0, e_1, ..., e_{n-1})`, with `e_0 = 1` where appropriate.
- Python coordinate arrays are zero based and use the same order.
- A sparse coordinate map omits exact zeros and stores keys in increasing basis
  order for canonical serialization.
- Repeated indices are not summed unless an expression explicitly declares
  Einstein summation or uses a contraction operator.
- Structure constants use
  `e_i * e_j = sum_k c[i,j,k] e_k`; Lie constants use
  `[e_i,e_j] = sum_k f[i,j,k] e_k`.

## 3. Scalar domains

- Integers serialize as decimal integers.
- Rationals are reduced with positive denominator and serialize as
  `{"numerator": p, "denominator": q}`.
- Exact zero has one canonical representation in each domain.
- Floating values are not silently converted to rationals.

## 4. Named composition-algebra fixtures

These are generic table fixtures in v0.0, not specialized public classes.

### 4.1 Complex and split-complex numbers

`algmul.C.v1` uses basis `(1,i)` and `i^2 = -1`.

`algmul.Cs.v1` uses basis `(1,j)` and `j^2 = +1`. The split conjugation sends
`j` to `-j` and the norm is `a^2-b^2` for `a+bj`.

### 4.2 Quaternions

`algmul.H.v1` uses basis `(1,i,j,k)` with

```text
i^2 = j^2 = k^2 = -1
i j = k,   j k = i,   k i = j
j i = -k,  k j = -i,  i k = -j
```

Conjugation negates `(i,j,k)`.

### 4.3 Split quaternions

`algmul.Hs.v1` follows the evaluated AlgMul table with basis `(1,i,j,k)`:

```text
i^2 = -1,  j^2 = +1,  k^2 = +1
i j = k,   j k = -i,  k i = j
j i = -k,  k j = i,   i k = -j
```

Conjugation negates `(i,j,k)`. This convention must not be confused with a
permuted basis used by another split-quaternion source.

### 4.4 Octonions

`algmul.O.v1` uses basis `(1,e1,...,e7)`, `e_a^2=-1`, anticommutation for
distinct imaginary units, and positively oriented triples

```text
(1,2,3), (1,4,5), (1,7,6), (2,4,6),
(2,5,7), (3,4,7), (3,6,5).
```

For a positive triple `(a,b,c)`, `e_a e_b=e_c`, `e_b e_c=e_a`, and
`e_c e_a=e_b`; reversing an ordered pair changes the sign. Conjugation negates
all imaginary units. Products with more than two factors require explicit
parentheses.

### 4.5 Split octonions

`algmul.Os.v1` is defined by the exact eight-dimensional table evaluated from
the pinned `AlgMul.wl` source, not by assuming that a verbal Fano-plane rule
matches another source:

| · | 1 | e1 | e2 | e3 | e4 | e5 | e6 | e7 |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | e1 | e2 | e3 | e4 | e5 | e6 | e7 |
| e1 | e1 | -1 | e3 | -e2 | -e5 | e4 | -e7 | e6 |
| e2 | e2 | -e3 | -1 | e1 | -e6 | e7 | e4 | -e5 |
| e3 | e3 | e2 | -e1 | -1 | -e7 | -e6 | e5 | e4 |
| e4 | e4 | e5 | e6 | e7 | 1 | e1 | e2 | e3 |
| e5 | e5 | -e4 | -e7 | e6 | -e1 | 1 | e3 | -e2 |
| e6 | e6 | e7 | -e4 | -e5 | -e2 | -e3 | 1 | e1 |
| e7 | e7 | -e6 | e5 | -e4 | -e3 | e2 | -e1 | 1 |

Thus `e1^2=e2^2=e3^2=-1` and
`e4^2=e5^2=e6^2=e7^2=+1`. The generated evaluated table is stored in
`legacy/generated/algmul-evaluated-surface.json` with the source hash and kernel
version. This table is the normative parity fixture; its classification as the
split octonions must also pass independent alternativity, conjugation, norm, and
signature tests before it is used as a correctness oracle.

## 5. Tensor products

For ordered bases `(a_i)` of `A` and `(b_j)` of `B`, the tensor basis is

```text
(a_0⊗b_0, ..., a_0⊗b_{m-1}, a_1⊗b_0, ..., a_{n-1}⊗b_{m-1}).
```

The left factor is the major index, matching AlgMul's `TChars` ordering.
Multiplication in the ordinary algebra tensor product is factorwise,

```text
(a⊗b)(c⊗d) = (ac)⊗(bd),
```

only when the scalar/braiding hypotheses for that construction are declared.
Other graded, braided, or twisted tensor products are distinct constructors.

For `A tensor J_n(B)`, the tensor carrier and the Jordan matrix parent remain
separate typed factors. `O tensor J_3(O)` is a required later fixture, not a
special case baked into the tensor implementation.

## 6. Matrices over arbitrary parents

- Matrix indices are displayed mathematically from 1 but stored from 0.
- Entries have one declared parent; Python shape never determines their algebra.
- Matrix addition is entrywise.
- Matrix multiplication uses the explicit ordered sum
  `C[i,j] = (((A[i,0]B[0,j] + A[i,1]B[1,j]) + ...)`. The additive parent must
  make this unambiguous; entry products retain their declared order.
- Associativity of matrix multiplication is not inferred when entry
  multiplication is nonassociative.
- Determinant, trace cyclicity, inverse, rank, and characteristic-polynomial
  identities are unavailable unless their hypotheses and chosen definitions are
  supplied.

Nested matrices and matrices over tensor products carry recursive parent data.

## 7. Terms and partial operations

- Multiplication strings with three or more factors are invalid unless the
  parser preserves parentheses or the parent declares associativity with
  evidence accepted for that context.
- A missing table entry means `Undefined`, never zero.
- An unevaluated search caused by a declared bound means `Indeterminate`.
- A malformed coordinate or parent mismatch means `Failed`/typed input error,
  not mathematical undefinedness.

## 8. Law conventions

- Commutator: `[x,y] = x*y - y*x`.
- Associator: `(x,y,z) = (x*y)*z - x*(y*z)`.
- Left, middle, and right nuclei use vanishing associators in the corresponding
  slot; the nucleus is their intersection.
- The center is the nucleus intersected with elements commuting with all
  elements.
- A derivation satisfies `D(x*y)=D(x)*y+x*D(y)`.
- Law results are `Proved`, `Disproved(counterexample)`, or
  `Inconclusive(bounds)`, never a bare boolean in evidence-bearing calculations.

## 9. Lie and Clifford fixture conventions

These conventions reserve interoperability rules for later releases:

- Lie bracket is `[x,y]=xy-yx` when induced from an associative product.
- Long roots have squared length `2` unless a manifest states otherwise.
- A signature `(p,q)` has `p` positive and `q` negative metric directions.
- Clifford generators satisfy
  `gamma(v)gamma(w)+gamma(w)gamma(v)=2 g(v,w) I`.
- A real form is identified by an explicit real structure/signature label, not
  inferred from a dimension or complexification.

No specialized Lie or Clifford API is promised by v0.0.

## 10. Convention manifest fields

Every stable manifest records:

```json
{
  "schemaType": "ConventionManifest",
  "schemaVersion": "1.0",
  "id": "algmul.O.v1",
  "basisOrder": ["1", "e1", "e2", "e3", "e4", "e5", "e6", "e7"],
  "coefficientDomain": "QQ",
  "multiplicationFixture": "sha256:...",
  "involutions": {"conjugation": "..."},
  "indexing": "zero-based-storage",
  "parenthesization": "explicit",
  "sources": ["source-anchor-id"],
  "status": "accepted"
}
```

Allowed status values are `provisional`, `accepted`, `deprecated`, and
`superseded`. Conversion maps between manifests are versioned mathematical maps
with round-trip tests and declared preserved structure.
