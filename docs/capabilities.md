# What can AnyAlgebra do today?

**Checked:** 2026-08-16
**Source version:** locally complete 0.1.0 source milestone

All 60 v0.1 tasks are accepted and `V01-060` passed the aggregate readiness
gate. This page describes locally verified source behavior, not a tagged or
published release.

AnyAlgebra currently has a usable general-purpose kernel for exact finite
algebraic calculations. It can define unfamiliar structures directly from
sorts, carriers, operations, relations, rules, bases, and structure constants.
It can then evaluate expressions, validate laws, produce counterexamples,
change presentations, recover operator structure, compute bounded invariants,
and save replayable evidence.

The package does not yet have the high-level family constructors planned for
later versions. There is no stable `Octonion`, `LieAlgebra`, `CliffordAlgebra`,
`GL(n, A)`, or `A tensor J_n(B)` interface. Familiar algebras currently enter
through the general kernel, named fixtures, or the experimental research
namespace.

## Capability map

| Task | Status | Current route |
|---|---|---|
| Exact integer and rational arithmetic | Available | `anyalgebra.core.domains.ZZ`, `QQ` |
| Explicit exact coercions | Available | `CoercionMap`, `CoercionGraph`, `coerce`, `common_parent` |
| Finite-rank vectors over an exact domain | Available | `Basis`, `FreeModule`, sparse elements |
| Arbitrary finite many-sorted structures | Available | sorts, carriers, signatures, operations, relations, and `StructureBuilder` |
| Partial operations | Available | four explicit outcomes: defined, undefined, indeterminate, and failed |
| Deductive or rule-generated systems | Available | bounded closure with a replayable derivation trace |
| Finite multilinear algebras | Available | multiplication tables, callables, or structure constants of declared arity |
| Generic multilinear element evaluation | Available | `evaluate_multilinear` for exact sparse elements of declared arity |
| Bounded finite-carrier census | Available | immutable `CensusSpec`, reference enumeration, symmetry pruning, and exact accounting |
| Canonical labels and isomorphism outcomes | Available | canonical operation tables, checked bijections, bounded negative and inconclusive evidence |
| Deterministic finite-algebra atlas | Available | build, load, validate, query, replay, and `anyalgebra-atlas` CLI |
| Exact matrices and linear maps | Available | matrix parents, matrix operations, span and solve utilities |
| Operator closure and structure recovery | Available | recover structure constants or return dependence/nonclosure witnesses |
| Presentation conversion | Available | explicit conversion graphs, route selection, inverses, and ambiguity rejection |
| Exact basis changes | Available | basis isomorphisms with transport certificates |
| Law checking | Available | associativity, commutativity, alternativity, flexibility, Jacobi, Jordan, and custom laws |
| Counterexample search | Available | exhaustive finite search, bounded sampling, and deterministic witness minimization |
| Elementary algebra analysis | Available | commutators, associators, nuclei, center, ideals, derivations, units, and fingerprints |
| Tensor-product bookkeeping | Limited | ordered tensor parents and pure tensors only; no general tensor arithmetic yet |
| Evidence and replay | Available | contracts, receipts, canonical JSON, replay, staleness, and claim-status controls |
| Backend comparison | Available | exact reference backend and differential comparison contracts |
| AlgMul migration audit | Available | static capture, bounded Mathematica capture, generated-name reconstruction, and disposition ledgers |
| `H`, `Hs`, `O`, and `Os` | Available as generic fixtures | exact named multiplication tables, not specialized element classes |
| Albert algebra and compact `F4` | Experimental | `anyalgebra.experimental.exceptional` |
| Magic-square catalogues for `n=2` and `n=3` | Experimental | `anyalgebra.experimental.magic_square` |
| Baker--Campbell--Hausdorff series through degree four | Experimental | `anyalgebra.experimental.bch` |
| Exact Lie-algebra checks on structure constants | Experimental | `anyalgebra.experimental.exact_linear`: Jacobi on every basis triple, trace forms, inertia, and modular rank |
| Stable Lie, Clifford, Jordan, and classical-group families | Not yet | planned for later project-driven versions |
| General `GL/SL/SO/SU(n, A)` | Not yet | no stable constructor or representation layer |
| General `A tensor J_n(B)` | Not yet | only lower-level ingredients and a specific Albert implementation |
| Manifolds, bundles, gauge gravity, and Schouten integration | Not yet | planned package-integration work |

## How to import it

The root package intentionally exports only `__version__`. Import working APIs
from their owning modules:

```python
from anyalgebra.core.domains import QQ, ZZ
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.structures.operations import PartialOperation
from anyalgebra.algebra.multilinear import (
    FiniteMultilinearStructure,
    evaluate_multilinear,
)
from anyalgebra.analysis.fingerprint import algebra_fingerprint
```

This is deliberate namespace policy, not an indication that the submodule APIs
are missing.

## Start with the executable examples

Run these from the repository root after `python -m pip install -e ".[dev]"`.

### Define an arbitrary partial multiplication

```powershell
python examples/partial_magma.py
```

Confirmed output:

```json
{"defined":"c","explicit_undefined":"table cell explicitly marked undefined","omitted_undefined":"table cell is not declared","totality":"partial"}
```

The example defines a three-element carrier and only four multiplication-table
cells. An explicitly undefined cell remains distinguishable from an omitted
cell. The package does not silently add a bottom element.

### Define an arbitrary many-sorted structure

```powershell
python examples/two_sorted_structure.py
```

Confirmed output:

```json
{"evaluation":"right","relation_false":false,"relation_true":true,"ternary_arity":3}
```

This exercises two sorts, a ternary operation, term evaluation, and a relation.
The structure is not required to be a magma, ring, or other named family.

### Compute bounded deductive closure

```powershell
python examples/deductive_system.py
```

Confirmed output:

```json
{"conclusions":["rain","rain_implies_wet","wet"],"premise_indices":[0,1],"status":"complete"}
```

Rules retain premise indices and a derivation trace. Cycles and incomplete
bounds have explicit outcomes.

### Build an exact algebra from a multiplication table

```powershell
python examples/rational_algebra.py
```

This constructs a rank-two noncommutative algebra over `QQ`, multiplies generic
sparse elements, round-trips its multiplication table and structure constants,
and transports the constants through an exact basis change. The confirmed
generic product is

```text
(p + 2q)(2p - q) = -3/2 p - 4q.
```

### Evaluate arbitrary sparse elements explicitly

```powershell
python examples/multilinear_evaluation.py
```

This v0.1 worktree example evaluates a generic nonassociative binary table, a
ternary map, quaternion and octonion sparse products, and split-composition
zero divisors through the same public evaluator. It keeps nested parentheses
explicit and makes no classification or physics claim.

### Find an exact octonion counterexample

```powershell
python examples/octonion_counterexample.py
```

Confirmed result in the `algmul.O.v1` convention:

```text
first nonzero associator after 85 checks: ('e1', 'e2', 'e4')
(e1*e2)*e4 - e1*(e2*e4) = 2*e7
```

The complete `8^3` basis grid is sufficient because the loaded product is
bilinear and the associator is trilinear.

### Analyze the named composition fixtures

```powershell
python examples/generic_composition_algebras.py
```

The current exact reports reproduce these facts:

| Fixture | Dimension | Associativity | Nucleus dimension | Center dimension | Derivation dimension |
|---|---:|---|---:|---:|---:|
| `algmul.H.v1` | 4 | proved | 4 | 1 | 3 |
| `algmul.Hs.v1` | 4 | proved | 4 | 1 | 3 |
| `algmul.O.v1` | 8 | disproved by witness | 1 | 1 | 14 |
| `algmul.Os.v1` | 8 | disproved by witness | 1 | 1 | 14 |

The fingerprint is isomorphism-rejection data, not a complete isomorphism
invariant. The ideal search is bounded and reports its incomplete status.

### Recover algebra structure from operators

```powershell
python examples/operator_recovery.py
```

The example confirms one closed operator family and returns exact certificates
for two failure modes:

- a linear dependence relation in the declared generator family;
- a product outside the declared span, with a separating functional and a
  replayable witness.

### Convert between presentations

```powershell
python examples/presentations.py
```

The example round-trips symbolic and coordinate presentations, then table and
structure presentations. A graph with two equally valid routes returns an
ambiguity error instead of selecting one silently.

### Save and replay a calculation

```powershell
python examples/evidence_replay.py
```

Confirmed result:

```text
round trip: true
exact replay: exact_match
changed dependency replay: stale
```

This layer keeps successful execution, mathematical outcome, and scientific
claim status separate.

## Experimental exceptional-algebra calculations

These modules are tested and useful for research, but their API may change.
Import them explicitly through `anyalgebra.experimental`.

### Albert algebra and compact F4 control

```powershell
python examples/f4_albert_certificate.py
```

The current certificate constructs `H_3(O)` in 27 coordinates and confirms:

- a 52-dimensional inner-derivation span;
- derivation-constraint rank 677 and nullity 52;
- exact closure of 1,378 unordered generator pairs;
- 52 positive exact LDL pivots for the compact trace Gram matrix; and
- numerical preservation of trace, trace form, and determinant under one
  truncated exponential action.

The last item is numerical. It is not a global group-coordinate proof.

### Degree-two and degree-three magic-square catalogues

```powershell
python examples/magic_square_certificate.py
```

The catalogue distinguishes raw carrier dimension from Lie-algebra dimension.
It records all 49 ordered division/split entries for `n=2` and `n=3`. The
degree-two formula produces `so(p,q)` from the summed norm signatures and can
materialize exact standard plane generators. The requested
`O tensor J2(O)` control checks all 120 generators and all 7,260
unordered-with-repetition commutators. The 49 degree-three real-form labels are
checked entry-by-entry against Tables 1--3 of
Cacciatori--Cerchiai--Marrani (2012). They are labels from the published
tables; no degree-three bracket is constructed here.

### Exact checks on structure constants

```powershell
python -m pytest -q tests/experimental/test_lie_algebra_checks.py
```

`anyalgebra.experimental.exact_linear` checks a table of basis brackets in
exact arithmetic. `jacobi_basis_triples` verifies antisymmetry and then the
Jacobi identity on every distinct basis triple, working on integer numerators
over one common denominator. `sparse_trace_gram` gives the trace form of a
family of operators, which for adjoint operators is the Killing form.
`exact_symmetric_inertia` returns the signature of a rational symmetric form,
and `modular_rank` gives a lower bound for a rational rank. The tests apply
them to `so(3)` and to the commutator algebra of the imaginary octonions,
which is antisymmetric but fails the Jacobi identity.

### Baker--Campbell--Hausdorff series

`anyalgebra.experimental.bch` derives `log(exp(X) exp(Y))` through degree four
in the free associative algebra and solves for its coordinates in a Hall
basis of the free Lie algebra. The numeric evaluator uses those derived
coefficients. The tests compare it exactly with the matrix logarithm on
nilpotent rational matrices.

## What should you use it for now?

AnyAlgebra is ready for small and medium exact calculations whose structure can
be expressed by a finite signature, finite basis, multiplication table,
structure constants, or finite operator family. It is also ready to serve as a
verification harness for experimental exceptional-algebra constructions.

It is not yet ready as a concise everyday interface for arbitrary named
division algebras, Lie algebras and representations, Clifford algebras,
classical groups over coefficient algebras, tensor-Jordan systems, or geometric
field theory. Those are the next layers to build on the kernel.

For the exhaustive worked manual, see
[AnyAlgebra through its mathematical tests](guides/mathematical-test-manual.md).
For the four composition fixtures, their exact conventions, and a complete
focused test inventory, see
[Using quaternions and octonions](guides/quaternions-and-octonions.md).
For exact import signatures, see the [v0.0 quickstart](api/v0.0-quickstart.md)
and [v0.0 API contract](api/api-v0.0.md).
