# AnyAlgebra through its mathematical tests

This document is a mathematics-first guide to the AnyAlgebra 0.0.1 source
milestone. The separate [current-status page](../status.md) records installation,
artifact, tag, and publication status. For a shorter task-to-example map, start
with [What can AnyAlgebra do today?](../capabilities.md). This guide uses the
test suite as an executable manual. Each worked example states what is being
constructed, what was checked, and what the result means. The final catalogue
covers every test module, including modules omitted from the worked path
because they concern malformed inputs, serialization defenses, packaging, or
release infrastructure.

The package is deliberately general. It does not yet provide specialized
public classes named `Octonion`, `LieAlgebra`, `CliffordAlgebra`, or
`JordanAlgebra`. The current source milestone supplies exact domains, free
modules, arbitrary signatures, total and partial structures, multilinear tables,
matrices, basis changes, finite validation, elementary algebra analysis,
morphisms, evidence records, and migration tooling. Familiar algebras appear
as named fixtures loaded through those generic mechanisms.

## Verified state

The following results were reproduced on 2026-08-11:

- the mathematics-facing selection under `core`, `structures`, `algebra`,
  `linear`, `maps`, `presentations`, `validation`, `analysis`, `fixtures`,
  `examples`, and `integration` passed **923 tests**.
- the exact-domain, parent-coexistence, partial-magma, two-sorted, deductive,
  presentation, rational-algebra, operator-recovery, octonion, and generic
  composition-algebra examples ran successfully.
- all **16 self-contained Python blocks** in this manual executed without an
  error.
- the last clean pre-experimental v0.0.1 suite passed **1,970 tests**, with one
  explicitly opt-in Wolfram 12.0 replay skip and one expected duplicate-archive
  warning from a corruption test. The exact v0.0.1 coverage record remains
  **14,578/14,578 statements** and **4,960/4,960 branches** across 60 canonical
  source files.

The [current-status page](../status.md) records later research additions. On
2026-08-14, the current tree passed **2,009 tests**, with one opt-in Wolfram
replay skipped and one expected corruption-fixture warning. Ruff, strict
MyPy, the project contract, and the frozen v0.0.1 governance checks pass. The
experimental additions remain outside this manual's closed stable-API claim.

The frozen coverage ledger shows that every stable-kernel path was executed.
It does not prove a general theorem, literal AlgMul equivalence, or a physics
claim. The tests below make narrower claims with explicit finite domains, exact
coefficient parents, conventions, and work bounds.

## Running the manual

From the repository root:

```powershell
python -m pytest -q
python -m pytest -q tests/core tests/structures tests/algebra tests/linear `
  tests/maps tests/presentations tests/validation tests/analysis `
  tests/fixtures tests/examples tests/integration
```

Run one example directly:

```powershell
python examples/exact_domains.py
python examples/rational_algebra.py
python examples/octonion_counterexample.py
```

Run one test by its node ID:

```powershell
python -m pytest -q `
  tests/analysis/test_fingerprint.py::test_derivation_controls_and_direct_leibniz_check
```

AnyAlgebra keeps the root namespace small. Most imports come from the module
that owns the concept, as the examples below show.

Code blocks that import an `examples` builder or construct every object locally
can be run as written. A few later blocks are abridged test fragments and use
fixture names such as `source`, `target`, or `nonassociative_fixture`. Their
surrounding text identifies the fixture, and the complete executable version is
named in the test-module catalogue.

## 1. Exact integers, rationals, and explicit coercion

`ZZ()` and `QQ()` are canonical exact parents. Rational values are normalized
by greatest common divisor and denominator sign. Booleans and floating-point
values are not silently treated as exact integers or rationals.

The following is adapted directly from
`tests/core/test_qq.py`:

```python
from anyalgebra.core.domains import QQ

left = QQ().element((-7, 12))
right = QQ().element((5, -18))  # normalized to -5/18
zero = QQ().element(0)
one = QQ().element(1)

assert left + right == QQ().element((-31, 36))
assert left - right == QQ().element((-11, 36))
assert left * right == QQ().element((35, 216))
assert left / right == QQ().element((21, 10))
assert -left == QQ().element((7, 12))
assert left + (-left) == zero
assert left / left == one
```

Confirmed result:

```text
All arithmetic remained exact and every result retained the canonical QQ parent.
```

Coercion is a declared graph, not an implicit Python conversion convention:

```python
from anyalgebra.core.coercions import CoercionGraph, CoercionMap, coerce, common_parent
from anyalgebra.core.domains import QQ, ZZ

zz_to_qq = CoercionMap(
    id="core.zz_to_qq.v1",
    source=ZZ(),
    target=QQ(),
    forward=lambda value: QQ().element(value.value),
    injective=True,
    exact=True,
    lossless=True,
)
graph = CoercionGraph().with_map(zz_to_qq)

integer = ZZ().element(6)
rational = QQ().element((2, 3))
embedded = coerce(integer, QQ(), graph=graph)
plan = common_parent((integer, rational), graph=graph)
values = plan.apply((integer, rational))

assert embedded == QQ().element(6)
assert plan.target is QQ()
assert plan.route_ids == (("core.zz_to_qq.v1",), ())
assert values == (QQ().element(6), QQ().element((2, 3)))
```

The executable test also declares a lossy `QQ -> ZZ` map and confirms that
automatic coercion refuses to call it:

```text
error_type = LossyCoercionError
lossy callable executed = false
```

The coercion tests cover unique paths, multiple-step paths, diamonds, cycles,
parallel paths, incomparable common parents, ambiguity, missing routes, and
literal endpoint identity. A lower numerical cost does not resolve semantic
ambiguity between two lossless routes.

## 2. Parent identity, free modules, and sparse elements

Two parents may have the same mathematical definition without being the same
literal computational parent. AnyAlgebra uses structural fingerprints for
comparison and literal parent identity for operations that would otherwise
silently reinterpret data.

```python
from anyalgebra.core.domains import ZZ
from anyalgebra.core.modules import Basis, FreeModule


def module():
    domain = ZZ()
    return FreeModule(
        domain, Basis(("e0", "e1"), coefficient_domain=domain), name="coexisting"
    )


source = module()
target = module()
x = source.element({0: 2, 1: -1})
y = target.element({0: 2, 1: -1})

assert source == target
assert source is not target
assert source.fingerprint() == target.fingerprint()
assert x != y
```

Confirmed behavior from `examples/coexisting_parents.py`:

```text
same display name: true
same fingerprint: true
same instance: false
same coordinates: [[0, 2], [1, -1]]
Python element equality: false
x + y: ModuleParentMismatchError
target.element(x): SparseElementConstructionError
```

An explicit whole-element map can transport between the parents. The example
uses `(x0,x1) -> (-x1,3x0)` and obtains `(1,6)` in the target parent. This
demonstrates that transport is not assumed to preserve raw coordinates.

Sparse module arithmetic is exact and removes coordinates that cancel:

```python
from anyalgebra.core.domains import QQ
from anyalgebra.core.modules import Basis, FreeModule

domain = QQ()
module = FreeModule(domain, Basis(("e0", "e1", "e2", "e3"), coefficient_domain=domain))
x = module.element({0: (-1, 2), 2: (3, 4)})
y = module.element({0: (1, 2), 2: (1, 4)})

assert x + y == module.element({2: 1})
assert x - y == module.element({0: -1, 2: (1, 2)})
assert -x == module.element({0: (1, 2), 2: (-3, 4)})
assert 2 * x == module.element({0: -1, 2: (3, 2)})
```

The boundary tests also cover rank-zero modules, uncertain coefficient
equality, source-mapping snapshots, preflight of all coordinates before any
coercion is called, and immutable parent fingerprints across hash seeds and
separate processes.

## 3. Arbitrary signatures and partial algebraic structures

The general structure layer begins with sorts, operation symbols, relation
symbols, and a signature. Arity may be zero, one, two, three, or higher. A
structure then binds finite carriers and interpretations to those symbols.

### A partial magma

```python
from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.structures.operations import PartialOperation
from anyalgebra.structures.outcomes import Defined, Undefined
from anyalgebra.structures.signatures import OperationSymbol

element = Sort("partial_magma_element")
carrier = FiniteCarrier(("a", "b", "c"), sort=element)
product = OperationSymbol("product", (element, element), element)
undefined = object()

operation = PartialOperation.from_table(
    product,
    ((element, carrier),),
    (
        (("a", "a"), "a"),
        (("a", "b"), "c"),
        (("b", "a"), "b"),
        (("b", "c"), undefined),
    ),
    undefined_marker=undefined,
)

assert operation.apply("a", "b") == Defined("c")
assert isinstance(operation.apply("b", "c"), Undefined)
assert isinstance(operation.apply("c", "c"), Undefined)
```

Confirmed result:

```text
a*b = c
b*c = Undefined("table cell explicitly marked undefined")
c*c = Undefined("table cell is not declared")
totality = partial
```

An explicitly undefined cell and an omitted cell remain distinguishable. No
bottom element is inserted into the carrier. Separate totalization tests add a
new strict bottom only when the caller explicitly requests it, and preserve an
embedding of the original carrier into the enlarged one.

### A many-sorted structure

The two-sorted example has point and color carriers, a ternary operation
`choose(point,color,point) -> point`, and a binary relation.

```python
from examples.two_sorted_structure import build_example
from anyalgebra.structures.outcomes import Defined

example = build_example()
evaluation = example["evaluation"]
relation = example["relation"]

assert evaluation.outcome == Defined("right")
assert relation.apply("right", "blue").holds is True
assert relation.apply("left", "blue").holds is False
assert example["operation"].symbol.arity == 3
```

Confirmed output:

```json
{"evaluation":"right","relation_false":false,"relation_true":true,"ternary_arity":3}
```

The tests check nullary and 257-ary boundaries, repeated input sorts,
unhashable carrier elements, operation and relation tables, callables,
predicates, sort mismatches, and coexistence of equal-looking structures.

## 4. Terms preserve parentheses

`Term` is an explicit syntax tree. It does not infer associativity or erase
parentheses. Evaluation returns one of four branches:

- `Defined(value)`.
- `Undefined(reason, witness=...)`.
- `Failed(error_type, message)`.
- `Indeterminate(reason)`.

For a binary symbol `mul`, these remain different terms:

```python
from anyalgebra.core.parents import Sort
from anyalgebra.structures.signatures import OperationSymbol
from anyalgebra.structures.terms import Term, Variable

s = Sort("s")
mul = OperationSymbol("mul", (s, s), s)
x, y, z = (Variable(name, s) for name in ("x", "y", "z"))
tx, ty, tz = (Term.variable(v) for v in (x, y, z))

left = Term.apply(mul, Term.apply(mul, tx, ty), tz)
right = Term.apply(mul, tx, Term.apply(mul, ty, tz))

assert left != right
```

Evaluation preflights every variable binding and symbol before invoking a user
callable. If a child is undefined, failed, or indeterminate, that exact outcome
and its path through the syntax tree are retained.

## 5. Rewriting and deductive closure

Rewriting uses ordered, first-applicable rules. It does not claim confluence.
Cycle detection and work-bound exhaustion are separate outcomes:

```python
from anyalgebra.core.parents import Sort
from anyalgebra.structures.rewrite import (
    BoundExhausted,
    CycleDetected,
    RewriteRule,
    RewriteSystem,
    normalize,
)
from anyalgebra.structures.signatures import OperationSymbol
from anyalgebra.structures.terms import Term

s = Sort("scalar")
a = Term.apply(OperationSymbol("a", (), s))
b = Term.apply(OperationSymbol("b", (), s))
system = RewriteSystem((RewriteRule(a, b), RewriteRule(b, a)))

cycle = normalize(a, system, max_steps=4)
bounded = normalize(a, system, max_steps=1)

assert isinstance(cycle, CycleDetected)
assert cycle.entry_index == 0
assert cycle.cycle_path == (a, b, a)
assert cycle.trace == (a, b, a)
assert bounded == BoundExhausted(b, (a, b), 1)
```

Ground deductive systems retain replayable premise indices:

```python
from anyalgebra.structures.deductive import CompleteClosure
from examples.deductive_system import build_example

result = build_example()["closure"]
assert isinstance(result, CompleteClosure)
assert tuple(term.symbol.name for term in result.conclusions) == (
    "rain",
    "rain_implies_wet",
    "wet",
)
assert result.derivations[0].premise_indices == (0, 1)
```

Confirmed output:

```json
{"conclusions":["rain","rain_implies_wet","wet"],"premise_indices":[0,1],"status":"complete"}
```

The deductive tests cover axioms, joins, synchronous rounds, cycles, repeated
premises, zero bounds, incomplete frontiers, and trace replay. They operate on
ground terms in v0.0.1. General unification is not yet claimed.

## 6. Multilinear algebras from tables, callables, and constants

`FiniteMultilinearStructure` is the current general finite algebra container.
It can be constructed from a basis-product table, sparse structure constants,
or one declared multilinear callable. Binary multiplication is common, but the
representation supports nullary, unary, ternary, and high arities.

The rank-two rational example declares

\[
p^2=0,\qquad pq=\frac32p,\qquad qp=-\frac12q,\qquad q^2=q.
\]

```python
from examples.rational_algebra import build_example

example = build_example()
module = example["module"]
structure = example["structure"]

assert structure.evaluate_basis(0, 0) == module.zero()
assert structure.evaluate_basis(0, 1) == module.element({0: (3, 2)})
assert structure.evaluate_basis(1, 0) == module.element({1: (-1, 2)})
assert structure.evaluate_basis(1, 1) == module.element({1: 1})
assert example["generic_product"] == module.element({0: (-3, 2), 1: -4})
assert example["table_round_trip"] == example["table"]
```

Here the generic product is

\[
(p+2q)(2p-q)=-\frac32p-4q.
\]

Confirmed result:

```text
generic product = [(-3/2) p] + [-4 q]
table -> structure -> table round trip = true
```

The structure-constant tests verify sparse canonical support, declared output
basis order, zero-dimensional boundaries, 257-ary singleton operations without
dense tensor allocation, exact coefficient parent checks, and equivalence of
table, callable, and sparse-constant construction routes.

### Ordered tensor products

The tensor-product API in v0.0.1 is a bounded parent and pure-tensor layer:

```python
from anyalgebra.algebra.tensor import pure_tensor, tensor_product
from anyalgebra.core.domains import QQ, ZZ
from anyalgebra.core.modules import Basis, FreeModule

module = FreeModule(ZZ(), Basis(("m",), coefficient_domain=ZZ()))
inner = tensor_product(ZZ(), module, max_factors=2)
inner_value = pure_tensor(inner, ZZ().element(2), module.element({0: 3}))
outer = tensor_product(QQ(), inner, max_factors=2)
value = pure_tensor(outer, QQ().element((1, 2)), inner_value)

assert value.parent is outer
assert value.factors == (QQ().element((1, 2)), inner_value)
```

Factor order and nesting are semantic. `A tensor (B tensor C)`,
`(A tensor B) tensor C`, and `A tensor B tensor C` remain distinct parents.
This release does not flatten nested tensors and does not define tensor
addition or multiplication. Those limitations are tested explicitly.

## 7. Exact matrices, row reduction, kernels, and solving

Matrices have explicit dimensions and an exact entry parent. The solver over
`QQ` returns typed outcomes for unique, dependent, and inconsistent systems.

```python
from anyalgebra.core.domains import QQ
from anyalgebra.linear.matrix import MatrixSpace
from anyalgebra.linear.solve import UniqueSolution, solve

A = MatrixSpace(2, 2, QQ()).element(((2, 1), (1, -1)))
b = MatrixSpace(2, 1, QQ()).element(((5,), (1,)))
report = solve(A, b)

assert isinstance(report, UniqueSolution)
assert report.solution.entry(0, 0) == QQ().element(2)
assert report.solution.entry(1, 0) == QQ().element(1)
reconstructed = A.matmul(report.solution)
assert reconstructed.entry(0, 0) == QQ().element(5)
assert reconstructed.entry(1, 0) == QQ().element(1)
```

The row-reduction tests also confirm

```text
rref([[0,2,4],[1,1,3]]) = [[1,0,1],[0,1,2]]
rank = 2
pivot columns = (0,1)
free columns = (2,)
kernel basis = ((-1,-2,1)^T,)
```

Underdetermined systems return a particular solution and an independently
checked kernel basis. Inconsistent systems return a contradictory reduced row.
Zero equations, zero unknowns, rectangular matrices, noncommutative entry
order, transpose, and exact work limits are all tested.

## 8. Span classification and operator recovery

The span layer distinguishes a unique coordinate decomposition, dependence,
and a vector outside the declared span. Outside-span reports carry a separating
functional.

Operator recovery applies a declared bilinear bracket to a finite family of
matrices and attempts to recover structure constants. Three outcomes are kept
distinct:

```python
from examples.operator_recovery import build_example
from anyalgebra.linear.recovery import RecoveryFailure, RecoverySuccess
from anyalgebra.linear.span import NonClosed

example = build_example()
closed = example["closed"]["report"]
dependent = example["dependent"]["report"]
nonclosed = example["nonclosed"]["report"]

assert isinstance(closed, RecoverySuccess)
assert isinstance(dependent, RecoveryFailure)
assert isinstance(nonclosed, RecoveryFailure)
assert isinstance(nonclosed.closure_evidence, NonClosed)
```

Confirmed report:

```text
closed family:
  bracket calls = 4
  [e0,e1] = e1, [e1,e0] = -e1
  all four reconstructions exact

dependent family:
  bracket calls = 0
  dependency relation = (-1,1)
  relation replayed to zero

nonclosed family:
  first failed pair = (0,0)
  separating pairing = 1
  outside-span witness replayed successfully
```

Dependence is checked before brackets are evaluated, so a dependent family
cannot accidentally produce spurious structure constants. Nonclosure stops at
the first declared pair and retains the product and separating witness.

## 9. Symbolic, coordinate, table, and structure presentations

Conversions form an explicit graph with stable IDs, directions, validity
domains, and optional inverse metadata. The system does not infer a source
presentation from Python container shape.

```python
from examples.presentations import build_example

example = build_example()
assert example["symbolic_round_trip"] == example["symbolic"]
assert example["table_round_trip"] == example["table"]
assert len(example["ambiguity_candidates"]) == 2
```

Confirmed result:

```text
symbolic -> coordinate -> symbolic: equal
table -> finite multilinear structure -> table: equal
two parallel symbolic-to-coordinate routes: ConversionAmbiguityError
automatic route selected: false
```

This matters for future exotic-algebra work. A list of coefficients, a matrix,
and a nested tensor should not be reinterpreted merely because their Python
shapes happen to resemble one another.

### Exact basis changes

```python
from anyalgebra.core.domains import QQ
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.presentations.basis_change import ExactBasisIsomorphism

domain = QQ()
source = FreeModule(domain, Basis(("e0", "e1"), coefficient_domain=domain))
target = FreeModule(domain, Basis(("f0", "f1"), coefficient_domain=domain))

change = ExactBasisIsomorphism.from_images(
    source,
    target,
    (target.element({0: 1, 1: 1}), target.element({0: 1, 1: -1})),
    (source.element({0: (1, 2), 1: (1, 2)}), source.element({0: (1, 2), 1: (-1, 2)})),
)
value = source.element({0: (1, 2), 1: 3})

assert change.forward(value) == target.element({0: (7, 2), 1: (-5, 2)})
assert change.inverse(change.forward(value)) == value
assert change.certificate.exact_certified
```

The constructor evaluates both composites on every declared basis vector.
Duplicate forward images produce an indexed certification failure. Structure
constants can be transported through a certified change and transported back.

## 10. Laws: proved, disproved, and inconclusive

Finite law validation returns one of three top-level reports:

- `Proved`, when the declared finite assignment domain is exhausted.
- `Disproved`, with an explicit counterexample.
- `Inconclusive`, when a bound, partial outcome, comparison gap, or sampled
  search prevents either conclusion.

For Boolean conjunction, the associativity law has three variables over a
two-element carrier, hence eight assignments:

```python
from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.structures.laws import Equation, Law
from anyalgebra.structures.operations import Operation
from anyalgebra.structures.signatures import OperationSymbol, Signature
from anyalgebra.structures.structure import StructureBuilder
from anyalgebra.structures.terms import Term, Variable
from anyalgebra.validation.validate import Proved, validate_law

bit = Sort("bit")
carrier = FiniteCarrier((0, 1), sort=bit)
product = OperationSymbol("and", (bit, bit), bit)
operation = Operation.from_table(
    product,
    ((bit, carrier),),
    tuple(((x, y), x & y) for x in carrier for y in carrier),
)
structure = (
    StructureBuilder(Signature((bit,), (product,)))
    .with_carrier(bit, carrier)
    .with_operation(product, operation)
    .freeze()
)
x, y, z = (Variable(name, bit) for name in "xyz")
tx, ty, tz = (Term.variable(v) for v in (x, y, z))
law = Law(
    "associative",
    (x, y, z),
    Equation(
        Term.apply(product, Term.apply(product, tx, ty), tz),
        Term.apply(product, tx, Term.apply(product, ty, tz)),
    ),
)

report = validate_law(structure, law)
assert isinstance(report, Proved)
assert (report.expected_assignments, report.evaluated_assignments) == (8, 8)
```

For subtraction modulo 3, the first lexicographic associativity failure occurs
at substitution indices `(0,0,1)` after two evaluations. The validator returns
that deterministic witness instead of a Boolean `False`.

Standard law factories cover associativity, commutativity, alternativity,
flexibility, Jacobi, Jordan, identities, and custom equations. Jacobi is built
from an explicit bracket, addition operation, and zero term. No vector-space
operation is inferred from notation.

Sampling has a separate report contract. A sampled pass remains
`Inconclusive`, even if the sample happens to equal the whole population unless
the exhaustive algorithm and its coverage conditions are satisfied. Witness
minimization replays earlier assignments and will not claim global minimality
if an earlier evaluation is indeterminate.

## 11. Morphisms, homomorphisms, and isomorphism checks

A `Morphism` contains one component map per sort and explicit source and target
structures. Validation checks operation preservation, relation preservation,
carrier membership, and inverse identities over the declared finite carriers.

```python
from anyalgebra.maps.core import Map, Morphism
from anyalgebra.maps.validate import Proved, validate_homomorphism, validate_inverse

# source and target are equal-looking finite structures built independently.
component = Map.from_callable(
    source.carriers[0], target.carriers[0], lambda value: value
)
forward = Morphism.from_components(
    source,
    target,
    (component,),
    preserved_operations=(source.signature.operations[0],),
    preserved_relations=(source.signature.relations[0],),
)
hom = validate_homomorphism(forward)
assert isinstance(hom, Proved)
assert (hom.operation_cases, hom.relation_cases) == (2, 2)
```

The corresponding inverse test checks four component cases and returns
`Proved`. Other tests exhibit operation counterexamples, invalid component
images, many-sorted maps with overlapping raw values, partial-operation gaps,
and inconclusive callables. A declaration that a map preserves an operation is
metadata until the validator checks it.

## 12. Commutators, associators, nuclei, centers, ideals, and derivations

The elementary analysis layer works on binary, total,
`FiniteMultilinearStructure` objects over exact `QQ` where its linear-algebra
algorithms apply.

For the test algebra

\[
e_0e_0=e_1,\qquad e_1e_0=e_1,
\]

with the other basis products zero:

```python
from anyalgebra.analysis.elementary import associator, commutator

algebra = nonassociative_fixture
e0 = algebra.module.element({0: 1})
e1 = algebra.module.element({1: 1})

assert commutator(algebra, e0, e1) == algebra.module.element({1: -1})
assert associator(algebra, e0, e0, e0) == e1
assert associator(algebra, e0 + e1, e0, e0) == algebra.module.element({1: 2})
```

The computed subspace dimensions are:

```text
left nucleus dimension   = 1, basis (-e0 + e1)
middle nucleus dimension = 1, basis (e1)
right nucleus dimension  = 1, basis (e1)
full nucleus dimension   = 0
```

For a separate associative noncommutative rank-two fixture, the nucleus has
dimension 2 and the center dimension 0. For a commutative associative fixture,
the full two-dimensional module is the center.

The derivation solver constructs the exact linear constraints for

\[
D(xy)=D(x)y+xD(y).
\]

Confirmed controls include:

```text
dual numbers over QQ: derivation dimension 1, 8 constraints, work count 128
two-dimensional zero algebra: derivation dimension 4
quaternions over QQ: derivation dimension 3
octonions over QQ: derivation dimension 14
```

Each reported derivation basis is independently substituted back into the
Leibniz equations. The ideal search deduplicates equal subspaces and reports
when its bounded classification is incomplete. The fingerprint therefore
records `ideal_classification_status` instead of pretending that a bounded
search is a theorem about all ideals.

## 13. Algebra fingerprints and basis invariance

The current fingerprint records dimension, unit status, commutativity,
associativity, nucleus and center dimensions, derivation dimension, bounded
ideal data, conventions, and algorithm metadata. It is explicitly marked
`isomorphism_rejection_only`. Equal fingerprints are not a complete
isomorphism proof.

An exact rational shear of the dual-number basis produces the same
fingerprint. A larger integration test transports the octonion structure and
checks all 512 ordered basis associators:

```python
source_fingerprint = algebra_fingerprint(source_octonions)
target_fingerprint = algebra_fingerprint(transported_octonions)

assert change.certificate.exact_certified
assert source_fingerprint == target_fingerprint
assert source_fingerprint.isomorphism_rejection_only

for i in range(8):
    for j in range(8):
        for k in range(8):
            assert change.forward(associator(source, e[i], e[j], e[k])) == (
                associator(
                    target,
                    change.forward(e[i]),
                    change.forward(e[j]),
                    change.forward(e[k]),
                )
            )
```

The test confirms basis transport of the calculation. It does not infer that
any two algebras with equal fingerprints are isomorphic.

## 14. Generic quaternion and octonion fixtures

Four named fixtures use exact multiplication tables:

- `algmul.H.v1` for quaternions.
- `algmul.Hs.v1` for split quaternions.
- `algmul.O.v1` for octonions.
- `algmul.Os.v1` for split octonions.

The names select conventions. All calculations run through generic modules,
structure constants, basis changes, and analysis functions.

The multiplication-table tests confirm the declared unit, generator squares,
quaternion orientations, octonion Fano orientation, split-octonion table, and
anticommutation of distinct imaginary generators. They exhaust all 64 ordered
quaternion basis triples and find an explicit octonion failure:

\[
(e_1e_2)e_4=e_7,\qquad e_1(e_2e_4)=-e_7,
\]

so

\[
[e_1,e_2,e_4]=(e_1e_2)e_4-e_1(e_2e_4)=2e_7.
\]

Run the witness finder:

```powershell
python examples/octonion_counterexample.py
```

Confirmed output:

```text
convention: algmul.O.v1
basis grid: all 8^3 ordered triples, lexicographic order, no sampling
first nonzero associator after 85 checks: ('e1', 'e2', 'e4')
associator: (e1*e2)*e4 - e1*(e2*e4) = 2*e7
```

The broader generic report confirmed:

| Fixture | Dimension | Associativity | Cases | Nucleus | Center | Derivations |
|---|---:|---|---:|---:|---:|---:|
| `algmul.H.v1` | 4 | Proved | 64/64 | 4 | 1 | 3 |
| `algmul.Hs.v1` | 4 | Proved | 64/64 | 4 | 1 | 3 |
| `algmul.O.v1` | 8 | Disproved | 85/512 | 1 | 1 | 14 |
| `algmul.Os.v1` | 8 | Disproved | 85/512 | 1 | 1 | 14 |

The octonion rows stop after the first decisive witness. A separate transport
test evaluates all 512 associators before and after a certified basis shear.
The first eight octonion triples happen to associate, but the sampled-prefix
report remains `Inconclusive` because 8/512 is not a proof.

## 15. Evidence, replay, and scientific claim boundaries

Calculations may be wrapped in a `CalculationContract` containing input hashes,
source anchors, convention manifests, declared bounds, and allowed outcomes.
Execution produces a typed result and a receipt. Replay distinguishes:

- an exact match.
- a fresh contract with a changed result.
- stale inputs, sources, conventions, or predecessors.
- unavailable dependencies or failed reruns.

The replay tests confirm successful, negative, and inconclusive outcomes. They
also verify that timestamps and display metadata do not change semantic
identity, while input or source changes do. Unknown serialized tags, tampered
hashes, cycles, and mutated predecessor chains fail closed.

Claim promotion is separate from calculation success. The evidence tests reject
attempts to promote a numerical run, passing test, backend agreement, or legacy
replay directly into a scientific claim. This is why the examples report their
finite domains and use phrases such as `infrastructure evidence`.

## 16. Persistence and backend agreement

Canonical JSON supports only registered, versioned semantic records. Tests
cover round trips, deterministic key ordering, exact rational payloads,
unknown versions, duplicate JSON keys, unsafe values, schema migrations, and
tamper detection.

The reference backend runs without optional mathematics packages. Differential
tests compare backend capabilities, requests, results, receipts, and artifacts.
Agreement is reported only when the same contract and semantic result are
compared. Backend agreement does not elevate the status of the mathematical or
scientific claim.

## 17. AlgMul migration tests

The legacy suite treats `AlgMul.wl` as migration evidence and a source of
feature requirements. It does not use AlgMul as a correctness oracle. Tests
cover:

- static parsing of Wolfram definitions without evaluation.
- evaluated runtime manifests when an explicit compatible kernel is available.
- dynamic factories such as `MakeProperty` and its 128 intended generated
  names.
- algebra registry and multiplication-table snapshots.
- generated-operation lifts and `MakeAlg` behavior.
- source hashes, kernel versions, dispositions, and replay boundaries.
- a clean Mathematica process adapter with bounded output and timeouts.

The licensed Wolfram 12.0 replay is intentionally opt-in. Ordinary test success
does not claim full AlgMul parity. The legacy ledger records preserve,
redesign, replace, drop, and defer decisions so later versions can close parity
family by family.

## 18. Complete test-module catalogue

The suite contains 108 test modules and 1,227 declared test functions.
Parametrization expands these to the larger executed-case count. The tables
below account for every module. Modules marked as defensive are still useful:
they establish that mathematical objects cannot be forged, silently coerced,
partially iterated past a bound, or serialized with altered semantics.

### Exact core

| Module | What it establishes |
|---|---|
| `core/test_basis.py` | Ordered bases, rank 0 and 1, duplicate labels, literal coefficient parents, immutability |
| `core/test_coerce.py` | Identity, unique and multistep coercion, parent checks, safe failure wrapping |
| `core/test_coercion_graph.py` | Persistent graphs, diamonds, parallel paths, cycles, stable path ordering |
| `core/test_coercion_map.py` | Exact map metadata, endpoint validation, callable identity boundaries |
| `core/test_common_parent.py` | Least lossless common parents, ambiguity, incomparability, cycles, lossy rejection |
| `core/test_domain_protocols.py` | Runtime domain protocols and typed construction/coercion diagnostics |
| `core/test_exact_domain_properties.py` | Bounded exhaustive `ZZ`/`QQ` normalization and exact-constructor equivalence |
| `core/test_free_module.py` | Rank boundaries, literal domain/basis ownership, structural parent equality |
| `core/test_immutability_cache_independence.py` | Frozen semantic data and cache-independent fingerprints |
| `core/test_module_arithmetic.py` | Exact sparse addition, subtraction, negation, scaling, cancellation, parent safety |
| `core/test_parent_builder.py` | Atomic, correctable builder lifecycle and complete issue reporting |
| `core/test_parent_fingerprint.py` | Reproducible canonical fingerprints across processes and hash seeds |
| `core/test_parent_mismatch.py` | Explicit whole-element transport between coexisting parents |
| `core/test_qq.py` | Canonical rational parent, exact field arithmetic, zero-division boundaries |
| `core/test_rational_kernel.py` | GCD/sign normalization, large integers, exact primitive arithmetic |
| `core/test_sorts_carriers.py` | Sorts, finite carriers, labels, unhashable values, conservative equality |
| `core/test_sparse_element.py` | Sparse canonicalization, exact coercion planning, zero uncertainty, literal parents |
| `core/test_zz.py` | Canonical integer parent and strict exact integer construction |

### General structures

| Module | What it establishes |
|---|---|
| `structures/test_callable_operation.py` | Total and partial callable operations across arities and safe result translation |
| `structures/test_deductive.py` | Ground closure, synchronous rounds, joins, cycles, bounds, replayable derivations |
| `structures/test_evaluate.py` | Parenthesized term evaluation and propagation of four outcome branches |
| `structures/test_law_ast.py` | Quantified equations, hypotheses, sort checks, immutable law syntax |
| `structures/test_outcomes.py` | Defined, undefined, failed, and indeterminate value semantics |
| `structures/test_partial_operation.py` | Partial tables, omitted versus explicit undefined cells, repeated sorts |
| `structures/test_relations.py` | Table and predicate relations, nullary/high arity, safe truth results |
| `structures/test_rewrite.py` | Ordered rewriting, matching, cycles, bounds, traces, deep substitutions |
| `structures/test_signature.py` | Arbitrary many-sorted operation and relation signatures |
| `structures/test_structure_builder.py` | Carrier and interpretation binding, freeze isolation, coexistence |
| `structures/test_symbols.py` | Nullary through high-arity symbols and notation as semantic metadata |
| `structures/test_terms.py` | Explicit syntax trees, parentheses, variables, sort-safe application |
| `structures/test_total_operation.py` | Complete operation tables, Cartesian coverage, application validation |
| `structures/test_totalize.py` | Explicit strict-bottom totalization and original-carrier embeddings |

### Multilinear algebra and tensor parents

| Module | What it establishes |
|---|---|
| `algebra/test_from_callable.py` | Compilation of one multilinear callable into exact constants |
| `algebra/test_from_structure_constants.py` | Sparse constants, arbitrary arity, basis evaluation, rank-zero behavior |
| `algebra/test_from_table.py` | Lexicographic basis tables and exact table-to-constant compilation |
| `algebra/test_tensor_product.py` | Ordered, nested tensor parents and pure tensors, with arithmetic intentionally absent |

### Exact linear algebra

| Module | What it establishes |
|---|---|
| `linear/test_exact_solve.py` | RREF, ranks, pivots, kernels, unique/dependent/inconsistent solves |
| `linear/test_matrix_space.py` | Rectangular matrices, exact products, transpose, zero shapes, work bounds |
| `linear/test_operator_recovery.py` | Closed recovery, dependencies, nonclosure, reconstruction, stateful brackets |
| `linear/test_span.py` | Unique span coordinates, dependence, outside witnesses, closure checks |

### Presentations and maps

| Module | What it establishes |
|---|---|
| `presentations/test_basis_change.py` | Exact basis isomorphisms, composite certificates, constant transport |
| `presentations/test_conversion_graph.py` | Explicit persistent conversion graphs, inverses, cycles, metadata |
| `presentations/test_convert.py` | Route planning, ambiguity, admissibility, multistep execution, round trips |
| `presentations/test_symbolic_coordinate_table.py` | Symbolic/coordinate and table/structure adapters, octonion table round trips |
| `maps/test_homomorphism.py` | Finite operation/relation preservation, inverses, isomorphism reports |
| `maps/test_map_records.py` | Generic, linear, multilinear, and many-sorted map contracts |

### Validation and witnesses

| Module | What it establishes |
|---|---|
| `validation/test_basis_reduction_gate.py` | Conditional certificates require every independent reduction hypothesis |
| `validation/test_exhaustive.py` | Exact finite proof, disproof, partial semantics, vacuity, work bounds |
| `validation/test_law_factories.py` | Associative, commutative, alternative, flexible, Jacobi, Jordan, custom laws |
| `validation/test_reports.py` | Sound report status, counts, scope, witnesses, theorem obligations |
| `validation/test_sampling.py` | Pinned schedules, sampled counterexamples, and nonproof status of passing samples |
| `validation/test_substitutions.py` | Many-sorted finite assignment enumeration and mixed-radix coordinates |
| `validation/test_witness_minimization.py` | Deterministic minimal counterexamples and limits on minimality claims |

### Algebra analysis and named fixtures

| Module | What it establishes |
|---|---|
| `analysis/test_elementary.py` | Products, commutators, associators, left/middle/right nuclei, nucleus, center |
| `analysis/test_fingerprint.py` | Derivations, bounded ideals, fingerprints, basis transport, rejection-only scope |
| `fixtures/test_composition_tables.py` | Exact `H`, `Hs`, `O`, `Os` tables, orientations, squares, anticommutation |

### Executable mathematical dossiers

| Module | What it establishes |
|---|---|
| `examples/test_generic_composition_algebras.py` | Exact reports for `H`, `Hs`, `O`, `Os`, sampling nonproof, basis invariance |
| `examples/test_operator_dossier.py` | Replay of routes, matrix parents, and operator-recovery witnesses |
| `examples/test_partial_and_deductive_release.py` | Partiality, explicit totalization, bounded closure, stable records |
| `examples/test_replay_dossier.py` | Canonical records, migrations, claim neutrality, replay and staleness |
| `examples/test_two_sorted_release.py` | Generic many-sorted calculations and evidence pipeline |
| `integration/test_coexisting_parents.py` | Parent coexistence from source and an isolated installed wheel |
| `integration/test_exact_domains_and_coercions.py` | Exact scalar grids and declared coercion graph family |
| `integration/test_required_linear_examples.py` | Presentation, rational-algebra, and operator-recovery programs |
| `integration/test_required_structure_examples.py` | Partial magma, two-sorted structure, and deductive examples |
| `integration/test_v0_0_exit_demo.py` | Five semantic object round trips and fail-closed record parsing |
| `integration/test_validation_basis_invariance.py` | Exact basis transport of quaternion and octonion analyses |

### Evidence and persistence

| Module | What it establishes |
|---|---|
| `evidence/test_claim_promotion.py` | No automatic promotion from software results to scientific claims |
| `evidence/test_contracts.py` | Calculation contracts, outcome policies, hashes, bounds, sources, conventions |
| `evidence/test_execution_lifecycle.py` | Deterministic execution state transitions |
| `evidence/test_replay.py` | Exact replay, mismatch, stale, unavailable, predecessor chains |
| `evidence/test_run_receipt.py` | Typed outcomes and canonical result receipts |
| `evidence/test_sources_conventions.py` | Source anchors and convention manifests |
| `evidence/test_staleness.py` | Direct and transitive staleness classification |
| `persistence/test_canonical_json.py` | Safe deterministic JSON and rejection of unsafe data |
| `persistence/test_migrations.py` | Explicit directional version migrations |
| `persistence/test_record_registry.py` | Closed type/version registry and exact reconstruction |

### Backends, legacy migration, and engineering proof

| Module | What it establishes |
|---|---|
| `backends/test_backend_protocol.py` | Capability and execution protocol boundaries |
| `backends/test_differential.py` | Contract-bound backend comparison and artifacts |
| `backends/test_reference_loading.py` | Dependency-free reference backend and optional capability absence |
| `legacy/test_algmul_generated_lifts.py` | Generic replacements for runtime-generated AlgMul families |
| `legacy/test_algmul_makealg.py` | Captured `MakeAlg` behaviors and deterministic recipe validation |
| `legacy/test_algmul_parity.py` | Pinned source identity and parity boundary |
| `legacy/test_algmul_runtime_capture.py` | Evaluated manifest protocol and required-symbol states |
| `legacy/test_algmul_static.py` | Nonexecuting Wolfram parser and static feature catalogue |
| `legacy/test_disposition_complete.py` | Every owned legacy record has a migration disposition |
| `legacy/test_generated_property_family.py` | Corrected replacements for the 128 intended generated properties |
| `legacy/test_manifest_models.py` | Immutable, canonical, cross-linked legacy evidence records |
| `legacy/test_mathematica_adapter.py` | Clean kernel command, version probe, timeout and output bounds |
| `docs/test_public_examples.py` | Public snippets remain executable |
| `performance/test_catastrophic_regressions.py` | Exact work bounds prevent catastrophic expansion |
| `packaging/test_release_artifacts.py` | Wheel and source distribution integrity |
| `test_clean_import.py` | Isolated import without optional mathematics packages |
| `test_package_metadata.py` | Version and intentionally small root namespace |
| `tools/test_adr_gate.py` | Accepted architecture decisions remain pinned |
| `tools/test_canonical_coverage.py` | Exact canonical source and branch coverage gate |
| `tools/test_contract_snapshot.py` | Frozen v0.0 API and test contract IDs |
| `tools/test_v0_0_release_readiness.py` | Release documents, artifacts, and semantic demo remain coherent |
| `tools/test_v0_0_traceability.py` | Feature, API, test, and task traceability |
| `tools/test_validate_project_contract.py` | Project-process registry shape and references |

### Coverage-hardening modules

These eight modules close defensive paths throughout the same production APIs.
They mainly test corrupt internal records, hostile iterators, impossible cache
states, malformed parser input, and sanitized failure translation:

- `coverage/test_v001_core_guards.py`.
- `coverage/test_v001_legacy_guards.py`.
- `coverage/test_v001_map_validation_guards.py`.
- `coverage/test_v001_misc_guards.py`.
- `coverage/test_v001_multilinear_guards.py`.
- `coverage/test_v001_presentation_guards.py`.
- `coverage/test_v001_semantic_guards.py`.
- `coverage/test_v001_structure_guards.py`.

They are less useful as a first tutorial, but they matter for arbitrary algebra
work. A malformed coefficient, incomplete record, hostile equality method, or
partially consumed iterator must not be mistaken for a proof, a zero, a valid
map, or a complete classification.

## 19. What the tests do not yet demonstrate

The current suite does not yet demonstrate:

- arbitrary-dimensional arithmetic over dedicated public `C`, `H`, `O`, split
  composition-algebra, Clifford, Lie, Jordan, or exceptional-algebra classes.
- multiplication of general tensor-product elements or matrix algebras over
  arbitrary nonassociative coefficient parents.
- `GL(n,A)`, `SL(n,A)`, `SO(n,A)`, or `SU(n,A)` constructors.
- stable constructors for the exceptional Lie algebras or their invariants.
- manifolds, tensor calculus, gauge fields, Schouten-kit integration, or
  metric-affine gravity.
- the proposed brane, A-theory, string-duality, or unified-field calculations.
- complete behavioral parity with AlgMul.

Those are roadmap targets. The tested foundation is already useful for exact
finite tables, arbitrary signatures, partiality, presentation control, bounded
law validation, linear recovery, elementary finite-dimensional analysis, and
evidence-bearing calculations. Later versions can build specialized objects on
these contracts without weakening parent identity, exactness, partiality, or
proof-status distinctions.

## 20. Suggested learning path

1. Run `examples/exact_domains.py` to learn exact parents and coercion.
2. Run `examples/coexisting_parents.py` to understand literal parent identity.
3. Run `examples/partial_magma.py` and `examples/two_sorted_structure.py`.
4. Run `examples/deductive_system.py` for replayable closure traces.
5. Run `examples/rational_algebra.py` for tables, generic products, and basis
   transport.
6. Run `examples/operator_recovery.py` for success, dependence, and nonclosure.
7. Run `examples/octonion_counterexample.py` for a compact exact witness.
8. Run `examples/generic_composition_algebras.py` for fingerprints, complete
   basis-grid reports, and the distinction between proof and sampling.
9. Read the validation and evidence tests before using a computational result
   in a research argument.

This order follows the package's design: define exact parents, construct a
structure, preserve its presentation and partiality, perform a bounded
calculation, and attach only the claim strength supported by the evidence.
