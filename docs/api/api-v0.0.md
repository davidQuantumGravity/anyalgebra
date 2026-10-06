# AnyAlgebra v0.0 public API

Status: Proposed  
Last updated: 2026-07-22

## 1. API policy

This is the bounded public API for v0.0. It is intentionally smaller than the
complete inventory in the implementation plan. Specialized algebra families
must be expressible as user-defined fixtures, but their convenience constructors
are not public until later releases.

The illustrative signatures below use Python typing syntax. Exact names may
change before implementation only through a documented spec and registry update.

Common rules:

- semantic objects are immutable and parent aware;
- constructors validate eagerly unless named `builder`;
- ambiguous coercions and conversions never choose silently;
- potentially partial operations return tagged outcomes;
- `validate_*` returns reports/witnesses, not unexplained booleans;
- every stable record implements safe `to_record()` and registered
  `from_record()` serialization;
- options objects are immutable and included in calculation receipts.

## 2. Domains and coercions

### Classes

| Name | Responsibility |
|---|---|
| `Domain[T]` | Parent for scalar values; owns normalization and exact arithmetic capabilities |
| `IntegerDomain` | Canonical exact integer domain |
| `RationalDomain` | Canonical exact rational domain |
| `DomainElement[T]` | Immutable normalized scalar with a domain parent |
| `CoercionMap[S,T]` | Typed conversion with exactness, injectivity, assumptions, and cost metadata |
| `CoercionGraph` | Immutable directed set of coercions and deterministic path resolution |
| `CoercionPolicy` | Explicit policy for common-parent and approximation choices |
| `CoercionPlan` | Resolved path(s), target, guarantees, and diagnostics |

Singleton constructors `ZZ()` and `QQ()` return canonical domain parents.

### Functions and methods

```python
domain.element(value) -> DomainElement
domain.normalize(value) -> canonical_value
coerce(value, target, *, graph, policy=None) -> DomainElement
common_parent(values, *, graph, policy=None) -> CoercionPlan
CoercionGraph.with_map(map) -> CoercionGraph
CoercionGraph.plan(source, target, *, policy=None) -> CoercionPlan
compare(left, right, *, map=None, graph=None) -> ComparisonResult
```

`ComparisonResult` is `Equal`, `Unequal(witness)`, or
`ComparisonIndeterminate(reason)`.

## 3. Parents, sorts, carriers, bases, and elements

| Name | Responsibility |
|---|---|
| `Parent` | Immutable semantic owner and fingerprint root |
| `ParentBuilder` | Staged construction followed by validated `freeze()` |
| `Sort` | Named sort in a many-sorted signature |
| `FiniteCarrier` | Ordered finite collection for one sort |
| `Basis` | Ordered, named basis over a coefficient domain |
| `FreeModule` | Finite free module parent |
| `Element` | Base protocol for immutable parented values |
| `SparseElement` | Canonical sparse coordinate element in a finite free module |

```python
FiniteCarrier(items, *, sort, labels=None) -> FiniteCarrier
Basis(labels, *, coefficient_domain) -> Basis
FreeModule(domain, basis, *, name=None) -> FreeModule
module.element(coordinates) -> SparseElement
module.zero() -> SparseElement
element.parent -> Parent
element.coordinates() -> Mapping[int, DomainElement]
parent.fingerprint() -> SemanticHash
builder.freeze() -> Parent
```

## 4. Signatures, terms, and laws

| Name | Responsibility |
|---|---|
| `OperationSymbol` | Name, input sorts, output sort, arity, and notation only |
| `RelationSymbol` | Name and ordered input sorts |
| `Signature` | Immutable collection of sort/operation/relation symbols |
| `Variable` | Typed variable |
| `Term` | Typed operation tree preserving parentheses |
| `Equation` | Pair of same-sort terms |
| `Law` | Quantified equation/relation plus hypotheses and partial semantics |
| `LawSet` | Named immutable collection of laws |
| `ValidationOptions` | Bounds, exactness route, ordering, and witness policy |
| `ValidationReport` | `Proved`, `Disproved`, or `Inconclusive` with coverage |

```python
OperationSymbol(name, inputs, output, *, notation=None) -> OperationSymbol
RelationSymbol(name, inputs) -> RelationSymbol
Signature(sorts, operations=(), relations=()) -> Signature
Term.variable(variable) -> Term
Term.apply(symbol, *arguments) -> Term
term.substitute(mapping) -> Term
Law(name, variables, conclusion, *, hypotheses=(), partial_semantics="strong")
validate_law(structure, law, *, options) -> ValidationReport
validate_laws(structure, laws, *, options) -> tuple[ValidationReport, ...]
```

## 5. Outcomes and errors

### Evaluation outcomes

```python
Defined(value)
Undefined(reason, witness=None)
Indeterminate(reason, bounds=None)
Failed(error, stage=None)
```

The closed `EvaluationOutcome[T]` union is serializable and supports explicit
pattern matching. `unwrap_defined(outcome)` is a convenience that returns the
value or raises the matching typed exception.

### Required exceptions

| Name | Used for |
|---|---|
| `AnyAlgebraError` | Root package exception |
| `ParentMismatchError` | No declared common parent |
| `CoercionAmbiguityError` | More than one valid coercion plan |
| `ConversionAmbiguityError` | More than one unselected presentation route |
| `MalformedDefinitionError` | Invalid signature, table, parent, or coordinates |
| `UnsupportedCapability` | Backend or object lacks the requested method |
| `NonClosureError` | Operation leaves a requested span/substructure |
| `NonUniqueDecompositionError` | Coordinates cannot be recovered uniquely |
| `SchemaError` | Unknown, unsafe, or malformed serialized record |
| `EvidenceError` | Invalid receipt transition or evidence promotion |

Mathematical undefinedness is not represented by an exception.

## 6. Operations, relations, and general structures

| Name | Responsibility |
|---|---|
| `Operation` | Typed total operation backed by a table or callable |
| `PartialOperation` | Typed operation returning `EvaluationOutcome` |
| `Relation` | Typed predicate returning a relation result |
| `Structure` | General many-sorted model of a signature |
| `StructureBuilder` | Registers carriers and interpretations, then freezes |
| `FiniteRelationalStructure` | Finite carriers with operations and relations |
| `PartialMagma` | One-sorted binary partial-operation convenience view |
| `ReductionRule` | Typed directed term/value rewrite |
| `ReductionSystem` | Rules plus bounded reachability/normalization |
| `InferenceRule` | Premises-to-conclusion relation over judgments |
| `DeductiveSystem` | Finite/bounded closure over inference rules |

```python
Operation.from_table(symbol, carriers, table) -> Operation
Operation.from_callable(symbol, carriers, function) -> Operation
PartialOperation.from_table(symbol, carriers, table, *, undefined_marker) -> PartialOperation
Relation.from_tuples(symbol, carriers, tuples) -> Relation
StructureBuilder(signature).with_carrier(sort, carrier) -> StructureBuilder
builder.with_operation(symbol, operation) -> StructureBuilder
builder.with_relation(symbol, relation) -> StructureBuilder
builder.freeze() -> Structure
structure.evaluate(term, environment) -> EvaluationOutcome
structure.apply(symbol, *arguments) -> EvaluationOutcome
structure.query(symbol, *arguments) -> RelationResult
reduction.reachable(source, target, *, bounds) -> SearchResult
reduction.normal_forms(value, *, bounds) -> SearchResult
deductive.closure(premises, *, bounds) -> SearchResult
totalize(partial_operation, *, bottom) -> Totalization
```

## 7. Finite multilinear structures

| Name | Responsibility |
|---|---|
| `MultilinearOperation` | An operation extended multilinearly from basis data |
| `FiniteMultilinearStructure` | Free-module carrier with one or more multilinear operations |
| `Algebra` | Convenience view for one bilinear multiplication; no associativity implied |
| `StructureConstants` | Canonical sparse tensor for a named operation |
| `GenericElement` | Symbolic linear combination over a basis and coefficient ring |

```python
FiniteMultilinearStructure.from_tables(module, operations, *, name=None)
FiniteMultilinearStructure.from_structure_constants(module, constants, *, name=None)
FiniteMultilinearStructure.from_callables(module, operations, *, name=None)
Algebra.from_table(domain, basis, table, *, unit=None, name=None)
algebra.multiply(left, right) -> SparseElement
algebra.basis_product(i, j) -> SparseElement
algebra.generic_element(prefix="x", *, coefficient_domain=None) -> GenericElement
structure.structure_constants(operation) -> StructureConstants
```

Table construction validates dimensions, coefficient parents, declared units,
and completeness. A partial multilinear structure uses `PartialOperation`; it is
not encoded by absent coefficients.

## 8. Maps and presentations

### Map objects

| Name | Responsibility |
|---|---|
| `Map` | Typed source/target mapping with declared guarantees |
| `LinearMap` | Exact linear map with coordinate/operator views |
| `MultilinearMap` | Typed multilinear map |
| `Morphism` | Map plus preserved operations/relations and validation report |
| `Isomorphism` | Invertible morphism with checked inverse |

### Presentation objects

| Name | Responsibility |
|---|---|
| `Presentation` | Base typed representation of semantic objects |
| `SymbolicPresentation` | Linear combinations of named basis elements |
| `CoordinatePresentation` | Dense/sparse coordinate vectors |
| `TablePresentation` | Finite operation/relation tables |
| `StructureConstantsPresentation` | Sparse tensors for multilinear operations |
| `OperatorPresentation` | Matrices/linear maps acting on a module |
| `Conversion` | Explicit typed map between presentation kinds |
| `ConversionGraph` | Immutable conversion routes and ambiguity detection |

```python
Conversion(source_kind, target_kind, forward, *, inverse=None, guarantees=())
ConversionGraph.with_conversion(conversion) -> ConversionGraph
graph.plan(source_kind, target_kind, *, route=None) -> ConversionPlan
convert(value, target_kind, *, graph, route=None) -> ConvertedValue
coordinates(element, basis=None) -> SparseElement
symbolic(element, basis=None) -> SymbolicExpression
change_basis(element_or_structure, isomorphism) -> ConvertedValue
```

Conversion plans expose intermediate representations, assumptions, exactness,
and whether the round trip is certified.

## 9. Exact linear algebra and operators

| Name | Responsibility |
|---|---|
| `VectorSpace` | Finite free module over a field-like exact domain |
| `MatrixSpace` | Matrices whose entries share a declared parent |
| `Matrix` | Immutable typed rectangular matrix |
| `SpanResult` | Unique, nonunique, outside-span, or failed decomposition |
| `ClosureReport` | Closed or smallest nonclosure witness |
| `OperatorFamily` | Ordered linear operators on one module |

```python
MatrixSpace(rows, columns, entry_parent) -> MatrixSpace
matrix_space.element(entries) -> Matrix
identity_matrix(module_or_size, *, entry_parent) -> Matrix
span_decompose(vector, generators) -> SpanResult
span_basis(generators) -> tuple[Element, ...]
closure(generators, operation, *, bounds=None) -> ClosureReport
operator_from_linear_map(map, *, source_basis=None, target_basis=None) -> Matrix
linear_map_from_operator(matrix, source, target) -> LinearMap
structure_constants_from_operators(operators, bracket, *, basis=None) -> RecoveryReport
```

`RecoveryReport` includes rank, dependencies, every closure witness, recovered
constants when unique, and a reconstructed-operator check.

## 10. Elementary algebra analysis

```python
commutator(algebra, x, y) -> Element
associator(algebra, x, y, z) -> Element
left_nucleus(algebra, *, options) -> SubspaceReport
middle_nucleus(algebra, *, options) -> SubspaceReport
right_nucleus(algebra, *, options) -> SubspaceReport
nucleus(algebra, *, options) -> SubspaceReport
center(algebra, *, options) -> SubspaceReport
ideals(algebra, *, options) -> SearchResult
derivation_algebra(algebra, *, options) -> LinearSolutionReport
validate_associative(algebra, *, options) -> ValidationReport
validate_commutative(algebra, *, options) -> ValidationReport
validate_alternative(algebra, *, options) -> ValidationReport
algebra_fingerprint(algebra, *, options) -> AlgebraFingerprint
```

The v0.0 fingerprint contains explicitly labeled computable data: dimension,
unit status, commutativity/associativity reports, nucleus/center dimensions,
annihilator dimensions, ideal data within bounds, and derivation dimension. It
is an isomorphism rejection tool, not a complete invariant.

## 11. Evidence and serialization

| Name | Responsibility |
|---|---|
| `SemanticHash` | Algorithm-tagged content hash |
| `SourceAnchor` | Exact source location and content identity |
| `ConventionManifest` | Named basis/sign/normalization contract |
| `CalculationContract` | Immutable pre-execution calculation specification |
| `ResultReceipt` | Immutable execution/outcome/evidence artifact |
| `ArtifactRef` | Media type, size, location, and hash |
| `ClaimRecord` | Claim-to-source-to-evidence links |
| `SerializerRegistry` | Allow-listed schema codecs and migrations |

```python
semantic_hash(value) -> SemanticHash
to_record(value, *, registry) -> Mapping[str, JSONValue]
from_record(record, *, registry) -> object
canonical_json(value, *, registry) -> bytes
write_artifact(value, destination, *, media_type) -> ArtifactRef
CalculationContract.create(...) -> CalculationContract
run_calculation(contract, callable, *, environment=None) -> ResultReceipt
verify_replay(receipt, *, resolver, environment=None) -> ReplayReport
promote_evidence(receipt, supporting_receipts, target_tier) -> ResultReceipt
```

`promote_evidence` validates the evidence rule and creates a new receipt; it does
not mutate the original.

## 12. Backend interfaces

```python
BackendCapabilities(name, version, operations, domains, exactness, limits)
backend.capabilities() -> BackendCapabilities
backend.supports(request) -> CapabilityResult
backend.execute(request, *, options) -> EvaluationOutcome
BackendRegistry.with_backend(backend) -> BackendRegistry
select_backend(request, *, registry, policy=None) -> BackendPlan
```

The package ships an exact `ReferenceBackend`. Optional adapters are separate
install extras and expose neutral values/results.

## 13. AlgMul audit API

The audit tooling may live under `anyalgebra.legacy.algmul`, but its records are
stable because parity is a release gate:

| Name | Responsibility |
|---|---|
| `AlgMulDefinitionRecord` | One literal or runtime-generated legacy definition |
| `AlgMulSurfaceManifest` | Source/kernel/load/registry/behavior/disposition inventory |
| `AlgMulAudit` | Static capture plus clean-kernel evaluated capture |
| `ParityDisposition` | Preserve exactly/conceptually, redesign, replace, drop, or defer |

```python
AlgMulAudit.capture_static(source) -> AlgMulSurfaceManifest
AlgMulAudit.capture_runtime(source, kernel) -> AlgMulSurfaceManifest
AlgMulAudit.exercise_generators(manifest, cases) -> AlgMulSurfaceManifest
manifest.assign_disposition(record_id, disposition, *, api_ids, test_ids, note)
manifest.validate_complete(*, owned_partitions) -> ValidationReport
```

Runtime factories such as `MakeProperty` are represented both as one factory
record and as the concrete symbols it creates. See
[algmul-parity.md](../legacy/algmul-parity.md).

## 14. Convenience operator policy

Python `+`, `-`, scalar `*`, algebra `*`, `@`, indexing, and powers may delegate
to the typed methods only when their meaning is unique for the parent. There is
no universal meaning assigned from syntax alone. Named methods remain the
normative route in examples and evidence receipts.

## 15. Deferred public families

The following are deliberately not v0.0 public names: specialized composition
algebra constructors, Cayley-Dickson towers, Clifford/gamma/spinor classes,
Lie/root/weight/representation classes, Jordan/Freudenthal/magic-square classes,
`GL`/`SL`/`SO`/`SU` constructors, invariant/orbit engines, manifolds/bundles,
Schouten geometry, variational tools, superalgebras, branes, and publication
campaign runners. Their future implementations must use, not bypass, the v0.0
parents, outcomes, conventions, evidence, and backend boundaries.

