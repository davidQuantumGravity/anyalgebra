# AnyAlgebra v0.1 public API contract

Status: implemented in the locally complete 0.1.0 source milestone. V01-060 passed the final readiness decision.
The concrete callables below are verified module
APIs. The root package continues to export only `__version__`.

The v0.1 surface adds exact sparse multilinear evaluation, a bounded
finite-carrier census, constructive comparison and capability-aware analysis,
and a deterministic content-addressed atlas. It does not stabilize named Lie,
Clifford, Jordan, exceptional, geometry, or physics interfaces.

## Shared records and errors

Public values are immutable and retain literal parent ownership. Ordered
collections use tuples. Serialized mappings use allow-listed, canonically
ordered records. A finite-carrier operation table is distinct from a
finite-basis vector algebra over `QQ` or another coefficient domain.
Canonical bytes are produced only by the declared serializers.

The implemented result types include `ReferenceEnumeration`, `CanonicalLabel`,
the `ComparisonOutcome` union, `AnalysisRecord`, `AtlasBuildResult`,
`LoadedAtlas`, and `AtlasQueryResult`. Bounded work remains explicit. It never
becomes a complete enumeration or a negative isomorphism result.

The implemented public error types for this surface are
`MultilinearEvaluationError`, `CensusSpecError`,
`ReferenceEnumerationError`, `CanonicalLabelError`,
`IsomorphismOutcomeError`, `AnalysisRecordError`, `AtlasBuildError`, and
`AtlasQueryError`. Their stable fields and bounded diagnostics are preferable
to platform-dependent exception chains.

<!-- stable-surface:begin -->

## Stable v0.1 surface

### `api.algebra.evaluate`

`evaluate_multilinear(structure, *elements, operation=None, graph=None)`

Module: `anyalgebra.algebra.multilinear`.

Returns: an exact `SparseElement` in the selected operation's output parent.
The function validates arity, literal parents, basis compatibility, operation
selection, and every requested coercion route before coefficient expansion.

Errors: `MultilinearEvaluationError` reports the first invalid arity, operand,
parent, basis, operation, coercion, or output-parent condition. The function
does not infer multiplication, basis maps, or parenthesization.

Owned by: `fe.v01.public_algebra_evaluation`.

Evidence: V01-003 through V01-010 and
`tests/algebra/test_multilinear_evaluate_elements.py`.

### `api.census.spec`

`CensusSpec.create(*, carrier_size, arity, constraints, equivalence, ordering, bounds, convention_refs=())`

Module: `anyalgebra.census.spec`.

Returns: an immutable content-addressed `CensusSpec`. It pins the carrier,
arity, constraints, equivalence group, ordering, bounds, and convention
references. Constraints and policies are allow-listed data records, not
serialized callables.

Errors: `CensusSpecError` localizes malformed, inconsistent, unsafe, or
oversized fields. Fixed-cost and aggregate bounds are checked before consuming
candidate iterables.

Owned by: `fe.v01.finite_corpus`.

Evidence: V01-011 through V01-015 and
`tests/census/test_spec_serialization.py`.

### `api.census.enumerate`

`enumerate_reference(spec)`

Module: `anyalgebra.census.reference`.

Returns: a `ReferenceEnumeration` containing the deterministic candidate
prefix, exact examined and accepted counts, work and memory charges, and either
complete status or an exact remaining frontier.

Errors: `ReferenceEnumerationError` rejects a non-`CensusSpec` value or a
resource contract that cannot be honored. Reaching a declared ceiling produces
an incomplete result rather than a false completeness claim.

Owned by: `fe.v01.enumeration`.

Evidence: V01-016 through V01-022 and
`tests/census/test_reference_enumerator.py`.

### `api.census.canonicalize`

`canonicalize_operation_table(core, source_outputs, equivalence)`

Module: `anyalgebra.census.canonical`.

Returns: a `CanonicalLabel` with the least table image under the declared
relabeling group, its semantic identifier, a source-to-canonical permutation,
and the exact orbit work count.

Errors: `CanonicalLabelError` rejects malformed tables, mismatched policy
ownership, empty candidate domains, or certificate failure. The returned
transport is replayed against every operation cell in the test oracle.

Owned by: `fe.v01.canonical_isomorphism`.

Evidence: V01-023 through V01-030 and
`tests/census/test_canonical_certificate.py`.

### `api.census.compare`

`compare_isomorphism(core, left_outputs, right_outputs, equivalence, *, max_permutations=None)`

Module: `anyalgebra.census.isomorphism`.

Returns: one `ComparisonOutcome`: `Isomorphic` with a checked bijection,
`InvariantNonIsomorphic` with an exact mismatch, `ExhaustiveNonIsomorphic` with
a completed no-map receipt, or `Inconclusive` with a bounded frontier.
Fingerprint equality never constructs an isomorphism.

Errors: `IsomorphismOutcomeError` rejects malformed source data or internally
inconsistent evidence. An exhausted caller bound returns `Inconclusive`.

Owned by: `fe.v01.canonical_isomorphism`.

Evidence: V01-025 through V01-030 and
`tests/census/test_isomorphism_negative.py`.

### `api.census.analyze`

`assemble_finite_carrier_analysis(core, outputs, equivalence, *, nilpotence_zero=None, max_nilpotence_index=None, subobject_options=None)`

Module: `anyalgebra.census.analysis_records`.

Returns: an `AnalysisRecord` whose fields state `complete`, `bounded`,
`rejection-only`, or `unsupported`. The record can include finite laws,
identity and zero profiles, nuclei, commutants, centers, substructures,
congruences, bounded nilpotence, and rejection fingerprints. Linear ideals and
derivations require the separate module-backed adapter.

Errors: `AnalysisRecordError` identifies malformed tables, incompatible
options, unsupported linear requests, or inconsistent nested evidence.

Owned by: `fe.v01.fingerprint_atlas`.

Evidence: V01-031 through V01-046 and
`tests/census/test_analysis_records.py`.

### `api.atlas.build`

`build_finite_algebra_atlas(specs, *, output_directory)`

Module: `anyalgebra.atlas.build`.

Returns: an `AtlasBuildResult`. The builder writes canonical census, orbit,
certificate, analysis, receipt, and manifest records in a private sibling
directory. It publishes the new target with one rename only after internal
verification succeeds.

Errors: `AtlasBuildError` rejects malformed or oversized spec collections,
nonfresh or unsafe paths, incomplete classification promotion, and atomic
publication failures. A failed build leaves no partial target.

Owned by: `fe.v01.atlas_evidence`.

Evidence: V01-043 through V01-048 and
`tests/atlas/test_evidence_replay.py`.

### `api.atlas.query`

`load_finite_algebra_atlas(directory); query_atlas(loaded, *, carrier_size=None, arity=None, canonical_id=None, laws=(), invariant_fields=())`

Module: `anyalgebra.atlas.query`.

Returns: `load_finite_algebra_atlas` produces a `LoadedAtlas` only after
canonical-byte, schema, exact-file-set, content-address, and role-link checks.
`query_atlas` returns an `AtlasQueryResult` ordered by canonical identifier.
Filters are allow-listed data values and never executable predicates.

Errors: `AtlasQueryError` rejects corrupt, partial, stale, unknown-version,
path-escaping, symlinked, or semantically inconsistent artifacts and malformed
filters. Loading and querying never repair an atlas.

Owned by: `fe.v01.atlas_evidence`.

Evidence: V01-047 through V01-052 and
`tests/atlas/test_evidence_replay.py`.

<!-- stable-surface:end -->

## Positive examples

The executable [finite algebra census and atlas manual](../guides/finite-algebra-census.md)
contains complete imports and asserted results. The central evaluation call is:

```python
product = evaluate_multilinear(structure, i, j)
assert product == k
```

The reference census uses `CensusSpec.create`, `enumerate_reference`,
`canonicalize_operation_table`, and `compare_isomorphism` on literal flat
operation tables. The reference atlas example builds all 19,683 binary tables
on a labeled carrier of size three, retains the 729 commutative tables in 129
certified relabeling orbits, then verifies and queries the published directory.

## Negative examples

These conditions fail before or at the declared boundary:

- the evaluator receives the wrong arity or an equal-looking element from a
  different literal parent;
- a census record contains a callable, unknown schema field, unsafe convention
  reference, or resource declaration beyond the hard ceilings;
- a comparison reaches `max_permutations` before exhausting the group, which
  returns `Inconclusive` instead of nonisomorphism;
- an atlas contains a changed digest, missing object, extra object, symlink,
  path-shaped content identifier, path traversal attempt, or unknown schema
  version.

## Boundary examples

- The empty carrier with positive operation arity has one total operation, the
  empty function, when its declared policies allow it.
- A nullary operation on the empty carrier is unsatisfiable because it cannot
  choose an output.
- A zero candidate ceiling retains a nonempty frontier if any candidate would
  otherwise require inspection.
- Equal fingerprints can reject a pair when they differ. Equal fingerprints
  do not provide a positive map.
- Finite-carrier nuclei and centers are subsets. The module-backed adapter
  computes linear subspaces under separate hypotheses.

## Determinism contract

Basis order, flattened table order, relabeling order, diagnostic order, query
order, and canonical JSON bytes are explicit. Locale, process identifiers,
temporary paths, hash iteration, and wall-clock time do not enter semantic
identities. V01-058 reproduced the reference atlas from separate wheel and
sdist installations with identical tree bytes and ordered canonical IDs.

## Stability boundary

Only the eight API IDs inside the stable-surface markers belong to v0.1. The
command-line atlas interface is tested and shipped, but its long-term stability
is not broader than these underlying records. Named composition families and
the experimental modules remain outside the v0.1 stable API.

## Task ownership

V01-003 through V01-048 implemented the evaluator, census, analysis, and atlas.
V01-049 through V01-055 supplied property, mutation, resource, security,
documentation, AlgMul handoff, and traceability evidence. V01-056 through
V01-059 supplied full-suite, artifact, reproduction, and documentation gates.
V01-060 marked the bounded local source milestone complete.
