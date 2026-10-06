# AlgMul parity and supersession contract

Status: v0.0-owned generic-surface ledger complete; v0.1-owned finite census,
evaluation, and composition controls reconciled; specialized families remain
deferred
Legacy source: `${ANYALGEBRA_ALGMUL_SOURCE}`
Source SHA-256: `3A13C9F47092B5B9814AA08873FF4F1DE61D4635DF68FC7F3A039A7DA5B0D00D`  
Bootstrap evaluation: Wolfram Language 12.0.0 for Windows, 2026-07-22

Set `ANYALGEBRA_ALGMUL_SOURCE` to the external `AlgMul.wl` file before an
explicit legacy capture or replay. The committed reference is symbolic so the
repository does not disclose or depend on one user's home directory.

## 1. Supersession rule

AnyAlgebra must supersede every sound observable AlgMul capability, including
behavior created dynamically while the package loads or while a user registers
an algebra. Supersession does not mean copying every Mathematica symbol. It means
that each legacy behavior receives one explicit disposition:

- `preserve-exactly`: externally meaningful behavior is reproduced;
- `preserve-concept`: the mathematical workflow remains, with a typed API;
- `redesign`: the capability remains but its interaction model changes;
- `replace`: a mathematically corrected or more general mechanism supersedes it;
- `drop`: confirmed defect or behavior with no valid use;
- `defer`: required capability is assigned to a later version and test owner.

No static definition, generated symbol, registry route, meaningful overload,
side effect, or intended stub is complete while its manifest entry is
unclassified.

## 2. Why static source search is insufficient

AlgMul is partly a program that writes or registers algebra programs.

### `MakeChar`

`MakeChar[chars, name]` mutates `alg["algs"]`, stores the basis in
`alg["chars"][name]`, strips trailing digits from the name, constructs a
multiplication symbol with `ToExpression[base <> "Mul"]`, and stores that
function in `alg["Mul"][name]`. Therefore `O2` and `O3` are registered at
runtime as distinct bases sharing `OMul`; their multiplication routing is not a
literal `O2Mul` definition.

At load time, `MakeCharSubVars[]` constructs 18 registrations:

```text
O, H, C; O2, H2, C2; O3, H3, C3;
Os, Hs, Cs; Os2, Hs2, Cs2; Os3, Hs3, Cs3.
```

Two additional calls register `CxH` and `D`, giving 20 evaluated entries in the
current registry. This is observable functionality even though the individual
registrations are not written as assignments in the source.

### `MakeTMul`

`MakeTMul` constructs a tensor-product basis, chooses a distinct second-factor
name when both inputs share a base, registers a new algebra name, and derives a
sparse structure tensor. The mutation and naming policy are runtime behaviors;
the new package must replace them with immutable tensor-product parents and
explicit factor identities.

### `MakeAlg` and structure registration

`MakeAlg` stores arbitrary characters and a structure-dot tensor, enabling
matrix-derived Lie structure data and other user-defined algebras. This is a
central arbitrary-structure capability, not merely a list of built-in exceptional
algebras.

### `MakeProperty`

The source intends `MakeProperty[name, arity, kernel]` to define sixteen public
variants for each named property:

```text
A, AT, AM, AMT
IsA, IsAT, IsAM, IsAMT
N, NT, NM, NMT
IsN, IsNT, IsNM, IsNMT
```

The prefixes distinguish algebraic/symbolic versus numerical evaluation and
scalar, tensor, matrix, and matrix-over-tensor containers. Eight load-time calls
request Commutative, Associative, Alternative, Flexible, PowerAssociative1,
PowerAssociative2, JIdentity, and Jacobi families. The intended surface is thus
128 generated names, none of which can be catalogued by searching only for
literal function definitions.

The pinned source has a decisive evaluated/static discrepancy: under Wolfram
12.0, the source loads 470 package-context names and registers 20 algebras, but
`MakeProperty` itself and all 128 requested generated property symbols are
absent. The bootstrap evaluator records
`completed-with-missing-required-symbols`; manually calling `MakeProperty`
remains unevaluated. Static inspection identifies malformed syntax in the
three-argument numerical-matrix branch near source line 1798, and a likely wrong
kernel reference in the two-argument `AMT` branch near line 1774. The intended
factory surface remains a parity obligation, but the broken evaluated behavior
must not be preserved.

This finding also means that code after the malformed expression may load while
one complete factory definition is silently absent. A successful `Get` return is
not enough to declare the package fully initialized.

## 3. Evaluated audit artifacts

- Audit script: [audit-algmul.wls](scripts/audit-algmul.wls)
- Bootstrap manifest:
  [algmul-evaluated-surface.json](generated/algmul-evaluated-surface.json)

The bootstrap manifest records the source hash, kernel version, contexts,
package/global names, down/own/up/sub-value counts, the 20-entry algebra registry,
the expected 128-name property surface, and an explicit generator exercise.

The v0.0 static and evaluated manifests now record messages, representative
behavior, source ranges, factory/generator routes, dispositions, API links, and
test links for every v0.0-owned record. The independent replacement tests use
documented mathematical definitions where possible; neither manifest makes
AlgMul output a correctness oracle. A live Wolfram replay is intentionally
opt-in because it requires a licensed compatible Wolfram installation.

## 4. Capability catalogue and disposition

| Partition | Representative legacy surface | v0.0 disposition | Typed replacement |
|---|---|---|---|
| Registry and bases | `alg`, `MakeChar*`, `AddAlgName`, `AlgToChars`, `CharsToAlg`, defaults | Redesign | immutable `Parent`, `Basis`, explicit registry instances |
| Arbitrary tables/structures | `MakeAlg`, `MakeStructDot`, `StructMul`, structure tensors | Preserve concept, generalize | `Structure`, `FiniteMultilinearStructure`, `StructureConstants` |
| Scalar composition products | `CMul`, `HMul`, `OMul`, split forms, `CxHMul`, `DMul` | Preserve exactly under named convention fixtures | generic `Algebra.from_table`; specialized constructors later |
| Symbolic/list conversion | `SymToList`, `ListToSym`, character lookup, coefficient extraction | Preserve concept | symbolic and coordinate presentations plus conversions |
| Arbitrary arrays | `ArrayToList`, `ListToArray`, sparse/depth helpers | Redesign | typed shapes, tensor/matrix parents, canonical sparse values |
| Tensor products | `TChars`, `TMul`, `MakeTMul`, tensor conjugations | Preserve concept, generalize | arbitrary finite tensor-product parent and maps |
| Matrices | `MMul`, `MTMul`, symbolic/list matrix conversions | Preserve concept, correct hypotheses | `MatrixSpace`, ordered products, recursive entry parents |
| Jordan matrices | `J2`, `J3`, Jordan products/tables/generators | Defer specialized API | generic matrix/algebra core now; Jordan release later |
| Product operators | commutator, anticommutator, associator, nested products | Preserve concept | named operations/terms; explicit parentheses |
| Property kernels | `Comm`, `Assoc`, alternatives, flexibility, Jordan/Jacobi expressions | Preserve concept, verify definitions | `Law`, `associator`, `commutator`, law factories |
| Property wrappers | `AProperty*`, `NProperty*`, tensor/matrix variants | Replace | one generic evaluator over typed parents/presentations |
| Generated property API | intended 128 `MakeProperty` outputs | Replace broken factory | typed `LawFactory`/validation; parity aliases only in migration layer if useful |
| Generic elements | algebraic and numeric vector/matrix/tensor element generators | Preserve concept | deterministic `GenericElement`; seeded strategies live in testing |
| Span/structure recovery | multiplication-to-structure tensors, matrix-to-Lie structure data | Preserve concept, strengthen witnesses | exact `span_decompose`, `closure`, `structure_constants_from_operators` |
| Conjugations/norms | scalar/tensor/matrix conjugation families | Defer family specifics | explicit involution maps and form objects |
| Clifford/geometric/Grassmann | `GAMul`, `GrMul`, `NAGMul`, index reducers | Audit now, defer public support | later exterior/Clifford/rewrite modules |
| Lie/group experiments | SO/SU generators, Lie products, exponentials, representation utilities | Audit now, defer | later Lie and representation APIs |
| Display/report tables | property grids, print banners, matrix forms | Redesign | structured reports rendered separately from computation |
| Global aliases/operators | default algebra, special Wolfram operators | Drop implicit state; optional migration syntax | explicit parent methods and named operations |
| Incomplete stubs/comments | `MakeT`/`MakeM`/`MakeMT` ideas and unfinished arbitrary-depth routes | Preserve intent only where mathematically sound | requirements and later APIs, not fake parity |

## 5. Required behavioral snapshots

For each retained partition, the final audit captures small deterministic cases:

1. basis registration and coexistence of numbered variants;
2. every basis product for `C`, `Cs`, `H`, `Hs`, `O`, and `Os`;
3. symbolic-to-list and list-to-symbol round trips with zero, rational, repeated,
   and out-of-basis terms;
4. tensor basis ordering and products for unrelated and repeated factors;
5. rectangular and square matrices, nested entries, and matrix-over-tensor
   multiplication;
6. structure constants recovered from a multiplication table and from matrices;
7. generic algebraic elements and seeded numeric elements;
8. all intended property families across scalar/tensor/matrix/container variants,
   including malformed legacy branches;
9. explicit undefined, message-emitting, or unevaluated cases;
10. registry/default mutations before and after representative calls.

Each snapshot has a definition-based oracle or is labeled reproduction-only.

## 6. Known legacy hazards requiring regression tests

- mutable package-global `alg` state and default algebra;
- name-based multiplication selection using stripped trailing digits;
- shape and expression-head dispatch that can confuse coordinates, matrices, and
  tensor arrays;
- incomplete tensor-product character recovery in `CharsToAlg`;
- comments acknowledging arbitrary-depth reconstruction uncertainty;
- nonassociative products whose nesting may be hidden by convenience syntax;
- random element functions without evidence-grade seed/domain metadata;
- repeated definitions of `OsMul` in the source;
- the missing evaluated `MakeProperty` factory and its 128 intended names;
- the `AMT` binary property branch passing the string-like `fooFoo` rather than
  the constructed `MT...` kernel;
- the malformed ternary numerical-matrix expression near line 1798;
- load banners that can suggest full support despite missing definitions.

The AnyAlgebra regression oracle is the mathematically intended and documented
behavior. A legacy defect is retained only as a captured negative fixture.

## 7. Mapping the generated property surface

The 128 intended names collapse into a small typed composition:

```text
property law
  × evaluation mode {symbolic, bounded-numeric}
  × carrier lift {scalar, tensor, matrix, matrix-over-tensor}
  × result mode {expression, validation report}
```

AnyAlgebra exposes those dimensions as arguments/objects rather than generating
public Python functions. The parity manifest maps each old name to the same law
ID, lift ID, evaluation options, and v0.0 test IDs. This preserves every useful
route while preventing 128 drifting implementations.

## 8. v0.0 parity exit criteria

- the static parser inventories every definition and `ToExpression`/factory site;
- a compatible clean kernel produces an evaluated manifest with messages;
- every intended and actually generated property name has a record even when
  absent because of a source defect;
- every registered algebra and multiplication route has basis-table snapshots;
- all v0.0-owned partitions map to API and test IDs;
- corrected replacements have a mathematical explanation and regression test;
- deferred partitions name an owning release/project;
- `validate_complete()` reports no unclassified v0.0-owned records;
- legacy output is never the only correctness oracle.

## 9. v0.0 ledger completion and remaining boundary

The v0.0 ledger is complete for its neutral generic ownership: static and
clean-kernel captures are source-hash-bound, the missing `MakeProperty` factory
and all 128 intended names are explicit records, and the manifest validator
rejects unknown names, unclassified records, or source-hash drift. The retained
fixtures cover registration, products, conversions, tensor and matrix lifting,
structure recovery, property-law mapping, and known defects.

This completion does not claim that every specialized AlgMul experiment has a
v0.0 API or that the legacy implementation is correct. Clifford, Lie, Jordan,
exceptional, geometry, and physics families remain deferred to explicitly
scoped releases. Maintainers may run the licensed Wolfram live replay as an
opt-in supplement; its absence is recorded as unavailable evidence, not as a
passing replay.

## 10. v0.1 supersession update

The table below records only behavior implemented and tested in the v0.1
worktree. “Supersedes” means the named useful behavior has a typed replacement;
it does not mean all symbols in the broader legacy partition are complete.

| Legacy behavior | v0.1 disposition | Implementation | Executable evidence | Remaining boundary |
|---|---|---|---|---|
| Expand arbitrary exact finite multilinear products, including nullary and higher arity | Supersedes the useful scalar evaluation concept | `src/anyalgebra/algebra/multilinear.py`, `evaluate_multilinear` | `tests/algebra/test_multilinear_evaluate_elements.py`; `tests/examples/test_multilinear_evaluation.py` | No implicit `x*y`, reassociation, or arbitrary basis coercion |
| Register one arbitrary finite operation table as census data | Replaces mutable name registration for this finite-carrier use case | `src/anyalgebra/census/spec.py`; `constraints.py`; `equivalence.py` | `tests/census/test_spec_core.py`; `test_constraints.py`; `test_equivalence_policy.py` | Not a general mutable AlgMul registry and not a vector algebra |
| Exhaust and constrain small operation-table families | New general capability beyond the legacy registry workflow | `src/anyalgebra/census/reference.py`; `filtering.py`; `pruning.py`; `reports.py` | `tests/census/test_reference_enumerator.py`; `test_constraint_filtering.py`; `test_local_pruning.py`; `test_enumeration_reports.py` | Hard v0.1 table, work, memory, and pruning bounds apply |
| Identify tables modulo declared relabelings | Replaces name or fingerprint coincidence with certificates | `src/anyalgebra/census/canonical.py`; `certificates.py`; `isomorphism.py`; `classify.py` | `tests/census/test_canonical_certificate.py`; `test_isomorphism_negative.py`; `test_orbit_classification.py` | Finite carriers only; vector-space basis equivalence is a different API |
| Evaluate commutative, associative, alternative, flexible, and related property kernels | Supersedes the corresponding finite-table law questions without recreating 128 generated names | `src/anyalgebra/census/analysis.py`; `src/anyalgebra/validation/` | `tests/census/test_law_profiles.py`; `tests/validation/test_standard_laws.py`; quaternion and octonion identity batteries | Matrix, tensor, and matrix-over-tensor lifts remain later work |
| Compute finite-carrier identities, zeros, nuclei, centers, substructures, and congruences | Adds capability-aware exact reports | `src/anyalgebra/census/analysis.py`; `subobjects.py`; `analysis_records.py` | `tests/census/test_element_profiles.py`; `test_finite_nuclei_center.py`; `test_substructures_congruences.py`; `test_analysis_records.py` | Linear ideals and derivations are rejected unless a module-backed algebra is supplied |
| Multiply the named `H`, `Hs`, `O`, and `Os` tables | Preserves exact named convention fixtures | `src/anyalgebra/fixtures/composition.py` | `tests/fixtures/test_composition_tables.py`; `tests/integration/test_composition_generic_evaluation.py` | This does not yet provide specialized named element classes |
| Conjugation, real part, trace, norm, and multiplication for those four tables | Supersedes this bounded composition subset of the formerly deferred conjugation/norm family | `src/anyalgebra/fixtures/composition_ops.py` | `tests/fixtures/test_composition_operations.py` | General tensor/matrix conjugations and arbitrary involution factories remain deferred |
| Associativity for `H/Hs`; alternativity, flexibility, Moufang forms, and norm composition for `O/Os` | Replaces generated property wrappers with exact basis/coefficient reductions and deterministic test-local scope records | `src/anyalgebra/algebra/multilinear.py`; `src/anyalgebra/fixtures/composition_ops.py` | `tests/fixtures/test_quaternion_identity_battery.py`; `test_octonion_identity_battery.py` | Results are convention- and coefficient-domain-scoped, not universal classification; the scope records are not yet public persistence types |
| Split isotropy, zero divisors, complementary idempotents, and nondivision witnesses | Adds explicit positive and negative controls | `src/anyalgebra/fixtures/composition_ops.py`; generic multilinear evaluator | `tests/fixtures/test_split_composition_controls.py` | Ordinary `H/O` controls prevent promotion of split witnesses to all fixtures |
| Transport products and checked invariants under exact rational basis changes | Preserves the convention-transport concept with literal parents and concrete inverse maps | `src/anyalgebra/presentations/basis_change.py` | `tests/integration/test_composition_basis_invariance.py`; `tests/presentations/test_basis_change.py` | Maps are currently transient Python records, not a versioned convention-map corpus |
| Publish and query a finite classification artifact | New content-addressed replacement for mutable report/global-state workflows | `src/anyalgebra/atlas/` | `tests/atlas/`; `tests/examples/test_finite_algebra_atlas.py` | Complete only for each explicitly declared bounded corpus |

The shipped reference atlas examines all 19,683 binary tables on three labelled
elements, retains the 729 commutative tables, and classifies them into 129
full-`S3` relabeling orbits. The exact scope and its 862-object closure are
asserted in `tests/examples/test_finite_algebra_atlas.py`. This is not a claim
about every order-three magma, AlgMul correctness, or exceptional physics.

## 11. Original NEW-02 handoff (now unscheduled)

v0.1 produces the following inputs for the convention-equivalence laboratory:

1. Four immutable source convention fixtures with IDs `algmul.H.v1`,
   `algmul.Hs.v1`, `algmul.O.v1`, and `algmul.Os.v1`; their ordered bases,
   complete signed multiplication tables, and source anchors live in
   `src/anyalgebra/fixtures/composition.py` and are checked by
   `tests/fixtures/test_composition_tables.py`.
2. Convention-scoped conjugation signs and canonical operation records from
   `src/anyalgebra/fixtures/composition_ops.py`, checked by
   `tests/fixtures/test_composition_operations.py`.
3. Exact identity reductions and minimal nonassociativity, isotropy,
   zero-divisor, and idempotent witnesses checked by the three composition
   battery suites under `tests/fixtures/`.
4. `ExactBasisIsomorphism` with concrete forward/inverse images and an exact
   inverse certificate in `src/anyalgebra/presentations/basis_change.py`, plus
   rational shear and signed-permutation controls in
   `tests/integration/test_composition_basis_invariance.py`.
5. Immutable `ConventionManifest`, `SourceAnchor`, semantic-hash, canonical
   serialization, and replay primitives in `src/anyalgebra/evidence/`, checked
   by `tests/evidence/` and `tests/persistence/`.
6. A finite-carrier canonical-map/certificate pattern in
   `src/anyalgebra/census/transport.py` and `certificates.py`. v0.2 may reuse its
   proof shape, but must not confuse carrier permutations with linear basis
   transformations.

The following are unresolved v0.2 work, not v0.1 deliverables:

- instantiate versioned `ConventionManifest` records for each external
  convention being compared, including source hashes and orientation data;
- serialize canonical vector-space maps rather than relying on transient test
  helpers;
- select at least two independent convention families and pin both sources;
- solve and verify exact basis/sign maps for products, conjugations, forms, and
  declared tensors;
- emit a smallest localized mismatch witness when no map satisfies all declared
  diagrams;
- add Clifford and Lie convention adapters only after their own typed parents
  exist; and
- keep equal fingerprints as candidate filters only, never positive evidence.

No v0.2 equivalence result, Clifford convention translation, or Lie convention
translation is claimed here.
