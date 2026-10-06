# AnyAlgebra testing strategy

Status: Normative for v0.0  
Last updated: 2026-07-22

## 1. Testing purpose

Tests must establish mathematical semantics, software behavior, and evidence
boundaries separately. A green test proves only its stated assertion over its
recorded domain. It does not by itself verify a conjecture, establish a theorem
outside finite bounds, or reproduce a research project.

Every public behavior has a stable test ID in
`.agents/project-process/test-matrix.json`. Test code names include that ID when
practical, and generated reports preserve it.

## 2. Test levels

| Level | Purpose | Required examples |
|---|---|---|
| Unit | One semantic object or algorithm | normalization, one table lookup, one outcome branch |
| Contract | Public API behavior independent of implementation | parent mismatch, coercion ambiguity, serialization tags |
| Property | Laws over generated or enumerated inputs | bilinearity, round trips, basis-change invariance |
| Exhaustive finite | Every case in a declared finite domain | partial table cells, finite carrier laws, conversion paths |
| Integration | Several neutral layers together | structure constants to operators to recovered constants |
| Differential | Two implementations on their common domain | reference backend versus optional adapter |
| Metamorphic | Behavior under a known transformation | basis change, relabeling, scalar extension |
| Regression | A previously observed defect or mismatch | malformed AlgMul generator, ambiguous span |
| Golden data | Stable canonical bytes or trusted small fixtures | JSON receipts, multiplication tables |
| Documentation | Published examples execute exactly | quick start and all v0.0 required examples |
| Packaging | Clean install and import boundaries | core installs without optional backends |
| Performance | Detect catastrophic regressions, not prove speed | bounded table and closure benchmarks |

## 3. Oracle hierarchy

Use the strongest available oracle and name it in the test:

1. direct consequence of a definition;
2. hand-checkable exact fixture with provenance;
3. theorem-backed reduction with hypotheses checked in code;
4. independently implemented exact algorithm;
5. external authoritative implementation with pinned version and convention;
6. legacy output used only as a parity comparison;
7. numerical or randomized evidence with explicit bounds and tolerance.

Legacy AlgMul output is never the only correctness oracle. For a parity test,
also use a definition, an independent implementation, or a known invariant when
possible. If no second oracle exists, the result is labeled reproduction-only.

## 4. Exhaustive and sampled claims

Every enumeration records:

- parent and convention manifest IDs;
- carrier, coefficient set, or generator domain;
- arity and term-depth bounds;
- ordering and symmetry reductions;
- number of cases expected and actually evaluated;
- skipped cases and reasons;
- algorithm/backend versions;
- content hashes for inputs and outputs.

The words “exhaustive,” “complete,” and “proved computationally” may appear only
with these bounds. Sampling records the random generator, seed, distribution,
sample size, and shrinker. Sampled success never becomes an exhaustive result.

## 5. Basis-reduction tests

Checking a polynomial identity only on basis vectors is valid only when a
recorded theorem applies. The test harness must verify the required
multilinearity or polarization assumptions. Otherwise it enumerates the declared
coefficient box or returns `Inconclusive`. The test report states which route was
used.

For multiplication-table algebras, table completeness is checked before law
tests. Undefined entries in a partial algebra are not treated as zero and must
be handled according to the law's declared partial semantics.

## 6. v0.0 mandatory test matrix

| Area | Minimum assertions |
|---|---|
| Domains | canonical `ZZ`/`QQ`; exact arithmetic; zero normalization; invalid constructors |
| Coercions | unique `ZZ -> QQ`; operand-order independence; ambiguity witness; lossy rejection |
| Parents | membership; same coordinates/different parents; immutable fingerprints; builder freeze |
| Carriers and modules | basis order; sparse zero removal; dimension checks; parent-preserving arithmetic |
| Signatures | arbitrary arity; many sorts; invalid sort maps; stable serialization |
| Partiality | all four outcome tags; unsupported capability; explicit totalization |
| Terms | variables; substitution; fully parenthesized evaluation; no implicit reassociation |
| Structures | table/callable construction; relation queries; reduction reachability; deductive closure |
| Multilinear algebra | structure constants; bilinear extension; generic element; tensor/matrix lifting |
| Presentations | all declared round trips; path ambiguity; basis change; malformed coordinates |
| Linear recovery | unique span; dependent generators; nonclosure witness; recovered constants |
| Laws | proved/disproved/inconclusive; smallest counterexample; hypothesis failures |
| Analysis | associator; commutator; nuclei; center; ideals; derivations; basis-change invariance |
| Serialization | canonical bytes; unknown tag rejection; hash mismatch; schema migration |
| Evidence | no self-promotion; negative receipts; stale inputs; supersession links; replay |
| Backends | exact reference behavior; missing optional backend; differential shared-domain cases |
| AlgMul | static plus evaluated manifest; generated family coverage; corrected-defect regressions |

## 7. Reference fixtures

The v0.0 fixture corpus includes:

- empty, singleton, and three-element carriers;
- a three-element partial magma with at least one undefined product;
- a two-sorted relational structure and a finite deductive-system fixture;
- `ZZ` and `QQ` modules of dimensions zero through four;
- rational associative algebras, including a commutative and a noncommutative
  example;
- generic quaternion, split-quaternion, octonion, and split-octonion tables under
  named conventions;
- a deliberately perturbed nonassociative table;
- linearly independent, dependent, nonspanning, and nonclosed operator families;
- canonical legacy snapshots from compatible Mathematica evaluation.

Fixtures carry source anchors and are immutable. A change to fixture semantics
requires a new fixture ID, not silent replacement.

## 8. Property and metamorphic testing

The following transformations must preserve their declared observables:

- exact invertible basis change;
- carrier relabeling;
- presentation round trip;
- structure transport along an isomorphism;
- tensor-factor identity insertion where typed;
- serialization round trip;
- deterministic reordering into canonical sparse form.

Failures are shrunk to a minimal term, table cell, conversion path, or generator
subset. The shrink result is stored in the receipt.

## 9. AlgMul testing

AlgMul testing has four independent layers:

1. static source inventory, including calls that construct symbol names;
2. clean-kernel load and before/after context diff;
3. generator exercise with registry and down-value snapshots;
4. behavioral parity/corrected-replacement fixtures.

The audit pins the source hash and Mathematica kernel version. A load that emits
messages or halts still produces a failure artifact. Generated functions are
grouped by factory pattern, but every generated name has its own manifest entry.
Known malformed definitions are tested against the mathematically intended
replacement and recorded as defects, not preserved exactly.

## 10. CI tiers

### Pull request: fast

- formatting, linting, type checks, schema validation;
- unit and contract tests;
- small property-test budget;
- documentation link and executable-snippet checks;
- registry traceability validation.

Target runtime: under ten minutes on a normal development machine.

### Main branch: full

- all fast checks;
- exhaustive v0.0 finite grids;
- integration, metamorphic, serialization, and packaging tests;
- reference-backend coverage report;
- optional-backend tests where available.

### Scheduled: reference and legacy

- clean Mathematica audit where a licensed compatible kernel is available;
- differential tests against external adapters;
- larger property budgets and mutation testing;
- performance baselines and artifact reproducibility.

Every collected test under `tests/legacy/` receives the `legacy` marker from
the repository `conftest.py`. This directory-level boundary prevents a newly
added legacy evidence test from silently entering the neutral lane when its
author forgets a function-level marker. Legacy tests consume committed receipts
or create synthetic temporary inputs; they do not read the external AlgMul or
research checkouts. A fresh clone can run `python -m pytest -q`. CI retains
separate neutral and legacy selections so their evidence and coverage reports
remain distinct.

The ordinary coverage report omits `src/anyalgebra/legacy/` for the same
reason: measuring source whose complete test lane was intentionally deselected
would make the neutral percentage meaningless. Historical stable coverage of
those modules remains in the frozen canonical ledger; live legacy behavior is
reported by the scheduled evidence lane rather than folded into the neutral
percentage.

### Research campaign

Project calculations run outside ordinary CI unless bounded and stable. They
produce receipts and may be promoted to regression fixtures only after review.

## 11. Coverage and quality gates

Traceability and behavior coverage are release gates; raw line coverage is a
diagnostic. For v0.0:

- every public function and documented branch has at least one contract test;
- partiality, evidence state transitions, coercion ambiguity, and deserialization
  rejection require complete branch coverage;
- every bug fix adds a regression test;
- no test is allowed to pass merely because an optional dependency is absent;
- expected failures include an issue/reason and expiry milestone;
- warnings are asserted or eliminated, never ignored globally.

Initial numeric coverage targets are 95% statement and 90% branch coverage for
the neutral package, rising where feasible. Missing a numeric target does not
permit a missing semantic contract test; conversely, hitting it does not prove
release readiness.

## 12. Test naming and organization

Recommended layout:

```text
tests/
  unit/
  contracts/
  properties/
  exhaustive/
  integration/
  differential/
  regression/
  golden/
  docs/
  packaging/
  fixtures/
```

Names follow `test_<area>_<observable>`. Parameter IDs are mathematical and
readable, such as `octonion-algmul-v1`, not positional numbers. Each research
fixture links to a project ID and evidence receipt when applicable.

## 13. Failure artifacts

On failure, retain the smallest relevant input, parent and convention IDs,
expected and actual outcome, backend and algorithm versions, seed/bounds,
traceback or mathematical witness, and artifact hashes. Sensitive or proprietary
paths are normalized before publication, but original local receipts may retain
resolvable source anchors.

## 14. Release decision

The v0.0 release is ready only when the acceptance checklist, process registry,
test report, AlgMul parity coverage, and evidence replay checks agree. A test
suite that passes while traceability or legacy dispositions remain incomplete is
not sufficient.

## 15. Flaky-test policy

Exact deterministic tests must not be retried to obtain a pass. A failure is
investigated as a product, fixture, environment, or test defect. Randomized tests
always print and retain their seed and shrunk witness. Timing-sensitive checks
use broad resource classifications outside the semantic suite.

A genuinely environment-dependent optional-backend test may be quarantined only
with an owner, issue, captured failure, expiry milestone, and replacement gate.
Quarantine never converts a required v0.0 test into a passing one. If a required
test cannot run in CI, release evidence must identify the controlled environment
where it ran and retain the artifact.

## 16. Mutation and performance testing

Mutation testing targets high-risk semantics first: parent checks, coercion
ambiguity, partiality tags, law-result promotion, canonical serialization,
evidence tiers, and AlgMul disposition completeness. Surviving mutations become
specific missing-test issues; a raw mutation percentage is not a scientific
quality claim.

Performance tests record machine, Python/backend version, dimensions, sparsity,
coefficient growth, wall time, peak memory, and result hash. Initial v0.0 budgets
are guardrails against catastrophic behavior, not promises:

- all pull-request tests finish within ten minutes on the reference developer
  class machine;
- required documentation examples finish within thirty seconds each;
- small finite-table and conversion operations have explicit case-count and
  memory ceilings chosen before implementation;
- a performance regression gate is enabled only after five stable baseline runs
  establish variance, with separate thresholds for time and memory.

No benchmark supports a public speed claim until compared under the same exact
semantics, fixtures, hardware, and warm/cold-cache policy.
