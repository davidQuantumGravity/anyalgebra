# AnyAlgebra evidence model

Status: Normative for v0.0  
Last updated: 2026-07-22

## 1. Goal

The evidence model prevents correct software execution, legacy reproduction,
finite computation, and scientific verification from being conflated. It makes
positive, negative, failed, and inconclusive calculations equally durable and
traceable.

## 2. Separate axes

Every result records three independent classifications.

### 2.1 Execution status

- `not_run`
- `running`
- `completed`
- `failed`
- `blocked`

### 2.2 Mathematical outcome

- `constructed`: a requested object satisfying checked obligations was built;
- `verified_within_domain`: a property holds over the exact declared domain;
- `counterexample`: a witness disproves the stated universal claim;
- `no_go_within_assumptions`: stated assumptions imply an obstruction;
- `inconclusive`: bounds or assumptions do not decide the question;
- `not_applicable`: the requested theorem or operation hypotheses fail;
- `reproduction_only`: an external/legacy result was reproduced without an
  independent correctness conclusion;
- `implementation_error`: software failed before a mathematical conclusion.

### 2.3 Evidence tier

| Tier | Meaning |
|---|---|
| `E0-unimplemented` | No executable calculation exists |
| `E1-executed` | Code ran and artifacts were captured |
| `E2-regression` | Stable fixtures and software tests pass |
| `E3-bounded-exact` | Exact exhaustive result on a recorded finite domain |
| `E4-cross-checked` | Independent algorithm/source agrees on the declared domain |
| `E5-proof-linked` | Formal or hand-auditable proof obligations connect computation to the claim |
| `E6-reproduced` | Independent environment reproduces hashes or equivalent certified outputs |

Tiers are descriptive, not a linear truth score. A counterexample can have E3 or
E4 evidence. E2 software quality alone does not verify a scientific claim.

## 3. Core records

### `SourceAnchor`

Identifies the exact source of an input claim or fixture: path/URI, content hash,
edition or commit, page/line/label, and an optional excerpt hash. A path without
a hash is locational metadata, not immutable provenance.

### `ConventionManifest`

Pins basis order, signs, involutions, normalization, indexing, parenthesization,
and named conversions. See [conventions-v0.0.md](../conventions/conventions-v0.0.md).

### `CalculationContract`

Defines before execution:

- contract and project IDs;
- precise question and acceptable outcome kinds;
- input object/artifact hashes;
- source anchors and convention manifests;
- algorithms, backends, and versions;
- assumptions and theorem hypotheses;
- enumeration/search/resource bounds;
- required cross-checks;
- expected artifacts and acceptance predicates.

Changing any semantic field creates a new contract ID.

### `ResultReceipt`

Records:

- contract hash and execution environment;
- timestamps and software revision;
- execution status, mathematical outcome, and evidence tier;
- result summary without stronger language than the tier permits;
- case counts, bounds, warnings, and smallest witnesses;
- output artifact hashes and logs;
- tests/cross-checks actually performed;
- dependency receipts;
- reviewer/reproduction attestations;
- supersedes/superseded-by links.

Receipts are immutable and content addressed.

### `ClaimRecord`

Links a manuscript or project claim to its source anchor, dependencies,
calculation contracts, current receipts, allowed public wording, and unresolved
obligations. Multiple competing hypotheses can link to the same evidence.

## 4. Promotion rules

Evidence can be promoted only by adding the missing artifact:

- E0 to E1 requires a captured run;
- E1 to E2 requires stable tests and fixtures;
- E2 to E3 requires exactness and complete declared-domain counts;
- E3 to E4 requires an independent method, not a second call through the same
  backend;
- E4 to E5 requires explicit proof obligations or a formal certificate;
- E5 to E6 requires independent replay/reproduction.

Code cannot promote its own receipt merely by setting a field. Promotion creates
a new receipt that references evidence satisfying the rule. A narrower exact
claim may have stronger evidence than a broad conjecture built from it.

## 5. Negative and inconclusive results

A counterexample receipt includes the smallest stable witness, all convention
data needed to evaluate it, and an independent direct check where feasible. A
no-go receipt lists its assumptions and does not generalize beyond them. An
inconclusive receipt records explored bounds and remaining branches so work can
resume without repeating the search.

“The proposed fibration is believed not to work,” for example, remains a
project hypothesis until the exact candidate, topology assumptions, and
obstruction are captured. Reconfirmation may yield a scoped no-go receipt; it
must not become a statement about every discrete or generalized construction.

## 6. Legacy evidence

AlgMul artifacts record source hash, kernel version, initialization result,
messages, generated symbols, registries, sample inputs/outputs, and known global
state. Matching AlgMul is `reproduction_only` until a definition or independent
oracle establishes correctness. Correcting a legacy defect records both the
legacy behavior and the mathematical reason for the replacement.

## 7. Staleness and dependency invalidation

A receipt is stale for a new claim evaluation when any semantic dependency has
changed: source content, convention, parent definition, algorithm, backend,
resource bound, or prerequisite receipt. Staleness does not delete historical
evidence. A dependency graph reports exactly which edge invalidated reuse.

Display-only metadata changes do not alter semantic hashes.

## 8. Serialization skeleton

```json
{
  "schemaType": "ResultReceipt",
  "schemaVersion": "1.0",
  "id": "receipt:sha256:...",
  "contract": "contract:sha256:...",
  "executionStatus": "completed",
  "mathematicalOutcome": "counterexample",
  "evidenceTier": "E3-bounded-exact",
  "summary": "The identity fails on the declared finite fixture.",
  "bounds": {"carrierSize": 8, "casesExpected": 512, "casesRun": 512},
  "witnesses": ["artifact:sha256:..."],
  "artifacts": ["artifact:sha256:..."],
  "dependencies": [],
  "supersedes": null
}
```

Canonical JSON and security rules follow ADR-0007.

## 9. Publication language

Generated prose uses constrained badges:

- `implemented` for E1 or above software availability;
- `regression-tested` for E2;
- `verified on <explicit domain>` for E3;
- `independently cross-checked on <domain>` for E4;
- `proof-linked` only for E5;
- `independently reproduced` only for E6;
- `counterexample found`, `no-go under assumptions`, or `inconclusive to bounds`
  according to the mathematical outcome.

The unqualified words “proved,” “confirmed,” “complete,” and “ruled out” are not
generated from E1–E4 alone.

## 10. v0.0 acceptance tests

- canonical receipt hashes are stable across serialization round trips;
- changing a convention or bound changes the contract hash;
- a completed run cannot set `verified_within_domain` without case counts or a
  theorem-backed certificate;
- two wrappers around one implementation fail the independence check;
- negative and inconclusive receipts replay without being converted to errors;
- dependency changes mark reuse stale while preserving the original receipt;
- a receipt cannot raise its own tier;
- publication badges never exceed the receipt's evidence.

