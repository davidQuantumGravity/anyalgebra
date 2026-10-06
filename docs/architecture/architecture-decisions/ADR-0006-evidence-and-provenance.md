# ADR-0006: Evidence and provenance

Status: Proposed  
Date: 2026-07-22

## Context

Many target calculations concern conjectures, proposed structures, negative
results, and computations whose validity depends on conventions and bounds. A
successful function call is not automatically evidence for a scientific claim.

## Decision

Calculations consume an immutable `CalculationContract` and emit a
`ResultReceipt`. The contract pins inputs, source anchors, convention manifests,
algorithm/backend versions, assumptions, bounds, and requested claim. The
receipt records outcome, artifacts, hashes, diagnostics, independent checks, and
the strongest justified evidence tier.

Evidence status and scientific conclusion are separate axes. Implementation
success cannot promote a claim. Negative and inconclusive results are first
class. Receipts are append-only and content-addressed; corrections create new
receipts linked to the superseded one.

The detailed model is normative in [evidence-model.md](../../evidence/evidence-model.md).

## Consequences

- Publication tables can be generated without overstating results.
- Small computations carry more metadata.
- Reproduction failures can be located at an exact dependency or convention.

## Rejected alternatives

- A single `verified: true` flag.
- Inferring claim status from green tests.
- Editing old receipts in place after a correction.

## Validation

Tests cover content hashes, stale dependency detection, supersession, negative
results, partial reproduction, and prohibited evidence-tier escalation.

