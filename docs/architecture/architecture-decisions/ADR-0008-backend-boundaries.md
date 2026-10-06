# ADR-0008: Backend boundaries

Status: Proposed  
Date: 2026-07-22

## Context

The package will eventually use pure Python, SymPy, NumPy/SciPy, SageMath, GAP,
Mathematica, Schouten-kit, and possibly proof assistants. Letting one backend's
types or semantics define the core would prevent arbitrary structures and make
results environment dependent.

## Decision

The neutral model owns parents, operations, outcomes, evidence, and
serialization. A small exact reference backend defines normative v0.0 behavior.
Adapters expose capability objects with declared domains, exactness, algorithms,
versions, and resource limits. Backend values are converted at the boundary and
do not leak into stable public records.

Backend selection is explicit or policy driven and is recorded in the
calculation contract. Missing capability is distinguished from mathematical
failure. Optimized and external backends pass differential tests against the
reference backend on their shared exact domain. Independent algorithms are
preferred for scientific cross-checks; two wrappers around the same routine do
not count as independent evidence.

Schouten-kit may remain a sibling repository and be referenced through an
optional path/development dependency. AnyAlgebra must not copy it into the core
or require it for import.

## Consequences

- Core semantics remain stable as performance tooling changes.
- Adapter translation and capability negotiation require explicit code.
- External tool output must carry version and convention metadata.

## Rejected alternatives

- Making SymPy expressions or NumPy arrays the universal element type.
- Import-time discovery of arbitrary sibling code.
- Treating every backend disagreement as a numerical tolerance issue.

## Validation

Tests cover absent backends, capability discovery, exact differential fixtures,
conversion isolation, version recording, sibling-path Schouten integration, and
backend disagreement receipts.

