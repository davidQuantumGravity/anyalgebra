# ADR-0004: Partiality and evaluation outcomes

Status: Proposed  
Date: 2026-07-22

## Context

A product may be mathematically undefined, a backend may not implement an
operation, a bounded search may be inconclusive, or a computation may crash.
Using `None`, exceptions, or one generic “unknown” loses essential meaning.

## Decision

Potentially partial evaluation returns one of four tagged outcomes:

- `Defined(value)`: the requested mathematical value was computed;
- `Undefined(reason, witness)`: the operation is mathematically not defined for
  these inputs under the declared structure;
- `Indeterminate(reason, bounds)`: the method cannot decide within declared
  assumptions or resource bounds;
- `Failed(error, stage)`: the requested computation should have been meaningful
  but execution failed or the input violated an implementation contract.

Unsupported optional capabilities raise or return a typed `UnsupportedCapability`
at the API boundary; they are not mathematical undefinedness. Law validation
uses separate `Proved`, `Disproved`, and `Inconclusive` results.

No partial operation is silently totalized. A user may explicitly adjoin a
bottom/error element, producing a new totalized parent and a recorded map.

## Consequences

- Negative mathematical results survive serialization and publication.
- Callers must handle outcomes explicitly.
- Convenience methods may unwrap `Defined` and raise typed exceptions, but the
  neutral API remains outcome based.

## Rejected alternatives

- `None` for every non-value.
- Exceptions for ordinary mathematical undefinedness.
- Returning zero for missing multiplication-table entries.

## Validation

Every outcome branch, serialization round trip, totalization, unsupported
backend, and bounded-search limit is tested independently.

