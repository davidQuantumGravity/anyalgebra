# ADR-0005: Immutability and caching

Status: Proposed  
Date: 2026-07-22

## Context

Mutable algebra definitions invalidate elements, hashes, caches, evidence, and
serialized results. Yet construction and expensive exact calculations benefit
from builders and caches.

## Decision

Parents, signatures, bases, laws, conversions, convention manifests, source
anchors, and receipts are immutable after validation. Builders may be mutable,
but `freeze()` validates and returns a new immutable value. Updating a structure
creates a distinct parent and fingerprint.

Caches are non-semantic optimization layers. Cache keys include the immutable
object fingerprint, algorithm version, backend identity, options, and resource
bounds. Cache contents never affect equality or serialization. Shared global
mutable registries are forbidden in the mathematical core; registry instances
are explicit dependencies.

## Consequences

- Reproducible hashing and safe parallel calculations become possible.
- Large constructions may require staged builders.
- Cache invalidation becomes versioned key selection rather than mutation.

## Rejected alternatives

- Mutating a multiplication table after elements exist.
- Process-global current algebra or convention.
- Serializing caches as if they were mathematical definitions.

## Validation

Tests attempt mutation, compare fingerprints before and after builder changes,
verify cache independence, and run concurrent calculations with separate
registries.

