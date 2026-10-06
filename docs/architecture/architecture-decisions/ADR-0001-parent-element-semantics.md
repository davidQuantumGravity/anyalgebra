# ADR-0001: Parent and element semantics

Status: Proposed  
Date: 2026-07-22

## Context

The same coordinate tuple can represent elements of unrelated algebras. Python
types alone cannot express basis, coefficient domain, operations, conventions,
or representation. Global registries, as used by legacy AlgMul workflows, also
prevent independent structures with similar names from coexisting safely.

## Decision

Every semantic element has exactly one immutable `Parent`. A parent owns the
element's carrier/module, coefficient domain, basis or coordinate policy,
operations, and convention identity. Construction validates membership.

Binary operations accept elements of one parent, or use an explicit unique
lossless coercion into a declared common parent. Parent mismatch is never solved
by comparing shapes or names. Parents are ordinary values: multiple structures
with identical display names may coexist, and structural identity includes all
semantic defining data.

Substructures, quotients, scalar extensions, tensor products, and presentations
create new parents linked by explicit maps. A presentation is not itself the
mathematical identity of an element.

## Consequences

- Elements remain unambiguous across simultaneous calculations.
- Caches and hashes can incorporate a stable parent fingerprint.
- Creating parents is more explicit than constructing bare lists.
- Cyclic definitions require builders that freeze into immutable parents.

## Rejected alternatives

- Inferring parents from Python classes or container shape.
- A mutable process-wide algebra registry.
- Treating equal coordinate arrays as equal mathematical elements.

## Validation

Tests must cover same coordinates in different parents, isomorphic but distinct
presentations, parent mismatch, explicit isomorphism transport, and coexistence
of two user-defined structures with the same label.

