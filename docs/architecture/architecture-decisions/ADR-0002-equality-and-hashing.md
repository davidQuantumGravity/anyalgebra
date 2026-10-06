# ADR-0002: Equality and hashing

Status: Proposed  
Date: 2026-07-22

## Context

Mathematical equality can mean structural equality, equality after coercion,
equality modulo relations, or an undecided symbolic proposition. Conflating
these meanings makes sets, dictionaries, simplification, and evidence unsound.

## Decision

Immutable parents use structural equality over canonical defining data. Elements
compare canonically inside one parent. Cross-parent mathematical comparison is a
named operation, `compare`, that may use an explicit common parent or map and
returns a tri-valued result: equal, unequal, or indeterminate.

Python `==` remains deterministic and boolean. It returns `False` for elements
of different parents unless the objects are literally the same canonical parent
and values compare canonically. It never performs lossy, ambiguous, symbolic, or
expensive coercion. Mathematical comparison after transport is explicit.

Only immutable canonically normalized objects are hashable. Objects containing
unknown symbolic equivalence, mutable external state, or noncanonical backend
values are unhashable until normalized or frozen.

## Consequences

- Dictionary behavior is stable and cheap.
- Users must distinguish object equality from proven mathematical equivalence.
- Symbolic equality cannot accidentally collapse to Python truthiness.

## Rejected alternatives

- Automatic coercion inside `__eq__`.
- Returning symbolic expressions from `__eq__`.
- Hashing display strings or floating approximations.

## Validation

Tests cover normalization, hash consistency, cross-parent comparison, ambiguous
coercions, symbolic indeterminacy, and equality after an explicit isomorphism.

