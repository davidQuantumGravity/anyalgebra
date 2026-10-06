# ADR-0003: Domain and coercion policy

Status: Proposed  
Date: 2026-07-22

## Context

Scientific algebra mixes integers, rationals, real approximations, symbolic
parameters, field extensions, and algebra elements. Silent promotion can lose
exactness or choose a mathematically unintended embedding.

## Decision

Coercions are immutable, typed maps in a directed graph. Each map declares
source, target, injectivity, surjectivity, exactness, cost, assumptions, and an
optional inverse. Automatic coercion is allowed only for a unique declared
lossless path. If two incomparable paths exist, the operation returns a coercion
ambiguity error and lists the paths.

Lossy conversion, branch choice, projection, quotienting, or approximation is a
named explicit conversion. Python numeric types enter through registered domain
constructors; they do not define the global promotion lattice.

Common-parent selection considers only declared embeddings and must be stable
under operand order. User policy can select among valid paths, but the policy is
part of the calculation contract and receipt.

## Consequences

- Exactness cannot disappear silently.
- Extension fields and split forms can coexist without name heuristics.
- Coercion graph construction and ambiguity tests become core responsibilities.

## Rejected alternatives

- NumPy-style dtype promotion as the mathematical policy.
- “Largest-looking” domain wins.
- Implicit numerical approximation when exact operations are unavailable.

## Validation

Required cases include `ZZ -> QQ`, forbidden implicit `QQ -> ZZ`, diamond-path
ambiguity, explicit approximation, scalar extension, and operand-order symmetry.

