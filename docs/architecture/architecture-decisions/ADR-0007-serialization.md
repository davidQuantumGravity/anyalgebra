# ADR-0007: Serialization and schema evolution

Status: Proposed  
Date: 2026-07-22

## Context

AnyAlgebra must exchange definitions, calculations, and evidence over many
years. Python pickles are unsafe and couple data to private class layouts.

## Decision

Stable records use tagged, schema-versioned JSON with canonical key ordering,
canonical exact-number encodings, explicit parent references, and content hashes.
Large arrays or artifacts may be stored externally, but the JSON record includes
media type, byte length, and cryptographic hash.

Deserialization uses an allow-listed type registry and never imports or executes
code named by input data. Unknown types or future schema versions are rejected
with structured diagnostics. Migrations are pure, version-to-version functions
with golden fixtures; original records remain preserved.

Semantic identity excludes nonessential display metadata. Canonical
serialization has one representation for normalized values, including rational
signs, sparse zero removal, basis order, and outcome tags.

## Consequences

- Data is inspectable and safe by default.
- Schema design precedes stable release of a type.
- Backend-native objects require neutral encodings or external artifacts.

## Rejected alternatives

- Pickle as a persistence contract.
- Class-name import hooks from untrusted JSON.
- Hashing pretty-printed or noncanonical output.

## Validation

Golden round trips, canonical-byte equality, unknown-tag rejection, migration
fixtures, malformed reference graphs, and hash mismatch tests are mandatory.

