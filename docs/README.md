# AnyAlgebra documentation

This directory holds the contracts that govern implementation and the guides
built on them. The documents have the following authority order:

1. A version specification defines what a release must implement.
2. Accepted architecture decisions define semantics shared across releases.
3. API, convention, testing, and evidence documents refine the version spec.
4. The legacy registry provides migration scope and traceability.

## Start here

- [What can AnyAlgebra do today?](capabilities.md)
- [Finite algebra census and atlas manual](guides/finite-algebra-census.md)
- [Using quaternions and octonions](guides/quaternions-and-octonions.md)
- [Current implementation and distribution status](status.md)
- [v0.1 API](api/api-v0.1.md)
- [Convenience layer](api/easy.md)
- [Composition algebras and involutions](api/composition.md)
- [v0.0 specification](specifications/v0.0-spec.md)
- [v0.0 API](api/api-v0.0.md)
- [Architecture decisions](architecture/architecture-decisions/README.md)
- [Testing strategy](testing/testing-strategy.md)
- [v0.0 conventions](conventions/conventions-v0.0.md)
- [Evidence model](evidence/evidence-model.md)
- [AlgMul parity contract](legacy/algmul-parity.md)
- [Mathematical test manual](guides/mathematical-test-manual.md)

The latest tracked implementation cycle is the locally complete 0.1.0 source
milestone. All 60 tasks are accepted and `V01-060` passed its aggregate
readiness gate. Version records under `.agents/` are internal engineering
evidence that is maintained privately and not distributed here; see the
repository-level README and the [current-status page](status.md) for the exact
distinction between source, milestone, artifact, tag, publication, and research
claims.

## Directory map

| Directory | Purpose |
|---|---|
| `specifications/` | Bounded release contracts and acceptance criteria |
| `architecture/architecture-decisions/` | ADRs for semantic choices that code must not decide accidentally |
| `api/` | Public Python interfaces owned by each release |
| `testing/` | Test levels, oracles, coverage, and release gates |
| `conventions/` | Named mathematical fixture conventions and translations |
| `evidence/` | Provenance, outcomes, receipts, and scientific claim rules |
| `legacy/` | AlgMul audit, parity decisions, evaluated manifests, and migration tools |
| `guides/` | Mathematics-first manuals built from executable examples and tests |

Generated files belong under the repository's `build/` directory and must
identify their source and generator. They are evidence artifacts, not
hand-edited authority.
