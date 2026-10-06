# AnyAlgebra test fixtures

This directory is the source-controlled, read-only input corpus for external
tests. Test code must never rewrite a fixture. A semantic correction creates a
new fixture ID and file; it does not silently replace an existing fixture.

The first path component declares one of these immutable namespaces:

| Namespace | Purpose | Required provenance |
| --- | --- | --- |
| `canonical/` | Hand-checkable exact inputs. | Fixture ID, definition or source anchor, convention ID, and canonical-content hash. |
| `published/` | Transcriptions from a published or authoritative source. | Fixture ID, source URL or bibliographic anchor, source revision/page, convention ID, and content hash. |
| `legacy/` | Captures from the pinned AlgMul environment. These are comparison evidence, never the sole correctness oracle. | Fixture ID, source hash, kernel/version details, capture command or receipt, convention ID, and content hash. |
| `generated/` | Deterministic inputs generated from a checked-in recipe. | Fixture ID, generator version, complete seed/parameters, convention ID, and content hash. |

Each fixture's metadata must identify its namespace, provenance fields, and
immutable content hash next to the payload or in a paired metadata record. A
fixture may be added only with a readable source anchor; unnamed local examples
do not belong here. Research-scoped data additionally names its project and
evidence receipt, and does not by itself establish a scientific result.

Failure diagnostics belong in `../artifacts/`, whose tracked `.gitkeep` keeps
the location available before the first failure. Retained artifacts record the
smallest relevant input, expected and actual outcome, fixture and convention
IDs, algorithm/backend versions, bounds or seed, traceback or witness, and
hashes. They are outputs, not fixtures, and must not be read as golden inputs.
