# Changelog

## Unreleased

- Licensed the repository under `AGPL-3.0-only` and documented the option to
  request separately negotiated proprietary terms.
- Restored strict MyPy compatibility with the current development toolchain by
  removing two casts that are now inferred as redundant.
- Prepared the repository for public visibility. Internal process records
  (`.agents/`), unpublished research notes and manuscripts (`docs/research/`),
  and rendered PDFs are no longer tracked. Tests that audit those records
  carry the `internal_records` marker and skip with an explicit reason when
  the records are absent.
- Rewrote the README around installation and a first calculation, added
  `CITATION.cff`, removed the CI steps that required the internal records, and
  raised the CI job time limits to fit the neutral suite.
- Made two coverage tests independent of how the interpreter checks runtime
  protocols. Before Python 3.12 that check reads each protocol property, so
  the tests failed on Python 3.11 although the library behaved correctly.
- Corrected the README command for the atlas example, which needs an output
  folder, and separated the status-page checks that run in any checkout from
  the two that read maintainer records.

### Experimental namespace review

An adversarial review checked the exact claims of `anyalgebra.experimental`
against independent computations and fixed the defects it found.

- Replaced the numerical exponential with one norm-scaled routine,
  `exact_linear.scaled_taylor_action`. The previous series returned wrong
  values without warning for large parameters (for example a rotation by 40)
  and was not linear in the vector at small scales. `one_parameter_action`
  now uses it.
- Added exact checks for tables of structure constants to
  `exact_linear`: the Jacobi identity on every basis triple, trace forms,
  inertia of a symmetric form, and modular rank.
- Hardened validation: operator entries are stored as exact fractions even
  when given as equal integers, non-finite coordinates are rejected, and the
  Albert derivation-dimension certificate requires a real prime and verifies
  its upper bound.
- The Baker--Campbell--Hausdorff evaluator now takes its coefficients from the
  exact Hall-basis derivation instead of repeating them.
- Attributed the 2-by-2 magic-square entries to their sources case by case.
- Made the exact computations faster without changing results: operator
  hashes are cached and operators are canonicalised once.
- Split continuous integration into a fast lane and a full lane. Tests marked
  `slow` run only in the full lane. Subprocess tests use one generous hang
  guard instead of per-test time budgets.
- Constructions of exceptional Lie algebras built on these tools are
  maintained privately until they are published.

## 0.1.0 - locally complete source milestone (2026-08-16)

Implemented the bounded `NEW-01` finite nonassociative-algebra census and
fingerprint atlas. V01-060 passed the final readiness and milestone-closure
gate. No Git tag or package-index publication is claimed.

- Added exact arbitrary-arity sparse multilinear evaluation with explicit
  operation, parent, ordered-basis, and coercion checks.
- Added immutable content-addressed census specifications, exhaustive reference
  enumeration, sound local pruning, exact orbit classification, canonical
  labels, and constructive positive, exhaustive negative, invariant-negative,
  and bounded-inconclusive isomorphism outcomes.
- Added complete or explicitly bounded finite laws, element profiles, nuclei,
  centers, substructures, congruences, module-backed ideals and derivations,
  rejection fingerprints, and uniform evidence-role records.
- Added pinned `H`, `Hs`, `O`, and `Os` controls for conjugation, norms,
  associativity or alternativity, Moufang identities, derivations, split
  isotropy, zero divisors, idempotents, and exact basis transport.
- Added an atomic content-addressed atlas format, strict loader, deterministic
  queries, evidence replay, command-line workflow, and executable manual.
- Exhausted the declared order-three binary commutative corpus: 19,683 raw
  tables, 729 accepted labeled tables, and 129 certified relabeling orbits.
- Added deterministic property, mutation, resource-bound, hostile-input,
  traceability, artifact, and cross-install reproduction gates.

The final post-close neutral selection contains 2,423 tests: 2,421 passed with
two platform-conditional symlink skips and 302 external-evidence tests
deselected. The coverage-instrumented V01-056 run passed 2,392 tests. Coverage is
20,320/21,093 statements and 6,529/7,052 branches. Ruff lint and formatting pass
over 314 files. Strict MyPy passes over 277 files.

Temporary 0.1.0 wheel and sdist candidates built, inspected, and clean-installed
without runtime dependencies or optional mathematics imports. Both reproduced
the same 862-object, 129-representative atlas tree and query results. These
artifacts were not retained, signed, tagged, or published.

The separate AlgMul lane remains fail-closed because the active external source
does not match its historical receipt: 265 tests pass, 36 source-bound tests
fail, and one licensed-Wolfram replay skips. No v0.1 result establishes general
classification, literal AlgMul parity, or a mathematical or scientific claim
outside the declared corpus.

## 0.0.1 - 2026-08-05

Coverage-hardening patch milestone, completed and locally verified.

Distribution note: the source and current UV editable-install metadata report
0.0.1. No retained 0.0.1 release archive, Git tag, or package-index publication
is claimed by this changelog entry.

Current-tree note: post-milestone experimental Albert-algebra and compact-F4
files were present during the 2026-08-12 audit, and further experimental
research modules were added later. These files remain outside the stable root
API and closed 60-file coverage record. On 2026-08-15 the neutral current-tree selection reported 1,745 passed
and 302 legacy tests deselected. The exceptional selection passed 111 focused
tests. Ruff format and lint pass for 220 scoped `src`, `tests`, `examples`, and
`tools` files; strict MyPy passes across 188 package/test files and 23 example
files.

After that clean run, an external Mathematica 12.0 save changed the linked
parent-folder `AlgMul.wl` from its pinned receipt. A deliberately unfiltered
rerun failed 33 legacy source-binding checks for that reason; the AnyAlgebra
implementation and 111-test exceptional selection did not regress. Neutral CI
now path-marks and deselects the complete legacy lane instead of depending on
individual marker coverage.

- Added meaningful boundary and failure-path tests across the neutral core,
  generalized structures, presentations, linear algebra, validation, evidence,
  backends, and legacy capture layers.
- Fixed malformed semantic-report input handling and removed defensive checks
  proven unreachable behind validated immutable constructors.
- Added an exact canonical-source coverage gate that excludes temporary package
  copies and requires every source statement and branch to be covered.
- Reached 100 percent statement and branch coverage over all 60 canonical
  `src/anyalgebra` files.

This patch increases execution-path evidence only. It does not claim literal
AlgMul behavioral parity, prove mathematical correctness, add specialized
algebra APIs, or establish any scientific result.

## 0.0.0 - 2026-07-25

Initial neutral AnyAlgebra foundation release.

- Exact `ZZ` and `QQ` domains, immutable parents and elements, explicit
  coercions, arbitrary many-sorted structures, partial operations, finite
  multilinear structures, conversions, exact recovery, bounded validation,
  fingerprints, safe records, receipts, replay, and a reference backend.
- Executable, bounded infrastructure examples and clean-install artifact
  verification.
- Static/evaluated AlgMul capture and v0.0-owned disposition coverage,
  including the missing `MakeProperty` factory and 128 intended names.

This is not a declaration of correctness for AlgMul, a specialized Lie,
Clifford, Jordan, exceptional, geometry, or physics API, or any scientific or
portfolio-project result. Licensed Wolfram live replay remains an opt-in
maintainer check.
