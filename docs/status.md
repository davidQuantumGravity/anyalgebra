# AnyAlgebra current status

**Checked:** 2026-08-30
**Authoritative for:** current implementation, verification, and distribution status

## Summary

AnyAlgebra is a **locally complete 0.1.0 source milestone**. All 60 sequential
v0.1 tasks have been implemented, reviewed, and accepted. `V01-060` passed all
eight aggregate gates. This is a local source-milestone claim, not a Git tag,
published distribution, or broad research-completion claim.

The v0.1 candidate implements the bounded `NEW-01` finite nonassociative-algebra
census and fingerprint-atlas slice. It also adds exact generic multilinear
evaluation and composition-algebra controls used to test that machinery. This
is an engineering and bounded-computation claim, not a general classification
of finite algebras or completion of the broader exceptional-physics program.

| Status dimension | Current state | Evidence or meaning |
|---|---|---|
| Source version | `0.1.0` | `src/anyalgebra/_version.py` |
| Closed source scope | 60 of 60 v0.1 tasks accepted | `V01-060` passed the aggregate readiness gate |
| Historical v0.0.1 coverage | 14,578/14,578 statements and 4,960/4,960 branches | Frozen 60-file historical ledger remains valid |
| Current standalone full suite | **2,807 passed, 2 skipped** in 647.05 seconds | Run with AlgMul and research paths deliberately set to nonexistent locations; skips are Windows symlink capability controls |
| Final v0.1 neutral suite | **2,421 passed, 2 skipped, 302 deselected** from 2,423 selected tests | The two skips are Windows symlink-creation controls; the separate legacy and licensed-backend lane is excluded |
| Coverage-instrumented suite | **2,392 passed, 2 skipped, 302 deselected** in 485.54 seconds | The frozen V01-056 run owns the measured coverage artifact |
| v0.1 source coverage | 20,320/21,093 statements (96.3353 percent), 6,529/7,052 branches (92.5837 percent), 95.3953 percent combined | `build/coverage-v01-final.json`, SHA-256 `FDBDD25FC998294B25B8DAEA685ACC5EE7215D8843F1B1367562E7BCC621E3C6` |
| Current static checks | Ruff passed across 361 files; strict MyPy passed across 313 package/test files and 33 example files | Rechecked after standalone-fixture migration |
| Traceability | 7 features, 7 actions, 6 stories, 12 tests, 5 components, and 8 APIs; zero findings | `V01-060` passed the final traceability and readiness checks |
| Reference corpus | 19,683 raw order-three binary tables; 729 commutative tables; 129 certified relabeling orbits | Complete only under the checked-in bounds and equivalence policy |
| Reference atlas | 862 built and verified objects | Includes 129 representatives, 12 associative, 7 idempotent, and 15 unique-identity matches |
| Atlas semantic hash | `sha256:245384be017e38895d684456272eddb5e81a2974220debf348f41e17561eb843` | Reproduced independently from the wheel and source archive |
| Candidate wheel | `anyalgebra-0.1.0-py3-none-any.whl`, 104 members, SHA-256 `13BCCFBA78F50EE3AE70B893EA5E57BE7F03B2FC54B70680BB02096692092484` | Temporary local artifact inspected and clean-installed |
| Candidate source archive | `anyalgebra-0.1.0.tar.gz`, 626 members, SHA-256 `9083FB88F2C6647EB241F2A6D6A1076B29DBCE5BB45B7D6FDFA4265F91045DAE` | Temporary local artifact inspected and clean-installed |
| Legacy AlgMul lane | **301 passed** with 2,508 non-legacy tests deselected | Uses committed receipts and synthetic inputs; no external AlgMul or research checkout is read |
| Git release | **No Git tag; not tagged** | A local candidate is not a tagged release |
| Publication | **Not published; publication is not claimed** | No package-index upload was attempted or verified |
| Research completion | Not claimed | `NEW-01` is a bounded computational dossier; experimental exceptional modules are not promoted to the v0.1 stable surface |

## What the candidate can do

The implemented v0.1 surface can:

- evaluate arbitrary-arity multilinear operations on exact sparse elements
  while enforcing literal-parent, operation, coercion, and expression-order
  rules;
- specify bounded finite-carrier censuses with declarative constraints,
  relabeling policies, deterministic ordering, and explicit resource ceilings;
- enumerate reference operation tables, prune by symmetry, construct canonical
  labels, and return checked isomorphism or bounded nonisomorphism outcomes;
- compute capability-aware law, unit, subobject, ideal, derivation, span, and
  fingerprint records where the subject and bounds support them;
- build, load, validate, query, and replay a versioned deterministic atlas; and
- exercise exact quaternion, split-quaternion, octonion, and split-octonion
  fixtures as finite-basis controls, including composition, alternativity,
  Moufang, isotropy, zero-divisor, idempotent, center, nucleus, derivation, and
  basis-invariance checks.

The public contracts and executable walkthrough are in
[the v0.1 API](api/api-v0.1.md) and
[the census manual](guides/finite-algebra-census.md).

## Evidence boundaries

The following claims are deliberately separate:

- **source version** is the value exported by the checked-out package source;
- **milestone complete** means every bounded task and final acceptance gate has
  passed;
- **artifact verified** means a wheel and source archive were built, inspected,
  and clean-installed locally;
- **tagged** means a corresponding Git tag exists and was inspected;
- **published** means the distribution was verified on its intended package
  index; and
- **research complete** applies only to an explicitly bounded project dossier
  with reproducible evidence.

At this checkpoint the source is 0.1.0, both candidate artifact formats are
verified, and the bounded local source milestone is complete. It is not tagged,
not published, and does not establish a broad mathematical or physical result.

## Experimental exceptional work

Modules under `anyalgebra.experimental` remain outside the stable v0.1 API.
They include bounded Albert/F4, magic-square, BCH, and related
exact-certificate experiments. Their exact checks do not by themselves prove
general construction theorems or a physics model. Stable API promotion still
requires a separate contract, independent identification gates, and explicit
claim scope.

The external `AlgMul.wl` working source is 194,003 bytes with SHA-256
`2752CF4A6146E97CAE0EB093725641F9A03D9AF2410D30CBBA7721594C60F98E`.
The historical audit pins a different 194,012-byte source. The 36 legacy
failures preserve that mismatch as evidence; they are not neutral v0.1 product
failures and are not silently redefined as parity.

## Reproduce the candidate checks

From the repository root:

```powershell
.venv\Scripts\python -m pytest -q -m "not legacy and not optional_backend"
.venv\Scripts\python tools/audit_v0_1_traceability.py --no-execute
.venv\Scripts\python tools/check_v0_1_artifacts.py
.venv\Scripts\python tools/reproduce_v0_1_atlas.py
.venv\Scripts\python tools/validate_project_contract.py .
```

The artifact and reproduction tools build temporary isolated environments.
They do not tag Git, retain a signed release bundle, or publish a distribution.
