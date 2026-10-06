# Using quaternions and octonions in AnyAlgebra

**Checked:** 2026-08-15  
**Source version:** 0.0.1; v0.1 implementation worktree

AnyAlgebra currently supplies exact named multiplication-table fixtures for
the quaternions, split quaternions, octonions, and split octonions. These are
instances of the generic `FiniteMultilinearStructure` kernel. They are not yet
specialized element classes, so `x * y` does not mean algebra multiplication.
The v0.1 worktree now provides `evaluate_multilinear` for exact arbitrary
sparse elements. This is implemented and tested, but v0.1 is not yet a closed
release.

The experimental exceptional-algebra module also has a more convenient
coordinate-level octonion multiplication function. That API supports the
Albert and compact `F4` research code, but it may change.

## Install and run

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

The root package exports only `__version__`. Import mathematical objects from
their owning modules.

## Load the four exact table conventions

```python
from anyalgebra.fixtures.composition import (
    octonion_fixture,
    quaternion_fixture,
    split_octonion_fixture,
    split_quaternion_fixture,
)

H = quaternion_fixture()
Hs = split_quaternion_fixture()
O = octonion_fixture()
Os = split_octonion_fixture()

assert H.name == "algmul.H.v1"
assert H.module.basis.labels == ("1", "i", "j", "k")
assert O.name == "algmul.O.v1"
assert O.module.basis.labels == ("1", "e1", "e2", "e3", "e4", "e5", "e6", "e7")
```

The convention identifiers are deliberate. They prevent a short name such as
`O` from silently selecting an orientation.

## Multiply basis elements

`evaluate_basis(a, b)` returns the exact sparse element represented by the
product of basis vectors at indices `a` and `b`.

```python
H = quaternion_fixture()
assert H.evaluate_basis(1, 2) == H.module.element({3: 1})  # i*j = k
assert H.evaluate_basis(2, 1) == H.module.element({3: -1})  # j*i = -k
assert H.evaluate_basis(1, 1) == H.module.element({0: -1})  # i^2 = -1

Hs = split_quaternion_fixture()
assert Hs.evaluate_basis(1, 1) == Hs.module.element({0: -1})
assert Hs.evaluate_basis(2, 2) == Hs.module.element({0: 1})
assert Hs.evaluate_basis(3, 3) == Hs.module.element({0: 1})

O = octonion_fixture()
assert O.evaluate_basis(1, 2) == O.module.element({3: 1})  # e1*e2 = e3
assert O.evaluate_basis(2, 1) == O.module.element({3: -1})  # e2*e1 = -e3

Os = split_octonion_fixture()
assert Os.evaluate_basis(4, 4) == Os.module.element({0: 1})  # e4^2 = +1
```

All coefficients here belong to the exact integer domain `ZZ`.

## Multiply arbitrary sparse elements

The kernel intentionally does not infer element multiplication from `x * y`.
Call the explicit evaluator so the operation, parent ownership, coercion
authority, and parentheses remain visible. This block is executed verbatim by
`tests/examples/test_multilinear_evaluation.py`.

```python
from anyalgebra.algebra.multilinear import evaluate_multilinear
from anyalgebra.fixtures.composition import octonion_fixture, quaternion_fixture

H = quaternion_fixture()
x = H.module.element({0: 1, 1: 2})  # 1 + 2i
y = H.module.element({2: 1, 3: 1})  # j + k
assert evaluate_multilinear(H, x, y) == H.module.element({2: -1, 3: 3})

O = octonion_fixture()
x = O.module.element({0: 1, 1: 1})  # 1 + e1
y = O.module.element({2: 1, 4: 1})  # e2 + e4
assert evaluate_multilinear(O, x, y) == O.module.element({2: 1, 3: 1, 4: 1, 5: 1})
```

The evaluator supports nullary and higher-arity multilinear maps as well as
binary products. It checks the declared arity and literal parents before
expansion and does not reassociate nested calls. An explicit `CoercionGraph`
can authorize coordinate-preserving scalar extension only when ordered basis
labels match exactly.

Run the complete introductory example:

```powershell
python examples/multilinear_evaluation.py
```

It constructs an unrelated nonassociative rank-two table, evaluates a ternary
map, computes the sparse products above, verifies left and right identity on
all four named fixtures, and exhibits exact zero-divisor products in `Hs` and
`Os`. These are finite-table controls, not classification or physics claims.

## Compute an associator

The public elementary-analysis API evaluates generic sparse elements and keeps
the parentheses explicit.

```python
from anyalgebra.analysis.elementary import associator

O = octonion_fixture()
e1 = O.module.element({1: 1})
e2 = O.module.element({2: 1})
e4 = O.module.element({4: 1})

value = associator(O, e1, e2, e4)
assert value == O.module.element({7: 2})
```

Thus, in the `algmul.O.v1` convention,

```text
(e1*e2)*e4 - e1*(e2*e4) = 2*e7.
```

The executable search finds this as the first nonzero associator after 85
lexicographically ordered basis triples:

```powershell
python examples/octonion_counterexample.py
```

## Analyze all four fixtures

```powershell
python examples/generic_composition_algebras.py
```

The example lifts the integral tables to the exact rational domain `QQ`, then
uses the generic fingerprint and elementary-analysis code. The confirmed
results are:

| Fixture | Dimension | Associativity | Nucleus | Center | Derivations |
|---|---:|---|---:|---:|---:|
| `algmul.H.v1` | 4 | proved | 4 | 1 | 3 |
| `algmul.Hs.v1` | 4 | proved | 4 | 1 | 3 |
| `algmul.O.v1` | 8 | disproved | 1 | 1 | 14 |
| `algmul.Os.v1` | 8 | disproved | 1 | 1 | 14 |

“Proved” here has a precise finite-table meaning. Bilinearity makes the
associator trilinear, so checking all `4^3 = 64` ordered basis triples proves
associativity for every rational linear combination in each quaternion
fixture. A single explicit basis witness disproves associativity for each
octonion fixture.

The fingerprint is bounded isomorphism-rejection data. It is not a complete
isomorphism invariant, and its ideal search reports that its canonical ideal
discovery is incomplete.

## Use the experimental coordinate octonions

The exceptional research module exposes direct coordinate multiplication for
octonions. Coordinates are ordered as `(1, e1, ..., e7)`.

```python
from anyalgebra.experimental.exceptional import (
    octonion_basis,
    octonion_multiply,
)

one, e1, e2, e3, e4, e5, e6, e7 = octonion_basis()
assert octonion_multiply(e1, e2) == e3

x = tuple(e1[i] + e2[i] for i in range(8))
assert octonion_multiply(x, e4) == tuple(e5[i] + e6[i] for i in range(8))
```

This implementation is used inside the 27-coordinate Albert algebra
`H_3(O)`.

## Focused test commands

Run the generic quaternion and octonion suite:

```powershell
python -m pytest -q `
  tests/examples/test_multilinear_evaluation.py `
  tests/fixtures/test_composition_tables.py `
  tests/examples/test_generic_composition_algebras.py `
  tests/integration/test_validation_basis_invariance.py `
  tests/analysis/test_fingerprint.py::test_composition_derivation_controls_over_qq `
  tests/presentations/test_symbolic_coordinate_table.py::test_octonion_table_and_structure_cells_round_trip `
  tests/maps/test_map_records.py::test_generic_map_accepts_current_finite_multilinear_structure_parents `
  tests/performance/test_catastrophic_regressions.py::test_composition_fixture_and_fingerprint_work_remain_bounded `
  tests/coverage/test_v001_misc_guards.py::test_fano_private_precondition_failure_is_explicit
```

Current result:

```text
39 passed in 6.14s
```

Run the octonion-dependent exceptional suite:

```powershell
python -m pytest -q `
  tests/experimental `
  tests/examples/test_f4_albert_certificate.py
```

The first suite tests the stable generic representation and fixtures plus the
in-progress v0.1 evaluator. The second tests experimental constructions that
depend on the declared octonion convention.

## Complete generic-fixture test inventory

### Public sparse evaluation: 5 cases

File: `tests/examples/test_multilinear_evaluation.py`

- Constructs an unrelated nonassociative table and a ternary map through the
  generic kernel, then checks the explicit nested and ternary results.
- Evaluates quaternion and octonion sparse products through
  `evaluate_multilinear` and verifies split-quaternion and split-octonion zero
  divisors.
- Checks two-sided identity on all four convention-pinned fixtures.
- Repeats the report and executes the example as a subprocess to require
  deterministic canonical JSON.
- Extracts and executes the guide's public evaluator Python block verbatim and
  checks that the capability page preserves the worktree/release distinction.

### Exact tables and conventions: 14 cases

File: `tests/fixtures/test_composition_tables.py`

- `test_named_fixtures_have_literal_basis_unity_and_declared_squares`, four
  cases. Confirms the exact parent, basis labels, unit, and every declared
  generator square for `H`, `Hs`, `O`, and `Os`.
- `test_quaternion_and_split_quaternion_orientations_are_explicit`. Confirms
  orientation and sign choices in both four-dimensional tables.
- `test_octonion_fano_rule_and_split_octonion_literal_table_are_auditable`.
  Checks all seven positive Fano triples and every one of the 64 cells in an
  independently transcribed split-octonion table.
- `test_all_fixture_imaginary_generators_anticommute`, four cases. Checks every
  distinct pair of imaginary generators in all four fixtures.
- `test_quaternion_fixture_is_associative_and_octonion_has_an_explicit_failure`.
  Checks all 64 quaternion basis associators and the explicit octonion witness.
- `test_generic_selector_and_invalid_fixture_identifier_boundary`. Confirms
  exact fixture identifiers and rejects aliases and invalid input types.
- `test_fixture_metadata_is_frozen_anchored_and_exact_id_selected`. Confirms
  immutable metadata and live convention-document anchors.
- `test_fixture_calls_construct_independent_generic_structure_parents`.
  Confirms repeated loads do not merge literal parents and reproduce the same
  table.

### Executable exact analysis: 12 cases

File: `tests/examples/test_generic_composition_algebras.py`

- `test_each_named_table_is_loaded_as_one_generic_qq_structure`, four cases.
- `test_associative_fixtures_have_complete_theorem_backed_reports`.
- `test_nonassociative_fixtures_have_deterministic_minimal_witnesses`.
- `test_report_states_finite_grid_infinite_coefficient_scope_and_bounds`.
- `test_sampled_pass_remains_inconclusive_and_cannot_replace_the_exact_report`.
- `test_exact_basis_change_preserves_only_the_declared_fingerprint_data`.
- `test_invalid_fixture_id_is_a_typed_boundary`.
- `test_executable_report_retains_the_invalid_identifier_boundary`.
- `test_cli_is_deterministic_json_and_claim_bounded`.

These checks confirm complete rational associativity reports for `H` and `Hs`,
minimal nonassociativity witnesses for `O` and `Os`, declared resource bounds,
honest sampling status, exact quaternion basis-change invariance, typed input
failures, and deterministic machine-readable output.

### Basis transport and executable witness: 3 cases

File: `tests/integration/test_validation_basis_invariance.py`

- `test_exact_shear_preserves_fingerprint_and_all_octonion_basis_associators`.
  Transports the octonion table through a non-permutation rational shear and
  checks all 512 basis associators exactly.
- `test_exact_shear_preserves_an_associative_fixture_at_its_complete_boundary`.
  Repeats the complete 64-triple quaternion associativity check after the
  exact shear.
- `test_octonion_counterexample_example_is_executable`. Confirms the witness,
  convention identifier, theorem hypothesis, and printed result.

### Analysis, presentation, API, and resource guards: 5 cases

- `tests/analysis/test_fingerprint.py::test_composition_derivation_controls_over_qq`
  confirms derivation dimensions 3 for `H` and 14 for `O`.
- `tests/presentations/test_symbolic_coordinate_table.py::test_octonion_table_and_structure_cells_round_trip`
  converts all 64 octonion cells from structure to table and back.
- `tests/maps/test_map_records.py::test_generic_map_accepts_current_finite_multilinear_structure_parents`
  confirms that generic map records accept distinct quaternion structure
  parents without merging them.
- `tests/performance/test_catastrophic_regressions.py::test_composition_fixture_and_fingerprint_work_remain_bounded`
  evaluates all 160 basis-product cells across the four fixtures and checks a
  deterministic bounded quaternion fingerprint.
- `tests/coverage/test_v001_misc_guards.py::test_fano_private_precondition_failure_is_explicit`
  confirms a typed error if an impossible Fano lookup reaches the private
  precondition guard.

The last three are supporting API, performance, and defensive tests. They do
not establish new quaternion or octonion identities.

## Complete octonion-dependent exceptional test inventory

### Albert algebra and compact F4: 9 cases

File: `tests/experimental/test_exceptional.py`

- `test_albert_basis_and_identity_are_exact` checks the 27-coordinate basis,
  trace-three identity, and identity action.
- `test_jordan_product_is_commutative_and_has_the_jordan_identity_on_basis`
  checks commutativity and a bounded basis-vector Jordan identity grid.
- `test_jordan_associator_is_the_inner_derivation_action` checks the stated
  associator and multiplication-operator commutator relation.
- `test_all_52_generators_are_exact_derivations_and_compact` checks the
  Leibniz rule, trace-form skewness, and annihilation of the identity.
- `test_full_derivation_space_has_dimension_52_by_two_sided_rank_bounds`
  checks rank 677 and nullity 52 through modular and rational calculations.
- `test_compact_trace_gram_is_exactly_positive_definite` checks 52 positive
  exact LDL pivots.
- `test_f4_generator_commutators_close_in_the_52_dimensional_span` checks all
  1,378 unordered pairs, including the diagonal.
- `test_exponentiated_generator_preserves_the_three_basic_invariants` checks
  numerical preservation of trace, trace form, and Albert determinant.
- `test_algmul_nested_action_order_is_explicit_on_octonions` distinguishes
  left-nested and right-nested octonion actions.

File: `tests/examples/test_f4_albert_certificate.py`, 3 additional cases.
These check the deterministic exact dimension, closure, compactness receipt,
the bounded numerical exponential receipt, and canonical JSON command output.

## Evidence tests that use octonion examples

Seven additional tests in `tests/evidence/test_contracts.py` and
`tests/evidence/test_sources_conventions.py` use an octonion-associator
calculation or `algmul.H.v1` and `algmul.O.v1` identifiers as evidence payloads.
They test canonical serialization, content addressing, immutability, manifest
references, and round trips. They do not evaluate quaternion or octonion
multiplication, so they are not counted in the mathematical and downstream
cases above.

## Important gaps

The current tests do not yet directly establish the following as stable,
named-family capabilities:

- a public `Quaternion` or `Octonion` element class with `x * y` syntax;
- direct stable APIs for conjugation, real and imaginary parts, norms, inverses,
  or division;
- exhaustive alternativity, flexibility, Moufang, and norm-composition tests on
  all four named fixtures;
- explicit zero-divisor and idempotent catalogues for the split algebras;
- Cayley-Dickson and Zorn-matrix construction equivalence;
- `G2` as the octonion derivation or automorphism group with representation
  data;
- tensor products such as `A tensor J_n(B)` through a stable general API;
- a stable matrix algebra over tensor products of composition algebras.

The generic law engine has some of the lower-level machinery needed for these
checks. Applying it systematically to `H`, `Hs`, `O`, and `Os`, then adding a
small ergonomic composition-algebra layer, is the clearest next vertical
slice.
