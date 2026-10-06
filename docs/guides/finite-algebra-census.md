# Finite algebra census and atlas manual

**Checked:** 2026-08-15  
**Source state:** v0.1 implementation worktree; release gates are not yet closed

This manual shows the usable v0.1 calculation path from exact sparse algebra
elements to a bounded, content-addressed finite-operation atlas. Every Python
block below is executed from an unrelated working directory by
`tests/docs/test_v0_1_manual.py`; every result shown in code is an assertion.

The examples are deliberately small. Their purpose is to make the semantics,
proof obligations, and failure boundaries inspectable. They are engineering
and finite-mathematics controls, with no physics claim.

## Two mathematical domains that must not be confused

A **finite-basis vector algebra** has a coefficient domain, an ordered basis,
and multilinear structure constants. Its elements are sparse linear
combinations, and bilinearity or multilinearity extends basis products to all
elements.

A **finite-carrier operation table** is a set such as `{0, 1, 2}` with one
declared total operation. It has no addition, scalar multiplication, ideals,
or derivations unless those structures are supplied separately. Its flat output
tuple lists the value at each lexicographically ordered input tuple.

The same word “algebra” is commonly used for both. AnyAlgebra keeps their
parents and APIs distinct so a carrier census cannot silently acquire linear
structure.

## Evidence vocabulary

The package distinguishes the following statements:

- A **proof within a declared finite domain** exhausts the exact grid named by
  the result. For a multilinear identity, a complete basis-grid reduction can
  extend the result to every linear combination under its stated hypotheses.
- A **counterexample** is a concrete assignment where an identity fails. It
  disproves that identity for the declared structure without classifying other
  structures.
- An **exhaustive corpus result** accounts for every candidate in one pinned
  carrier, arity, constraint set, equivalence group, ordering, and bound set.
- A result marked **bounded and inconclusive** stopped with work remaining. It
  is neither a proof nor a negative classification.
- A **rejection-only invariant** can prove that two objects are different when
  it mismatches. Equality of that invariant is not an isomorphism proof.
- **Experimental context** may motivate a calculation, but experimental code
  and AlgMul output are cross-check material rather than correctness oracles.

## Stable v0.1 calculation map

| Purpose | Current callable | Meaning |
|---|---|---|
| Sparse exact evaluation | `evaluate_multilinear` | Expand the declared immutable tensor without inferring `x*y` or changing parentheses |
| Freeze a corpus | `CensusSpec.create` | Bind carrier, arity, constraints, relabelings, ordering, conventions, and ceilings |
| Reference traversal | `enumerate_reference` | Visit flat tables in exact rank order and retain an honest frontier |
| Canonical label | `canonicalize_operation_table` | Minimize under the explicitly allowed permutation group |
| Compare tables | `compare_isomorphism` | Return an invariant rejection, checked map, exhaustive no-map receipt, or bounded frontier |
| Capability-aware analysis | `assemble_finite_carrier_analysis` | Separate complete, bounded, rejection-only, and unsupported fields |
| Publish an atlas | `build_finite_algebra_atlas` | Compose census, classification, analysis, certificates, and receipts atomically |
| Load and filter | `query_atlas` | Query a fully verified non-executing object store |

These are the concrete worktree callables. The version-level API document uses
role-oriented façade names in a few places; registry reconciliation is a later
v0.1 release task and must not be mistaken for functionality that already
exists.

## 1. Evaluate exact sparse elements

The quaternion fixture is a finite-basis vector algebra over the exact integer
domain. The explicit evaluator checks literal parent ownership before doing any
coefficient work.

```python
from anyalgebra.algebra.multilinear import (
    MultilinearEvaluationError,
    evaluate_multilinear,
)
from anyalgebra.fixtures.composition import quaternion_fixture

H = quaternion_fixture()
x = H.module.element({0: 1, 1: 2})  # 1 + 2i
y = H.module.element({2: 1, 3: 1})  # j + k
product = evaluate_multilinear(H, x, y)

assert product == H.module.element({2: -1, 3: 3})
assert H.evaluate_basis(1, 2) == H.module.element({3: 1})
assert H.evaluate_basis(2, 1) == H.module.element({3: -1})

other_H = quaternion_fixture()
try:
    evaluate_multilinear(H, x, other_H.module.element({0: 1}))
except MultilinearEvaluationError as error:
    assert error.code == "parent"
    assert error.argument_index == 1
else:
    raise AssertionError("a same-shaped but distinct parent was accepted")
```

This is exact expansion, not operator overloading. Higher arity and nullary
maps use the same evaluator. A coercion graph can authorize only an explicit
coordinate-preserving scalar extension between equal ordered bases.

## 2. Declare, enumerate, constrain, and classify a corpus

For a binary operation on two labelled elements there are four table cells and
therefore `2^4 = 16` raw tables. Commutativity requires the two off-diagonal
cells to agree, leaving `2^3 = 8` accepted labelled tables. Classification
below is under the full two-element relabeling group, not under an unstated
notion of equivalence.

```python
from anyalgebra.census.bounds import CensusOrdering, EnumerationBounds
from anyalgebra.census.classify import classify_reference_orbits
from anyalgebra.census.constraints import CensusConstraint, ConstraintSet
from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.reference import enumerate_reference
from anyalgebra.census.spec import CensusSpec, CensusSpecCore

core = CensusSpecCore.create(
    carrier_size=2,
    arity=2,
    corpus_name="manual-commutative-order-two",
)
spec = CensusSpec.create(
    carrier_size=2,
    arity=2,
    constraints=ConstraintSet.create(core, (CensusConstraint.commutative(),)),
    equivalence=EquivalencePolicy.relabeling(core),
    ordering=CensusOrdering.reference(),
    bounds=EnumerationBounds.create(
        max_candidates=16,
        max_orbits=16,
        max_work_units=16,
        max_memory_bytes=100_000,
    ),
)

enumeration = enumerate_reference(spec)
classification = classify_reference_orbits(enumeration)

assert enumeration.complete is True
assert enumeration.total_candidate_count == 2**4 == 16
assert tuple(item.candidate_index for item in enumeration.candidates) == tuple(
    range(16)
)
assert classification.accepted_labeled_count == 2**3 == 8
assert classification.rejected_labeled_count == 8
assert classification.orbit_count == 4
assert sum(orbit.accepted_member_count for orbit in classification.orbits) == 8
assert all(
    member.verification.transport_verified and member.verification.identifier_verified
    for orbit in classification.orbits
    for member in orbit.members
)
```

The reference enumerator intentionally visits all 16 raw tables. Constraints
are reapplied before orbit membership, making the raw, accepted, and rejected
counts independently visible.

## 3. Canonicalize, compare, and analyze one table

The XOR table `(0, 1, 1, 0)` is the group operation of the cyclic group of
order two. Canonicalization gives an identifier only relative to the declared
core and equivalence policy. A positive comparison still carries a checked
map. A negative comparison below is certified by an exact invariant mismatch,
while a zero-permutation budget remains inconclusive for a harder pair.

```python
from anyalgebra.census.analysis_records import assemble_finite_carrier_analysis
from anyalgebra.census.canonical import canonicalize_operation_table
from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.isomorphism import (
    Inconclusive,
    InvariantNonIsomorphic,
    Isomorphic,
    compare_isomorphism,
)
from anyalgebra.census.spec import CensusSpecCore

core = CensusSpecCore.create(
    carrier_size=2,
    arity=2,
    corpus_name="manual-xor-analysis",
)
policy = EquivalencePolicy.relabeling(core)
xor = (0, 1, 1, 0)
zero = (0, 0, 0, 0)

label = canonicalize_operation_table(core, xor, policy)
positive = compare_isomorphism(core, xor, xor, policy)
negative = compare_isomorphism(core, zero, xor, policy)
bounded = compare_isomorphism(
    core,
    (1, 0, 1, 0),
    (1, 1, 0, 0),
    policy,
    max_permutations=0,
)
analysis = assemble_finite_carrier_analysis(core, xor, policy)
fields = {field.name: field.status for field in analysis.fields}
laws = {result.name: result.status for result in analysis.law_profile.results}

assert label.canonical_outputs <= xor
assert isinstance(positive, Isomorphic)
assert positive.witness.verify_cells() is True
assert isinstance(negative, InvariantNonIsomorphic)
assert negative.permutations_examined == 0
assert isinstance(bounded, Inconclusive)
assert bounded.complete is False and bounded.remaining_count > 0
assert laws["associativity"] == "proved_on_complete_grid"
assert laws["commutativity"] == "proved_on_complete_grid"
assert fields["finite_carrier_laws"] == "complete"
assert fields["invariant_colors"] == "rejection_only"
assert fields["linear_ideals"] == "unsupported"
```

The final three assertions matter as much as the algebraic answer. The carrier
law grid is complete, the fingerprint role is rejection-only, and a bare
two-element carrier does not acquire linear ideals.

## 4. Use H, Hs, O, and Os through generic operations

The four convention-pinned fixtures are ordinary quaternions, split
quaternions, octonions, and split octonions. Conjugation, real part, trace,
quadratic norm, and multiplication are attached only after the exact fixture
table is recognized.

```python
from anyalgebra.fixtures.composition import (
    octonion_fixture,
    quaternion_fixture,
    split_octonion_fixture,
    split_quaternion_fixture,
)
from anyalgebra.fixtures.composition_ops import composition_operations

fixtures = (
    quaternion_fixture(),
    split_quaternion_fixture(),
    octonion_fixture(),
    split_octonion_fixture(),
)
assert tuple(item.name for item in fixtures) == (
    "algmul.H.v1",
    "algmul.Hs.v1",
    "algmul.O.v1",
    "algmul.Os.v1",
)

for algebra in fixtures:
    operations = composition_operations(algebra)
    value = algebra.module.element({0: 2, 1: -3})
    assert operations.conjugate(operations.conjugate(value)) == value
    assert operations.trace(value).value == 4

for algebra, generator in (
    (fixtures[1], 2),  # j^2 = +1 in Hs
    (fixtures[3], 4),  # e4^2 = +1 in Os
):
    operations = composition_operations(algebra)
    plus = algebra.module.element({0: 1, generator: 1})
    minus = algebra.module.element({0: 1, generator: -1})
    assert operations.quadratic_norm(plus).value == 0
    assert operations.quadratic_norm(minus).value == 0
    assert operations.multiply(plus, minus) == algebra.module.zero()
    assert operations.multiply(minus, plus) == algebra.module.zero()

for algebra in (fixtures[0], fixtures[2]):
    operations = composition_operations(algebra)
    plus = algebra.module.element({0: 1, 1: 1})
    minus = algebra.module.element({0: 1, 1: -1})
    assert operations.quadratic_norm(plus).value == 2
    assert operations.multiply(plus, minus) == algebra.module.element({0: 2})
```

The split witnesses prove isotropy and zero divisors in these two pinned split
forms. The ordinary controls prevent the false conclusion that the same
witness pattern holds in H or O. The package also has complete basis-reduction
certificates for quaternion associativity and octonion alternativity,
flexibility, Moufang identities, and norm composition. Those certificates are
bounded exact computations, not a classification of all composition algebras.

## 5. Build, verify, and query an atlas

Atlas publication uses a fresh absolute directory. Construction happens in a
private sibling directory and becomes visible through one rename. Loading then
checks canonical bytes, schema versions, content hashes, the exact file set,
and every cross-record role before returning a `LoadedAtlas`.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from anyalgebra.atlas.build import build_finite_algebra_atlas
from anyalgebra.atlas.query import load_finite_algebra_atlas, query_atlas
from anyalgebra.census.bounds import CensusOrdering, EnumerationBounds
from anyalgebra.census.constraints import ConstraintSet
from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.spec import CensusSpec, CensusSpecCore

core = CensusSpecCore.create(
    carrier_size=2,
    arity=2,
    corpus_name="manual-atlas-order-two",
)
spec = CensusSpec.create(
    carrier_size=2,
    arity=2,
    constraints=ConstraintSet.create(core, ()),
    equivalence=EquivalencePolicy.relabeling(core),
    ordering=CensusOrdering.reference(),
    bounds=EnumerationBounds.create(
        max_candidates=16,
        max_orbits=16,
        max_work_units=100,
        max_memory_bytes=1_000_000,
    ),
)

with TemporaryDirectory() as raw:
    target = Path(raw) / "atlas"
    build = build_finite_algebra_atlas((spec,), output_directory=target)
    loaded = load_finite_algebra_atlas(target)
    all_matches = query_atlas(loaded, carrier_size=2, arity=2)
    associative = query_atlas(loaded, laws=("associativity",))

    assert build.atlas.header.status == "complete"
    assert build.atlas.corpora[0].accepted_labeled_count == 16
    assert build.atlas.header.representative_count == 10
    assert loaded.verified_object_count == build.object_count
    assert len(all_matches.matches) == 10
    assert associative.matches
    assert all(
        item.law_statuses["associativity"] == "proved_on_complete_grid"
        for item in associative.matches
    )
```

Queries do not execute serialized payloads. They filter verified records by
carrier size, arity, canonical identifier, allow-listed proved laws, and
allow-listed carrier invariants.

## 6. Wrap a build in a calculation contract and replay it

The evidence layer freezes input hashes, algorithms, bounds, assumptions,
cross-checks, and expected artifacts before execution. Repeating the same
semantic calculation in a different environment reproduces the atlas identity
and gives an exact replay report. Execution evidence remains at E1 and does not
promote a scientific claim.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from anyalgebra.atlas.evidence import (
    replay_atlas_calculation,
    run_atlas_calculation,
)
from anyalgebra.census.bounds import CensusOrdering, EnumerationBounds
from anyalgebra.census.constraints import ConstraintSet
from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.spec import CensusSpec, CensusSpecCore
from anyalgebra.evidence.run import ExecutionEnvironment

core = CensusSpecCore.create(
    carrier_size=1,
    arity=2,
    corpus_name="manual-replay-singleton",
)
spec = CensusSpec.create(
    carrier_size=1,
    arity=2,
    constraints=ConstraintSet.create(core, ()),
    equivalence=EquivalencePolicy.relabeling(core),
    ordering=CensusOrdering.reference(),
    bounds=EnumerationBounds.create(
        max_candidates=1,
        max_orbits=1,
        max_work_units=100,
        max_memory_bytes=100_000,
    ),
)

with TemporaryDirectory() as raw:
    root = Path(raw)
    first = run_atlas_calculation(
        (spec,),
        output_directory=root / "first",
        environment=ExecutionEnvironment.create(platform="first"),
    )
    second = run_atlas_calculation(
        (spec,),
        output_directory=root / "second",
        environment=ExecutionEnvironment.create(platform="second"),
    )
    replay = replay_atlas_calculation(first.receipt, second.receipt)

    assert first.build.atlas.semantic_hash == second.build.atlas.semantic_hash
    assert first.contract == second.contract
    assert first.receipt.evidence_tier == "E1-executed"
    assert first.receipt.mathematical_outcome == "verified_within_domain"
    assert replay.status == "exact_match"
    assert replay.stale_edges == ()
```

“Verified within domain” means exactly the singleton corpus declared by this
contract. It is not a theorem about arbitrary algebras and is no physics claim.

## 7. Preserve an interrupted frontier

A candidate ceiling smaller than the raw corpus returns a deterministic
frontier. Classification refuses it because not all tables have been examined.
The partial run is bounded and inconclusive rather than negative or complete.

```python
from anyalgebra.census.bounds import CensusOrdering, EnumerationBounds
from anyalgebra.census.classify import (
    OrbitClassificationError,
    classify_reference_orbits,
)
from anyalgebra.census.constraints import ConstraintSet
from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.reference import enumerate_reference
from anyalgebra.census.spec import CensusSpec, CensusSpecCore

core = CensusSpecCore.create(
    carrier_size=2,
    arity=2,
    corpus_name="manual-bounded-prefix",
)
spec = CensusSpec.create(
    carrier_size=2,
    arity=2,
    constraints=ConstraintSet.create(core, ()),
    equivalence=EquivalencePolicy.relabeling(core),
    ordering=CensusOrdering.reference(),
    bounds=EnumerationBounds.create(
        max_candidates=3,
        max_orbits=16,
        max_work_units=100,
        max_memory_bytes=1_000_000,
    ),
)
prefix = enumerate_reference(spec)

assert prefix.complete is False
assert prefix.examined_count == 3
assert prefix.next_candidate_index == 3
assert prefix.stop_reason == "max_candidates"
try:
    classify_reference_orbits(prefix)
except OrbitClassificationError as error:
    assert error.field == "enumeration"
else:
    raise AssertionError("an incomplete stream was promoted to classification")
```

## The shipped order-three reference atlas

`examples/finite_algebra_atlas.py` is the largest complete binary carrier
control admitted by the current one-million-table reference batch ceiling:

- A three-element binary table has nine cells: `3^9 = 19,683` raw tables.
- Commutativity leaves six unordered input pairs: `3^6 = 729` accepted labelled
  tables.
- Full `S3` relabeling partitions those 729 tables into 129 certified orbits.
- The published example retains 729 member certificates and 862 verified
  content-addressed objects.
- Its analysis finds 12 associative, 129 commutative, 129 flexible, and 7
  idempotent orbits; 15 have a unique identity and 15 have a unique zero.

These counts are asserted in `tests/examples/test_finite_algebra_atlas.py`, and
the atlas is rebuilt, reloaded, queried, and hash-checked there. The scope is
only binary commutative operation tables on the labelled carrier `{0,1,2}`
modulo full carrier relabeling. It is not a census of all order-three magmas,
all finite algebras, exceptional structures, string vacua, or physical models.

Run the example into a fresh directory with the documented command-line
interface. The command itself is not presented as a transcript because output
paths are intentionally user-selected and every displayed result in this
manual must remain asserted by executable Python.

## Where to go next

- [Quaternion and octonion guide](quaternions-and-octonions.md) inventories the
  named composition fixtures and their generic and experimental tests.
- [v0.1 work bounds](../testing/v0.1-work-bounds.md) lists exact candidate,
  permutation, law-grid, subset, partition, and byte ceilings.
- [Evidence model](../evidence/evidence-model.md) defines E0–E4 engineering
  evidence and the independent scientific-claim states.
- [v0.1 API contract](../api/api-v0.1.md) records the version-level target
  surface. Until the milestone closes, this manual’s executable imports are the
  more reliable guide to the current worktree.
- [AlgMul parity contract](../legacy/algmul-parity.md) explains what is migrated,
  redesigned, deferred, or retained only as historical evidence.

The current atlas is a test bed for the generic machinery. It does not validate
E8, a brane model, an octic invariant, or any proposed unification. Those are
future projects that require their own conventions, independent derivations,
contracts, and claim review.
