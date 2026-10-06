"""Independent formulas and local brute-force reproduction of tiny corpora."""

from __future__ import annotations

from itertools import permutations, product
import json
from pathlib import Path
from typing import cast

from anyalgebra.census import CensusSpec
from anyalgebra.census.bounds import CensusOrdering, EnumerationBounds
from anyalgebra.census.constraints import CensusConstraint, ConstraintSet
from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.filtering import compile_constraint_filter
from anyalgebra.census.reference import enumerate_reference
from anyalgebra.census.spec import CensusSpecCore


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "v01_reference_corpora.json"
TOP_KEYS = frozenset(("schemaVersion", "status", "corpora"))
ENTRY_KEYS = frozenset(
    (
        "id",
        "carrierSize",
        "arity",
        "constraintProfile",
        "expectedRawCount",
        "expectedAcceptedCount",
        "independentDerivation",
        "independentMethod",
        "specHash",
    )
)


def _entries() -> tuple[dict[str, object], ...]:
    record = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert type(record) is dict and set(record) == TOP_KEYS
    assert record["schemaVersion"] == 1
    assert record["status"] == "bounded-control-evidence"
    entries = record["corpora"]
    assert type(entries) is list
    assert all(type(item) is dict and set(item) == ENTRY_KEYS for item in entries)
    assert [item["id"] for item in entries] == sorted(item["id"] for item in entries)
    return cast(tuple[dict[str, object], ...], tuple(entries))


def _constraints(profile: str) -> tuple[CensusConstraint, ...]:
    profiles = {
        "none": (),
        "identity0_two_sided": (
            CensusConstraint.identity(element=0, side="two_sided"),
        ),
        "commutative": (CensusConstraint.commutative(),),
        "idempotent": (CensusConstraint.idempotent(),),
        "commutative_idempotent": (
            CensusConstraint.commutative(),
            CensusConstraint.idempotent(),
        ),
        "quasigroup": (CensusConstraint.quasigroup(),),
    }
    return profiles[profile]


def _spec(entry: dict[str, object]) -> CensusSpec:
    size = cast(int, entry["carrierSize"])
    arity = cast(int, entry["arity"])
    core = CensusSpecCore.create(
        carrier_size=size,
        arity=arity,
        corpus_name=cast(str, entry["id"]),
    )
    return CensusSpec.create(
        carrier_size=size,
        arity=arity,
        constraints=ConstraintSet.create(
            core, _constraints(cast(str, entry["constraintProfile"]))
        ),
        equivalence=EquivalencePolicy.literal(core),
        ordering=CensusOrdering.reference(),
        bounds=EnumerationBounds.create(
            max_candidates=100_000,
            max_orbits=100_000,
            max_work_units=1_000_000,
            max_memory_bytes=100_000_000,
        ),
    )


def _latin_square_count(size: int) -> int:
    rows = tuple(permutations(range(size)))
    return sum(
        all(len({row[column] for row in table}) == size for column in range(size))
        for table in product(rows, repeat=size)
    )


def _power(base: int, exponent: int) -> int:
    result = 1
    for _ in range(exponent):
        result *= base
    return result


def _independent_count(entry: dict[str, object]) -> int:
    size = cast(int, entry["carrierSize"])
    arity = cast(int, entry["arity"])
    method = entry["independentMethod"]
    if method == "all_total_operations":
        return _power(size, _power(size, arity))
    if method == "two_sided_identity_cells":
        return _power(size, size * size - (2 * size - 1))
    if method == "unordered_binary_input_pairs":
        return _power(size, size * (size + 1) // 2)
    if method == "fixed_diagonal_cells":
        return _power(size, size * size - size)
    if method == "unordered_off_diagonal_pairs":
        return _power(size, size * (size - 1) // 2)
    if method == "independent_latin_square_rows":
        return _latin_square_count(size)
    raise AssertionError(f"unknown fixture derivation method: {method}")


def test_every_fixture_has_a_nonempty_independent_derivation_and_exact_spec_hash() -> (
    None
):
    entries = _entries()

    assert len(entries) >= 10
    for entry in entries:
        assert type(entry["independentDerivation"]) is str
        assert len(entry["independentDerivation"]) >= 24
        spec = _spec(entry)
        assert entry["specHash"] == str(spec.semantic_hash)


def test_independent_formulas_and_latin_square_enumeration_match_frozen_counts() -> (
    None
):
    for entry in _entries():
        expected = cast(int, entry["expectedAcceptedCount"])
        assert _independent_count(entry) == expected


def test_every_frozen_count_is_reproduced_by_local_unpruned_brute_force() -> None:
    for entry in _entries():
        spec = _spec(entry)
        raw = enumerate_reference(spec)
        compiled = compile_constraint_filter(spec)
        accepted = sum(compiled.evaluate(item).accepted for item in raw.candidates)

        assert raw.complete is True
        assert raw.total_candidate_count == entry["expectedRawCount"]
        assert accepted == entry["expectedAcceptedCount"]
