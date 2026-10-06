"""Evidence-shape contracts for finite-table isomorphism outcomes."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
from typing import cast

import pytest

from anyalgebra.census.canonical import canonicalize_operation_table
from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.isomorphism import (
    Inconclusive,
    Isomorphic,
    IsomorphismOutcomeError,
    NonIsomorphic,
    isomorphism_outcome_canonical_bytes,
    isomorphism_outcome_record,
)
from anyalgebra.census.permutations import generate_allowed_permutations
from anyalgebra.census.spec import CensusSpecCore
from anyalgebra.census.transport import (
    TableTransportCertificate,
    transport_operation_table,
)
from anyalgebra.core.parents import SemanticHash


def _core(size: int = 2, arity: int = 2) -> CensusSpecCore:
    return CensusSpecCore.create(
        carrier_size=size,
        arity=arity,
        corpus_name=f"isomorphism-outcomes-{size}-{arity}",
    )


def _positive() -> Isomorphic:
    core = _core()
    policy = EquivalencePolicy.relabeling(core)
    certificate = transport_operation_table(
        core,
        (0, 0, 0, 1),
        generate_allowed_permutations(policy)[1],
    )
    return Isomorphic.from_transport(certificate)


def _negative() -> NonIsomorphic:
    core = _core()
    policy = EquivalencePolicy.relabeling(core)
    left = canonicalize_operation_table(core, (0, 0, 0, 0), policy)
    right = canonicalize_operation_table(core, (0, 0, 0, 1), policy)
    assert left.canonical_id != right.canonical_id
    return NonIsomorphic.from_canonical_labels(left, right)


def _inconclusive() -> Inconclusive:
    core = _core(3, 1)
    policy = EquivalencePolicy.relabeling(core)
    return Inconclusive.create(
        core=core,
        equivalence=policy,
        left_outputs=(2, 2, 0),
        right_outputs=(1, 0, 0),
        bound_name="max_permutations",
        bound_limit=2,
        examined_count=2,
        remaining_count=4,
    )


def test_isomorphic_requires_and_retains_a_verified_concrete_bijection() -> None:
    result = _positive()

    assert result.status == "isomorphic"
    assert result.complete is True
    assert result.witness.verify_cells() is True
    assert result.left_outputs == result.witness.source_outputs
    assert result.right_outputs == result.witness.target_outputs
    assert result.permutation_images == result.witness.permutation.images


def test_a_matching_fingerprint_or_arbitrary_object_cannot_construct_positive() -> None:
    matched_fingerprint = object()
    with pytest.raises(IsomorphismOutcomeError) as caught:
        Isomorphic.from_transport(cast(TableTransportCertificate, matched_fingerprint))
    assert caught.value.field == "witness"


def test_nonisomorphic_requires_distinct_exhaustive_canonical_identifiers() -> None:
    result = _negative()

    assert result.status == "non_isomorphic"
    assert result.complete is True
    assert result.left_canonical_id != result.right_canonical_id
    with pytest.raises(IsomorphismOutcomeError) as caught:
        NonIsomorphic.from_canonical_labels(result.left_label, result.left_label)
    assert caught.value.field == "canonical_id"


def test_inconclusive_structurally_requires_an_exhausted_bound_and_remaining_work() -> (
    None
):
    result = _inconclusive()

    assert result.status == "bounded_inconclusive"
    assert result.complete is False
    assert result.examined_count == result.bound_limit
    assert result.remaining_count > 0
    with pytest.raises(IsomorphismOutcomeError) as caught:
        Inconclusive.create(
            core=result.core,
            equivalence=result.equivalence,
            left_outputs=result.left_outputs,
            right_outputs=result.right_outputs,
            bound_name="max_permutations",
            bound_limit=2,
            examined_count=1,
            remaining_count=4,
        )
    assert caught.value.field == "examined_count"


def test_outcome_records_have_disjoint_required_fields_and_schema_tags() -> None:
    records = tuple(
        isomorphism_outcome_record(value)
        for value in (_positive(), _negative(), _inconclusive())
    )
    iso, noniso, inconclusive = records

    assert len({record["schemaType"] for record in records}) == 3
    assert "witnessPermutation" in iso
    assert "leftCanonicalId" not in iso and "bound" not in iso
    assert "leftCanonicalId" in noniso
    assert "witnessPermutation" not in noniso and "bound" not in noniso
    assert "bound" in inconclusive
    assert "witnessPermutation" not in inconclusive
    assert "leftCanonicalId" not in inconclusive


@pytest.mark.parametrize("factory", [_positive, _negative, _inconclusive])
def test_canonical_bytes_and_content_hash_are_exact_and_deterministic(
    factory: object,
) -> None:
    result = factory()  # type: ignore[operator]
    first = isomorphism_outcome_canonical_bytes(result)
    second = isomorphism_outcome_canonical_bytes(result)
    record = json.loads(first)
    body = {key: value for key, value in record.items() if key != "contentHash"}
    encoded_body = json.dumps(
        body,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    assert first == second
    assert record["contentHash"] == {
        "algorithm": "sha256",
        "digest": hashlib.sha256(encoded_body).hexdigest(),
    }
    assert result.semantic_hash == SemanticHash(
        "sha256", record["contentHash"]["digest"]
    )


def test_outcomes_are_sealed_immutable_and_content_drift_fails_closed() -> None:
    result = _positive()
    with pytest.raises(IsomorphismOutcomeError, match="factory-owned"):
        type(result)()
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("Derived", (type(result),), {})
    with pytest.raises(FrozenInstanceError):
        result.complete = False  # type: ignore[misc]

    object.__setattr__(result, "left_outputs", result.right_outputs)
    with pytest.raises(IsomorphismOutcomeError) as caught:
        isomorphism_outcome_record(result)
    assert caught.value.field in {"witness", "semantic_hash"}


def test_nonexact_outcome_and_impossible_empty_nullary_core_fail_closed() -> None:
    with pytest.raises(IsomorphismOutcomeError) as caught:
        isomorphism_outcome_record(cast(Isomorphic, None))
    assert caught.value.field == "outcome"

    core = _core(0, 0)
    with pytest.raises(IsomorphismOutcomeError, match="no operation tables"):
        Inconclusive.create(
            core=core,
            equivalence=EquivalencePolicy.literal(core),
            left_outputs=(),
            right_outputs=(),
            bound_name="max_permutations",
            bound_limit=0,
            examined_count=0,
            remaining_count=1,
        )
