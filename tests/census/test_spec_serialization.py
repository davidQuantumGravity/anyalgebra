"""Contracts for the complete immutable, content-addressed census spec."""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from anyalgebra.census import CensusSpec, CensusSpecError
from anyalgebra.census.bounds import CensusOrdering, EnumerationBounds
from anyalgebra.census.constraints import CensusConstraint, CensusTerm, ConstraintSet
from anyalgebra.census.equivalence import EquivalencePolicy, SortBlock
from anyalgebra.census.serialization import (
    census_spec_canonical_bytes,
    census_spec_from_canonical_bytes,
    census_spec_from_record,
    census_spec_record,
)
from anyalgebra.census.spec import CensusSpecCore
from anyalgebra.core.parents import SemanticHash
from anyalgebra.persistence.registry import SchemaError


ROOT = Path(__file__).resolve().parents[2]


def _hash(digit: str) -> SemanticHash:
    return SemanticHash("sha256", digit * 64)


def _core(
    *,
    size: int = 3,
    arity: int = 2,
    name: str = "pointed-commutative-control",
) -> CensusSpecCore:
    points = () if size == 0 else (("zero", 0),)
    return CensusSpecCore.create(
        carrier_size=size,
        arity=arity,
        distinguished_elements=points,
        corpus_name=name,
    )


def _bounds(*, candidates: int = 20_000) -> EnumerationBounds:
    return EnumerationBounds.create(
        max_candidates=candidates,
        max_orbits=20_000,
        max_work_units=1_000_000,
        max_memory_bytes=16_000_000,
        max_observed_milliseconds=5_000,
    )


def _spec(
    *,
    core: CensusSpecCore | None = None,
    constraints: tuple[CensusConstraint, ...] | None = None,
    literal: bool = False,
    references: tuple[SemanticHash, ...] = (),
    bounds: EnumerationBounds | None = None,
) -> CensusSpec:
    selected = _core() if core is None else core
    declarations = (
        (
            CensusConstraint.commutative(),
            CensusConstraint.identity(element=0, side="two_sided"),
        )
        if constraints is None
        else constraints
    )
    constraint_set = ConstraintSet.create(selected, declarations)
    equivalence = (
        EquivalencePolicy.literal(selected)
        if literal
        else EquivalencePolicy.relabeling(selected)
    )
    return CensusSpec.create(
        carrier_size=selected.carrier_size,
        arity=selected.arity,
        constraints=constraint_set,
        equivalence=equivalence,
        ordering=CensusOrdering.reference(),
        bounds=_bounds() if bounds is None else bounds,
        convention_refs=references,
    )


def test_public_factory_has_the_frozen_signature_and_all_semantic_axes() -> None:
    spec = _spec(references=(_hash("b"), _hash("a")))

    assert str(inspect.signature(CensusSpec.create)) == (
        "(*, carrier_size: 'int', arity: 'int', constraints: 'ConstraintSet', "
        "equivalence: 'EquivalencePolicy', ordering: 'CensusOrdering', "
        "bounds: 'EnumerationBounds', convention_refs: "
        "'Iterable[SemanticHash]' = ()) -> 'CensusSpec'"
    )
    assert spec.carrier_size == 3
    assert spec.arity == 2
    assert spec.core.corpus_name == "pointed-commutative-control"
    assert spec.core.distinguished_elements == (("zero", 0),)
    assert spec.constraints.constraints[0].kind == "commutative"
    assert spec.equivalence.fixed_elements == (0,)
    assert spec.ordering == CensusOrdering.reference()
    assert spec.bounds == _bounds()
    assert spec.convention_refs == (_hash("a"), _hash("b"))
    assert spec.schema_version == 1
    assert spec.algorithm_contract_version == "anyalgebra.census.reference.v1"
    assert type(spec.semantic_hash) is SemanticHash


def test_canonical_record_bytes_and_identifier_round_trip_exactly() -> None:
    original = _spec(references=(_hash("a"), _hash("b")))
    record = census_spec_record(original)
    encoded = census_spec_canonical_bytes(original)
    from_record = census_spec_from_record(record)
    from_bytes = census_spec_from_canonical_bytes(encoded)

    assert json.loads(encoded) == record
    assert record["contentHash"] == {
        "algorithm": "sha256",
        "digest": original.semantic_hash.digest,
    }
    assert from_record == original
    assert from_bytes == original
    assert from_record.semantic_hash == original.semantic_hash
    assert census_spec_record(from_record) == record
    assert census_spec_canonical_bytes(from_bytes) == encoded


def test_unordered_declarations_produce_identical_bytes_across_process_order() -> None:
    core = _core(size=4, name="canonical-order-control")
    left_block = SortBlock.create(name="left", elements=(0, 2))
    right_block = SortBlock.create(name="right", elements=(1, 3))
    declarations = (
        CensusConstraint.idempotent(),
        CensusConstraint.commutative(),
    )
    first = CensusSpec.create(
        carrier_size=4,
        arity=2,
        constraints=ConstraintSet.create(core, declarations),
        equivalence=EquivalencePolicy.relabeling(
            core, sort_blocks=(right_block, left_block)
        ),
        ordering=CensusOrdering.reference(),
        bounds=_bounds(candidates=1 << 63 - 1),
        convention_refs=(_hash("b"), _hash("a")),
    )
    second = CensusSpec.create(
        carrier_size=4,
        arity=2,
        constraints=ConstraintSet.create(core, reversed(declarations)),
        equivalence=EquivalencePolicy.relabeling(
            core, sort_blocks=(left_block, right_block)
        ),
        ordering=CensusOrdering.reference(),
        bounds=_bounds(candidates=1 << 63 - 1),
        convention_refs=(_hash("a"), _hash("b")),
    )

    assert first == second
    assert census_spec_canonical_bytes(first) == census_spec_canonical_bytes(second)


def test_canonical_bytes_are_independent_of_python_hash_seed() -> None:
    program = """
from anyalgebra.census import CensusSpec
from anyalgebra.census.bounds import CensusOrdering, EnumerationBounds
from anyalgebra.census.constraints import CensusConstraint, ConstraintSet
from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.serialization import census_spec_canonical_bytes
from anyalgebra.census.spec import CensusSpecCore
from anyalgebra.core.parents import SemanticHash
core = CensusSpecCore.create(
    carrier_size=3,
    arity=2,
    distinguished_elements={"zero": 0},
    corpus_name="hash-seed-control",
)
declarations = {
    "commutative": CensusConstraint.commutative(),
    "idempotent": CensusConstraint.idempotent(),
}
spec = CensusSpec.create(
    carrier_size=3,
    arity=2,
    constraints=ConstraintSet.create(
        core,
        (declarations[name] for name in {"commutative", "idempotent"}),
    ),
    equivalence=EquivalencePolicy.relabeling(core, fixed_elements={2}),
    ordering=CensusOrdering.reference(),
    bounds=EnumerationBounds.create(
        max_candidates=20000,
        max_orbits=20000,
        max_work_units=100000,
        max_memory_bytes=1000000,
    ),
    convention_refs={
        SemanticHash("sha256", "a" * 64),
        SemanticHash("sha256", "b" * 64),
    },
)
print(census_spec_canonical_bytes(spec).hex())
"""
    outputs = []
    for seed in ("1", "947"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        outputs.append(
            subprocess.run(
                [sys.executable, "-c", program],
                cwd=ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )

    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    ("mutation", "code"),
    (
        (lambda record: record.pop("bounds"), "parser_failed"),
        (lambda record: record.update({"extra": 1}), "parser_failed"),
        (lambda record: record.update({"schemaVersion": 2}), "unknown_schema_version"),
        (lambda record: record.update({"carrierSize": 2}), "parser_failed"),
        (
            lambda record: cast(dict[str, object], record["contentHash"]).update(
                {"digest": "0" * 64}
            ),
            "parser_failed",
        ),
    ),
)
def test_missing_extra_future_version_and_tampering_fail_closed(
    mutation: Callable[[dict[str, object]], object], code: str
) -> None:
    record = census_spec_record(_spec())
    mutation(record)

    with pytest.raises(SchemaError) as caught:
        census_spec_from_record(record)
    assert caught.value.code == code


@pytest.mark.parametrize(
    "payload",
    (
        b"not json",
        b'{"schemaType":"anyalgebra.census.spec","schemaType":"duplicate"}',
        b'{"schemaType":"anyalgebra.census.spec","schemaVersion":1.0}',
        b"[]",
        b"\xff",
    ),
)
def test_byte_parser_rejects_malformed_duplicate_nonexact_and_nonrecord_input(
    payload: bytes,
) -> None:
    with pytest.raises(SchemaError):
        census_spec_from_canonical_bytes(payload)


def test_byte_parser_rejects_noncanonical_but_semantically_equivalent_json() -> None:
    record = census_spec_record(_spec())
    padded = json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8")

    with pytest.raises(SchemaError) as caught:
        census_spec_from_canonical_bytes(padded)
    assert caught.value.code == "noncanonical_census_spec"


def test_factory_rejects_cross_core_axes_before_constructing_a_spec() -> None:
    left = _core(name="left")
    right = _core(name="right")

    with pytest.raises(CensusSpecError) as caught:
        CensusSpec.create(
            carrier_size=3,
            arity=2,
            constraints=ConstraintSet.create(left, ()),
            equivalence=EquivalencePolicy.relabeling(right),
            ordering=CensusOrdering.reference(),
            bounds=_bounds(),
        )
    assert caught.value.field == "constraints"
    assert "same core" in caught.value.reason


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("carrier_size", True),
        ("arity", -1),
        ("constraints", ()),
        ("equivalence", "literal"),
        ("ordering", "reference"),
        ("bounds", None),
    ),
)
def test_factory_rejects_nonexact_axis_values(field: str, value: object) -> None:
    core = _core()
    values: dict[str, object] = {
        "carrier_size": 3,
        "arity": 2,
        "constraints": ConstraintSet.create(core, ()),
        "equivalence": EquivalencePolicy.relabeling(core),
        "ordering": CensusOrdering.reference(),
        "bounds": _bounds(),
    }
    values[field] = value

    with pytest.raises(CensusSpecError) as caught:
        CensusSpec.create(**values)  # type: ignore[arg-type]
    assert caught.value.field == field


def test_convention_references_are_exact_bounded_unique_and_snapshotted() -> None:
    source = [_hash("b"), _hash("a")]
    spec = _spec(references=cast(tuple[SemanticHash, ...], iter(source)))
    source.clear()
    assert spec.convention_refs == (_hash("a"), _hash("b"))

    with pytest.raises(CensusSpecError, match="duplicate"):
        _spec(references=(_hash("a"), _hash("a")))
    with pytest.raises(CensusSpecError, match="exact SemanticHash"):
        _spec(references=cast(tuple[SemanticHash, ...], ("bad",)))
    with pytest.raises(CensusSpecError, match="limit"):
        _spec(references=tuple(_hash("a") for _ in range(257)))


def test_empty_carrier_boundary_is_serializable_without_special_case_drift() -> None:
    core = _core(size=0, arity=1, name="empty-unary-control")
    spec = _spec(core=core, constraints=(), literal=True)
    reloaded = census_spec_from_canonical_bytes(census_spec_canonical_bytes(spec))

    assert reloaded.core.input_tuple_count == 0
    assert reloaded.core.candidate_count == 1
    assert reloaded.equivalence.sort_blocks == ()
    assert reloaded == spec


def test_every_allow_listed_constraint_and_nested_term_round_trips() -> None:
    core = _core(size=3, name="complete-constraint-codec-control")
    x = CensusTerm.variable(0)
    y = CensusTerm.variable(1)
    equation = CensusConstraint.equation(
        CensusTerm.apply(CensusTerm.apply(x, y), x),
        CensusTerm.apply(x, CensusTerm.apply(y, x)),
        variable_count=2,
    )
    declarations = (
        CensusConstraint.commutative(),
        CensusConstraint.idempotent(),
        CensusConstraint.identity(element=0, side="left"),
        CensusConstraint.quasigroup(),
        equation,
    )
    original = _spec(core=core, constraints=declarations)
    reloaded = census_spec_from_canonical_bytes(census_spec_canonical_bytes(original))

    assert reloaded.constraints == original.constraints
    assert reloaded == original


def test_nullary_value_constraint_round_trips() -> None:
    core = _core(size=2, arity=0, name="nullary-codec-control")
    original = _spec(
        core=core,
        constraints=(CensusConstraint.nullary_value(element=1),),
        literal=True,
    )

    assert (
        census_spec_from_canonical_bytes(census_spec_canonical_bytes(original))
        == original
    )


def test_spec_is_factory_owned_sealed_immutable_and_safely_represented() -> None:
    spec = _spec()

    with pytest.raises(CensusSpecError, match="factory-owned"):
        CensusSpec()
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("Derived", (CensusSpec,), {})
    with pytest.raises(FrozenInstanceError):
        spec.arity = 3  # type: ignore[misc]
    assert not hasattr(spec, "__dict__")
    assert "pointed-commutative-control" not in repr(spec)
    assert spec.semantic_hash.digest in repr(spec)
