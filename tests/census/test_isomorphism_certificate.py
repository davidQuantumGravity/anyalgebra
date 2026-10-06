"""Independent cell replay for claimed finite-table isomorphism maps."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from anyalgebra.census.certificates import (
    IsomorphismCertificate,
    IsomorphismCertificateError,
    verify_isomorphism_certificate,
)
from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.isomorphism import search_isomorphism
from anyalgebra.census.permutations import CarrierPermutation
from anyalgebra.census.spec import CensusSpecCore
from anyalgebra.census.transport import transport_operation_table


def _core(size: int, arity: int = 2, *, pointed: bool = False) -> CensusSpecCore:
    return CensusSpecCore.create(
        carrier_size=size,
        arity=arity,
        distinguished_elements=(("zero", 0),) if pointed else (),
        corpus_name=f"isomorphism-certificate-{size}-{arity}-{pointed}",
    )


def _claim(
    certificate: IsomorphismCertificate,
    *,
    source: tuple[int, ...] | None = None,
    target: tuple[int, ...] | None = None,
    images: tuple[int, ...] | None = None,
    direction: str | None = None,
    policy: EquivalencePolicy | None = None,
) -> IsomorphismCertificate:
    return IsomorphismCertificate.claim(
        core=certificate.core,
        equivalence=policy or certificate.equivalence,
        source_outputs=source if source is not None else certificate.source_outputs,
        target_outputs=target if target is not None else certificate.target_outputs,
        permutation_images=(
            images if images is not None else certificate.permutation_images
        ),
        direction=direction or certificate.direction,
    )


def test_search_map_verifies_without_search_or_canonical_identifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _core(3, 1)
    policy = EquivalencePolicy.relabeling(core)
    search = search_isomorphism(core, (0, 0, 0), (1, 1, 1), policy)
    assert search.witness is not None
    certificate = IsomorphismCertificate.from_transport(search.witness)

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("search or transport helper invoked")

    monkeypatch.setattr("anyalgebra.census.isomorphism.search_isomorphism", forbidden)
    monkeypatch.setattr(
        "anyalgebra.census.transport.transport_operation_table", forbidden
    )
    result = verify_isomorphism_certificate(certificate)

    assert result.source_outputs == certificate.source_outputs
    assert result.target_outputs == certificate.target_outputs
    assert result.permutation_images == certificate.permutation_images
    assert result.checked_tuple_count == core.input_tuple_count
    assert result.bijective is True
    assert result.policy_preserving is True
    assert result.operation_commutes is True
    assert not hasattr(result, "canonical_id")


def test_nonbijective_map_fails_closed() -> None:
    core = _core(3, 1)
    policy = EquivalencePolicy.relabeling(core)
    certificate = IsomorphismCertificate.claim(
        core=core,
        equivalence=policy,
        source_outputs=(0, 0, 0),
        target_outputs=(1, 1, 1),
        permutation_images=(1, 1, 2),
    )
    with pytest.raises(IsomorphismCertificateError) as caught:
        verify_isomorphism_certificate(certificate)
    assert caught.value.field == "permutation_images"


def test_policy_breaking_map_fails_closed() -> None:
    core = _core(3, 1, pointed=True)
    policy = EquivalencePolicy.relabeling(core)
    certificate = IsomorphismCertificate.claim(
        core=core,
        equivalence=policy,
        source_outputs=(0, 1, 2),
        target_outputs=(0, 1, 2),
        permutation_images=(1, 0, 2),
    )
    with pytest.raises(IsomorphismCertificateError) as caught:
        verify_isomorphism_certificate(certificate)
    assert caught.value.field == "permutation_images"


@pytest.mark.parametrize("field", ["source", "target"])
def test_tampered_tables_fail_cell_replay(field: str) -> None:
    core = _core(3, 1)
    policy = EquivalencePolicy.relabeling(core)
    permutation = CarrierPermutation.create(policy, (1, 2, 0))
    transport = transport_operation_table(core, (0, 0, 0), permutation)
    certificate = IsomorphismCertificate.from_transport(transport)
    changes = {field: (0, 1, 2)}

    with pytest.raises(IsomorphismCertificateError) as caught:
        verify_isomorphism_certificate(_claim(certificate, **changes))  # type: ignore[arg-type]
    assert caught.value.field == "operation"


def test_wrong_direction_map_and_direction_tag_fail_closed() -> None:
    core = _core(3, 1)
    policy = EquivalencePolicy.relabeling(core)
    permutation = CarrierPermutation.create(policy, (1, 2, 0))
    transport = transport_operation_table(core, (0, 0, 0), permutation)
    certificate = IsomorphismCertificate.from_transport(transport)

    with pytest.raises(IsomorphismCertificateError) as caught:
        verify_isomorphism_certificate(
            _claim(certificate, images=permutation.inverse().images)
        )
    assert caught.value.field == "operation"
    with pytest.raises(IsomorphismCertificateError) as caught:
        verify_isomorphism_certificate(_claim(certificate, direction="right_to_left"))
    assert caught.value.field == "direction"


def test_nullary_and_empty_carrier_boundaries_verify() -> None:
    nullary = _core(2, 0)
    policy = EquivalencePolicy.relabeling(nullary)
    permutation = CarrierPermutation.create(policy, (1, 0))
    certificate = IsomorphismCertificate.from_transport(
        transport_operation_table(nullary, (0,), permutation)
    )
    assert verify_isomorphism_certificate(certificate).checked_tuple_count == 1

    empty = _core(0, 1)
    empty_policy = EquivalencePolicy.relabeling(empty)
    empty_certificate = IsomorphismCertificate.from_transport(
        transport_operation_table(empty, (), CarrierPermutation.identity(empty_policy))
    )
    assert verify_isomorphism_certificate(empty_certificate).checked_tuple_count == 0


def test_nonexact_and_no_table_inputs_fail_closed() -> None:
    with pytest.raises(IsomorphismCertificateError) as caught:
        verify_isomorphism_certificate(cast(IsomorphismCertificate, None))
    assert caught.value.field == "certificate"

    impossible = _core(0, 0)
    certificate = IsomorphismCertificate.claim(
        core=impossible,
        equivalence=EquivalencePolicy.literal(impossible),
        source_outputs=(),
        target_outputs=(),
        permutation_images=(),
    )
    with pytest.raises(IsomorphismCertificateError, match="no operation tables"):
        verify_isomorphism_certificate(certificate)


def test_claim_and_verification_are_sealed_immutable_and_deterministic() -> None:
    core = _core(2)
    policy = EquivalencePolicy.relabeling(core)
    certificate = IsomorphismCertificate.from_transport(
        transport_operation_table(
            core,
            (0, 0, 0, 0),
            CarrierPermutation.identity(policy),
        )
    )
    first = verify_isomorphism_certificate(certificate)
    second = verify_isomorphism_certificate(certificate)

    assert first == second
    assert repr(first) == repr(second)
    with pytest.raises(IsomorphismCertificateError, match="factory-owned"):
        type(first)()
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("Derived", (type(first),), {})
    with pytest.raises(FrozenInstanceError):
        first.operation_commutes = False  # type: ignore[misc]
