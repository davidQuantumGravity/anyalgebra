"""Independent verification contracts for canonical-label certificates."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from anyalgebra.census import canonical as canonical_module
from anyalgebra.census.canonical import canonicalize_operation_table
from anyalgebra.census.certificates import (
    CanonicalCertificate,
    CanonicalCertificateError,
    verify_canonical_certificate,
)
from anyalgebra.census.equivalence import EquivalencePolicy, SortBlock
from anyalgebra.census.spec import CensusSpecCore
from anyalgebra.core.parents import SemanticHash


def _core(size: int, arity: int = 2) -> CensusSpecCore:
    return CensusSpecCore.create(
        carrier_size=size,
        arity=arity,
        corpus_name=f"certificate-{size}-{arity}",
    )


def _nontrivial_certificate() -> CanonicalCertificate:
    core = _core(3, 1)
    policy = EquivalencePolicy.relabeling(core)
    label = canonicalize_operation_table(core, (2, 2, 0), policy)
    assert label.source_outputs != label.canonical_outputs
    return CanonicalCertificate.from_label(label)


def _claim(
    certificate: CanonicalCertificate,
    *,
    policy: EquivalencePolicy | None = None,
    source: tuple[int, ...] | None = None,
    target: tuple[int, ...] | None = None,
    images: tuple[int, ...] | None = None,
    canonical_id: SemanticHash | None = None,
) -> CanonicalCertificate:
    return CanonicalCertificate.claim(
        core=certificate.core,
        equivalence=policy or certificate.equivalence,
        source_outputs=source or certificate.source_outputs,
        canonical_outputs=target or certificate.canonical_outputs,
        permutation_images=images or certificate.permutation_images,
        canonical_id=canonical_id or certificate.canonical_id,
    )


def test_valid_certificate_reconstructs_both_tables_without_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    certificate = _nontrivial_certificate()

    def forbidden_search(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("canonical search was invoked")

    monkeypatch.setattr(
        canonical_module, "canonicalize_operation_table", forbidden_search
    )
    result = verify_canonical_certificate(certificate)

    assert result.reconstructed_source == certificate.source_outputs
    assert result.reconstructed_target == certificate.canonical_outputs
    assert result.checked_cell_count == certificate.core.input_tuple_count * 2
    assert result.transport_verified is True
    assert result.identifier_verified is True
    assert result.minimality_verified is False
    assert result.canonical_id == certificate.canonical_id


@pytest.mark.parametrize("field", ["mapping", "source", "target", "identifier"])
def test_tampered_certificate_fields_fail_closed(field: str) -> None:
    certificate = _nontrivial_certificate()
    changes: dict[str, object] = {}
    if field == "mapping":
        changes["images"] = tuple(range(certificate.core.carrier_size))
    elif field == "source":
        changes["source"] = (0, 0, 0)
    elif field == "target":
        changes["target"] = (2, 2, 2)
    else:
        changes["canonical_id"] = SemanticHash("sha256", "0" * 64)

    with pytest.raises(CanonicalCertificateError) as caught:
        verify_canonical_certificate(_claim(certificate, **changes))  # type: ignore[arg-type]
    assert caught.value.field in {
        "permutation_images",
        "source_outputs",
        "canonical_outputs",
        "canonical_id",
    }


def test_tampered_policy_fails_even_when_the_mapping_remains_allowed() -> None:
    certificate = _nontrivial_certificate()
    policy = EquivalencePolicy.relabeling(
        certificate.core,
        sort_blocks=(
            SortBlock.create(name="all", elements=range(certificate.core.carrier_size)),
        ),
    )
    assert policy.allows(certificate.permutation_images)

    with pytest.raises(CanonicalCertificateError) as caught:
        verify_canonical_certificate(_claim(certificate, policy=policy))
    assert caught.value.field == "canonical_id"


def test_literal_expectations_use_exact_literal_ownership() -> None:
    certificate = _nontrivial_certificate()
    same_value_core = _core(3, 1)
    same_value_policy = EquivalencePolicy.relabeling(same_value_core)

    with pytest.raises(CanonicalCertificateError) as caught:
        verify_canonical_certificate(
            certificate,
            expected_policy=same_value_policy,
        )
    assert caught.value.field == "expected_policy"


def test_nullary_and_empty_carrier_boundaries_verify() -> None:
    nullary_core = _core(2, 0)
    nullary_label = canonicalize_operation_table(
        nullary_core,
        (1,),
        EquivalencePolicy.literal(nullary_core),
    )
    nullary = verify_canonical_certificate(
        CanonicalCertificate.from_label(nullary_label)
    )
    assert nullary.reconstructed_source == nullary.reconstructed_target == (1,)
    assert nullary.checked_cell_count == 2

    empty_core = _core(0, 1)
    empty_label = canonicalize_operation_table(
        empty_core,
        (),
        EquivalencePolicy.relabeling(empty_core),
    )
    empty = verify_canonical_certificate(CanonicalCertificate.from_label(empty_label))
    assert empty.reconstructed_source == empty.reconstructed_target == ()
    assert empty.checked_cell_count == 0


def test_impossible_empty_nullary_core_and_nonexact_input_fail_closed() -> None:
    impossible = _core(0, 0)
    policy = EquivalencePolicy.literal(impossible)
    claim = CanonicalCertificate.claim(
        core=impossible,
        equivalence=policy,
        source_outputs=(),
        canonical_outputs=(),
        permutation_images=(),
        canonical_id=SemanticHash("sha256", "0" * 64),
    )
    with pytest.raises(CanonicalCertificateError, match="no operation tables"):
        verify_canonical_certificate(claim)
    with pytest.raises(CanonicalCertificateError) as caught:
        verify_canonical_certificate(cast(CanonicalCertificate, None))
    assert caught.value.field == "certificate"


def test_claim_and_verification_are_sealed_immutable_and_deterministic() -> None:
    certificate = _nontrivial_certificate()
    first = verify_canonical_certificate(certificate)
    second = verify_canonical_certificate(certificate)

    assert first == second
    assert repr(first) == repr(second)
    with pytest.raises(CanonicalCertificateError, match="factory-owned"):
        type(certificate)()
    with pytest.raises(CanonicalCertificateError, match="factory-owned"):
        type(first)()
    with pytest.raises(TypeError, match="cannot be subclassed"):
        type("Derived", (type(first),), {})
    with pytest.raises(FrozenInstanceError):
        first.transport_verified = False  # type: ignore[misc]
