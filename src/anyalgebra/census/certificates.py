"""Independent verification of canonical-table transport certificates.

The verifier deliberately does not establish that a claimed target is the
least member of an orbit.  It establishes the narrower replayable facts that
the declared relabeling transports the source to the target in both directions
and that the target and policy have the claimed canonical content identifier.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
import json

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import SemanticHash

from .canonical import CanonicalLabel
from .equivalence import EquivalencePolicy
from .permutations import CarrierPermutation, PermutationGenerationError
from .spec import CensusSpecCore
from .table_codes import OperationTableCodeError, rank_operation_table
from .transport import TableTransportCertificate


class CanonicalCertificateError(AnyAlgebraError, ValueError):
    """A canonical-certificate claim failed structural or semantic replay."""

    def __init__(self, *, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid canonical certificate {field}: {reason}")


def _policy_record(policy: EquivalencePolicy) -> dict[str, object]:
    return {
        "fixedElements": list(policy.fixed_elements),
        "kind": policy.kind,
        "sortBlocks": [
            {"elements": list(block.elements), "name": block.name}
            for block in policy.sort_blocks
        ],
    }


def _canonical_bytes(
    core: CensusSpecCore,
    outputs: tuple[int, ...],
    policy: EquivalencePolicy,
) -> bytes:
    record = {
        "arity": core.arity,
        "carrierSize": core.carrier_size,
        "equivalence": _policy_record(policy),
        "outputs": list(outputs),
        "schemaType": "anyalgebra.census.canonical_table",
        "schemaVersion": 1,
    }
    return json.dumps(
        record,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _flat_index(arguments: tuple[int, ...], size: int) -> int:
    index = 0
    for argument in arguments:
        index = index * size + argument
    return index


def _exact_tuple(value: object, field: str) -> tuple[int, ...]:
    if type(value) is not tuple or any(type(item) is not int for item in value):
        raise CanonicalCertificateError(
            field=field, reason="must be an exact tuple of built-in ints"
        )
    return value


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class CanonicalCertificate:
    """An immutable, explicitly unverified canonical-label claim."""

    core: CensusSpecCore
    equivalence: EquivalencePolicy
    source_outputs: tuple[int, ...]
    canonical_outputs: tuple[int, ...]
    permutation_images: tuple[int, ...]
    canonical_id: SemanticHash
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise CanonicalCertificateError(field="certificate", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("CanonicalCertificate cannot be subclassed")

    @classmethod
    def claim(
        cls,
        *,
        core: CensusSpecCore,
        equivalence: EquivalencePolicy,
        source_outputs: tuple[int, ...],
        canonical_outputs: tuple[int, ...],
        permutation_images: tuple[int, ...],
        canonical_id: SemanticHash,
    ) -> CanonicalCertificate:
        """Snapshot an untrusted claim without asserting its mathematical facts."""
        if cls is not CanonicalCertificate:
            raise CanonicalCertificateError(
                field="certificate", reason="factory requires exact type"
            )
        if type(core) is not CensusSpecCore:
            raise CanonicalCertificateError(
                field="core", reason="must be an exact CensusSpecCore"
            )
        if type(equivalence) is not EquivalencePolicy or equivalence.core is not core:
            raise CanonicalCertificateError(
                field="equivalence",
                reason="must belong to the literal certificate core",
            )
        source = _exact_tuple(source_outputs, "source_outputs")
        target = _exact_tuple(canonical_outputs, "canonical_outputs")
        images = _exact_tuple(permutation_images, "permutation_images")
        if type(canonical_id) is not SemanticHash:
            raise CanonicalCertificateError(
                field="canonical_id", reason="must be an exact SemanticHash"
            )
        value = object.__new__(CanonicalCertificate)
        for field, item in (
            ("core", core),
            ("equivalence", equivalence),
            ("source_outputs", source),
            ("canonical_outputs", target),
            ("permutation_images", images),
            ("canonical_id", canonical_id),
        ):
            object.__setattr__(value, field, item)
        return value

    @classmethod
    def from_label(cls, label: CanonicalLabel) -> CanonicalCertificate:
        """Extract the replayable claim from one exact search-produced label."""
        if type(label) is not CanonicalLabel:
            raise CanonicalCertificateError(
                field="label", reason="must be an exact CanonicalLabel"
            )
        return cls.claim(
            core=label.core,
            equivalence=label.equivalence,
            source_outputs=label.source_outputs,
            canonical_outputs=label.canonical_outputs,
            permutation_images=label.certificate.permutation.images,
            canonical_id=label.canonical_id,
        )

    def __eq__(self, other: object) -> bool:
        return type(other) is CanonicalCertificate and (
            self.core,
            self.equivalence,
            self.source_outputs,
            self.canonical_outputs,
            self.permutation_images,
            self.canonical_id,
        ) == (
            other.core,
            other.equivalence,
            other.source_outputs,
            other.canonical_outputs,
            other.permutation_images,
            other.canonical_id,
        )

    def __repr__(self) -> str:
        return (
            "CanonicalCertificate("
            f"canonical_id={str(self.canonical_id)!r}, "
            f"carrier_size={self.core.carrier_size}, arity={self.core.arity})"
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class CanonicalCertificateVerification:
    """Successful bounded replay, excluding any canonical-minimality claim."""

    certificate: CanonicalCertificate
    reconstructed_source: tuple[int, ...]
    reconstructed_target: tuple[int, ...]
    canonical_bytes: bytes
    canonical_id: SemanticHash
    checked_cell_count: int
    transport_verified: bool
    identifier_verified: bool
    minimality_verified: bool
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise CanonicalCertificateError(field="verification", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("CanonicalCertificateVerification cannot be subclassed")

    @classmethod
    def _create(
        cls,
        certificate: CanonicalCertificate,
        source: tuple[int, ...],
        target: tuple[int, ...],
        encoded: bytes,
        canonical_id: SemanticHash,
    ) -> CanonicalCertificateVerification:
        value = object.__new__(CanonicalCertificateVerification)
        for field, item in (
            ("certificate", certificate),
            ("reconstructed_source", source),
            ("reconstructed_target", target),
            ("canonical_bytes", encoded),
            ("canonical_id", canonical_id),
            ("checked_cell_count", certificate.core.input_tuple_count * 2),
            ("transport_verified", True),
            ("identifier_verified", True),
            ("minimality_verified", False),
        ):
            object.__setattr__(value, field, item)
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is CanonicalCertificateVerification and (
            self.certificate,
            self.reconstructed_source,
            self.reconstructed_target,
            self.canonical_bytes,
            self.canonical_id,
            self.checked_cell_count,
            self.transport_verified,
            self.identifier_verified,
            self.minimality_verified,
        ) == (
            other.certificate,
            other.reconstructed_source,
            other.reconstructed_target,
            other.canonical_bytes,
            other.canonical_id,
            other.checked_cell_count,
            other.transport_verified,
            other.identifier_verified,
            other.minimality_verified,
        )

    def __repr__(self) -> str:
        return (
            "CanonicalCertificateVerification("
            f"canonical_id={str(self.canonical_id)!r}, "
            f"checked_cell_count={self.checked_cell_count}, "
            "minimality_verified=False)"
        )


def _validated_table(
    core: CensusSpecCore, outputs: tuple[int, ...], field: str
) -> None:
    try:
        rank_operation_table(core, outputs)
    except OperationTableCodeError as error:
        raise CanonicalCertificateError(
            field=field, reason="is not a valid table for the certificate core"
        ) from error


def verify_canonical_certificate(
    certificate: CanonicalCertificate,
    *,
    expected_policy: EquivalencePolicy | None = None,
) -> CanonicalCertificateVerification:
    """Replay transport and identity without invoking canonical search.

    This verifies a concrete orbit map and canonical-table content address.  It
    intentionally leaves ``minimality_verified`` false because proving that the
    target is lexicographically least requires a separate exhaustive or trusted
    canonicalization argument.
    """
    if type(certificate) is not CanonicalCertificate:
        raise CanonicalCertificateError(
            field="certificate", reason="must be an exact CanonicalCertificate"
        )
    core = certificate.core
    if core.candidate_count == 0:
        raise CanonicalCertificateError(
            field="source_outputs", reason="the declared core has no operation tables"
        )
    if expected_policy is not None and (
        type(expected_policy) is not EquivalencePolicy
        or expected_policy is not certificate.equivalence
    ):
        raise CanonicalCertificateError(
            field="expected_policy",
            reason="must be the literal certificate policy",
        )
    _validated_table(core, certificate.source_outputs, "source_outputs")
    _validated_table(core, certificate.canonical_outputs, "canonical_outputs")
    try:
        permutation = CarrierPermutation.create(
            certificate.equivalence, certificate.permutation_images
        )
    except PermutationGenerationError as error:
        raise CanonicalCertificateError(
            field="permutation_images",
            reason="is not an allowed carrier relabeling",
        ) from error

    size = core.carrier_size
    target = [0] * core.input_tuple_count
    for old_flat, old_inputs in enumerate(product(range(size), repeat=core.arity)):
        new_inputs = tuple(permutation(argument) for argument in old_inputs)
        target[_flat_index(new_inputs, size)] = permutation(
            certificate.source_outputs[old_flat]
        )
    reconstructed_target = tuple(target)
    if reconstructed_target != certificate.canonical_outputs:
        raise CanonicalCertificateError(
            field="canonical_outputs",
            reason="does not match forward transport of the claimed source",
        )

    inverse = permutation.inverse()
    source = [0] * core.input_tuple_count
    for new_flat, new_inputs in enumerate(product(range(size), repeat=core.arity)):
        old_inputs = tuple(inverse(argument) for argument in new_inputs)
        source[_flat_index(old_inputs, size)] = inverse(
            certificate.canonical_outputs[new_flat]
        )
    reconstructed_source = tuple(source)
    if reconstructed_source != certificate.source_outputs:
        raise CanonicalCertificateError(
            field="source_outputs",
            reason="does not match inverse transport of the claimed target",
        )

    encoded = _canonical_bytes(
        core, certificate.canonical_outputs, certificate.equivalence
    )
    canonical_id = SemanticHash("sha256", hashlib.sha256(encoded).hexdigest())
    if canonical_id != certificate.canonical_id:
        raise CanonicalCertificateError(
            field="canonical_id",
            reason="does not match the target table and equivalence policy",
        )
    return CanonicalCertificateVerification._create(
        certificate,
        reconstructed_source,
        reconstructed_target,
        encoded,
        canonical_id,
    )


_ISOMORPHISM_DIRECTION = "left_to_right_old_index_to_new_index"


class IsomorphismCertificateError(AnyAlgebraError, ValueError):
    """An untrusted positive-map claim failed independent replay."""

    def __init__(
        self,
        *,
        field: str,
        reason: str,
        inputs: tuple[int, ...] | None = None,
    ) -> None:
        self.field = field
        self.reason = reason
        self.inputs = inputs
        suffix = "" if inputs is None else f" at inputs {inputs!r}"
        super().__init__(f"invalid isomorphism certificate {field}: {reason}{suffix}")


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class IsomorphismCertificate:
    """Immutable untrusted claim of one explicit left-to-right table map."""

    core: CensusSpecCore
    equivalence: EquivalencePolicy
    source_outputs: tuple[int, ...]
    target_outputs: tuple[int, ...]
    permutation_images: tuple[int, ...]
    direction: str
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise IsomorphismCertificateError(
            field="certificate", reason="is factory-owned"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("IsomorphismCertificate cannot be subclassed")

    @classmethod
    def claim(
        cls,
        *,
        core: CensusSpecCore,
        equivalence: EquivalencePolicy,
        source_outputs: tuple[int, ...],
        target_outputs: tuple[int, ...],
        permutation_images: tuple[int, ...],
        direction: str = _ISOMORPHISM_DIRECTION,
    ) -> IsomorphismCertificate:
        """Snapshot untrusted fields without asserting that they commute."""
        if cls is not IsomorphismCertificate:
            raise IsomorphismCertificateError(
                field="certificate", reason="factory requires exact type"
            )
        if type(core) is not CensusSpecCore:
            raise IsomorphismCertificateError(
                field="core", reason="must be an exact CensusSpecCore"
            )
        if type(equivalence) is not EquivalencePolicy or equivalence.core is not core:
            raise IsomorphismCertificateError(
                field="equivalence",
                reason="must belong to the literal certificate core",
            )
        for field, item in (
            ("source_outputs", source_outputs),
            ("target_outputs", target_outputs),
            ("permutation_images", permutation_images),
        ):
            if type(item) is not tuple or any(type(value) is not int for value in item):
                raise IsomorphismCertificateError(
                    field=field, reason="must be an exact tuple of built-in ints"
                )
        if type(direction) is not str:
            raise IsomorphismCertificateError(
                field="direction", reason="must be an exact built-in str"
            )
        value = object.__new__(IsomorphismCertificate)
        for stored_field, stored_item in (
            ("core", core),
            ("equivalence", equivalence),
            ("source_outputs", source_outputs),
            ("target_outputs", target_outputs),
            ("permutation_images", permutation_images),
            ("direction", direction),
        ):
            object.__setattr__(value, stored_field, stored_item)
        return value

    @classmethod
    def from_transport(
        cls, witness: TableTransportCertificate
    ) -> IsomorphismCertificate:
        """Extract an untrusted replay claim from an existing transport."""
        if type(witness) is not TableTransportCertificate:
            raise IsomorphismCertificateError(
                field="witness", reason="must be an exact TableTransportCertificate"
            )
        return cls.claim(
            core=witness.core,
            equivalence=witness.permutation.policy,
            source_outputs=witness.source_outputs,
            target_outputs=witness.target_outputs,
            permutation_images=witness.permutation.images,
        )

    def __eq__(self, other: object) -> bool:
        return type(other) is IsomorphismCertificate and (
            self.core,
            self.equivalence,
            self.source_outputs,
            self.target_outputs,
            self.permutation_images,
            self.direction,
        ) == (
            other.core,
            other.equivalence,
            other.source_outputs,
            other.target_outputs,
            other.permutation_images,
            other.direction,
        )

    def __repr__(self) -> str:
        return (
            "IsomorphismCertificate("
            f"carrier_size={self.core.carrier_size}, arity={self.core.arity}, "
            f"direction={self.direction!r})"
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class IsomorphismCertificateVerification:
    """Successful direct commuting-square replay of one claimed map."""

    certificate: IsomorphismCertificate
    source_outputs: tuple[int, ...]
    target_outputs: tuple[int, ...]
    permutation_images: tuple[int, ...]
    checked_tuple_count: int
    bijective: bool
    policy_preserving: bool
    operation_commutes: bool
    direction: str
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise IsomorphismCertificateError(
            field="verification", reason="is factory-owned"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("IsomorphismCertificateVerification cannot be subclassed")

    @classmethod
    def _create(
        cls, certificate: IsomorphismCertificate
    ) -> IsomorphismCertificateVerification:
        value = object.__new__(IsomorphismCertificateVerification)
        for field, item in (
            ("certificate", certificate),
            ("source_outputs", certificate.source_outputs),
            ("target_outputs", certificate.target_outputs),
            ("permutation_images", certificate.permutation_images),
            ("checked_tuple_count", certificate.core.input_tuple_count),
            ("bijective", True),
            ("policy_preserving", True),
            ("operation_commutes", True),
            ("direction", certificate.direction),
        ):
            object.__setattr__(value, field, item)
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is IsomorphismCertificateVerification and (
            self.certificate,
            self.source_outputs,
            self.target_outputs,
            self.permutation_images,
            self.checked_tuple_count,
            self.bijective,
            self.policy_preserving,
            self.operation_commutes,
            self.direction,
        ) == (
            other.certificate,
            other.source_outputs,
            other.target_outputs,
            other.permutation_images,
            other.checked_tuple_count,
            other.bijective,
            other.policy_preserving,
            other.operation_commutes,
            other.direction,
        )

    def __repr__(self) -> str:
        return (
            "IsomorphismCertificateVerification("
            f"checked_tuple_count={self.checked_tuple_count}, "
            "bijective=True, policy_preserving=True, operation_commutes=True)"
        )


def _isomorphism_table(
    core: CensusSpecCore, outputs: tuple[int, ...], field: str
) -> None:
    try:
        rank_operation_table(core, outputs)
    except OperationTableCodeError as error:
        raise IsomorphismCertificateError(
            field=field, reason="is not a valid table for the certificate core"
        ) from error


def verify_isomorphism_certificate(
    certificate: IsomorphismCertificate,
) -> IsomorphismCertificateVerification:
    """Check bijectivity, policy, direction, and every operation tuple directly."""
    if type(certificate) is not IsomorphismCertificate:
        raise IsomorphismCertificateError(
            field="certificate", reason="must be an exact IsomorphismCertificate"
        )
    core = certificate.core
    if core.candidate_count == 0:
        raise IsomorphismCertificateError(
            field="source_outputs", reason="the declared core has no operation tables"
        )
    _isomorphism_table(core, certificate.source_outputs, "source_outputs")
    _isomorphism_table(core, certificate.target_outputs, "target_outputs")
    if certificate.direction != _ISOMORPHISM_DIRECTION:
        raise IsomorphismCertificateError(
            field="direction", reason="must declare left-to-right old-to-new mapping"
        )
    images = certificate.permutation_images
    if (
        len(images) != core.carrier_size
        or any(type(item) is not int for item in images)
        or set(images) != set(range(core.carrier_size))
    ):
        raise IsomorphismCertificateError(
            field="permutation_images", reason="must be a carrier bijection"
        )
    try:
        allowed = certificate.equivalence.allows(images)
    except Exception as error:
        raise IsomorphismCertificateError(
            field="permutation_images", reason="policy membership check failed"
        ) from error
    if not allowed:
        raise IsomorphismCertificateError(
            field="permutation_images", reason="bijection is forbidden by the policy"
        )
    for source_flat, source_inputs in enumerate(
        product(range(core.carrier_size), repeat=core.arity)
    ):
        target_inputs = tuple(images[item] for item in source_inputs)
        target_flat = _flat_index(target_inputs, core.carrier_size)
        if (
            certificate.target_outputs[target_flat]
            != images[certificate.source_outputs[source_flat]]
        ):
            raise IsomorphismCertificateError(
                field="operation",
                reason="left-to-right commuting equation failed",
                inputs=source_inputs,
            )
    return IsomorphismCertificateVerification._create(certificate)


__all__ = (
    "CanonicalCertificate",
    "CanonicalCertificateError",
    "CanonicalCertificateVerification",
    "IsomorphismCertificate",
    "IsomorphismCertificateError",
    "IsomorphismCertificateVerification",
    "verify_canonical_certificate",
    "verify_isomorphism_certificate",
)
