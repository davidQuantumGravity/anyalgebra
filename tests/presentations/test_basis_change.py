"""Contract tests for constructive finite exact basis changes."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from anyalgebra.algebra.multilinear import StructureConstants
from anyalgebra.core.domains import QQ, ZZ
from anyalgebra.core.elements import SparseElement
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.presentations.basis_change import (
    BasisChangeError,
    ExactBasisIsomorphism,
    change_basis,
)


def _module(rank: int, *, qq: bool = False) -> FreeModule:
    domain = QQ() if qq else ZZ()
    return FreeModule(
        domain,
        Basis(tuple(f"e{index}" for index in range(rank)), coefficient_domain=domain),
    )


def _basis(module: FreeModule, index: int, coefficient: object = 1) -> SparseElement:
    return module.element({index: coefficient})


def _identity(
    module: FreeModule, target: FreeModule | None = None
) -> ExactBasisIsomorphism:
    actual_target = module if target is None else target
    return ExactBasisIsomorphism.from_images(
        module,
        actual_target,
        tuple(_basis(actual_target, index) for index in range(module.rank)),
        tuple(_basis(module, index) for index in range(module.rank)),
    )


def test_identity_permutation_certificate_reversal_and_element_round_trips() -> None:
    source = _module(2)
    target = _module(2)
    permutation = ExactBasisIsomorphism.from_images(
        source,
        target,
        (_basis(target, 1), _basis(target, 0)),
        (_basis(source, 1), _basis(source, 0)),
    )
    value = source.element({0: 3, 1: -2})
    transported = permutation.forward(value)
    assert transported.parent is target
    assert transported == target.element({0: -2, 1: 3})
    assert permutation.inverse(transported) == value
    assert permutation.reversed().forward(transported) == value
    certificate = permutation.certificate
    assert certificate.exact_certified is True
    assert certificate.algorithm == "explicit-basis-composites-v1"
    assert certificate.expected_forward_after_inverse_count == 2
    assert certificate.evaluated_forward_after_inverse_count == 2
    assert certificate.expected_inverse_after_forward_count == 2
    assert certificate.evaluated_inverse_after_forward_count == 2
    assert change_basis(value, permutation) == transported


def test_rational_scaling_basis_change_and_negative_certification_witnesses() -> None:
    source = _module(2, qq=True)
    target = _module(2, qq=True)
    change = ExactBasisIsomorphism.from_images(
        source,
        target,
        (target.element({0: 1, 1: 1}), target.element({0: 1, 1: -1})),
        (
            source.element({0: (1, 2), 1: (1, 2)}),
            source.element({0: (1, 2), 1: (-1, 2)}),
        ),
    )
    value = source.element({0: (1, 2), 1: 3})
    assert change.forward(value) == target.element({0: (7, 2), 1: (-5, 2)})
    assert change.inverse(change.forward(value)) == value
    with pytest.raises(BasisChangeError, match="forward-after-inverse") as caught:
        ExactBasisIsomorphism.from_images(
            source,
            target,
            (_basis(target, 0), _basis(target, 0)),
            (_basis(source, 0), _basis(source, 1)),
        )
    assert caught.value.side == "forward-after-inverse"
    assert caught.value.index == 1


def test_factory_parent_count_type_rank_and_record_boundaries() -> None:
    module = _module(1)
    foreign = _module(1)
    with pytest.raises(BasisChangeError, match="exact FreeModule"):
        ExactBasisIsomorphism.from_images(object(), module, (), ())
    with pytest.raises(BasisChangeError, match="literal coefficient domain"):
        ExactBasisIsomorphism.from_images(module, _module(1, qq=True), (), ())
    with pytest.raises(BasisChangeError, match="equal finite rank"):
        ExactBasisIsomorphism.from_images(module, _module(2), (), ())
    with pytest.raises(BasisChangeError, match="wrong image count"):
        ExactBasisIsomorphism.from_images(module, module, (), ())
    with pytest.raises(BasisChangeError, match="literal target"):
        ExactBasisIsomorphism.from_images(
            module, module, (foreign.zero(),), (module.zero(),)
        )
    identity = _identity(module)
    for record in (identity, identity.certificate):
        assert not hasattr(record, "__dict__")
        with pytest.raises(TypeError, match="unhashable"):
            hash(record)
        assert "0x" not in repr(record)
    with pytest.raises(FrozenInstanceError):
        identity.source = foreign  # type: ignore[misc]
    with pytest.raises(BasisChangeError, match="from_images"):
        ExactBasisIsomorphism(module, module, (), ())
    with pytest.raises(BasisChangeError, match="literal source"):
        identity.forward(foreign.zero())


def test_zero_rank_512_boundary_and_513_rejection() -> None:
    zero = _module(0)
    identity = _identity(zero)
    assert identity.forward(zero.zero()) == zero.zero()
    module = _module(512)
    boundary = _identity(module)
    assert boundary.certificate.source_rank == 512
    with pytest.raises(BasisChangeError, match="maximum 512 basis images"):
        ExactBasisIsomorphism.from_images(_module(513), _module(513), (), ())


def test_structure_constant_transport_permutation_and_reverse_equality() -> None:
    source = _module(2)
    target = _module(2)
    change = ExactBasisIsomorphism.from_images(
        source,
        target,
        (_basis(target, 1), _basis(target, 0)),
        (_basis(source, 1), _basis(source, 0)),
    )
    constants = StructureConstants.from_sparse(
        (source.basis, source.basis),
        source.basis,
        (
            ((0, 0, 0), ZZ().element(1)),
            ((0, 1, 1), ZZ().element(2)),
            ((1, 0, 0), ZZ().element(-1)),
        ),
    )
    transported = change_basis(constants, change)
    assert type(transported) is StructureConstants
    assert transported.arity == 2
    restored = change_basis(transported, change.reversed())
    assert restored == constants


@pytest.mark.parametrize("arity", (0, 1, 3, 257))
def test_constants_nullary_unary_ternary_and_rank_one_high_arity(arity: int) -> None:
    module = _module(1)
    change = _identity(module)
    constants = StructureConstants.from_sparse(
        tuple(module.basis for _ in range(arity)),
        module.basis,
        (((*(0 for _ in range(arity)), 0), ZZ().element(1)),),
    )
    assert change_basis(constants, change) == constants


def test_constants_wrong_basis_type_and_expansion_limit() -> None:
    source = _module(2, qq=True)
    target = _module(2, qq=True)
    dense = ExactBasisIsomorphism.from_images(
        source,
        target,
        (target.element({0: 1, 1: 1}), target.element({0: 1, 1: -1})),
        (
            source.element({0: (1, 2), 1: (1, 2)}),
            source.element({0: (1, 2), 1: (-1, 2)}),
        ),
    )
    constants = StructureConstants.from_sparse(
        tuple(source.basis for _ in range(17)), source.basis, ()
    )
    with pytest.raises(BasisChangeError, match="work limit") as caught:
        change_basis(constants, dense)
    assert caught.value.observed == 65_537
    assert caught.value.maximum == 65_536
    wrong = StructureConstants.from_sparse((target.basis,), target.basis, ())
    with pytest.raises(BasisChangeError, match="literal source"):
        change_basis(wrong, dense)
    with pytest.raises(BasisChangeError, match="SparseElement or StructureConstants"):
        change_basis(object(), dense)


def test_mixed_qq_structure_constant_hand_oracle_and_reverse() -> None:
    source = _module(2, qq=True)
    target = _module(2, qq=True)
    change = ExactBasisIsomorphism.from_images(
        source,
        target,
        (target.element({0: 1, 1: 1}), target.element({0: 1, 1: -1})),
        (
            source.element({0: (1, 2), 1: (1, 2)}),
            source.element({0: (1, 2), 1: (-1, 2)}),
        ),
    )
    source_constants = StructureConstants.from_sparse(
        (source.basis, source.basis),
        source.basis,
        (
            ((0, 0, 0), QQ().element(1)),
            ((0, 1, 1), QQ().element(1)),
            ((1, 0, 1), QQ().element(-1)),
            ((1, 1, 0), QQ().element(1)),
        ),
    )
    expected = StructureConstants.from_sparse(
        (target.basis, target.basis),
        target.basis,
        (
            ((0, 0, 0), QQ().element((1, 2))),
            ((0, 0, 1), QQ().element((1, 2))),
            ((0, 1, 0), QQ().element((-1, 2))),
            ((0, 1, 1), QQ().element((1, 2))),
            ((1, 0, 0), QQ().element((1, 2))),
            ((1, 0, 1), QQ().element((-1, 2))),
            ((1, 1, 0), QQ().element((1, 2))),
            ((1, 1, 1), QQ().element((1, 2))),
        ),
    )
    transported = cast(StructureConstants, change_basis(source_constants, change))
    assert transported == expected
    assert transported.coefficients_for_basis(
        0, 1
    ) != transported.coefficients_for_basis(1, 0)
    assert change_basis(transported, change.reversed()) == source_constants


def test_image_snapshot_subclass_and_change_basis_boundaries() -> None:
    module = _module(1)

    class BrokenIter:
        def __iter__(self) -> object:
            raise RuntimeError("secret iter payload")

    class BrokenNext:
        def __iter__(self) -> BrokenNext:
            return self

        def __next__(self) -> object:
            raise RuntimeError("secret next payload")

    for images in (BrokenIter(), BrokenNext()):
        with pytest.raises(BasisChangeError, match="could not be iterated") as caught:
            ExactBasisIsomorphism.from_images(module, module, images, ())
        assert "secret" not in str(caught.value)
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None

    class SparseSubclass(SparseElement):
        pass

    with pytest.raises(BasisChangeError, match="forward image"):
        ExactBasisIsomorphism.from_images(
            module, module, (object.__new__(SparseSubclass),), (_basis(module, 0),)
        )
    foreign = _module(1)
    with pytest.raises(BasisChangeError, match="inverse image"):
        ExactBasisIsomorphism.from_images(
            module, module, (_basis(module, 0),), (foreign.zero(),)
        )
    identity = _identity(module)
    with pytest.raises(BasisChangeError, match="isomorphism must"):
        change_basis(module.zero(), object())
    with pytest.raises(BasisChangeError, match="literal source"):
        change_basis(foreign.zero(), identity)

    class IsomorphismSubclass(ExactBasisIsomorphism):
        pass

    from anyalgebra.presentations.basis_change import BasisChangeCertificate

    class CertificateSubclass(BasisChangeCertificate):
        pass

    with pytest.raises(BasisChangeError, match="exact ExactBasisIsomorphism"):
        IsomorphismSubclass.from_images(
            module, module, (_basis(module, 0),), (_basis(module, 0),)
        )
    with pytest.raises(BasisChangeError, match="exact BasisChangeCertificate"):
        CertificateSubclass._create(0, 0)


def test_exact_512_image_iterable_and_zero_certificate_counts() -> None:
    zero = _identity(_module(0)).certificate
    assert (
        zero.expected_forward_after_inverse_count,
        zero.evaluated_forward_after_inverse_count,
        zero.expected_inverse_after_forward_count,
        zero.evaluated_inverse_after_forward_count,
    ) == (0, 0, 0, 0)
    module = _module(512)
    accepted = ExactBasisIsomorphism.from_images(
        module,
        module,
        (_basis(module, index) for index in range(512)),
        (_basis(module, index) for index in range(512)),
    )
    assert accepted.certificate.source_rank == 512
    with pytest.raises(BasisChangeError, match="exceeds maximum") as caught:
        ExactBasisIsomorphism.from_images(
            module,
            module,
            (module.zero() for _ in range(513)),
            (_basis(module, index) for index in range(512)),
        )
    assert caught.value.observed == 513
    assert caught.value.maximum == 512
