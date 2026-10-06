"""Constructive exact finite basis changes for free modules and constants.

Only explicitly supplied mutually inverse basis images are accepted.  This
requires a domain-supplied exact ``one`` plus working ``SparseElement.add`` and
``SparseElement.scale`` capabilities; it intentionally never infers or inverts
matrices and is not a general module-isomorphism decision procedure.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from typing import cast

from anyalgebra.algebra.multilinear import StructureConstants
from anyalgebra.core.domains import DomainElement
from anyalgebra.core.elements import SparseElement
from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.modules import FreeModule


_MAX_BASIS_IMAGES = 512
_MAX_EXPANSION_WORK = 65_536


class BasisChangeError(AnyAlgebraError, ValueError):
    """One exact basis-change declaration, certification, or transport failed."""

    def __init__(
        self,
        reason: str,
        *,
        side: str | None = None,
        index: int | None = None,
        observed: int | None = None,
        maximum: int | None = None,
    ) -> None:
        self.reason = reason
        self.side = side
        self.index = index
        self.observed = observed
        self.maximum = maximum
        detail = reason
        if side is not None:
            detail = f"{detail}; side={side}"
        if index is not None:
            detail = f"{detail}; index={index}"
        if observed is not None and maximum is not None:
            detail = f"{detail}; observed={observed}; maximum={maximum}"
        super().__init__(f"invalid exact basis change: {detail}")


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class BasisChangeCertificate:
    """Finite computational witness for the supplied image composites."""

    algorithm: str
    source_rank: int
    target_rank: int
    expected_forward_after_inverse_count: int
    evaluated_forward_after_inverse_count: int
    expected_inverse_after_forward_count: int
    evaluated_inverse_after_forward_count: int
    exact_certified: bool
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise BasisChangeError("certificate is produced by ExactBasisIsomorphism")

    @classmethod
    def _create(cls, source_rank: int, target_rank: int) -> BasisChangeCertificate:
        if cls is not BasisChangeCertificate:
            raise BasisChangeError("factory requires exact BasisChangeCertificate")
        certificate = object.__new__(cls)
        object.__setattr__(certificate, "algorithm", "explicit-basis-composites-v1")
        object.__setattr__(certificate, "source_rank", source_rank)
        object.__setattr__(certificate, "target_rank", target_rank)
        object.__setattr__(
            certificate, "expected_forward_after_inverse_count", target_rank
        )
        object.__setattr__(
            certificate, "evaluated_forward_after_inverse_count", target_rank
        )
        object.__setattr__(
            certificate, "expected_inverse_after_forward_count", source_rank
        )
        object.__setattr__(
            certificate, "evaluated_inverse_after_forward_count", source_rank
        )
        object.__setattr__(certificate, "exact_certified", True)
        return certificate

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        return (
            "BasisChangeCertificate(algorithm='explicit-basis-composites-v1', "
            f"source_rank={self.source_rank}, target_rank={self.target_rank}, "
            f"exact_certified={self.exact_certified})"
        )


def _exception_name(error: Exception) -> str:
    name = type(error).__name__
    return name if type(name) is str and name.isidentifier() else "exception"


def _module(value: object, field: str) -> FreeModule:
    if type(value) is not FreeModule:
        raise BasisChangeError(f"{field} must be an exact FreeModule")
    if value.rank > _MAX_BASIS_IMAGES:
        raise BasisChangeError(
            "module rank exceeds maximum 512 basis images",
            observed=value.rank,
            maximum=_MAX_BASIS_IMAGES,
        )
    return value


def _snapshot_images(value: object, field: str) -> tuple[object, ...]:
    if isinstance(value, str | bytes):
        raise BasisChangeError(f"{field} must be an ordered image iterable")
    iterator_failure: str | None = None
    try:
        iterator = iter(cast(Iterable[object], value))
    except Exception as error:
        iterator = None
        iterator_failure = _exception_name(error)
    if iterator_failure is not None:
        raise BasisChangeError(f"{field} could not be iterated ({iterator_failure})")
    assert iterator is not None
    images: list[object] = []
    iteration_failure: str | None = None
    for _ in range(_MAX_BASIS_IMAGES + 1):
        try:
            image = next(iterator)
        except StopIteration:
            break
        except Exception as error:
            iteration_failure = _exception_name(error)
            break
        images.append(image)
    if iteration_failure is not None:
        raise BasisChangeError(f"{field} could not be iterated ({iteration_failure})")
    if len(images) > _MAX_BASIS_IMAGES:
        raise BasisChangeError(
            f"{field} exceeds maximum 512 basis images",
            observed=len(images),
            maximum=_MAX_BASIS_IMAGES,
        )
    return tuple(images)


def _basis_vector(module: FreeModule, index: int) -> SparseElement:
    failure: str | None = None
    try:
        one = module.domain.element(1)
        vector = module.element({index: one})
    except Exception as error:
        failure = _exception_name(error)
        vector = None
    if (
        failure is not None
        or type(vector) is not SparseElement
        or vector.parent is not module
    ):
        raise BasisChangeError(
            f"could not construct exact basis vector ({failure or 'invalid-result'})",
            index=index,
        )
    return vector


def _linear_apply(
    element: SparseElement,
    images: tuple[SparseElement, ...],
    target: FreeModule,
    *,
    side: str,
) -> SparseElement:
    if type(element) is not SparseElement:
        raise BasisChangeError("value must be an exact SparseElement", side=side)
    result = target.zero()
    for index, coefficient in element.coordinates().items():
        failure: str | None = None
        try:
            term = images[index].scale(coefficient)
            result = result.add(term)
        except Exception as error:
            failure = _exception_name(error)
        if failure is not None:
            raise BasisChangeError(
                f"linear extension arithmetic failed ({failure})",
                side=side,
                index=index,
            )
    return result


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ExactBasisIsomorphism:
    """A finite exact linear isomorphism certified from explicit inverse images."""

    source: FreeModule
    target: FreeModule
    forward_images: tuple[SparseElement, ...]
    inverse_images: tuple[SparseElement, ...]
    certificate: BasisChangeCertificate
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise BasisChangeError("use ExactBasisIsomorphism.from_images")

    @classmethod
    def from_images(
        cls,
        source: object,
        target: object,
        forward_images: object,
        inverse_images: object,
    ) -> ExactBasisIsomorphism:
        """Validate supplied images and certify both finite basis composites."""
        if cls is not ExactBasisIsomorphism:
            raise BasisChangeError("factory requires exact ExactBasisIsomorphism")
        source_parent = _module(source, "source")
        target_parent = _module(target, "target")
        if source_parent.domain is not target_parent.domain:
            raise BasisChangeError(
                "source and target must share the literal coefficient domain"
            )
        if source_parent.rank != target_parent.rank:
            raise BasisChangeError("source and target must have equal finite rank")
        forward_raw = _snapshot_images(forward_images, "forward_images")
        inverse_raw = _snapshot_images(inverse_images, "inverse_images")
        if len(forward_raw) != source_parent.rank:
            raise BasisChangeError("forward_images has the wrong image count")
        if len(inverse_raw) != target_parent.rank:
            raise BasisChangeError("inverse_images has the wrong image count")
        for index, image in enumerate(forward_raw):
            if type(image) is not SparseElement or image.parent is not target_parent:
                raise BasisChangeError(
                    "forward image must be an exact SparseElement of the "
                    "literal target",
                    side="forward",
                    index=index,
                )
        for index, image in enumerate(inverse_raw):
            if type(image) is not SparseElement or image.parent is not source_parent:
                raise BasisChangeError(
                    "inverse image must be an exact SparseElement of the "
                    "literal source",
                    side="inverse",
                    index=index,
                )
        forward = cast(tuple[SparseElement, ...], forward_raw)
        inverse = cast(tuple[SparseElement, ...], inverse_raw)
        # forward-after-inverse acts on target-basis images.
        for index in range(target_parent.rank):
            actual = _linear_apply(
                inverse[index], forward, target_parent, side="forward-after-inverse"
            )
            if actual != _basis_vector(target_parent, index):
                raise BasisChangeError(
                    "forward-after-inverse composite is not the target basis identity",
                    side="forward-after-inverse",
                    index=index,
                )
        # inverse-after-forward acts on source-basis images.
        for index in range(source_parent.rank):
            actual = _linear_apply(
                forward[index], inverse, source_parent, side="inverse-after-forward"
            )
            if actual != _basis_vector(source_parent, index):
                raise BasisChangeError(
                    "inverse-after-forward composite is not the source basis identity",
                    side="inverse-after-forward",
                    index=index,
                )
        value = object.__new__(cls)
        object.__setattr__(value, "source", source_parent)
        object.__setattr__(value, "target", target_parent)
        object.__setattr__(value, "forward_images", forward)
        object.__setattr__(value, "inverse_images", inverse)
        object.__setattr__(
            value,
            "certificate",
            BasisChangeCertificate._create(source_parent.rank, target_parent.rank),
        )
        return value

    def forward(self, value: object) -> SparseElement:
        """Apply the certified source-to-target linear extension."""
        if type(value) is not SparseElement or value.parent is not self.source:
            raise BasisChangeError("forward value must have the literal source parent")
        return _linear_apply(value, self.forward_images, self.target, side="forward")

    def inverse(self, value: object) -> SparseElement:
        """Apply the certified target-to-source linear extension."""
        if type(value) is not SparseElement or value.parent is not self.target:
            raise BasisChangeError("inverse value must have the literal target parent")
        return _linear_apply(value, self.inverse_images, self.source, side="inverse")

    def reversed(self) -> ExactBasisIsomorphism:
        """Return the independently rechecked opposite exact basis isomorphism."""
        return ExactBasisIsomorphism.from_images(
            self.target, self.source, self.inverse_images, self.forward_images
        )

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        return f"ExactBasisIsomorphism(rank={self.source.rank}, exact_certified=True)"


def _evaluate_constants(
    constants: StructureConstants,
    arguments: tuple[SparseElement, ...],
    source: FreeModule,
    work: list[int],
) -> SparseElement:
    result = source.zero()
    supports = tuple(tuple(argument.coordinates().items()) for argument in arguments)
    for selected in product(*supports):
        work[0] += 1
        if work[0] > _MAX_EXPANSION_WORK:
            raise BasisChangeError(
                "basis-change multilinear expansion exceeds the declared work limit",
                observed=work[0],
                maximum=_MAX_EXPANSION_WORK,
            )
        indices = tuple(index for index, _ in selected)
        output = source.element(dict(constants.coefficients_for_basis(*indices)))
        for _, coefficient in selected:
            output = output.scale(coefficient)
        result = result.add(output)
    return result


def _transport_constants(
    constants: StructureConstants, isomorphism: ExactBasisIsomorphism
) -> StructureConstants:
    if constants.coefficient_domain is not isomorphism.source.domain:
        raise BasisChangeError(
            "constants must have the literal source coefficient parent"
        )
    if constants.output_basis is not isomorphism.source.basis or any(
        basis is not isomorphism.source.basis for basis in constants.input_bases
    ):
        raise BasisChangeError(
            "constants must use the literal source basis in every slot"
        )
    target = isomorphism.target
    work = [0]
    entries: list[tuple[tuple[int, ...], DomainElement[object]]] = []
    for input_indices in product(range(target.rank), repeat=constants.arity):
        arguments = tuple(isomorphism.inverse_images[index] for index in input_indices)
        output = isomorphism.forward(
            _evaluate_constants(constants, arguments, isomorphism.source, work)
        )
        for output_index, coefficient in output.coordinates().items():
            entries.append(((*input_indices, output_index), coefficient))
    try:
        return StructureConstants.from_sparse(
            tuple(target.basis for _ in range(constants.arity)), target.basis, entries
        )
    except Exception as error:
        failure = _exception_name(error)
    raise BasisChangeError(
        f"transported constants could not be constructed ({failure})"
    )


def change_basis(
    value_or_structure: object, isomorphism: object
) -> SparseElement | StructureConstants:
    """Transport one source-parent element or source-basis sparse constants."""
    if type(isomorphism) is not ExactBasisIsomorphism:
        raise BasisChangeError("isomorphism must be an exact ExactBasisIsomorphism")
    if type(value_or_structure) is SparseElement:
        return isomorphism.forward(value_or_structure)
    if type(value_or_structure) is StructureConstants:
        return _transport_constants(value_or_structure, isomorphism)
    raise BasisChangeError("value must be an exact SparseElement or StructureConstants")
