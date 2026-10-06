"""Constructive deterministic permutation groups for census equivalence policies."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

from anyalgebra.core.errors import AnyAlgebraError

from .equivalence import EquivalencePolicy


MAX_GENERATED_PERMUTATIONS = 1_000_000


class PermutationGenerationError(AnyAlgebraError, ValueError):
    """A permutation, policy, or bounded generation request was invalid."""

    def __init__(
        self, *, field: str, reason: str, requested: int | None = None
    ) -> None:
        self.field = field
        self.reason = reason
        self.requested = requested
        super().__init__(f"invalid carrier permutation {field}: {reason}")


def _policy(value: object) -> EquivalencePolicy:
    if type(value) is not EquivalencePolicy:
        raise PermutationGenerationError(
            field="policy", reason="must be an exact EquivalencePolicy"
        )
    return value


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class CarrierPermutation:
    """One policy-owned old-index-to-new-index carrier bijection."""

    policy: EquivalencePolicy
    images: tuple[int, ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise PermutationGenerationError(field="permutation", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("CarrierPermutation cannot be subclassed")

    @classmethod
    def create(
        cls, policy: EquivalencePolicy, images: tuple[int, ...]
    ) -> CarrierPermutation:
        if cls is not CarrierPermutation:
            raise PermutationGenerationError(
                field="permutation", reason="factory requires exact type"
            )
        checked = _policy(policy)
        if type(images) is not tuple:
            raise PermutationGenerationError(
                field="images", reason="must be an exact tuple"
            )
        try:
            allowed = checked.allows(images)
        except Exception as error:
            raise PermutationGenerationError(
                field="images", reason="must be a carrier bijection"
            ) from error
        if not allowed:
            raise PermutationGenerationError(
                field="images", reason="bijection is forbidden by the policy"
            )
        return cls._create(checked, images)

    @classmethod
    def _create(
        cls, policy: EquivalencePolicy, images: tuple[int, ...]
    ) -> CarrierPermutation:
        value = object.__new__(CarrierPermutation)
        object.__setattr__(value, "policy", policy)
        object.__setattr__(value, "images", images)
        return value

    @classmethod
    def identity(cls, policy: EquivalencePolicy) -> CarrierPermutation:
        checked = _policy(policy)
        return cls.create(checked, tuple(range(checked.carrier_size)))

    def __call__(self, old_index: int) -> int:
        if type(old_index) is not int or not 0 <= old_index < len(self.images):
            raise PermutationGenerationError(
                field="element", reason="must be a carrier index"
            )
        return self.images[old_index]

    def inverse(self) -> CarrierPermutation:
        """Return the new-index-to-old-index inverse under the same literal policy."""
        images = [0] * len(self.images)
        for old, new in enumerate(self.images):
            images[new] = old
        return CarrierPermutation._create(self.policy, tuple(images))

    def then(self, after: CarrierPermutation) -> CarrierPermutation:
        """Compose old-to-new maps as ``after(self(old))``."""
        if type(after) is not CarrierPermutation or after.policy is not self.policy:
            raise PermutationGenerationError(
                field="policy", reason="composition requires the literal same policy"
            )
        return CarrierPermutation._create(
            self.policy, tuple(after.images[image] for image in self.images)
        )

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is CarrierPermutation
            and self.policy == other.policy
            and self.images == other.images
        )

    def __repr__(self) -> str:
        if len(self.images) <= 16:
            return (
                f"CarrierPermutation(size={len(self.images)}, images={self.images!r})"
            )
        return f"CarrierPermutation(size={len(self.images)})"


def generate_allowed_permutations(
    policy: EquivalencePolicy,
) -> tuple[CarrierPermutation, ...]:
    """Generate exactly the declared group, sorted by complete image tuples."""
    checked = _policy(policy)
    if checked.kind == "literal":
        return (CarrierPermutation.identity(checked),)
    if checked.permutation_count > MAX_GENERATED_PERMUTATIONS:
        raise PermutationGenerationError(
            field="permutation_count",
            reason="generation safety limit exceeded",
            requested=checked.permutation_count,
        )
    fixed = set(checked.fixed_elements)
    block_sources = tuple(
        tuple(element for element in block.elements if element not in fixed)
        for block in checked.sort_blocks
    )
    choices = tuple(tuple(permutations(source)) for source in block_sources)
    image_records: list[tuple[int, ...]] = []
    for block_images in product(*choices):
        images = list(range(checked.carrier_size))
        for sources, targets in zip(block_sources, block_images, strict=True):
            for source, target in zip(sources, targets, strict=True):
                images[source] = target
        image_records.append(tuple(images))
    ordered = tuple(sorted(image_records))
    if len(ordered) != checked.permutation_count:
        raise PermutationGenerationError(
            field="permutation_count", reason="constructive count mismatch"
        )
    return tuple(CarrierPermutation._create(checked, images) for images in ordered)


__all__ = (
    "MAX_GENERATED_PERMUTATIONS",
    "CarrierPermutation",
    "PermutationGenerationError",
    "generate_allowed_permutations",
)
