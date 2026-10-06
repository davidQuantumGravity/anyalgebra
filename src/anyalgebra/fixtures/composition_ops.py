"""Convention-scoped generic maps for the four named composition fixtures."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import cast

from anyalgebra.algebra.multilinear import (
    FiniteMultilinearStructure,
    evaluate_multilinear,
)
from anyalgebra.core.domains import DomainElement, ZZ
from anyalgebra.core.elements import SparseElement
from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import SemanticHash

from .composition import ProductTable, _FIXTURES, _CompositionFixtureMetadata


_SCHEMA_VERSION = 1


class CompositionOperationError(AnyAlgebraError, ValueError):
    """A named composition operation violated its exact convention boundary."""

    def __init__(self, *, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid composition operation {field}: {reason}")


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class CompositionOperations:
    """Generic exact maps attached to one validated named fixture parent."""

    structure: FiniteMultilinearStructure
    identifier: str
    basis_labels: tuple[str, ...]
    conjugation_signs: tuple[int, ...]
    source_anchors: tuple[str, ...]
    algorithm: str
    algorithm_version: int
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise CompositionOperationError(field="operations", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("CompositionOperations cannot be subclassed")

    def __eq__(self, other: object) -> bool:
        return type(other) is CompositionOperations and (
            self.structure is other.structure
            and self.identifier == other.identifier
            and self.conjugation_signs == other.conjugation_signs
            and self.semantic_hash == other.semantic_hash
        )

    def _element(self, value: object, *, field: str) -> SparseElement:
        if (
            type(value) is not SparseElement
            or value.parent is not self.structure.module
        ):
            raise CompositionOperationError(
                field=field, reason="must belong to the literal fixture module"
            )
        for coefficient in value.coordinates().values():
            if coefficient.parent is not ZZ() or type(coefficient.value) is not int:
                raise CompositionOperationError(
                    field=field, reason="has a malformed exact ZZ coefficient"
                )
        return value

    def conjugate(self, value: SparseElement) -> SparseElement:
        """Fix the scalar basis coordinate and negate every imaginary one."""
        checked = self._element(value, field="element")
        return self.structure.module.element(
            {
                index: cast(int, coefficient.value) * self.conjugation_signs[index]
                for index, coefficient in checked.coordinates().items()
            }
        )

    def real_part(self, value: SparseElement) -> DomainElement[int]:
        """Return the exact coordinate of the convention's distinguished unit."""
        checked = self._element(value, field="element")
        coefficient = checked.coordinates().get(0)
        if coefficient is None:
            return ZZ().element(0)
        return cast(DomainElement[int], coefficient)

    def trace(self, value: SparseElement) -> DomainElement[int]:
        """Return ``x + conjugate(x)`` as its exact scalar coefficient."""
        return ZZ().element(2 * self.real_part(value).value)

    def multiply(self, left: SparseElement, right: SparseElement) -> SparseElement:
        """Evaluate the fixture's literal declared product on two exact elements."""
        checked_left = self._element(left, field="left")
        checked_right = self._element(right, field="right")
        try:
            return evaluate_multilinear(self.structure, checked_left, checked_right)
        except Exception as error:
            raise CompositionOperationError(
                field="product", reason="exact generic multiplication failed"
            ) from error

    def quadratic_norm(self, value: SparseElement) -> DomainElement[int]:
        """Return the scalar coefficient of ``x * conjugate(x)`` exactly."""
        checked = self._element(value, field="element")
        product = self.multiply(checked, self.conjugate(checked))
        coordinates = product.coordinates()
        if any(index != 0 for index in coordinates):
            raise CompositionOperationError(
                field="quadratic_norm", reason="product is not scalar"
            )
        coefficient = coordinates.get(0)
        if coefficient is None:
            return ZZ().element(0)
        return cast(DomainElement[int], coefficient)


def _signed_table(structure: FiniteMultilinearStructure) -> ProductTable:
    rows: list[tuple[tuple[int, int], ...]] = []
    for left in range(structure.module.rank):
        row: list[tuple[int, int]] = []
        for right in range(structure.module.rank):
            coordinates = tuple(
                structure.evaluate_basis(left, right).coordinates().items()
            )
            if len(coordinates) != 1:
                raise CompositionOperationError(
                    field="structure", reason="basis product is not one signed basis"
                )
            output, coefficient = coordinates[0]
            if coefficient.parent is not ZZ() or type(coefficient.value) is not int:
                raise CompositionOperationError(
                    field="structure",
                    reason="basis product coefficient is not exact ZZ",
                )
            row.append((coefficient.value, output))
        rows.append(tuple(row))
    return tuple(rows)


def _metadata(structure: object) -> _CompositionFixtureMetadata:
    if type(structure) is not FiniteMultilinearStructure:
        raise CompositionOperationError(
            field="structure", reason="must be exact FiniteMultilinearStructure"
        )
    if structure.module.domain is not ZZ() or structure.arity != 2:
        raise CompositionOperationError(
            field="structure", reason="must be a binary ZZ fixture"
        )
    candidate = next(
        (metadata for metadata in _FIXTURES if metadata.identifier == structure.name),
        None,
    )
    if (
        candidate is None
        or structure.module.basis.labels != candidate.basis_labels
        or _signed_table(structure) != candidate.product_table
    ):
        raise CompositionOperationError(
            field="structure", reason="does not match a named exact convention table"
        )
    return candidate


def composition_operations(
    structure: FiniteMultilinearStructure,
) -> CompositionOperations:
    """Attach the generic composition maps after exact table validation."""
    metadata = _metadata(structure)
    value = object.__new__(CompositionOperations)
    for field, item in (
        ("structure", structure),
        ("identifier", metadata.identifier),
        ("basis_labels", metadata.basis_labels),
        ("conjugation_signs", (1,) + (-1,) * (len(metadata.basis_labels) - 1)),
        ("source_anchors", metadata.source_anchors),
        ("algorithm", "anyalgebra.named_composition_maps"),
        ("algorithm_version", 1),
        ("semantic_hash", SemanticHash("sha256", "0" * 64)),
    ):
        object.__setattr__(value, field, item)
    object.__setattr__(value, "semantic_hash", _semantic_hash(_body(value)))
    return value


def _encoded(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _semantic_hash(body: dict[str, object]) -> SemanticHash:
    return SemanticHash("sha256", hashlib.sha256(_encoded(body)).hexdigest())


def _body(value: CompositionOperations) -> dict[str, object]:
    return {
        "algorithm": value.algorithm,
        "algorithmVersion": value.algorithm_version,
        "basisLabels": list(value.basis_labels),
        "conjugationSigns": list(value.conjugation_signs),
        "identifier": value.identifier,
        "productTable": [
            [[coefficient, output] for coefficient, output in row]
            for row in _signed_table(value.structure)
        ],
        "schemaType": "anyalgebra.fixtures.composition_operations",
        "schemaVersion": _SCHEMA_VERSION,
        "sourceAnchors": list(value.source_anchors),
    }


def composition_operations_record(value: CompositionOperations) -> dict[str, object]:
    if type(value) is not CompositionOperations:
        raise CompositionOperationError(
            field="operations", reason="must be exact CompositionOperations"
        )
    expected = composition_operations(value.structure)
    if value != expected:
        raise CompositionOperationError(field="operations", reason="content drift")
    body = _body(value)
    semantic_hash = _semantic_hash(body)
    if semantic_hash != value.semantic_hash:
        raise CompositionOperationError(field="semantic_hash", reason="content drift")
    return {
        **body,
        "contentHash": {
            "algorithm": semantic_hash.algorithm,
            "digest": semantic_hash.digest,
        },
    }


def composition_operations_canonical_bytes(value: CompositionOperations) -> bytes:
    return _encoded(composition_operations_record(value))


__all__ = (
    "CompositionOperationError",
    "CompositionOperations",
    "composition_operations",
    "composition_operations_canonical_bytes",
    "composition_operations_record",
)
