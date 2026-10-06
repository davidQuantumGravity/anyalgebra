"""Tests for immutable named sorts and ordered finite carriers."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import (
    CarrierDefinitionError,
    CarrierLabelNotFoundError,
    CarrierMemberNotFoundError,
    FiniteCarrier,
    Sort,
    SortDefinitionError,
)


class _StringSubclass(str):
    """A string subtype used to prove exact built-in-string validation."""


class _BrokenEquality:
    """An item whose equality cannot define finite-carrier membership."""

    def __eq__(self, other: object) -> bool:
        """Reject every non-identical equality comparison."""
        raise RuntimeError("comparison unavailable")


@pytest.mark.parametrize("name", ["", " ", " points", "points ", 3, True])
def test_sort_rejects_malformed_names(name: object) -> None:
    """Sort names are nonempty, unpadded, exact built-in strings."""
    with pytest.raises(SortDefinitionError) as caught:
        Sort(name)  # type: ignore[arg-type]

    assert isinstance(caught.value, AnyAlgebraError)
    assert caught.value.name is name
    assert caught.value.reason


def test_sort_is_an_immutable_structural_value() -> None:
    """Equal names define equal sorts without canonicalizing their instances."""
    left = Sort("points")
    right = Sort("points")

    assert left == right
    assert left is not right
    assert hash(left) == hash(right)
    assert repr(left) == "Sort(name='points')"
    with pytest.raises(FrozenInstanceError):
        left.name = "vectors"  # type: ignore[misc]

    with pytest.raises(SortDefinitionError):
        Sort(_StringSubclass("points"))


def test_carrier_preserves_declared_order_sort_instance_and_labels() -> None:
    """Iteration, indexing, and labeled lookup share one declared order."""
    sort = Sort("points")
    source_items = ["origin", "north", "east"]
    source_labels = ["o", "n", "e"]
    carrier = FiniteCarrier(source_items, sort=sort, labels=source_labels)
    source_items.append("west")
    source_labels.append("w")

    assert carrier.sort is sort
    assert carrier.items == ("origin", "north", "east")
    assert carrier.labels == ("o", "n", "e")
    assert tuple(carrier) == carrier.items
    assert len(carrier) == 3
    assert carrier[0] == "origin"
    assert carrier[-1] == "east"
    assert carrier.index("north") == 1
    assert carrier.label_for("east") == "e"
    assert carrier.index_for_label("n") == 1
    assert carrier.item_for_label("e") == "east"
    assert "north" in carrier
    assert "west" not in carrier


def test_singleton_unlabeled_carrier_is_the_smallest_valid_carrier() -> None:
    """A singleton is valid while omitted labels remain explicitly absent."""
    carrier = FiniteCarrier([[]], sort=Sort("mutable-payloads"))

    assert carrier.items == ([],)
    assert carrier.labels is None
    assert [] in carrier
    assert carrier.index([]) == 0
    with pytest.raises(CarrierLabelNotFoundError) as caught:
        carrier.index_for_label("only")
    assert caught.value.reason == "carrier has no labels"


def test_empty_carrier_is_rejected() -> None:
    """The v0.0 finite-carrier slice requires at least one declared item."""
    with pytest.raises(CarrierDefinitionError) as caught:
        FiniteCarrier((), sort=Sort("empty"))

    assert caught.value.reason == "carrier must contain at least one item"


def test_duplicate_items_report_the_first_conflict_without_hashing() -> None:
    """Equality scanning supports unhashable items and reports first indices."""
    with pytest.raises(CarrierDefinitionError) as caught:
        FiniteCarrier([[1], [2], [1], [1]], sort=Sort("lists"))

    error = caught.value
    assert error.reason == "duplicate carrier item"
    assert error.index == 2
    assert error.conflicting_index == 0
    assert "0 and 2" in str(error)
    assert "object at" not in str(error)


def test_unusable_item_equality_is_a_typed_definition_diagnostic() -> None:
    """A carrier rejects item equality that cannot produce a truth value."""
    with pytest.raises(CarrierDefinitionError) as caught:
        FiniteCarrier(
            [_BrokenEquality(), _BrokenEquality()], sort=Sort("broken-equality")
        )

    assert caught.value.reason == "carrier item equality comparison failed"
    assert caught.value.index == 1
    assert caught.value.conflicting_index is None


def test_label_validation_and_first_duplicate_are_deterministic() -> None:
    """Labels must align with items and be unique exact unpadded strings."""
    sort = Sort("points")
    malformed: list[object] = [
        ["a"],
        "ab",
        ["a", ""],
        ["a", " b"],
        ["a", _StringSubclass("b")],
    ]
    for labels in malformed:
        with pytest.raises(CarrierDefinitionError):
            FiniteCarrier([1, 2], sort=sort, labels=labels)  # type: ignore[arg-type]

    with pytest.raises(CarrierDefinitionError) as caught:
        FiniteCarrier([1, 2, 3, 4], sort=sort, labels=["a", "b", "a", "a"])

    assert caught.value.reason == "duplicate carrier label"
    assert caught.value.index == 2
    assert caught.value.conflicting_index == 0


def test_carrier_rejects_malformed_sort_and_item_collection() -> None:
    """Construction failures are normalized to task-local typed errors."""
    with pytest.raises(CarrierDefinitionError, match="exact Sort"):
        FiniteCarrier([1], sort="numbers")  # type: ignore[arg-type]

    with pytest.raises(CarrierDefinitionError, match="iterable"):
        FiniteCarrier(7, sort=Sort("numbers"))  # type: ignore[arg-type]


def test_absent_member_and_label_have_distinct_typed_diagnostics() -> None:
    """Failed item and label queries preserve their query kind and sort."""
    sort = Sort("points")
    carrier = FiniteCarrier([10, 20], sort=sort, labels=["a", "b"])

    with pytest.raises(CarrierMemberNotFoundError) as member_caught:
        carrier.index(30)
    assert member_caught.value.sort is sort
    assert member_caught.value.member == 30
    assert "item is not a member" in str(member_caught.value)

    with pytest.raises(CarrierLabelNotFoundError) as label_caught:
        carrier.item_for_label("z")
    assert label_caught.value.sort is sort
    assert label_caught.value.label == "z"
    assert label_caught.value.reason == "label is not present"

    with pytest.raises(CarrierLabelNotFoundError, match="built-in str"):
        carrier.index_for_label(_StringSubclass("a"))


def test_carrier_is_immutable_structural_and_deliberately_unhashable() -> None:
    """Carrier equality is structural; universal hashing cannot admit list items."""
    original_sort = Sort("points")
    carrier = FiniteCarrier([[1], [2]], sort=original_sort, labels=["a", "b"])
    equal_looking = FiniteCarrier([[1], [2]], sort=Sort("points"), labels=("a", "b"))
    different_order = FiniteCarrier([[2], [1]], sort=Sort("points"), labels=("a", "b"))

    assert carrier == equal_looking
    assert carrier.sort is original_sort
    assert carrier != different_order
    with pytest.raises(TypeError, match="unhashable"):
        hash(carrier)
    with pytest.raises(FrozenInstanceError):
        carrier.items = (3,)  # type: ignore[misc]


def test_indexing_rejects_non_integer_and_out_of_range_queries() -> None:
    """Index access does not silently accept bool or expose tuple mutation."""
    carrier = FiniteCarrier(["a"], sort=Sort("letters"))

    with pytest.raises(TypeError, match="built-in int"):
        carrier[True]
    with pytest.raises(IndexError, match="out of range"):
        carrier[1]
