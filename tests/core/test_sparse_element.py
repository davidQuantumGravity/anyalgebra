"""Contract tests for canonical immutable sparse free-module elements."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from itertools import permutations

import pytest

from anyalgebra.core.coercions import CoercionGraph, CoercionMap
from anyalgebra.core.domains import DomainElement, QQ, ZZ
from anyalgebra.core.elements import (
    CoordinateIndexError,
    SparseElement,
    SparseElementConstructionError,
)
from anyalgebra.core.errors import (
    CoercionAmbiguityError,
    CoercionError,
    LossyCoercionError,
)
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.core.rational import Rational


def _module(domain: object, rank: int) -> FreeModule:
    """Build a finite module with stable ordered test labels."""
    labels = tuple(f"e{index}" for index in range(rank))
    return FreeModule(domain, Basis(labels, coefficient_domain=domain))  # type: ignore[arg-type]


def _zz_to_qq_graph(*, calls: list[int] | None = None) -> CoercionGraph:
    """Declare the canonical exact integer embedding used by coefficient tests."""

    def forward(value: DomainElement[int]) -> DomainElement[Rational]:
        if calls is not None:
            calls.append(value.value)
        return QQ().element(value.value)

    return CoercionGraph().with_map(
        CoercionMap(
            id="core.zz_to_qq.v1",
            source=ZZ(),
            target=QQ(),
            forward=forward,
            injective=True,
            exact=True,
            lossless=True,
        )
    )


def test_sparse_elements_cover_rank_zero_one_and_four_over_zz_and_qq() -> None:
    """Finite modules construct their unique zero and exact sparse values."""
    for domain in (ZZ(), QQ()):
        for rank in (0, 1, 4):
            module = _module(domain, rank)
            zero = module.zero()

            assert zero.parent is module
            assert isinstance(zero.coordinates(), Mapping)
            assert tuple(zero.coordinates().items()) == ()
            assert module.element({}) == zero
            if rank:
                value = module.element({rank - 1: 3})
                assert value.parent is module
                assert value.coordinates()[rank - 1].parent is domain
                assert value.coordinates()[rank - 1].value == domain.normalize(3)


def test_sparse_canonicalizes_raw_and_same_domain_coefficients() -> None:
    """Construction normalizes rational literals and preserves owned elements."""
    module = _module(QQ(), 4)
    owned = QQ().element((-2, -4))
    value = module.element({3: (0, -9), 2: (6, 8), 0: owned, 1: -0})

    assert tuple(value.coordinates().items()) == (
        (0, owned),
        (2, QQ().element(Rational(3, 4))),
    )
    assert value.coordinates()[0] is owned


def test_sparse_order_and_value_are_independent_of_mapping_insertion_order() -> None:
    """Every insertion permutation produces the same ascending canonical support."""
    module = _module(QQ(), 4)
    entries = ((3, 0), (0, (-1, 2)), (2, (3, 4)), (1, (0, 17)))
    values = [module.element(dict(order)) for order in permutations(entries)]

    assert all(value == values[0] for value in values)
    assert all(tuple(value.coordinates()) == (0, 2) for value in values)
    assert tuple(values[0].coordinates().items()) == (
        (0, QQ().element((-1, 2))),
        (2, QQ().element((3, 4))),
    )


def test_sparse_does_not_alias_source_or_expose_mutable_coordinates() -> None:
    """Input mappings and returned coordinate views cannot mutate an element."""
    module = _module(ZZ(), 4)
    source = {2: 7, 0: 5}
    value = module.element(source)
    source.clear()
    coordinates = value.coordinates()

    assert tuple(coordinates.items()) == ((0, ZZ().element(5)), (2, ZZ().element(7)))
    with pytest.raises(TypeError):
        coordinates[1] = ZZ().element(9)  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        value.parent = module  # type: ignore[misc]
    with pytest.raises(AttributeError):
        value._coordinate_pairs += ((1, ZZ().element(9)),)  # type: ignore[misc]


def test_sparse_direct_construction_cannot_bypass_module_validation() -> None:
    """The public class cannot forge out-of-range or foreign coordinate pairs."""
    module = _module(ZZ(), 1)

    with pytest.raises(SparseElementConstructionError, match=r"FreeModule\.element"):
        SparseElement(module, ((9, QQ().element(1)),))


@pytest.mark.parametrize("index", [True, False, -1, 4, "0", 1.0, None])
def test_sparse_rejects_invalid_coordinate_indices_before_conversion(
    index: object,
) -> None:
    """Indices are exact built-in integers in the module basis range."""
    module = _module(ZZ(), 4)

    with pytest.raises(CoordinateIndexError) as caught:
        module.element({index: 1})  # type: ignore[dict-item]

    assert caught.value.coordinate_index is index
    assert "coordinate index" in str(caught.value)


def test_sparse_preflights_every_index_before_any_coefficient_conversion() -> None:
    """An invalid later index wins over an invalid earlier coefficient value."""
    with pytest.raises(CoordinateIndexError) as caught:
        _module(ZZ(), 4).element({0: object(), 4: 1})

    assert caught.value.coordinate_index == 4


def test_rank_zero_rejects_every_nonempty_sparse_coordinate() -> None:
    """The zero-rank parent admits only its empty-support element."""
    with pytest.raises(CoordinateIndexError) as caught:
        _module(QQ(), 0).element({0: 0})

    assert caught.value.coordinate_index == 0
    assert caught.value.reason == "coordinate index is outside the module basis range"


@pytest.mark.parametrize("coordinates", [[], (), "", object(), ((0, 1),)])
def test_sparse_requires_an_actual_mapping(coordinates: object) -> None:
    """Dense and pair-sequence presentations are deliberately deferred."""
    with pytest.raises(SparseElementConstructionError) as caught:
        _module(ZZ(), 1).element(coordinates)  # type: ignore[arg-type]

    assert caught.value.reason.startswith("sparse coordinates")
    assert "Mapping" in str(caught.value)


def test_sparse_accepts_unique_explicit_lossless_coefficient_transport() -> None:
    """A foreign coefficient enters only through one supplied lossless route."""
    module = _module(QQ(), 2)
    integer = ZZ().element(7)
    value = module.element({1: integer}, graph=_zz_to_qq_graph())

    assert value.coordinates()[1] == QQ().element(7)
    assert value.coordinates()[1].parent is QQ()


def test_sparse_rejects_foreign_coefficients_without_an_explicit_graph() -> None:
    """A DomainElement is never reinterpreted from its payload or shape."""
    module = _module(QQ(), 1)

    with pytest.raises(SparseElementConstructionError) as caught:
        module.element({0: ZZ().element(1)})

    assert caught.value.coordinate_index == 0
    assert caught.value.coefficient_parent is ZZ()
    assert "foreign coefficient" in str(caught.value)


def test_sparse_refuses_ambiguous_lossy_and_missing_routes_without_calls() -> None:
    """Planning failures happen before any user coercion callable can run."""
    module = _module(QQ(), 1)
    integer = ZZ().element(2)
    calls: list[str] = []

    def forward(value: DomainElement[int]) -> DomainElement[Rational]:
        calls.append("called")
        return QQ().element(value.value)

    first = CoercionMap(
        id="route.a",
        source=ZZ(),
        target=QQ(),
        forward=forward,
        injective=True,
        exact=True,
        lossless=True,
    )
    second = CoercionMap(
        id="route.b",
        source=ZZ(),
        target=QQ(),
        forward=forward,
        injective=True,
        exact=True,
        lossless=True,
    )
    ambiguous = CoercionGraph().with_map(second).with_map(first)
    with pytest.raises(CoercionAmbiguityError):
        module.element({0: integer}, graph=ambiguous)

    lossy = CoercionGraph().with_map(
        CoercionMap(
            id="route.lossy",
            source=ZZ(),
            target=QQ(),
            forward=forward,
            exact=True,
        )
    )
    with pytest.raises(LossyCoercionError):
        module.element({0: integer}, graph=lossy)
    with pytest.raises(CoercionError):
        module.element({0: integer}, graph=CoercionGraph())
    assert calls == []


def test_sparse_plans_all_foreign_routes_before_applying_an_earlier_one() -> None:
    """A later route failure prevents an earlier unique map from having effects."""
    first_source = _UncertainDomain()
    second_source = _UncertainDomain()
    module = _module(QQ(), 2)
    calls: list[str] = []

    def edge(
        identifier: str, source: _UncertainDomain, *, lossless: bool = True
    ) -> CoercionMap[object, Rational]:
        def forward(value: DomainElement[object]) -> DomainElement[Rational]:
            calls.append(identifier)
            return QQ().element(value.value)

        return CoercionMap(
            id=identifier,
            source=source,
            target=QQ(),
            forward=forward,
            injective=lossless,
            exact=True,
            lossless=lossless,
        )

    coordinates = {0: first_source.element(1), 1: second_source.element(2)}
    ambiguous = (
        CoercionGraph()
        .with_map(edge("first.unique", first_source))
        .with_map(edge("second.a", second_source))
        .with_map(edge("second.b", second_source))
    )
    with pytest.raises(CoercionAmbiguityError):
        module.element(coordinates, graph=ambiguous)
    assert calls == []

    lossy = (
        CoercionGraph()
        .with_map(edge("first.unique", first_source))
        .with_map(edge("second.lossy", second_source, lossless=False))
    )
    with pytest.raises(LossyCoercionError):
        module.element(coordinates, graph=lossy)
    assert calls == []


def test_sparse_validates_graph_before_domain_construction_effects() -> None:
    """Even raw coefficient normalization waits until graph validation succeeds."""
    domain = _RecordingDomain()
    module = _module(domain, 1)

    with pytest.raises(TypeError, match="graph"):
        module.element({0: 3}, graph=object())  # type: ignore[arg-type]
    assert domain.calls == []


def test_sparse_element_input_is_idempotent_only_for_literal_same_parent() -> None:
    """Equal-looking separate modules never share an element implicitly."""
    left = _module(ZZ(), 1)
    right = _module(ZZ(), 1)
    value = left.element({0: 4})

    assert left == right
    assert left.element(value) is value
    with pytest.raises(SparseElementConstructionError) as caught:
        right.element(value)
    assert (
        caught.value.reason == "SparseElement input must have the literal module parent"
    )
    assert "literal module parent" in str(caught.value)

    with pytest.raises(SparseElementConstructionError):
        right.element(value, graph=_zz_to_qq_graph())


class _IndeterminateEquality:
    """A comparison outcome which is deliberately not the built-in true value."""


class _UncertainElement:
    """A protocol element whose equality cannot prove zero."""

    def __init__(self, parent: _UncertainDomain, value: object) -> None:
        self._parent = parent
        self._value = value

    @property
    def parent(self) -> _UncertainDomain:
        return self._parent

    @property
    def value(self) -> object:
        return self._value

    def __eq__(self, other: object) -> object:  # type: ignore[override]
        return _IndeterminateEquality()


class _UncertainDomain:
    """An exact fixture domain that cannot establish equality to its zero."""

    def normalize(self, value: object) -> object:
        return value

    def element(self, value: object) -> _UncertainElement:
        if type(value) is _UncertainElement and value.parent is self:
            return value
        return _UncertainElement(self, value)


class _RecordingDomain(_UncertainDomain):
    """A domain fixture exposing every attempted element construction."""

    def __init__(self) -> None:
        self.calls: list[object] = []

    def element(self, value: object) -> _UncertainElement:
        self.calls.append(value)
        return super().element(value)


def test_sparse_retains_coefficients_when_zero_equality_is_uncertain() -> None:
    """Only an exact built-in True equality result authorizes zero removal."""
    domain = _UncertainDomain()
    module = _module(domain, 1)
    value = module.element({0: 0})

    assert tuple(value.coordinates()) == (0,)
    assert value.coordinates()[0].value == 0


def test_sparse_does_not_truth_coerce_uncertain_coefficient_equality() -> None:
    """Truthy non-bool comparisons do not silently establish element equality."""
    domain = _UncertainDomain()
    module = _module(domain, 1)
    coefficient = domain.element("x")
    same_identity = module.element({0: coefficient})
    same_identity_again = module.element({0: coefficient})
    distinct_uncertain = module.element({0: domain.element("x")})

    assert same_identity == same_identity_again
    assert same_identity != distinct_uncertain


def test_sparse_equality_requires_literal_parent_and_is_unhashable() -> None:
    """Canonical coordinates do not merge distinct module parent instances."""
    left = _module(ZZ(), 1)
    right = _module(ZZ(), 1)
    x = left.element({0: 3})
    same = left.element({0: 3})
    foreign = right.element({0: 3})

    assert x == same
    assert x != foreign
    assert x != object()
    with pytest.raises(TypeError, match="unhashable"):
        hash(x)


def test_sparse_arithmetic_slice_does_not_define_algebra_multiplication() -> None:
    """Module arithmetic does not guess a later algebra product or transport."""
    value = _module(ZZ(), 1).element({0: 1})

    for attribute in ("multiply", "product", "transport", "change_parent"):
        assert not hasattr(value, attribute)
    with pytest.raises(TypeError):
        _ = value * value
