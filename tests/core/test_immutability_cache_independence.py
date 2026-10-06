"""Regression coverage for v0.0 frozen-value and nonsemantic-cache boundaries."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from anyalgebra.core.coercions import CoercionGraph, CoercionMap
from anyalgebra.core.domains import DomainElement, QQ, ZZ
from anyalgebra.core.elements import SparseElementMap
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.core.parents import (
    FiniteCarrier,
    FrozenBuilderError,
    ParentBuildIssue,
    ParentBuilder,
    SemanticHash,
    Sort,
)
from anyalgebra.core.rational import Rational


def _zz_to_qq(value: DomainElement[int]) -> DomainElement[Rational]:
    """Embed an exact integer in the canonical rational parent."""
    return QQ().element(value.value)


def _module(*, name: str | None = None) -> FreeModule:
    """Build one small parent with a stable, ordinary basis."""
    return FreeModule(ZZ(), Basis(("e0", "e1"), coefficient_domain=ZZ()), name=name)


def _coercion_fixture() -> tuple[CoercionMap[int, Rational], CoercionGraph]:
    """Return one immutable edge and the persistent graph that contains it."""
    edge = CoercionMap(
        id="immutability.zz_to_qq.v1",
        source=ZZ(),
        target=QQ(),
        forward=_zz_to_qq,
        injective=True,
        exact=True,
        lossless=True,
    )
    return edge, CoercionGraph().with_map(edge)


def test_current_frozen_semantic_values_reject_mutation_of_defining_data() -> None:
    """Every implemented frozen value rejects ordinary semantic-data mutation."""
    module = _module(name="V")
    edge, graph = _coercion_fixture()
    plan = graph.plan(ZZ(), QQ())
    sparse_map = SparseElementMap(module, module, lambda value: value)
    frozen_fields = (
        (SemanticHash("sha256", "0" * 64), "digest", "f" * 64),
        (ParentBuildIssue("missing", "domain", "message"), "code", "changed"),
        (Sort("scalar"), "name", "changed"),
        (FiniteCarrier(("x",), sort=Sort("carrier")), "items", ("y",)),
        (module.basis, "labels", ("changed",)),
        (module, "name", "changed"),
        (ZZ().element(3), "value", 4),
        (QQ().element((3, 4)), "value", Rational(4, 5)),
        (Rational(3, 4), "numerator", 5),
        (edge, "cost", 8),
        (graph, "_topology", ()),
        (plan, "target", ZZ()),
        (module.element({0: 3}), "parent", _module(name="other")),
        (sparse_map, "source", _module(name="other")),
    )

    for value, field, replacement in frozen_fields:
        with pytest.raises(FrozenInstanceError):
            setattr(value, field, replacement)


def test_canonical_domain_parents_cannot_accumulate_runtime_state() -> None:
    """Canonical scalar parents have no writable instance dictionary or cache slot."""
    for domain in (ZZ(), QQ()):
        cache_attribute = "runtime_cache"
        with pytest.raises(AttributeError):
            setattr(domain, cache_attribute, object())
        assert not hasattr(domain, cache_attribute)


def test_builder_is_mutable_only_before_freeze_and_frozen_output_is_independent() -> (
    None
):
    """A builder may change its draft, while its successful output remains fixed."""
    basis = Basis(("e0",), coefficient_domain=ZZ())
    builder = ParentBuilder().with_domain(ZZ()).with_basis(basis).with_name("draft")
    assert builder.with_name("final") is builder

    frozen = builder.freeze()
    before = frozen.fingerprint()

    with pytest.raises(FrozenBuilderError):
        builder.with_name("changed after freeze")
    with pytest.raises(FrozenBuilderError):
        builder.freeze()

    assert frozen.name == "final"
    assert frozen.basis is basis
    assert frozen.fingerprint() == before
    assert frozen.fingerprint() is not before


def test_fresh_fingerprints_and_coordinate_records_are_equal_and_address_free() -> None:
    """Observable semantic records are recomputable, equal, and cache-independent."""
    module = _module(name="display-only")
    reconstructed = _module(name="different-display-name")
    value = module.element({1: -2, 0: 3})
    equal_value = module.element({0: 3, 1: -2})
    fingerprint_before = module.fingerprint()
    coordinates_before = value.coordinates()
    coordinate_record_before = tuple(coordinates_before.items())

    def semantic_snapshot() -> tuple[object, ...]:
        """Capture only equality, hash, fingerprint, and coordinate semantics."""
        current_fingerprint = module.fingerprint()
        return (
            module == reconstructed,
            value == equal_value,
            current_fingerprint,
            current_fingerprint == reconstructed.fingerprint(),
            hash(current_fingerprint),
            tuple(value.coordinates().items()),
        )

    baseline = semantic_snapshot()
    external_cache: dict[SemanticHash, object] = {}
    external_cache[fingerprint_before] = {"phase": "populated"}
    after_population = semantic_snapshot()
    external_cache[fingerprint_before] = ["mutated", "cache", "value"]
    after_mutation = semantic_snapshot()
    external_cache.clear()
    after_clear = semantic_snapshot()
    external_cache[module.fingerprint()] = ("replacement", object())
    after_replacement = semantic_snapshot()

    with pytest.raises(FrozenInstanceError):
        module.name = "altered"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        value.parent = _module(name="foreign")  # type: ignore[misc]
    with pytest.raises(TypeError):
        coordinates_before[0] = ZZ().element(99)  # type: ignore[index]

    fingerprint_after = module.fingerprint()
    coordinates_after = value.coordinates()

    assert fingerprint_after == fingerprint_before
    assert fingerprint_after is not fingerprint_before
    assert len(external_cache) == 1
    assert (
        after_population
        == after_mutation
        == after_clear
        == after_replacement
        == baseline
    )
    assert coordinates_after is not coordinates_before
    assert tuple(coordinates_after.items()) == coordinate_record_before
    assert tuple(coordinates_after.items()) == (
        (0, ZZ().element(3)),
        (1, ZZ().element(-2)),
    )
    for representation in (repr(module), repr(value), repr(fingerprint_after)):
        assert "0x" not in representation


def test_hash_boundaries_are_stable_for_hashable_values_and_explicit_elsewhere() -> (
    None
):
    """Hashable records stay equal; broad parent classes stay unhashable."""
    module = _module()
    edge, graph = _coercion_fixture()
    plan = graph.plan(ZZ(), QQ())
    hashable_values = (
        SemanticHash("sha256", "1" * 64),
        ParentBuildIssue("missing", "domain", "message"),
        Sort("scalar"),
        ZZ().element(3),
        QQ().element((3, 4)),
        Rational(3, 4),
        edge,
        graph,
        plan,
    )
    before = tuple(hash(value) for value in hashable_values)

    with pytest.raises(FrozenInstanceError):
        edge.cost = 5  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        graph._topology = ()  # type: ignore[misc]

    assert tuple(hash(value) for value in hashable_values) == before
    for deliberately_unhashable in (
        FiniteCarrier(("x",), sort=Sort("carrier")),
        module.basis,
        module,
        module.element({0: 1}),
        SparseElementMap(module, module, lambda value: value),
    ):
        with pytest.raises(TypeError, match="unhashable"):
            hash(deliberately_unhashable)
