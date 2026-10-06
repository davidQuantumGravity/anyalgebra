"""Corrected typed inventory for AlgMul's missing ``MakeProperty`` family."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import cast

import pytest

from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.legacy import property_mapping
from anyalgebra.legacy.models import Disposition
from anyalgebra.legacy.property_mapping import (
    PropertyMappingError,
    intended_property_members,
    makeproperty_law,
    property_factory_record,
    property_law_factory,
    property_laws,
    property_surfaces,
    property_validation_policy,
)
from anyalgebra.structures.laws import Law
from anyalgebra.structures.operations import Operation, PartialOperation
from anyalgebra.structures.outcomes import Indeterminate
from anyalgebra.structures.signatures import OperationSymbol, Signature
from anyalgebra.structures.structure import Structure, StructureBuilder
from anyalgebra.structures.terms import Term
from anyalgebra.validation.validate import Disproved, Inconclusive, Proved, validate_law


_FIXTURE = (
    Path(__file__).parents[1] / "fixtures" / "legacy" / "makeproperty-intent.json"
)
_REPO_ROOT = Path(__file__).parents[2]


def _fixture() -> dict[str, object]:
    """Load the immutable source-pinned intended-family manifest."""
    value = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def _members(payload: dict[str, object]) -> list[dict[str, object]]:
    """Return only the fixture's checked, explicit generated-name records."""
    value = payload["members"]
    assert type(value) is list and all(type(item) is dict for item in value)
    return cast(list[dict[str, object]], value)


def _receipt_path(payload: dict[str, object]) -> Path:
    """Resolve only the fixture-declared repository-relative receipt path."""
    binding = payload["receipt"]
    assert type(binding) is dict
    declared = binding["path"]
    assert type(declared) is str
    relative = Path(declared)
    assert not relative.is_absolute()
    resolved = (_REPO_ROOT / relative).resolve()
    assert resolved.is_relative_to(_REPO_ROOT.resolve())
    return resolved


def _receipt(payload: dict[str, object]) -> dict[str, object]:
    """Load the separately captured, pinned evaluated-surface receipt."""
    value = json.loads(_receipt_path(payload).read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def _records(value: object, *, field: str) -> list[dict[str, object]]:
    """Check a JSON list consists only of explicit object records."""
    assert type(value) is list and all(type(item) is dict for item in value), field
    return cast(list[dict[str, object]], value)


def _structure(
    signature: Signature,
    carrier: FiniteCarrier,
    operations: tuple[Operation | PartialOperation, ...],
) -> Structure:
    """Build one exact one-sorted finite model in declaration order."""
    builder = StructureBuilder(signature).with_carrier(carrier.sort, carrier)
    for symbol, operation in zip(signature.operations, operations, strict=True):
        builder.with_operation(symbol, operation)
    return builder.freeze()


def _binary(
    symbol: OperationSymbol, carrier: FiniteCarrier, function: object
) -> Operation:
    """Make a total finite binary operation from one hand-checkable table rule."""
    assert callable(function)
    return Operation.from_table(
        symbol,
        ((carrier.sort, carrier),),
        tuple(
            ((left, right), function(left, right))
            for left in carrier
            for right in carrier
        ),
    )


def _exact_term_shape(
    term: Term, operations: tuple[tuple[str, OperationSymbol], ...]
) -> tuple[object, ...]:
    """Expose one complete tree while requiring the literal operation instances."""

    if term.variable_value is not None:
        assert term.symbol is None
        assert term.arguments == ()
        return ("variable", term.variable_value.name)
    assert term.symbol is not None
    label = next(
        (name for name, symbol in operations if term.symbol is symbol),
        None,
    )
    assert label is not None
    return (
        label,
        tuple(_exact_term_shape(argument, operations) for argument in term.arguments),
    )


def test_makeproperty_fixture_enumerates_the_complete_intended_surface() -> None:
    payload = _fixture()
    members = _members(payload)
    assert len(members) == 128
    assert tuple(item["name"] for item in members) == tuple(
        member.name for member in intended_property_members()
    )


def test_fixture_is_canonical_source_pinned_and_matches_typed_mapping() -> None:
    payload = _fixture()
    source = payload["source"]
    assert type(source) is dict
    assert source == {
        "path": "${ANYALGEBRA_ALGMUL_SOURCE}",
        "sha256": "3A13C9F47092B5B9814AA08873FF4F1DE61D4635DF68FC7F3A039A7DA5B0D00D",
        "anchors": source["anchors"],
    }
    assert _FIXTURE.read_bytes() == (
        json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    )
    actual = tuple(
        {
            "name": member.name,
            "lawName": member.law_name,
            "arity": member.arity,
            "kernel": member.kernel,
            "typedFactory": member.typed_factory,
            "mode": member.mode,
            "lift": member.lift,
            "result": member.result,
            "disposition": member.disposition.value,
            "replacementRoute": member.replacement_route,
            "resultRoute": member.result_route,
            "oracle": member.oracle,
            "legacyStatus": member.legacy_status,
            "caveat": member.caveat,
        }
        for member in intended_property_members()
    )
    assert tuple(_members(payload)) == actual


def test_fixture_laws_surfaces_policy_and_factory_are_exact_typed_records() -> None:
    payload = _fixture()
    assert payload["laws"] == [
        {
            "name": law.name,
            "arity": law.arity,
            "kernel": law.kernel,
            "typedFactory": law.typed_factory,
            "disposition": law.disposition.value,
            "replacementRoute": law.replacement_route,
            "oracle": law.oracle,
            "caveat": law.caveat,
        }
        for law in property_laws()
    ]
    assert payload["surfaces"] == [
        {
            "prefix": surface.prefix,
            "mode": surface.mode,
            "lift": surface.lift,
            "result": surface.result,
            "disposition": surface.disposition.value,
            "resultRoute": surface.result_route,
            "oracle": surface.oracle,
        }
        for surface in property_surfaces()
    ]
    factory = property_factory_record()
    assert factory.disposition is Disposition.REPLACE
    assert payload["factory"] == {
        "name": factory.name,
        "legacyStatus": factory.legacy_status,
        "disposition": factory.disposition.value,
        "replacementRoute": factory.replacement_route,
        "oracle": factory.oracle,
        "note": factory.note,
    }
    policy = property_validation_policy()
    assert payload["validationPolicy"] == {
        "primaryApi": policy.primary_api,
        "outcomes": list(policy.outcomes),
        "sampledPass": policy.sampled_pass,
        "counterexampleOrder": policy.counterexample_order,
        "basisReduction": policy.basis_reduction,
        "fingerprints": policy.fingerprints,
    }


def test_metadata_factory_identifiers_resolve_and_makeproperty_uses_dispatch() -> None:
    for law in property_laws():
        factory = property_law_factory(law.name)
        assert factory.__name__ == law.typed_factory
        assert getattr(property_mapping, law.typed_factory) is factory

    bit = Sort("bit")
    product = OperationSymbol("product", (bit, bit), bit)
    result = makeproperty_law("Flexible", product)
    assert result.name == "legacy:Flexible"
    assert type(property_mapping._LAW_FACTORY_REGISTRY).__name__ == "mappingproxy"


def test_factory_resolver_rejects_same_name_callable_impostor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def flexible_law(*_args: object, **_kwargs: object) -> Law:
        raise AssertionError("resolver must reject before dispatch")

    monkeypatch.setattr(property_mapping, "flexible_law", flexible_law)
    with pytest.raises(PropertyMappingError) as raised:
        property_law_factory("Flexible")
    assert raised.value.field == "typed_factory"


def test_pinned_runtime_receipt_confirms_missing_factory_and_surface() -> None:
    payload = _fixture()
    source = payload["source"]
    assert type(source) is dict
    receipt_binding = payload["receipt"]
    assert type(receipt_binding) is dict
    receipt_path = _receipt_path(payload)
    receipt = _receipt(payload)
    assert (
        hashlib.sha256(receipt_path.read_bytes()).hexdigest().upper()
        == receipt_binding["sha256"]
    )
    for field in (
        "schemaType",
        "schemaVersion",
        "kernelVersion",
        "loadStatus",
        "expectedPropertySymbolCount",
        "propertySymbolsPresentCount",
        "propertySymbolsDefinedCount",
    ):
        assert receipt[field] == receipt_binding[field]
    assert receipt["sourceSHA256"] == source["sha256"]
    assert receipt["missingRequiredSymbols"] == [
        receipt_binding["missingRequiredSymbol"]
    ]
    assert receipt["propertyFactoryExerciseNames"] == []
    assert receipt["propertyFactoryExerciseRecords"] == []
    symbols = _records(receipt["propertySymbols"], field="propertySymbols")
    assert len(symbols) == receipt_binding["expectedPropertySymbolCount"] == 128
    assert all(
        item["present"] is False and item["defined"] is False for item in symbols
    )


def test_fixture_laws_and_prefix_order_reconcile_receipt_and_mapping() -> None:
    payload = _fixture()
    calls = [
        (item["name"], str(item["arity"]), item["kernel"])
        for item in cast(list[dict[str, object]], payload["laws"])
    ]
    assert calls == [
        ("Commutative", "2", "Comm"),
        ("Associative", "3", "Assoc"),
        ("Alternative", "2", "Alternative"),
        ("Flexible", "2", "Flexible"),
        ("PowerAssociative1", "1", "PowerAssociative1"),
        ("PowerAssociative2", "1", "PowerAssociative2"),
        ("JIdentity", "2", "JIdentity"),
        ("Jacobi", "3", "JacobiIdentity"),
    ]
    prefixes = tuple(
        item["prefix"] for item in cast(list[dict[str, object]], payload["surfaces"])
    )
    assert prefixes == (
        "A",
        "AT",
        "AM",
        "AMT",
        "IsA",
        "IsAT",
        "IsAM",
        "IsAMT",
        "N",
        "NT",
        "NM",
        "NMT",
        "IsN",
        "IsNT",
        "IsNM",
        "IsNMT",
    )
    receipt_names = tuple(
        item["name"]
        for item in _records(
            _receipt(payload)["propertySymbols"], field="propertySymbols"
        )
    )
    independent_names = tuple(
        prefix + cast(str, law) for prefix in prefixes for law, _, _ in calls
    )
    assert receipt_names == independent_names
    assert set(item["name"] for item in _members(payload)) == set(independent_names)
    assert {member.name for member in intended_property_members()} == set(
        independent_names
    )


def test_source_anchors_bind_every_typed_kernel_definition() -> None:
    payload = _fixture()
    source = payload["source"]
    assert type(source) is dict
    anchors = source["anchors"]
    assert type(anchors) is list
    assert [anchor["id"] for anchor in anchors] == [
        "comm_assoc",
        "jidentity",
        "property_kernels",
        "makeproperty_factory",
        "factory_calls",
        "jacobi",
    ]
    for anchor in anchors:
        assert type(anchor) is dict
        first, last = (int(part) for part in anchor["lines"].split("-"))
        assert 1 <= first <= last <= 3981
        assert anchor["statements"]


def test_literal_prefix_order_is_shared_by_all_three_factory_arity_branches() -> None:
    payload = _fixture()
    expected = tuple(surface.prefix for surface in property_surfaces())
    fixture_prefixes = tuple(
        item["prefix"] for item in cast(list[dict[str, object]], payload["surfaces"])
    )
    assert fixture_prefixes == expected
    for arity in (1, 2, 3):
        law_names = {
            item["name"]
            for item in cast(list[dict[str, object]], payload["laws"])
            if item["arity"] == arity
        }
        generated = {
            item["name"] for item in _members(payload) if item["lawName"] in law_names
        }
        assert generated == {
            prefix + cast(str, law) for prefix in expected for law in law_names
        }


def test_missing_factory_and_every_intended_name_have_disposition_and_oracle() -> None:
    payload = _fixture()
    factory = payload["factory"]
    assert type(factory) is dict
    assert factory["disposition"] == Disposition.REPLACE.value
    assert factory["replacementRoute"] == "typed_property_registry"
    members = _members(payload)
    assert all(item["disposition"] and item["oracle"] for item in members)
    assert len({item["name"] for item in members}) == 128
    assert [law.name for law in property_laws()] == [
        "Commutative",
        "Associative",
        "Alternative",
        "Flexible",
        "PowerAssociative1",
        "PowerAssociative2",
        "JIdentity",
        "Jacobi",
    ]
    assert [surface.prefix for surface in property_surfaces()] == [
        "A",
        "AT",
        "AM",
        "AMT",
        "IsA",
        "IsAT",
        "IsAM",
        "IsAMT",
        "N",
        "NT",
        "NM",
        "NMT",
        "IsN",
        "IsNT",
        "IsNM",
        "IsNMT",
    ]


def test_known_legacy_generator_defects_are_localized_not_preserved() -> None:
    defects = {
        member.name: member.caveat
        for member in intended_property_members()
        if member.caveat.startswith("legacy_defect:")
    }
    amt = "legacy_defect: AMT arity-2 branch passes raw fooFoo, not MT-prefixed kernel"
    nm = "legacy_defect: NM arity-3 branch omits a comma after ToExpression[M<>fooFoo]"
    assert defects == {
        "AMTCommutative": amt,
        "AMTAlternative": amt,
        "AMTFlexible": amt,
        "AMTJIdentity": amt,
        "NMAssociative": nm,
        "NMJacobi": nm,
    }


def test_typed_laws_validate_against_a_hand_checkable_finite_model() -> None:
    bit = Sort("bit")
    product = OperationSymbol("and", (bit, bit), bit)
    bracket = OperationSymbol("zero_bracket", (bit, bit), bit)
    addition = OperationSymbol("xor", (bit, bit), bit)
    zero = OperationSymbol("zero", (), bit)
    carrier = FiniteCarrier((0, 1), sort=bit)
    structure = _structure(
        Signature((bit,), (product, bracket, addition, zero)),
        carrier,
        (
            _binary(product, carrier, lambda left, right: left & right),
            _binary(bracket, carrier, lambda _left, _right: 0),
            _binary(addition, carrier, lambda left, right: left ^ right),
            Operation.from_table(zero, ((bit, carrier),), (((), 0),)),
        ),
    )
    for mapping in property_laws():
        if mapping.name == "Jacobi":
            law = makeproperty_law(mapping.name, bracket, addition=addition, zero=zero)
        else:
            law = makeproperty_law(mapping.name, product)
        report = validate_law(structure, law)
        assert type(report) is Proved
        assert report.expected_assignments == len(carrier) ** mapping.arity
        assert report.enumeration_policy == "finite_substitution_domain_lexicographic"


def test_literal_legacy_power_jordan_and_jacobi_parenthesizations_are_typed() -> None:
    carrier_sort = Sort("carrier")
    product = OperationSymbol("product", (carrier_sort, carrier_sort), carrier_sort)
    addition = OperationSymbol("addition", (carrier_sort, carrier_sort), carrier_sort)
    zero = OperationSymbol("zero", (), carrier_sort)
    x = ("variable", "x")
    y = ("variable", "y")
    z = ("variable", "z")
    square = ("product", (x, x))

    power_one = makeproperty_law("PowerAssociative1", product)
    assert tuple(variable.name for variable in power_one.variables) == ("x",)
    assert _exact_term_shape(power_one.conclusion.left, (("product", product),)) == (
        "product",
        (square, x),
    )
    assert _exact_term_shape(power_one.conclusion.right, (("product", product),)) == (
        "product",
        (x, square),
    )

    power_two = makeproperty_law("PowerAssociative2", product)
    assert tuple(variable.name for variable in power_two.variables) == ("x",)
    assert _exact_term_shape(power_two.conclusion.left, (("product", product),)) == (
        "product",
        (("product", (square, x)), x),
    )
    assert _exact_term_shape(power_two.conclusion.right, (("product", product),)) == (
        "product",
        (square, square),
    )

    jordan = makeproperty_law("JIdentity", product)
    assert tuple(variable.name for variable in jordan.variables) == ("x", "y")
    assert _exact_term_shape(jordan.conclusion.left, (("product", product),)) == (
        "product",
        (("product", (x, y)), square),
    )
    assert _exact_term_shape(jordan.conclusion.right, (("product", product),)) == (
        "product",
        (x, ("product", (y, square))),
    )

    jacobi = makeproperty_law("Jacobi", product, addition=addition, zero=zero)
    assert tuple(variable.name for variable in jacobi.variables) == ("x", "y", "z")
    operations = (
        ("bracket", product),
        ("addition", addition),
        ("zero", zero),
    )
    first = ("bracket", (x, ("bracket", (y, z))))
    second = ("bracket", (z, ("bracket", (x, y))))
    third = ("bracket", (y, ("bracket", (z, x))))
    assert _exact_term_shape(jacobi.conclusion.left, operations) == (
        "addition",
        (("addition", (first, second)), third),
    )
    assert _exact_term_shape(jacobi.conclusion.right, operations) == ("zero", ())


def test_validation_keeps_deterministic_disproof_and_inconclusive_boundaries() -> None:
    residue = Sort("residue")
    subtract = OperationSymbol("subtract", (residue, residue), residue)
    carrier = FiniteCarrier((0, 1, 2), sort=residue)
    structure = _structure(
        Signature((residue,), (subtract,)),
        carrier,
        (_binary(subtract, carrier, lambda left, right: (left - right) % 3),),
    )
    disproof = validate_law(structure, makeproperty_law("Associative", subtract))
    assert type(disproof) is Disproved
    assert disproof.witness is not None
    assert disproof.witness.substitution_indices == (0, 0, 1)

    probe = OperationSymbol("probe", (residue, residue), residue)
    incomplete = _structure(
        Signature((residue,), (probe,)),
        carrier,
        (
            PartialOperation.from_callable(
                probe,
                ((residue, carrier),),
                lambda _left, _right: Indeterminate("declared finite bound"),
            ),
        ),
    )
    report = validate_law(incomplete, makeproperty_law("Commutative", probe))
    assert type(report) is Inconclusive
    assert report.witness is not None
    assert report.witness.substitution_indices == (0, 0)


def test_factory_requires_explicit_typed_operations_and_jacobi_additive_data() -> None:
    bit = Sort("bit")
    foreign = Sort("foreign")
    product = OperationSymbol("product", (bit, bit), bit)
    addition = OperationSymbol("addition", (bit, bit), bit)
    zero = OperationSymbol("zero", (), bit)
    assert makeproperty_law("Jacobi", product, addition=addition, zero=zero).name == (
        "legacy:Jacobi"
    )
    with pytest.raises(PropertyMappingError, match="intended eight-law"):
        makeproperty_law("NotALaw", product)
    with pytest.raises(PropertyMappingError, match="built-in str"):
        makeproperty_law(cast(str, 7), product)
    with pytest.raises(PropertyMappingError, match="exact OperationSymbol"):
        makeproperty_law("Flexible", cast(OperationSymbol, object()))
    with pytest.raises(PropertyMappingError, match="requires explicit addition"):
        makeproperty_law("Jacobi", product)
    with pytest.raises(PropertyMappingError, match="only accepted for Jacobi"):
        makeproperty_law("Flexible", product, addition=addition, zero=zero)
    with pytest.raises(PropertyMappingError, match="closed binary"):
        makeproperty_law("Flexible", zero)
    with pytest.raises(PropertyMappingError, match="partial_semantics"):
        makeproperty_law("Flexible", product, partial_semantics="sampled")
    with pytest.raises(PropertyMappingError, match="exact OperationSymbol"):
        makeproperty_law(
            "Jacobi", product, addition=addition, zero=cast(OperationSymbol, object())
        )
    with pytest.raises(PropertyMappingError, match="nullary OperationSymbol"):
        makeproperty_law("Jacobi", product, addition=addition, zero=product)
    with pytest.raises(PropertyMappingError, match="nullary OperationSymbol"):
        makeproperty_law(
            "Jacobi",
            product,
            addition=addition,
            zero=OperationSymbol("foreign_zero", (), foreign),
        )
    with pytest.raises(PropertyMappingError, match="bracket sort"):
        makeproperty_law(
            "Jacobi",
            product,
            addition=OperationSymbol("foreign_addition", (foreign, foreign), foreign),
            zero=zero,
        )


def test_custom_law_routes_preserve_declared_partial_semantics() -> None:
    carrier_sort = Sort("carrier")
    product = OperationSymbol("product", (carrier_sort, carrier_sort), carrier_sort)
    strong = makeproperty_law("PowerAssociative1", product)
    conditional = makeproperty_law(
        "PowerAssociative1", product, partial_semantics="definedness_conditional"
    )
    assert strong.partial_semantics == "strong"
    assert conditional.partial_semantics == "definedness_conditional"
    assert conditional.conclusion == strong.conclusion
    assert conditional.variables == strong.variables
