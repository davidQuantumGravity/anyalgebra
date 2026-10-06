"""Contract tests for finite many-sorted substitution enumeration."""

from __future__ import annotations

from collections.abc import ItemsView, Iterator, Mapping
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.structures.evaluate import evaluate_term
from anyalgebra.structures.outcomes import Defined
from anyalgebra.structures.signatures import Signature
from anyalgebra.structures.structure import StructureBuilder
from anyalgebra.structures.terms import Term, Variable
from anyalgebra.validation.domains import (
    FiniteSubstitutionDomain,
    Substitution,
    SubstitutionDomainError,
)


def test_many_sorted_order_mapping_and_evaluator_integration() -> None:
    point = Sort("point")
    color = Sort("color")
    left = Variable("left", point)
    shade = Variable("shade", color)
    right = Variable("right", point)
    points = FiniteCarrier(("L", "R"), sort=point, labels=("a", "b"))
    colors = FiniteCarrier(("red", "blue"), sort=color, labels=("r", "b"))
    domain = FiniteSubstitutionDomain.from_bindings(
        (left, shade, right), ((point, points), (color, colors))
    )
    assignments = tuple(domain)
    assert domain.assignment_count == len(domain) == 8
    assert [assignment.member_values for assignment in assignments] == [
        ("L", "red", "L"),
        ("L", "red", "R"),
        ("L", "blue", "L"),
        ("L", "blue", "R"),
        ("R", "red", "L"),
        ("R", "red", "R"),
        ("R", "blue", "L"),
        ("R", "blue", "R"),
    ]
    first = assignments[0]
    assert tuple(first) == (left, shade, right)
    assert tuple(first.items()) == ((left, "L"), (shade, "red"), (right, "L"))
    assert first[left] == "L" and Variable("left", point) in first
    assert first.carrier_indices == (0, 0, 0)
    structure = (
        StructureBuilder(Signature((point, color), (), ()))
        .with_carrier(point, points)
        .with_carrier(color, colors)
        .freeze()
    )
    evaluated = evaluate_term(structure, Term.variable(right), assignments[-1])
    assert type(evaluated.outcome) is Defined and evaluated.outcome.value == "R"


def test_zero_singleton_repeated_sort_indexing_and_repeatability() -> None:
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    empty = FiniteSubstitutionDomain.from_bindings((), ())
    assert empty.assignment_count == 1 and tuple(
        tuple(item.items()) for item in empty
    ) == ((),)
    first = Variable("first", bit)
    second = Variable("second", bit)
    domain = FiniteSubstitutionDomain.from_bindings((first, second), ((bit, carrier),))
    assert [assignment.carrier_indices for assignment in domain] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert [assignment.member_values for assignment in domain] == [
        assignment.member_values for assignment in domain
    ]
    assert domain[2].member_values == (1, 0)
    assert domain[-1].member_values == (1, 1)
    with pytest.raises(IndexError):
        domain[4]


def test_unhashable_values_structure_factory_and_sealed_records() -> None:
    item = ["mutable"]
    sort = Sort("value")
    variable = Variable("x", sort)
    carrier = FiniteCarrier((item,), sort=sort)
    structure = (
        StructureBuilder(Signature((sort,), (), ()))
        .with_carrier(sort, carrier)
        .freeze()
    )
    domain = FiniteSubstitutionDomain.from_structure((variable,), structure)
    assignment = domain[0]
    assert assignment[Variable("x", sort)] is item
    assert "mutable" not in repr(assignment) and "0x" not in repr(assignment)
    assert not hasattr(domain, "__dict__") and not hasattr(assignment, "__dict__")
    with pytest.raises(FrozenInstanceError):
        assignment.member_values = ()  # type: ignore[misc]
    with pytest.raises(TypeError):
        hash(assignment)
    with pytest.raises(SubstitutionDomainError):
        Substitution()
    with pytest.raises(SubstitutionDomainError):
        FiniteSubstitutionDomain()


def test_bounds_and_malformed_inputs_are_preflighted_without_enumeration() -> None:
    bit = Sort("bit")
    carrier = FiniteCarrier((0, 1), sort=bit)
    variables = tuple(Variable(f"x{index}", bit) for index in range(17))
    with pytest.raises(SubstitutionDomainError) as overflow:
        FiniteSubstitutionDomain.from_bindings(variables, ((bit, carrier),))
    assert (overflow.value.kind, overflow.value.observed, overflow.value.maximum) == (
        "assignments",
        131072,
        65536,
    )
    maximum = tuple(Variable(f"y{index}", Sort(f"s{index}")) for index in range(512))
    singleton_bindings = tuple(
        (variable.sort, FiniteCarrier((index,), sort=variable.sort))
        for index, variable in enumerate(maximum)
    )
    accepted = FiniteSubstitutionDomain.from_bindings(maximum, singleton_bindings)
    assert accepted.assignment_count == 1 and accepted[0].carrier_indices == (0,) * 512
    binary_limit = tuple(Variable(f"b{index}", bit) for index in range(16))
    at_limit = FiniteSubstitutionDomain.from_bindings(binary_limit, ((bit, carrier),))
    assert at_limit.assignment_count == 65_536
    assert next(iter(at_limit)).carrier_indices == (0,) * 16
    with pytest.raises(SubstitutionDomainError):
        FiniteSubstitutionDomain.from_bindings(
            (variables[0], variables[0]), ((bit, carrier),)
        )
    with pytest.raises(SubstitutionDomainError):
        FiniteSubstitutionDomain.from_bindings((variables[0],), ())
    with pytest.raises(SubstitutionDomainError):
        FiniteSubstitutionDomain.from_bindings((), ((bit, carrier),))


def test_binding_validation_and_one_pass_snapshot_boundaries() -> None:
    bit = Sort("bit")
    other = Sort("other")
    variable = Variable("x", bit)
    carrier = FiniteCarrier((0,), sort=bit)
    foreign = FiniteCarrier(("x",), sort=other)

    def once(values: tuple[object, ...]) -> Iterator[object]:
        yield from values

    assert (
        FiniteSubstitutionDomain.from_bindings(
            once((variable,)), once(((bit, carrier),))
        )[0][variable]
        == 0
    )
    malformed = (
        ("bad variables", ((bit, carrier),)),
        ((object(),), ((bit, carrier),)),
        ((variable,), ("not-a-binding",)),
        ((variable,), ((object(), carrier),)),
        ((variable,), ((bit, object()),)),
        ((variable,), ((bit, foreign),)),
        ((variable,), ((bit, carrier), (bit, carrier))),
        ((variable,), ((bit, carrier), (other, foreign))),
    )
    for variables, bindings in malformed:
        with pytest.raises(SubstitutionDomainError):
            FiniteSubstitutionDomain.from_bindings(variables, bindings)
    too_many_variables = tuple(Variable(f"v{index}", bit) for index in range(513))
    with pytest.raises(SubstitutionDomainError) as variable_bound:
        FiniteSubstitutionDomain.from_bindings(too_many_variables, ((bit, carrier),))
    assert (variable_bound.value.kind, variable_bound.value.observed) == (
        "variables",
        513,
    )
    many_sorts = tuple(Sort(f"t{index}") for index in range(513))
    many_bindings = tuple(
        (sort, FiniteCarrier((index,), sort=sort))
        for index, sort in enumerate(many_sorts)
    )
    with pytest.raises(SubstitutionDomainError) as binding_bound:
        FiniteSubstitutionDomain.from_bindings((), many_bindings)
    assert (binding_bound.value.kind, binding_bound.value.observed) == (
        "carriers",
        513,
    )


class HostileIter:
    def __iter__(self) -> Iterator[object]:
        raise RuntimeError("secret iterator payload")


class HostileNext:
    def __iter__(self) -> HostileNext:
        return self

    def __next__(self) -> object:
        raise RuntimeError("secret next payload")


class HostileMapping(Mapping[object, object]):
    def __getitem__(self, key: object) -> object:
        raise KeyError(key)

    def __iter__(self) -> Iterator[object]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self) -> ItemsView[object, object]:
        raise RuntimeError("secret mapping payload")


def test_hostile_inputs_and_subclass_factories_are_rejected_safely() -> None:
    bit = Sort("bit")
    variable = Variable("x", bit)
    carrier = FiniteCarrier((0,), sort=bit)

    class DomainSubclass(FiniteSubstitutionDomain):
        pass

    class SubstitutionSubclass(Substitution):
        pass

    for variables, bindings in (
        (HostileIter(), ((bit, carrier),)),
        (HostileNext(), ((bit, carrier),)),
        ((variable,), HostileMapping()),
    ):
        with pytest.raises(SubstitutionDomainError) as caught:
            FiniteSubstitutionDomain.from_bindings(variables, bindings)
        assert "secret" not in str(caught.value)
    with pytest.raises(SubstitutionDomainError):
        DomainSubclass._create((), ())
    with pytest.raises(SubstitutionDomainError):
        DomainSubclass.from_bindings((), ())
    with pytest.raises(SubstitutionDomainError):
        DomainSubclass.from_structure((), object())
    with pytest.raises(SubstitutionDomainError):
        SubstitutionSubclass._create((), (), ())


def test_mapping_and_equality_boundaries_are_conservative_and_safe() -> None:
    sort = Sort("safe")
    variable = Variable("x", sort)

    class NoisyMember:
        def __repr__(self) -> str:
            return "secret-member-repr"

    member = NoisyMember()
    carrier = FiniteCarrier((member,), sort=sort)
    domain = FiniteSubstitutionDomain.from_bindings((variable,), {sort: carrier})
    first = domain[0]
    second = domain[0]
    assert first is not second and first != second
    assert list(first.keys()) == [variable]
    assert list(first.values()) == [member]
    assert list(first.items()) == [(variable, member)]
    assert object() not in first
    with pytest.raises(KeyError):
        _ = first[cast(Variable, object())]
    with pytest.raises(KeyError):
        _ = first[Variable("missing", sort)]
    with pytest.raises(TypeError):
        _ = domain[True]
    assert len(first) == 1
    assert "secret-member-repr" not in repr(first)
    assert "secret-member-repr" not in repr(domain)
    with pytest.raises(SubstitutionDomainError):
        FiniteSubstitutionDomain.from_structure((variable,), object())
