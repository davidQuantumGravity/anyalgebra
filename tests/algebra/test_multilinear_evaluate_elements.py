"""External contracts for exact generic multilinear element evaluation."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest

from anyalgebra.algebra.multilinear import (
    FiniteMultilinearStructure,
    MultilinearEvaluationError,
    MultilinearOperation,
    StructureConstants,
    evaluate_multilinear,
)
from anyalgebra.core.coercions import CoercionGraph, CoercionMap
from anyalgebra.core.domains import Domain, DomainElement, QQ, ZZ
from anyalgebra.core.elements import SparseElement
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.core.rational import Rational


def _module(
    rank: int = 2,
    *,
    domain: Domain[object] | None = None,
    labels: tuple[str, ...] | None = None,
) -> FreeModule:
    coefficient_domain = ZZ() if domain is None else domain
    basis_labels = (
        tuple(f"e{index}" for index in range(rank)) if labels is None else labels
    )
    return FreeModule(
        coefficient_domain,
        Basis(basis_labels, coefficient_domain=coefficient_domain),
    )


def _structure(
    module: FreeModule,
    arity: int,
    entries: tuple[tuple[tuple[int, ...], int], ...],
) -> FiniteMultilinearStructure:
    constants = StructureConstants.from_sparse(
        (module.basis for _ in range(arity)),
        module.basis,
        tuple((key, module.domain.element(value)) for key, value in entries),
    )
    return FiniteMultilinearStructure.from_structure_constants(module, constants)


def _zz_to_qq_graph() -> CoercionGraph:
    def forward(value: DomainElement[int]) -> DomainElement[Rational]:
        return QQ().element(value.value)

    return CoercionGraph().with_map(
        CoercionMap(
            id="test.v01.zz_to_qq",
            source=ZZ(),
            target=QQ(),
            forward=forward,
            injective=True,
            exact=True,
            lossless=True,
        )
    )


def test_nullary_unary_binary_and_ternary_evaluation_is_exact() -> None:
    module = _module()
    nullary = _structure(module, 0, (((1,), 7),))
    unary = _structure(module, 1, (((0, 1), 2), ((1, 0), -1)))
    binary = _structure(
        module,
        2,
        (
            ((0, 0, 0), 1),
            ((0, 1, 1), 1),
            ((1, 0, 1), -1),
            ((1, 1, 0), 2),
        ),
    )
    ternary = _structure(module, 3, (((0, 1, 0, 1), 3),))

    assert evaluate_multilinear(nullary) == module.element({1: 7})
    assert evaluate_multilinear(unary, module.element({0: 2, 1: 3})) == (
        module.element({0: -3, 1: 4})
    )
    assert evaluate_multilinear(
        binary, module.element({0: 2, 1: 3}), module.element({0: -1, 1: 4})
    ) == module.element({0: 22, 1: 11})
    assert evaluate_multilinear(
        ternary,
        module.element({0: 2}),
        module.element({1: 5}),
        module.element({0: -1}),
    ) == module.element({1: -30})


def test_zero_inputs_and_rank_zero_boundaries_return_declared_parent_zero() -> None:
    module = _module()
    binary = _structure(module, 2, (((0, 0, 1), 5),))
    rank_zero = _module(0)
    empty_nullary = _structure(rank_zero, 0, ())

    result = evaluate_multilinear(binary, module.zero(), module.element({0: 4}))

    assert result == module.zero()
    assert result.parent is module
    assert evaluate_multilinear(empty_nullary) == rank_zero.zero()


def test_selected_operation_must_be_the_structures_literal_operation() -> None:
    module = _module(1)
    structure = _structure(module, 1, (((0, 0), 2),))
    value = module.element({0: 3})

    assert evaluate_multilinear(
        structure, value, operation=structure.operation
    ) == module.element({0: 6})

    other = _structure(module, 1, (((0, 0), 3),))
    with pytest.raises(MultilinearEvaluationError) as caught:
        evaluate_multilinear(structure, value, operation=other.operation)
    assert caught.value.code == "operation"
    assert caught.value.argument_index is None


@pytest.mark.parametrize("provided", ((), (object(),), (object(), object(), object())))
def test_wrong_arity_and_non_elements_fail_before_expansion(
    provided: tuple[object, ...],
) -> None:
    module = _module(1)
    structure = _structure(module, 2, (((0, 0, 0), 1),))

    with pytest.raises(MultilinearEvaluationError) as caught:
        evaluate_multilinear(structure, *cast(tuple[SparseElement, ...], provided))

    expected_code = "arity" if len(provided) != 2 else "element"
    assert caught.value.code == expected_code
    assert caught.value.argument_index == (None if expected_code == "arity" else 0)


def test_literal_parent_mismatch_fails_without_coercion_authority() -> None:
    module = _module(1)
    foreign = _module(1)
    structure = _structure(module, 1, (((0, 0), 1),))

    with pytest.raises(MultilinearEvaluationError) as caught:
        evaluate_multilinear(structure, foreign.element({0: 2}))

    assert caught.value.code == "parent"
    assert caught.value.argument_index == 0
    assert caught.value.expected_parent is module
    assert caught.value.actual_parent is foreign


def test_explicit_lossless_scalar_extension_preserves_equal_basis_coordinates() -> None:
    source = _module(2, domain=ZZ(), labels=("one", "u"))
    target = _module(2, domain=QQ(), labels=("one", "u"))
    structure = _structure(
        target,
        2,
        (((0, 0, 0), 1), ((0, 1, 1), 1), ((1, 0, 1), 1)),
    )

    result = evaluate_multilinear(
        structure,
        source.element({0: 2, 1: 3}),
        source.element({0: -1, 1: 4}),
        graph=_zz_to_qq_graph(),
    )

    assert result.parent is target
    assert result == target.element({0: -2, 1: 5})


def test_graph_never_infers_a_basis_map_or_unavailable_coercion() -> None:
    source = _module(1, domain=ZZ(), labels=("foreign",))
    target = _module(1, domain=QQ(), labels=("declared",))
    structure = _structure(target, 1, (((0, 0), 1),))

    with pytest.raises(MultilinearEvaluationError) as wrong_basis:
        evaluate_multilinear(structure, source.element({0: 2}), graph=_zz_to_qq_graph())
    assert wrong_basis.value.code == "basis"
    assert wrong_basis.value.argument_index == 0

    equal_basis = _module(1, domain=ZZ(), labels=("declared",))
    with pytest.raises(MultilinearEvaluationError) as missing_route:
        evaluate_multilinear(
            structure, equal_basis.element({0: 2}), graph=CoercionGraph()
        )
    assert missing_route.value.code == "coercion"
    assert missing_route.value.argument_index == 0


def test_invalid_structure_operation_and_graph_have_typed_diagnostics() -> None:
    module = _module(1)
    structure = _structure(module, 1, (((0, 0), 1),))
    value = module.element({0: 1})

    cases: tuple[tuple[Callable[[], SparseElement], str], ...] = (
        (
            lambda: evaluate_multilinear(cast(FiniteMultilinearStructure, object())),
            "structure",
        ),
        (
            lambda: evaluate_multilinear(
                structure, value, operation=cast(MultilinearOperation, object())
            ),
            "operation",
        ),
        (
            lambda: evaluate_multilinear(
                structure, value, graph=cast(CoercionGraph, object())
            ),
            "graph",
        ),
    )
    for invoke, code in cases:
        with pytest.raises(MultilinearEvaluationError) as caught:
            invoke()
        assert caught.value.code == code
        assert caught.value.argument_index is None


def test_nonassociative_parenthesization_is_caller_explicit() -> None:
    module = _module(2, labels=("a", "b"))
    structure = _structure(
        module,
        2,
        (((0, 0, 1), 1), ((1, 0, 0), 1)),
    )
    a = module.element({0: 1})

    left_nested = evaluate_multilinear(
        structure, evaluate_multilinear(structure, a, a), a
    )
    right_nested = evaluate_multilinear(
        structure, a, evaluate_multilinear(structure, a, a)
    )

    assert left_nested == a
    assert right_nested == module.zero()
    assert left_nested != right_nested


def test_evaluation_does_not_mutate_sparse_inputs_or_structure_constants() -> None:
    module = _module(2)
    structure = _structure(module, 2, (((0, 1, 0), 3),))
    left = module.element({0: 2})
    right = module.element({1: 5})
    left_before = tuple(left.coordinates().items())
    right_before = tuple(right.coordinates().items())
    entries_before = structure.operation.constants.entries

    result = evaluate_multilinear(structure, left, right)

    assert result == module.element({0: 30})
    assert tuple(left.coordinates().items()) == left_before
    assert tuple(right.coordinates().items()) == right_before
    assert structure.operation.constants.entries == entries_before
    assert isinstance(result, SparseElement)
