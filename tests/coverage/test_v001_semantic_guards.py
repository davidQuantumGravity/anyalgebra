from __future__ import annotations

from typing import cast

import pytest

from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.persistence import semantic
from anyalgebra.structures.deductive import DeductiveSystem
from anyalgebra.structures.operations import PartialOperation
from anyalgebra.structures.signatures import OperationSymbol
from anyalgebra.structures.terms import Term
from anyalgebra.validation.validate import Disproved
from examples import v0_0_exit_demo


class _StringSubclass(str):
    pass


def test_semantic_sequence_and_cell_guards_reject_each_malformed_shape() -> None:
    invalid_strings: tuple[tuple[object, dict[str, object]], ...] = (
        (object(), {}),
        ([], {}),
        (["a", "b"], {"maximum": 1}),
        ([""], {}),
        (["a", "a"], {}),
    )
    for value, options in invalid_strings:
        with pytest.raises(ValueError):
            semantic._strings(value, "field", **options)  # type: ignore[arg-type]

    for value, basis in (
        (object(), ("e",)),
        ([], ("e",)),
        ([object()], ("e",)),
        ([[1, 0]], ("e",)),
        ([[True]], ("e",)),
    ):
        with pytest.raises(ValueError, match="cells"):
            semantic._cells(value, basis)

    with pytest.raises(ValueError, match="record"):
        semantic._mapping(object())


def test_partial_record_rejects_every_table_declaration_hazard() -> None:
    carrier = ["a"]
    invalid_tables: tuple[object, ...] = (
        object(),
        [("a", "a", "a")] * (semantic._MAX_TABLE + 1),
        [object()],
        [("a", "a")],
        [("missing", "a", "a")],
        [(_StringSubclass("a"), "a", "a")],
        [("a", "a", _StringSubclass("a"))],
        [("a", "a", "a"), ("a", "a", None)],
    )
    for table in invalid_tables:
        with pytest.raises(ValueError, match="table"):
            semantic.FinitePartialOperationRecord.create(carrier, table)


def test_deductive_record_rejects_every_rule_declaration_hazard() -> None:
    invalid_rules: tuple[object, ...] = (
        object(),
        [],
        [(("a",), "b")] * (semantic._MAX_RULES + 1),
        [object()],
        [(("a",), "b", "extra")],
        [(("a",), "")],
    )
    for rules in invalid_rules:
        with pytest.raises(ValueError, match="rules"):
            semantic.RuleGeneratedDeductiveRecord.create(rules)


def test_semantic_exact_type_and_associativity_guards() -> None:
    with pytest.raises(ValueError, match="algebra"):
        semantic.AlgebraPresentationsRecord.create(object())
    with pytest.raises(ValueError, match="algebra"):
        semantic.AlgebraPresentationBundle.from_algebra(object())
    with pytest.raises(ValueError, match="validation_report"):
        semantic.LawValidationReportRecord.create(
            ("e",), ("e",), ("e", "e", "e")
        ).rebuild_and_validate()


def test_semantic_value_encoders_reject_wrong_shape_before_serializing() -> None:
    with pytest.raises(ValueError, match="partial_operation"):
        semantic._partial_value_encoder(cast(PartialOperation, object()))
    with pytest.raises(ValueError, match="deductive_system"):
        semantic._deductive_value_encoder(DeductiveSystem(()))
    with pytest.raises(ValueError, match="validation_report"):
        semantic._report_value_encoder(cast(Disproved, object()))

    sort = Sort("element")
    carrier = FiniteCarrier(("a",), sort=sort)
    marker = object()
    unary = PartialOperation.from_table(
        OperationSymbol("u", (sort,), sort),
        ((sort, carrier),),
        [(("a",), marker)],
        undefined_marker=marker,
    )
    with pytest.raises(ValueError, match="partial_operation"):
        semantic._partial_value_encoder(unary)


def test_ground_term_encoder_rejects_variables_nonatoms_and_mixed_sorts() -> None:
    left = Sort("left")
    right = Sort("right")
    variable = Term._make(
        variable_value=None,
        symbol=None,
        arguments=(),
        sort=left,
    )
    with pytest.raises(ValueError, match="deductive_system"):
        semantic._term_name(variable, None)

    atom = Term.apply(OperationSymbol("a", (), left))
    nonatom = Term.apply(OperationSymbol("u", (left,), left), atom)
    with pytest.raises(ValueError, match="deductive_system"):
        semantic._term_name(nonatom, None)

    mismatched_output = Term._make(
        variable_value=None,
        symbol=OperationSymbol("b", (), right),
        arguments=(),
        sort=left,
    )
    with pytest.raises(ValueError, match="deductive_system"):
        semantic._term_name(mismatched_output, None)
    with pytest.raises(ValueError, match="deductive_system"):
        semantic._term_name(Term.apply(OperationSymbol("b", (), right)), left)


def test_semantic_encoders_detect_mutated_internal_invariants() -> None:
    algebra = v0_0_exit_demo._fixture()[2]
    coefficient = algebra.evaluate_basis(0, 0).coordinates()[0]
    object.__setattr__(coefficient, "value", True)
    with pytest.raises(ValueError, match="algebra"):
        semantic._table_algebra_from_value(algebra)

    malformed_structure = v0_0_exit_demo._fixture()[-1]
    object.__setattr__(malformed_structure, "structure", object())
    with pytest.raises(ValueError, match="validation_report"):
        semantic._report_value_encoder(malformed_structure)

    missing_witness = v0_0_exit_demo._fixture()[-1]
    object.__setattr__(missing_witness, "witness", None)
    with pytest.raises(ValueError, match="validation_report"):
        semantic._report_value_encoder(missing_witness)

    mismatched_replay = v0_0_exit_demo._fixture()[-1]
    object.__setattr__(
        mismatched_replay,
        "evaluated_assignments",
        mismatched_replay.evaluated_assignments + 1,
    )
    with pytest.raises(ValueError, match="validation_report"):
        semantic._report_value_encoder(mismatched_replay)
