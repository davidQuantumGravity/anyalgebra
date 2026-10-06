"""Allow-listed declarative semantic records used by the v0.0 exit demo.

This module intentionally serializes a bounded neutral subset: finite string
carriers, ground deduction rules, integer multiplication tables, and a finite
law-check declaration.  Records contain data only; decoding never imports a
name, resolves a callable, or evaluates input text.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import product
from typing import cast

from anyalgebra.algebra.multilinear import BasisProductTable, FiniteMultilinearStructure
from anyalgebra.core.domains import IntegerDomain, ZZ
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.core.parents import FiniteCarrier, Sort
from anyalgebra.linear.matrix import Matrix, MatrixSpace
from anyalgebra.persistence.registry import JSONValue, SerializerRegistry
from anyalgebra.structures.deductive import DeductiveSystem, InferenceRule
from anyalgebra.structures.operations import Operation, PartialOperation
from anyalgebra.structures.signatures import OperationSymbol, Signature
from anyalgebra.structures.structure import Structure, StructureBuilder
from anyalgebra.structures.terms import Term
from anyalgebra.validation.factories import associative_law
from anyalgebra.validation.validate import Disproved, ValidationReport, validate_law


_MAX_CARRIER = 32
_MAX_TABLE = 1024
_MAX_RULES = 128


def _strings(
    value: object, field: str, *, maximum: int = _MAX_CARRIER, unique: bool = True
) -> tuple[str, ...]:
    if type(value) not in (tuple, list):
        raise ValueError(field)
    items: tuple[object, ...] = tuple(cast(Iterable[object], value))
    if (
        not items
        or len(items) > maximum
        or any(type(item) is not str or not item for item in items)
    ):
        raise ValueError(field)
    if unique and len(set(items)) != len(items):
        raise ValueError(field)
    return cast(tuple[str, ...], items)


def _cells(value: object, basis: tuple[str, ...]) -> tuple[tuple[int, ...], ...]:
    if type(value) not in (tuple, list):
        raise ValueError("cells")
    cells: tuple[object, ...] = tuple(cast(Iterable[object], value))
    if len(cells) != len(basis) ** 2:
        raise ValueError("cells")
    normalized: list[tuple[int, ...]] = []
    for cell in cells:
        if type(cell) not in (tuple, list):
            raise ValueError("cells")
        vector: tuple[object, ...] = tuple(cast(Iterable[object], cell))
        if len(vector) != len(basis) or any(type(item) is not int for item in vector):
            raise ValueError("cells")
        normalized.append(cast(tuple[int, ...], vector))
    return tuple(normalized)


def _atom(name: str, sort: Sort) -> Term:
    return Term.apply(OperationSymbol(name, (), sort))


def _mapping(value: object) -> Mapping[str, object]:
    """Require the inert mapping snapshot passed by ``SerializerRegistry``."""
    if not isinstance(value, Mapping):
        raise ValueError("record")
    return cast(Mapping[str, object], value)


def _require_keys(record: object, expected: set[str]) -> Mapping[str, object]:
    """Reject missing and ignored fields before a parser sees semantic data."""
    mapping = _mapping(record)
    if set(mapping) != expected:
        raise ValueError("record_keys")
    return mapping


@dataclass(frozen=True, slots=True)
class FinitePartialOperationRecord:
    """A finite binary partial operation, including its declared undefined cells."""

    carrier: tuple[str, ...]
    table: tuple[tuple[str, str, str | None], ...]

    @classmethod
    def create(cls, carrier: object, table: object) -> FinitePartialOperationRecord:
        values = _strings(tuple(cast(Iterable[object], carrier)), "carrier")
        if type(table) not in (tuple, list):
            raise ValueError("table")
        table_items = cast(tuple[object, ...] | list[object], table)
        if len(table_items) > _MAX_TABLE:
            raise ValueError("table")
        entries: list[tuple[str, str, str | None]] = []
        seen: set[tuple[str, str]] = set()
        for entry in table_items:
            if type(entry) not in (tuple, list):
                raise ValueError("table")
            entry_values = cast(tuple[object, ...] | list[object], entry)
            if len(entry_values) != 3:
                raise ValueError("table")
            left, right, output = cast(tuple[object, object, object], entry_values)
            if (
                left not in values
                or right not in values
                or (output is not None and output not in values)
            ):
                raise ValueError("table")
            if type(left) is not str or type(right) is not str:
                raise ValueError("table")
            if output is not None and type(output) is not str:
                raise ValueError("table")
            key = (left, right)
            if key in seen:
                raise ValueError("table")
            seen.add(key)
            entries.append((left, right, output))
        return cls(values, tuple(entries))

    def rebuild(self) -> PartialOperation:
        sort = Sort("element")
        carrier = FiniteCarrier(self.carrier, sort=sort)
        marker = object()
        return PartialOperation.from_table(
            OperationSymbol("partial_product", (sort, sort), sort),
            ((sort, carrier),),
            tuple(
                ((left, right), marker if output is None else output)
                for left, right, output in self.table
            ),
            undefined_marker=marker,
        )


@dataclass(frozen=True, slots=True)
class RuleGeneratedDeductiveRecord:
    """Ground finite rules and premises; no parser or executable rule language."""

    rules: tuple[tuple[tuple[str, ...], str], ...]

    @classmethod
    def create(cls, rules: object) -> RuleGeneratedDeductiveRecord:
        if type(rules) not in (tuple, list):
            raise ValueError("rules")
        rule_items = cast(tuple[object, ...] | list[object], rules)
        if not rule_items or len(rule_items) > _MAX_RULES:
            raise ValueError("rules")
        declared_rules: list[tuple[tuple[str, ...], str]] = []
        for rule in rule_items:
            if type(rule) not in (tuple, list):
                raise ValueError("rules")
            rule_values = cast(tuple[object, ...] | list[object], rule)
            if len(rule_values) != 2:
                raise ValueError("rules")
            raw_premises, conclusion = cast(tuple[object, object], rule_values)
            rule_premises = _strings(raw_premises, "rules")
            if type(conclusion) is not str or not conclusion:
                raise ValueError("rules")
            declared_rules.append((rule_premises, conclusion))
        return cls(tuple(declared_rules))

    def rebuild(self) -> DeductiveSystem:
        sort = Sort("judgment")
        atoms: dict[str, Term] = {}

        def atom(name: str) -> Term:
            if name not in atoms:
                atoms[name] = _atom(name, sort)
            return atoms[name]

        rules = tuple(
            InferenceRule(tuple(atom(name) for name in premises), atom(conclusion))
            for premises, conclusion in self.rules
        )
        return DeductiveSystem(rules)


@dataclass(frozen=True, slots=True)
class TableAlgebraRecord:
    """A total bilinear integer multiplication table, with no named family claim."""

    basis: tuple[str, ...]
    cells: tuple[tuple[int, ...], ...]

    @classmethod
    def create(cls, basis: object, cells: object) -> TableAlgebraRecord:
        labels = _strings(basis, "basis")
        return cls(labels, _cells(cells, labels))

    def rebuild(self) -> FiniteMultilinearStructure:
        domain = ZZ()
        module = FreeModule(
            domain, Basis(self.basis, coefficient_domain=domain), name="exit-demo"
        )
        table = BasisProductTable.from_cells(
            module,
            2,
            tuple(
                module.element(
                    {
                        index: coefficient
                        for index, coefficient in enumerate(cell)
                        if coefficient
                    }
                )
                for cell in self.cells
            ),
        )
        return FiniteMultilinearStructure.from_tables(module, table, name="exit-demo")

    def product(self, left: int, right: int) -> tuple[int, ...]:
        return self.cells[left * len(self.basis) + right]


def _table_algebra_from_value(value: FiniteMultilinearStructure) -> TableAlgebraRecord:
    """Extract a bounded integer table from a neutral total algebra value."""
    labels = value.module.basis.labels
    if (
        type(value) is not FiniteMultilinearStructure
        or type(value.module.domain) is not IntegerDomain
        or value.module.domain is not ZZ()
        or value.arity != 2
        or len(labels) > _MAX_CARRIER
    ):
        raise ValueError("algebra")
    cells: list[tuple[int, ...]] = []
    for left, right in product(range(len(labels)), repeat=2):
        coordinates = value.evaluate_basis(left, right).coordinates()
        cell: list[int] = []
        for index in range(len(labels)):
            if index not in coordinates:
                cell.append(0)
                continue
            coefficient = coordinates[index]
            if (
                type(coefficient.parent) is not IntegerDomain
                or coefficient.parent is not ZZ()
                or type(coefficient.value) is not int
            ):
                raise ValueError("algebra")
            cell.append(coefficient.value)
        cells.append(tuple(cell))
    return TableAlgebraRecord.create(labels, cells)


@dataclass(frozen=True, slots=True)
class AlgebraPresentationsRecord:
    """One algebra's symbolic, coordinate, constants, and left-matrix views."""

    algebra: TableAlgebraRecord

    @classmethod
    def create(cls, algebra: object) -> AlgebraPresentationsRecord:
        if type(algebra) is not TableAlgebraRecord:
            raise ValueError("algebra")
        return cls(algebra)

    def rebuild(
        self,
    ) -> tuple[
        tuple[str, ...],
        tuple[tuple[int, ...], ...],
        tuple[tuple[int, ...], ...],
        tuple[Matrix, ...],
    ]:
        algebra = self.algebra.rebuild()
        rank = len(self.algebra.basis)
        matrices: list[Matrix] = []
        for left in range(rank):
            matrix_space = MatrixSpace(rank, rank, ZZ())
            matrices.append(
                matrix_space.element(
                    tuple(
                        tuple(
                            self.algebra.product(left, column)[row]
                            for column in range(rank)
                        )
                        for row in range(rank)
                    )
                )
            )
        constants = tuple(
            self.algebra.product(left, right)
            for left, right in product(range(rank), repeat=2)
        )
        coordinates = tuple(
            tuple(1 if index == basis_index else 0 for index in range(rank))
            for basis_index in range(rank)
        )
        # Construction establishes a total multilinear operation.
        assert algebra.arity == 2
        return self.algebra.basis, coordinates, constants, tuple(matrices)


@dataclass(frozen=True, slots=True)
class AlgebraPresentationBundle:
    """A semantic aggregate presenting one exact algebra in four declared forms."""

    algebra: FiniteMultilinearStructure

    @classmethod
    def from_algebra(cls, algebra: object) -> AlgebraPresentationBundle:
        if type(algebra) is not FiniteMultilinearStructure:
            raise ValueError("algebra")
        _table_algebra_from_value(algebra)
        return cls(algebra)

    def presentations(
        self,
    ) -> tuple[
        tuple[str, ...],
        tuple[tuple[int, ...], ...],
        tuple[tuple[int, ...], ...],
        tuple[Matrix, ...],
    ]:
        """Return symbolic, coordinate, structure-constant, and matrix forms."""
        return AlgebraPresentationsRecord.create(
            _table_algebra_from_value(self.algebra)
        ).rebuild()


@dataclass(frozen=True, slots=True)
class LawValidationReportRecord:
    """A finite associativity-check declaration and its exact counterexample data."""

    carrier: tuple[str, ...]
    table: tuple[str, ...]
    expected_witness: tuple[str, str, str]

    @classmethod
    def create(
        cls, carrier: object, table: object, expected_witness: object
    ) -> LawValidationReportRecord:
        values = _strings(tuple(cast(Iterable[object], carrier)), "carrier")
        products = _strings(table, "table", maximum=_MAX_TABLE, unique=False)
        witness = _strings(
            expected_witness, "expected_witness", maximum=3, unique=False
        )
        if (
            len(products) != len(values) ** 2
            or len(witness) != 3
            or any(value not in values for value in products + witness)
        ):
            raise ValueError("table")
        return cls(values, products, (witness[0], witness[1], witness[2]))

    def rebuild_and_validate(self) -> ValidationReport:
        sort = Sort("element")
        carrier = FiniteCarrier(self.carrier, sort=sort)
        symbol = OperationSymbol("product", (sort, sort), sort)
        operation = Operation.from_table(
            symbol,
            ((sort, carrier),),
            tuple(
                (
                    (left, right),
                    self.table[left_index * len(self.carrier) + right_index],
                )
                for left_index, left in enumerate(self.carrier)
                for right_index, right in enumerate(self.carrier)
            ),
        )
        structure = (
            StructureBuilder(Signature((sort,), (symbol,), ()))
            .with_carrier(sort, carrier)
            .with_operation(symbol, operation)
            .freeze()
        )
        report = validate_law(structure, associative_law(symbol))
        if type(report) is not Disproved or report.witness is None:
            raise ValueError("validation_report")
        witness = tuple(carrier[index] for index in report.witness.substitution_indices)
        if witness != self.expected_witness:
            raise ValueError("expected_witness")
        return report


def _partial_encoder(value: FinitePartialOperationRecord) -> dict[str, JSONValue]:
    return {"carrier": list(value.carrier), "table": [list(row) for row in value.table]}


def _partial_parser(record: object) -> FinitePartialOperationRecord:
    mapping = _require_keys(record, {"schemaType", "schemaVersion", "carrier", "table"})
    return FinitePartialOperationRecord.create(
        mapping.get("carrier"), mapping.get("table")
    )


def _deductive_encoder(value: RuleGeneratedDeductiveRecord) -> dict[str, JSONValue]:
    return {
        "rules": [[list(premises), conclusion] for premises, conclusion in value.rules],
    }


def _deductive_parser(record: object) -> RuleGeneratedDeductiveRecord:
    mapping = _require_keys(record, {"schemaType", "schemaVersion", "rules"})
    return RuleGeneratedDeductiveRecord.create(mapping.get("rules"))


def _algebra_encoder(value: TableAlgebraRecord) -> dict[str, JSONValue]:
    return {"basis": list(value.basis), "cells": [list(cell) for cell in value.cells]}


def _algebra_parser(record: object) -> TableAlgebraRecord:
    mapping = _require_keys(record, {"schemaType", "schemaVersion", "basis", "cells"})
    return TableAlgebraRecord.create(mapping.get("basis"), mapping.get("cells"))


def _report_encoder(value: LawValidationReportRecord) -> dict[str, JSONValue]:
    return {
        "carrier": list(value.carrier),
        "table": list(value.table),
        "expectedWitness": list(value.expected_witness),
    }


def _report_parser(record: object) -> LawValidationReportRecord:
    mapping = _require_keys(
        record,
        {"schemaType", "schemaVersion", "carrier", "table", "expectedWitness"},
    )
    return LawValidationReportRecord.create(
        mapping.get("carrier"),
        mapping.get("table"),
        mapping.get("expectedWitness"),
    )


def _partial_value_encoder(value: PartialOperation) -> dict[str, JSONValue]:
    if type(value) is not PartialOperation or value.symbol.arity != 2:
        raise ValueError("partial_operation")
    carrier = value.output_carrier
    if (
        type(carrier) is not FiniteCarrier
        or len(value.input_carriers) != 2
        or any(input_carrier is not carrier for input_carrier in value.input_carriers)
        or value.symbol.inputs != (carrier.sort, carrier.sort)
        or value.symbol.output is not carrier.sort
        or value.carrier_bindings != ((carrier.sort, carrier),)
        or any(type(item) is not str for item in carrier)
    ):
        raise ValueError("partial_operation")
    outputs = {indices: output for indices, output in value.table}
    undefined = set(value.undefined_indices)
    table = tuple(
        (
            left,
            right,
            None
            if (left_index, right_index) in undefined
            else outputs.get((left_index, right_index)),
        )
        for left_index, left in enumerate(carrier)
        for right_index, right in enumerate(carrier)
        if (left_index, right_index) in undefined
        or (left_index, right_index) in outputs
    )
    return _partial_encoder(FinitePartialOperationRecord.create(carrier, table))


def _partial_value_parser(record: object) -> PartialOperation:
    return _partial_parser(record).rebuild()


def _term_name(term: Term, sort: Sort | None) -> tuple[str, Sort]:
    if term.symbol is None or term.symbol.arity != 0 or term.arguments:
        raise ValueError("deductive_system")
    if term.symbol.output is not term.sort:
        raise ValueError("deductive_system")
    if sort is not None and term.sort is not sort:
        raise ValueError("deductive_system")
    return term.symbol.name, term.sort


def _deductive_value_encoder(value: DeductiveSystem) -> dict[str, JSONValue]:
    if type(value) is not DeductiveSystem or not value.rules:
        raise ValueError("deductive_system")
    sort: Sort | None = None
    rules: list[tuple[tuple[str, ...], str]] = []
    for rule in value.rules:
        names: list[str] = []
        for term in rule.premises:
            name, sort = _term_name(term, sort)
            names.append(name)
        conclusion, sort = _term_name(rule.conclusion, sort)
        rules.append((tuple(names), conclusion))
    return _deductive_encoder(RuleGeneratedDeductiveRecord.create(rules))


def _deductive_value_parser(record: object) -> DeductiveSystem:
    return _deductive_parser(record).rebuild()


def _algebra_value_encoder(value: FiniteMultilinearStructure) -> dict[str, JSONValue]:
    return _algebra_encoder(_table_algebra_from_value(value))


def _algebra_value_parser(record: object) -> FiniteMultilinearStructure:
    return _algebra_parser(record).rebuild()


def _bundle_encoder(value: AlgebraPresentationBundle) -> dict[str, JSONValue]:
    return _algebra_value_encoder(value.algebra)


def _bundle_parser(record: object) -> AlgebraPresentationBundle:
    return AlgebraPresentationBundle.from_algebra(_algebra_value_parser(record))


def _report_value_encoder(value: Disproved) -> dict[str, JSONValue]:
    if type(value) is not Disproved:
        raise ValueError("validation_report")
    structure = value.structure
    if (
        type(structure) is not Structure
        or len(structure.carriers) != 1
        or len(structure.operations) != 1
        or structure.relations
    ):
        raise ValueError("validation_report")
    operation = structure.operations[0]
    carrier = operation.output_carrier
    if (
        type(operation) is not Operation
        or type(carrier) is not FiniteCarrier
        or structure.carriers != (carrier,)
        or operation.input_carriers != (carrier, carrier)
        or operation.symbol.inputs != (carrier.sort, carrier.sort)
        or operation.symbol.output is not carrier.sort
        or operation.carrier_bindings != ((carrier.sort, carrier),)
        or any(type(item) is not str for item in carrier)
        or value.law != associative_law(operation.symbol)
    ):
        raise ValueError("validation_report")
    outputs = {indices: output for indices, output in operation.table}
    if len(outputs) != len(carrier) ** 2 or value.witness is None:
        raise ValueError("validation_report")
    witness = tuple(carrier[index] for index in value.witness.substitution_indices)
    record = LawValidationReportRecord.create(
        carrier,
        tuple(
            outputs[(left, right)]
            for left, right in product(range(len(carrier)), repeat=2)
        ),
        witness,
    )
    rebuilt = record.rebuild_and_validate()
    if (
        type(rebuilt) is not Disproved
        or rebuilt.witness is None
        or rebuilt.witness.substitution_indices != value.witness.substitution_indices
        or rebuilt.evaluated_assignments != value.evaluated_assignments
    ):
        raise ValueError("validation_report")
    return _report_encoder(record)


def _report_value_parser(record: object) -> Disproved:
    return cast(Disproved, _report_parser(record).rebuild_and_validate())


def semantic_record_registry() -> SerializerRegistry:
    """Return the fixed v0.0 semantic-record allow-list used by the exit demo."""
    return (
        SerializerRegistry()
        .with_codec(
            "finite_partial_operation",
            1,
            PartialOperation,
            _partial_value_encoder,
            _partial_value_parser,
        )
        .with_codec(
            "rule_generated_deductive",
            1,
            DeductiveSystem,
            _deductive_value_encoder,
            _deductive_value_parser,
        )
        .with_codec(
            "table_algebra",
            1,
            FiniteMultilinearStructure,
            _algebra_value_encoder,
            _algebra_value_parser,
        )
        .with_codec(
            "algebra_presentations",
            1,
            AlgebraPresentationBundle,
            _bundle_encoder,
            _bundle_parser,
        )
        .with_codec(
            "law_validation_report",
            1,
            Disproved,
            _report_value_encoder,
            _report_value_parser,
        )
    )
