"""Conservative eligibility gate for later basis/theorem reduction routes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import Sort
from anyalgebra.structures.laws import Law
from anyalgebra.structures.signatures import OperationSymbol
from anyalgebra.structures.structure import Structure
from anyalgebra.structures.terms import Term
from anyalgebra.validation.validate import Proved


_MAX_OBLIGATIONS = 512


class BasisReductionDefinitionError(AnyAlgebraError, ValueError):
    """A reduction-gate declaration was malformed before any theorem use."""

    def __init__(self, *, field: str, reason: str, index: int | None = None) -> None:
        self.field, self.reason, self.index = field, reason, index
        location = field if index is None else f"{field}[{index}]"
        super().__init__(f"invalid basis reduction {location}: {reason}")


def _report(value: object) -> Proved:
    """Require exact exhaustive proof evidence; labels and booleans never suffice."""
    if type(value) is not Proved:
        raise BasisReductionDefinitionError(
            field="evidence", reason="must be an exact Proved exhaustive report"
        )
    if value.expected_assignments != value.evaluated_assignments:
        raise BasisReductionDefinitionError(
            field="evidence", reason="must exhaust its declared assignment scope"
        )
    return value


def _string(value: object, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise BasisReductionDefinitionError(
            field=field, reason="must be a non-empty trimmed built-in str"
        )
    return value


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class MultilinearityObligation:
    operation: OperationSymbol
    slots: tuple[int, ...]
    evidence: Proved
    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self, operation: OperationSymbol, slots: Iterable[int], evidence: Proved
    ) -> None:
        if type(operation) is not OperationSymbol:
            raise BasisReductionDefinitionError(
                field="operation", reason="must be an exact OperationSymbol"
            )
        try:
            values = tuple(slots)
        except Exception:
            raise BasisReductionDefinitionError(
                field="slots", reason="must be iterable"
            ) from None
        if len(values) > operation.arity:
            raise BasisReductionDefinitionError(
                field="slots", reason="has too many declared slots"
            )
        for index, slot in enumerate(values):
            if type(slot) is not int or slot < 0 or slot >= operation.arity:
                raise BasisReductionDefinitionError(
                    field="slots",
                    reason="must contain valid built-in slot indices",
                    index=index,
                )
            if slot in values[:index]:
                raise BasisReductionDefinitionError(
                    field="slots", reason="contains a duplicate slot", index=index
                )
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "slots", values)
        object.__setattr__(self, "evidence", _report(evidence))

    def __repr__(self) -> str:
        return f"MultilinearityObligation(slot_count={len(self.slots)})"


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ClosureObligation:
    operation: OperationSymbol
    sort: Sort
    evidence: Proved
    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self, operation: OperationSymbol, sort: Sort, evidence: Proved
    ) -> None:
        if type(operation) is not OperationSymbol:
            raise BasisReductionDefinitionError(
                field="operation", reason="must be an exact OperationSymbol"
            )
        if type(sort) is not Sort:
            raise BasisReductionDefinitionError(
                field="sort", reason="must be an exact Sort"
            )
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "sort", sort)
        object.__setattr__(self, "evidence", _report(evidence))

    def __repr__(self) -> str:
        return "ClosureObligation()"


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ScalarDomainObligation:
    scope: str
    characteristic_condition: str
    polarization_condition: str
    evidence: Proved
    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self,
        scope: str,
        characteristic_condition: str,
        polarization_condition: str,
        evidence: Proved,
    ) -> None:
        object.__setattr__(self, "scope", _string(scope, "scope"))
        object.__setattr__(
            self,
            "characteristic_condition",
            _string(characteristic_condition, "characteristic_condition"),
        )
        object.__setattr__(
            self,
            "polarization_condition",
            _string(polarization_condition, "polarization_condition"),
        )
        object.__setattr__(self, "evidence", _report(evidence))

    def __repr__(self) -> str:
        return "ScalarDomainObligation()"


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class PremiseObligation:
    premise_index: int
    evidence: Proved
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, premise_index: int, evidence: Proved) -> None:
        if type(premise_index) is not int or premise_index < 0:
            raise BasisReductionDefinitionError(
                field="premise_index", reason="must be a non-negative built-in int"
            )
        object.__setattr__(self, "premise_index", premise_index)
        object.__setattr__(self, "evidence", _report(evidence))

    def __repr__(self) -> str:
        return f"PremiseObligation(premise_index={self.premise_index})"


Obligation = (
    MultilinearityObligation
    | ClosureObligation
    | ScalarDomainObligation
    | PremiseObligation
)


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class BasisReductionReady:
    structure: Structure
    law: Law
    obligations: tuple[Obligation, ...]
    is_conditional: bool
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise BasisReductionDefinitionError(field="result", reason="is gate-owned")

    def __repr__(self) -> str:
        return (
            "BasisReductionReady("
            f"obligation_count={len(self.obligations)}, is_conditional=True)"
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class BasisReductionDenied:
    structure: Structure
    law: Law
    obligations: tuple[Obligation, ...]
    issues: tuple[str, ...]
    is_conditional: bool
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise BasisReductionDefinitionError(field="result", reason="is gate-owned")

    def __repr__(self) -> str:
        return (
            f"BasisReductionDenied(issue_count={len(self.issues)}, is_conditional=True)"
        )


def _symbols(term: Term) -> tuple[OperationSymbol, ...]:
    """Discover symbols iteratively before any evaluator or theorem execution."""
    pending: list[Term] = [term]
    found: list[OperationSymbol] = []
    visited = 0
    while pending:
        node = pending.pop()
        visited += 1
        if visited > _MAX_OBLIGATIONS:
            raise BasisReductionDefinitionError(
                field="law", reason="term syntax exceeds declared maximum node count"
            )
        if node.symbol is not None:
            found.append(node.symbol)
        pending.extend(reversed(node.arguments))
    return tuple(found)


def _snapshot(values: Iterable[Obligation]) -> tuple[Obligation, ...]:
    try:
        iterator = iter(values)
    except Exception:
        raise BasisReductionDefinitionError(
            field="obligations", reason="could not be snapshotted"
        ) from None
    result: list[Obligation] = []
    for _ in range(_MAX_OBLIGATIONS + 1):
        try:
            item = next(iterator)
        except StopIteration:
            break
        except Exception:
            raise BasisReductionDefinitionError(
                field="obligations", reason="could not be snapshotted"
            ) from None
        if type(item) not in (
            MultilinearityObligation,
            ClosureObligation,
            ScalarDomainObligation,
            PremiseObligation,
        ):
            raise BasisReductionDefinitionError(
                field="obligations",
                reason="must contain exact obligation records",
                index=len(result),
            )
        result.append(item)
    if len(result) > _MAX_OBLIGATIONS:
        raise BasisReductionDefinitionError(
            field="obligations", reason="exceeds declared maximum", index=len(result)
        )
    return tuple(result)


def _provenance(evidence: Proved, structure: Structure, law: Law) -> bool:
    return (
        evidence.structure is structure
        and evidence.law is not law
        and evidence.expected_assignments == evidence.evaluated_assignments
        and evidence.algorithm == "anyalgebra.exhaustive_finite_law"
        and evidence.algorithm_version == 1
    )


def _declares_sort(structure: Structure, sort: Sort) -> bool:
    """Use the signature's structural sort semantics without evaluating it."""
    return any(declared == sort for declared in structure.signature.sorts)


def _declares_operation(structure: Structure, symbol: OperationSymbol) -> bool:
    """Use the signature's structural operation semantics without applying it."""
    return any(declared == symbol for declared in structure.signature.operations)


def _target_operations(structure: Structure, law: Law) -> tuple[OperationSymbol, ...]:
    """Preflight target syntax before consuming evidence or invoking operations."""
    for variable in law.variables:
        if not _declares_sort(structure, variable.sort):
            raise BasisReductionDefinitionError(
                field="variables", reason="variable sort is not declared by structure"
            )
    operations: list[OperationSymbol] = []
    for equation in (law.conclusion, *law.hypotheses):
        if not _declares_sort(structure, equation.left.sort):
            raise BasisReductionDefinitionError(
                field="law", reason="equation result sort is not declared by structure"
            )
        for symbol in (*_symbols(equation.left), *_symbols(equation.right)):
            if not _declares_operation(structure, symbol):
                raise BasisReductionDefinitionError(
                    field="law",
                    reason="law operation symbol is not declared by structure",
                )
            if not any(symbol is prior for prior in operations):
                operations.append(symbol)
    return tuple(operations)


def basis_reduction_gate(
    structure: Structure, law: Law, obligations: Iterable[Obligation]
) -> BasisReductionReady | BasisReductionDenied:
    """Return only theorem-route eligibility; never a proof of ``law`` itself."""
    if type(structure) is not Structure:
        raise BasisReductionDefinitionError(
            field="structure", reason="must be an exact Structure"
        )
    if type(law) is not Law:
        raise BasisReductionDefinitionError(field="law", reason="must be an exact Law")
    operations = _target_operations(structure, law)
    supplied = _snapshot(obligations)
    issues: list[str] = []
    valid = tuple(
        item for item in supplied if _provenance(item.evidence, structure, law)
    )
    if len(valid) != len(supplied):
        issues.append("invalid_provenance")
    evidence_reports: list[Proved] = []
    for item in valid:
        if any(item.evidence is prior for prior in evidence_reports):
            issues.append("duplicate_evidence_report")
        evidence_reports.append(item.evidence)
    seen: list[tuple[object, ...]] = []
    for item in valid:
        if type(item) is MultilinearityObligation:
            keys: list[tuple[object, ...]] = [
                ("multilinearity", item.operation, slot) for slot in item.slots
            ]
        elif type(item) is ClosureObligation:
            keys = [("closure", item.operation, item.sort)]
        elif type(item) is ScalarDomainObligation:
            keys = [("scalar",)]
        else:
            assert type(item) is PremiseObligation
            keys = [("premise", item.premise_index)]
        for key in keys:
            if key in seen:
                issues.append("duplicate_obligation")
            seen.append(key)
        if type(item) is MultilinearityObligation:
            if not any(item.operation is operation for operation in operations):
                issues.append("irrelevant_multilinearity")
        elif type(item) is ClosureObligation:
            if not any(item.operation is operation for operation in operations):
                issues.append("irrelevant_closure")
            elif item.sort is not item.operation.output:
                issues.append("invalid_closure_sort")
        elif type(item) is PremiseObligation:
            if item.premise_index >= len(law.hypotheses):
                issues.append("invalid_premise_index")
            elif item.evidence.law.conclusion is not law.hypotheses[item.premise_index]:
                issues.append("invalid_premise_evidence")
    for operation in operations:
        slots = {
            slot
            for item in valid
            if type(item) is MultilinearityObligation and item.operation is operation
            for slot in item.slots
        }
        for slot in range(operation.arity):
            if slot not in slots:
                issues.append(f"missing_multilinearity:{operation.name}:{slot}")
        if not any(
            type(item) is ClosureObligation
            and item.operation is operation
            and item.sort is operation.output
            for item in valid
        ):
            issues.append(f"missing_closure:{operation.name}:{operation.output.name}")
    result_sort = law.conclusion.left.sort
    if operations and not any(
        type(item) is ClosureObligation and item.sort is result_sort for item in valid
    ):
        issues.append(f"missing_result_closure:{result_sort.name}")
    if not any(type(item) is ScalarDomainObligation for item in valid):
        issues.append("missing_scalar_domain")
    for index in range(len(law.hypotheses)):
        if not any(
            type(item) is PremiseObligation
            and item.premise_index == index
            and item.evidence.law.conclusion is law.hypotheses[index]
            for item in valid
        ):
            issues.append(f"missing_premise:{index}")
    target = BasisReductionReady if not issues else BasisReductionDenied
    value = object.__new__(target)
    object.__setattr__(value, "structure", structure)
    object.__setattr__(value, "law", law)
    object.__setattr__(value, "obligations", supplied)
    object.__setattr__(value, "is_conditional", True)
    if target is BasisReductionDenied:
        object.__setattr__(value, "issues", tuple(issues))
    return cast(BasisReductionReady | BasisReductionDenied, value)
