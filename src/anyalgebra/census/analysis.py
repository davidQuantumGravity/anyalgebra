"""Complete finite-grid law analysis for one total operation table."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
from itertools import product
import json
from typing import TypeAlias, cast

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import SemanticHash

from .constraints import CensusConstraint, CensusTerm, ConstraintError, ConstraintSet
from .equivalence import EquivalencePolicy
from .partition import compute_invariant_partition
from .spec import CensusSpecCore
from .table_codes import OperationTableCodeError, rank_operation_table


_SCHEMA_VERSION = 1


class FiniteLawAnalysisError(AnyAlgebraError, ValueError):
    """A finite law input or retained complete result was invalid."""

    def __init__(self, *, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"invalid finite law analysis {field}: {reason}")


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


def _hash_record(value: SemanticHash) -> dict[str, str]:
    return {"algorithm": value.algorithm, "digest": value.digest}


def _flat(arguments: tuple[int, ...], size: int) -> int:
    index = 0
    for argument in arguments:
        index = index * size + argument
    return index


def _apply(outputs: tuple[int, ...], size: int, *arguments: int) -> int:
    return outputs[_flat(arguments, size)]


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class LawWitness:
    """Lexicographically smallest failed assignment and its two values."""

    assignment: tuple[int, ...]
    left_value: int
    right_value: int
    relation: str
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise FiniteLawAnalysisError(field="witness", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("LawWitness cannot be subclassed")

    @classmethod
    def _create(
        cls,
        assignment: tuple[int, ...],
        left_value: int,
        right_value: int,
        relation: str,
    ) -> LawWitness:
        value = object.__new__(LawWitness)
        for field, item in (
            ("assignment", assignment),
            ("left_value", left_value),
            ("right_value", right_value),
            ("relation", relation),
        ):
            object.__setattr__(value, field, item)
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is LawWitness and (
            self.assignment,
            self.left_value,
            self.right_value,
            self.relation,
        ) == (
            other.assignment,
            other.left_value,
            other.right_value,
            other.relation,
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class ProvedLaw:
    name: str
    status: str
    variable_count: int
    assignment_count: int
    evaluated_count: int

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise FiniteLawAnalysisError(field="proved_law", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("ProvedLaw cannot be subclassed")

    @classmethod
    def _create(cls, name: str, variables: int, count: int) -> ProvedLaw:
        value = object.__new__(ProvedLaw)
        for field, item in (
            ("name", name),
            ("status", "proved_on_complete_grid"),
            ("variable_count", variables),
            ("assignment_count", count),
            ("evaluated_count", count),
        ):
            object.__setattr__(value, field, item)
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is ProvedLaw and _law_identity(self) == _law_identity(other)


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class DisprovedLaw:
    name: str
    status: str
    variable_count: int
    assignment_count: int
    evaluated_count: int
    witness: LawWitness

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise FiniteLawAnalysisError(field="disproved_law", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("DisprovedLaw cannot be subclassed")

    @classmethod
    def _create(
        cls, name: str, variables: int, count: int, witness: LawWitness
    ) -> DisprovedLaw:
        value = object.__new__(DisprovedLaw)
        for field, item in (
            ("name", name),
            ("status", "disproved_on_complete_grid"),
            ("variable_count", variables),
            ("assignment_count", count),
            ("evaluated_count", count),
            ("witness", witness),
        ):
            object.__setattr__(value, field, item)
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is DisprovedLaw and _law_identity(self) == _law_identity(
            other
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class UnsupportedLaw:
    name: str
    status: str
    reason: str
    variable_count: int
    assignment_count: int
    evaluated_count: int

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise FiniteLawAnalysisError(field="unsupported_law", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("UnsupportedLaw cannot be subclassed")

    @classmethod
    def _create(cls, name: str, reason: str) -> UnsupportedLaw:
        value = object.__new__(UnsupportedLaw)
        for field, item in (
            ("name", name),
            ("status", "unsupported"),
            ("reason", reason),
            ("variable_count", 0),
            ("assignment_count", 0),
            ("evaluated_count", 0),
        ):
            object.__setattr__(value, field, item)
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is UnsupportedLaw and _law_identity(self) == _law_identity(
            other
        )


LawResult: TypeAlias = ProvedLaw | DisprovedLaw | UnsupportedLaw


def _law_identity(value: LawResult) -> tuple[object, ...]:
    witness = value.witness if isinstance(value, DisprovedLaw) else None
    reason = value.reason if isinstance(value, UnsupportedLaw) else None
    return (
        value.name,
        value.status,
        value.variable_count,
        value.assignment_count,
        value.evaluated_count,
        witness,
        reason,
    )


_LawEvaluator: TypeAlias = Callable[[tuple[int, ...]], tuple[int, int, str]]
MAX_FINITE_LAW_ASSIGNMENTS = 20_000_000


def _complete_law(
    name: str,
    variable_count: int,
    size: int,
    evaluator: _LawEvaluator,
) -> LawResult:
    count = size**variable_count
    witness: LawWitness | None = None
    for assignment in product(range(size), repeat=variable_count):
        left, right, relation = evaluator(assignment)
        if left != right and witness is None:
            witness = LawWitness._create(assignment, left, right, relation)
    if witness is None:
        return ProvedLaw._create(name, variable_count, count)
    return DisprovedLaw._create(name, variable_count, count, witness)


def _term_value(
    term: CensusTerm,
    assignment: tuple[int, ...],
    outputs: tuple[int, ...],
    size: int,
) -> int:
    if term.kind == "variable":
        return assignment[cast(int, term.index)]
    if term.kind == "constant":
        return cast(int, term.index)
    arguments = tuple(
        _term_value(argument, assignment, outputs, size) for argument in term.arguments
    )
    return _apply(outputs, size, *arguments)


def _equation_count(constraint: CensusConstraint) -> int:
    return cast(int, dict(constraint.parameters)["variable_count"])


def _preflight_law_work(
    core: CensusSpecCore, equations: tuple[CensusConstraint, ...]
) -> None:
    """Reject an aggregate complete grid before any assignment traversal."""
    size = core.carrier_size
    requested = core.input_tuple_count
    if core.arity > 0:
        requested += size
    if core.arity == 2:
        requested += size**3 + 5 * size**2
    requested += sum(size ** _equation_count(item) for item in equations)
    if requested > MAX_FINITE_LAW_ASSIGNMENTS:
        raise FiniteLawAnalysisError(
            field="work",
            reason=(
                "aggregate assignment grid exceeds hard limit "
                f"{MAX_FINITE_LAW_ASSIGNMENTS}"
            ),
        )


def _standard_laws(
    core: CensusSpecCore, outputs: tuple[int, ...]
) -> tuple[LawResult, ...]:
    size = core.carrier_size
    results: list[LawResult] = [
        ProvedLaw._create("totality", core.arity, core.input_tuple_count)
    ]
    if core.arity == 0:
        results.append(UnsupportedLaw._create("idempotence", "requires_input"))
    else:
        results.append(
            _complete_law(
                "idempotence",
                1,
                size,
                lambda a: (
                    _apply(outputs, size, *(a[0] for _ in range(core.arity))),
                    a[0],
                    "f(x,...,x)=x",
                ),
            )
        )
    binary_names = (
        "commutativity",
        "associativity",
        "flexibility",
        "left_alternativity",
        "right_alternativity",
        "alternativity",
    )
    if core.arity != 2:
        results.extend(
            UnsupportedLaw._create(name, "requires_binary_operation")
            for name in binary_names
        )
        return tuple(results)

    def op(left: int, right: int) -> int:
        return _apply(outputs, size, left, right)

    def commutativity(a: tuple[int, ...]) -> tuple[int, int, str]:
        return op(a[0], a[1]), op(a[1], a[0]), "xy=yx"

    def associativity(a: tuple[int, ...]) -> tuple[int, int, str]:
        return (
            op(op(a[0], a[1]), a[2]),
            op(a[0], op(a[1], a[2])),
            "(xy)z=x(yz)",
        )

    def flexibility(a: tuple[int, ...]) -> tuple[int, int, str]:
        return (
            op(op(a[0], a[1]), a[0]),
            op(a[0], op(a[1], a[0])),
            "(xy)x=x(yx)",
        )

    def left_alternativity(a: tuple[int, ...]) -> tuple[int, int, str]:
        return (
            op(op(a[0], a[0]), a[1]),
            op(a[0], op(a[0], a[1])),
            "(xx)y=x(xy)",
        )

    def right_alternativity(a: tuple[int, ...]) -> tuple[int, int, str]:
        return (
            op(a[0], op(a[1], a[1])),
            op(op(a[0], a[1]), a[1]),
            "x(yy)=(xy)y",
        )

    results.extend(
        (
            _complete_law("commutativity", 2, size, commutativity),
            _complete_law("associativity", 3, size, associativity),
            _complete_law("flexibility", 2, size, flexibility),
            _complete_law("left_alternativity", 2, size, left_alternativity),
            _complete_law("right_alternativity", 2, size, right_alternativity),
            _complete_law(
                "alternativity", 2, size, lambda a: _alternativity_values(op, a)
            ),
        )
    )
    return tuple(results)


def _alternativity_values(
    operation: Callable[[int, int], int], assignment: tuple[int, ...]
) -> tuple[int, int, str]:
    x, y = assignment
    left_left = operation(operation(x, x), y)
    left_right = operation(x, operation(x, y))
    if left_left != left_right:
        return left_left, left_right, "(xx)y=x(xy)"
    return (
        operation(x, operation(y, y)),
        operation(operation(x, y), y),
        "x(yy)=(xy)y",
    )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class FiniteLawProfile:
    core: CensusSpecCore
    outputs: tuple[int, ...]
    equations: tuple[CensusConstraint, ...]
    results: tuple[LawResult, ...]
    complete: bool
    declared_grid_count: int
    total_evaluations: int
    algorithm: str
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise FiniteLawAnalysisError(field="profile", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("FiniteLawProfile cannot be subclassed")

    @classmethod
    def _create(
        cls,
        core: CensusSpecCore,
        outputs: tuple[int, ...],
        equations: tuple[CensusConstraint, ...],
        results: tuple[LawResult, ...],
    ) -> FiniteLawProfile:
        declared = sum(result.assignment_count for result in results)
        evaluated = sum(result.evaluated_count for result in results)
        value = object.__new__(FiniteLawProfile)
        for field, item in (
            ("core", core),
            ("outputs", outputs),
            ("equations", equations),
            ("results", results),
            ("complete", True),
            ("declared_grid_count", declared),
            ("total_evaluations", evaluated),
            ("algorithm", "finite_full_grid_laws_v1"),
            ("semantic_hash", SemanticHash("sha256", "0" * 64)),
        ):
            object.__setattr__(value, field, item)
        object.__setattr__(value, "semantic_hash", _semantic_hash(_body(value)))
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is FiniteLawProfile and (
            self.core,
            self.outputs,
            self.equations,
            self.results,
            self.complete,
            self.declared_grid_count,
            self.total_evaluations,
            self.algorithm,
            self.semantic_hash,
        ) == (
            other.core,
            other.outputs,
            other.equations,
            other.results,
            other.complete,
            other.declared_grid_count,
            other.total_evaluations,
            other.algorithm,
            other.semantic_hash,
        )


def _result_record(value: LawResult) -> dict[str, object]:
    record: dict[str, object] = {
        "assignmentCount": value.assignment_count,
        "evaluatedCount": value.evaluated_count,
        "name": value.name,
        "status": value.status,
        "variableCount": value.variable_count,
    }
    if isinstance(value, DisprovedLaw):
        record["witness"] = {
            "assignment": list(value.witness.assignment),
            "leftValue": value.witness.left_value,
            "relation": value.witness.relation,
            "rightValue": value.witness.right_value,
        }
    elif isinstance(value, UnsupportedLaw):
        record["reason"] = value.reason
    return record


def _body(value: FiniteLawProfile) -> dict[str, object]:
    return {
        "algorithm": value.algorithm,
        "arity": value.core.arity,
        "carrierSize": value.core.carrier_size,
        "complete": value.complete,
        "declaredGridCount": value.declared_grid_count,
        "outputs": list(value.outputs),
        "results": [_result_record(result) for result in value.results],
        "schemaType": "anyalgebra.census.finite_law_profile",
        "schemaVersion": _SCHEMA_VERSION,
        "totalEvaluations": value.total_evaluations,
    }


def analyze_finite_laws(
    core: CensusSpecCore,
    outputs: tuple[int, ...],
    *,
    equations: tuple[CensusConstraint, ...] = (),
) -> FiniteLawProfile:
    if type(core) is not CensusSpecCore:
        raise FiniteLawAnalysisError(
            field="core", reason="must be exact CensusSpecCore"
        )
    if core.candidate_count == 0:
        raise FiniteLawAnalysisError(
            field="outputs", reason="core has no operation tables"
        )
    if type(outputs) is not tuple:
        raise FiniteLawAnalysisError(field="outputs", reason="must be an exact tuple")
    try:
        rank_operation_table(core, outputs)
    except OperationTableCodeError as error:
        raise FiniteLawAnalysisError(field="outputs", reason="invalid table") from error
    if type(equations) is not tuple or any(
        type(item) is not CensusConstraint for item in equations
    ):
        raise FiniteLawAnalysisError(
            field="equations", reason="must be an exact tuple of equations"
        )
    if any(item.kind != "equation" for item in equations):
        raise FiniteLawAnalysisError(
            field="equations", reason="only equation constraints are accepted"
        )
    try:
        canonical_equations = ConstraintSet.create(core, equations).constraints
    except ConstraintError as error:
        raise FiniteLawAnalysisError(
            field="equations", reason="equation validation failed"
        ) from error
    _preflight_law_work(core, canonical_equations)
    results = list(_standard_laws(core, outputs))
    for index, equation in enumerate(canonical_equations):
        assert equation.left is not None and equation.right is not None
        variable_count = _equation_count(equation)

        def evaluate_equation(
            assignment: tuple[int, ...], item: CensusConstraint = equation
        ) -> tuple[int, int, str]:
            assert item.left is not None and item.right is not None
            return (
                _term_value(item.left, assignment, outputs, core.carrier_size),
                _term_value(item.right, assignment, outputs, core.carrier_size),
                "declared_equation",
            )

        results.append(
            _complete_law(
                f"equation_{index:03d}",
                variable_count,
                core.carrier_size,
                evaluate_equation,
            )
        )
    return FiniteLawProfile._create(core, outputs, canonical_equations, tuple(results))


def law_profile_record(value: FiniteLawProfile) -> dict[str, object]:
    if type(value) is not FiniteLawProfile:
        raise FiniteLawAnalysisError(
            field="profile", reason="must be exact FiniteLawProfile"
        )
    expected = analyze_finite_laws(value.core, value.outputs, equations=value.equations)
    if value != expected:
        raise FiniteLawAnalysisError(field="profile", reason="content drift")
    body = _body(value)
    semantic_hash = _semantic_hash(body)
    if semantic_hash != value.semantic_hash:
        raise FiniteLawAnalysisError(field="semantic_hash", reason="content drift")
    return {**body, "contentHash": _hash_record(semantic_hash)}


def law_profile_canonical_bytes(value: FiniteLawProfile) -> bytes:
    return _encoded(law_profile_record(value))


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class FiniteElementProfile:
    element: int
    color: int
    left_identity: bool
    right_identity: bool
    identity: bool
    left_zero: bool
    right_zero: bool
    zero: bool
    idempotent: bool
    nilpotence_index: int | None

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise FiniteLawAnalysisError(field="element_profile", reason="is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("FiniteElementProfile cannot be subclassed")

    @classmethod
    def _create(
        cls,
        element: int,
        color: int,
        *,
        left_identity: bool,
        right_identity: bool,
        left_zero: bool,
        right_zero: bool,
        idempotent: bool,
        nilpotence_index: int | None,
    ) -> FiniteElementProfile:
        value = object.__new__(FiniteElementProfile)
        for field, item in (
            ("element", element),
            ("color", color),
            ("left_identity", left_identity),
            ("right_identity", right_identity),
            ("identity", left_identity and right_identity),
            ("left_zero", left_zero),
            ("right_zero", right_zero),
            ("zero", left_zero and right_zero),
            ("idempotent", idempotent),
            ("nilpotence_index", nilpotence_index),
        ):
            object.__setattr__(value, field, item)
        return value

    def __eq__(self, other: object) -> bool:
        return type(other) is FiniteElementProfile and _element_identity(
            self
        ) == _element_identity(other)


def _element_identity(value: FiniteElementProfile) -> tuple[object, ...]:
    return (
        value.element,
        value.color,
        value.left_identity,
        value.right_identity,
        value.identity,
        value.left_zero,
        value.right_zero,
        value.zero,
        value.idempotent,
        value.nilpotence_index,
    )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class FiniteElementAnalysis:
    core: CensusSpecCore
    outputs: tuple[int, ...]
    equivalence: EquivalencePolicy
    elements: tuple[FiniteElementProfile, ...]
    colors: tuple[int, ...]
    left_identities: tuple[int, ...]
    right_identities: tuple[int, ...]
    identities: tuple[int, ...]
    left_zeros: tuple[int, ...]
    right_zeros: tuple[int, ...]
    zeros: tuple[int, ...]
    idempotents: tuple[int, ...]
    nilpotent_elements: tuple[int, ...]
    left_identity_status: str
    right_identity_status: str
    identity_status: str
    left_zero_status: str
    right_zero_status: str
    zero_status: str
    nilpotence_status: str
    nilpotence_zero: int | None
    max_nilpotence_index: int | None
    examined_element_count: int
    complete: bool
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise FiniteLawAnalysisError(
            field="element_analysis", reason="is factory-owned"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("FiniteElementAnalysis cannot be subclassed")

    def __eq__(self, other: object) -> bool:
        return type(other) is FiniteElementAnalysis and (
            self.core,
            self.outputs,
            self.equivalence,
            self.elements,
            self.nilpotence_zero,
            self.max_nilpotence_index,
            self.semantic_hash,
        ) == (
            other.core,
            other.outputs,
            other.equivalence,
            other.elements,
            other.nilpotence_zero,
            other.max_nilpotence_index,
            other.semantic_hash,
        )


def _presence_status(values: tuple[int, ...], *, supported: bool) -> str:
    if not supported:
        return "unsupported_nonbinary"
    if not values:
        return "absent"
    if len(values) == 1:
        return "unique"
    return "nonunique"


def _nilpotence_declaration(
    core: CensusSpecCore,
    zero: object,
    bound: object,
) -> tuple[int | None, int | None]:
    if zero is None and bound is None:
        return None, None
    if (
        core.arity != 2
        or type(zero) is not int
        or not 0 <= zero < core.carrier_size
        or type(bound) is not int
        or bound < 1
    ):
        raise FiniteLawAnalysisError(
            field="nilpotence",
            reason="requires binary operation, carrier zero, and positive exact bound",
        )
    return zero, bound


def _nilpotence_index(
    outputs: tuple[int, ...], size: int, element: int, zero: int, bound: int
) -> int | None:
    power = element
    if power == zero:
        return 1
    for exponent in range(2, bound + 1):
        power = _apply(outputs, size, power, element)
        if power == zero:
            return exponent
    return None


def analyze_finite_elements(
    core: CensusSpecCore,
    outputs: tuple[int, ...],
    equivalence: EquivalencePolicy,
    *,
    nilpotence_zero: int | None = None,
    max_nilpotence_index: int | None = None,
) -> FiniteElementAnalysis:
    if type(core) is not CensusSpecCore:
        raise FiniteLawAnalysisError(
            field="core", reason="must be exact CensusSpecCore"
        )
    if core.candidate_count == 0:
        raise FiniteLawAnalysisError(
            field="outputs", reason="core has no operation tables"
        )
    if type(equivalence) is not EquivalencePolicy or equivalence.core is not core:
        raise FiniteLawAnalysisError(
            field="equivalence", reason="must belong to literal analysis core"
        )
    if type(outputs) is not tuple:
        raise FiniteLawAnalysisError(field="outputs", reason="must be exact tuple")
    try:
        rank_operation_table(core, outputs)
    except OperationTableCodeError as error:
        raise FiniteLawAnalysisError(field="outputs", reason="invalid table") from error
    zero, bound = _nilpotence_declaration(core, nilpotence_zero, max_nilpotence_index)
    partition = compute_invariant_partition(core, outputs, equivalence)
    binary = core.arity == 2
    profiles: list[FiniteElementProfile] = []
    for element in range(core.carrier_size):
        left_identity = binary and all(
            _apply(outputs, core.carrier_size, element, item) == item
            for item in range(core.carrier_size)
        )
        right_identity = binary and all(
            _apply(outputs, core.carrier_size, item, element) == item
            for item in range(core.carrier_size)
        )
        left_zero = binary and all(
            _apply(outputs, core.carrier_size, element, item) == element
            for item in range(core.carrier_size)
        )
        right_zero = binary and all(
            _apply(outputs, core.carrier_size, item, element) == element
            for item in range(core.carrier_size)
        )
        idempotent = (
            core.arity > 0
            and _apply(
                outputs,
                core.carrier_size,
                *(element for _ in range(core.arity)),
            )
            == element
        )
        nil_index = (
            _nilpotence_index(outputs, core.carrier_size, element, zero, bound)
            if zero is not None and bound is not None
            else None
        )
        profiles.append(
            FiniteElementProfile._create(
                element,
                partition.colors[element],
                left_identity=left_identity,
                right_identity=right_identity,
                left_zero=left_zero,
                right_zero=right_zero,
                idempotent=idempotent,
                nilpotence_index=nil_index,
            )
        )
    elements = tuple(profiles)
    lists = {
        "left_identities": tuple(x.element for x in elements if x.left_identity),
        "right_identities": tuple(x.element for x in elements if x.right_identity),
        "identities": tuple(x.element for x in elements if x.identity),
        "left_zeros": tuple(x.element for x in elements if x.left_zero),
        "right_zeros": tuple(x.element for x in elements if x.right_zero),
        "zeros": tuple(x.element for x in elements if x.zero),
        "idempotents": tuple(x.element for x in elements if x.idempotent),
        "nilpotent_elements": tuple(
            x.element for x in elements if x.nilpotence_index is not None
        ),
    }
    value = object.__new__(FiniteElementAnalysis)
    for field, item in (
        ("core", core),
        ("outputs", outputs),
        ("equivalence", equivalence),
        ("elements", elements),
        ("colors", partition.colors),
        *lists.items(),
        (
            "left_identity_status",
            _presence_status(lists["left_identities"], supported=binary),
        ),
        (
            "right_identity_status",
            _presence_status(lists["right_identities"], supported=binary),
        ),
        ("identity_status", _presence_status(lists["identities"], supported=binary)),
        ("left_zero_status", _presence_status(lists["left_zeros"], supported=binary)),
        ("right_zero_status", _presence_status(lists["right_zeros"], supported=binary)),
        ("zero_status", _presence_status(lists["zeros"], supported=binary)),
        (
            "nilpotence_status",
            "computed_left_associated_bounded"
            if zero is not None
            else "unsupported_not_declared",
        ),
        ("nilpotence_zero", zero),
        ("max_nilpotence_index", bound),
        ("examined_element_count", core.carrier_size),
        ("complete", True),
        ("semantic_hash", SemanticHash("sha256", "0" * 64)),
    ):
        object.__setattr__(value, field, item)
    object.__setattr__(value, "semantic_hash", _semantic_hash(_element_body(value)))
    return value


def _element_body(value: FiniteElementAnalysis) -> dict[str, object]:
    return {
        "arity": value.core.arity,
        "carrierSize": value.core.carrier_size,
        "colors": list(value.colors),
        "complete": value.complete,
        "elements": [
            {
                "color": item.color,
                "element": item.element,
                "idempotent": item.idempotent,
                "identity": item.identity,
                "leftIdentity": item.left_identity,
                "leftZero": item.left_zero,
                "nilpotenceIndex": item.nilpotence_index,
                "rightIdentity": item.right_identity,
                "rightZero": item.right_zero,
                "zero": item.zero,
            }
            for item in value.elements
        ],
        "examinedElementCount": value.examined_element_count,
        "identities": list(value.identities),
        "identityStatus": value.identity_status,
        "idempotents": list(value.idempotents),
        "leftIdentities": list(value.left_identities),
        "leftIdentityStatus": value.left_identity_status,
        "leftZeroStatus": value.left_zero_status,
        "leftZeros": list(value.left_zeros),
        "maxNilpotenceIndex": value.max_nilpotence_index,
        "nilpotenceStatus": value.nilpotence_status,
        "nilpotenceZero": value.nilpotence_zero,
        "nilpotentElements": list(value.nilpotent_elements),
        "outputs": list(value.outputs),
        "rightIdentities": list(value.right_identities),
        "rightIdentityStatus": value.right_identity_status,
        "rightZeroStatus": value.right_zero_status,
        "rightZeros": list(value.right_zeros),
        "schemaType": "anyalgebra.census.finite_element_analysis",
        "schemaVersion": _SCHEMA_VERSION,
        "zeroStatus": value.zero_status,
        "zeros": list(value.zeros),
    }


def element_profile_record(value: FiniteElementAnalysis) -> dict[str, object]:
    if type(value) is not FiniteElementAnalysis:
        raise FiniteLawAnalysisError(
            field="element_analysis", reason="must be exact FiniteElementAnalysis"
        )
    expected = analyze_finite_elements(
        value.core,
        value.outputs,
        value.equivalence,
        nilpotence_zero=value.nilpotence_zero,
        max_nilpotence_index=value.max_nilpotence_index,
    )
    if value != expected:
        raise FiniteLawAnalysisError(field="element_analysis", reason="content drift")
    body = _element_body(value)
    semantic_hash = _semantic_hash(body)
    if semantic_hash != value.semantic_hash:
        raise FiniteLawAnalysisError(field="semantic_hash", reason="content drift")
    return {**body, "contentHash": _hash_record(semantic_hash)}


def element_profile_canonical_bytes(value: FiniteElementAnalysis) -> bytes:
    return _encoded(element_profile_record(value))


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class FiniteMagmaSubobjectAnalysis:
    """Complete carrier-level nuclei, commutant, and center evidence."""

    core: CensusSpecCore
    outputs: tuple[int, ...]
    status: str
    left_nucleus: tuple[int, ...]
    middle_nucleus: tuple[int, ...]
    right_nucleus: tuple[int, ...]
    nucleus: tuple[int, ...]
    commutant: tuple[int, ...]
    center: tuple[int, ...]
    associator_assignments_checked: int
    commutator_pairs_checked: int
    complete: bool
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise FiniteLawAnalysisError(
            field="magma_subobjects", reason="is factory-owned"
        )

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("FiniteMagmaSubobjectAnalysis cannot be subclassed")

    def __eq__(self, other: object) -> bool:
        return type(other) is FiniteMagmaSubobjectAnalysis and (
            self.core,
            self.outputs,
            self.status,
            self.left_nucleus,
            self.middle_nucleus,
            self.right_nucleus,
            self.nucleus,
            self.commutant,
            self.center,
            self.associator_assignments_checked,
            self.commutator_pairs_checked,
            self.complete,
            self.semantic_hash,
        ) == (
            other.core,
            other.outputs,
            other.status,
            other.left_nucleus,
            other.middle_nucleus,
            other.right_nucleus,
            other.nucleus,
            other.commutant,
            other.center,
            other.associator_assignments_checked,
            other.commutator_pairs_checked,
            other.complete,
            other.semantic_hash,
        )


def _validated_finite_table(
    core: object, outputs: object
) -> tuple[CensusSpecCore, tuple[int, ...]]:
    if type(core) is not CensusSpecCore:
        raise FiniteLawAnalysisError(
            field="core", reason="must be exact CensusSpecCore"
        )
    if core.candidate_count == 0:
        raise FiniteLawAnalysisError(
            field="outputs", reason="core has no operation tables"
        )
    if type(outputs) is not tuple:
        raise FiniteLawAnalysisError(field="outputs", reason="must be exact tuple")
    try:
        rank_operation_table(core, outputs)
    except OperationTableCodeError as error:
        raise FiniteLawAnalysisError(field="outputs", reason="invalid table") from error
    return core, outputs


def analyze_finite_magma_subobjects(
    core: CensusSpecCore, outputs: tuple[int, ...]
) -> FiniteMagmaSubobjectAnalysis:
    """Exhaust the defining carrier equations for binary magma subobjects."""
    checked_core, checked_outputs = _validated_finite_table(core, outputs)
    size = checked_core.carrier_size
    binary = checked_core.arity == 2
    left_flags = [binary for _ in range(size)]
    middle_flags = [binary for _ in range(size)]
    right_flags = [binary for _ in range(size)]
    commute_flags = [binary for _ in range(size)]
    associator_checks = 0
    commutator_checks = 0
    if binary:
        for candidate in range(size):
            for first in range(size):
                for second in range(size):
                    first_second = _apply(checked_outputs, size, first, second)
                    left_flags[candidate] &= _apply(
                        checked_outputs,
                        size,
                        _apply(checked_outputs, size, candidate, first),
                        second,
                    ) == _apply(checked_outputs, size, candidate, first_second)
                    middle_flags[candidate] &= _apply(
                        checked_outputs,
                        size,
                        _apply(checked_outputs, size, first, candidate),
                        second,
                    ) == _apply(
                        checked_outputs,
                        size,
                        first,
                        _apply(checked_outputs, size, candidate, second),
                    )
                    right_flags[candidate] &= _apply(
                        checked_outputs, size, first_second, candidate
                    ) == _apply(
                        checked_outputs,
                        size,
                        first,
                        _apply(checked_outputs, size, second, candidate),
                    )
                    associator_checks += 3
                commute_flags[candidate] &= _apply(
                    checked_outputs, size, candidate, first
                ) == _apply(checked_outputs, size, first, candidate)
                commutator_checks += 1
    left = tuple(item for item, keep in enumerate(left_flags) if keep)
    middle = tuple(item for item, keep in enumerate(middle_flags) if keep)
    right = tuple(item for item, keep in enumerate(right_flags) if keep)
    commutant = tuple(item for item, keep in enumerate(commute_flags) if keep)
    nucleus = tuple(
        item
        for item in range(size)
        if left_flags[item] and middle_flags[item] and right_flags[item]
    )
    center = tuple(item for item in nucleus if commute_flags[item])
    value = object.__new__(FiniteMagmaSubobjectAnalysis)
    for field, item in (
        ("core", checked_core),
        ("outputs", checked_outputs),
        ("status", "computed_complete" if binary else "unsupported_nonbinary"),
        ("left_nucleus", left),
        ("middle_nucleus", middle),
        ("right_nucleus", right),
        ("nucleus", nucleus),
        ("commutant", commutant),
        ("center", center),
        ("associator_assignments_checked", associator_checks),
        ("commutator_pairs_checked", commutator_checks),
        ("complete", binary),
        ("semantic_hash", SemanticHash("sha256", "0" * 64)),
    ):
        object.__setattr__(value, field, item)
    object.__setattr__(value, "semantic_hash", _semantic_hash(_magma_body(value)))
    return value


def _magma_body(value: FiniteMagmaSubobjectAnalysis) -> dict[str, object]:
    return {
        "arity": value.core.arity,
        "associatorAssignmentsChecked": value.associator_assignments_checked,
        "carrierSize": value.core.carrier_size,
        "center": list(value.center),
        "commutant": list(value.commutant),
        "commutatorPairsChecked": value.commutator_pairs_checked,
        "complete": value.complete,
        "leftNucleus": list(value.left_nucleus),
        "middleNucleus": list(value.middle_nucleus),
        "nucleus": list(value.nucleus),
        "outputs": list(value.outputs),
        "rightNucleus": list(value.right_nucleus),
        "schemaType": "anyalgebra.census.finite_magma_subobject_analysis",
        "schemaVersion": _SCHEMA_VERSION,
        "status": value.status,
    }


def finite_magma_subobject_record(
    value: FiniteMagmaSubobjectAnalysis,
) -> dict[str, object]:
    if type(value) is not FiniteMagmaSubobjectAnalysis:
        raise FiniteLawAnalysisError(
            field="magma_subobjects",
            reason="must be exact FiniteMagmaSubobjectAnalysis",
        )
    expected = analyze_finite_magma_subobjects(value.core, value.outputs)
    if value != expected:
        raise FiniteLawAnalysisError(field="magma_subobjects", reason="content drift")
    body = _magma_body(value)
    semantic_hash = _semantic_hash(body)
    if semantic_hash != value.semantic_hash:
        raise FiniteLawAnalysisError(field="semantic_hash", reason="content drift")
    return {**body, "contentHash": _hash_record(semantic_hash)}


def finite_magma_subobject_canonical_bytes(
    value: FiniteMagmaSubobjectAnalysis,
) -> bytes:
    return _encoded(finite_magma_subobject_record(value))


__all__ = (
    "MAX_FINITE_LAW_ASSIGNMENTS",
    "DisprovedLaw",
    "FiniteElementAnalysis",
    "FiniteElementProfile",
    "FiniteLawAnalysisError",
    "FiniteLawProfile",
    "FiniteMagmaSubobjectAnalysis",
    "LawResult",
    "LawWitness",
    "ProvedLaw",
    "UnsupportedLaw",
    "analyze_finite_elements",
    "analyze_finite_laws",
    "analyze_finite_magma_subobjects",
    "element_profile_canonical_bytes",
    "element_profile_record",
    "finite_magma_subobject_canonical_bytes",
    "finite_magma_subobject_record",
    "law_profile_canonical_bytes",
    "law_profile_record",
)
