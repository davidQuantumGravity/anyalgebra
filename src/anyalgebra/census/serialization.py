"""Strict canonical codec and content addressing for :class:`CensusSpec`."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from typing import NoReturn, cast

from anyalgebra.core.parents import SemanticHash
from anyalgebra.persistence.canonical import canonical_json
from anyalgebra.persistence.registry import JSONValue, SchemaError, SerializerRegistry

from .bounds import CensusOrdering, EnumerationBounds
from .constraints import CensusConstraint, CensusTerm, ConstraintSet
from .equivalence import EquivalencePolicy, SortBlock
from .spec import CensusSpec, CensusSpecCore


_SCHEMA_TYPE = "anyalgebra.census.spec"
_SCHEMA_VERSION = 1
_MAX_BYTES = 1_048_576
_MAX_SEQUENCE = 256
_TOP_LEVEL_KEYS = frozenset(
    (
        "schemaType",
        "schemaVersion",
        "algorithmContractVersion",
        "arity",
        "bounds",
        "carrierSize",
        "constraints",
        "contentHash",
        "conventionRefs",
        "corpusName",
        "distinguishedElements",
        "equivalence",
        "ordering",
    )
)


def _hash_record(value: SemanticHash) -> dict[str, JSONValue]:
    record: dict[str, JSONValue] = {
        "algorithm": value.algorithm,
        "digest": value.digest,
    }
    return record


def _term_record(value: CensusTerm | None) -> dict[str, JSONValue] | None:
    if value is None:
        return None
    return {
        "arguments": [
            cast(dict[str, JSONValue], _term_record(argument))
            for argument in value.arguments
        ],
        "index": value.index,
        "kind": value.kind,
    }


def _constraint_record(value: CensusConstraint) -> dict[str, JSONValue]:
    return {
        "kind": value.kind,
        "left": _term_record(value.left),
        "parameters": [
            {"name": name, "value": item} for name, item in value.parameters
        ],
        "right": _term_record(value.right),
    }


def _semantic_body(value: CensusSpec) -> dict[str, JSONValue]:
    """Return every defining axis, excluding only the derived content hash."""
    return {
        "algorithmContractVersion": value.algorithm_contract_version,
        "arity": value.arity,
        "bounds": {
            "maxCandidates": value.bounds.max_candidates,
            "maxMemoryBytes": value.bounds.max_memory_bytes,
            "maxObservedMilliseconds": value.bounds.max_observed_milliseconds,
            "maxOrbits": value.bounds.max_orbits,
            "maxWorkUnits": value.bounds.max_work_units,
        },
        "carrierSize": value.carrier_size,
        "constraints": [
            _constraint_record(constraint)
            for constraint in value.constraints.constraints
        ],
        "conventionRefs": [
            _hash_record(reference) for reference in value.convention_refs
        ],
        "corpusName": value.core.corpus_name,
        "distinguishedElements": [
            {"element": element, "name": name}
            for name, element in value.core.distinguished_elements
        ],
        "equivalence": {
            "fixedElements": list(value.equivalence.fixed_elements),
            "kind": value.equivalence.kind,
            "sortBlocks": [
                {"elements": list(block.elements), "name": block.name}
                for block in value.equivalence.sort_blocks
            ],
        },
        "ordering": {
            "candidateOrder": value.ordering.candidate_order,
            "constraintOrder": value.ordering.constraint_order,
            "inputTupleOrder": value.ordering.input_tuple_order,
            "outputDigitSignificance": value.ordering.output_digit_significance,
            "permutationOrder": value.ordering.permutation_order,
            "schemaVersion": value.ordering.schema_version,
        },
    }


def _semantic_bytes(value: CensusSpec) -> bytes:
    record: dict[str, JSONValue] = {
        "schemaType": _SCHEMA_TYPE,
        "schemaVersion": _SCHEMA_VERSION,
        **_semantic_body(value),
    }
    return json.dumps(
        record,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def census_spec_semantic_hash(value: CensusSpec) -> SemanticHash:
    """Hash the schema-tagged semantic body, excluding its derived hash field."""
    if type(value) is not CensusSpec:
        raise SchemaError("invalid_census_spec")
    return SemanticHash("sha256", hashlib.sha256(_semantic_bytes(value)).hexdigest())


def _encode(value: CensusSpec) -> dict[str, JSONValue]:
    if type(value) is not CensusSpec:
        raise SchemaError("invalid_census_spec")
    if census_spec_semantic_hash(value) != value.semantic_hash:
        raise SchemaError("census_spec_hash_drift")
    return {**_semantic_body(value), "contentHash": _hash_record(value.semantic_hash)}


def _mapping(value: object, keys: frozenset[str], field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(field)
    return value


def _sequence(value: object, field: str) -> tuple[object, ...]:
    if type(value) is not tuple or len(value) > _MAX_SEQUENCE:
        raise ValueError(field)
    return value


def _hash_value(value: object, field: str) -> SemanticHash:
    record = _mapping(value, frozenset(("algorithm", "digest")), field)
    return SemanticHash(record["algorithm"], record["digest"])  # type: ignore[arg-type]


def _term(value: object, field: str) -> CensusTerm | None:
    if value is None:
        return None
    record = _mapping(value, frozenset(("arguments", "index", "kind")), field)
    kind = record["kind"]
    index = record["index"]
    arguments = tuple(
        _term(item, field) for item in _sequence(record["arguments"], field)
    )
    if any(argument is None for argument in arguments):
        raise ValueError(field)
    checked_arguments = cast(tuple[CensusTerm, ...], arguments)
    if kind == "variable" and type(index) is int and not arguments:
        return CensusTerm.variable(index)
    if kind == "constant" and type(index) is int and not arguments:
        return CensusTerm.constant(index)
    if kind == "apply" and index is None:
        return CensusTerm.apply(*checked_arguments)
    raise ValueError(field)


def _parameters(value: object) -> tuple[tuple[str, int | str], ...]:
    result: list[tuple[str, int | str]] = []
    for item in _sequence(value, "parameters"):
        record = _mapping(item, frozenset(("name", "value")), "parameters")
        name, parameter = record["name"], record["value"]
        if type(name) is not str or type(parameter) not in (int, str):
            raise ValueError("parameters")
        result.append((name, cast(int | str, parameter)))
    return tuple(result)


def _constraint(value: object) -> CensusConstraint:
    record = _mapping(
        value,
        frozenset(("kind", "left", "parameters", "right")),
        "constraints",
    )
    kind = record["kind"]
    parameters = _parameters(record["parameters"])
    left = _term(record["left"], "left")
    right = _term(record["right"], "right")
    values = dict(parameters)
    element = values.get("element")
    side = values.get("side")
    if (
        kind == "nullary_value"
        and parameters == (("element", element),)
        and type(element) is int
        and left is None
        and right is None
    ):
        return CensusConstraint.nullary_value(element=element)
    if (
        kind == "identity"
        and parameters == (("element", element), ("side", side))
        and type(element) is int
        and type(side) is str
        and left is None
        and right is None
    ):
        return CensusConstraint.identity(element=element, side=side)
    if kind == "commutative" and not parameters and left is None and right is None:
        return CensusConstraint.commutative()
    if kind == "idempotent" and not parameters and left is None and right is None:
        return CensusConstraint.idempotent()
    if kind == "quasigroup" and not parameters and left is None and right is None:
        return CensusConstraint.quasigroup()
    variable_count = values.get("variable_count")
    if (
        kind == "equation"
        and parameters == (("variable_count", variable_count),)
        and type(variable_count) is int
        and left is not None
        and right is not None
    ):
        return CensusConstraint.equation(left, right, variable_count=variable_count)
    raise ValueError("constraints")


def _core(record: Mapping[str, object]) -> CensusSpecCore:
    points = tuple(
        (
            _mapping(item, frozenset(("element", "name")), "distinguishedElements")[
                "name"
            ],
            _mapping(item, frozenset(("element", "name")), "distinguishedElements")[
                "element"
            ],
        )
        for item in _sequence(record["distinguishedElements"], "distinguishedElements")
    )
    return CensusSpecCore.create(
        carrier_size=record["carrierSize"],  # type: ignore[arg-type]
        arity=record["arity"],  # type: ignore[arg-type]
        distinguished_elements=points,  # type: ignore[arg-type]
        corpus_name=record["corpusName"],  # type: ignore[arg-type]
    )


def _equivalence(value: object, core: CensusSpecCore) -> EquivalencePolicy:
    record = _mapping(
        value, frozenset(("fixedElements", "kind", "sortBlocks")), "equivalence"
    )
    blocks = tuple(
        SortBlock.create(
            name=_mapping(item, frozenset(("elements", "name")), "sortBlocks")["name"],  # type: ignore[arg-type]
            elements=_sequence(
                _mapping(item, frozenset(("elements", "name")), "sortBlocks")[
                    "elements"
                ],
                "sortBlocks",
            ),  # type: ignore[arg-type]
        )
        for item in _sequence(record["sortBlocks"], "sortBlocks")
    )
    if record["kind"] == "literal":
        result = EquivalencePolicy.literal(core)
    elif record["kind"] == "carrier_relabeling":
        result = EquivalencePolicy.relabeling(
            core,
            fixed_elements=_sequence(record["fixedElements"], "fixedElements"),  # type: ignore[arg-type]
            sort_blocks=blocks,
        )
    else:
        raise ValueError("equivalence")
    if result.fixed_elements != _sequence(record["fixedElements"], "fixedElements"):
        raise ValueError("equivalence")
    if result.sort_blocks != blocks:
        raise ValueError("equivalence")
    return result


def _ordering(value: object) -> CensusOrdering:
    keys = frozenset(
        (
            "candidateOrder",
            "constraintOrder",
            "inputTupleOrder",
            "outputDigitSignificance",
            "permutationOrder",
            "schemaVersion",
        )
    )
    record = _mapping(value, keys, "ordering")
    reference = CensusOrdering.reference()
    expected: dict[str, object] = {
        "candidateOrder": reference.candidate_order,
        "constraintOrder": reference.constraint_order,
        "inputTupleOrder": reference.input_tuple_order,
        "outputDigitSignificance": reference.output_digit_significance,
        "permutationOrder": reference.permutation_order,
        "schemaVersion": reference.schema_version,
    }
    if dict(record) != expected:
        raise ValueError("ordering")
    return reference


def _bounds(value: object) -> EnumerationBounds:
    record = _mapping(
        value,
        frozenset(
            (
                "maxCandidates",
                "maxMemoryBytes",
                "maxObservedMilliseconds",
                "maxOrbits",
                "maxWorkUnits",
            )
        ),
        "bounds",
    )
    return EnumerationBounds.create(
        max_candidates=record["maxCandidates"],  # type: ignore[arg-type]
        max_orbits=record["maxOrbits"],  # type: ignore[arg-type]
        max_work_units=record["maxWorkUnits"],  # type: ignore[arg-type]
        max_memory_bytes=record["maxMemoryBytes"],  # type: ignore[arg-type]
        max_observed_milliseconds=record["maxObservedMilliseconds"],  # type: ignore[arg-type]
    )


def _parse(value: object) -> CensusSpec:
    record = _mapping(value, _TOP_LEVEL_KEYS, "spec")
    if record["algorithmContractVersion"] != "anyalgebra.census.reference.v1":
        raise ValueError("algorithmContractVersion")
    core = _core(record)
    constraints = ConstraintSet.create(
        core,
        tuple(
            _constraint(item)
            for item in _sequence(record["constraints"], "constraints")
        ),
    )
    result = CensusSpec.create(
        carrier_size=record["carrierSize"],  # type: ignore[arg-type]
        arity=record["arity"],  # type: ignore[arg-type]
        constraints=constraints,
        equivalence=_equivalence(record["equivalence"], core),
        ordering=_ordering(record["ordering"]),
        bounds=_bounds(record["bounds"]),
        convention_refs=tuple(
            _hash_value(item, "conventionRefs")
            for item in _sequence(record["conventionRefs"], "conventionRefs")
        ),
    )
    if _hash_value(record["contentHash"], "contentHash") != result.semantic_hash:
        raise ValueError("contentHash")
    return result


_REGISTRY = SerializerRegistry().with_codec(
    _SCHEMA_TYPE, _SCHEMA_VERSION, CensusSpec, _encode, _parse
)


def census_spec_registry() -> SerializerRegistry:
    """Return the immutable allow-list containing only the census-spec codec."""
    return _REGISTRY


def census_spec_record(value: CensusSpec) -> dict[str, object]:
    """Return a fresh, versioned, content-addressed record."""
    return cast(dict[str, object], _REGISTRY.to_record(value))


def census_spec_from_record(record: object) -> CensusSpec:
    """Strictly parse one allow-listed record without imports or callbacks."""
    return cast(CensusSpec, _REGISTRY.from_record(record))


def census_spec_canonical_bytes(value: CensusSpec) -> bytes:
    """Encode one spec as canonical UTF-8 JSON including its verified hash."""
    return canonical_json(value, registry=_REGISTRY)


def _reject_number(_: str) -> NoReturn:
    raise ValueError("nonexact number")


def _parse_integer(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        raise ValueError("integer limit")
    return int(value)


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def census_spec_from_canonical_bytes(data: object) -> CensusSpec:
    """Parse only exact canonical bytes and reject malformed or altered records."""
    if type(data) is not bytes or len(data) > _MAX_BYTES:
        raise SchemaError("invalid_census_spec_bytes")
    try:
        decoded = data.decode("utf-8")
        record = json.loads(
            decoded,
            object_pairs_hook=_object,
            parse_constant=_reject_number,
            parse_float=_reject_number,
            parse_int=_parse_integer,
        )
    except Exception:
        failed = True
    else:
        failed = False
    if failed:
        raise SchemaError("invalid_census_spec_bytes")
    if type(record) is not dict:
        raise SchemaError("invalid_census_spec_bytes")
    result = census_spec_from_record(record)
    if census_spec_canonical_bytes(result) != data:
        raise SchemaError("noncanonical_census_spec")
    return result


__all__ = (
    "census_spec_canonical_bytes",
    "census_spec_from_canonical_bytes",
    "census_spec_from_record",
    "census_spec_record",
    "census_spec_registry",
)
