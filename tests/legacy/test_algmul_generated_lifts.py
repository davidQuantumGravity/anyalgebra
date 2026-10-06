"""Deterministic, non-oracular evidence checks for generated AlgMul lifts."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "legacy" / "generated-lifts.json"
_RUNTIME = Path(__file__).parents[1] / "fixtures" / "legacy" / "runtime-load.json"
_SOURCE_PATH = "${ANYALGEBRA_ALGMUL_SOURCE}"
_SOURCE = {
    "path": _SOURCE_PATH,
    "sha256": "3A13C9F47092B5B9814AA08873FF4F1DE61D4635DF68FC7F3A039A7DA5B0D00D",
    "bytes": 194012,
}
_RECIPE_SHA256 = "18ECB90B72F225E0C8B6AB5EDE4181295324D7FED7511AD17DC7F8F6F85B0614"
_EMPTY_DEFINITION_HASH = (
    "9220276CACAEAC75CFD1905068507388D617C29233DB39CFD3062E71144A4E01"
)
_ALG_BEFORE_HASH = "75568E6BB1A75B98FD0A56C2BE3340CC2CAD6E1F6B27CFA47B4DD9BA4CA7359B"
_COMPLEX_TABLE = [
    [[[1, 0], [0, 1]], [[0, 1], [-1, 0]]],
    [[[1, 0], [0, 1]], [[0, 1], [-1, 0]]],
]
_SUBSTITUTION = {
    "ACTION": (
        "replace the sole {{ACTION}} token with scenario.action as Wolfram "
        "Language input"
    ),
    "TARGETS": (
        "replace the sole {{TARGETS}} token with a Wolfram list of quoted "
        "scenario.targets"
    ),
    "EXTRACT": (
        "replace the sole {{EXTRACT}} token with scenario.extract as Wolfram "
        "Language input"
    ),
}
_HEX = frozenset("0123456789ABCDEF")
_FUNCTIONS = (
    "AlgMul`MakeChar",
    "AlgMul`MakeTMul",
    "AlgMul`TMul",
    "AlgMul`MMul",
    "AlgMul`MTMul",
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, indent=2).encode("utf-8") + b"\n"


def _render_recipe(template: object, scenario: object) -> str:
    if type(template) is not str or type(scenario) is not dict:
        raise ValueError("recipe renderer received an invalid input")
    action = scenario.get("action")
    targets = scenario.get("targets")
    extract = scenario.get("extract")
    if (
        type(action) is not str
        or type(targets) is not list
        or any(type(item) is not str for item in targets)
        or type(extract) is not str
    ):
        raise ValueError("recipe renderer received an invalid scenario")
    rendered = template
    wl_targets = (
        "{" + ",".join(json.dumps(item, ensure_ascii=True) for item in targets) + "}"
    )
    for token, replacement in (
        ("{{ACTION}}", action),
        ("{{TARGETS}}", wl_targets),
        ("{{EXTRACT}}", extract),
    ):
        if rendered.count(token) != 1:
            raise ValueError("recipe template placeholder is invalid")
        rendered = rendered.replace(token, replacement)
    if any(token in rendered for token in ("{{ACTION}}", "{{TARGETS}}", "{{EXTRACT}}")):
        raise ValueError("recipe renderer left a placeholder")
    return rendered


def _fixture() -> dict[str, Any]:
    value = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def _runtime() -> dict[str, Any]:
    value = json.loads(_RUNTIME.read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def _mapping(value: object, keys: set[str], field: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{field} has an invalid shape")
    return value


def _hash(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{field} must be uppercase SHA-256")
    return value


def _count(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _coordinates(value: object, field: str) -> list[list[int]]:
    if (
        type(value) is not list
        or len(value) != 2
        or any(
            type(row) is not list
            or len(row) != 2
            or any(type(item) is not int for item in row)
            for row in value
        )
    ):
        raise ValueError(f"{field} must be a 2x2 integer coordinate array")
    return value


def _symbol(value: object, expected_name: str) -> None:
    record = _mapping(
        value,
        {
            "name",
            "context",
            "attributes",
            "ownValueCount",
            "downValueCount",
            "upValueCount",
            "subValueCount",
            "definitionHash",
        },
        "generated symbol",
    )
    if record["name"] != expected_name or not expected_name.startswith(
        record["context"]
    ):
        raise ValueError("generated symbol identity is invalid")
    if record["attributes"] != []:
        raise ValueError("generated symbol attributes are invalid")
    for name in ("ownValueCount", "downValueCount", "upValueCount", "subValueCount"):
        if _count(record[name], f"generated symbol {name}") != 0:
            raise ValueError("generated symbol must be undefined")
    if (
        _hash(record["definitionHash"], "generated symbol definition hash")
        != _EMPTY_DEFINITION_HASH
    ):
        raise ValueError("generated symbol definition hash is invalid")


def _validate(value: object) -> dict[str, Any]:
    root = _mapping(
        value,
        {
            "schemaType",
            "schemaVersion",
            "source",
            "kernel",
            "captureRecipe",
            "scenarioContextMutations",
            "load",
            "definitions",
            "makeChar",
            "makeTMul",
            "lifts",
            "independentOracle",
        },
        "fixture",
    )
    if (
        root["schemaType"] != "AnyAlgebra.AlgMulGeneratedLifts"
        or root["schemaVersion"] != "2"
    ):
        raise ValueError("fixture schema is invalid")
    if _mapping(root["source"], set(_SOURCE), "source") != _SOURCE:
        raise ValueError("source binding is invalid")
    kernel = _mapping(root["kernel"], {"versionNumber", "freshProcess"}, "kernel")
    if kernel != {"versionNumber": "12.0", "freshProcess": True}:
        raise ValueError("kernel binding is invalid")
    recipe = _mapping(
        root["captureRecipe"],
        {
            "schema",
            "freshProcessPerScenario",
            "adapter",
            "template",
            "scenarios",
            "sha256",
            "substitution",
        },
        "capture recipe",
    )
    recipe_body = {name: item for name, item in recipe.items() if name != "sha256"}
    recipe_bytes = json.dumps(
        recipe_body, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    expected_scenarios = [
        ("load", "Null", 5, ("registryNames", "makePropertyPresent")),
        ("make-char", 'AlgMul`MakeChar[3,"Q9","q"]', 4, ("route", "chars")),
        (
            "same-maketmul",
            'AlgMul`MakeTMul["C","C","CxC"]',
            0,
            ("basis", "structureDotHash"),
        ),
        (
            "cross-maketmul",
            'AlgMul`MakeTMul["C","H2","AuditCxH2"]',
            4,
            ("basis", "structureDotHash"),
        ),
        ("lifts", None, 0, ("TMulCoordinateCxH", "MTMul", "MMul")),
    ]
    if (
        recipe["schema"] != "AnyAlgebra.AlgMulGeneratedLiftsRecipe.v1"
        or recipe["freshProcessPerScenario"] is not True
        or recipe["adapter"]
        != {
            "module": "anyalgebra.legacy.mathematica.MathematicaAdapter",
            "requiredVersion": "12.0",
            "timeoutSeconds": 120,
            "maxOutputBytes": 1_048_576,
        }
        or recipe["sha256"] != _RECIPE_SHA256
        or hashlib.sha256(recipe_bytes).hexdigest().upper() != _RECIPE_SHA256
        or type(recipe["template"]) is not str
        or "{{ACTION}}" not in recipe["template"]
        or "{{TARGETS}}" not in recipe["template"]
        or "{{EXTRACT}}" not in recipe["template"]
        or "ExportString" not in recipe["template"]
        or 'v86zzSha[""];v86zzSnapshot[]' not in recipe["template"]
        or "v86zzBefore=v86zzSnapshot[]" not in recipe["template"]
        or "v86zzAfter=v86zzSnapshot[]" not in recipe["template"]
        or type(recipe["scenarios"]) is not list
        or len(recipe["scenarios"]) != len(expected_scenarios)
        or recipe["substitution"] != _SUBSTITUTION
    ):
        raise ValueError("capture recipe binding is invalid")
    for scenario, (identifier, action, target_count, evidence_keys) in zip(
        recipe["scenarios"], expected_scenarios, strict=True
    ):
        if (
            type(scenario) is not dict
            or set(scenario) != {"id", "action", "targets", "extract"}
            or scenario["id"] != identifier
            or (action is not None and scenario["action"] != action)
            or (identifier == "lifts" and "AlgMul`MTMul" not in scenario["action"])
            or type(scenario["targets"]) is not list
            or len(scenario["targets"]) != target_count
            or type(scenario["extract"]) is not str
            or any(key not in scenario["extract"] for key in evidence_keys)
        ):
            raise ValueError("capture recipe scenarios are invalid")
    context_mutations = _mapping(
        root["scenarioContextMutations"],
        {"load", "make-char", "same-maketmul", "cross-maketmul", "lifts"},
        "scenario context mutations",
    )
    expected_context_mutations = {
        "load": {"added": []},
        "make-char": {"added": []},
        "same-maketmul": {"added": []},
        "cross-maketmul": {"added": ["AlgMul`e1AlgMul`"]},
        "lifts": {"added": []},
    }
    if context_mutations != expected_context_mutations:
        raise ValueError("scenario context mutations are invalid")
    load = _mapping(
        root["load"],
        {
            "registryNames",
            "messageTag",
            "sourceLine",
            "rawMessageSha256",
            "makePropertyPresent",
        },
        "load",
    )
    runtime = _runtime()
    raw_message = runtime["load"]["messages"][0]
    if (
        load["registryNames"]
        != [item["name"] for item in runtime["registries"]["algebras"]]
        or load["messageTag"] != "Syntax::sntx"
        or load["sourceLine"] != 1414
        or load["rawMessageSha256"]
        != hashlib.sha256(raw_message.encode("utf-8")).hexdigest().upper()
        or load["makePropertyPresent"] is not False
    ):
        raise ValueError("load evidence is invalid")
    definitions = root["definitions"]
    if type(definitions) is not list or len(definitions) != len(_FUNCTIONS):
        raise ValueError("function definitions are incomplete")
    fields = {
        "name",
        "attributes",
        "ownValueCount",
        "downValueCount",
        "upValueCount",
        "subValueCount",
        "definitionHash",
    }
    expected = {
        item["name"]: {field: item[field] for field in fields}
        for item in runtime["symbols"]
        if item["name"] in _FUNCTIONS
    }
    observed: dict[str, dict[str, Any]] = {}
    for record in definitions:
        item = _mapping(record, fields, "definition")
        name = item["name"]
        if type(name) is not str or name in observed:
            raise ValueError("function definitions are duplicate")
        if type(item["attributes"]) is not list or any(
            type(attribute) is not str for attribute in item["attributes"]
        ):
            raise ValueError("definition attributes are invalid")
        for count_name in (
            "ownValueCount",
            "downValueCount",
            "upValueCount",
            "subValueCount",
        ):
            _count(item[count_name], f"definition {count_name}")
        _hash(item["definitionHash"], "definition hash")
        observed[name] = item
    if observed != expected:
        raise ValueError("function definitions disagree with runtime load")
    make_char = _mapping(
        root["makeChar"],
        {
            "operation",
            "arguments",
            "result",
            "messages",
            "globalMutation",
            "contextMutation",
            "registryMutation",
            "symbols",
            "algDefinition",
            "route",
        },
        "MakeChar",
    )
    if (
        make_char["operation"] != "MakeChar"
        or make_char["arguments"] != [3, "Q9", "q"]
        or make_char["result"] != ["1", "q1", "q2"]
        or make_char["messages"] != []
    ):
        raise ValueError("MakeChar observation is invalid")
    if make_char["registryMutation"] != {
        "beforeCount": 20,
        "added": ["Q9"],
        "afterCount": 21,
    }:
        raise ValueError("MakeChar registry mutation is invalid")
    if make_char["globalMutation"] != {
        "before": [],
        "added": ["Q", "QMul", "q1", "q2"],
        "after": ["Q", "QMul", "q1", "q2"],
    }:
        raise ValueError("MakeChar global mutation is invalid")
    if make_char["contextMutation"] != context_mutations["make-char"]:
        raise ValueError("MakeChar context mutation is invalid")
    symbols = make_char["symbols"]
    names = ["Global`Q", "Global`q1", "Global`q2", "Global`QMul"]
    if type(symbols) is not list or len(symbols) != len(names):
        raise ValueError("MakeChar symbols are invalid")
    for record, name in zip(symbols, names, strict=True):
        _symbol(record, name)
    _alg_hashes(
        make_char["algDefinition"],
        "MakeChar",
        "415C8C6D0C49B99C17C2A78C109B179724B9B2C5042AC813566D780C1745F315",
    )
    if make_char["route"] != {"value": "Global`QMul", "definitionPresent": False}:
        raise ValueError("MakeChar route is invalid")
    tensor = _mapping(root["makeTMul"], {"sameFactor", "crossFactor"}, "MakeTMul")
    _tensor_record(
        tensor["sameFactor"],
        ["C", "C", "CxC"],
        ["1", "AlgMul`e1", "AlgMul`e1", "AlgMul`e1"],
        [4, 4, 4],
        0,
        "04656E8036AB7CE8323D4209CDE1513E2521819AC428E9C0AFD16521F4324E11",
    )
    _tensor_record(
        tensor["crossFactor"],
        ["C", "H2", "AuditCxH2"],
        [
            "1",
            "AlgMul`f1",
            "AlgMul`f2",
            "AlgMul`f3",
            "AlgMul`e1",
            "AlgMul`e1AlgMul`f1",
            "AlgMul`e1AlgMul`f2",
            "AlgMul`e1AlgMul`f3",
        ],
        [8, 8, 8],
        4,
        "BA629EF0DF5858DA12CA972AF988A39563690154CD77E8730A3ECE0A64F19395",
    )
    lifts = root["lifts"]
    if type(lifts) is not list or len(lifts) != 3:
        raise ValueError("lift observations are invalid")
    _oracle(root["independentOracle"])
    _lifts(lifts, root["independentOracle"])
    return root


def _alg_hashes(value: object, field: str, expected_after: str) -> None:
    record = _mapping(value, {"before", "after"}, f"{field} alg definition")
    if (
        _hash(record["before"], f"{field} before hash") != _ALG_BEFORE_HASH
        or _hash(record["after"], f"{field} after hash") != expected_after
    ):
        raise ValueError(f"{field} alg mutation hashes are invalid")


def _tensor_record(
    value: object,
    arguments: list[str],
    expected_basis: list[str],
    dimensions: list[int],
    symbol_count: int,
    expected_alg_after: str,
) -> None:
    record = _mapping(
        value,
        {
            "operation",
            "arguments",
            "result",
            "messages",
            "globalMutation",
            "contextMutation",
            "registryMutation",
            "basis",
            "structureDot",
            "algDefinition",
            "symbols",
        },
        "MakeTMul record",
    )
    if (
        record["operation"] != "MakeTMul"
        or record["arguments"] != arguments
        or record["result"] != arguments[-1]
        or record["messages"] != []
    ):
        raise ValueError("MakeTMul observation is invalid")
    if record["registryMutation"] != {
        "beforeCount": 20,
        "added": [arguments[-1]],
        "afterCount": 21,
    }:
        raise ValueError("MakeTMul registry mutation is invalid")
    expected_global_mutations = {
        ("C", "C", "CxC"): {"before": [], "added": [], "after": []},
        ("C", "H2", "AuditCxH2"): {
            "before": [],
            "added": ["H"],
            "after": ["H"],
        },
    }
    argument_key: tuple[str, str, str] = (arguments[0], arguments[1], arguments[2])
    if record["globalMutation"] != expected_global_mutations[argument_key]:
        raise ValueError("MakeTMul global mutation is invalid")
    expected_context_mutations = {
        ("C", "C", "CxC"): {"added": []},
        ("C", "H2", "AuditCxH2"): {"added": ["AlgMul`e1AlgMul`"]},
    }
    if record["contextMutation"] != expected_context_mutations[argument_key]:
        raise ValueError("MakeTMul context mutation is invalid")
    structure = _mapping(
        record["structureDot"],
        {"dimensions", "nonzeroEntries", "hash"},
        "structure tensor",
    )
    if structure["dimensions"] != dimensions or structure["nonzeroEntries"] != 0:
        raise ValueError("structure tensor observation is invalid")
    expected_hashes: dict[tuple[str, ...], str] = {
        (
            "C",
            "C",
            "CxC",
        ): "1709960F90E4FBD48672A5F7AC7D6EBC798371B885E84441C246FA57EEB378B2",
        (
            "C",
            "H2",
            "AuditCxH2",
        ): "ED18CFDD36794CFFDE182B3EC97CD3BD6AF17C65AFEEEA787C84538142D349FB",
    }
    expected_hash = expected_hashes[tuple(arguments)]
    if _hash(structure["hash"], "structure tensor hash") != expected_hash:
        raise ValueError("structure tensor hash is invalid")
    _alg_hashes(record["algDefinition"], "MakeTMul", expected_alg_after)
    if record["basis"] != expected_basis:
        raise ValueError("MakeTMul basis is invalid")
    if type(record["symbols"]) is not list or len(record["symbols"]) != symbol_count:
        raise ValueError("MakeTMul symbols are invalid")
    if symbol_count:
        expected_names = [
            "Global`H",
            "AlgMul`e1AlgMul`f1",
            "AlgMul`e1AlgMul`f2",
            "AlgMul`e1AlgMul`f3",
        ]
        for item, expected_name in zip(record["symbols"], expected_names, strict=True):
            _symbol(item, expected_name)


def _lifts(records: list[Any], oracle: Any) -> None:
    first, second, third = records
    if set(first) != {
        "operation",
        "algebras",
        "left",
        "right",
        "result",
        "messages",
        "classification",
    }:
        raise ValueError("TMul lift record shape is invalid")
    if first["operation"] != "TMul" or first["algebras"] != ["C", "H"]:
        raise ValueError("TMul lift identity is invalid")
    if (
        first["left"] != [[0, 0, 0, 0], [0, 1, 0, 0]]
        or first["right"] != first["left"]
        or first["result"] != [[0] * 4, [0] * 4]
        or first["messages"] != []
        or first["classification"] != "legacy-defect"
    ):
        raise ValueError("TMul 2x4 lift is invalid")
    if set(second) != {
        "operation",
        "algebras",
        "left",
        "right",
        "result",
        "messages",
        "classification",
    }:
        raise ValueError("MTMul lift record shape is invalid")
    if second["operation"] != "MTMul" or second["algebras"] != ["C", "C2"]:
        raise ValueError("MTMul lift identity is invalid")
    if (
        second["left"] != oracle["matrixLeft"]
        or second["right"] != oracle["matrixRight"]
        or second["result"] != oracle["legacyObservedMatrix"]
        or second["messages"] != []
        or second["classification"] != "legacy-defect"
    ):
        raise ValueError("MTMul summation lift is invalid")
    if set(third) != {
        "operation",
        "algebras",
        "kernel",
        "left",
        "right",
        "result",
        "messages",
        "classification",
    }:
        raise ValueError("MMul lift record shape is invalid")
    if (
        third["operation"] != "MMul"
        or third["algebras"] != ["H"]
        or third["classification"] != "explicit-kernel-positive"
        or third["kernel"] != "Function[{a,b,ignored},HMul[a,b]]"
        or third["left"] != [[[0, 1, 0, 0], [0, 0, 1, 0]]]
        or third["right"] != [[[0, 1, 0, 0]], [[0, 0, 1, 0]]]
        or third["messages"] != []
    ):
        raise ValueError("explicit MMul lift is invalid")
    if third["result"] != [[[-2, 0, 0, 0]]]:
        raise ValueError("explicit MMul result is invalid")


def _oracle(value: object) -> None:
    record = _mapping(
        value,
        {
            "factorTables",
            "squareInput",
            "squareExpected",
            "matrixLeft",
            "matrixRight",
            "matrixExpected",
            "legacyObservedMatrix",
        },
        "independent oracle",
    )
    tables = record["factorTables"]
    if type(tables) is not list or tables != _COMPLEX_TABLE:
        raise ValueError("oracle factor tables are invalid")
    checked_tables = [_factor_table(table) for table in tables]
    square = _coordinates(record["squareInput"], "oracle square input")
    if (
        _tensor_product_multiply(square, square, checked_tables)
        != record["squareExpected"]
    ):
        raise ValueError("oracle square is invalid")
    if (
        _matrix_tensor_product_multiply(
            record["matrixLeft"], record["matrixRight"], checked_tables
        )
        != record["matrixExpected"]
    ):
        raise ValueError("oracle matrix lift is invalid")
    if (
        record["squareExpected"] != [[1, 0], [0, 0]]
        or record["matrixExpected"] != [[[[2, 0], [0, 0]]]]
        or record["legacyObservedMatrix"] != [[[[0, 0], [0, 0]]]]
    ):
        raise ValueError("oracle observations are invalid")
    if record["legacyObservedMatrix"] == record["matrixExpected"]:
        raise ValueError("oracle must not normalize the legacy defect")


def _factor_table(value: object) -> list[list[list[int]]]:
    if type(value) is not list or len(value) != 2:
        raise ValueError("factor table must be 2x2")
    for row in value:
        if type(row) is not list or len(row) != 2:
            raise ValueError("factor table must be 2x2")
        for product in row:
            if (
                type(product) is not list
                or len(product) != 2
                or any(type(item) is not int for item in product)
            ):
                raise ValueError("factor table products must be integer pairs")
    return value


def _tensor_product_multiply(
    left: object, right: object, tables: list[list[list[list[int]]]]
) -> list[list[int]]:
    a = _coordinates(left, "left tensor coordinate")
    b = _coordinates(right, "right tensor coordinate")
    first, second = tables
    result = [[0, 0], [0, 0]]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for right_second in range(2):
                    scale = a[i][j] * b[k][right_second]
                    for p in range(2):
                        for q in range(2):
                            result[p][q] += (
                                scale * first[i][k][p] * second[j][right_second][q]
                            )
    return result


def _matrix_tensor_product_multiply(
    left: object, right: object, tables: list[list[list[list[int]]]]
) -> list[list[list[list[int]]]]:
    if (
        type(left) is not list
        or type(right) is not list
        or len(left) != 1
        or len(right) != 2
        or any(type(row) is not list for row in left + right)
    ):
        raise ValueError("oracle matrices must be 1x2 and 2x1")
    if len(left[0]) != 2 or any(len(row) != 1 for row in right):
        raise ValueError("oracle matrix dimensions are invalid")
    total = [[0, 0], [0, 0]]
    for index in range(2):
        product = _tensor_product_multiply(left[0][index], right[index][0], tables)
        for i in range(2):
            for j in range(2):
                total[i][j] += product[i][j]
    return [[total]]


def test_generated_lifts_fixture_is_canonical_and_bound_to_v85() -> None:
    payload = _validate(_fixture())

    assert _FIXTURE.read_bytes() == _canonical(payload)
    assert payload["source"] == _SOURCE
    assert {item["name"] for item in payload["definitions"]} == set(_FUNCTIONS)


def test_recipe_renders_each_isolated_scenario_with_declared_extract() -> None:
    recipe = _validate(_fixture())["captureRecipe"]

    for scenario in recipe["scenarios"]:
        rendered = _render_recipe(recipe["template"], scenario)
        assert 'Get[Environment["ANYALGEBRA_ALGMUL_SOURCE"]]' in rendered
        assert "v86zzEvidence=(" in rendered
        assert "ExportString" in rendered
        assert "v86zzRecord/@{" in rendered
        assert "v86zzRecord/@[" not in rendered
        assert 'v86zzHarnessGlobals=Names["Global`*"]' in rendered
        assert 'Complement[Names["Global`*"],v86zzHarnessGlobals]' in rendered
        assert "v86zzNonzero=.;" in rendered
        assert 'v86zzSha[""];v86zzSnapshot[]' in rendered
        assert "::shdw" not in rendered
        assert scenario["action"] in rendered
        assert scenario["extract"] in rendered


def test_generated_names_mutations_and_tensor_hashes_are_exact() -> None:
    payload = _validate(_fixture())

    assert [item["name"] for item in payload["makeChar"]["symbols"]] == [
        "Global`Q",
        "Global`q1",
        "Global`q2",
        "Global`QMul",
    ]
    assert payload["makeTMul"]["sameFactor"]["basis"] == [
        "1",
        "AlgMul`e1",
        "AlgMul`e1",
        "AlgMul`e1",
    ]
    assert payload["scenarioContextMutations"] == {
        "load": {"added": []},
        "make-char": {"added": []},
        "same-maketmul": {"added": []},
        "cross-maketmul": {"added": ["AlgMul`e1AlgMul`"]},
        "lifts": {"added": []},
    }
    assert payload["makeChar"]["globalMutation"]["added"] == [
        "Q",
        "QMul",
        "q1",
        "q2",
    ]
    assert payload["makeTMul"]["sameFactor"]["globalMutation"] == {
        "before": [],
        "added": [],
        "after": [],
    }
    assert payload["makeTMul"]["crossFactor"]["globalMutation"] == {
        "before": [],
        "added": ["H"],
        "after": ["H"],
    }


def test_table_convolution_oracle_exposes_tensor_and_matrix_lift_defects() -> None:
    payload = _validate(_fixture())
    oracle = payload["independentOracle"]

    assert _tensor_product_multiply(
        oracle["squareInput"], oracle["squareInput"], oracle["factorTables"]
    ) == [[1, 0], [0, 0]]
    assert _matrix_tensor_product_multiply(
        oracle["matrixLeft"], oracle["matrixRight"], oracle["factorTables"]
    ) == [[[[2, 0], [0, 0]]]]
    assert oracle["legacyObservedMatrix"] != oracle["matrixExpected"]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: value["source"].update({"bytes": 1}), "source binding"),
        (lambda value: value["source"].update({"path": "wrong"}), "source binding"),
        (
            lambda value: value["definitions"].append(
                deepcopy(value["definitions"][0])
            ),
            "function definitions",
        ),
        (
            lambda value: value["definitions"][0].update({"definitionHash": "a" * 64}),
            "definition hash",
        ),
        (
            lambda value: value["makeChar"]["route"].update(
                {"definitionPresent": True}
            ),
            "MakeChar route",
        ),
        (
            lambda value: value["makeTMul"]["sameFactor"]["structureDot"].update(
                {"hash": "0" * 64}
            ),
            "structure tensor",
        ),
        (
            lambda value: value["lifts"][0].update({"result": [[0, 0], [0, 0]]}),
            "TMul 2x4",
        ),
        (
            lambda value: value["independentOracle"].update(
                {"legacyObservedMatrix": value["independentOracle"]["matrixExpected"]}
            ),
            "oracle observations",
        ),
        (
            lambda value: value["captureRecipe"].update(
                {"freshProcessPerScenario": False}
            ),
            "capture recipe binding",
        ),
        (
            lambda value: value["makeChar"]["symbols"][0].update({"downValueCount": 1}),
            "must be undefined",
        ),
        (
            lambda value: value["makeTMul"]["crossFactor"].update({"basis": ["1"] * 8}),
            "basis",
        ),
        (
            lambda value: value["makeTMul"]["crossFactor"]["globalMutation"].update(
                {"added": []}
            ),
            "global mutation",
        ),
        (
            lambda value: value["scenarioContextMutations"]["cross-maketmul"].update(
                {"added": []}
            ),
            "scenario context mutations",
        ),
        (
            lambda value: value["makeTMul"]["sameFactor"]["contextMutation"].update(
                {"added": ["fabricated`"]}
            ),
            "MakeTMul context mutation",
        ),
        (
            lambda value: value["makeTMul"]["crossFactor"]["symbols"][0].update(
                {"name": "Global`notH"}
            ),
            "symbol identity",
        ),
        (
            lambda value: value["makeTMul"]["sameFactor"]["algDefinition"].update(
                {"after": _ALG_BEFORE_HASH}
            ),
            "mutation hashes",
        ),
        (
            lambda value: value["lifts"][1].update({"classification": "fabricated"}),
            "summation lift",
        ),
        (
            lambda value: value["lifts"][2].update({"kernel": "fabricated"}),
            "explicit MMul",
        ),
        (
            lambda value: value["independentOracle"].update({"factorTables": []}),
            "factor tables",
        ),
    ],
)
def test_generated_lifts_validator_rejects_tampering(
    mutator: Any, message: str
) -> None:
    tampered = deepcopy(_fixture())
    mutator(tampered)

    with pytest.raises(ValueError, match=message):
        _validate(tampered)


def test_convolution_rejects_wrong_coordinate_and_matrix_shapes() -> None:
    tables = _fixture()["independentOracle"]["factorTables"]
    with pytest.raises(ValueError, match="2x2 integer"):
        _tensor_product_multiply([[1]], [[1, 0], [0, 0]], tables)
    with pytest.raises(ValueError, match="1x2 and 2x1"):
        _matrix_tensor_product_multiply([], [], tables)
