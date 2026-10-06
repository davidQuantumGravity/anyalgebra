"""Strict, non-oracular runtime evidence for the legacy ``MakeAlg`` surface."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "legacy" / "makealg.json"
_FIXTURE_DIR = _FIXTURE.parent
_SOURCE = {
    "path": "${ANYALGEBRA_ALGMUL_SOURCE}",
    "sha256": "3A13C9F47092B5B9814AA08873FF4F1DE61D4635DF68FC7F3A039A7DA5B0D00D",
    "bytes": 194012,
}
_LINKS = {
    "runtimeLoad": {
        "path": "runtime-load.json",
        "sha256": "AEA955F5484DDAB0B8BB1FAF9A2417095DABB889B3467EBC67571E0CDA34E2EE",
    },
    "generatedLifts": {
        "path": "generated-lifts.json",
        "sha256": "EE288652276C3BC318B17CB28C5006D9E7D582A16F566397CE84929C4FA8A904",
    },
}
_ALGS = [
    "O",
    "H",
    "C",
    "O2",
    "H2",
    "C2",
    "O3",
    "H3",
    "C3",
    "Os",
    "Hs",
    "Cs",
    "Os2",
    "Hs2",
    "Cs2",
    "Os3",
    "Hs3",
    "Cs3",
    "CxH",
    "D",
]
_BASE_ROUTES = {
    "O": "OMul",
    "H": "HMul",
    "C": "CMul",
    "Os": "OsMul",
    "Hs": "HsMul",
    "Cs": "CsMul",
}
_HEX = frozenset("0123456789ABCDEF")
_RECIPE_SHA256 = "FAAAAE43E86155366C4D15D8678E58EE3F3784594BFBA39E810DD41C8826496D"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, indent=2).encode() + b"\n"


def _fixture() -> dict[str, Any]:
    value = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def _mapping(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{label} has an invalid shape")
    return value


def _hash(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or set(value) - _HEX:
        raise ValueError(f"{label} must be uppercase SHA-256")
    return value


def _recipe_digest(recipe: dict[str, Any]) -> str:
    body = {key: item for key, item in recipe.items() if key != "sha256"}
    return (
        hashlib.sha256(
            json.dumps(
                body, ensure_ascii=True, sort_keys=True, separators=(",", ":")
            ).encode()
        )
        .hexdigest()
        .upper()
    )


def _contract(value: object) -> dict[str, Any]:
    root = _mapping(
        value,
        {
            "schemaType",
            "schemaVersion",
            "source",
            "boundFixtures",
            "kernel",
            "capture",
            "captureRecipe",
            "liveReplay",
            "symbols",
            "load",
            "makeAlg",
            "constructors",
            "repeatedLoad",
            "independentOracle",
            "limitations",
        },
        "fixture",
    )
    if (
        root["schemaType"] != "AnyAlgebra.AlgMulMakeAlgAudit"
        or root["schemaVersion"] != "1"
    ):
        raise ValueError("fixture schema is invalid")
    if _mapping(root["source"], set(_SOURCE), "source") != _SOURCE:
        raise ValueError("source binding is invalid")
    links = _mapping(root["boundFixtures"], set(_LINKS), "bound fixtures")
    for key, binding in _LINKS.items():
        if _mapping(links[key], set(binding), f"{key} binding") != binding:
            raise ValueError(f"{key} binding is invalid")
        if (
            hashlib.sha256((_FIXTURE_DIR / binding["path"]).read_bytes())
            .hexdigest()
            .upper()
            != binding["sha256"]
        ):
            raise ValueError(f"{key} hash is stale")
    if _mapping(
        root["kernel"], {"versionNumber", "freshProcess", "adapter"}, "kernel"
    ) != {
        "versionNumber": "12.0",
        "freshProcess": True,
        "adapter": "anyalgebra.legacy.mathematica.MathematicaAdapter",
    }:
        raise ValueError("kernel binding is invalid")
    capture = _mapping(
        root["capture"],
        {"schema", "freshProcessPerScenario", "prewarmedBaseline", "rawFraming"},
        "capture",
    )
    if (
        capture["schema"] != "AnyAlgebra.AlgMulMakeAlgRecipe.v1"
        or capture["freshProcessPerScenario"] is not True
    ):
        raise ValueError("capture isolation is invalid")
    if _mapping(
        capture["prewarmedBaseline"], {"context", "contextPath", "globals"}, "baseline"
    ) != {
        "context": "Global`",
        "contextPath": ["System`", "Global`"],
        "globals": ["args", "dims", "WolframScript"],
    }:
        raise ValueError("prewarmed baseline is invalid")
    framing = _mapping(
        capture["rawFraming"],
        {
            "begin",
            "end",
            "payload",
            "loadMessages",
            "sameProcessSecondLoadMessages",
            "helperGlobalDelta",
        },
        "raw framing",
    )
    if framing != {
        "begin": "V87_BEGIN",
        "end": "V87_END",
        "payload": "ExportString[association,RawJSON]",
        "loadMessages": [
            "HoldForm[AlgMul`n::shdw]",
            "HoldForm[AlgMul`s::shdw]",
            "HoldForm[AlgMul`x::shdw]",
        ],
        "sameProcessSecondLoadMessages": [],
        "helperGlobalDelta": [],
    }:
        raise ValueError("raw framing is invalid")
    _recipe(root["captureRecipe"])
    _live_replay(root["liveReplay"], root["load"])
    symbols = _mapping(
        root["symbols"],
        {
            "AlgMul`alg",
            "AlgMul`AddAlgName",
            "AlgMul`MakeChar",
            "AlgMul`MakeAlg",
            "AlgMul`MakeSOnFund",
            "AlgMul`MakeSUnFund",
            "AlgMul`MakeSPnFund",
            "AlgMul`AlgToChars",
            "AlgMul`CharsToAlg",
            "AlgMul`StructMul",
        },
        "symbols",
    )
    for name, digest in symbols.items():
        if not name.startswith("AlgMul`"):
            raise ValueError("symbol identity is invalid")
        _hash(digest, "symbol definition hash")
    load = _mapping(root["load"], {"algebras", "routes"}, "load")
    if (
        load["algebras"] != _ALGS
        or type(load["routes"]) is not dict
        or set(load["routes"]) != set(_ALGS)
    ):
        raise ValueError("registry is invalid")
    for algebra in _ALGS:
        base = algebra.rstrip("0123456789")
        expected = _BASE_ROUTES.get(base)
        if expected is None and algebra == "CxH":
            expected = "CxHMul"
        if expected is None and algebra == "D":
            expected = "DMul"
        if load["routes"][algebra] != f"AlgMul`{expected}":
            raise ValueError("multiplication lookup is invalid")
    _makealg(root["makeAlg"])
    _constructors(root["constructors"])
    _repeated(root["repeatedLoad"])
    _oracle(root["independentOracle"])
    limitations = _mapping(
        root["limitations"], {"observedDefects", "nonGoals"}, "limitations"
    )
    if limitations["nonGoals"] != [
        "No V88 MakeProperty exercise.",
        "No V89 dispositions.",
        "No V90 parity closure.",
        "No mathematical or scientific correctness claims.",
    ]:
        raise ValueError("non-goals are invalid")
    return root


def _recipe(value: object) -> None:
    recipe = _mapping(
        value,
        {
            "schema",
            "freshProcessPerScenario",
            "adapter",
            "template",
            "scenarios",
            "substitution",
            "sha256",
        },
        "capture recipe",
    )
    if (
        recipe["schema"] != "AnyAlgebra.AlgMulMakeAlgCaptureRecipe.v1"
        or recipe["freshProcessPerScenario"] is not True
    ):
        raise ValueError("capture recipe identity is invalid")
    adapter = _mapping(
        recipe["adapter"],
        {"module", "requiredVersion", "timeoutSeconds", "maxOutputBytes"},
        "capture adapter",
    )
    if adapter != {
        "module": "anyalgebra.legacy.mathematica.MathematicaAdapter",
        "requiredVersion": "12.0",
        "timeoutSeconds": 120,
        "maxOutputBytes": 1_048_576,
    }:
        raise ValueError("capture adapter is invalid")
    template = recipe["template"]
    if type(template) is not str or any(
        template.count(token) != 1
        for token in ("{{ACTION}}", "{{TARGETS}}", "{{EXTRACT}}")
    ):
        raise ValueError("capture template placeholders are invalid")
    required_order = (
        "v87zzSnapshot[]:=",
        'v87zzLoad=Get[Environment["ANYALGEBRA_ALGMUL_SOURCE"]]',
        'v87zzPrewarm={v87zzSnapshot[],v87zzSha[""],Names["Global`*"],Contexts[],ExportString[<||>,"RawJSON"]};',
        'v87zzHarnessGlobals=Names["Global`*"];v87zzHarnessContexts=Contexts[];',
        "v87zzBefore=v87zzSnapshot[];",
        "v87zzResult=({{ACTION}});",
        "v87zzEvidence=({{EXTRACT}});v87zzSymbols=(v87zzRecord/@{{TARGETS}});",
        "v87zzAfter=v87zzSnapshot[];",
        'Print["V87_BEGIN"];',
        'Print["V87_END"];',
    )
    positions = [template.find(text) for text in required_order]
    if -1 in positions or positions != sorted(positions):
        raise ValueError("capture template lacks its isolation contract")
    scenarios = recipe["scenarios"]
    expected_ids = [
        "load",
        "valid-makealg",
        "boundary-makealg",
        "wrong-arity",
        "son-one",
        "son-two",
        "sun-one-default",
        "sun-two",
        "spn-one",
        "spn-two",
        "repeat-named-makealg-second-get",
    ]
    if (
        type(scenarios) is not list
        or [item.get("id") for item in scenarios if type(item) is dict] != expected_ids
    ):
        raise ValueError("capture scenarios are invalid")
    for scenario in scenarios:
        record = _mapping(scenario, {"id", "action", "targets", "extract"}, "scenario")
        if (
            type(record["action"]) is not str
            or type(record["extract"]) is not str
            or type(record["targets"]) is not list
            or any(type(target) is not str for target in record["targets"])
        ):
            raise ValueError("capture scenario has an invalid field")
    substitution = _mapping(
        recipe["substitution"], {"ACTION", "TARGETS", "EXTRACT"}, "substitution"
    )
    if any(
        "sole" not in instruction
        for instruction in substitution.values()
        if type(instruction) is str
    ) or not all(type(instruction) is str for instruction in substitution.values()):
        raise ValueError("capture substitution contract is invalid")
    digest = _recipe_digest(recipe)
    if (
        _hash(recipe["sha256"], "capture recipe hash") != digest
        or digest != _RECIPE_SHA256
    ):
        raise ValueError("capture recipe hash is invalid")


def _live_replay(value: object, load: object) -> None:
    replay = _mapping(
        value,
        {"sourceLoadResult", "sourceLoadMessageTags", "scenarios"},
        "live replay",
    )
    if replay["sourceLoadResult"] != "Null" or not all(
        type(tag) is str for tag in replay["sourceLoadMessageTags"]
    ):
        raise ValueError("live replay load evidence is invalid")
    scenarios = replay["scenarios"]
    if type(scenarios) is not list or len(scenarios) != 11:
        raise ValueError("live replay scenarios are invalid")
    expected_ids = [
        "load",
        "valid-makealg",
        "boundary-makealg",
        "wrong-arity",
        "son-one",
        "son-two",
        "sun-one-default",
        "sun-two",
        "spn-one",
        "spn-two",
        "repeat-named-makealg-second-get",
    ]
    if [item.get("id") for item in scenarios if type(item) is dict] != expected_ids:
        raise ValueError("live replay order is invalid")
    for item in scenarios:
        record = _mapping(
            item,
            {
                "id",
                "actionResult",
                "actionMessageTags",
                "before",
                "after",
                "evidence",
                "symbols",
            },
            "live replay scenario",
        )
        for snapshot_name in ("before", "after"):
            snapshot = _mapping(
                record[snapshot_name],
                {"algDefinitionHash", "contexts", "globals"},
                f"live replay {snapshot_name}",
            )
            if snapshot["contexts"] != [] or type(snapshot["globals"]) is not list:
                raise ValueError("live replay snapshot is invalid")
            _hash(snapshot["algDefinitionHash"], "live replay alg hash")
        if (
            not all(type(tag) is str for tag in record["actionMessageTags"])
            or type(record["actionResult"]) is not str
            or type(record["evidence"]) is not dict
            or type(record["symbols"]) is not list
        ):
            raise ValueError("live replay evidence is invalid")
        for symbol in record["symbols"]:
            _mapping(
                symbol,
                {
                    "attributes",
                    "context",
                    "definitionHash",
                    "downValueCount",
                    "name",
                    "ownValueCount",
                    "subValueCount",
                    "upValueCount",
                },
                "live replay target definition",
            )
            _hash(symbol["definitionHash"], "live replay target hash")


def _expected_live_payload(
    fixture: dict[str, Any], scenario: dict[str, Any]
) -> dict[str, Any]:
    replay = fixture["liveReplay"]
    load = fixture["load"]
    assert type(replay) is dict and type(load) is dict
    expected = next(
        item
        for item in replay["scenarios"]
        if type(item) is dict and item["id"] == scenario["id"]
    )
    assert type(expected) is dict
    snapshots = {
        name: {
            **expected[name],
            "algebras": load["algebras"],
            "routes": load["routes"],
        }
        for name in ("before", "after")
    }
    return {
        "sourceLoadResult": replay["sourceLoadResult"],
        "sourceLoadMessageTags": replay["sourceLoadMessageTags"],
        "actionResult": expected["actionResult"],
        "actionMessageTags": expected["actionMessageTags"],
        "before": snapshots["before"],
        "after": snapshots["after"],
        "evidence": expected["evidence"],
        "symbols": expected["symbols"],
    }


def _render_recipe(recipe: dict[str, Any], scenario: dict[str, Any]) -> str:
    template = recipe["template"]
    if type(template) is not str:
        raise ValueError("capture template is invalid")
    action = scenario.get("action")
    extract = scenario.get("extract")
    targets = scenario.get("targets")
    if (
        type(action) is not str
        or type(extract) is not str
        or type(targets) is not list
        or any(type(target) is not str for target in targets)
    ):
        raise ValueError("capture targets are invalid")
    rendered = template
    replacements = (
        ("{{ACTION}}", action),
        ("{{TARGETS}}", "{" + ",".join(json.dumps(target) for target in targets) + "}"),
        ("{{EXTRACT}}", extract),
    )
    for token, replacement in replacements:
        if type(replacement) is not str or rendered.count(token) != 1:
            raise ValueError("capture rendering is invalid")
        rendered = rendered.replace(token, replacement)
    return rendered


def _makealg(value: object) -> None:
    item = _mapping(value, {"valid", "boundary", "wrongArity"}, "MakeAlg")
    valid = _mapping(
        item["valid"],
        {
            "name",
            "chars",
            "structDot",
            "structMul",
            "algToChars",
            "charsToAlg",
            "registryDelta",
            "route",
            "globalDelta",
        },
        "valid MakeAlg",
    )
    if valid != {
        "name": "AuditC",
        "chars": "{1, auditE1}",
        "structDot": "{{{1, 0}, {0, 1}}, {{0, 1}, {-1, 0}}}",
        "structMul": "{1, 0}",
        "algToChars": "{1, auditE1}",
        "charsToAlg": "{}",
        "registryDelta": [],
        "route": 'AlgMul`alg["Mul"]["AuditC"]',
        "globalDelta": ["auditE1"],
    }:
        raise ValueError("valid MakeAlg evidence is invalid")
    boundary = _mapping(
        item["boundary"],
        {"arguments", "storedChars", "storedStructDot", "registered", "route"},
        "boundary MakeAlg",
    )
    if boundary != {
        "arguments": [42, "not-a-list", 7],
        "storedChars": '"not-a-list"',
        "storedStructDot": "7",
        "registered": False,
        "route": 'AlgMul`alg["Mul"][42]',
    }:
        raise ValueError("boundary MakeAlg evidence is invalid")
    if _mapping(item["wrongArity"], {"input", "result", "messages"}, "wrong arity") != {
        "input": "AlgMul`MakeAlg[]",
        "result": "AlgMul`MakeAlg[]",
        "messages": [],
    }:
        raise ValueError("wrong-arity evidence is invalid")


def _constructors(value: object) -> None:
    records = _mapping(
        value,
        {"sonOne", "sonTwo", "sunOneDefault", "sunTwo", "spnOne", "spnTwo"},
        "constructors",
    )
    expected = {
        "sonOne": (
            "{auditG[1], auditG[2]}",
            ["SparseArray::list"],
            "Transpose",
            ["auditG"],
        ),
        "sonTwo": (
            "{auditG[1], auditG[2]}",
            ["SparseArray::list"],
            "Transpose",
            ["auditG"],
        ),
        "sunOneDefault": (
            "{AlgMul`g[1], AlgMul`g[2]}",
            ["SparseArray::list"],
            "Transpose",
            [],
        ),
        "sunTwo": (
            "{auditG[1], auditG[2]}",
            ["SparseArray::list"],
            "Transpose",
            ["auditG"],
        ),
        "spnOne": ("{}", [], 'AlgMul`alg["StructDot"]', []),
        "spnTwo": ("{}", [], 'AlgMul`alg["StructDot"]', []),
    }
    for key, (result, messages, head, globals_) in expected.items():
        record = _mapping(
            records[key],
            {
                "call",
                "result",
                "messages",
                "structDotHead",
                "registered",
                "routeDefined",
                "globalDelta",
            },
            f"{key} constructor",
        )
        if (
            record["result"] != result
            or record["messages"] != messages
            or record["structDotHead"] != head
            or record["registered"] is not False
            or record["routeDefined"] is not False
            or record["globalDelta"] != globals_
        ):
            raise ValueError("constructor evidence is invalid")


def _repeated(value: object) -> None:
    item = _mapping(value, {"afterNamedMakeAlg", "afterSecondGet"}, "repeated load")
    keys = {
        "algebraCount",
        "registered",
        "chars",
        "structDot",
        "route",
        "globals",
        "algDefinitionHash",
    }
    first = _mapping(item["afterNamedMakeAlg"], keys, "first load")
    second = _mapping(item["afterSecondGet"], keys, "second load")
    if (
        first["algebraCount"] != 20
        or second["algebraCount"] != 20
        or first["registered"] is not False
        or second["registered"] is not False
    ):
        raise ValueError("repeated registry state is invalid")
    if first["chars"] != "{1, auditReloadE1}" or not second["chars"].startswith(
        'AlgMul`alg["chars"]'
    ):
        raise ValueError("repeated custom character state is invalid")
    if first["structDot"].startswith("AlgMul`") or not second["structDot"].startswith(
        'AlgMul`alg["StructDot"]'
    ):
        raise ValueError("repeated structure state is invalid")
    if first["route"] != second["route"] or first["globals"] != second["globals"]:
        raise ValueError("repeated mutation state is invalid")
    _hash(first["algDefinitionHash"], "first alg hash")
    if (
        _hash(second["algDefinitionHash"], "second alg hash")
        != _contract_hash("AlgMul`alg")
        or first["algDefinitionHash"] == second["algDefinitionHash"]
    ):
        raise ValueError("repeated definition state is invalid")


def _contract_hash(name: str) -> str:
    return (
        "75568E6BB1A75B98FD0A56C2BE3340CC2CAD6E1F6B27CFA47B4DD9BA4CA7359B"
        if name == "AlgMul`alg"
        else ""
    )


def _oracle(value: object) -> None:
    item = _mapping(
        value,
        {"structureTensor", "left", "right", "expected", "classification"},
        "independent oracle",
    )
    tensor, left, right = item["structureTensor"], item["left"], item["right"]
    if not (
        type(tensor) is list
        and type(left) is list
        and type(right) is list
        and len(tensor) == len(left) == len(right) == 2
    ):
        raise ValueError("oracle shape is invalid")
    actual = [
        sum(left[i] * tensor[i][j][k] * right[k] for i in range(2) for k in range(2))
        for j in range(2)
    ]
    if (
        actual != item["expected"]
        or item["expected"] != [1, 0]
        or "does not endorse" not in item["classification"]
    ):
        raise ValueError("independent oracle is invalid")


def test_makealg_fixture_is_canonical_and_bound_to_accepted_evidence() -> None:
    payload = _contract(_fixture())
    assert _FIXTURE.read_bytes() == _canonical(payload)


def test_makealg_captures_registry_lookup_mutations_and_repeated_load() -> None:
    payload = _contract(_fixture())
    assert len(payload["load"]["algebras"]) == 20
    assert payload["makeAlg"]["valid"]["registryDelta"] == []
    assert (
        payload["repeatedLoad"]["afterNamedMakeAlg"]["globals"]
        == payload["repeatedLoad"]["afterSecondGet"]["globals"]
    )


def test_makealg_recorded_replay_has_only_declared_dynamic_globals() -> None:
    payload = _contract(_fixture())
    replay = payload["liveReplay"]
    assert replay["sourceLoadResult"] == "Null"
    assert (
        replay["sourceLoadMessageTags"]
        == payload["capture"]["rawFraming"]["loadMessages"]
    )
    expected = {
        "load": [],
        "valid-makealg": ["auditE1"],
        "boundary-makealg": [],
        "wrong-arity": [],
        "son-one": ["auditG"],
        "son-two": ["auditG"],
        "sun-one-default": [],
        "sun-two": ["auditG"],
        "spn-one": [],
        "spn-two": [],
        "repeat-named-makealg-second-get": ["auditReloadE1"],
    }
    for scenario in replay["scenarios"]:
        assert scenario["before"]["globals"] == []
        assert scenario["before"]["contexts"] == []
        assert scenario["after"]["globals"] == expected[scenario["id"]]
        assert scenario["after"]["contexts"] == []
        assert all(
            not name.startswith("v87zz") for name in scenario["after"]["globals"]
        )


def test_makealg_recipe_renders_each_isolated_scenario() -> None:
    recipe = _fixture()["captureRecipe"]
    assert type(recipe) is dict
    for scenario in recipe["scenarios"]:
        assert type(scenario) is dict
        rendered = _render_recipe(recipe, scenario)
        assert not any(
            token in rendered for token in ("{{ACTION}}", "{{TARGETS}}", "{{EXTRACT}}")
        )
        assert "v87zzPrewarm" in rendered
        assert 'Complement[Names["Global`*"],v87zzHarnessGlobals]' in rendered
        assert "Complement[Contexts[],v87zzHarnessContexts]" in rendered


def test_makealg_renderer_is_three_exact_literal_replacements() -> None:
    recipe = _fixture()["captureRecipe"]
    assert type(recipe) is dict
    scenario = recipe["scenarios"][1]
    assert type(scenario) is dict
    template = recipe["template"]
    assert type(template) is str
    expected = template
    for token, replacement in (
        ("{{ACTION}}", scenario["action"]),
        (
            "{{TARGETS}}",
            "{" + ",".join(json.dumps(target) for target in scenario["targets"]) + "}",
        ),
        ("{{EXTRACT}}", scenario["extract"]),
    ):
        assert type(replacement) is str
        expected = expected.replace(token, replacement)
    assert _render_recipe(recipe, scenario) == expected


def _replace_recipe_text_and_rehash(
    value: dict[str, Any], original: str, replacement: str
) -> None:
    recipe = value["captureRecipe"]
    assert type(recipe) is dict
    template = recipe["template"]
    assert type(template) is str and template.count(original) == 1
    recipe["template"] = template.replace(original, replacement)
    recipe["sha256"] = _recipe_digest(recipe)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["source"].update({"sha256": "0" * 64}),
        lambda value: value["boundFixtures"]["runtimeLoad"].update(
            {"sha256": "0" * 64}
        ),
        lambda value: value["kernel"].update({"versionNumber": "13.0"}),
        lambda value: value["capture"]["rawFraming"].update({"end": "wrong"}),
        lambda value: value["captureRecipe"].update({"sha256": "0" * 64}),
        lambda value: _replace_recipe_text_and_rehash(
            value,
            'v87zzPrewarm={v87zzSnapshot[],v87zzSha[""],Names["Global`*"],Contexts[],ExportString[<||>,"RawJSON"]};',
            "v87zzPrewarm={};",
        ),
        lambda value: _replace_recipe_text_and_rehash(
            value,
            'v87zzHarnessGlobals=Names["Global`*"];v87zzHarnessContexts=Contexts[];',
            "v87zzHarnessGlobals={};v87zzHarnessContexts={};",
        ),
        lambda value: _replace_recipe_text_and_rehash(
            value, "v87zzBefore=v87zzSnapshot[];", "v87zzAfter=v87zzSnapshot[];"
        ),
        lambda value: value["captureRecipe"]["scenarios"][0].update(
            {"action": "wrong[]"}
        ),
        lambda value: value["captureRecipe"].update({"template": "broken"}),
        lambda value: value["symbols"].update({"AlgMul`MakeAlg": "bad"}),
        lambda value: value["load"]["routes"].update({"O2": "AlgMul`O2Mul"}),
        lambda value: value["makeAlg"]["valid"].update({"registryDelta": ["AuditC"]}),
        lambda value: value["makeAlg"]["boundary"].update({"registered": True}),
        lambda value: value["constructors"]["sunOneDefault"].update(
            {"globalDelta": ["g"]}
        ),
        lambda value: value["repeatedLoad"]["afterSecondGet"].update(
            {"chars": "{1, auditReloadE1}"}
        ),
        lambda value: value["independentOracle"].update({"expected": [0, 1]}),
        lambda value: value["limitations"].update({"nonGoals": []}),
    ],
)
def test_makealg_validator_rejects_each_critical_tamper(mutator: Any) -> None:
    tampered = deepcopy(_fixture())
    mutator(tampered)
    with pytest.raises(ValueError):
        _contract(tampered)
