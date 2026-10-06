from __future__ import annotations

# ruff: noqa: E501

import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import anyalgebra
import anyalgebra.validation
import anyalgebra.validation.validate as validation_validate

import pytest


HERE = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "traceability", HERE / "tools" / "audit_v0_0_traceability.py"
)
assert SPEC and SPEC.loader
traceability = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = traceability
SPEC.loader.exec_module(traceability)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_repo(tmp_path: Path) -> tuple[Path, set[str]]:
    root = tmp_path
    (root / "src" / "anyalgebra").mkdir(parents=True)
    (root / "src" / "anyalgebra" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "anyalgebra" / "sample.py").write_text(
        "def build(value):\n    return value\n", encoding="utf-8"
    )
    process = root / ".agents" / "project-process"
    scope = {
        "apiIds": ["api.sample.build"],
        "testIds": ["test.sample.build"],
        "featureIds": ["fe.sample"],
        "actionIds": ["action.sample"],
        "storyIds": ["us.sample"],
        "componentIds": ["component.sample"],
        "stateIds": [],
        "decisionIds": ["ADR-0001"],
    }
    write_json(process / "versions" / "v0.0" / "scope.json", scope)
    write_json(
        process / "versions" / "v0.0" / "contract-snapshot.json",
        {"counts": {"apiIds": 1, "testIds": 1}},
    )
    base = {
        "featureIds": ["fe.sample"],
        "docPaths": ["docs/a.md"],
        "evidencePaths": ["evidence/a.json"],
    }
    write_json(
        process / "features.json",
        [
            {
                "id": "fe.sample",
                "actionIds": ["action.sample"],
                "storyIds": ["us.sample"],
                "testIds": ["test.sample.build"],
                **base,
            }
        ],
    )
    write_json(
        process / "actions.json",
        [
            {
                "id": "action.sample",
                "storyIds": ["us.sample"],
                "testIds": ["test.sample.build"],
                **base,
            }
        ],
    )
    write_json(
        process / "user-stories.json",
        [
            {
                "id": "us.sample",
                "actionIds": ["action.sample"],
                "testIds": ["test.sample.build"],
                **base,
            }
        ],
    )
    write_json(
        process / "components.json",
        [
            {
                "id": "component.sample",
                "featureIds": ["fe.sample"],
                "testIds": ["test.sample.build"],
            }
        ],
    )
    write_json(process / "states.json", [])
    write_json(
        process / "api-contracts.json",
        [
            {
                "id": "api.sample.build",
                "signature": "build(value)",
                "featureIds": ["fe.sample"],
            }
        ],
    )
    write_json(
        process / "test-matrix.json",
        [
            {
                "id": "test.sample.build",
                "featureIds": ["fe.sample"],
                "actionIds": ["action.sample"],
                "storyIds": ["us.sample"],
                "nodeIds": ["tests/test_sample.py::test_build"],
            }
        ],
    )
    quickstart = root / "docs" / "api"
    quickstart.mkdir(parents=True)
    (quickstart / "v0.0-quickstart.md").write_text(
        "| `api.sample.build` | `from anyalgebra.sample import build` | "
        "`build(value)` |\n",
        encoding="utf-8",
    )
    adr = (
        root / "docs" / "architecture" / "architecture-decisions" / "ADR-0001-sample.md"
    )
    adr.parent.mkdir(parents=True)
    adr.write_text("# ADR", encoding="utf-8")
    write_json(
        process / "versions" / "v0.0" / "adr-gate.json",
        {
            "entries": [
                {
                    "adrId": "ADR-0001",
                    "path": (
                        "docs/architecture/architecture-decisions/ADR-0001-sample.md"
                    ),
                    "gateDecision": "accepted",
                }
            ]
        },
    )
    write_json(
        process
        / "versions"
        / "v0.0"
        / "implementation"
        / "implementation-progress.json",
        {"accepted": [{"taskId": f"V00-{number:03d}"} for number in range(91, 99)]},
    )
    log = process / "versions" / "v0.0" / "skill-run-log.md"
    log.write_text("accepted evidence", encoding="utf-8")
    return root, {"tests/test_sample.py::test_build"}


def codes(report: Any) -> set[str]:
    return {finding.code for finding in report.findings}


def copied_release_repo(tmp_path: Path) -> tuple[Path, set[str]]:
    """Copy the actual release surface so mutation checks cannot use a toy graph."""
    root = tmp_path / f"release-{len(tuple(tmp_path.iterdir()))}"
    shutil.copytree(
        HERE,
        root,
        ignore=shutil.ignore_patterns(
            ".git",
            ".venv",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "build",
            "__pycache__",
        ),
    )
    nodes = {
        node
        for owners in (
            *traceability.TEST_OWNERS.values(),
            *traceability.RELEASE_EXAMPLE_OWNERS.values(),
        )
        for node in owners
    }
    return root, nodes


def release_audit(root: Path, nodes: set[str]) -> Any:
    """Run the real reviewed scope without subprocess execution in mutation tests."""
    return traceability.audit(root, collected_node_ids=nodes, execute_nodes=False)


def test_positive_fixture_resolves_concrete_callable_nodes_and_evidence(
    tmp_path: Path,
) -> None:
    root, nodes = make_repo(tmp_path)
    report = traceability.audit(root, collected_node_ids=nodes, execute_nodes=False)
    assert report.ok
    assert report.counts["apiIds"] == 1
    assert report.counts["testIds"] == 1
    assert report.ownership["api.sample.build"] == (
        "anyalgebra.sample:build",
        "build(value)",
    )
    assert "v91_v98_log_sha256" in report.evidence


def test_temp_audit_restores_preloaded_anyalgebra_module_graph(tmp_path: Path) -> None:
    root, nodes = make_repo(tmp_path)
    before = {
        name: module
        for name, module in sys.modules.items()
        if name == "anyalgebra" or name.startswith("anyalgebra.")
    }
    report = traceability.audit(root, collected_node_ids=nodes, execute_nodes=False)
    assert report.ok
    for name, module in before.items():
        assert sys.modules[name] is module
    assert anyalgebra.validation.validate is validation_validate


def test_tampered_scope_and_missing_registry_are_typed_findings(tmp_path: Path) -> None:
    root, nodes = make_repo(tmp_path)
    scope_path = (
        root / ".agents" / "project-process" / "versions" / "v0.0" / "scope.json"
    )
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    scope["apiIds"].append("api.missing")
    write_json(scope_path, scope)
    report = traceability.audit(root, collected_node_ids=nodes, execute_nodes=False)
    assert {"SCOPED_COUNT_MISMATCH", "ORPHAN_SCOPE_ID", "API_TARGET_MISSING"} <= codes(
        report
    )


def test_duplicate_node_bad_symbol_and_bad_node_are_rejected(tmp_path: Path) -> None:
    root, nodes = make_repo(tmp_path)
    process = root / ".agents" / "project-process"
    tests = json.loads((process / "test-matrix.json").read_text(encoding="utf-8"))
    tests.append(
        {
            "id": "test.other",
            "featureIds": ["fe.sample"],
            "nodeIds": ["tests/test_sample.py::test_build"],
        }
    )
    write_json(process / "test-matrix.json", tests)
    scope_path = process / "versions" / "v0.0" / "scope.json"
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    scope["testIds"].append("test.other")
    write_json(scope_path, scope)
    quickstart = root / "docs" / "api" / "v0.0-quickstart.md"
    quickstart.write_text(
        "| `api.sample.build` | `from anyalgebra.sample import missing` | "
        "`missing(value)` |\n",
        encoding="utf-8",
    )
    report = traceability.audit(root, collected_node_ids=nodes, execute_nodes=False)
    assert {"DUPLICATE_TEST_NODE_OWNER", "API_TARGET_UNRESOLVED"} <= codes(report)


def test_missing_node_metadata_and_unsafe_module_are_rejected(tmp_path: Path) -> None:
    root, nodes = make_repo(tmp_path)
    process = root / ".agents" / "project-process"
    tests = json.loads((process / "test-matrix.json").read_text(encoding="utf-8"))
    tests[0].pop("nodeIds")
    write_json(process / "test-matrix.json", tests)
    quickstart = root / "docs" / "api" / "v0.0-quickstart.md"
    quickstart.write_text(
        "| `api.sample.build` | `from os import system` | `system(value)` |\n",
        encoding="utf-8",
    )
    report = traceability.audit(root, collected_node_ids=nodes, execute_nodes=False)
    assert {"TEST_NODE_OWNERSHIP_MISSING", "UNSAFE_API_MODULE"} <= codes(report)


@pytest.mark.internal_records
def test_reciprocal_action_story_component_state_and_contract_tampering_is_typed(
    tmp_path: Path,
) -> None:
    root, _ = make_repo(tmp_path)
    process = root / ".agents" / "project-process"
    scope = json.loads(
        (process / "versions" / "v0.0" / "scope.json").read_text(encoding="utf-8")
    )
    registries = {
        field: json.loads((process / filename).read_text(encoding="utf-8"))
        for field, filename in traceability.REGISTRIES.items()
    }
    registries["featureIds"][0].update(
        {
            "actionIds": ["action.sample"],
            "storyIds": ["us.sample"],
            "testIds": ["test.sample.build"],
        }
    )
    registries["actionIds"][0]["featureIds"] = []
    registries["storyIds"][0]["featureIds"] = []
    registries["componentIds"][0]["testIds"] = []
    registries["stateIds"] = [{"id": "state.sample", "featureIds": []}]
    scope["stateIds"] = ["state.sample"]
    parsed = {
        field: traceability._records(payload, field, [])
        for field, payload in registries.items()
    }
    findings: list[Any] = []
    ownership: dict[str, tuple[str, ...]] = {}
    traceability._check_registry_graph(parsed, scope, findings, ownership)
    assert {
        "RECIPROCAL_LINK_MISSING",
        "COMPONENT_TEST_LINK_MISSING",
        "STATE_FEATURE_LINK_MISSING",
    } <= {finding.code for finding in findings}
    contracts = {
        entry["id"]: entry
        for entry in json.loads(
            (HERE / ".agents" / "project-process" / "api-contracts.json").read_text(
                encoding="utf-8"
            )
        )
    }
    contracts["api.domain.construct_exact"] = {
        **contracts["api.domain.construct_exact"],
        "signature": "changed(value)",
    }
    contract_findings: list[Any] = []
    traceability._check_api_targets(
        HERE,
        contracts,
        tuple(traceability.API_SPECS),
        contract_findings,
        {},
    )
    assert "API_CONTRACT_CALL_MISMATCH" in {
        finding.code for finding in contract_findings
    }


@pytest.mark.internal_records
def test_reviewed_api_contract_rejects_nonfirst_call_kind_default_and_alias_mutations(
    monkeypatch: Any,
) -> None:
    """Every named callable is checked; not merely a representative first call."""
    contracts = {
        entry["id"]: entry
        for entry in json.loads(
            (HERE / ".agents" / "project-process" / "api-contracts.json").read_text(
                encoding="utf-8"
            )
        )
    }
    contracts["api.serialization.safe_record"] = {
        **contracts["api.serialization.safe_record"],
        "signature": "SerializerRegistry.to_record(value); SerializerRegistry.from_record(record); unrelated(value, *, registry)",
    }
    findings: list[Any] = []
    traceability._check_api_targets(
        HERE, contracts, tuple(traceability.API_SPECS), findings, {}
    )
    assert "API_CONTRACT_SIGNATURE_MISMATCH" in {item.code for item in findings}

    original = traceability.API_SPECS["api.presentation.convert"]
    bad_default = traceability.CallableSpec(
        original[0].module,
        original[0].symbol,
        original[0].member,
        (
            *original[0].parameters[:-1],
            ("route", traceability.P.KEYWORD_ONLY, "not-none"),
        ),
    )
    monkeypatch.setitem(
        traceability.API_SPECS, "api.presentation.convert", (bad_default,)
    )
    findings = []
    traceability._check_api_targets(
        HERE, contracts, tuple(traceability.API_SPECS), findings, {}
    )
    assert "API_SIGNATURE_MISMATCH" in {item.code for item in findings}

    bad_kind = traceability.CallableSpec(
        original[0].module,
        original[0].symbol,
        original[0].member,
        (
            *original[0].parameters[:-1],
            ("route", traceability.P.POSITIONAL_OR_KEYWORD, None),
        ),
    )
    monkeypatch.setitem(traceability.API_SPECS, "api.presentation.convert", (bad_kind,))
    findings = []
    traceability._check_api_targets(
        HERE, contracts, tuple(traceability.API_SPECS), findings, {}
    )
    assert "API_SIGNATURE_MISMATCH" in {item.code for item in findings}

    monkeypatch.setitem(
        traceability.PUBLISHED_CALLS,
        "api.presentation.convert",
        "convert(value, target_kind)",
    )
    findings = []
    traceability._check_api_targets(
        HERE, contracts, tuple(traceability.API_SPECS), findings, {}
    )
    assert "API_PUBLISHED_CALL_MISMATCH" in {item.code for item in findings}


@pytest.mark.internal_records
def test_full_release_copy_rejects_same_count_scope_and_component_state_substitution(
    tmp_path: Path,
) -> None:
    root, nodes = copied_release_repo(tmp_path)
    process = root / ".agents" / "project-process"
    scope_path = process / "versions" / "v0.0" / "scope.json"
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    scope["apiIds"][0] = "api.substituted"
    write_json(scope_path, scope)
    assert "REVIEWED_SCOPE_SET_MISMATCH" in codes(release_audit(root, nodes))

    root, nodes = copied_release_repo(tmp_path)
    process = root / ".agents" / "project-process"
    scope_path = process / "versions" / "v0.0" / "scope.json"
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    scope["apiIds"] = [f"api.substituted.{index}" for index in range(23)]
    scope["testIds"] = [f"test.substituted.{index}" for index in range(22)]
    write_json(scope_path, scope)
    assert "REVIEWED_SCOPE_SET_MISMATCH" in codes(release_audit(root, nodes))

    root, nodes = copied_release_repo(tmp_path)
    process = root / ".agents" / "project-process"
    scope_path = process / "versions" / "v0.0" / "scope.json"
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    components = json.loads((process / "components.json").read_text(encoding="utf-8"))
    states = json.loads((process / "states.json").read_text(encoding="utf-8"))
    scope["componentIds"][0] = "component.substituted"
    scope["stateIds"][0] = "state.substituted"
    components[0]["id"] = "component.substituted"
    states[0]["id"] = "state.substituted"
    write_json(scope_path, scope)
    write_json(process / "components.json", components)
    write_json(process / "states.json", states)
    report = release_audit(root, nodes)
    assert "REVIEWED_SCOPE_SET_MISMATCH" in codes(report)
    registries = {
        field: traceability._records(
            json.loads((process / filename).read_text(encoding="utf-8")), field, []
        )
        for field, filename in traceability.REGISTRIES.items()
    }
    findings: list[Any] = []
    traceability._check_scoped_owners(root, registries, scope, findings, {})
    assert {"COMPONENT_OWNER_SET_MISMATCH", "STATE_OWNER_SET_MISMATCH"} <= {
        item.code for item in findings
    }


@pytest.mark.internal_records
def test_full_release_copy_rejects_graph_docs_and_evidence_tampering(
    tmp_path: Path,
) -> None:
    root, nodes = copied_release_repo(tmp_path)
    process = root / ".agents" / "project-process"
    actions_path = process / "actions.json"
    actions = json.loads(actions_path.read_text(encoding="utf-8"))
    actions[0]["testIds"] = actions[0]["testIds"][1:]
    write_json(actions_path, actions)
    parsed = {
        field: traceability._records(
            json.loads((process / filename).read_text(encoding="utf-8")), field, []
        )
        for field, filename in traceability.REGISTRIES.items()
    }
    scope = json.loads(
        (process / "versions" / "v0.0" / "scope.json").read_text(encoding="utf-8")
    )
    findings: list[Any] = []
    traceability._check_registry_graph(parsed, scope, findings, {})
    assert "RECIPROCAL_LINK_MISSING" in {item.code for item in findings}

    root, nodes = copied_release_repo(tmp_path)
    (root / "docs" / "specifications" / "v0.0-spec.md").write_text("", encoding="utf-8")
    assert "RELEASE_DOCUMENT_CONTENT_MISMATCH" in codes(release_audit(root, nodes))

    root, nodes = copied_release_repo(tmp_path)
    (
        root / ".agents" / "project-process" / "versions" / "v0.0" / "skill-run-log.md"
    ).write_bytes(b"arbitrary")
    assert "SKILL_LOG_SECTION_MISSING" in codes(release_audit(root, nodes))

    root, _ = copied_release_repo(tmp_path)
    progress_path = (
        root
        / ".agents"
        / "project-process"
        / "versions"
        / "v0.0"
        / "implementation"
        / "implementation-progress.json"
    )
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    entry = next(item for item in progress["accepted"] if item["taskId"] == "V00-091")
    entry["verification"]["stableFiles"] = "two_sorted_release.py SHA-256 " + "0" * 64
    write_json(progress_path, progress)
    findings = []
    traceability._accepted_evidence(root, findings, strict=True)
    assert "EVIDENCE_STABLE_FILE_HASH_MISMATCH" in {item.code for item in findings}

    root, _ = copied_release_repo(tmp_path)
    progress_path = (
        root
        / ".agents"
        / "project-process"
        / "versions"
        / "v0.0"
        / "implementation"
        / "implementation-progress.json"
    )
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    entry = next(item for item in progress["accepted"] if item["taskId"] == "V00-091")
    entry["verification"]["independentAudit"] = "arbitrary"
    write_json(progress_path, progress)
    findings = []
    traceability._accepted_evidence(root, findings, strict=True)
    assert "EVIDENCE_INDEPENDENT_AUDIT_CONTENT_MISMATCH" in {
        item.code for item in findings
    }

    root, _ = copied_release_repo(tmp_path)
    log_path = (
        root / ".agents" / "project-process" / "versions" / "v0.0" / "skill-run-log.md"
    )
    log_path.write_text(
        "\n".join(f"## V00-{number:03d}" for number in range(91, 99)),
        encoding="utf-8",
    )
    findings = []
    traceability._accepted_evidence(root, findings, strict=True)
    assert "SKILL_LOG_SECTION_CONTENT_MISMATCH" in {item.code for item in findings}


@pytest.mark.internal_records
def test_audit_reports_encoding_import_and_subprocess_failures_without_traceback(
    tmp_path: Path, monkeypatch: Any
) -> None:
    root, nodes = copied_release_repo(tmp_path)
    (root / "docs" / "api" / "v0.0-quickstart.md").write_bytes(b"\xff")
    assert "API_DOCUMENTATION_UNREADABLE" in codes(release_audit(root, nodes))

    root, nodes = copied_release_repo(tmp_path)
    (root / "src" / "anyalgebra" / "core" / "domains.py").write_text(
        "raise RuntimeError('controlled import failure')\n", encoding="utf-8"
    )
    assert "API_TARGET_UNRESOLVED" in codes(release_audit(root, nodes))

    def timeout(*args: Any, **kwargs: Any) -> Any:
        raise traceability.subprocess.TimeoutExpired("pytest", 1)

    monkeypatch.setattr(traceability.subprocess, "run", timeout)
    _, collection = traceability.collect_pytest_nodes(HERE)
    assert collection is not None and collection.code == "PYTEST_COLLECTION_TIMEOUT"

    def os_error(*args: Any, **kwargs: Any) -> Any:
        raise OSError("controlled process failure")

    monkeypatch.setattr(traceability.subprocess, "run", os_error)
    assert (
        traceability._run_selected_pytest(HERE, ("missing::node",), "test") is not None
    )


@pytest.mark.internal_records
def test_temp_audit_isolates_and_restores_reviewed_tools_namespace(
    tmp_path: Path,
) -> None:
    """A preloaded runtime-capture tool cannot conceal a copied-repo failure."""
    original = importlib.import_module("tools.capture_algmul_runtime")
    parent = importlib.import_module("tools")
    original_attribute = parent.capture_algmul_runtime
    root, nodes = copied_release_repo(tmp_path)
    (root / "tools" / "capture_algmul_runtime.py").write_text(
        "raise RuntimeError('controlled tool import failure')\n", encoding="utf-8"
    )
    assert "API_TARGET_UNRESOLVED" in codes(release_audit(root, nodes))
    assert sys.modules["tools.capture_algmul_runtime"] is original
    assert importlib.import_module("tools").capture_algmul_runtime is original_attribute


@pytest.mark.internal_records
def test_temp_state_audit_isolates_and_restores_cached_owner_module(
    tmp_path: Path,
) -> None:
    """Copied state seams cannot reuse the current evidence module cache."""
    original = importlib.import_module("anyalgebra.evidence.run")
    original_status = original.ExecutionStatus
    root, nodes = copied_release_repo(tmp_path)
    target = root / "src" / "anyalgebra" / "evidence" / "run.py"
    target.write_text(
        target.read_text(encoding="utf-8").replace(
            '    BLOCKED = "blocked"', '    BLOCKED = "not_blocked"'
        ),
        encoding="utf-8",
    )
    assert "STATE_CODE_VALUES_MISMATCH" in codes(release_audit(root, nodes))
    assert sys.modules["anyalgebra.evidence.run"] is original
    assert (
        importlib.import_module("anyalgebra.evidence.run").ExecutionStatus
        is original_status
    )


@pytest.mark.internal_records
def test_state_audit_binds_code_owned_outcomes_and_terminal_lifecycle() -> None:
    """The lifecycle and terminal receipt subset are both independently owned."""
    process = HERE / ".agents" / "project-process"
    registries = {
        field: traceability._records(
            json.loads((process / filename).read_text(encoding="utf-8")), field, []
        )
        for field, filename in traceability.REGISTRIES.items()
    }
    scope = json.loads(
        (process / "versions" / "v0.0" / "scope.json").read_text(encoding="utf-8")
    )
    findings: list[Any] = []
    traceability._check_scoped_owners(HERE, registries, scope, findings, {})
    codes_seen = {item.code for item in findings}
    assert "STATE_CODE_VALUES_MISMATCH" not in codes_seen
