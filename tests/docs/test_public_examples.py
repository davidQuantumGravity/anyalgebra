"""Executable contract for the concise, current v0.0 public quickstart."""

from __future__ import annotations

# ruff: noqa: E501

from dataclasses import dataclass
import importlib
import inspect
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import pytest

from anyalgebra.persistence.registry import SchemaError, SerializerRegistry
from anyalgebra.structures.outcomes import (
    Defined,
    Failed,
    FailedOutcomeError,
    Indeterminate,
    IndeterminateOutcomeError,
    Undefined,
    UndefinedOutcomeError,
    unwrap_defined,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
QUICKSTART = REPOSITORY_ROOT / "docs" / "api" / "v0.0-quickstart.md"


@dataclass(frozen=True)
class PublicApi:
    """One scoped API ID, its explicit import seam, and callable entrypoint."""

    identifier: str
    module: str
    imported_name: str
    attribute: str | None
    published_call: str
    parameter_names: tuple[str, ...]

    @property
    def import_statement(self) -> str:
        """Return the literal import promised by the quickstart table."""
        return f"from {self.module} import {self.imported_name}"


PUBLIC_APIS = (
    PublicApi(
        "api.domain.construct_exact", "anyalgebra.core.domains", "ZZ", None, "ZZ()", ()
    ),
    PublicApi(
        "api.coercion.plan_apply",
        "anyalgebra.core.coercions",
        "common_parent",
        None,
        "common_parent(values, *, graph)",
        ("values", "graph"),
    ),
    PublicApi(
        "api.parent.freeze",
        "anyalgebra.core.parents",
        "ParentBuilder",
        "freeze",
        "ParentBuilder.freeze()",
        ("self",),
    ),
    PublicApi(
        "api.module.element",
        "anyalgebra.core.modules",
        "FreeModule",
        "element",
        "FreeModule.element(coordinates, *, graph=None)",
        ("self", "coordinates", "graph"),
    ),
    PublicApi(
        "api.structure.build",
        "anyalgebra.structures.structure",
        "StructureBuilder",
        None,
        "StructureBuilder(signature)",
        ("signature",),
    ),
    PublicApi(
        "api.structure.evaluate",
        "anyalgebra.structures.evaluate",
        "evaluate_term",
        None,
        "evaluate_term(structure, term, environment)",
        ("structure", "term", "environment"),
    ),
    PublicApi(
        "api.structure.deductive_closure",
        "anyalgebra.structures.deductive",
        "closure",
        None,
        "closure(premises, system, *, max_rounds)",
        ("premises", "system", "max_rounds"),
    ),
    PublicApi(
        "api.partiality.totalize",
        "anyalgebra.structures.partiality",
        "totalize",
        None,
        "totalize(partial_operation, *, bottom)",
        ("partial_operation", "bottom"),
    ),
    PublicApi(
        "api.algebra.from_structure_constants",
        "anyalgebra.algebra.multilinear",
        "FiniteMultilinearStructure",
        "from_structure_constants",
        "FiniteMultilinearStructure.from_structure_constants(module, constants, *, name=None)",
        ("module", "constants", "name"),
    ),
    PublicApi(
        "api.presentation.convert",
        "anyalgebra.presentations.convert",
        "convert",
        None,
        "convert(value, target_kind, *, graph, source_kind, route=None)",
        ("value", "target_kind", "graph", "source_kind", "route"),
    ),
    PublicApi(
        "api.presentation.change_basis",
        "anyalgebra.presentations.basis_change",
        "change_basis",
        None,
        "change_basis(value_or_structure, isomorphism)",
        ("value_or_structure", "isomorphism"),
    ),
    PublicApi(
        "api.matrix.construct",
        "anyalgebra.linear.matrix",
        "MatrixSpace",
        None,
        "MatrixSpace(rows, columns, entry_parent)",
        ("rows", "columns", "entry_parent"),
    ),
    PublicApi(
        "api.linear.recover_structure_constants",
        "anyalgebra.linear.recovery",
        "recover_structure_constants",
        None,
        "recover_structure_constants(operators, declared_bracket)",
        ("operators", "declared_bracket"),
    ),
    PublicApi(
        "api.validation.validate_law",
        "anyalgebra.validation.validate",
        "validate_law",
        None,
        "validate_law(structure, law)",
        ("structure", "law"),
    ),
    PublicApi(
        "api.analysis.algebra_fingerprint",
        "anyalgebra.analysis.fingerprint",
        "algebra_fingerprint",
        None,
        "algebra_fingerprint(algebra, *, options=None)",
        ("algebra", "options"),
    ),
    PublicApi(
        "api.serialization.safe_record",
        "anyalgebra.persistence.registry",
        "SerializerRegistry",
        "from_record",
        "SerializerRegistry.from_record(record)",
        ("self", "record"),
    ),
    PublicApi(
        "api.evidence.contract",
        "anyalgebra.evidence.contracts",
        "CalculationContract",
        "create",
        "CalculationContract.create(*, contract_name, project_id, question, acceptable_outcomes, input_hashes, source_anchors, convention_manifests, algorithms, backend_request, backend_name, backend_version, assumptions, theorem_hypotheses, bounds, required_cross_checks, expected_artifacts, acceptance_predicates)",
        (
            "contract_name",
            "project_id",
            "question",
            "acceptable_outcomes",
            "input_hashes",
            "source_anchors",
            "convention_manifests",
            "algorithms",
            "backend_request",
            "backend_name",
            "backend_version",
            "assumptions",
            "theorem_hypotheses",
            "bounds",
            "required_cross_checks",
            "expected_artifacts",
            "acceptance_predicates",
        ),
    ),
    PublicApi(
        "api.evidence.run",
        "anyalgebra.evidence.run",
        "run_calculation",
        None,
        "run_calculation(contract, calculation, *, environment=None)",
        ("contract", "calculation", "environment"),
    ),
    PublicApi(
        "api.evidence.replay",
        "anyalgebra.evidence.replay",
        "verify_replay",
        None,
        "verify_replay(receipt, *, resolver, environment=None)",
        ("receipt", "resolver", "environment"),
    ),
    PublicApi(
        "api.backend.capabilities",
        "anyalgebra.backends.reference",
        "ReferenceBackend",
        "capabilities",
        "ReferenceBackend.capabilities()",
        ("self",),
    ),
    PublicApi(
        "api.legacy.capture_static",
        "anyalgebra.legacy.algmul_static",
        "capture_static",
        None,
        "capture_static(source_path)",
        ("source_path",),
    ),
    PublicApi(
        "api.legacy.capture_runtime",
        "tools.capture_algmul_runtime",
        "capture_runtime",
        None,
        "capture_runtime(source_path, adapter, *, expected_source_sha256='3A13C9F47092B5B9814AA08873FF4F1DE61D4635DF68FC7F3A039A7DA5B0D00D', max_payload_bytes=1048576)",
        ("source_path", "adapter", "expected_source_sha256", "max_payload_bytes"),
    ),
    PublicApi(
        "api.legacy.validate_complete",
        "anyalgebra.legacy.models",
        "AlgMulManifest",
        "validate_complete",
        "AlgMulManifest.validate_complete(*, owned_partitions)",
        ("self", "owned_partitions"),
    ),
)


def _quickstart_text() -> str:
    """Read the one checked document from a path independent of the test CWD."""
    return QUICKSTART.read_text(encoding="utf-8")


def _object_for(api: PublicApi) -> Any:
    """Resolve the documented import and, when needed, its published method."""
    imported = getattr(importlib.import_module(api.module), api.imported_name)
    return imported if api.attribute is None else getattr(imported, api.attribute)


def test_scoped_import_map_resolves_all_23_api_ids_with_current_signatures() -> None:
    """The quickstart publishes only callable v0.0 seams from the scope registry."""
    document = _quickstart_text()
    assert len(PUBLIC_APIS) == 23
    for api in PUBLIC_APIS:
        expected_row = (
            f"| `{api.identifier}` | `{api.import_statement}` | "
            f"`{api.published_call}` |"
        )
        assert expected_row in document
        resolved = _object_for(api)
        assert callable(resolved)
        parameters = tuple(inspect.signature(resolved).parameters)
        assert parameters[: len(api.parameter_names)] == api.parameter_names


def test_all_python_fences_execute_from_an_unrelated_working_directory(
    tmp_path: Path,
) -> None:
    """Published code is parsed and run directly instead of copied into this test."""
    snippets = re.findall(r"```python\n(.*?)\n```", _quickstart_text(), flags=re.DOTALL)
    assert snippets
    environment = os.environ.copy()
    source_path = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = (
        source_path + os.pathsep + environment.get("PYTHONPATH", "")
    )
    for index, snippet in enumerate(snippets):
        script = tmp_path / f"quickstart-{index}.py"
        script.write_text(snippet, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=tmp_path,
            check=False,
            capture_output=True,
            text=True,
            timeout=1800,  # a hang guard, not a performance budget
            env=environment,
        )
        assert completed.returncode == 0, completed.stderr


def test_outcomes_cover_defined_partial_bounded_failed_and_rejection_boundaries() -> (
    None
):
    """Outcome branches remain tagged and invalid serialization stays outside them."""
    assert unwrap_defined(Defined("value")) == "value"
    with pytest.raises(UndefinedOutcomeError):
        unwrap_defined(Undefined("table cell is undefined"))
    with pytest.raises(IndeterminateOutcomeError):
        unwrap_defined(Indeterminate("finite bound exhausted", {"cases": 8}))
    with pytest.raises(FailedOutcomeError):
        unwrap_defined(Failed("backend returned a stable error", "execute"))
    with pytest.raises(SchemaError, match="unknown_schema_type"):
        SerializerRegistry().from_record(
            {"schemaType": "untrusted", "schemaVersion": 1}
        )


def test_quickstart_states_the_required_release_boundaries() -> None:
    """The concise guide preserves partiality, routes, bounds, and release limits."""
    document = " ".join(_quickstart_text().lower().split())
    for phrase in (
        "exact parents",
        "partial operations",
        "explicit conversion routes",
        "negative result and an inconclusive result",
        "optional wolfram adapter",
        "not an algmul correctness oracle",
        "does not provide specialized",
    ):
        assert phrase in document
