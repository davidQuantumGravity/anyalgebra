"""Executable contract for the mathematics-first v0.1 census manual."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / "docs" / "guides" / "finite-algebra-census.md"


def _text() -> str:
    return MANUAL.read_text(encoding="utf-8")


def test_manual_is_linked_from_both_public_entrypoints() -> None:
    expected = "docs/guides/finite-algebra-census.md"
    assert expected in (ROOT / "README.md").read_text(encoding="utf-8")
    assert "guides/finite-algebra-census.md" in (ROOT / "docs" / "README.md").read_text(
        encoding="utf-8"
    )


def test_manual_separates_mathematical_domains_and_evidence_statuses() -> None:
    normalized = " ".join(_text().lower().split())
    for phrase in (
        "finite-basis vector algebra",
        "finite-carrier operation table",
        "proof within a declared finite domain",
        "counterexample",
        "exhaustive corpus result",
        "bounded and inconclusive",
        "rejection-only invariant",
        "experimental context",
        "no physics claim",
    ):
        assert phrase in normalized


def test_manual_maps_every_stable_v0_1_api_and_reference_result() -> None:
    document = _text()
    for name in (
        "evaluate_multilinear",
        "CensusSpec.create",
        "enumerate_reference",
        "canonicalize_operation_table",
        "compare_isomorphism",
        "assemble_finite_carrier_analysis",
        "build_finite_algebra_atlas",
        "query_atlas",
    ):
        assert f"`{name}`" in document
    for result in (
        "19,683",
        "729",
        "129",
        "862",
        "3^9",
        "3^6",
    ):
        assert result in document


def test_every_python_block_is_an_asserting_executable_example(tmp_path: Path) -> None:
    snippets = re.findall(r"```python\n(.*?)\n```", _text(), flags=re.DOTALL)
    assert len(snippets) >= 5
    environment = os.environ.copy()
    source = str(ROOT / "src")
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    for index, snippet in enumerate(snippets):
        assert "assert " in snippet
        script = tmp_path / f"finite-algebra-manual-{index}.py"
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


def test_manual_contains_no_unasserted_transcript_blocks() -> None:
    document = _text()
    assert "```text" not in document
    assert ">>>" not in document
