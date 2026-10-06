"""Executable contracts for the public multilinear-evaluation example."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any, cast


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "multilinear_evaluation.py"
GUIDE = ROOT / "docs" / "guides" / "quaternions-and-octonions.md"


def _example() -> dict[str, Any]:
    return runpy.run_path(str(EXAMPLE))


def _python_block_after(heading: str) -> str:
    text = GUIDE.read_text(encoding="utf-8")
    section = text.split(heading, maxsplit=1)[1]
    return section.split("```python", maxsplit=1)[1].split("```", maxsplit=1)[0]


def test_example_constructs_generic_nonassociative_and_ternary_controls() -> None:
    namespace = _example()
    built = cast(dict[str, Any], namespace["build_example"]())

    generic = built["generic"]
    a = generic.module.element({0: 1})
    assert built["left_nested"] == a
    assert built["right_nested"] == generic.module.zero()
    assert built["ternary_result"] == generic.module.element({1: -30})


def test_example_evaluates_sparse_composition_products_and_split_zero_divisors() -> (
    None
):
    report = cast(dict[str, Any], _example()["run_example"]())

    assert report["quaternion"] == {
        "fixture": "algmul.H.v1",
        "product": [[2, -1], [3, 3]],
    }
    assert report["octonion"] == {
        "fixture": "algmul.O.v1",
        "product": [[2, 1], [3, 1], [4, 1], [5, 1]],
    }
    assert report["split_zero_divisors"] == {
        "algmul.Hs.v1": [],
        "algmul.Os.v1": [],
    }
    assert report["identity_checks"] == {
        "algmul.H.v1": True,
        "algmul.Hs.v1": True,
        "algmul.O.v1": True,
        "algmul.Os.v1": True,
    }


def test_example_report_and_command_output_are_canonical_and_deterministic() -> None:
    namespace = _example()
    first = cast(dict[str, Any], namespace["run_example"]())
    second = cast(dict[str, Any], namespace["run_example"]())
    completed = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert first == second
    assert (
        completed.stdout
        == json.dumps(first, sort_keys=True, separators=(",", ":")) + "\n"
    )
    assert first["scope"] == (
        "exact finite-table evaluation controls only; no classification or "
        "physics claim"
    )


def test_documented_public_evaluator_snippet_executes_verbatim() -> None:
    block = _python_block_after("## Multiply arbitrary sparse elements")

    assert "evaluate_multilinear" in block
    assert "def multiply" not in block
    exec(compile(block, str(GUIDE), "exec"), {})


def test_capability_page_states_the_milestone_and_release_boundary() -> None:
    text = (ROOT / "docs" / "capabilities.md").read_text(encoding="utf-8")

    assert "Generic multilinear element evaluation" in text
    assert "locally complete 0.1.0 source milestone" in text
    assert "V01-060" in text
    assert "not a tagged or" in text
    assert "published release" in text
