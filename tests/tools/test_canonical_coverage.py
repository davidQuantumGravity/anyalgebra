"""Tests for the exact canonical-source coverage gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parents[2]
TOOL_PATH = HERE / "tools" / "check_canonical_coverage.py"
SPEC = importlib.util.spec_from_file_location("canonical_coverage", TOOL_PATH)
assert SPEC and SPEC.loader
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def _repository(tmp_path: Path, summary: dict[str, int]) -> tuple[Path, Path]:
    source = tmp_path / "src/anyalgebra/__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n", encoding="utf-8")
    report = tmp_path / "coverage.json"
    report.write_text(
        json.dumps(
            {
                "files": {
                    "src\\anyalgebra\\__init__.py": {"summary": summary},
                    "C:/Temp/pytest/anyalgebra/__init__.py": {
                        "summary": {
                            "num_statements": 1,
                            "covered_lines": 0,
                            "num_branches": 0,
                            "covered_branches": 0,
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return tmp_path, report


@pytest.mark.skipif(
    not (HERE / checker.DEFAULT_REPORT).is_file(),
    reason="the frozen local coverage report is a regenerable build artifact",
)
def test_current_canonical_coverage_is_exactly_complete() -> None:
    result = checker.validate(root=HERE, require_source_closure=False)
    assert result.errors == ()
    assert result.files == 60
    assert (result.covered_statements, result.statements) == (14578, 14578)
    assert (result.covered_branches, result.branches) == (4960, 4960)


def test_gate_ignores_temporary_copies_but_rejects_real_gaps(tmp_path: Path) -> None:
    root, report = _repository(
        tmp_path,
        {
            "num_statements": 2,
            "covered_lines": 1,
            "num_branches": 2,
            "covered_branches": 1,
        },
    )
    result = checker.validate(report, root=root)
    assert result.errors == (
        "incomplete statement coverage: src/anyalgebra/__init__.py (1/2)",
        "incomplete branch coverage: src/anyalgebra/__init__.py (1/2)",
    )


def test_gate_keeps_research_experiments_outside_the_frozen_release(
    tmp_path: Path,
) -> None:
    root, report = _repository(
        tmp_path,
        {
            "num_statements": 1,
            "covered_lines": 1,
            "num_branches": 0,
            "covered_branches": 0,
        },
    )
    experimental = root / "src/anyalgebra/experimental/candidate.py"
    experimental.parent.mkdir(parents=True)
    experimental.write_text("candidate = 1\n", encoding="utf-8")

    result = checker.validate(report, root=root)

    assert result.errors == ()
    assert result.files == 1


def test_gate_rejects_missing_source_records_and_malformed_reports(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src/anyalgebra/__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("", encoding="utf-8")
    report = tmp_path / "coverage.json"
    report.write_text('{"files": {}}', encoding="utf-8")
    result = checker.validate(report, root=tmp_path)
    assert result.errors == (
        "missing canonical source records: src/anyalgebra/__init__.py",
    )
    assert checker.validate(tmp_path / "absent.json", root=tmp_path).errors == (
        "coverage report is unreadable",
    )


def test_frozen_report_can_allow_only_later_source_additions(tmp_path: Path) -> None:
    root, report = _repository(
        tmp_path,
        {
            "num_statements": 1,
            "covered_lines": 1,
            "num_branches": 0,
            "covered_branches": 0,
        },
    )
    added = root / "src/anyalgebra/later.py"
    added.write_text("later = 1\n", encoding="utf-8")

    strict = checker.validate(report, root=root)
    historical = checker.validate(report, root=root, require_source_closure=False)

    assert strict.errors == (
        "missing canonical source records: src/anyalgebra/later.py",
    )
    assert historical.errors == ()
    assert historical.files == 1
