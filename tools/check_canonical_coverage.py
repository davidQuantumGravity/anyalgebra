"""Fail closed unless canonical AnyAlgebra source has exact full coverage."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast


DEFAULT_REPORT = Path("build/coverage-v001-final.json")
SOURCE_PREFIX = "src/anyalgebra/"
# The v0.0.1 canonical report is immutable release evidence. Research modules
# have their own executable receipts and must not silently mutate that ledger.
EXPERIMENTAL_PREFIX = "src/anyalgebra/experimental/"


@dataclass(frozen=True, slots=True)
class CoverageResult:
    """One deterministic canonical-tree coverage verdict."""

    files: int
    covered_statements: int
    statements: int
    covered_branches: int
    branches: int
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.errors


def _integer(value: object, label: str, errors: list[str]) -> int:
    if type(value) is not int or value < 0:
        errors.append(f"{label} must be a nonnegative exact integer")
        return 0
    return value


def validate(
    report_path: Path | str = DEFAULT_REPORT,
    *,
    root: Path | str | None = None,
    require_source_closure: bool = True,
) -> CoverageResult:
    """Validate exact coverage, optionally requiring the current source set.

    Frozen historical reports can still prove exact coverage of every source
    file they record after later versions append new modules.  Callers checking
    a current report retain the stricter source-closure default.
    """
    repository = (
        Path(__file__).resolve().parents[1] if root is None else Path(root).resolve()
    )
    report = repository / report_path
    errors: list[str] = []
    try:
        payload = cast(object, json.loads(report.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return CoverageResult(0, 0, 0, 0, 0, ("coverage report is unreadable",))
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("files"), Mapping
    ):
        return CoverageResult(0, 0, 0, 0, 0, ("coverage report has no file map",))

    file_map = cast(Mapping[object, object], payload["files"])
    canonical: dict[str, Mapping[object, object]] = {}
    for raw_name, raw_record in file_map.items():
        if not isinstance(raw_name, str):
            errors.append("coverage file name must be a string")
            continue
        name = raw_name.replace("\\", "/")
        if not name.startswith(SOURCE_PREFIX) or name.startswith(EXPERIMENTAL_PREFIX):
            continue
        if not isinstance(raw_record, Mapping):
            errors.append(f"coverage record is malformed: {name}")
            continue
        if name in canonical:
            errors.append(f"duplicate canonical coverage record: {name}")
            continue
        canonical[name] = raw_record

    expected = {
        path.relative_to(repository).as_posix()
        for path in (repository / SOURCE_PREFIX).rglob("*.py")
        if not path.relative_to(repository).as_posix().startswith(EXPERIMENTAL_PREFIX)
    }
    missing_files = sorted(expected - set(canonical))
    extra_files = sorted(set(canonical) - expected)
    if require_source_closure and missing_files:
        errors.append("missing canonical source records: " + ", ".join(missing_files))
    if extra_files:
        errors.append("unexpected canonical source records: " + ", ".join(extra_files))

    covered_statements = statements = covered_branches = branches = 0
    for name, record in sorted(canonical.items()):
        summary = record.get("summary")
        if not isinstance(summary, Mapping):
            errors.append(f"coverage summary is malformed: {name}")
            continue
        file_statements = _integer(
            summary.get("num_statements"), f"{name} statements", errors
        )
        file_covered_statements = _integer(
            summary.get("covered_lines"), f"{name} covered statements", errors
        )
        file_branches = _integer(
            summary.get("num_branches"), f"{name} branches", errors
        )
        file_covered_branches = _integer(
            summary.get("covered_branches"), f"{name} covered branches", errors
        )
        statements += file_statements
        covered_statements += file_covered_statements
        branches += file_branches
        covered_branches += file_covered_branches
        if file_covered_statements != file_statements:
            errors.append(
                f"incomplete statement coverage: {name} "
                f"({file_covered_statements}/{file_statements})"
            )
        if file_covered_branches != file_branches:
            errors.append(
                f"incomplete branch coverage: {name} "
                f"({file_covered_branches}/{file_branches})"
            )

    return CoverageResult(
        len(canonical),
        covered_statements,
        statements,
        covered_branches,
        branches,
        tuple(errors),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", nargs="?", default=str(DEFAULT_REPORT))
    parser.add_argument("--root", default=None)
    args = parser.parse_args(argv)
    result = validate(args.report, root=args.root)
    if result.errors:
        for error in result.errors:
            print(f"ERROR: {error}")
        return 1
    print(
        "canonical coverage complete: "
        f"{result.files} files, "
        f"{result.covered_statements}/{result.statements} statements, "
        f"{result.covered_branches}/{result.branches} branches"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
