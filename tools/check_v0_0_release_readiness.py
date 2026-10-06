"""Fail-closed release-readiness checks for the bounded AnyAlgebra v0.0 release.

This checker deliberately verifies release evidence, not mathematical truth.  It
does not treat a passing test suite, an AlgMul capture, or an exit-demo fixture
as confirmation of a physics result or of all legacy behavior.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import subprocess
import sys
from typing import cast


PROCESS = Path(".agents/project-process")
VERSION_DIR = PROCESS / "versions" / "v0.0"
REGISTRY_PATHS = (
    PROCESS / "features.json",
    PROCESS / "actions.json",
    PROCESS / "user-stories.json",
    PROCESS / "test-matrix.json",
    PROCESS / "components.json",
)
REQUIRED_DOCUMENTS = (
    Path("README.md"),
    Path("CHANGELOG.md"),
    Path("docs/legacy/algmul-parity.md"),
    VERSION_DIR / "acceptance-checklist.md",
    VERSION_DIR / "release-notes.md",
    VERSION_DIR / "carryover.md",
)
STALE_RELEASE_PHRASES = (
    "pre-v0.0",
    "no implementation exists yet",
    "remains planned",
    "remaining audit work before implementation",
)
EXIT_DEMO = Path("examples/v0_0_exit_demo.py")
EXIT_DEMO_TEST = Path("tests/integration/test_v0_0_exit_demo.py")
COMPATIBLE_PATCH_VERSIONS = frozenset({"0.0.0", "0.0.1"})


class ReadinessResult:
    """A deterministic, non-mutating local release-readiness result."""

    def __init__(self, errors: tuple[str, ...]) -> None:
        self.errors = errors

    @property
    def ready(self) -> bool:
        """Whether every fail-closed local release condition holds."""
        return not self.errors


def _is_compatible_or_later_version(version: str) -> bool:
    if version in COMPATIBLE_PATCH_VERSIONS:
        return True
    fields = version.split(".", 2)
    if len(fields) < 2:
        return False
    try:
        major_minor = (int(fields[0]), int(fields[1]))
    except ValueError:
        return False
    return major_minor > (0, 0)


def _read_json(path: Path, errors: list[str]) -> object | None:
    try:
        return cast(object, json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        errors.append(f"missing required JSON: {path.as_posix()}")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        errors.append(f"malformed required JSON: {path.as_posix()}")
    return None


def _version_from_source(path: Path, errors: list[str]) -> str | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        errors.append("package version source is unreadable")
        return None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__version__"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    errors.append("package version source has no literal __version__")
    return None


def _check_documents(root: Path, errors: list[str], *, strict_release: bool) -> None:
    for relative in REQUIRED_DOCUMENTS:
        path = root / relative
        try:
            text = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            errors.append(
                f"missing or unreadable release document: {relative.as_posix()}"
            )
            continue
        if not text:
            errors.append(f"empty release document: {relative.as_posix()}")

    readme = (root / "README.md").read_text(encoding="utf-8")
    release_text = "\n".join(
        (root / relative).read_text(encoding="utf-8").lower()
        for relative in (
            Path("README.md"),
            Path("docs/legacy/algmul-parity.md"),
            VERSION_DIR / "release-notes.md",
        )
    )
    for phrase in STALE_RELEASE_PHRASES:
        if phrase in release_text:
            errors.append(f"release documentation retains stale phrase: {phrase}")
    if "no python package can be installed" in readme.lower():
        errors.append("README retains the pre-v0.0 installation claim")
    if "No project/scientific claim" not in readme:
        errors.append("README lacks the release claim boundary")

    parity = (root / "docs/legacy/algmul-parity.md").read_text(encoding="utf-8")
    if "licensed Wolfram" not in parity:
        errors.append("AlgMul parity document lacks the licensed-Wolfram limitation")
    if strict_release:
        checklist = (root / VERSION_DIR / "acceptance-checklist.md").read_text(
            encoding="utf-8"
        )
        if any("- [ ]" in line for line in checklist.splitlines()):
            errors.append("release acceptance checklist has unchecked items")


def _check_scope_and_registries(
    root: Path, errors: list[str], *, strict_release: bool
) -> None:
    scope = _read_json(root / VERSION_DIR / "scope.json", errors)
    if not isinstance(scope, Mapping):
        return
    if scope.get("version") != "v0.0":
        errors.append("scope does not identify v0.0")
    accepted_statuses = (
        {"released"}
        if strict_release
        else {
            "release-candidate",
            "released",
        }
    )
    if scope.get("status") not in accepted_statuses:
        expected = "released" if strict_release else "release-candidate or released"
        errors.append(f"scope is not marked {expected}")

    for relative in REGISTRY_PATHS:
        payload = _read_json(root / relative, errors)
        if not isinstance(payload, list):
            continue
        for entry in payload:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("id"), str):
                errors.append(f"malformed registry entry: {relative.as_posix()}")
                continue
            owner = entry.get("version")
            if isinstance(owner, str) and owner not in {"v0.0", "v0.0.1"}:
                continue
            if entry.get("status") != "verified":
                errors.append(f"unverified scoped registry entry: {entry['id']}")


def _release_hash_candidates(path: Path) -> set[str]:
    """Return the live hash and an exact pre-append JSON-list hash when present."""
    raw = path.read_bytes()
    candidates = {hashlib.sha256(raw).hexdigest().upper()}
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return candidates
    if not isinstance(payload, list):
        return candidates
    first_later = next(
        (
            entry
            for entry in payload
            if isinstance(entry, Mapping)
            and isinstance(entry.get("version"), str)
            and entry["version"] not in {"v0.0", "v0.0.1"}
        ),
        None,
    )
    if not isinstance(first_later, Mapping) or not isinstance(
        first_later.get("id"), str
    ):
        return candidates
    needle = json.dumps(first_later["id"]).encode("utf-8")
    identifier_position = raw.find(needle)
    record_start = raw.rfind(b"{", 0, identifier_position)
    separator = raw.rfind(b",", 0, record_start)
    if identifier_position < 0 or record_start < 0 or separator < 0:
        return candidates
    newline = b"\r\n" if b"\r\n" in raw[:record_start] else b"\n"
    historical = raw[:separator] + newline + b"]" + newline
    candidates.add(hashlib.sha256(historical).hexdigest().upper())
    return candidates


def _check_recorded_release_evidence(root: Path, errors: list[str]) -> None:
    """Require recorded final-gate evidence without rerunning expensive gates."""
    path = root / VERSION_DIR / "test-results.md"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        errors.append("missing readable final release test-results ledger")
        return
    lowered = text.lower()
    required_tokens = (
        "v00-100",
        "fullpytest",
        "coverage",
        "traceability",
        "exit demo",
        "artifact",
        "clean-install",
        "reproducibility",
        "v00-100 supersession",
    )
    missing = [token for token in required_tokens if token not in lowered]
    if missing:
        errors.append("final release evidence is incomplete: " + ", ".join(missing))
    for relative in (
        PROCESS / "actions.json",
        VERSION_DIR / "contract-snapshot.json",
    ):
        if not any(
            candidate in text.upper()
            for candidate in _release_hash_candidates(root / relative)
        ):
            errors.append(
                "V00-099 mutable evidence lacks V00-100 live hash: "
                + relative.as_posix()
            )


def _check_exit_demo(root: Path, errors: list[str], *, execute: bool) -> None:
    if not (root / EXIT_DEMO).is_file():
        errors.append("missing v0.0 semantic exit demo")
    if not (root / EXIT_DEMO_TEST).is_file():
        errors.append("missing v0.0 semantic exit-demo integration test")
    if not execute or errors:
        return
    try:
        completed = subprocess.run(
            [sys.executable, str(root / EXIT_DEMO)],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        errors.append(f"v0.0 semantic exit demo could not run: {type(error).__name__}")
        return
    if completed.returncode:
        errors.append("v0.0 semantic exit demo exited nonzero")
        return
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        errors.append("v0.0 semantic exit demo did not emit JSON")
        return
    round_trips = payload.get("round_trips") if isinstance(payload, Mapping) else None
    record_types = payload.get("record_types") if isinstance(payload, Mapping) else None
    if (
        not isinstance(round_trips, Mapping)
        or not round_trips
        or not all(value is True for value in round_trips.values())
        or not isinstance(record_types, list)
        or len(record_types) != 5
    ):
        errors.append(
            "v0.0 semantic exit demo lacks five successful semantic round trips"
        )


def validate(
    root: Path | str | None = None,
    *,
    execute_exit_demo: bool = False,
    strict_release: bool = False,
) -> ReadinessResult:
    """Return a fail-closed readiness result without changing the repository."""
    repository = (
        Path(__file__).resolve().parents[1] if root is None else Path(root).resolve()
    )
    errors: list[str] = []
    version = _version_from_source(repository / "src/anyalgebra/_version.py", errors)
    if version is not None and not _is_compatible_or_later_version(version):
        errors.append(f"package version is not a compatible v0.0 patch: {version}")
    _check_documents(repository, errors, strict_release=strict_release)
    _check_scope_and_registries(repository, errors, strict_release=strict_release)
    _check_exit_demo(repository, errors, execute=execute_exit_demo)
    if strict_release:
        _check_recorded_release_evidence(repository, errors)
    return ReadinessResult(tuple(sorted(set(errors))))


def main(argv: Sequence[str] | None = None) -> int:
    """Print the local release-readiness decision and return its shell status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository_root", nargs="?", type=Path)
    parser.add_argument("--execute-exit-demo", action="store_true")
    parser.add_argument("--release", action="store_true")
    arguments = parser.parse_args(argv)
    result = validate(
        arguments.repository_root,
        execute_exit_demo=arguments.execute_exit_demo,
        strict_release=arguments.release,
    )
    if result.ready:
        print("v0.0 release readiness valid")
        return 0
    print("v0.0 release readiness blocked")
    for error in result.errors:
        print(f"- {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
