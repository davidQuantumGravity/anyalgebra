"""Read-only validation for the pinned AnyAlgebra v0.0 contract IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast


SNAPSHOT_PATH = Path(".agents/project-process/versions/v0.0/contract-snapshot.json")
SOURCE_PATHS: tuple[str, ...] = (
    ".agents/project-process/process-config.json",
    ".agents/project-process/versions/v0.0/scope.json",
    ".agents/project-process/api-contracts.json",
    ".agents/project-process/test-matrix.json",
)
EXPECTED_SCOPE_VERSION = "v0.0"
EXPECTED_COUNTS = {"apiIds": 23, "testIds": 22}
EXPECTED_ORDERING = "scope-list-order"
SHA256_LENGTH = 64


def _load_json(path: Path, errors: list[str], label: str) -> object | None:
    try:
        return cast(object, json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        errors.append(f"missing {label}")
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        errors.append(f"malformed {label}")
    return None


def _source_hash(path: Path, errors: list[str], relative_path: str) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        errors.append(f"missing source file: {relative_path}")
    except OSError:
        errors.append(f"unreadable source file: {relative_path}")
    return None


def _string_list(value: object, label: str, errors: list[str]) -> list[str] | None:
    if not isinstance(value, list):
        errors.append(f"{label} must be a list")
        return None
    if not all(isinstance(identifier, str) for identifier in value):
        errors.append(f"{label} must contain only string IDs")
        return None
    return value


def _duplicates(identifiers: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for identifier in identifiers:
        if identifier in seen and identifier not in duplicates:
            duplicates.append(identifier)
        seen.add(identifier)
    return duplicates


def _validate_id_list(
    label: str,
    identifiers: list[str] | None,
    expected: list[str],
    errors: list[str],
    *,
    comparison_label: str | None = None,
    allow_appended: bool = False,
) -> None:
    if identifiers is None:
        return
    duplicates = _duplicates(identifiers)
    if duplicates:
        errors.append(f"{label} has duplicate IDs: {duplicates}")
    missing = [identifier for identifier in expected if identifier not in identifiers]
    if missing:
        errors.append(f"{label} is missing IDs: {missing}")
    extra = [identifier for identifier in identifiers if identifier not in expected]
    if extra and not allow_appended:
        errors.append(f"{label} has extra IDs: {extra}")
    ordered = (
        identifiers[: len(expected)] == expected
        if allow_appended
        else identifiers == expected
    )
    if comparison_label is not None and not ordered:
        errors.append(f"{label} do not match {comparison_label} order")


def _snapshot_sources(
    snapshot: Mapping[str, object], errors: list[str]
) -> dict[str, str] | None:
    source_entries = snapshot.get("sources")
    if not isinstance(source_entries, list):
        errors.append("snapshot sources must be a list")
        return None

    hashes: dict[str, str] = {}
    for entry in source_entries:
        if not isinstance(entry, Mapping):
            errors.append("snapshot source entry must be an object")
            continue
        path = entry.get("path")
        source_hash = entry.get("sha256")
        if not isinstance(path, str) or path not in SOURCE_PATHS:
            errors.append("snapshot source entry has an invalid path")
            continue
        if (
            not isinstance(source_hash, str)
            or len(source_hash) != SHA256_LENGTH
            or any(character not in "0123456789abcdef" for character in source_hash)
        ):
            errors.append(f"snapshot source entry has an invalid sha256: {path}")
            continue
        if path in hashes:
            errors.append(f"snapshot sources has duplicate path: {path}")
            continue
        hashes[path] = source_hash

    missing = [path for path in SOURCE_PATHS if path not in hashes]
    if missing:
        errors.append(f"snapshot sources is missing paths: {missing}")
    return hashes


def _registry_ids(payload: object, label: str, errors: list[str]) -> list[str] | None:
    if not isinstance(payload, list):
        errors.append(f"{label} must be a list")
        return None
    identifiers: list[str] = []
    for entry in payload:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("id"), str):
            errors.append(f"{label} has an entry without a string id")
            return None
        identifiers.append(entry["id"])
    return identifiers


def _validate_snapshot_shape(
    snapshot: Mapping[str, object], errors: list[str]
) -> tuple[list[str] | None, list[str] | None, dict[str, str] | None]:
    if snapshot.get("schemaVersion") != "1.0":
        errors.append("snapshot schemaVersion must be 1.0")
    if snapshot.get("scopeVersion") != EXPECTED_SCOPE_VERSION:
        errors.append(f"snapshot scopeVersion must be {EXPECTED_SCOPE_VERSION}")
    if snapshot.get("ordering") != EXPECTED_ORDERING:
        errors.append(f"snapshot ordering must be {EXPECTED_ORDERING}")

    counts = snapshot.get("counts")
    if not isinstance(counts, Mapping):
        errors.append("snapshot counts must be an object")
    else:
        for field, expected_count in EXPECTED_COUNTS.items():
            actual_count = counts.get(field)
            if isinstance(actual_count, bool) or actual_count != expected_count:
                errors.append(
                    f"snapshot {field} count must be {expected_count}: {actual_count}"
                )

    api_ids = _string_list(snapshot.get("apiIds"), "snapshot apiIds", errors)
    test_ids = _string_list(snapshot.get("testIds"), "snapshot testIds", errors)
    for field, identifiers in (("apiIds", api_ids), ("testIds", test_ids)):
        if identifiers is not None and len(identifiers) != EXPECTED_COUNTS[field]:
            errors.append(
                f"snapshot {field} length must be {EXPECTED_COUNTS[field]}: "
                f"{len(identifiers)}"
            )
    if api_ids is not None:
        _validate_id_list("snapshot apiIds", api_ids, api_ids, errors)
    if test_ids is not None:
        _validate_id_list("snapshot testIds", test_ids, test_ids, errors)
    return api_ids, test_ids, _snapshot_sources(snapshot, errors)


def validate(repository_root: Path | str | None = None) -> list[str]:
    """Return deterministic drift errors without modifying ``repository_root``."""

    root = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else Path(repository_root).resolve()
    )
    errors: list[str] = []
    snapshot = _load_json(root / SNAPSHOT_PATH, errors, "snapshot JSON")
    if snapshot is None:
        return errors
    if not isinstance(snapshot, Mapping):
        return [*errors, "snapshot must be an object"]

    api_ids, test_ids, expected_hashes = _validate_snapshot_shape(snapshot, errors)
    if expected_hashes is None:
        return errors

    scope_version = snapshot.get("scopeVersion")
    payloads: dict[str, object] = {}
    for relative_path in SOURCE_PATHS:
        source_path = root / relative_path
        payload = _load_json(source_path, errors, f"source JSON: {relative_path}")
        if payload is not None:
            payloads[relative_path] = payload

    config = payloads.get(SOURCE_PATHS[0])
    active_version = None
    if isinstance(config, Mapping):
        versioning = config.get("versioning")
        active_version = (
            versioning.get("activeVersion") if isinstance(versioning, Mapping) else None
        )
    advanced = isinstance(active_version, str) and active_version != scope_version
    for relative_path in SOURCE_PATHS:
        source_path = root / relative_path
        shared_append_only = relative_path in (SOURCE_PATHS[2], SOURCE_PATHS[3])
        compare_hash = not (
            advanced and (relative_path == SOURCE_PATHS[0] or shared_append_only)
        )
        source_hash = _source_hash(source_path, errors, relative_path)
        if (
            compare_hash
            and source_hash is not None
            and expected_hashes.get(relative_path) != source_hash
        ):
            errors.append(f"source hash mismatch: {relative_path}")

    if isinstance(config, Mapping):
        versioning = config.get("versioning")
        active_version = (
            versioning.get("activeVersion") if isinstance(versioning, Mapping) else None
        )
        if not isinstance(active_version, str):
            errors.append("process-config.json activeVersion must be a string")
    elif SOURCE_PATHS[0] in payloads:
        errors.append("process-config.json must be an object")

    scope = payloads.get(SOURCE_PATHS[1])
    if isinstance(scope, Mapping):
        if scope.get("version") != scope_version:
            errors.append(
                "scope version does not match snapshot: "
                f"{scope.get('version')} != {scope_version}"
            )
        scope_api_ids = _string_list(scope.get("apiIds"), "scope.json apiIds", errors)
        scope_test_ids = _string_list(
            scope.get("testIds"), "scope.json testIds", errors
        )
        if api_ids is not None:
            _validate_id_list(
                "scope.json apiIds",
                scope_api_ids,
                api_ids,
                errors,
                comparison_label="snapshot",
            )
        if test_ids is not None:
            _validate_id_list(
                "scope.json testIds",
                scope_test_ids,
                test_ids,
                errors,
                comparison_label="snapshot",
            )
    elif SOURCE_PATHS[1] in payloads:
        errors.append("scope.json must be an object")

    api_registry_ids = _registry_ids(
        payloads.get(SOURCE_PATHS[2]), "api-contracts.json", errors
    )
    test_registry_ids = _registry_ids(
        payloads.get(SOURCE_PATHS[3]), "test-matrix.json", errors
    )
    if api_ids is not None:
        _validate_id_list(
            "api-contracts.json IDs",
            api_registry_ids,
            api_ids,
            errors,
            comparison_label="snapshot",
            allow_appended=advanced,
        )
    if test_ids is not None:
        _validate_id_list(
            "test-matrix.json IDs",
            test_registry_ids,
            test_ids,
            errors,
            comparison_label="snapshot",
            allow_appended=advanced,
        )
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checker and return a shell status without writing a snapshot."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repository_root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    arguments = parser.parse_args(argv)
    errors = validate(arguments.repository_root)
    if not errors:
        print("contract snapshot valid")
        return 0
    print("contract snapshot invalid")
    for error in errors:
        print(f"- {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
