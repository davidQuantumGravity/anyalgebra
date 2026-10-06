"""Read-only validation for AnyAlgebra's machine-readable project contract."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePath, PureWindowsPath
from typing import Any


PROCESS_PATH = Path(".agents") / "project-process"
REGISTRIES: tuple[tuple[str, str], ...] = (
    ("features.json", "feature"),
    ("actions.json", "action"),
    ("api-contracts.json", "api"),
    ("components.json", "component"),
    ("user-stories.json", "story"),
    ("test-matrix.json", "test"),
    ("states.json", "state"),
    ("routes.json", "route"),
    ("events.json", "event"),
    ("permissions.json", "permission"),
)
REFERENCE_KINDS = {
    "featureIds": "feature",
    "actionIds": "action",
    "apiIds": "api",
    "componentIds": "component",
    "storyIds": "story",
    "testIds": "test",
    "stateIds": "state",
    "routeIds": "route",
    "eventIds": "event",
    "permissionIds": "permission",
}
ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
VERSION_PATTERN = re.compile(r"^v[0-9]+(?:\.[0-9]+)+$")


def _load_json(path: Path, errors: list[str], label: str) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"missing required registry: {label}")
    except (OSError, json.JSONDecodeError):
        errors.append(f"malformed JSON: {label}")
    return None


def _registry_ids(filename: str, payload: Any, errors: list[str]) -> set[str]:
    if not isinstance(payload, list):
        errors.append(f"registry must be a list: {filename}")
        return set()

    identifiers: set[str] = set()
    for entry in payload:
        if not isinstance(entry, Mapping):
            errors.append(f"registry entry must be an object: {filename}")
            continue
        identifier = entry.get("id")
        if not isinstance(identifier, str) or not ID_PATTERN.fullmatch(identifier):
            errors.append(f"invalid id in {filename}")
            continue
        if identifier in identifiers:
            errors.append(f"duplicate id in {filename}: {identifier}")
            continue
        identifiers.add(identifier)
    return identifiers


def _iter_references(payload: Any) -> Iterable[tuple[str, str]]:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key in REFERENCE_KINDS and isinstance(value, list):
                for identifier in value:
                    if isinstance(identifier, str):
                        yield key, identifier
            yield from _iter_references(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _iter_references(value)


def _validate_references(
    label: str,
    payload: Any,
    known_ids: Mapping[str, set[str]],
    available_kinds: set[str],
    errors: list[str],
) -> None:
    for field, identifier in _iter_references(payload):
        kind = REFERENCE_KINDS[field]
        if kind not in available_kinds:
            continue
        if identifier not in known_ids[kind]:
            errors.append(f"unknown {field} reference in {label}: {identifier}")


def _is_safe_scoped_path(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    windows_path = PureWindowsPath(value)
    native_path = PurePath(value)
    if windows_path.is_absolute() or windows_path.drive or native_path.is_absolute():
        return False
    return all(part not in {"", ".", ".."} for part in native_path.parts)


def _validate_queue(
    scope_version: str,
    queue: Any,
    progress: Any,
    errors: list[str],
) -> None:
    if not isinstance(queue, Mapping):
        errors.append("implementation queue must be an object")
        return
    queue_version = queue.get("version")
    if queue_version != scope_version:
        errors.append(
            "implementation queue version does not match scope version: "
            f"{queue_version} != {scope_version}"
        )
    queue_id = queue.get("queueId")
    tasks = queue.get("tasks")
    if not isinstance(tasks, list):
        errors.append("implementation queue tasks must be a list")
    else:
        task_ids: set[str] = set()
        for task in tasks:
            if not isinstance(task, Mapping):
                errors.append("implementation queue task must be an object")
                continue
            task_id = task.get("id")
            if not isinstance(task_id, str) or not task_id:
                errors.append("implementation queue task has invalid id")
                continue
            if task_id in task_ids:
                errors.append(f"duplicate implementation task id: {task_id}")
            task_ids.add(task_id)
            work = task.get("work")
            if not isinstance(work, Mapping):
                errors.append(f"implementation task has invalid work: {task_id}")
                continue
            allowed_paths = work.get("allowedPaths")
            deliverables = work.get("deliverables")
            if allowed_paths != deliverables:
                errors.append(
                    "implementation task scoped paths differ from deliverables: "
                    f"{task_id}"
                )
            if isinstance(allowed_paths, list):
                for path in allowed_paths:
                    if not _is_safe_scoped_path(path):
                        errors.append(f"unsafe scoped path in {task_id}: {path}")
            else:
                errors.append(
                    f"implementation task has invalid allowedPaths: {task_id}"
                )

    if not isinstance(progress, Mapping):
        errors.append("implementation progress must be an object")
        return
    if progress.get("version") != scope_version:
        errors.append("implementation progress version does not match scope version")
    if progress.get("queueId") != queue_id:
        errors.append(
            "implementation progress queueId does not match implementation queue"
        )


def validate(repository_root: Path | str | None = None) -> list[str]:
    """Return deterministic contract errors without changing ``repository_root``."""

    root = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else Path(repository_root).resolve()
    )
    process = root / PROCESS_PATH
    errors: list[str] = []
    config = _load_json(
        process / "process-config.json",
        errors,
        ".agents/project-process/process-config.json",
    )
    payloads: dict[str, Any] = {}
    known_ids: dict[str, set[str]] = {}
    for filename, kind in REGISTRIES:
        label = f".agents/project-process/{filename}"
        payload = _load_json(process / filename, errors, label)
        if payload is None:
            continue
        payloads[filename] = payload
        known_ids[kind] = _registry_ids(filename, payload, errors)
    available_kinds = set(known_ids)

    if not isinstance(config, Mapping):
        errors.append("process configuration must be an object")
        return errors
    versioning = config.get("versioning")
    if not isinstance(versioning, Mapping):
        errors.append("process configuration must define versioning")
        return errors
    active_version = versioning.get("activeVersion")
    versions_path = versioning.get("versionsPath")
    if not isinstance(active_version, str) or not VERSION_PATTERN.fullmatch(
        active_version
    ):
        errors.append("process configuration has invalid activeVersion")
        return errors
    if not isinstance(versions_path, str) or not _is_safe_scoped_path(versions_path):
        errors.append("process configuration has unsafe versionsPath")
        return errors

    scope_label = f".agents/project-process/{versions_path}/{active_version}/scope.json"
    scope = _load_json(
        process / versions_path / active_version / "scope.json", errors, scope_label
    )
    if not isinstance(scope, Mapping):
        if scope is not None:
            errors.append("scope must be an object")
        return errors
    scope_version = scope.get("version")
    if scope_version != active_version:
        errors.append(
            "scope version does not match active version: "
            f"{scope_version} != {active_version}"
        )
        return errors
    _validate_references("scope.json", scope, known_ids, available_kinds, errors)

    decisions = process / "decisions.md"
    decision_ids = (
        set(re.findall(r"ADR-[0-9]{4}", decisions.read_text(encoding="utf-8")))
        if decisions.is_file()
        else set()
    )
    decision_references = scope.get("decisionIds")
    if not isinstance(decision_references, list):
        errors.append("scope must define decisionIds as a list")
    else:
        for decision_id in decision_references:
            if decision_id not in decision_ids:
                errors.append(
                    f"unknown decisionIds reference in scope.json: {decision_id}"
                )

    for filename, payload in payloads.items():
        _validate_references(filename, payload, known_ids, available_kinds, errors)

    version_dir = process / versions_path / active_version / "implementation"
    queue = _load_json(
        version_dir / "implementation-tasks.json",
        errors,
        f".agents/project-process/{versions_path}/{active_version}/implementation/implementation-tasks.json",
    )
    progress = _load_json(
        version_dir / "implementation-progress.json",
        errors,
        f".agents/project-process/{versions_path}/{active_version}/implementation/implementation-progress.json",
    )
    if queue is not None and progress is not None:
        _validate_queue(active_version, queue, progress, errors)
    return errors


def main(argv: Sequence[str] | None = None) -> int:
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
        print("project contract valid")
        return 0
    print("project contract invalid")
    for error in errors:
        print(f"- {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
