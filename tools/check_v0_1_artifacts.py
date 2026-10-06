"""Build, inspect, and clean-install the exact AnyAlgebra v0.1 artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, cast


EXPECTED_NAME = "anyalgebra"
EXPECTED_VERSION = "0.1.0"
EXPECTED_REQUIRES_PYTHON = ">=3.11"
BASE_TOOL = Path(__file__).with_name("check_release_artifacts.py")


class V01ArtifactError(RuntimeError):
    """A stable v0.1-specific artifact contract failure."""


def _base_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_anyalgebra_release_artifact_base", BASE_TOOL
    )
    if spec is None or spec.loader is None:
        raise V01ArtifactError("base artifact checker is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _field(value: object, name: str) -> object:
    if not hasattr(value, name):
        raise V01ArtifactError(f"artifact evidence lacks {name}")
    return getattr(value, name)


def _tuple(value: object, label: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise V01ArtifactError(f"{label} must be a tuple")
    return value


def _artifact_record(value: object) -> dict[str, object]:
    record = {
        "filename": _field(value, "filename"),
        "kind": _field(value, "kind"),
        "memberCount": _field(value, "member_count"),
        "metadataName": _field(value, "metadata_name"),
        "metadataVersion": _field(value, "metadata_version"),
        "requiresPython": _field(value, "requires_python"),
        "runtimeDependencies": list(
            _tuple(_field(value, "runtime_dependencies"), "runtime dependencies")
        ),
        "sha256": _field(value, "sha256"),
    }
    if (
        not all(
            type(record[key]) is str
            for key in (
                "filename",
                "kind",
                "metadataName",
                "metadataVersion",
                "requiresPython",
                "sha256",
            )
        )
        or type(record["memberCount"]) is not int
    ):
        raise V01ArtifactError("artifact inspection fields are malformed")
    if (
        record["metadataName"] != EXPECTED_NAME
        or record["metadataVersion"] != EXPECTED_VERSION
        or record["requiresPython"] != EXPECTED_REQUIRES_PYTHON
        or record["runtimeDependencies"]
    ):
        raise V01ArtifactError("artifact metadata does not match v0.1")
    digest = cast(str, record["sha256"])
    if len(digest) != 64 or any(
        character not in "0123456789ABCDEFabcdef" for character in digest
    ):
        raise V01ArtifactError("artifact SHA-256 is malformed")
    return record


def _installation_record(value: object) -> dict[str, object]:
    attempted = _tuple(
        _field(value, "optional_modules_attempted"), "optional import attempts"
    )
    imported = _tuple(_field(value, "optional_modules_imported"), "optional imports")
    verified = _field(value, "reference_backend_verified")
    version = _field(value, "version")
    artifact = _field(value, "artifact")
    if (
        type(artifact) is not str
        or version != EXPECTED_VERSION
        or attempted
        or imported
        or verified is not True
    ):
        raise V01ArtifactError("clean-install evidence does not match v0.1")
    return {
        "artifact": artifact,
        "optionalModulesAttempted": [],
        "optionalModulesImported": [],
        "referenceBackendVerified": True,
        "version": EXPECTED_VERSION,
    }


def audit(
    repository_root: Path | str,
    *,
    artifacts: Iterable[Path | str] | None = None,
    inspect_only: bool = False,
    allow_network: bool = False,
    runner: Callable[..., Any] | None = None,
) -> dict[str, object]:
    """Return deterministic evidence for exactly one wheel and one sdist."""
    root = Path(repository_root).resolve()
    base = _base_tool()
    metadata = base.expected_metadata(root)
    if (
        metadata.name != EXPECTED_NAME
        or metadata.version != EXPECTED_VERSION
        or metadata.requires_python != EXPECTED_REQUIRES_PYTHON
        or metadata.runtime_dependencies
    ):
        raise V01ArtifactError(
            "repository package version must be 0.1.0 with no runtime dependencies"
        )
    execute = base.run_release_artifact_check if runner is None else runner
    result = execute(
        root,
        allow_network=allow_network,
        artifacts=artifacts,
        inspect_only=inspect_only,
    )
    inspection = _field(result, "inspection")
    errors = _tuple(_field(inspection, "errors"), "inspection errors")
    if errors or _field(inspection, "ok") is not True:
        raise V01ArtifactError("underlying archive inspection failed")
    artifact_records = tuple(
        sorted(
            (
                _artifact_record(value)
                for value in _tuple(_field(inspection, "artifacts"), "artifacts")
            ),
            key=lambda item: (str(item["kind"]), str(item["filename"])),
        )
    )
    if len(artifact_records) != 2 or {item["kind"] for item in artifact_records} != {
        "wheel",
        "sdist",
    }:
        raise V01ArtifactError("artifact set must contain one wheel and one sdist")
    raw_installations = _tuple(_field(result, "installations"), "installations")
    if inspect_only:
        if raw_installations:
            raise V01ArtifactError("inspect-only run contains installation evidence")
        installation_records: tuple[dict[str, object], ...] = ()
    else:
        installation_records = tuple(
            sorted(
                (_installation_record(value) for value in raw_installations),
                key=lambda item: str(item["artifact"]),
            )
        )
        if {item["artifact"] for item in installation_records} != {
            item["filename"] for item in artifact_records
        }:
            raise V01ArtifactError(
                "clean-install evidence must cover both distribution artifacts"
            )
    return {
        "artifacts": list(artifact_records),
        "installations": list(installation_records),
        "name": EXPECTED_NAME,
        "ok": True,
        "schemaType": "anyalgebra.v0.1.artifact-audit",
        "schemaVersion": 1,
        "version": EXPECTED_VERSION,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument("--artifact", action="append", type=Path, dest="artifacts")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        record = audit(
            arguments.repository,
            artifacts=arguments.artifacts,
            inspect_only=arguments.inspect_only,
            allow_network=arguments.allow_network,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        stage = getattr(error, "stage", "v0.1-artifact")
        print(
            json.dumps(
                {
                    "detail": "artifact gate failed",
                    "ok": False,
                    "stage": stage if type(stage) is str else "v0.1-artifact",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
