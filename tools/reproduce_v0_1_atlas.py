"""Rebuild and compare the v0.1 reference atlas from wheel and sdist installs."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from typing import cast


BASE_TOOL = Path(__file__).with_name("check_release_artifacts.py")
SPEC_FIXTURE = Path("tests/fixtures/v01_atlas_spec.json")
EXPECTED_VERSION = "0.1.0"
EXPECTED_ATLAS_HASH = (
    "sha256:245384be017e38895d684456272eddb5e81a2974220debf348f41e17561eb843"
)
EXPECTED_SPEC_HASH = (
    "sha256:361253fe9e43ad6aac9dadcc3aa8d0fc97572a4d8f58939aa282134d1a596dea"
)
EXPECTED_QUERY_COUNTS = {
    "all": 129,
    "associative": 12,
    "idempotent": 7,
    "uniqueIdentity": 15,
}
EXPECTED_OBJECT_COUNT = 862
OPTIONAL_MODULES = (
    "gap",
    "mathematica",
    "numpy",
    "sage",
    "sageall",
    "schouten_kit",
    "scipy",
    "sympy",
    "wolframclient",
)


class AtlasReproductionError(RuntimeError):
    """A stable cross-artifact atlas reproduction failure."""


def _fixture_payload(path: Path) -> bytes:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise AtlasReproductionError("reference spec fixture is unavailable") from error
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise AtlasReproductionError(
            "reference spec fixture must have one transport newline"
        )
    canonical = payload[:-1]
    if not canonical:
        raise AtlasReproductionError("reference spec fixture is empty")
    return canonical


def _base_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_anyalgebra_atlas_reproduction_base", BASE_TOOL
    )
    if spec is None or spec.loader is None:
        raise AtlasReproductionError("artifact support is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise AtlasReproductionError(f"{label} is malformed")
    return cast(Mapping[str, object], value)


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or any(type(item) is not str for item in value):
        raise AtlasReproductionError(f"{label} is malformed")
    return cast(list[str], value)


def _digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789ABCDEFabcdef" for character in value)
    ):
        raise AtlasReproductionError(f"{label} is malformed")
    return value


def _semantic(value: object) -> dict[str, object]:
    source = _mapping(value, "semantic evidence")
    expected_keys = {
        "atlasSemanticHash",
        "atlasTreeSha256",
        "canonicalIdsSha256",
        "objectCount",
        "queryCounts",
        "specSemanticHash",
        "verifiedObjectCount",
    }
    if set(source) != expected_keys:
        raise AtlasReproductionError("semantic evidence is malformed")
    query_counts = _mapping(source["queryCounts"], "query counts")
    if dict(query_counts) != EXPECTED_QUERY_COUNTS:
        raise AtlasReproductionError("semantic evidence drifts from frozen reference")
    if (
        source["atlasSemanticHash"] != EXPECTED_ATLAS_HASH
        or source["specSemanticHash"] != EXPECTED_SPEC_HASH
        or source["objectCount"] != EXPECTED_OBJECT_COUNT
        or source["verifiedObjectCount"] != EXPECTED_OBJECT_COUNT
    ):
        raise AtlasReproductionError("semantic evidence drifts from frozen reference")
    return {
        "atlasSemanticHash": EXPECTED_ATLAS_HASH,
        "atlasTreeSha256": _digest(source["atlasTreeSha256"], "atlas tree hash"),
        "canonicalIdsSha256": _digest(
            source["canonicalIdsSha256"], "canonical identifier hash"
        ),
        "objectCount": EXPECTED_OBJECT_COUNT,
        "queryCounts": dict(EXPECTED_QUERY_COUNTS),
        "specSemanticHash": EXPECTED_SPEC_HASH,
        "verifiedObjectCount": EXPECTED_OBJECT_COUNT,
    }


def _run_record(value: object) -> tuple[dict[str, object], dict[str, object]]:
    source = _mapping(value, "run evidence")
    expected_keys = {
        "artifact",
        "artifactKind",
        "optionalModulesAttempted",
        "optionalModulesImported",
        "packageVersion",
        "semantic",
    }
    if set(source) != expected_keys:
        raise AtlasReproductionError("run evidence is malformed")
    artifact = source["artifact"]
    kind = source["artifactKind"]
    if type(artifact) is not str or kind not in {"wheel", "sdist"}:
        raise AtlasReproductionError("run evidence is malformed")
    if source["packageVersion"] != EXPECTED_VERSION:
        raise AtlasReproductionError("installed package version is not 0.1.0")
    attempted = _string_list(
        source["optionalModulesAttempted"], "optional module attempts"
    )
    imported = _string_list(source["optionalModulesImported"], "optional imports")
    if attempted or imported:
        raise AtlasReproductionError("optional modules entered atlas reproduction")
    runtime = {
        "artifact": artifact,
        "artifactKind": kind,
        "optionalModulesAttempted": [],
        "optionalModulesImported": [],
        "packageVersion": EXPECTED_VERSION,
    }
    return runtime, _semantic(source["semantic"])


def compare_reproductions(runs: Iterable[object]) -> dict[str, object]:
    """Validate and merge two runs while separating runtime from semantics."""
    values = tuple(runs)
    parsed = tuple(_run_record(value) for value in values)
    kinds = [runtime["artifactKind"] for runtime, _ in parsed]
    if len(parsed) != 2 or set(kinds) != {"wheel", "sdist"}:
        raise AtlasReproductionError("requires one wheel and one sdist")
    semantic = parsed[0][1]
    if parsed[1][1] != semantic:
        raise AtlasReproductionError("cross-artifact semantic outputs differ")
    runtime_records = sorted(
        (runtime for runtime, _ in parsed),
        key=lambda item: (str(item["artifactKind"]), str(item["artifact"])),
    )
    return {
        "ok": True,
        "runs": runtime_records,
        "schemaType": "anyalgebra.v0.1.atlas-reproduction",
        "schemaVersion": 1,
        "semantic": semantic,
    }


def _reproduction_command() -> str:
    return f"""
import hashlib, importlib.abc, json, sys
from pathlib import Path
denied = {OPTIONAL_MODULES!r}
class Audit(importlib.abc.MetaPathFinder):
    def __init__(self): self.attempted = []
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.partition('.')[0]
        if root in denied and root not in self.attempted: self.attempted.append(root)
        return None
audit = Audit()
sys.meta_path.insert(0, audit)
import anyalgebra
from anyalgebra.atlas.build import build_finite_algebra_atlas
from anyalgebra.atlas.query import load_finite_algebra_atlas, query_atlas
from anyalgebra.census.serialization import census_spec_from_canonical_bytes
spec = census_spec_from_canonical_bytes(Path(sys.argv[1]).read_bytes())
output = Path(sys.argv[2])
build = build_finite_algebra_atlas((spec,), output_directory=output)
loaded = load_finite_algebra_atlas(output)
all_matches = query_atlas(loaded)
associative = query_atlas(loaded, laws=('associativity',))
idempotent = query_atlas(loaded, laws=('idempotence',))
identities = query_atlas(loaded, invariant_fields=(('identity_status', 'unique'),))
tree = hashlib.sha256()
for path in sorted(item for item in output.rglob('*') if item.is_file()):
    tree.update(path.relative_to(output).as_posix().encode('utf-8'))
    tree.update(b'\\0')
    tree.update(path.read_bytes())
    tree.update(b'\\0')
identifiers = '\\n'.join(str(item.canonical_id) for item in all_matches.matches)
loaded_optional = sorted(name for name in denied if name in sys.modules)
record = {{
    'packageVersion': anyalgebra.__version__,
    'optionalModulesAttempted': sorted(audit.attempted),
    'optionalModulesImported': loaded_optional,
    'semantic': {{
        'atlasSemanticHash': str(build.atlas.semantic_hash),
        'atlasTreeSha256': tree.hexdigest(),
        'canonicalIdsSha256': hashlib.sha256(identifiers.encode('utf-8')).hexdigest(),
        'objectCount': build.object_count,
        'queryCounts': {{
            'all': len(all_matches.matches),
            'associative': len(associative.matches),
            'idempotent': len(idempotent.matches),
            'uniqueIdentity': len(identities.matches),
        }},
        'specSemanticHash': str(spec.semantic_hash),
        'verifiedObjectCount': loaded.verified_object_count,
    }},
}}
print(json.dumps(record, allow_nan=False, sort_keys=True, separators=(',', ':')))
"""


def reproduce(
    repository_root: Path | str,
    *,
    artifacts: Iterable[Path | str] | None = None,
    allow_network: bool = False,
) -> dict[str, object]:
    """Build or inspect both artifacts and reproduce the atlas from each."""
    root = Path(repository_root).resolve()
    fixture = root / SPEC_FIXTURE
    if not fixture.is_file():
        raise AtlasReproductionError("reference spec fixture is unavailable")
    base = _base_tool()
    metadata = base.expected_metadata(root)
    if metadata.version != EXPECTED_VERSION or metadata.runtime_dependencies:
        raise AtlasReproductionError("repository metadata is not the v0.1 contract")
    with tempfile.TemporaryDirectory(
        prefix="anyalgebra-v01-reproduction-"
    ) as temporary:
        temporary_root = Path(temporary)
        artifact_directory = temporary_root / "artifacts"
        artifact_directory.mkdir()
        if artifacts is None:
            artifact_paths = base._build(
                root, artifact_directory, allow_network=allow_network
            )
        else:
            artifact_paths = tuple(Path(value).resolve() for value in artifacts)
        inspection = base.inspect_release_artifacts(artifact_paths, metadata)
        if not inspection.ok:
            raise AtlasReproductionError("artifact inspection failed")
        kinds = {item.filename: item.kind for item in inspection.artifacts}
        unrelated = temporary_root / "unrelated-cwd"
        unrelated.mkdir()
        copied_fixture = temporary_root / "v01_atlas_spec.json"
        copied_fixture.write_bytes(_fixture_payload(fixture))
        records: list[dict[str, object]] = []
        for index, artifact in enumerate(sorted(artifact_paths, key=lambda p: p.name)):
            python = base._venv_python(temporary_root / f"install-{index}")
            base._run(
                base._uv_install_command(python, artifact, allow_network=allow_network),
                cwd=unrelated,
                stage="install",
                environment=base._environment(),
            )
            completed = base._run(
                [
                    python,
                    "-I",
                    "-c",
                    _reproduction_command(),
                    copied_fixture,
                    temporary_root / f"atlas-{index}",
                ],
                cwd=unrelated,
                stage="reproduce",
                environment=base._environment(),
            )
            try:
                record = _mapping(json.loads(completed.stdout), "subprocess evidence")
            except json.JSONDecodeError as error:
                raise AtlasReproductionError(
                    "subprocess evidence is malformed"
                ) from error
            records.append(
                {
                    "artifact": artifact.name,
                    "artifactKind": kinds[artifact.name],
                    **dict(record),
                }
            )
        return compare_reproductions(records)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument("--artifact", action="append", type=Path, dest="artifacts")
    parser.add_argument("--allow-network", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        record = reproduce(
            arguments.repository,
            artifacts=arguments.artifacts,
            allow_network=arguments.allow_network,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        stage = getattr(error, "stage", "atlas-reproduction")
        print(
            json.dumps(
                {
                    "detail": "atlas reproduction failed",
                    "ok": False,
                    "stage": stage if type(stage) is str else "atlas-reproduction",
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
