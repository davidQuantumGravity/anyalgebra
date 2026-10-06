"""Build, inspect, and clean-install AnyAlgebra release artifacts.

The checker deliberately keeps artifacts, virtual environments, and the
unrelated import working directory inside a temporary directory.  It does not
write to the source tree, and its default build route is offline.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
from dataclasses import asdict, dataclass
from email import policy
from email.message import Message
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterable, Sequence
import venv
import zipfile


OPTIONAL_MODULE_NAMES: tuple[str, ...] = (
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
_REQUIRED_PACKAGE_FILES = frozenset(
    {
        "anyalgebra/__init__.py",
        "anyalgebra/_version.py",
        "anyalgebra/py.typed",
        "anyalgebra/backends/base.py",
        "anyalgebra/backends/reference.py",
    }
)
_FORBIDDEN_PARTS = frozenset(
    {
        ".agents",
        ".coverage",
        ".git",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "docs",
        "tests",
    }
)
_SECRET_PARTS = frozenset({".env", "credentials", "secrets", "id_rsa"})
_LOCAL_PATH = re.compile(
    r"(?<![A-Za-z0-9+.-])(?:[A-Za-z]:[\\/]|/(?:Users|home)/|file://)"
)
_SECRET_VALUE = re.compile(r"(?:api[_-]?key|password|secret|token)\s*[=:]", re.I)


class ReleaseArtifactFailure(RuntimeError):
    """A typed build, inspection, or clean-install failure."""

    def __init__(self, stage: str, detail: str, *, output: str = "") -> None:
        self.stage = stage
        self.detail = detail
        self.output = output
        super().__init__(f"{stage} failed: {detail}")


@dataclass(frozen=True)
class ReleaseMetadata:
    """Expected publishable metadata derived from the repository contract."""

    name: str
    version: str
    requires_python: str
    runtime_dependencies: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactInspection:
    """Stable evidence for one archive after static validation."""

    filename: str
    kind: str
    sha256: str
    member_count: int
    metadata_name: str
    metadata_version: str
    requires_python: str
    runtime_dependencies: tuple[str, ...]


@dataclass(frozen=True)
class InspectionResult:
    """Structured result for archive-only checks, including negative evidence."""

    artifacts: tuple[ArtifactInspection, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class InstallEvidence:
    """Evidence from one dependency-free, installed-artifact import."""

    artifact: str
    version: str
    package_location: str
    purelib_location: str
    optional_modules_attempted: tuple[str, ...]
    optional_modules_imported: tuple[str, ...]
    reference_backend_verified: bool


@dataclass(frozen=True)
class ReleaseCheckResult:
    """Complete successful release-check evidence."""

    metadata: ReleaseMetadata
    inspection: InspectionResult
    installations: tuple[InstallEvidence, ...]


def _normalise_dependency(value: str) -> str:
    return value.split(";", 1)[0].strip()


def _runtime_dependencies(message: Message) -> tuple[str, ...]:
    values = message.get_all("Requires-Dist", [])
    result = []
    for value in values:
        item = str(value).strip()
        marker = item.partition(";")[2].lower()
        if "extra ==" not in marker:
            result.append(_normalise_dependency(item))
    return tuple(sorted(result))


def expected_metadata(repository_root: Path | str) -> ReleaseMetadata:
    """Read the package metadata contract without importing the source package."""
    root = Path(repository_root).resolve()
    try:
        import tomllib

        with (root / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream)["project"]
        source = root / "src" / "anyalgebra" / "_version.py"
        parsed = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        version = next(
            node.value.value
            for node in parsed.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__version__"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        dependencies = project.get("dependencies", [])
        if not all(isinstance(item, str) for item in dependencies):
            raise TypeError("project dependencies must be strings")
        return ReleaseMetadata(
            name=str(project["name"]),
            version=version,
            requires_python=str(project["requires-python"]),
            runtime_dependencies=tuple(
                sorted(map(_normalise_dependency, dependencies))
            ),
        )
    except (
        KeyError,
        OSError,
        SyntaxError,
        StopIteration,
        TypeError,
        ValueError,
    ) as error:
        raise ReleaseArtifactFailure(
            "metadata", "repository metadata is not readable"
        ) from error


def _safe_member_name(name: str) -> str | None:
    path = PurePosixPath(name)
    if (
        not name
        or not path.parts
        or "\\" in name
        or path.is_absolute()
        or ":" in path.parts[0]
    ):
        return None
    if any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path.as_posix()


def _forbidden_member(name: str, *, wheel: bool) -> str | None:
    parts = PurePosixPath(name).parts
    lowered = {part.lower() for part in parts}
    excluded = _FORBIDDEN_PARTS if wheel else _FORBIDDEN_PARTS - {"docs", "tests"}
    if lowered & excluded or name.lower().endswith(".pyc"):
        return "contains excluded wheel/source cache or bytecode"
    if wheel and ("docs" in parts or "tests" in parts):
        return "contains excluded tests or documentation"
    if lowered & _SECRET_PARTS:
        return "contains a secret-bearing filename"
    return None


def _metadata_message(
    payload: bytes, filename: str, errors: list[str]
) -> Message | None:
    try:
        message = BytesParser(policy=policy.strict).parsebytes(payload)
    except Exception:
        errors.append(f"{filename}: invalid core metadata")
        return None
    if message.defects:
        errors.append(f"{filename}: malformed core metadata")
        return None
    text = payload.decode("utf-8", errors="replace")
    if _LOCAL_PATH.search(text) or _SECRET_VALUE.search(text):
        errors.append(
            f"{filename}: metadata contains a local path or secret-like value"
        )
    return message


def _inspect_metadata(
    message: Message,
    expected: ReleaseMetadata,
    filename: str,
    errors: list[str],
) -> tuple[str, str, str, tuple[str, ...]]:
    name, version = str(message.get("Name", "")), str(message.get("Version", ""))
    requires_python = str(message.get("Requires-Python", ""))
    dependencies = _runtime_dependencies(message)
    if name != expected.name:
        errors.append(
            f"{filename}: metadata name {name!r} does not match {expected.name!r}"
        )
    if version != expected.version:
        errors.append(
            f"{filename}: metadata version {version!r} does not match "
            f"{expected.version!r}"
        )
    if requires_python != expected.requires_python:
        errors.append(
            f"{filename}: Requires-Python {requires_python!r} does not match "
            f"{expected.requires_python!r}"
        )
    if dependencies != expected.runtime_dependencies:
        errors.append(
            f"{filename}: runtime dependencies {dependencies!r} do not match "
            f"{expected.runtime_dependencies!r}"
        )
    return name, version, requires_python, dependencies


def _record_errors(
    members: dict[str, bytes], record_name: str, filename: str
) -> list[str]:
    errors: list[str] = []
    try:
        rows = list(csv.reader(members[record_name].decode("utf-8").splitlines()))
    except (KeyError, UnicodeDecodeError, csv.Error):
        return [f"{filename}: invalid wheel RECORD"]
    recorded: set[str] = set()
    for row in rows:
        if len(row) != 3 or not row[0] or row[0] in recorded:
            errors.append(f"{filename}: invalid wheel RECORD row")
            continue
        path, digest, size = row
        recorded.add(path)
        if path not in members:
            errors.append(f"{filename}: RECORD references missing member {path!r}")
            continue
        if path == record_name:
            if digest or size:
                errors.append(f"{filename}: RECORD must not hash itself")
            continue
        try:
            algorithm, encoded = digest.split("=", 1)
            if algorithm not in {"sha256", "sha384", "sha512"}:
                raise ValueError("weak RECORD digest algorithm")
            actual = hashlib.new(algorithm, members[path]).digest()
            expected = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
            if actual != expected or str(len(members[path])) != size:
                errors.append(f"{filename}: RECORD integrity mismatch for {path!r}")
        except (ValueError, TypeError):
            errors.append(f"{filename}: invalid RECORD digest for {path!r}")
    missing = set(members) - recorded
    if missing:
        errors.append(f"{filename}: RECORD omits member {sorted(missing)[0]!r}")
    return errors


def _inspect_wheel(
    path: Path, expected: ReleaseMetadata
) -> tuple[ArtifactInspection | None, list[str]]:
    errors: list[str] = []
    members: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                name = _safe_member_name(info.filename)
                if name is None:
                    errors.append(f"{path.name}: unsafe archive path {info.filename!r}")
                    continue
                if name in members:
                    errors.append(f"{path.name}: duplicate archive member {name!r}")
                if info.is_dir():
                    continue
                if info.external_attr >> 16 & 0o170000 == 0o120000:
                    errors.append(f"{path.name}: symbolic-link archive member {name!r}")
                    continue
                forbidden = _forbidden_member(name, wheel=True)
                if forbidden:
                    errors.append(f"{path.name}: {forbidden}: {name!r}")
                members[name] = archive.read(info)
    except (OSError, zipfile.BadZipFile):
        return None, [f"{path.name}: unreadable wheel archive"]
    metadata_names = [name for name in members if name.endswith(".dist-info/METADATA")]
    record_names = [name for name in members if name.endswith(".dist-info/RECORD")]
    if len(metadata_names) != 1 or len(record_names) != 1:
        errors.append(
            f"{path.name}: wheel must contain exactly one METADATA and RECORD"
        )
        return None, errors
    missing = _REQUIRED_PACKAGE_FILES - set(members)
    if missing:
        errors.append(
            f"{path.name}: missing required package file {sorted(missing)[0]!r}"
        )
    message = _metadata_message(members[metadata_names[0]], path.name, errors)
    if message is None:
        return None, errors
    metadata = _inspect_metadata(message, expected, path.name, errors)
    errors.extend(_record_errors(members, record_names[0], path.name))
    return ArtifactInspection(
        path.name, "wheel", _sha256(path), len(members), *metadata
    ), errors


def _inspect_sdist(
    path: Path, expected: ReleaseMetadata
) -> tuple[ArtifactInspection | None, list[str]]:
    errors: list[str] = []
    members: dict[str, bytes] = {}
    try:
        with tarfile.open(path, "r:gz") as archive:
            for info in archive.getmembers():
                name = _safe_member_name(info.name)
                if name is None:
                    errors.append(f"{path.name}: unsafe archive path {info.name!r}")
                    continue
                if not info.isfile():
                    if not info.isdir():
                        errors.append(
                            f"{path.name}: non-regular archive member {name!r}"
                        )
                    continue
                if name in members:
                    errors.append(f"{path.name}: duplicate archive member {name!r}")
                    continue
                forbidden = _forbidden_member(name, wheel=False)
                if forbidden:
                    errors.append(f"{path.name}: {forbidden}: {name!r}")
                extracted = archive.extractfile(info)
                if extracted is None:
                    errors.append(f"{path.name}: unreadable member {name!r}")
                else:
                    members[name] = extracted.read()
    except (OSError, tarfile.TarError):
        return None, [f"{path.name}: unreadable sdist archive"]
    roots = {PurePosixPath(name).parts[0] for name in members}
    if not roots:
        return None, [*errors, f"{path.name}: empty sdist archive"]
    if len(roots) != 1:
        return None, [*errors, f"{path.name}: sdist must use one top-level directory"]
    root = next(iter(roots))
    if not root.startswith("anyalgebra-"):
        errors.append(f"{path.name}: unexpected sdist top-level directory {root!r}")
    relative = {
        name.removeprefix(root + "/"): payload for name, payload in members.items()
    }
    required_source_files = {f"src/{name}" for name in _REQUIRED_PACKAGE_FILES}
    missing = required_source_files - set(relative)
    if missing:
        errors.append(
            f"{path.name}: missing required package file {sorted(missing)[0]!r}"
        )
    if "PKG-INFO" not in relative:
        return None, [*errors, f"{path.name}: missing PKG-INFO"]
    message = _metadata_message(relative["PKG-INFO"], path.name, errors)
    if message is None:
        return None, errors
    metadata = _inspect_metadata(message, expected, path.name, errors)
    return ArtifactInspection(
        path.name, "sdist", _sha256(path), len(members), *metadata
    ), errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_release_artifacts(
    artifacts: Iterable[Path | str], expected: ReleaseMetadata
) -> InspectionResult:
    """Inspect artifacts without extracting them or mutating their source tree."""
    checked: list[ArtifactInspection] = []
    errors: list[str] = []
    seen_kinds: set[str] = set()
    for raw_path in artifacts:
        path = Path(raw_path)
        if path.suffix == ".whl":
            result, item_errors = _inspect_wheel(path, expected)
            kind = "wheel"
        elif path.name.endswith(".tar.gz"):
            result, item_errors = _inspect_sdist(path, expected)
            kind = "sdist"
        else:
            result, item_errors, kind = (
                None,
                [f"{path.name}: unsupported artifact type"],
                "unknown",
            )
        if kind in seen_kinds:
            errors.append(f"{path.name}: duplicate {kind} artifact")
        seen_kinds.add(kind)
        if result is not None:
            checked.append(result)
        errors.extend(item_errors)
    for required in ("sdist", "wheel"):
        if required not in seen_kinds:
            errors.append(f"missing required {required} artifact")
    return InspectionResult(tuple(checked), tuple(sorted(errors)))


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        environment.pop(name, None)
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return environment


def _run(
    command: Sequence[str | Path], *, cwd: Path, stage: str, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            [str(part) for part in command],
            cwd=cwd,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        raise ReleaseArtifactFailure(stage, "could not start command") from error
    if completed.returncode:
        raise ReleaseArtifactFailure(
            stage,
            f"command exited {completed.returncode}",
            output=(completed.stdout + completed.stderr).strip(),
        )
    return completed


def _venv_python(destination: Path) -> Path:
    try:
        venv.EnvBuilder(with_pip=True, clear=False).create(destination)
    except OSError as error:
        raise ReleaseArtifactFailure(
            "install", "could not create virtual environment"
        ) from error
    for candidate in (
        destination / "Scripts" / "python.exe",
        destination / "bin" / "python",
    ):
        if candidate.is_file():
            return candidate
    raise ReleaseArtifactFailure(
        "install", "virtual environment has no Python interpreter"
    )


def _import_command() -> str:
    return f"""
import hashlib, importlib, json, sys, sysconfig
from pathlib import Path
denied = {OPTIONAL_MODULE_NAMES!r}
class Audit:
    def __init__(self): self.attempted = []
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.partition('.')[0]
        if root in denied:
            self.attempted.append(root)
            raise ImportError('unexpected optional import: ' + root)
audit = Audit()
sys.meta_path.insert(0, audit)
try:
    package = importlib.import_module('anyalgebra')
    from anyalgebra.backends.base import BackendOptions, BackendRequest
    from anyalgebra.backends.reference import ReferenceBackend
    from anyalgebra.core.parents import SemanticHash
    from anyalgebra.structures.outcomes import Defined
finally:
    sys.meta_path.remove(audit)
source = Path(package.__file__).resolve()
purelib = Path(sysconfig.get_paths()['purelib']).resolve()
try:
    source.relative_to(purelib)
except ValueError:
    raise RuntimeError('package did not import from installed purelib')
if audit.attempted:
    raise RuntimeError('optional modules attempted: ' + ', '.join(audit.attempted))
loaded = tuple(name for name in denied if name in sys.modules)
if loaded:
    raise RuntimeError('optional modules imported: ' + ', '.join(loaded))
value = SemanticHash('sha256', hashlib.sha256(b'release-artifact-check').hexdigest())
request = BackendRequest.create(
    operation='hash_identity', domain='semantic_hash', algorithm='identity',
    input_hashes=(value,), limits=(('inputs', 1),),
)
outcome = ReferenceBackend().execute(
    request, options=BackendOptions.create(limits=(('inputs', 1),)),
)
if type(outcome) is not Defined or outcome.value.value_hash != value:
    raise RuntimeError('reference backend quickstart failed')
print(json.dumps({{
    'version': package.__version__,
    'package_location': source.as_posix(),
    'purelib_location': purelib.as_posix(),
    'optional_modules_attempted': audit.attempted,
    'optional_modules_imported': loaded,
    'reference_backend_verified': True,
}}, sort_keys=True))
"""


def _clean_install(
    artifact: Path, temporary_root: Path, *, allow_network: bool
) -> InstallEvidence:
    python = _venv_python(temporary_root / f"install-{artifact.suffix.lstrip('.')}")
    environment = _environment()
    command = _uv_install_command(python, artifact, allow_network=allow_network)
    _run(
        command,
        cwd=temporary_root / "unrelated-cwd",
        stage="install",
        environment=environment,
    )
    completed = _run(
        [python, "-I", "-c", _import_command()],
        cwd=temporary_root / "unrelated-cwd",
        stage="import",
        environment=environment,
    )
    try:
        payload = json.loads(completed.stdout)
        return InstallEvidence(
            artifact.name,
            str(payload["version"]),
            str(payload["package_location"]),
            str(payload["purelib_location"]),
            tuple(payload["optional_modules_attempted"]),
            tuple(payload["optional_modules_imported"]),
            bool(payload["reference_backend_verified"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ReleaseArtifactFailure(
            "import", "invalid clean-install evidence"
        ) from error


def _uv_install_command(
    python: Path, artifact: Path, *, allow_network: bool
) -> list[str | Path]:
    """Use uv for install so an sdist shares the build frontend's cache policy."""
    uv = shutil.which("uv")
    if uv is None:
        raise ReleaseArtifactFailure(
            "install", "uv is unavailable; no bundled install frontend was found"
        )
    command: list[str | Path] = [uv]
    if allow_network:
        # Windows' native trust store is more reliable than uv's bundled roots.
        command.append("--native-tls")
    else:
        command.append("--offline")
    command.extend(("pip", "install", "--python", python, "--no-deps", artifact))
    return command


def _build(
    repository_root: Path, output: Path, *, allow_network: bool
) -> tuple[Path, ...]:
    uv = shutil.which("uv")
    if uv is None:
        raise ReleaseArtifactFailure(
            "build", "uv is unavailable; no bundled build frontend was found"
        )
    command: list[str | Path] = [uv]
    if allow_network:
        # Windows' native trust store is more reliable than uv's bundled roots.
        command.append("--native-tls")
    command.extend(
        (
            "build",
            "--no-create-gitignore",
            "--out-dir",
            output,
        )
    )
    if not allow_network:
        command.append("--offline")
    _run(
        [*command, repository_root],
        cwd=output.parent,
        stage="build",
        environment=_environment(),
    )
    artifacts = tuple(
        sorted(output.glob("anyalgebra-*.whl"))
        + sorted(output.glob("anyalgebra-*.tar.gz"))
    )
    return artifacts


def run_release_artifact_check(
    repository_root: Path | str,
    *,
    allow_network: bool = False,
    artifacts: Iterable[Path | str] | None = None,
    inspect_only: bool = False,
) -> ReleaseCheckResult:
    """Build (unless supplied), inspect, and clean-install each release artifact."""
    root = Path(repository_root).resolve()
    metadata = expected_metadata(root)
    if artifacts is not None:
        artifact_paths = tuple(Path(item).resolve() for item in artifacts)
        inspection = inspect_release_artifacts(artifact_paths, metadata)
        if not inspection.ok:
            raise ReleaseArtifactFailure("inspect", "; ".join(inspection.errors))
        if inspect_only:
            return ReleaseCheckResult(metadata, inspection, ())
        with tempfile.TemporaryDirectory(
            prefix="anyalgebra-release-check-"
        ) as temporary:
            temporary_root = Path(temporary)
            (temporary_root / "unrelated-cwd").mkdir()
            installations = tuple(
                _clean_install(item, temporary_root, allow_network=allow_network)
                for item in artifact_paths
            )
            return ReleaseCheckResult(metadata, inspection, installations)
    with tempfile.TemporaryDirectory(prefix="anyalgebra-release-check-") as temporary:
        temporary_root = Path(temporary)
        output = temporary_root / "artifacts"
        output.mkdir()
        artifact_paths = _build(root, output, allow_network=allow_network)
        inspection = inspect_release_artifacts(artifact_paths, metadata)
        if not inspection.ok:
            raise ReleaseArtifactFailure("inspect", "; ".join(inspection.errors))
        if inspect_only:
            return ReleaseCheckResult(metadata, inspection, ())
        (temporary_root / "unrelated-cwd").mkdir()
        installations = tuple(
            _clean_install(item, temporary_root, allow_network=allow_network)
            for item in artifact_paths
        )
        return ReleaseCheckResult(metadata, inspection, installations)


def main(argv: Sequence[str] | None = None) -> int:
    """Print a JSON evidence record suitable for a release receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repository_root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--artifact", action="append", type=Path, dest="artifacts")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="permit a missing build requirement to be downloaded",
    )
    arguments = parser.parse_args(argv)
    try:
        result = run_release_artifact_check(
            arguments.repository_root,
            allow_network=arguments.allow_network,
            artifacts=arguments.artifacts,
            inspect_only=arguments.inspect_only,
        )
    except ReleaseArtifactFailure as error:
        print(
            json.dumps(
                {
                    "detail": error.detail,
                    "ok": False,
                    "output": error.output,
                    "stage": error.stage,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(asdict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
