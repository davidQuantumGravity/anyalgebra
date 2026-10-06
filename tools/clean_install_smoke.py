"""Build a wheel, install it in a clean environment, and verify core import."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import venv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


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


@dataclass(frozen=True)
class SmokeResult:
    """Evidence collected from one isolated installed-wheel import."""

    version: str
    package_location: str
    purelib_location: str
    optional_modules_attempted: tuple[str, ...]
    optional_modules_imported: tuple[str, ...]


class SmokeFailure(RuntimeError):
    """A build, install, or import stage failure with its captured output."""

    def __init__(self, stage: str, detail: str, *, output: str = "") -> None:
        self.stage = stage
        self.detail = detail
        self.output = output
        message = f"{stage} failed: {detail}"
        if output:
            message = f"{message}\n{output}"
        super().__init__(message)


class OptionalImportAudit:
    """Record and reject imports whose top-level name is an optional backend."""

    def __init__(self, denied_roots: tuple[str, ...]) -> None:
        self._denied_roots = frozenset(denied_roots)
        self.attempted_roots: list[str] = []

    def find_spec(
        self, fullname: str, path: object = None, target: object = None
    ) -> None:
        """Reject an attempted optional import before another finder handles it."""
        del path, target
        root = fullname.partition(".")[0]
        if root in self._denied_roots:
            self.attempted_roots.append(root)
            raise ImportError(f"unexpected optional import: {root}")


def find_single_wheel(wheel_directory: Path) -> Path:
    """Return the sole AnyAlgebra wheel or report a deterministic build failure."""

    wheels = sorted(wheel_directory.glob("anyalgebra-*.whl"))
    if not wheels:
        raise SmokeFailure("build", "no wheel was produced")
    if len(wheels) != 1:
        names = ", ".join(wheel.name for wheel in wheels)
        raise SmokeFailure("build", f"expected one wheel, found: {names}")
    return wheels[0]


def _environment_for_subprocess() -> dict[str, str]:
    environment = os.environ.copy()
    for variable in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        environment.pop(variable, None)
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return environment


def _run(
    command: Sequence[str | Path], *, cwd: Path, stage: str
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [str(part) for part in command],
        check=False,
        cwd=cwd,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=_environment_for_subprocess(),
        text=True,
    )
    if completed.returncode != 0:
        output = "".join((completed.stdout, completed.stderr)).strip()
        raise SmokeFailure(
            stage, f"command exited {completed.returncode}", output=output
        )
    return completed


def _create_environment(destination: Path, *, stage: str) -> Path:
    try:
        venv.EnvBuilder(with_pip=True, clear=False).create(destination)
    except (OSError, subprocess.SubprocessError) as error:
        raise SmokeFailure(stage, "could not create virtual environment") from error

    candidates = (
        destination / "Scripts" / "python.exe",
        destination / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SmokeFailure(stage, "virtual environment has no Python interpreter")


def _import_check_command() -> str:
    optional_modules = repr(OPTIONAL_MODULE_NAMES)
    return f"""
import importlib
import json
import sys
import sysconfig
from pathlib import Path

optional_modules = {optional_modules}
class OptionalImportAudit:
    def __init__(self, denied_roots):
        self.denied_roots = frozenset(denied_roots)
        self.attempted_roots = []

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.partition('.')[0]
        if root in self.denied_roots:
            self.attempted_roots.append(root)
            raise ImportError('unexpected optional import: ' + root)

audit = OptionalImportAudit(optional_modules)
sys.meta_path.insert(0, audit)
try:
    package = importlib.import_module('anyalgebra')
finally:
    sys.meta_path.remove(audit)
package_location = Path(package.__file__).resolve()
purelib_location = Path(sysconfig.get_paths()['purelib']).resolve()
try:
    package_location.relative_to(purelib_location)
except ValueError as error:
    raise RuntimeError('anyalgebra did not import from installed purelib') from error
if audit.attempted_roots:
    raise RuntimeError(
        'optional modules attempted by core: ' + ', '.join(audit.attempted_roots)
    )
imported_optional_modules = tuple(
    name for name in optional_modules if name in sys.modules
)
if imported_optional_modules:
    raise RuntimeError(
        'optional modules imported by core: ' + ', '.join(imported_optional_modules)
    )
print(json.dumps({{
    'optional_modules_attempted': audit.attempted_roots,
    'optional_modules_imported': imported_optional_modules,
    'package_location': package_location.as_posix(),
    'purelib_location': purelib_location.as_posix(),
    'version': package.__version__,
}}, sort_keys=True))
"""


def run_clean_install_smoke(repository_root: Path | str | None = None) -> SmokeResult:
    """Build, install, and import AnyAlgebra without optional systems.

    All environments and artifacts are created inside one temporary directory.
    The import subprocess runs from an empty temporary working directory with
    Python isolated mode enabled, so a source-tree import cannot satisfy it.
    """

    root = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else Path(repository_root).resolve()
    )
    if not (root / "pyproject.toml").is_file():
        raise SmokeFailure("build", f"no pyproject.toml at repository root: {root}")

    with tempfile.TemporaryDirectory(prefix="anyalgebra-clean-install-") as temporary:
        temporary_root = Path(temporary)
        build_environment = _create_environment(
            temporary_root / "build-environment", stage="build"
        )
        wheel_directory = temporary_root / "wheels"
        wheel_directory.mkdir()
        _run(
            (
                build_environment,
                "-m",
                "pip",
                "--isolated",
                "wheel",
                "--no-deps",
                "--wheel-dir",
                wheel_directory,
                root,
            ),
            cwd=temporary_root,
            stage="build",
        )
        wheel = find_single_wheel(wheel_directory)

        install_environment = _create_environment(
            temporary_root / "install-environment", stage="install"
        )
        _run(
            (
                install_environment,
                "-m",
                "pip",
                "--isolated",
                "install",
                "--no-deps",
                wheel,
            ),
            cwd=temporary_root,
            stage="install",
        )
        import_result = _run(
            (install_environment, "-I", "-c", _import_check_command()),
            cwd=temporary_root,
            stage="import",
        )

    try:
        payload = json.loads(import_result.stdout)
        return SmokeResult(
            version=str(payload["version"]),
            package_location=str(payload["package_location"]),
            purelib_location=str(payload["purelib_location"]),
            optional_modules_attempted=tuple(payload["optional_modules_attempted"]),
            optional_modules_imported=tuple(payload["optional_modules_imported"]),
        )
    except (TypeError, ValueError, KeyError) as error:
        raise SmokeFailure("import", "produced an invalid evidence record") from error


def main(argv: Sequence[str] | None = None) -> int:
    """Run the clean install smoke check and print a stable evidence record."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repository_root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    arguments = parser.parse_args(argv)
    try:
        result = run_clean_install_smoke(arguments.repository_root)
    except SmokeFailure as error:
        print(error, file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "optional_modules_imported": result.optional_modules_imported,
                "optional_modules_attempted": result.optional_modules_attempted,
                "package_location": result.package_location,
                "purelib_location": result.purelib_location,
                "version": result.version,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
