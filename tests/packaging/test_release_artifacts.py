"""External contracts for the V00-097 release-artifact checker."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import sys
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = REPOSITORY_ROOT / "tools" / "check_release_artifacts.py"


def _load_checker() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "check_release_artifacts", CHECKER_PATH
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _record_hash(payload: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode("ascii")
    return f"sha256={digest.rstrip('=')}"


def _wheel(tmp_path: Path, *, metadata: str | None = None) -> Path:
    """Create the smallest safe, internally consistent wheel archive."""
    artifact = tmp_path / "anyalgebra-0.0.0.dev0-py3-none-any.whl"
    files = {
        "anyalgebra/__init__.py": b'__version__ = "0.0.0.dev0"\n',
        "anyalgebra/_version.py": b'__version__ = "0.0.0.dev0"\n',
        "anyalgebra/py.typed": b"",
        "anyalgebra/backends/base.py": b"",
        "anyalgebra/backends/reference.py": b"",
        "anyalgebra-0.0.0.dev0.dist-info/METADATA": (
            metadata
            or "Metadata-Version: 2.3\nName: anyalgebra\nVersion: 0.0.0.dev0\n"
            "Requires-Python: >=3.11\n\n"
        ).encode(),
        "anyalgebra-0.0.0.dev0.dist-info/WHEEL": (
            b"Wheel-Version: 1.0\nTag: py3-none-any\n"
        ),
    }
    rows = [
        [name, _record_hash(payload), str(len(payload))]
        for name, payload in files.items()
    ]
    rows.append(["anyalgebra-0.0.0.dev0.dist-info/RECORD", "", ""])
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    files["anyalgebra-0.0.0.dev0.dist-info/RECORD"] = output.getvalue().encode()
    with zipfile.ZipFile(artifact, "w") as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return artifact


def _sdist(tmp_path: Path, *, extra_name: str | None = None) -> Path:
    """Create the matching safe sdist used by archive-only tests."""
    artifact = tmp_path / "anyalgebra-0.0.0.dev0.tar.gz"
    root = "anyalgebra-0.0.0.dev0"
    files = {
        "src/anyalgebra/__init__.py": b'__version__ = "0.0.0.dev0"\n',
        "src/anyalgebra/_version.py": b'__version__ = "0.0.0.dev0"\n',
        "src/anyalgebra/py.typed": b"",
        "src/anyalgebra/backends/base.py": b"",
        "src/anyalgebra/backends/reference.py": b"",
        "PKG-INFO": (
            b"Metadata-Version: 2.3\nName: anyalgebra\nVersion: 0.0.0.dev0\n"
            b"Requires-Python: >=3.11\n\n"
        ),
    }
    if extra_name is not None:
        files[extra_name] = b"excluded"
    with tarfile.open(artifact, "w:gz") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(f"{root}/{name}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return artifact


def test_inspect_accepts_matching_safe_wheel_and_sdist(tmp_path: Path) -> None:
    """Positive archive inspection reports deterministic metadata and hashes."""
    checker = _load_checker()

    result = checker.inspect_release_artifacts(
        (_wheel(tmp_path), _sdist(tmp_path)),
        checker.ReleaseMetadata(
            name="anyalgebra",
            version="0.0.0.dev0",
            requires_python=">=3.11",
            runtime_dependencies=(),
        ),
    )

    assert result.ok
    assert {item.kind for item in result.artifacts} == {"wheel", "sdist"}
    assert all(len(item.sha256) == 64 for item in result.artifacts)


@pytest.mark.parametrize("unsafe_name", ("../escape.py", "/absolute.py", "C:/local.py"))
def test_inspection_rejects_unsafe_archive_paths(
    tmp_path: Path, unsafe_name: str
) -> None:
    """Traversal, absolute, and Windows-drive paths never pass inspection."""
    checker = _load_checker()
    artifact = _wheel(tmp_path)
    with zipfile.ZipFile(artifact, "a") as archive:
        archive.writestr(unsafe_name, b"unsafe")

    result = checker.inspect_release_artifacts(
        (artifact,),
        checker.ReleaseMetadata("anyalgebra", "0.0.0.dev0", ">=3.11", ()),
    )

    assert not result.ok
    assert any("unsafe archive path" in error for error in result.errors)


def test_inspection_rejects_tampered_wheel_record_and_metadata(tmp_path: Path) -> None:
    """A stale RECORD digest and unexpected runtime dependency are release blockers."""
    checker = _load_checker()
    artifact = _wheel(
        tmp_path,
        metadata=(
            "Metadata-Version: 2.3\nName: anyalgebra\nVersion: 9.9\n"
            "Requires-Python: >=3.11\nRequires-Dist: surprise\n\n"
        ),
    )
    with (
        pytest.warns(UserWarning, match="Duplicate name"),
        zipfile.ZipFile(artifact, "a") as archive,
    ):
        archive.writestr("anyalgebra/_version.py", b"tampered\n")

    result = checker.inspect_release_artifacts(
        (artifact,),
        checker.ReleaseMetadata("anyalgebra", "0.0.0.dev0", ">=3.11", ()),
    )

    assert not result.ok
    assert any("RECORD" in error for error in result.errors)
    assert any("version" in error for error in result.errors)
    assert any("runtime dependencies" in error for error in result.errors)


def test_metadata_allows_https_but_rejects_actual_local_paths(tmp_path: Path) -> None:
    """URL schemes are not drive paths, while explicit user paths remain blocked."""
    checker = _load_checker()
    safe_dir = tmp_path / "safe"
    unsafe_dir = tmp_path / "unsafe"
    safe_dir.mkdir()
    unsafe_dir.mkdir()
    safe = _wheel(
        safe_dir,
        metadata=(
            "Metadata-Version: 2.3\nName: anyalgebra\nVersion: 0.0.0.dev0\n"
            "Requires-Python: >=3.11\n"
            "Project-URL: Repository, https://example.invalid/anyalgebra\n\n"
        ),
    )
    unsafe = _wheel(
        unsafe_dir,
        metadata=(
            "Metadata-Version: 2.3\nName: anyalgebra\nVersion: 0.0.0.dev0\n"
            "Requires-Python: >=3.11\nDescription: C:/Users/private/source\n\n"
        ),
    )
    expected = checker.ReleaseMetadata("anyalgebra", "0.0.0.dev0", ">=3.11", ())

    _, safe_errors = checker._inspect_wheel(safe, expected)
    _, unsafe_errors = checker._inspect_wheel(unsafe, expected)

    assert safe_errors == []
    assert any("local path" in error for error in unsafe_errors)


@pytest.mark.parametrize("member", (".CoVeRaGe", ".AgEnTs/internal.py"))
def test_inspection_rejects_case_insensitive_internal_cache_members(
    tmp_path: Path, member: str
) -> None:
    """Neither archive kind may distribute internal or coverage cache files."""
    checker = _load_checker()
    wheel = _wheel(tmp_path)
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr(member, b"excluded")
    sdist = _sdist(tmp_path, extra_name=member)

    result = checker.inspect_release_artifacts(
        (wheel, sdist),
        checker.ReleaseMetadata("anyalgebra", "0.0.0.dev0", ">=3.11", ()),
    )

    assert not result.ok
    assert sum("excluded wheel/source cache" in error for error in result.errors) == 2


def test_uv_install_command_reuses_build_frontend_cache_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Sdist installation uses uv's cache, not pip's unrelated no-index cache."""
    checker = _load_checker()
    monkeypatch.setattr(checker.shutil, "which", lambda name: "uv.exe")
    python = tmp_path / "venv" / "Scripts" / "python.exe"
    artifact = tmp_path / "anyalgebra-0.0.0.dev0.tar.gz"

    offline = checker._uv_install_command(python, artifact, allow_network=False)
    online = checker._uv_install_command(python, artifact, allow_network=True)

    assert offline == [
        "uv.exe",
        "--offline",
        "pip",
        "install",
        "--python",
        python,
        "--no-deps",
        artifact,
    ]
    assert online[:2] == ["uv.exe", "--native-tls"]
    assert "--offline" not in online
