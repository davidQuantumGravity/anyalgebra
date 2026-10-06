"""Contracts for the bounded v0.1 distribution-artifact gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "check_v0_1_artifacts.py"


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("check_v01_artifacts", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repository(tmp_path: Path, version: str = "0.1.0") -> Path:
    (tmp_path / "src/anyalgebra").mkdir(parents=True)
    (tmp_path / "src/anyalgebra/_version.py").write_text(
        f'__version__ = "{version}"\n', encoding="utf-8"
    )
    (tmp_path / "pyproject.toml").write_text(
        "\n".join(
            (
                "[project]",
                'name = "anyalgebra"',
                'requires-python = ">=3.11"',
                'dynamic = ["version"]',
                "dependencies = []",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return tmp_path


def _result() -> SimpleNamespace:
    artifacts = (
        SimpleNamespace(
            filename="anyalgebra-0.1.0.tar.gz",
            kind="sdist",
            sha256="B" * 64,
            member_count=100,
            metadata_name="anyalgebra",
            metadata_version="0.1.0",
            requires_python=">=3.11",
            runtime_dependencies=(),
        ),
        SimpleNamespace(
            filename="anyalgebra-0.1.0-py3-none-any.whl",
            kind="wheel",
            sha256="A" * 64,
            member_count=80,
            metadata_name="anyalgebra",
            metadata_version="0.1.0",
            requires_python=">=3.11",
            runtime_dependencies=(),
        ),
    )
    installations = tuple(
        SimpleNamespace(
            artifact=item.filename,
            version="0.1.0",
            package_location="C:/temporary/purelib/anyalgebra/__init__.py",
            purelib_location="C:/temporary/purelib",
            optional_modules_attempted=(),
            optional_modules_imported=(),
            reference_backend_verified=True,
        )
        for item in artifacts
    )
    return SimpleNamespace(
        metadata=SimpleNamespace(
            name="anyalgebra",
            version="0.1.0",
            requires_python=">=3.11",
            runtime_dependencies=(),
        ),
        inspection=SimpleNamespace(artifacts=artifacts, errors=(), ok=True),
        installations=installations,
    )


def test_audit_requires_exact_v0_1_metadata_and_two_clean_installs(
    tmp_path: Path,
) -> None:
    module = _module()
    calls: list[tuple[Path, bool]] = []

    def runner(root: Path, **kwargs: object) -> SimpleNamespace:
        calls.append((root, bool(kwargs["inspect_only"])))
        return _result()

    record = module.audit(_repository(tmp_path), runner=runner)

    assert calls == [(tmp_path.resolve(), False)]
    assert record["ok"] is True
    assert record["version"] == "0.1.0"
    assert [item["kind"] for item in record["artifacts"]] == ["sdist", "wheel"]
    assert len(record["installations"]) == 2
    assert all(item["referenceBackendVerified"] for item in record["installations"])
    assert "temporary" not in json.dumps(record)


def test_audit_rejects_old_version_before_build(tmp_path: Path) -> None:
    module = _module()
    calls: list[object] = []

    with pytest.raises(module.V01ArtifactError, match=r"version must be 0\.1\.0"):
        module.audit(
            _repository(tmp_path, "0.0.1"),
            runner=lambda *args, **kwargs: calls.append((args, kwargs)),
        )

    assert calls == []


def test_audit_rejects_missing_archive_kind_and_install_evidence(
    tmp_path: Path,
) -> None:
    module = _module()
    result = _result()
    result.inspection.artifacts = result.inspection.artifacts[:1]
    result.installations = ()

    with pytest.raises(module.V01ArtifactError, match="one wheel and one sdist"):
        module.audit(_repository(tmp_path), runner=lambda *args, **kwargs: result)


def test_inspect_only_record_is_canonical_and_omits_install_claims(
    tmp_path: Path,
) -> None:
    module = _module()
    result = _result()
    result.installations = ()

    first = module.audit(
        _repository(tmp_path),
        inspect_only=True,
        runner=lambda *args, **kwargs: result,
    )
    second = module.audit(
        tmp_path,
        inspect_only=True,
        runner=lambda *args, **kwargs: result,
    )

    assert first == second
    assert first["installations"] == []
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )
