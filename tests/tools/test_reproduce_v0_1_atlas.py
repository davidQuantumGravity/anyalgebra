"""Contracts for cross-artifact reproduction of the v0.1 reference atlas."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "reproduce_v0_1_atlas.py"


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("reproduce_v01_atlas", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _semantic() -> dict[str, object]:
    return {
        "atlasSemanticHash": (
            "sha256:245384be017e38895d684456272eddb5e81a2974220debf348f41e17561eb843"
        ),
        "atlasTreeSha256": "A" * 64,
        "canonicalIdsSha256": "B" * 64,
        "objectCount": 862,
        "queryCounts": {
            "all": 129,
            "associative": 12,
            "idempotent": 7,
            "uniqueIdentity": 15,
        },
        "specSemanticHash": (
            "sha256:361253fe9e43ad6aac9dadcc3aa8d0fc97572a4d8f58939aa282134d1a596dea"
        ),
        "verifiedObjectCount": 862,
    }


def _run(kind: str) -> dict[str, object]:
    suffix = ".tar.gz" if kind == "sdist" else "-py3-none-any.whl"
    return {
        "artifact": f"anyalgebra-0.1.0{suffix}",
        "artifactKind": kind,
        "packageVersion": "0.1.0",
        "optionalModulesAttempted": [],
        "optionalModulesImported": [],
        "semantic": _semantic(),
    }


def test_compare_accepts_exact_wheel_and_sdist_semantic_identity() -> None:
    module = _module()

    record = module.compare_reproductions((_run("wheel"), _run("sdist")))

    assert record["ok"] is True
    assert record["semantic"] == _semantic()
    assert [item["artifactKind"] for item in record["runs"]] == ["sdist", "wheel"]
    assert all("semantic" not in item for item in record["runs"])
    assert "packageLocation" not in json.dumps(record)


def test_compare_rejects_cross_artifact_semantic_difference() -> None:
    module = _module()
    wheel = _run("wheel")
    sdist = copy.deepcopy(_run("sdist"))
    sdist_semantic = sdist["semantic"]
    assert isinstance(sdist_semantic, dict)
    sdist_semantic["atlasTreeSha256"] = "C" * 64

    with pytest.raises(module.AtlasReproductionError, match="semantic outputs differ"):
        module.compare_reproductions((wheel, sdist))


@pytest.mark.parametrize(
    "runs, message",
    [
        ((_run("wheel"),), "one wheel and one sdist"),
        ((_run("wheel"), _run("wheel")), "one wheel and one sdist"),
    ],
)
def test_compare_rejects_missing_or_duplicate_artifact_kind(
    runs: tuple[dict[str, object], ...], message: str
) -> None:
    module = _module()

    with pytest.raises(module.AtlasReproductionError, match=message):
        module.compare_reproductions(runs)


def test_compare_rejects_optional_import_or_frozen_reference_drift() -> None:
    module = _module()
    wheel = _run("wheel")
    sdist = _run("sdist")
    wheel["optionalModulesImported"] = ["numpy"]

    with pytest.raises(module.AtlasReproductionError, match="optional modules"):
        module.compare_reproductions((wheel, sdist))

    wheel = _run("wheel")
    wheel_semantic = wheel["semantic"]
    assert isinstance(wheel_semantic, dict)
    wheel_semantic["objectCount"] = 861
    sdist = copy.deepcopy(wheel)
    sdist["artifact"] = "anyalgebra-0.1.0.tar.gz"
    sdist["artifactKind"] = "sdist"
    with pytest.raises(module.AtlasReproductionError, match="frozen reference"):
        module.compare_reproductions((wheel, sdist))


def test_fixture_payload_strips_exactly_one_transport_newline(tmp_path: Path) -> None:
    module = _module()
    fixture = tmp_path / "spec.json"
    fixture.write_bytes(b"{}\n")

    assert module._fixture_payload(fixture) == b"{}"

    fixture.write_bytes(b"{}")
    with pytest.raises(module.AtlasReproductionError, match="one transport newline"):
        module._fixture_payload(fixture)
    fixture.write_bytes(b"{}\n\n")
    with pytest.raises(module.AtlasReproductionError, match="one transport newline"):
        module._fixture_payload(fixture)
