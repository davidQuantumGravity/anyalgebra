"""Tests for the bounded, evidence-only AlgMul runtime-load capture tool."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from anyalgebra.legacy.mathematica import MathematicaExecutionState


_TOOL = Path(__file__).parents[2] / "tools" / "capture_algmul_runtime.py"
_SPEC = importlib.util.spec_from_file_location("capture_algmul_runtime", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
runtime = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = runtime
_SPEC.loader.exec_module(runtime)


def _source(tmp_path: Path, contents: bytes = b'BeginPackage["AlgMul`"]') -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "AlgMul.wl"
    path.write_bytes(contents)
    return path


def _payload(source: Path, *, version: str = "12.0") -> dict[str, Any]:
    data = source.read_bytes()
    package_names = sorted(
        runtime._REQUIRED_SYMBOLS, key=lambda item: item.casefold().replace("$", "~")
    )
    return {
        "schemaType": "AnyAlgebra.AlgMulRuntimeLoad",
        "schemaVersion": "1",
        "source": {
            "path": str(source.resolve()),
            "sha256": hashlib.sha256(data).hexdigest().upper(),
            "bytes": len(data),
        },
        "kernel": {"version": "12.0 test", "versionNumber": version},
        "load": {
            "status": "loaded",
            "resultHead": "Symbol",
            "initialContext": "Global`",
            "contextAfterLoad": "Global`",
            "initialContextPath": ["System`", "Global`"],
            "contextPathAfterLoad": ["System`", "Global`"],
            "messages": ["message"],
            "output": "load output",
        },
        "names": {"newPackage": package_names, "newGlobal": []},
        "requiredSymbols": list(runtime._REQUIRED_SYMBOLS),
        "missingRequiredSymbols": [],
        "symbols": [
            {
                "name": name,
                "attributes": ["Protected"],
                "ownValueCount": 0,
                "downValueCount": 1,
                "upValueCount": 0,
                "subValueCount": 0,
                "definitionHash": ("ABCDEF" * 11)[:64],
            }
            for name in package_names
        ],
        "registries": {"algebras": [], "routes": []},
    }


class _Adapter:
    def __init__(
        self,
        stdout: str,
        *,
        state: MathematicaExecutionState = MathematicaExecutionState.SUCCEEDED,
    ) -> None:
        self.stdout = stdout
        self.state = state
        self.version = "12.0"
        self.stderr = "kernel stderr"
        self.diagnostic = "test transport"

    def execute(self, script: str) -> SimpleNamespace:
        assert "ALG_MUL_RUNTIME_BEGIN" in script
        assert "FileHash" in script
        assert "$VersionNumber" in script
        return SimpleNamespace(
            state=self.state,
            version=self.version,
            stdout=self.stdout,
            stderr=self.stderr,
            diagnostic=self.diagnostic,
        )


def _framed(payload: object, *, noisy: bool = False) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    prefix = "banner\nGeneral::warn: retained\n" if noisy else ""
    return f"{prefix}ALG_MUL_RUNTIME_BEGIN\n{encoded}\nALG_MUL_RUNTIME_END\ntrailer\n"


def _capture(source: Path, adapter: object, **kwargs: object) -> Any:
    return runtime.capture_runtime(
        source,
        adapter,
        expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        **kwargs,
    )


def test_capture_accepts_a_framed_payload_and_is_deterministic(tmp_path: Path) -> None:
    source = _source(tmp_path, 'BeginPackage["AlgMul`"]\nπ'.encode())
    payload = _payload(source)
    receipt = _capture(source, _Adapter(_framed(payload, noisy=True)))

    assert receipt.state is runtime.RuntimeCaptureState.SUCCEEDED
    assert receipt.payload == payload
    assert receipt.canonical_bytes == runtime.canonical_runtime_bytes(payload)
    assert b"started" not in receipt.canonical_bytes


def test_capture_refuses_mismatch_before_kernel_execution(tmp_path: Path) -> None:
    source = _source(tmp_path)
    adapter = _Adapter(_framed(_payload(source)))

    with pytest.raises(runtime.RuntimeCaptureError, match="pinned"):
        runtime.capture_runtime(source, adapter, expected_source_sha256="0" * 64)


@pytest.mark.parametrize(
    "payload",
    [
        {"schemaType": "wrong"},
        {"schemaType": "AnyAlgebra.AlgMulRuntimeLoad", "schemaVersion": "1"},
    ],
)
def test_capture_rejects_malformed_payload(tmp_path: Path, payload: object) -> None:
    source = _source(tmp_path)
    with pytest.raises(runtime.RuntimeCaptureError, match="payload"):
        _capture(source, _Adapter(_framed(payload)))


def test_capture_rejects_duplicate_json_keys_and_duplicate_frames(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    duplicate = '{"schemaType":"AnyAlgebra.AlgMulRuntimeLoad","schemaType":"x"}'
    with pytest.raises(runtime.RuntimeCaptureError, match="duplicate"):
        _capture(
            source, _Adapter(f"ALG_MUL_RUNTIME_BEGIN\n{duplicate}\nALG_MUL_RUNTIME_END")
        )
    payload = _payload(source)
    with pytest.raises(runtime.RuntimeCaptureError, match="exactly one"):
        _capture(source, _Adapter(_framed(payload) + _framed(payload)))


def test_capture_rejects_payload_version_and_source_binding_mismatches(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    payload = _payload(source, version="13.0")
    with pytest.raises(runtime.RuntimeCaptureError, match="kernel version"):
        _capture(source, _Adapter(_framed(payload)))
    payload = _payload(source)
    payload["source"] = {**payload["source"], "bytes": 99}
    with pytest.raises(runtime.RuntimeCaptureError, match="source bytes"):
        _capture(source, _Adapter(_framed(payload)))


def test_capture_exposes_typed_adapter_failure_and_missing_required_state(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    failed = _capture(
        source,
        _Adapter("", state=MathematicaExecutionState.TIMED_OUT),
    )
    assert failed.state is runtime.RuntimeCaptureState.ADAPTER_FAILURE
    payload = _payload(source)
    payload["names"] = {
        "newPackage": [
            item for item in runtime._REQUIRED_SYMBOLS if item != "AlgMul`MakeAlg"
        ],
        "newGlobal": [],
    }
    payload["names"]["newPackage"].sort(
        key=lambda item: item.casefold().replace("$", "~")
    )
    payload["symbols"] = [
        item for item in payload["symbols"] if item["name"] != "AlgMul`MakeAlg"
    ]
    payload["missingRequiredSymbols"] = ["AlgMul`MakeAlg"]
    payload["load"] = {
        **payload["load"],
        "status": "completed-with-missing-required-symbols",
    }
    missing = _capture(source, _Adapter(_framed(payload)))
    assert missing.state is runtime.RuntimeCaptureState.MISSING_REQUIRED_SYMBOLS


def test_output_bounds_path_escaping_and_atomic_refusal(tmp_path: Path) -> None:
    source = _source(tmp_path / "sp ace", b"x")
    payload = _payload(source)
    output = tmp_path / "out π.json"
    receipt = _capture(source, _Adapter(_framed(payload)))
    runtime.write_runtime_fixture(output, receipt)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError):
        runtime.write_runtime_fixture(output, receipt)
    with pytest.raises(runtime.RuntimeCaptureError, match="output"):
        _capture(source, _Adapter("x" * 200), max_payload_bytes=64)


def test_control_flow_is_not_reclassified(tmp_path: Path) -> None:
    source = _source(tmp_path)

    class Interrupting:
        def execute(self, script: str) -> SimpleNamespace:
            del script
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _capture(source, Interrupting())


@pytest.mark.parametrize("signal", [SystemExit, GeneratorExit])
def test_other_control_flow_is_not_reclassified(
    tmp_path: Path, signal: type[BaseException]
) -> None:
    source = _source(tmp_path)

    class Interrupting:
        def execute(self, script: str) -> SimpleNamespace:
            del script
            raise signal

    with pytest.raises(signal):
        _capture(source, Interrupting())


def test_fixture_is_current_evidence_with_complete_registries() -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "legacy" / "runtime-load.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    canonical = runtime.canonical_runtime_bytes(payload)

    assert canonical == fixture.read_bytes()
    assert len(payload["registries"]["algebras"]) == 20
    assert len(payload["registries"]["routes"]) == 40
    assert all(
        not record["structureDotAvailable"]
        and record["structureDotHead"] is None
        and record["structureDotHash"] is None
        for record in payload["registries"]["algebras"]
    )


def test_validator_rejects_symbol_and_registry_contradictions(tmp_path: Path) -> None:
    source = _source(tmp_path)
    payload = _payload(source)
    payload["symbols"] = payload["symbols"][:-1]
    with pytest.raises(runtime.RuntimeCaptureError, match="cover"):
        _capture(source, _Adapter(_framed(payload)))
    payload = _payload(source)
    payload["registries"] = {
        "algebras": [
            {
                "name": "O",
                "basisEntries": ["1", "e1"],
                "basisHash": "B" * 64,
                "multiplicationRoute": 'AlgMul`alg["Mul"]["wrong"]',
                "multiplicationValue": "AlgMul`OMul",
                "structureDotRoute": 'AlgMul`alg["StructDot"]["O"]',
                "structureDotAvailable": False,
                "structureDotHead": None,
                "structureDotHash": None,
            }
        ],
        "routes": [
            'AlgMul`alg["Mul"]["wrong"]',
            'AlgMul`alg["StructDot"]["O"]',
        ],
    }
    with pytest.raises(runtime.RuntimeCaptureError, match="route"):
        _capture(source, _Adapter(_framed(payload)))


def test_receipt_returns_fresh_payload_and_atomic_failure_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    receipt = _capture(source, _Adapter(_framed(_payload(source))))
    first = receipt.payload
    assert first is not None
    first["schemaType"] = "forged"
    assert receipt.payload is not None
    assert receipt.payload["schemaType"] == "AnyAlgebra.AlgMulRuntimeLoad"
    output = tmp_path / "atomic.json"

    def fail_link(source_name: object, target_name: object) -> None:
        del source_name, target_name
        raise OSError("link refused")

    monkeypatch.setattr(runtime.os, "link", fail_link)
    with pytest.raises(OSError, match="link refused"):
        runtime.write_runtime_fixture(output, receipt)
    assert not output.exists()
    assert not list(tmp_path.glob(".atomic.json.*"))


def test_private_bounds_and_frame_rejections_are_explicit(tmp_path: Path) -> None:
    source = _source(tmp_path)
    with pytest.raises(runtime.RuntimeCaptureError, match="regular"):
        runtime._read_source(tmp_path / "missing.wl")
    with pytest.raises(runtime.RuntimeCaptureError, match="regular"):
        runtime._read_source(tmp_path)
    empty = _source(tmp_path / "empty", b"")
    with pytest.raises(runtime.RuntimeCaptureError, match="bounded"):
        runtime._read_source(empty)
    with pytest.raises(runtime.RuntimeCaptureError, match="complete frame"):
        runtime._framed_payload("no frame", 100)
    with pytest.raises(runtime.RuntimeCaptureError, match="one JSON object"):
        runtime._framed_payload("ALG_MUL_RUNTIME_BEGIN\n[]\nALG_MUL_RUNTIME_END", 100)
    with pytest.raises(runtime.RuntimeCaptureError, match="strict JSON"):
        runtime._framed_payload(
            "ALG_MUL_RUNTIME_BEGIN\n{bad}\nALG_MUL_RUNTIME_END", 100
        )
    assert "FromCharacterCode" in runtime._wl_string('π\\"')
    assert (
        runtime.RuntimeCaptureReceipt._create(
            runtime.RuntimeCaptureState.ADAPTER_FAILURE, None, "absence"
        ).payload
        is None
    )
    payload = _payload(source)
    with pytest.raises(runtime.RuntimeCaptureError, match="max payload"):
        _capture(source, _Adapter(_framed(payload)), max_payload_bytes=0)
    with pytest.raises(runtime.RuntimeCaptureError, match="expected source hash"):
        runtime.capture_runtime(
            source, _Adapter(_framed(payload)), expected_source_sha256="x"
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.update({"schemaType": "wrong"}), "schema"),
        (lambda p: p["load"].update({"status": "unknown"}), "status"),
        (
            lambda p: p["names"].update({"newGlobal": list(p["names"]["newPackage"])}),
            "overlap",
        ),
        (lambda p: p.update({"missingRequiredSymbols": ["not-required"]}), "required"),
        (lambda p: p["symbols"][0].update({"definitionHash": "bad"}), "hash"),
    ],
)
def test_validator_rejects_remaining_untrusted_shapes(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None], message: str
) -> None:
    source = _source(tmp_path)
    payload = _payload(source)
    mutate(payload)
    with pytest.raises(runtime.RuntimeCaptureError, match=message):
        _capture(source, _Adapter(_framed(payload)))


def test_load_failure_and_fixture_write_boundaries(tmp_path: Path) -> None:
    source = _source(tmp_path)
    payload = _payload(source)
    payload["load"]["status"] = "failed"
    failed = _capture(source, _Adapter(_framed(payload)))
    assert failed.state is runtime.RuntimeCaptureState.LOAD_FAILED
    absence = runtime.RuntimeCaptureReceipt._create(
        runtime.RuntimeCaptureState.ADAPTER_FAILURE, None, "absent"
    )
    with pytest.raises(runtime.RuntimeCaptureError, match="adapter-failure"):
        runtime.write_runtime_fixture(tmp_path / "x.json", absence)
    receipt = _capture(source, _Adapter(_framed(_payload(source))))
    with pytest.raises(runtime.RuntimeCaptureError, match="parent"):
        runtime.write_runtime_fixture(tmp_path / "none" / "x.json", receipt)
    output = tmp_path / "replace.json"
    output.write_text("old", encoding="utf-8")
    runtime.write_runtime_fixture(output, receipt, allow_overwrite=True)
    assert output.read_bytes() == receipt.canonical_bytes


def test_cli_uses_explicit_adapter_and_writes_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    output = tmp_path / "cli.json"
    payload = _payload(source)

    class FakeAdapter:
        @classmethod
        def create(cls, **kwargs: object) -> FakeAdapter:
            assert kwargs["required_version"] == "12.0"
            assert kwargs["max_output_bytes"] == 1_048_576
            return cls()

        def execute(self, script: str) -> SimpleNamespace:
            assert "FromCharacterCode" in script
            return SimpleNamespace(
                state=MathematicaExecutionState.SUCCEEDED,
                version="12.0",
                stdout=_framed(payload),
            )

    import anyalgebra.legacy.mathematica as mathematica

    monkeypatch.setattr(mathematica, "MathematicaAdapter", FakeAdapter)
    assert (
        runtime.main(
            [
                "--source",
                str(source),
                "--output",
                str(output),
                "--executable",
                "fake-kernel",
                "--required-version",
                "12.0",
                "--expected-source-sha256",
                hashlib.sha256(source.read_bytes()).hexdigest(),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_small_internal_guards_and_untrusted_receipts(tmp_path: Path) -> None:
    source = _source(tmp_path)
    with pytest.raises(TypeError, match="factory"):
        runtime.RuntimeCaptureReceipt()
    with pytest.raises(runtime.RuntimeCaptureError, match="invalid text"):
        runtime._text("", "text")
    with pytest.raises(runtime.RuntimeCaptureError, match="exceeds"):
        runtime._list("not-list", "items")
    with pytest.raises(runtime.RuntimeCaptureError, match="unique"):
        runtime._names(["x", "x"], "names")
    with pytest.raises(runtime.RuntimeCaptureError, match="invalid count"):
        runtime._count(True, "count")
    with pytest.raises(runtime.RuntimeCaptureError, match="non-finite"):
        runtime._reject_json_constant("NaN")
    with pytest.raises(runtime.RuntimeCaptureError, match="non-finite"):
        runtime._framed_payload("ALG_MUL_RUNTIME_BEGIN\nNaN\nALG_MUL_RUNTIME_END", 100)
    forged = runtime.RuntimeCaptureReceipt._create(
        runtime.RuntimeCaptureState.SUCCEEDED, b"{}\n", "forged"
    )
    with pytest.raises(runtime.RuntimeCaptureError, match="untrusted shape"):
        runtime.write_runtime_fixture(tmp_path / "forged.json", forged)
    valid = _capture(source, _Adapter(_framed(_payload(source))))
    mismatched = runtime.RuntimeCaptureReceipt._create(
        runtime.RuntimeCaptureState.LOAD_FAILED,
        valid.canonical_bytes,
        "forged diagnostic",
    )
    with pytest.raises(runtime.RuntimeCaptureError, match="receipt state"):
        runtime.write_runtime_fixture(tmp_path / "mismatched.json", mismatched)


@pytest.mark.parametrize(
    ("change", "pattern"),
    [
        (lambda p: p["source"].update({"sha256": "G" * 64}), "source hash"),
        (
            lambda p: (
                p["names"].update({"newPackage": p["names"]["newPackage"][1:]}),
                p.update({"missingRequiredSymbols": []}),
            ),
            "missing-required",
        ),
        (lambda p: p["symbols"].append(p["symbols"][0].copy()), "duplicate symbol"),
    ],
)
def test_remaining_source_and_symbol_guards(
    tmp_path: Path, change: Callable[[dict[str, Any]], object], pattern: str
) -> None:
    source = _source(tmp_path)
    payload = _payload(source)
    change(payload)
    with pytest.raises(runtime.RuntimeCaptureError, match=pattern):
        _capture(source, _Adapter(_framed(payload)))


@pytest.mark.parametrize(
    ("change", "pattern"),
    [
        (lambda r: r.update({"basisEntries": []}), "basis entries"),
        (lambda r: r.update({"basisHash": "x"}), "basis hash"),
        (lambda r: r.update({"multiplicationValue": "$Failed"}), "multiplication"),
        (lambda r: r.update({"structureDotAvailable": "false"}), "availability"),
        (
            lambda r: r.update(
                {
                    "structureDotAvailable": True,
                    "structureDotHead": "Symbol",
                    "structureDotHash": "Z" * 64,
                }
            ),
            "structure-dot hash",
        ),
        (
            lambda r: r.update(
                {"structureDotHead": "not-null", "structureDotHash": None}
            ),
            "unavailable structure-dot",
        ),
    ],
)
def test_registry_guard_matrix(
    tmp_path: Path, change: Callable[[dict[str, Any]], object], pattern: str
) -> None:
    source = _source(tmp_path)
    payload = _payload(source)
    record: dict[str, Any] = {
        "name": "O",
        "basisEntries": ["1"],
        "basisHash": "A" * 64,
        "multiplicationRoute": 'AlgMul`alg["Mul"]["O"]',
        "multiplicationValue": "AlgMul`OMul",
        "structureDotRoute": 'AlgMul`alg["StructDot"]["O"]',
        "structureDotAvailable": False,
        "structureDotHead": None,
        "structureDotHash": None,
    }
    change(record)
    payload["registries"] = {
        "algebras": [record],
        "routes": [
            'AlgMul`alg["Mul"]["O"]',
            'AlgMul`alg["StructDot"]["O"]',
        ],
    }
    with pytest.raises(runtime.RuntimeCaptureError, match=pattern):
        _capture(source, _Adapter(_framed(payload)))


def test_remaining_evidence_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    monkeypatch.setattr(
        runtime.Path,
        "read_bytes",
        lambda self: (_ for _ in ()).throw(OSError("denied")),
    )
    with pytest.raises(runtime.RuntimeCaptureError, match="could not be read"):
        runtime._read_source(source)
    monkeypatch.undo()

    payload = _payload(source)
    payload["names"]["newPackage"].reverse()
    with pytest.raises(runtime.RuntimeCaptureError, match="deterministic order"):
        _capture(source, _Adapter(_framed(payload)))

    payload = _payload(source)
    payload["source"]["sha256"] = "B" * 64
    with pytest.raises(runtime.RuntimeCaptureError, match="source identity"):
        _capture(source, _Adapter(_framed(payload)))

    payload = _payload(source)
    payload["names"]["newPackage"] = [
        name for name in payload["names"]["newPackage"] if name != "AlgMul`MakeAlg"
    ]
    payload["symbols"] = [
        record for record in payload["symbols"] if record["name"] != "AlgMul`MakeAlg"
    ]
    payload["missingRequiredSymbols"] = ["AlgMul`MakeAlg"]
    with pytest.raises(runtime.RuntimeCaptureError, match="loaded status"):
        _capture(source, _Adapter(_framed(payload)))

    payload = _payload(source)
    payload["load"]["status"] = "completed-with-missing-required-symbols"
    with pytest.raises(runtime.RuntimeCaptureError, match="load status"):
        _capture(source, _Adapter(_framed(payload)))

    payload = _payload(source)
    payload["symbols"].reverse()
    with pytest.raises(runtime.RuntimeCaptureError, match="name order"):
        _capture(source, _Adapter(_framed(payload)))


def test_registry_order_receipt_and_adapter_failure_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    record: dict[str, Any] = {
        "name": "O",
        "basisEntries": ["1"],
        "basisHash": "A" * 64,
        "multiplicationRoute": 'AlgMul`alg["Mul"]["O"]',
        "multiplicationValue": "AlgMul`OMul",
        "structureDotRoute": 'AlgMul`alg["StructDot"]["O"]',
        "structureDotAvailable": True,
        "structureDotHead": "Symbol",
        "structureDotHash": "B" * 64,
    }
    payload = _payload(source)
    payload["registries"] = {
        "algebras": [record],
        "routes": [
            'AlgMul`alg["Mul"]["O"]',
            'AlgMul`alg["StructDot"]["O"]',
        ],
    }
    assert (
        _capture(source, _Adapter(_framed(payload))).state
        is runtime.RuntimeCaptureState.SUCCEEDED
    )

    duplicate = _payload(source)
    duplicate["registries"] = {
        "algebras": [record, record.copy()],
        "routes": payload["registries"]["routes"],
    }
    with pytest.raises(runtime.RuntimeCaptureError, match="duplicate registry"):
        _capture(source, _Adapter(_framed(duplicate)))
    routes = _payload(source)
    routes["registries"] = {"algebras": [record], "routes": []}
    with pytest.raises(runtime.RuntimeCaptureError, match="routes"):
        _capture(source, _Adapter(_framed(routes)))

    fixture = Path(__file__).parents[1] / "fixtures" / "legacy" / "runtime-load.json"
    pinned = json.loads(fixture.read_text(encoding="utf-8"))
    pinned["registries"]["algebras"][0], pinned["registries"]["algebras"][1] = (
        pinned["registries"]["algebras"][1],
        pinned["registries"]["algebras"][0],
    )
    pinned["source"] = _payload(source)["source"]
    with monkeypatch.context() as patch:
        patch.setattr(
            runtime,
            "PINNED_ALGMUL_SHA256",
            hashlib.sha256(source.read_bytes()).hexdigest().upper(),
        )
        with pytest.raises(runtime.RuntimeCaptureError, match="registry order"):
            _capture(source, _Adapter(_framed(pinned)))

    no_version = SimpleNamespace(
        state=MathematicaExecutionState.SUCCEEDED,
        version=None,
        stdout=_framed(_payload(source)),
    )
    with pytest.raises(runtime.RuntimeCaptureError, match="preflight version"):
        _capture(
            source,
            type("NoVersion", (), {"execute": lambda self, script: no_version})(),
        )

    receipt = _capture(source, _Adapter(_framed(_payload(source))))
    noncanonical = runtime.RuntimeCaptureReceipt._create(
        receipt.state, receipt.canonical_bytes + b"\n", receipt.diagnostic
    )
    invalid = runtime.RuntimeCaptureReceipt._create(
        receipt.state, b"{", receipt.diagnostic
    )
    with pytest.raises(runtime.RuntimeCaptureError, match="canonical"):
        runtime.write_runtime_fixture(tmp_path / "noncanonical.json", noncanonical)
    with pytest.raises(runtime.RuntimeCaptureError, match="invalid"):
        runtime.write_runtime_fixture(tmp_path / "invalid.json", invalid)
    monkeypatch.setattr(
        runtime.os,
        "link",
        lambda source, destination: (_ for _ in ()).throw(FileExistsError()),
    )
    with pytest.raises(FileExistsError, match="refusing"):
        runtime.write_runtime_fixture(tmp_path / "race.json", receipt)


def test_cli_surfaces_typed_adapter_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingAdapter:
        @classmethod
        def create(cls, **kwargs: object) -> FailingAdapter:
            del kwargs
            return cls()

        def execute(self, script: str) -> SimpleNamespace:
            del script
            return SimpleNamespace(
                state=MathematicaExecutionState.TIMED_OUT,
                diagnostic="bounded timeout",
            )

    import anyalgebra.legacy.mathematica as mathematica

    monkeypatch.setattr(mathematica, "MathematicaAdapter", FailingAdapter)
    source = _source(tmp_path)
    with pytest.raises(runtime.RuntimeCaptureError, match="bounded timeout"):
        runtime.main(
            [
                "--source",
                str(source),
                "--output",
                str(tmp_path / "failure.json"),
                "--executable",
                "fake-kernel",
                "--expected-source-sha256",
                hashlib.sha256(source.read_bytes()).hexdigest(),
            ]
        )
