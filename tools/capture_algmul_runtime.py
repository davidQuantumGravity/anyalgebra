"""Capture one bounded, clean-kernel AlgMul load-state receipt.

This is maintainer tooling, deliberately outside neutral imports.  The receipt
is legacy evidence only: it records what one pinned source did in one fresh
kernel, without treating that result as a mathematical correctness claim.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Protocol

from anyalgebra.legacy.algmul_static import MAX_SOURCE_BYTES, PINNED_ALGMUL_SHA256
from anyalgebra.legacy.mathematica import MathematicaExecutionState


_SCHEMA_TYPE = "AnyAlgebra.AlgMulRuntimeLoad"
_SCHEMA_VERSION = "1"
_FRAME_BEGIN = "ALG_MUL_RUNTIME_BEGIN"
_FRAME_END = "ALG_MUL_RUNTIME_END"
# V84's public factory accepts at most 1 MiB of aggregate process output.
# Keep the load receipt cap within that independent optional-adapter boundary.
_MAX_PAYLOAD_BYTES = 1_048_576
_MAX_ITEMS = 4096
_MAX_TEXT = 131_072
_MAX_SYMBOLS = 2048
_REQUIRED_SYMBOLS = (
    "AlgMul`MMul",
    "AlgMul`MTMul",
    "AlgMul`MakeAlg",
    "AlgMul`MakeChar",
    "AlgMul`MakeProperty",
    "AlgMul`OMul",
    "AlgMul`TMul",
)
_EXPECTED_REGISTRY_NAMES = (
    "O",
    "H",
    "C",
    "O2",
    "H2",
    "C2",
    "O3",
    "H3",
    "C3",
    "Os",
    "Hs",
    "Cs",
    "Os2",
    "Hs2",
    "Cs2",
    "Os3",
    "Hs3",
    "Cs3",
    "CxH",
    "D",
)


class RuntimeCaptureState(StrEnum):
    """Typed outcome of a runtime-load attempt."""

    SUCCEEDED = "succeeded"
    ADAPTER_FAILURE = "adapter_failure"
    LOAD_FAILED = "load_failed"
    MISSING_REQUIRED_SYMBOLS = "missing_required_symbols"


class RuntimeCaptureError(ValueError):
    """The external runtime payload is malformed, untrusted, or out of scope."""


class RuntimeAdapter(Protocol):
    """Minimal injected V84 adapter surface used without leaking process types."""

    def execute(self, script: str, /) -> object: ...


@dataclass(frozen=True, slots=True, init=False)
class RuntimeCaptureReceipt:
    """Immutable runtime result; payload exists only after full validation."""

    state: RuntimeCaptureState
    canonical_bytes: bytes | None
    diagnostic: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("RuntimeCaptureReceipt is factory-owned")

    @classmethod
    def _create(
        cls,
        state: RuntimeCaptureState,
        canonical_bytes: bytes | None,
        diagnostic: str,
    ) -> RuntimeCaptureReceipt:
        value = object.__new__(cls)
        object.__setattr__(value, "state", state)
        object.__setattr__(value, "canonical_bytes", canonical_bytes)
        object.__setattr__(value, "diagnostic", diagnostic)
        return value

    @property
    def payload(self) -> dict[str, object] | None:
        """Return a fresh parsed copy; callers cannot mutate stored evidence."""
        if self.canonical_bytes is None:
            return None
        value = json.loads(
            self.canonical_bytes,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_json_constant,
        )
        assert type(value) is dict
        return value


def _no_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise RuntimeCaptureError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_json_constant(value: str) -> object:
    raise RuntimeCaptureError(f"non-finite JSON constant is forbidden: {value}")


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def canonical_runtime_bytes(payload: object) -> bytes:
    """Return the normalized fixture bytes after rejecting an untrusted shape."""
    checked = _validate_payload(
        payload, source_path=None, source_bytes=None, version=None
    )
    return _canonical(checked)


def _read_source(path_value: str | Path) -> tuple[Path, bytes]:
    try:
        path = Path(path_value).resolve(strict=True)
        metadata = path.stat()
    except (OSError, RuntimeError, TypeError) as error:
        raise RuntimeCaptureError("source must be a readable regular file") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeCaptureError("source must be a readable regular file")
    try:
        data = path.read_bytes()
    except OSError as error:
        raise RuntimeCaptureError("source could not be read") from error
    if not data or len(data) > MAX_SOURCE_BYTES:
        raise RuntimeCaptureError("source exceeds the bounded runtime capture policy")
    return path, data


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _wl_string(value: str) -> str:
    """Encode a filesystem path as a literal WL string, never executable code."""
    return (
        "FromCharacterCode[{"
        + ",".join(str(item) for item in value.encode("utf-8"))
        + '},"UTF8"]'
    )


def _script(source_path: Path) -> str:
    """Return one fresh-kernel audit payload, framed apart from incidental output."""
    source = _wl_string(str(source_path))
    required = "{" + ",".join(_wl_string(item) for item in _REQUIRED_SYMBOLS) + "}"
    variables = (
        f"source={source},required={required},beforeGlobal,beforePackage,"
        "initialContext,initialPath,"
        "contextAfter,pathAfter,result,messages,messageTemp,outputTemp,"
        "messageStream,outputStream,loadOutput,messageOutput,afterGlobal,"
        "afterPackage,newGlobal,newPackage,symbolRecord,summaries,missing,"
        "status,algebraNames,registries,payload,sha256,basisEntries,basis,"
        "multiplication,cleanup,auditHashInput,auditBasisValue,auditSymbolName,"
        "auditHeldSymbol"
    )
    return f'''Module[{{{variables}}},
sha256[auditHashInput_]:=ToUpperCase[IntegerString[Hash[auditHashInput,"SHA256"],16,64]];
basisEntries[auditBasisValue_]:=If[ListQ[auditBasisValue],(ToString[#,InputForm]&/@auditBasisValue),{{}}];
beforeGlobal=Names["Global`*"];beforePackage=Names["AlgMul`*"];
initialContext=$Context;initialPath=$ContextPath;
messageTemp=CreateTemporary[];outputTemp=CreateTemporary[];messageStream=OpenWrite[messageTemp,PageWidth->Infinity];outputStream=OpenWrite[outputTemp,PageWidth->Infinity];
cleanup:=((Check[Close[messageStream],Null];Check[Close[outputStream],Null];messageOutput=Check[Import[messageTemp,"Text"],""];loadOutput=Check[Import[outputTemp,"Text"],""];Check[DeleteFile[messageTemp],Null];Check[DeleteFile[outputTemp],Null]));
result=CheckAbort[Block[{{$Messages={{messageStream}},$Output={{outputStream}}}},Get[source]],$Aborted];cleanup[];messages=If[messageOutput==="",{{}},{{messageOutput}}];
contextAfter=$Context;pathAfter=$ContextPath;afterGlobal=Names["Global`*"];afterPackage=Names["AlgMul`*"];
newGlobal=Sort[Complement[afterGlobal,beforeGlobal]];newPackage=Sort[Complement[afterPackage,beforePackage]];
symbolRecord[auditSymbolName_String]:=ToExpression[auditSymbolName,InputForm,Function[auditHeldSymbol,<|"name"->auditSymbolName,"attributes"->(ToString[#,InputForm]&/@Attributes[auditHeldSymbol]),"ownValueCount"->Length[OwnValues[auditHeldSymbol]],"downValueCount"->Length[DownValues[auditHeldSymbol]],"upValueCount"->Length[UpValues[auditHeldSymbol]],"subValueCount"->Length[SubValues[auditHeldSymbol]],"definitionHash"->sha256[ToString[{{OwnValues[auditHeldSymbol],DownValues[auditHeldSymbol],UpValues[auditHeldSymbol],SubValues[auditHeldSymbol],Attributes[auditHeldSymbol]}},InputForm]]|>,HoldAll]];
summaries=symbolRecord/@Join[newPackage,newGlobal];missing=Complement[required,afterPackage];
status=Which[result===$Aborted,"timeout",result===$Failed,"failed",Length[missing]>0,"completed-with-missing-required-symbols",True,"loaded"];
algebraNames=Check[ToExpression["AlgMul`alg[\\\"algs\\\"]"],{{}}];
registries=If[ListQ[algebraNames],(With[{{key=If[StringQ[#],#,ToString[#,InputForm]],route=("AlgMul`alg[\\\"StructDot\\\"][\\\""<>If[StringQ[#],#,ToString[#,InputForm]]<>"\\\"]")}},With[{{available=Quiet[Check[ToExpression[route,InputForm,Function[Null,ValueQ[#],HoldAll]],False]],basis=Check[ToExpression["AlgMul`alg[\\\"chars\\\"][\\\""<>key<>"\\\"]"],{{}}],multiplication=Check[ToExpression["AlgMul`alg[\\\"Mul\\\"][\\\""<>key<>"\\\"]"],$Failed]}},<|"name"->key,"basisEntries"->basisEntries[basis],"basisHash"->sha256[ToString[basis,InputForm]],"multiplicationRoute"->("AlgMul`alg[\\\"Mul\\\"][\\\""<>key<>"\\\"]"),"multiplicationValue"->ToString[multiplication,InputForm],"structureDotRoute"->route,"structureDotAvailable"->available,"structureDotHead"->If[available,ToString[Head[ToExpression[route]],InputForm],Null],"structureDotHash"->If[available,sha256[ToString[ToExpression[route],InputForm]],Null]|>]]&/@algebraNames),{{}}];
payload=<|"schemaType"->{_wl_string(_SCHEMA_TYPE)},"schemaVersion"->{_wl_string(_SCHEMA_VERSION)},"source"-><|"path"->source,"sha256"->ToUpperCase[IntegerString[FileHash[source,"SHA256"],16,64]],"bytes"->FileByteCount[source]|>,"kernel"-><|"version"->$Version,"versionNumber"->With[{{v=ToString[$VersionNumber,InputForm]}},If[StringEndsQ[v,"."],v<>"0",v]]|>,"load"-><|"status"->status,"resultHead"->ToString[Head[result],InputForm],"initialContext"->initialContext,"contextAfterLoad"->contextAfter,"initialContextPath"->initialPath,"contextPathAfterLoad"->pathAfter,"messages"->messages,"output"->loadOutput|>,"names"-><|"newPackage"->newPackage,"newGlobal"->newGlobal|>,"requiredSymbols"->required,"missingRequiredSymbols"->missing,"symbols"->summaries,"registries"-><|"algebras"->registries,"routes"->Sort[DeleteDuplicates[Join[Lookup[registries,"multiplicationRoute"],Lookup[registries,"structureDotRoute"]]]]|>|>;
Print["{_FRAME_BEGIN}"];Print[ExportString[payload,"RawJSON","Compact"->True]];Print["{_FRAME_END}"]]
'''


def _mapping(value: object, field: str, keys: frozenset[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise RuntimeCaptureError(f"payload {field} has an untrusted shape")
    return value


def _text(
    value: object, field: str, *, maximum: int = _MAX_TEXT, allow_empty: bool = False
) -> str:
    if (
        type(value) is not str
        or (not allow_empty and not value)
        or len(value.encode("utf-8")) > maximum
        or "\x00" in value
    ):
        raise RuntimeCaptureError(f"payload {field} has invalid text")
    return value


def _list(value: object, field: str, *, maximum: int = _MAX_ITEMS) -> list[object]:
    if type(value) is not list or len(value) > maximum:
        raise RuntimeCaptureError(f"payload {field} exceeds its bound")
    return value


def _names(value: object, field: str, *, ordered: bool = True) -> list[str]:
    names = [_text(item, field, maximum=1024) for item in _list(value, field)]
    if len(names) != len(set(names)):
        raise RuntimeCaptureError(f"payload {field} must be unique")
    if ordered and names != sorted(
        names, key=lambda item: item.casefold().replace("$", "~")
    ):
        raise RuntimeCaptureError(f"payload {field} must have deterministic order")
    return names


def _count(value: object, field: str, *, maximum: int = _MAX_ITEMS) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        raise RuntimeCaptureError(f"payload {field} has invalid count")
    return value


def _validate_payload(
    payload: object,
    *,
    source_path: Path | None,
    source_bytes: bytes | None,
    version: str | None,
) -> dict[str, object]:
    root = _mapping(
        payload,
        "root",
        frozenset(
            (
                "schemaType",
                "schemaVersion",
                "source",
                "kernel",
                "load",
                "names",
                "requiredSymbols",
                "missingRequiredSymbols",
                "symbols",
                "registries",
            )
        ),
    )
    if root["schemaType"] != _SCHEMA_TYPE or root["schemaVersion"] != _SCHEMA_VERSION:
        raise RuntimeCaptureError(
            "payload schema is not the trusted runtime-load schema"
        )
    source = _mapping(root["source"], "source", frozenset(("path", "sha256", "bytes")))
    path = _text(source["path"], "source.path", maximum=8192)
    digest = _text(source["sha256"], "source.sha256", maximum=64)
    if len(digest) != 64 or any(char not in "0123456789ABCDEF" for char in digest):
        raise RuntimeCaptureError("payload source hash is invalid")
    byte_count = _count(source["bytes"], "source.bytes", maximum=MAX_SOURCE_BYTES)
    if source_path is not None and source_bytes is not None:
        if path != str(source_path) or digest != _sha256(source_bytes):
            raise RuntimeCaptureError(
                "payload source identity does not bind the executed source"
            )
        if byte_count != len(source_bytes):
            raise RuntimeCaptureError(
                "payload source bytes do not bind the executed source"
            )
    kernel = _mapping(root["kernel"], "kernel", frozenset(("version", "versionNumber")))
    _text(kernel["version"], "kernel.version")
    kernel_number = _text(kernel["versionNumber"], "kernel.versionNumber", maximum=64)
    if version is not None and kernel_number != version:
        raise RuntimeCaptureError(
            "payload kernel version does not match the adapter probe"
        )
    load = _mapping(
        root["load"],
        "load",
        frozenset(
            (
                "status",
                "resultHead",
                "initialContext",
                "contextAfterLoad",
                "initialContextPath",
                "contextPathAfterLoad",
                "messages",
                "output",
            )
        ),
    )
    status = _text(load["status"], "load.status", maximum=128)
    if status not in {
        "loaded",
        "completed-with-missing-required-symbols",
        "failed",
        "timeout",
    }:
        raise RuntimeCaptureError("payload load status is invalid")
    for key in ("resultHead", "initialContext", "contextAfterLoad"):
        _text(load[key], f"load.{key}")
    _text(load["output"], "load.output", allow_empty=True)
    for key in ("initialContextPath", "contextPathAfterLoad", "messages"):
        [_text(item, f"load.{key}") for item in _list(load[key], f"load.{key}")]
    names = _mapping(root["names"], "names", frozenset(("newPackage", "newGlobal")))
    package_names = _names(names["newPackage"], "names.newPackage")
    global_names = _names(names["newGlobal"], "names.newGlobal")
    if set(package_names) & set(global_names):
        raise RuntimeCaptureError("payload new-name partitions overlap")
    required = _names(root["requiredSymbols"], "requiredSymbols", ordered=False)
    missing = _names(
        root["missingRequiredSymbols"], "missingRequiredSymbols", ordered=False
    )
    if tuple(required) != _REQUIRED_SYMBOLS or any(
        item not in required for item in missing
    ):
        raise RuntimeCaptureError("payload required-symbol receipt is contradictory")
    expected_missing = [item for item in required if item not in package_names]
    if missing != expected_missing:
        raise RuntimeCaptureError("payload missing-required receipt is contradictory")
    if status == "loaded" and missing:
        raise RuntimeCaptureError(
            "payload loaded status contradicts missing-required symbols"
        )
    if status == "completed-with-missing-required-symbols" and not missing:
        raise RuntimeCaptureError(
            "payload load status contradicts missing-required symbols"
        )
    symbols = _list(root["symbols"], "symbols", maximum=_MAX_SYMBOLS)
    seen_symbols: set[str] = set()
    symbol_order: list[str] = []
    for item in symbols:
        record = _mapping(
            item,
            "symbol",
            frozenset(
                (
                    "name",
                    "attributes",
                    "ownValueCount",
                    "downValueCount",
                    "upValueCount",
                    "subValueCount",
                    "definitionHash",
                )
            ),
        )
        name = _text(record["name"], "symbol.name", maximum=1024)
        if name in seen_symbols:
            raise RuntimeCaptureError("payload contains duplicate symbol records")
        seen_symbols.add(name)
        symbol_order.append(name)
        _names(record["attributes"], "symbol.attributes")
        for key in ("ownValueCount", "downValueCount", "upValueCount", "subValueCount"):
            _count(record[key], f"symbol.{key}")
        definition_hash = _text(
            record["definitionHash"], "symbol.definitionHash", maximum=64
        )
        if len(definition_hash) != 64 or any(
            char not in "0123456789ABCDEF" for char in definition_hash
        ):
            raise RuntimeCaptureError("payload symbol definition hash is invalid")
    if seen_symbols != set(package_names) | set(global_names):
        raise RuntimeCaptureError(
            "payload symbol summaries do not cover new names exactly"
        )
    if symbol_order != [*package_names, *global_names]:
        raise RuntimeCaptureError("payload symbol summaries are not in name order")
    registries = _mapping(
        root["registries"], "registries", frozenset(("algebras", "routes"))
    )
    registry_names: set[str] = set()
    registry_order: list[str] = []
    expected_routes: set[str] = set()
    for item in _list(registries["algebras"], "registries.algebras"):
        record = _mapping(
            item,
            "registry",
            frozenset(
                (
                    "name",
                    "basisEntries",
                    "basisHash",
                    "multiplicationRoute",
                    "multiplicationValue",
                    "structureDotRoute",
                    "structureDotAvailable",
                    "structureDotHead",
                    "structureDotHash",
                )
            ),
        )
        name = _text(record["name"], "registry.name")
        if name in registry_names:
            raise RuntimeCaptureError("payload contains duplicate registry names")
        registry_names.add(name)
        registry_order.append(name)
        basis_entries = _list(record["basisEntries"], "registry.basisEntries")
        if not basis_entries:
            raise RuntimeCaptureError("payload registry basis entries are empty")
        for entry in basis_entries:
            _text(entry, "registry.basisEntries")
        for key in (
            "basisHash",
            "multiplicationRoute",
            "multiplicationValue",
            "structureDotRoute",
        ):
            _text(record[key], f"registry.{key}")
        basis_hash = record["basisHash"]
        if (
            type(basis_hash) is not str
            or len(basis_hash) != 64
            or any(char not in "0123456789ABCDEF" for char in basis_hash)
        ):
            raise RuntimeCaptureError("payload registry basis hash is invalid")
        if record["multiplicationValue"] == "$Failed":
            raise RuntimeCaptureError("payload registry multiplication is unavailable")
        if type(record["structureDotAvailable"]) is not bool:
            raise RuntimeCaptureError("payload registry availability is invalid")
        if record["structureDotAvailable"]:
            _text(record["structureDotHead"], "registry.structureDotHead")
            structure_hash = _text(
                record["structureDotHash"], "registry.structureDotHash"
            )
            if len(structure_hash) != 64 or any(
                char not in "0123456789ABCDEF" for char in structure_hash
            ):
                raise RuntimeCaptureError(
                    "payload registry structure-dot hash is invalid"
                )
        elif (
            record["structureDotHead"] is not None
            or record["structureDotHash"] is not None
        ):
            raise RuntimeCaptureError(
                "payload unavailable structure-dot is contradictory"
            )
        expected_multiplication = f'AlgMul`alg["Mul"]["{name}"]'
        expected_structure_dot = f'AlgMul`alg["StructDot"]["{name}"]'
        if (
            record["multiplicationRoute"] != expected_multiplication
            or record["structureDotRoute"] != expected_structure_dot
        ):
            raise RuntimeCaptureError("payload registry route does not bind its name")
        expected_routes.update((expected_multiplication, expected_structure_dot))
    routes = _names(registries["routes"], "registries.routes")
    if set(routes) != expected_routes:
        raise RuntimeCaptureError(
            "payload registry routes do not match registry records"
        )
    if (
        digest == PINNED_ALGMUL_SHA256
        and tuple(registry_order) != _EXPECTED_REGISTRY_NAMES
    ):
        raise RuntimeCaptureError(
            "pinned payload registry order is incomplete or changed"
        )
    return root


def _framed_payload(stdout: object, maximum: int) -> dict[str, object]:
    if type(stdout) is not str or len(stdout.encode("utf-8")) > maximum:
        raise RuntimeCaptureError("runtime output exceeds its declared bound")
    text = _text(stdout, "adapter.stdout", maximum=maximum)
    lines = text.splitlines()
    begins = [index for index, line in enumerate(lines) if line == _FRAME_BEGIN]
    ends = [index for index, line in enumerate(lines) if line == _FRAME_END]
    if len(begins) != 1 or len(ends) != 1 or begins[0] >= ends[0] - 1:
        raise RuntimeCaptureError(
            "runtime output must contain exactly one complete frame"
        )
    encoded = "\n".join(lines[begins[0] + 1 : ends[0]])
    try:
        payload = json.loads(
            encoded,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, RuntimeCaptureError):
            raise
        raise RuntimeCaptureError("runtime payload is not valid strict JSON") from error
    if type(payload) is not dict:
        raise RuntimeCaptureError("runtime payload must be one JSON object")
    return payload


def capture_runtime(
    source_path: str | Path,
    adapter: RuntimeAdapter,
    *,
    expected_source_sha256: str = PINNED_ALGMUL_SHA256,
    max_payload_bytes: int = _MAX_PAYLOAD_BYTES,
) -> RuntimeCaptureReceipt:
    """Capture one clean runtime load only after pinning the source receipt."""
    path, source_bytes = _read_source(source_path)
    expected = _text(expected_source_sha256, "expected source hash", maximum=64)
    if len(expected) != 64 or any(
        char not in "0123456789ABCDEFabcdef" for char in expected
    ):
        raise RuntimeCaptureError("expected source hash is invalid")
    if _sha256(source_bytes) != expected.upper():
        raise RuntimeCaptureError(
            "pinned source hash does not match; kernel was not executed"
        )
    if (
        type(max_payload_bytes) is not int
        or not 1 <= max_payload_bytes <= _MAX_PAYLOAD_BYTES
    ):
        raise RuntimeCaptureError("max payload bytes is outside the bounded policy")
    try:
        execution = adapter.execute(_script(path))
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    state = getattr(execution, "state", None)
    if state is not MathematicaExecutionState.SUCCEEDED:
        detail = _text(
            str(getattr(execution, "diagnostic", "adapter did not succeed")),
            "adapter diagnostic",
        )
        return RuntimeCaptureReceipt._create(
            RuntimeCaptureState.ADAPTER_FAILURE, None, detail
        )
    version = getattr(execution, "version", None)
    stdout = getattr(execution, "stdout", None)
    if type(version) is not str or not version:
        raise RuntimeCaptureError(
            "successful adapter execution omitted its preflight version"
        )
    payload = _framed_payload(stdout, max_payload_bytes)
    checked = _validate_payload(
        payload, source_path=path, source_bytes=source_bytes, version=version
    )
    canonical = _canonical(checked)
    load = checked["load"]
    assert type(load) is dict
    status = load["status"]
    if status in {"failed", "timeout"}:
        result_state = RuntimeCaptureState.LOAD_FAILED
    elif checked["missingRequiredSymbols"]:
        result_state = RuntimeCaptureState.MISSING_REQUIRED_SYMBOLS
    else:
        result_state = RuntimeCaptureState.SUCCEEDED
    return RuntimeCaptureReceipt._create(
        result_state, canonical, "validated clean-kernel load receipt"
    )


def write_runtime_fixture(
    output_path: str | Path,
    receipt: RuntimeCaptureReceipt,
    *,
    allow_overwrite: bool = False,
) -> Path:
    """Atomically write validated fixture bytes, never replacing by default."""
    if type(receipt) is not RuntimeCaptureReceipt or receipt.canonical_bytes is None:
        raise RuntimeCaptureError("cannot write an adapter-failure receipt")
    try:
        payload = json.loads(
            receipt.canonical_bytes,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_json_constant,
        )
        checked = _validate_payload(
            payload, source_path=None, source_bytes=None, version=None
        )
        if _canonical(checked) != receipt.canonical_bytes:
            raise RuntimeCaptureError("runtime fixture bytes are not canonical")
        load = checked["load"]
        assert type(load) is dict
        expected_state = (
            RuntimeCaptureState.LOAD_FAILED
            if load["status"] in {"failed", "timeout"}
            else RuntimeCaptureState.MISSING_REQUIRED_SYMBOLS
            if checked["missingRequiredSymbols"]
            else RuntimeCaptureState.SUCCEEDED
        )
        if (
            receipt.state is not expected_state
            or receipt.diagnostic != "validated clean-kernel load receipt"
        ):
            raise RuntimeCaptureError("runtime fixture receipt state is untrusted")
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        if type(error) is RuntimeCaptureError:
            raise
        raise RuntimeCaptureError("runtime fixture bytes are invalid") from error
    path = Path(output_path).resolve(strict=False)
    if path.exists() and not allow_overwrite:
        raise FileExistsError("refusing to overwrite an existing runtime fixture")
    parent = path.parent
    if not parent.is_dir():
        raise RuntimeCaptureError("output parent directory does not exist")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(receipt.canonical_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        if allow_overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                raise FileExistsError(
                    "refusing to overwrite an existing runtime fixture"
                ) from None
            temporary.unlink()
    except BaseException:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise
    return path


def main(arguments: list[str] | None = None) -> int:
    """CLI with explicit source, output, executable, and deliberate overwrite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--required-version")
    parser.add_argument("--max-payload-bytes", type=int, default=_MAX_PAYLOAD_BYTES)
    parser.add_argument(
        "--expected-source-sha256",
        default=PINNED_ALGMUL_SHA256,
        help="expected source digest; defaults to the historical pinned receipt",
    )
    options = parser.parse_args(arguments)
    from anyalgebra.legacy.mathematica import MathematicaAdapter

    adapter = MathematicaAdapter.create(
        executable=options.executable,
        required_version=options.required_version,
        timeout_seconds=options.timeout_seconds,
        max_output_bytes=options.max_payload_bytes,
    )
    receipt = capture_runtime(
        options.source,
        adapter,
        expected_source_sha256=options.expected_source_sha256,
        max_payload_bytes=options.max_payload_bytes,
    )
    if receipt.canonical_bytes is None:
        raise RuntimeCaptureError(receipt.diagnostic)
    write_runtime_fixture(
        options.output, receipt, allow_overwrite=options.allow_overwrite
    )
    print(f"runtime capture state={receipt.state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
