"""Isolated maintainer tooling for an optional WolframScript kernel.

This module is deliberately not imported by the neutral package.  It starts a
fresh external process for each request and exposes only immutable Python
records: neither ``subprocess`` handles nor Wolfram expressions cross the
boundary.  It is an execution transport for the later evaluated AlgMul audit,
not an AlgMul correctness oracle or runtime-manifest parser.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
import os
import re
import subprocess
from threading import Event, Lock, Thread
import time
from typing import IO, Protocol, cast


_MAX_EXECUTABLE_CHARS = 4096
_MAX_SCRIPT_BYTES = 1_048_576
_MAX_OUTPUT_BYTES = 1_048_576
_MAX_TIMEOUT_SECONDS = 3600
_MAX_ENVIRONMENT_VALUE_CHARS = 32_768
_CLEANUP_TIMEOUT_SECONDS = 1
_SAFE_ENVIRONMENT_NAMES = frozenset(
    {"COMSPEC", "PATH", "SystemDrive", "SystemRoot", "TEMP", "TMP", "WINDIR"}
)
_VERSION_CODE = (
    "With[{v = ToString[$VersionNumber, InputForm]}, "
    'If[StringEndsQ[v, "."], v <> "0", v]]'
)
_VERSION_RE = re.compile(r"^(\d+(?:\.\d+)*)$")


class MathematicaExecutionState(StrEnum):
    """Exhaustive success, absence, and process-boundary states."""

    SUCCEEDED = "succeeded"
    EXECUTABLE_UNAVAILABLE = "executable_unavailable"
    VERSION_UNAVAILABLE = "version_unavailable"
    VERSION_INCOMPATIBLE = "version_incompatible"
    LAUNCH_FAILED = "launch_failed"
    TIMED_OUT = "timed_out"
    PROCESS_FAILED = "process_failed"
    PROTOCOL_ERROR = "protocol_error"
    OUTPUT_LIMIT_EXCEEDED = "output_limit_exceeded"


def _text(value: object, field: str, *, maximum: int) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError(f"{field} must be a nonempty string without NUL")
    if len(value) > maximum or any(
        ord(char) < 32 and char not in "\n\r\t" for char in value
    ):
        raise ValueError(f"{field} exceeds its safe character policy")
    return value


def _positive_int(value: object, field: str, *, maximum: int) -> int:
    if type(value) is not int or not 0 < value <= maximum:
        raise ValueError(f"{field} must be an integer in 1..{maximum}")
    return value


def _bytes(value: str) -> int:
    return len(value.encode("utf-8"))


def _bounded_output(stdout: object, stderr: object, maximum: int) -> tuple[str, str]:
    if type(stdout) is not str or type(stderr) is not str:
        raise ValueError("stdout and stderr must be built-in strings")
    if (
        "\x00" in stdout
        or "\x00" in stderr
        or any(ord(char) < 32 and char not in "\n\r\t" for char in stdout)
        or any(ord(char) < 32 and char not in "\n\r\t" for char in stderr)
        or _bytes(stdout) + _bytes(stderr) > maximum
    ):
        raise ValueError("output exceeds its bounded neutral transport")
    return stdout, stderr


def _version_components(value: str) -> tuple[int, ...] | None:
    match = _VERSION_RE.fullmatch(value.strip())
    if match is None:
        return None
    components = tuple(int(item) for item in value.strip().split("."))
    normalized = components
    while len(normalized) > 1 and normalized[-1] == 0:
        normalized = normalized[:-1]
    return normalized


def _environment(value: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    source: Mapping[str, str] = (
        {
            name: os.environ[name]
            for name in _SAFE_ENVIRONMENT_NAMES
            if name in os.environ
        }
        if value is None
        else value
    )
    if len(source) > len(_SAFE_ENVIRONMENT_NAMES):
        raise ValueError("environment contains too many entries")
    checked: list[tuple[str, str]] = []
    for name, item in source.items():
        if (
            type(name) is not str
            or name not in _SAFE_ENVIRONMENT_NAMES
            or type(item) is not str
            or not item
            or len(item) > _MAX_ENVIRONMENT_VALUE_CHARS
            or "\x00" in item
        ):
            raise ValueError("environment contains an unsupported entry")
        checked.append((name, item))
    return tuple(sorted(checked))


@dataclass(frozen=True, slots=True)
class KernelCommand:
    """One explicit argv-only request to a new WolframScript process."""

    argv: tuple[str, ...]
    purpose: str
    timeout_seconds: int
    max_output_bytes: int
    clean_kernel: bool
    environment: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if (
            type(self.argv) is not tuple
            or len(self.argv) != 3
            or self.argv[1] != "-code"
            or any(
                type(part) is not str or not part or "\x00" in part
                for part in self.argv
            )
        ):
            raise ValueError("argv must be one bounded executable/-code/code tuple")
        _text(self.argv[0], "executable", maximum=_MAX_EXECUTABLE_CHARS)
        _text(self.argv[2], "code", maximum=_MAX_SCRIPT_BYTES)
        if _bytes(self.argv[2]) > _MAX_SCRIPT_BYTES:
            raise ValueError("code exceeds its UTF-8 byte bound")
        if type(self.purpose) is not str or self.purpose not in {"version", "execute"}:
            raise ValueError("purpose must be version or execute")
        if self.purpose == "version" and self.argv[2] != _VERSION_CODE:
            raise ValueError("version commands must use the canonical probe code")
        _positive_int(
            self.timeout_seconds, "timeout_seconds", maximum=_MAX_TIMEOUT_SECONDS
        )
        _positive_int(
            self.max_output_bytes,
            "max_output_bytes",
            maximum=_MAX_OUTPUT_BYTES,
        )
        if type(self.clean_kernel) is not bool or not self.clean_kernel:
            raise ValueError("only clean-kernel commands are supported")
        if type(self.environment) is not tuple or any(
            type(item) is not tuple or len(item) != 2 for item in self.environment
        ):
            raise ValueError("environment must be an immutable tuple")
        if len({name for name, _ in self.environment}) != len(self.environment):
            raise ValueError("environment must not repeat a name")
        if _environment(dict(self.environment)) != self.environment:
            raise ValueError("environment must be a canonical safe snapshot")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("KernelCommand cannot be subclassed")


@dataclass(frozen=True, slots=True, init=False)
class KernelProcess:
    """Bounded, backend-neutral completion data returned by a runner."""

    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    output_limited: bool

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("KernelProcess is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("KernelProcess cannot be subclassed")

    @classmethod
    def create(
        cls,
        *,
        exit_code: int | None,
        stdout: str,
        stderr: str,
        timed_out: bool = False,
        output_limited: bool = False,
        max_output_bytes: int = _MAX_OUTPUT_BYTES,
    ) -> KernelProcess:
        if cls is not KernelProcess:
            raise TypeError("KernelProcess factory requires the exact class")
        maximum = _positive_int(
            max_output_bytes, "max_output_bytes", maximum=_MAX_OUTPUT_BYTES
        )
        if exit_code is not None and (type(exit_code) is not int):
            raise ValueError("exit_code must be an integer or None")
        if type(timed_out) is not bool or type(output_limited) is not bool:
            raise ValueError("process flags must be built-in booleans")
        checked_stdout, checked_stderr = _bounded_output(stdout, stderr, maximum)
        value = object.__new__(KernelProcess)
        for name, item in (
            ("exit_code", exit_code),
            ("stdout", checked_stdout),
            ("stderr", checked_stderr),
            ("timed_out", timed_out),
            ("output_limited", output_limited),
        ):
            object.__setattr__(value, name, item)
        return value

    def _assert(self, maximum: int) -> None:
        if type(self.exit_code) is not int and self.exit_code is not None:
            raise ValueError("exit_code is invalid")
        if type(self.timed_out) is not bool or type(self.output_limited) is not bool:
            raise ValueError("process flags are invalid")
        _bounded_output(self.stdout, self.stderr, maximum)


class MathematicaRunner(Protocol):
    """Injectable process boundary used by deterministic unit tests."""

    def __call__(self, command: KernelCommand, /) -> KernelProcess: ...


@dataclass(frozen=True, slots=True, init=False)
class MathematicaExecution:
    """Typed result with neutral text and version metadata only."""

    state: MathematicaExecutionState
    version: str | None
    version_source: str | None
    stdout: str
    stderr: str
    exit_code: int | None
    diagnostic: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("MathematicaExecution is adapter-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("MathematicaExecution cannot be subclassed")


def _trim_to_bytes(value: str, maximum: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= maximum:
        return value
    return encoded[:maximum].decode("utf-8", errors="ignore")


def _plain_output(value: str) -> str:
    """Remove terminal/control bytes before a subprocess result becomes public."""
    return "".join(char for char in value if ord(char) >= 32 or char in "\n\r\t")


def _process_result(
    state: MathematicaExecutionState,
    *,
    version: str | None = None,
    process: KernelProcess | None = None,
    diagnostic: str,
) -> MathematicaExecution:
    if process is None:
        stdout, stderr, exit_code = "", "", None
    else:
        stdout, stderr, exit_code = process.stdout, process.stderr, process.exit_code
    value = object.__new__(MathematicaExecution)
    for name, item in (
        ("state", state),
        ("version", version),
        ("version_source", "pre_payload_clean_probe" if version else None),
        ("stdout", stdout),
        ("stderr", stderr),
        ("exit_code", exit_code),
        ("diagnostic", diagnostic),
    ):
        object.__setattr__(value, name, item)
    return value


def _read_process(command: KernelCommand) -> KernelProcess:
    """Run argv without a shell, streaming both pipes until a hard bound.

    Two readers avoid a stdout/stderr pipe deadlock.  Either an elapsed timeout
    or aggregate output overflow terminates the child before its neutral record
    is returned.  This function is intentionally internal; callers receive no
    ``Popen`` object.
    """

    process = subprocess.Popen(
        command.argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        env=dict(command.environment),
    )
    assert process.stdout is not None
    assert process.stderr is not None
    captured_stdout = bytearray()
    captured_stderr = bytearray()
    output_limited = Event()
    reader_failed = Event()
    reader_errors: list[str] = []
    lock = Lock()

    def drain(readable: IO[bytes], target: bytearray) -> None:
        while True:
            try:
                chunk = readable.read(8192)
            except (OSError, ValueError) as error:
                with lock:
                    reader_errors.append(type(error).__name__)
                reader_failed.set()
                return
            if not chunk:
                return
            with lock:
                remaining = (
                    command.max_output_bytes
                    - len(captured_stdout)
                    - len(captured_stderr)
                )
                if remaining > 0:
                    target.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    output_limited.set()
            if output_limited.is_set():
                with suppress(OSError):
                    process.terminate()
                return

    stdout_thread = Thread(
        target=drain,
        args=(process.stdout, captured_stdout),
        daemon=True,
    )
    stderr_thread = Thread(
        target=drain,
        args=(process.stderr, captured_stderr),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    deadline = time.monotonic() + command.timeout_seconds
    timed_out = False
    while (
        process.poll() is None
        and not output_limited.is_set()
        and not reader_failed.is_set()
    ):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            with suppress(OSError):
                process.terminate()
            break
        output_limited.wait(min(remaining, 0.02))
    if output_limited.is_set() or reader_failed.is_set():
        with suppress(OSError):
            process.terminate()
    cleanup_error: OSError | None = None
    cleanup_cause: subprocess.TimeoutExpired | None = None
    try:
        exit_code = process.wait(timeout=_CLEANUP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        with suppress(OSError):
            process.kill()
        try:
            exit_code = process.wait(timeout=_CLEANUP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as error:
            cleanup_error = OSError("kernel process did not stop after bounded cleanup")
            cleanup_cause = error
            exit_code = None
    if cleanup_error is not None:
        with suppress(OSError):
            process.stdout.close()
        with suppress(OSError):
            process.stderr.close()
    stdout_thread.join(timeout=_CLEANUP_TIMEOUT_SECONDS)
    stderr_thread.join(timeout=_CLEANUP_TIMEOUT_SECONDS)
    if stdout_thread.is_alive() or stderr_thread.is_alive():
        output_limited.set()
        with suppress(OSError):
            process.stdout.close()
        with suppress(OSError):
            process.stderr.close()
        stdout_thread.join(timeout=_CLEANUP_TIMEOUT_SECONDS)
        stderr_thread.join(timeout=_CLEANUP_TIMEOUT_SECONDS)
    if stdout_thread.is_alive() or stderr_thread.is_alive():
        raise OSError("kernel output readers did not stop after bounded cleanup")
    if cleanup_error is not None:
        raise cleanup_error from cleanup_cause
    if reader_errors:
        raise OSError("kernel output reader failed: " + reader_errors[0])
    stdout = _trim_to_bytes(
        _plain_output(captured_stdout.decode("utf-8", errors="replace")),
        command.max_output_bytes,
    )
    remaining = command.max_output_bytes - _bytes(stdout)
    stderr = _trim_to_bytes(
        _plain_output(captured_stderr.decode("utf-8", errors="replace")),
        remaining,
    )
    return KernelProcess.create(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        output_limited=output_limited.is_set(),
        max_output_bytes=command.max_output_bytes,
    )


@dataclass(frozen=True, slots=True, init=False)
class MathematicaAdapter:
    """Fresh-process optional adapter, intentionally limited to execution."""

    executable: str
    required_version: str | None
    timeout_seconds: int
    max_script_bytes: int
    max_output_bytes: int
    environment: tuple[tuple[str, str], ...]
    runner: MathematicaRunner

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("MathematicaAdapter is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("MathematicaAdapter cannot be subclassed")

    @classmethod
    def create(
        cls,
        *,
        executable: str,
        runner: MathematicaRunner | None = None,
        required_version: str | None = None,
        timeout_seconds: int = 180,
        max_script_bytes: int = 65_536,
        max_output_bytes: int = 65_536,
        environment: Mapping[str, str] | None = None,
    ) -> MathematicaAdapter:
        if cls is not MathematicaAdapter:
            raise TypeError("MathematicaAdapter factory requires the exact class")
        checked_executable = _text(
            executable, "executable", maximum=_MAX_EXECUTABLE_CHARS
        )
        checked_required = None
        if required_version is not None:
            checked_required = _text(required_version, "required_version", maximum=64)
            if _version_components(checked_required) is None:
                raise ValueError("required_version must be a numeric version")
        checked_runner: MathematicaRunner = _read_process if runner is None else runner
        if not callable(checked_runner):
            raise ValueError("runner must be callable")
        value = object.__new__(MathematicaAdapter)
        for name, item in (
            ("executable", checked_executable),
            ("required_version", checked_required),
            (
                "timeout_seconds",
                _positive_int(
                    timeout_seconds,
                    "timeout_seconds",
                    maximum=_MAX_TIMEOUT_SECONDS,
                ),
            ),
            (
                "max_script_bytes",
                _positive_int(
                    max_script_bytes,
                    "max_script_bytes",
                    maximum=_MAX_SCRIPT_BYTES,
                ),
            ),
            (
                "max_output_bytes",
                _positive_int(
                    max_output_bytes,
                    "max_output_bytes",
                    maximum=_MAX_OUTPUT_BYTES,
                ),
            ),
            ("environment", _environment(environment)),
            ("runner", checked_runner),
        ):
            object.__setattr__(value, name, item)
        return value

    def _command(self, *, purpose: str, code: str) -> KernelCommand:
        return KernelCommand(
            argv=(self.executable, "-code", code),
            purpose=purpose,
            timeout_seconds=self.timeout_seconds,
            max_output_bytes=self.max_output_bytes,
            clean_kernel=True,
            environment=self.environment,
        )

    def _invoke(self, command: KernelCommand) -> MathematicaExecution | KernelProcess:
        try:
            process = self.runner(command)
        except FileNotFoundError:
            return _process_result(
                MathematicaExecutionState.EXECUTABLE_UNAVAILABLE,
                diagnostic="FileNotFoundError: optional executable is unavailable",
            )
        except OSError as error:
            return _process_result(
                MathematicaExecutionState.LAUNCH_FAILED,
                diagnostic=f"{type(error).__name__}: optional process could not launch",
            )
        except Exception as error:
            return _process_result(
                MathematicaExecutionState.LAUNCH_FAILED,
                diagnostic=f"{type(error).__name__}: runner failed before completion",
            )
        if type(process) is not KernelProcess:
            return _process_result(
                MathematicaExecutionState.PROTOCOL_ERROR,
                diagnostic="runner returned a non-neutral process record",
            )
        try:
            process._assert(self.max_output_bytes)
        except (AttributeError, TypeError, ValueError):
            return _process_result(
                MathematicaExecutionState.PROTOCOL_ERROR,
                diagnostic="runner returned an invalid neutral process record",
            )
        if process.output_limited:
            return _process_result(
                MathematicaExecutionState.OUTPUT_LIMIT_EXCEEDED,
                process=process,
                diagnostic="kernel output exceeded the declared bound",
            )
        if process.timed_out:
            return _process_result(
                MathematicaExecutionState.TIMED_OUT,
                process=process,
                diagnostic="kernel exceeded the declared timeout",
            )
        return process

    def execute(self, script: str) -> MathematicaExecution:
        """Probe the fresh kernel version, then run one bounded script in another.

        A version probe failure is deliberately distinct from a missing program;
        a caller can therefore decide whether to install, retarget, or defer the
        later runtime audit.  The payload is not run unless the probe succeeds.
        """

        checked_script = _text(script, "script", maximum=self.max_script_bytes)
        if _bytes(checked_script) > self.max_script_bytes:
            raise ValueError("script exceeds max_script_bytes")
        probe = self._invoke(self._command(purpose="version", code=_VERSION_CODE))
        if type(probe) is MathematicaExecution:
            return probe
        probe = cast(KernelProcess, probe)
        if probe.exit_code != 0:
            return _process_result(
                MathematicaExecutionState.VERSION_UNAVAILABLE,
                process=probe,
                diagnostic="kernel version probe exited unsuccessfully",
            )
        version = probe.stdout.strip()
        components = _version_components(version)
        if components is None:
            return _process_result(
                MathematicaExecutionState.VERSION_UNAVAILABLE,
                process=probe,
                diagnostic="kernel version probe did not produce one numeric version",
            )
        if self.required_version is not None and components != _version_components(
            self.required_version
        ):
            return _process_result(
                MathematicaExecutionState.VERSION_INCOMPATIBLE,
                version=version,
                process=probe,
                diagnostic="kernel version does not match the declared requirement",
            )
        payload = self._invoke(self._command(purpose="execute", code=checked_script))
        if type(payload) is MathematicaExecution:
            return _process_result(
                payload.state,
                version=version,
                process=KernelProcess.create(
                    exit_code=payload.exit_code,
                    stdout=payload.stdout,
                    stderr=payload.stderr,
                    max_output_bytes=self.max_output_bytes,
                ),
                diagnostic=payload.diagnostic,
            )
        payload = cast(KernelProcess, payload)
        if payload.exit_code != 0:
            return _process_result(
                MathematicaExecutionState.PROCESS_FAILED,
                version=version,
                process=payload,
                diagnostic="kernel payload exited unsuccessfully",
            )
        return _process_result(
            MathematicaExecutionState.SUCCEEDED,
            version=version,
            process=payload,
            diagnostic="kernel payload completed in a fresh process",
        )
