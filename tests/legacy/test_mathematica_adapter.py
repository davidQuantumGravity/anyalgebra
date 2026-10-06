"""Tests for the isolated, optional Mathematica process boundary (V00-084)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError
import subprocess
from threading import Event
import threading
from typing import Any

import pytest

import anyalgebra.legacy.mathematica as mathematica
from anyalgebra.legacy.mathematica import (
    KernelCommand,
    KernelProcess,
    MathematicaAdapter,
    MathematicaExecutionState,
)


Runner = Callable[[KernelCommand], KernelProcess]


class _FakePipe:
    def __init__(self, chunks: list[object] | None = None) -> None:
        self.chunks = [] if chunks is None else list(chunks)
        self.close_calls = 0

    def read(self, size: int) -> bytes:
        del size
        if not self.chunks:
            return b""
        item = self.chunks.pop(0)
        if isinstance(item, BaseException):
            raise item
        assert type(item) is bytes
        return item

    def close(self) -> None:
        self.close_calls += 1


class _BlockingPipe(_FakePipe):
    def __init__(self) -> None:
        super().__init__()
        self.started = Event()
        self.released = Event()

    def read(self, size: int) -> bytes:
        del size
        self.started.set()
        self.released.wait()
        return b""

    def close(self) -> None:
        super().close()
        self.released.set()


class _FakePopen:
    def __init__(
        self,
        *,
        stdout: _FakePipe | None = None,
        stderr: _FakePipe | None = None,
        polls: list[int | None] | None = None,
        waits: list[object] | None = None,
    ) -> None:
        self.stdout = _FakePipe() if stdout is None else stdout
        self.stderr = _FakePipe() if stderr is None else stderr
        self.polls = list[int | None]([0]) if polls is None else list(polls)
        self.waits = list[object]([0]) if waits is None else list(waits)
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_timeouts: list[float] = []

    def poll(self) -> int | None:
        return self.polls.pop(0) if self.polls else 0

    def wait(self, timeout: float) -> int:
        self.wait_timeouts.append(timeout)
        item = self.waits.pop(0) if self.waits else 0
        if isinstance(item, BaseException):
            raise item
        assert type(item) is int
        return item

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1


def _transport_command(
    *, timeout_seconds: int = 1, max_output_bytes: int = 64
) -> KernelCommand:
    return KernelCommand(
        argv=("wolframscript", "-code", "Print[1]"),
        purpose="execute",
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
        clean_kernel=True,
        environment=(("PATH", "test-path"),),
    )


def _adapter(runner: Runner, **kwargs: Any) -> MathematicaAdapter:
    options: dict[str, Any] = {"timeout_seconds": 3}
    options.update(kwargs)
    return MathematicaAdapter.create(
        executable="wolframscript",
        runner=runner,
        **options,
    )


def _success_runner(commands: list[KernelCommand]) -> Runner:
    def run(command: KernelCommand) -> KernelProcess:
        commands.append(command)
        if command.purpose == "version":
            return KernelProcess.create(exit_code=0, stdout="14.1\n", stderr="")
        return KernelProcess.create(exit_code=0, stdout="answer\n", stderr="")

    return run


def test_clean_kernel_command_and_version_metadata_are_deterministic() -> None:
    commands: list[KernelCommand] = []
    result = _adapter(_success_runner(commands)).execute("2 + 2")

    assert result.state is MathematicaExecutionState.SUCCEEDED
    assert result.version == "14.1"
    assert result.version_source == "pre_payload_clean_probe"
    assert result.stdout == "answer\n"
    assert result.stderr == ""
    assert result.exit_code == 0
    assert [command.purpose for command in commands] == ["version", "execute"]
    assert commands[0].argv == (
        "wolframscript",
        "-code",
        "With[{v = ToString[$VersionNumber, InputForm]}, "
        'If[StringEndsQ[v, "."], v <> "0", v]]',
    )
    assert "Print" not in commands[0].argv[2]
    assert commands[1].argv == ("wolframscript", "-code", "2 + 2")
    assert all(command.clean_kernel for command in commands)
    assert all(command.timeout_seconds == 3 for command in commands)


def test_absent_executable_is_a_typed_result() -> None:
    def missing(command: KernelCommand) -> KernelProcess:
        del command
        raise FileNotFoundError("not installed")

    result = _adapter(missing).execute("1")

    assert result.state is MathematicaExecutionState.EXECUTABLE_UNAVAILABLE
    assert result.version is None
    assert result.stdout == ""
    assert result.stderr == ""
    assert result.exit_code is None
    assert "FileNotFoundError" in result.diagnostic


def test_unknown_and_incompatible_versions_do_not_execute_payload() -> None:
    unknown_commands: list[KernelCommand] = []

    def unknown(command: KernelCommand) -> KernelProcess:
        unknown_commands.append(command)
        return KernelProcess.create(exit_code=0, stdout="not-a-version\n", stderr="")

    unknown_result = _adapter(unknown).execute("dangerous[]")
    assert unknown_result.state is MathematicaExecutionState.VERSION_UNAVAILABLE
    assert [command.purpose for command in unknown_commands] == ["version"]

    incompatible_commands: list[KernelCommand] = []
    incompatible_result = _adapter(
        _success_runner(incompatible_commands), required_version="13.3"
    ).execute("dangerous[]")
    assert incompatible_result.state is MathematicaExecutionState.VERSION_INCOMPATIBLE
    assert incompatible_result.version == "14.1"
    assert [command.purpose for command in incompatible_commands] == ["version"]


def test_wolframscript_top_level_version_output_is_a_valid_probe_fixture() -> None:
    commands: list[KernelCommand] = []

    def wolframscript_style_runner(command: KernelCommand) -> KernelProcess:
        commands.append(command)
        if command.purpose == "version":
            return KernelProcess.create(exit_code=0, stdout="12.0\r\n", stderr="")
        return KernelProcess.create(exit_code=0, stdout="4\r\n", stderr="")

    result = _adapter(wolframscript_style_runner).execute("2 + 2")

    assert result.state is MathematicaExecutionState.SUCCEEDED
    assert result.version == "12.0"
    assert result.stdout == "4\r\n"
    assert [command.purpose for command in commands] == ["version", "execute"]


@pytest.mark.parametrize(
    ("process", "expected"),
    [
        (
            KernelProcess.create(exit_code=None, stdout="", stderr="", timed_out=True),
            MathematicaExecutionState.TIMED_OUT,
        ),
        (
            KernelProcess.create(exit_code=2, stdout="", stderr="bad"),
            MathematicaExecutionState.PROCESS_FAILED,
        ),
        (
            KernelProcess.create(
                exit_code=None, stdout="partial", stderr="", output_limited=True
            ),
            MathematicaExecutionState.OUTPUT_LIMIT_EXCEEDED,
        ),
    ],
)
def test_payload_process_failures_are_typed(
    process: KernelProcess, expected: MathematicaExecutionState
) -> None:
    calls = 0

    def runner(command: KernelCommand) -> KernelProcess:
        nonlocal calls
        calls += 1
        if command.purpose == "version":
            return KernelProcess.create(exit_code=0, stdout="14.1", stderr="")
        return process

    result = _adapter(runner).execute("x")

    assert calls == 2
    assert result.state is expected
    assert result.version == "14.1"


def test_bad_runner_protocol_and_launch_failure_are_typed() -> None:
    def malformed(command: KernelCommand) -> KernelProcess:
        del command
        return object()  # type: ignore[return-value]

    malformed_result = _adapter(malformed).execute("x")
    assert malformed_result.state is MathematicaExecutionState.PROTOCOL_ERROR

    def broken(command: KernelCommand) -> KernelProcess:
        del command
        raise OSError("process creation failed")

    broken_result = _adapter(broken).execute("x")
    assert broken_result.state is MathematicaExecutionState.LAUNCH_FAILED
    assert "OSError" in broken_result.diagnostic


def test_bounds_and_control_flow_are_not_silently_erased() -> None:
    with pytest.raises(ValueError, match="script"):
        _adapter(_success_runner([]), max_script_bytes=3).execute("1234")

    def interrupted(command: KernelCommand) -> KernelProcess:
        del command
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _adapter(interrupted).execute("x")

    with pytest.raises(ValueError, match="output"):
        KernelProcess.create(
            exit_code=0, stdout="toolong", stderr="", max_output_bytes=3
        )


def test_system_and_generator_exit_propagate_from_an_injected_runner() -> None:
    def system_exit(command: KernelCommand) -> KernelProcess:
        del command
        raise SystemExit(7)

    with pytest.raises(SystemExit):
        _adapter(system_exit).execute("x")

    def generator_exit(command: KernelCommand) -> KernelProcess:
        del command
        raise GeneratorExit

    with pytest.raises(GeneratorExit):
        _adapter(generator_exit).execute("x")


def test_factories_reject_bool_as_int_and_combined_output_overflow() -> None:
    with pytest.raises(ValueError, match="exit_code"):
        KernelProcess.create(exit_code=True, stdout="", stderr="")
    with pytest.raises(ValueError, match="timeout_seconds"):
        _adapter(_success_runner([]), timeout_seconds=True)
    with pytest.raises(ValueError, match="output"):
        KernelProcess.create(exit_code=0, stdout="abc", stderr="d", max_output_bytes=3)
    with pytest.raises(ValueError, match="output"):
        KernelProcess.create(exit_code=0, stdout="\x1b[2J", stderr="")


def test_script_bound_is_measured_in_utf8_bytes_not_characters() -> None:
    two_non_ascii_characters = chr(0xE9) * 2
    assert len(two_non_ascii_characters) == 2
    with pytest.raises(ValueError, match="script"):
        _adapter(_success_runner([]), max_script_bytes=3).execute(
            two_non_ascii_characters
        )


def test_utf8_script_limit_and_ambiguous_version_output_stop_before_payload() -> None:
    with pytest.raises(ValueError, match="script"):
        _adapter(_success_runner([]), max_script_bytes=3).execute(chr(0xE9) * 2)

    commands: list[KernelCommand] = []

    def ambiguous(command: KernelCommand) -> KernelProcess:
        commands.append(command)
        return KernelProcess.create(exit_code=0, stdout="14.0\n14.1\n", stderr="")

    result = _adapter(ambiguous).execute("payload[]")
    assert result.state is MathematicaExecutionState.VERSION_UNAVAILABLE
    assert [command.purpose for command in commands] == ["version"]


def test_command_process_and_result_records_are_immutable_and_sealed() -> None:
    commands: list[KernelCommand] = []
    result = _adapter(_success_runner(commands)).execute("1")
    process = KernelProcess.create(exit_code=0, stdout="", stderr="")

    def overwrite(record: object, attribute: str) -> None:
        setattr(record, attribute, "changed")

    with pytest.raises(FrozenInstanceError):
        overwrite(commands[0], "purpose")
    with pytest.raises(FrozenInstanceError):
        overwrite(process, "stdout")
    with pytest.raises(FrozenInstanceError):
        overwrite(result, "stdout")
    with pytest.raises(TypeError, match="factory-owned"):
        KernelProcess(exit_code=0, stdout="", stderr="")
    with pytest.raises(TypeError, match="adapter-owned"):
        type(result)(
            state=MathematicaExecutionState.SUCCEEDED,
            version="14.1",
            version_source="forged",
            stdout="",
            stderr="",
            exit_code=0,
            diagnostic="forged",
        )


def test_forged_process_record_is_rejected_at_the_runner_boundary() -> None:
    forged = object.__new__(KernelProcess)
    object.__setattr__(forged, "exit_code", True)
    object.__setattr__(forged, "stdout", "")
    object.__setattr__(forged, "stderr", "")
    object.__setattr__(forged, "timed_out", False)
    object.__setattr__(forged, "output_limited", False)

    def runner(command: KernelCommand) -> KernelProcess:
        del command
        return forged

    result = _adapter(runner).execute("x")
    assert result.state is MathematicaExecutionState.PROTOCOL_ERROR


def test_command_rejects_noncanonical_version_code_and_environment_shape() -> None:
    with pytest.raises(ValueError, match="canonical probe"):
        KernelCommand(
            argv=("wolframscript", "-code", "arbitrary[]"),
            purpose="version",
            timeout_seconds=1,
            max_output_bytes=1,
            clean_kernel=True,
            environment=(),
        )
    with pytest.raises(ValueError, match="environment"):
        KernelCommand(
            argv=("wolframscript", "-code", "x"),
            purpose="execute",
            timeout_seconds=1,
            max_output_bytes=1,
            clean_kernel=True,
            environment=(("TEMP",),),  # type: ignore[arg-type]
        )


def test_default_transport_uses_bounded_argv_only_popen_and_drains_both_pipes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakePopen(
        stdout=_FakePipe([b"out\x1b[31m\n"]),
        stderr=_FakePipe([b"err\n"]),
    )
    observed: dict[str, object] = {}

    def popen(argv: tuple[str, ...], **kwargs: object) -> _FakePopen:
        observed["argv"] = argv
        observed.update(kwargs)
        return process

    monkeypatch.setattr("anyalgebra.legacy.mathematica.subprocess.Popen", popen)
    result = mathematica._read_process(_transport_command())

    assert result.exit_code == 0
    assert result.stdout == "out[31m\n"
    assert result.stderr == "err\n"
    assert not result.timed_out
    assert not result.output_limited
    assert observed == {
        "argv": ("wolframscript", "-code", "Print[1]"),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "shell": False,
        "env": {"PATH": "test-path"},
    }
    assert process.wait_timeouts == [mathematica._CLEANUP_TIMEOUT_SECONDS]


def test_default_transport_caps_aggregate_output_and_terminates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakePopen(stdout=_FakePipe([b"abcdef"]), polls=[None])
    monkeypatch.setattr(
        "anyalgebra.legacy.mathematica.subprocess.Popen",
        lambda *args, **kwargs: process,
    )

    result = mathematica._read_process(_transport_command(max_output_bytes=4))

    assert result.output_limited
    assert not result.timed_out
    assert result.stdout == "abcd"
    assert result.stderr == ""
    assert process.terminate_calls >= 1


def test_default_transport_times_out_then_escalates_to_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakePopen(
        polls=[None],
        waits=[subprocess.TimeoutExpired("wolframscript", 1), -9],
    )
    moments = iter((0.0, 2.0))
    monkeypatch.setattr(
        "anyalgebra.legacy.mathematica.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(
        "anyalgebra.legacy.mathematica.time.monotonic", lambda: next(moments)
    )

    result = mathematica._read_process(_transport_command(timeout_seconds=1))

    assert result.timed_out
    assert result.exit_code == -9
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.wait_timeouts == [
        mathematica._CLEANUP_TIMEOUT_SECONDS,
        mathematica._CLEANUP_TIMEOUT_SECONDS,
    ]


def test_default_transport_surfaces_reader_failure_without_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakePopen(stdout=_FakePipe([OSError("read failed")]))
    monkeypatch.setattr(
        "anyalgebra.legacy.mathematica.subprocess.Popen",
        lambda *args, **kwargs: process,
    )

    with pytest.raises(OSError, match="reader failed"):
        mathematica._read_process(_transport_command())


def test_default_transport_closes_stalled_reader_with_bounded_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakePopen(stdout=_BlockingPipe())
    monkeypatch.setattr(mathematica, "_CLEANUP_TIMEOUT_SECONDS", 0.001)
    monkeypatch.setattr(
        "anyalgebra.legacy.mathematica.subprocess.Popen",
        lambda *args, **kwargs: process,
    )

    result = mathematica._read_process(_transport_command())

    assert result.output_limited
    assert process.stdout.close_calls == 1
    assert process.stderr.close_calls == 1


def test_default_transport_rejects_unstoppable_reader_after_bounded_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocking = _BlockingPipe()

    def close_without_release() -> None:
        blocking.close_calls += 1

    monkeypatch.setattr(blocking, "close", close_without_release)
    process = _FakePopen(stdout=blocking)
    monkeypatch.setattr(mathematica, "_CLEANUP_TIMEOUT_SECONDS", 0.001)
    monkeypatch.setattr(
        "anyalgebra.legacy.mathematica.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    try:
        with pytest.raises(OSError, match="readers did not stop"):
            mathematica._read_process(_transport_command())
    finally:
        blocking.released.set()


def test_default_transport_reports_a_second_bounded_wait_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakePopen(
        waits=[
            subprocess.TimeoutExpired("wolframscript", 1),
            subprocess.TimeoutExpired("wolframscript", 1),
        ]
    )
    monkeypatch.setattr(
        "anyalgebra.legacy.mathematica.subprocess.Popen",
        lambda *args, **kwargs: process,
    )

    with pytest.raises(OSError, match="did not stop"):
        mathematica._read_process(_transport_command())
    assert process.kill_calls == 1


def test_second_wait_timeout_closes_and_joins_blocked_readers_before_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocking = _BlockingPipe()
    process = _FakePopen(
        stdout=blocking,
        waits=[
            subprocess.TimeoutExpired("wolframscript", 1),
            subprocess.TimeoutExpired("wolframscript", 1),
        ],
    )
    before = {
        thread.ident for thread in threading.enumerate() if thread.ident is not None
    }
    monkeypatch.setattr(mathematica, "_CLEANUP_TIMEOUT_SECONDS", 0.001)
    monkeypatch.setattr(
        "anyalgebra.legacy.mathematica.subprocess.Popen",
        lambda *args, **kwargs: process,
    )

    with pytest.raises(OSError, match="kernel process did not stop"):
        mathematica._read_process(_transport_command())

    after = {
        thread.ident for thread in threading.enumerate() if thread.ident is not None
    }
    assert after <= before
    assert process.stdout.close_calls == 1
    assert process.stderr.close_calls == 1
    assert process.wait_timeouts == [0.001, 0.001]


def test_default_transport_waits_once_for_an_exiting_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakePopen(polls=[None, 0])
    monkeypatch.setattr(
        "anyalgebra.legacy.mathematica.subprocess.Popen",
        lambda *args, **kwargs: process,
    )

    result = mathematica._read_process(_transport_command())

    assert result.exit_code == 0


def test_default_transport_handles_decode_expansion_at_the_byte_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakePopen(stdout=_FakePipe([b"\xff\xff\xff\xff"]))
    monkeypatch.setattr(
        "anyalgebra.legacy.mathematica.subprocess.Popen",
        lambda *args, **kwargs: process,
    )

    result = mathematica._read_process(_transport_command(max_output_bytes=4))

    assert result.stdout == "�"
    assert not result.output_limited


def test_constructor_and_adapter_failure_boundaries_are_typed() -> None:
    with pytest.raises(ValueError, match="argv"):
        KernelCommand(
            argv=("wolframscript", "-file", "x"),
            purpose="execute",
            timeout_seconds=1,
            max_output_bytes=1,
            clean_kernel=True,
            environment=(),
        )
    with pytest.raises(ValueError, match="purpose"):
        KernelCommand(
            argv=("wolframscript", "-code", "x"),
            purpose="other",
            timeout_seconds=1,
            max_output_bytes=1,
            clean_kernel=True,
            environment=(),
        )
    with pytest.raises(ValueError, match="clean-kernel"):
        KernelCommand(
            argv=("wolframscript", "-code", "x"),
            purpose="execute",
            timeout_seconds=1,
            max_output_bytes=1,
            clean_kernel=False,
            environment=(),
        )
    with pytest.raises(ValueError, match="repeat"):
        KernelCommand(
            argv=("wolframscript", "-code", "x"),
            purpose="execute",
            timeout_seconds=1,
            max_output_bytes=1,
            clean_kernel=True,
            environment=(("PATH", "one"), ("PATH", "two")),
        )
    with pytest.raises(ValueError, match="canonical"):
        KernelCommand(
            argv=("wolframscript", "-code", "x"),
            purpose="execute",
            timeout_seconds=1,
            max_output_bytes=1,
            clean_kernel=True,
            environment=(("TEMP", "one"), ("PATH", "two")),
        )
    with pytest.raises(ValueError, match="flags"):
        KernelProcess.create(exit_code=0, stdout="", stderr="", timed_out=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="stdout"):
        KernelProcess.create(exit_code=0, stdout=object(), stderr="")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="numeric"):
        _adapter(_success_runner([]), required_version="14.")
    with pytest.raises(ValueError, match="environment"):
        _adapter(_success_runner([]), environment={"NOT_ALLOWED": "x"})
    system_drive_adapter = _adapter(
        _success_runner([]), environment={"SystemDrive": "C:"}
    )
    assert system_drive_adapter.environment == (("SystemDrive", "C:"),)
    with pytest.raises(ValueError, match="too many"):
        _adapter(
            _success_runner([]),
            environment={
                "PATH": "1",
                "TEMP": "1",
                "TMP": "1",
                "WINDIR": "1",
                "SystemRoot": "1",
                "COMSPEC": "1",
                "EXTRA": "1",
                "EXTRA2": "1",
            },
        )
    with pytest.raises(ValueError, match="callable"):
        _adapter(object())  # type: ignore[arg-type]


def test_sealed_classes_and_unclassified_runner_exception_are_not_silent() -> None:
    commands: list[KernelCommand] = []
    execution_type = type(_adapter(_success_runner(commands)).execute("x"))
    for record_type in (
        KernelCommand,
        KernelProcess,
        MathematicaAdapter,
        execution_type,
    ):
        with pytest.raises(TypeError, match="cannot be subclassed"):
            type("ForbiddenSubclass", (record_type,), {})
    with pytest.raises(TypeError, match="factory-owned"):
        MathematicaAdapter()

    def unexpected(command: KernelCommand) -> KernelProcess:
        del command
        raise RuntimeError("unexpected")

    result = _adapter(unexpected).execute("x")
    assert result.state is MathematicaExecutionState.LAUNCH_FAILED


def test_probe_nonzero_and_forged_flags_do_not_become_payload_success() -> None:
    def nonzero_probe(command: KernelCommand) -> KernelProcess:
        del command
        return KernelProcess.create(exit_code=1, stdout="", stderr="error")

    result = _adapter(nonzero_probe).execute("payload[]")
    assert result.state is MathematicaExecutionState.VERSION_UNAVAILABLE

    forged = object.__new__(KernelProcess)
    object.__setattr__(forged, "exit_code", 0)
    object.__setattr__(forged, "stdout", "")
    object.__setattr__(forged, "stderr", "")
    object.__setattr__(forged, "timed_out", "false")
    object.__setattr__(forged, "output_limited", False)

    def forged_runner(command: KernelCommand) -> KernelProcess:
        del command
        return forged

    forged_result = _adapter(forged_runner).execute("payload[]")
    assert forged_result.state is MathematicaExecutionState.PROTOCOL_ERROR


def test_remaining_text_utf8_and_exact_factory_guards() -> None:
    with pytest.raises(ValueError, match="nonempty string"):
        MathematicaAdapter.create(executable="")

    oversized_utf8 = "é" * (mathematica._MAX_SCRIPT_BYTES // 2 + 1)
    with pytest.raises(ValueError, match="UTF-8 byte bound"):
        KernelCommand(
            argv=("wolframscript", "-code", oversized_utf8),
            purpose="execute",
            timeout_seconds=1,
            max_output_bytes=64,
            clean_kernel=True,
            environment=(),
        )

    with pytest.raises(TypeError, match="exact class"):
        KernelProcess.create.__func__(  # type: ignore[attr-defined]
            object, exit_code=0, stdout="", stderr=""
        )
    with pytest.raises(TypeError, match="exact class"):
        MathematicaAdapter.create.__func__(  # type: ignore[attr-defined]
            object, executable="wolframscript"
        )


def test_transport_handles_a_pipe_after_the_shared_output_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakePopen(
        stdout=_FakePipe([b"abcd"]),
        stderr=_FakePipe([b"efgh"]),
        polls=[0],
        waits=[0],
    )
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    result = mathematica._read_process(_transport_command(max_output_bytes=4))
    assert result.output_limited
    assert len(result.stdout.encode()) + len(result.stderr.encode()) == 4
