"""Adversarial contracts for V00-079's neutral backend declarations."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, cast, get_args

import pytest

from anyalgebra.backends import base
from anyalgebra.backends.base import (
    MAX_DECLARATIONS,
    MAX_RESOURCE_LIMIT,
    BackendBoundaryError,
    BackendCapabilities,
    BackendOptions,
    BackendProvenance,
    BackendRequest,
    BackendResult,
    BackendSupported,
    BackendUnsupported,
    CapabilitySpec,
    UnsupportedReason,
    negotiate,
)
from anyalgebra.core.parents import SemanticHash
from anyalgebra.structures.outcomes import EvaluationOutcome


def h(char: str) -> SemanticHash:
    return SemanticHash("sha256", hashlib.sha256(char.encode("utf-8")).hexdigest())


def spec(
    *,
    operation: str = "rank",
    domain: str = "QQ",
    algorithm: str = "direct",
    exact: bool = True,
    preconditions: object = (),
    conventions: object = (h("c"),),
) -> CapabilitySpec:
    return CapabilitySpec.create(
        operation=operation,
        domain=domain,
        algorithm=algorithm,
        exact=exact,
        precondition_hashes=preconditions,
        convention_hashes=conventions,
    )


def caps(**changes: object) -> BackendCapabilities:
    values: dict[str, object] = {
        "name": "reference",
        "version": "0.0",
        "specifications": (spec(), spec(operation="solve", algorithm="rref")),
        "convention_hashes": (h("c"),),
        "limits": (("cases", 2),),
    }
    values.update(changes)
    return BackendCapabilities.create(**values)


def request(**changes: object) -> BackendRequest:
    values: dict[str, object] = {
        "operation": "rank",
        "domain": "QQ",
        "algorithm": "direct",
        "input_hashes": (h("a"),),
        "limits": (("cases", 2),),
        "convention_hashes": (h("c"),),
    }
    values.update(changes)
    return BackendRequest.create(**values)


def test_negotiation_matches_one_full_specification_and_binds_metadata() -> None:
    cap, req = caps(), request()
    result = negotiate(cap, req)
    assert type(result) is BackendSupported
    assert result.provenance.specification_hash == spec().semantic_hash
    assert result.provenance.convention_hashes == (h("c"),)
    assert cap.operations == ("rank", "solve")
    assert cap.domains == ("QQ",)
    assert cap.algorithms == ("direct", "rref")
    assert cap.exactness == (("rank", "QQ", "direct"), ("solve", "QQ", "rref"))
    assert BackendUnsupported not in get_args(EvaluationOutcome)
    assert not hasattr(result, "backend")


@pytest.mark.parametrize(
    ("req", "reason"),
    (
        (request(operation="missing"), UnsupportedReason.MISSING_OPERATION),
        (request(domain="ZZ"), UnsupportedReason.MISSING_DOMAIN),
        (request(algorithm="rref"), UnsupportedReason.MISSING_ALGORITHM),
        (request(limits=(("cases", 3),)), UnsupportedReason.RESOURCE_LIMIT),
        (request(convention_hashes=(h("d"),)), UnsupportedReason.CONVENTION),
        (request(precondition_hashes=(h("p"),)), UnsupportedReason.PRECONDITION),
    ),
)
def test_typed_unsupported_reasons(
    req: BackendRequest, reason: UnsupportedReason
) -> None:
    result = negotiate(caps(), req)
    assert type(result) is BackendUnsupported and result.reason is reason


def test_algorithms_are_bound_to_operation_domain_specifications() -> None:
    cap = caps(
        specifications=(
            spec(operation="op1", domain="QQ", algorithm="a"),
            spec(operation="op2", domain="QQ", algorithm="b"),
        )
    )
    first = negotiate(cap, request(operation="op1", algorithm="a"))
    assert type(first) is BackendSupported
    wrong = negotiate(cap, request(operation="op2", algorithm="a"))
    assert (
        type(wrong) is BackendUnsupported
        and wrong.reason is UnsupportedReason.MISSING_ALGORITHM
    )


def test_exactness_and_preconditions_are_spec_scoped() -> None:
    precondition = h("p")
    inexact = caps(specifications=(spec(exact=False),))
    inexact_result = negotiate(inexact, request())
    assert type(inexact_result) is BackendUnsupported
    assert inexact_result.reason is UnsupportedReason.INEXACT_CAPABILITY
    declared = caps(specifications=(spec(preconditions=(precondition,)),))
    assert (
        type(negotiate(declared, request(precondition_hashes=(precondition,))))
        is BackendSupported
    )
    mixed = caps(
        specifications=(
            spec(algorithm="direct", exact=True),
            spec(algorithm="rref", exact=False),
        )
    )
    assert mixed.exactness == (("rank", "QQ", "direct"),)


def test_supported_factory_cannot_falsely_promote_unsupported_request() -> None:
    with pytest.raises(BackendBoundaryError):
        BackendSupported.create(request(algorithm="missing"), caps())


def test_neutral_result_binds_options_provenance_request_and_hash_only() -> None:
    cap, req = caps(), request()
    provenance = BackendProvenance.create(req, cap)
    options = BackendOptions.create(convention_hashes=(h("c"),), limits=(("cases", 2),))
    result = BackendResult.create(
        request=req, provenance=provenance, options=options, value_hash=h("v")
    )
    assert result.value_hash == h("v") and not hasattr(result, "value")
    with pytest.raises(BackendBoundaryError):
        BackendResult.create(
            request=req,
            provenance=provenance,
            options=BackendOptions.create(),
            value_hash=h("v"),
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (("digest", "0" * 63), ("algorithm", "sha1")),
)
def test_corrupted_semantic_hashes_are_revalidated_at_neutral_boundaries(
    field: str, replacement: str
) -> None:
    corrupted = h("a")
    object.__setattr__(corrupted, field, replacement)
    cap, req = caps(), request()
    provenance = BackendProvenance.create(req, cap)
    options = BackendOptions.create(convention_hashes=(h("c"),), limits=(("cases", 2),))

    for construct in (
        lambda: spec(preconditions=(corrupted,)),
        lambda: caps(convention_hashes=(corrupted,)),
        lambda: request(input_hashes=(corrupted,)),
        lambda: BackendOptions.create(convention_hashes=(corrupted,), limits=()),
        lambda: BackendResult.create(
            request=req, provenance=provenance, options=options, value_hash=corrupted
        ),
    ):
        with pytest.raises(BackendBoundaryError):
            construct()

    object.__setattr__(cap.semantic_hash, field, replacement)
    with pytest.raises(BackendBoundaryError):
        BackendProvenance.create(req, cap)


class Broken:
    def __iter__(self) -> Broken:
        return self

    def __next__(self) -> object:
        raise RuntimeError("hostile")


class Interrupted:
    def __iter__(self) -> Interrupted:
        return self

    def __next__(self) -> object:
        raise KeyboardInterrupt


class Exited:
    def __iter__(self) -> Exited:
        return self

    def __next__(self) -> object:
        raise SystemExit(7)


def test_bounded_iterables_malformed_data_and_interrupts() -> None:
    operations = tuple(spec(operation=f"o{n}") for n in range(MAX_DECLARATIONS))
    assert len(caps(specifications=operations).specifications) == MAX_DECLARATIONS
    failures = (
        lambda: caps(specifications=(*operations, spec(operation="extra"))),
        lambda: caps(specifications=Broken()),
        lambda: caps(limits=(("cases", MAX_RESOURCE_LIMIT + 1),)),
        lambda: request(input_hashes="bad"),
        lambda: request(input_hashes=("bad",)),
        lambda: request(limits=(("cases", 1, 2),)),
        lambda: spec(exact=cast(bool, 1)),
    )
    for construct in failures:
        with pytest.raises(BackendBoundaryError):
            construct()
    with pytest.raises(KeyboardInterrupt):
        request(input_hashes=Interrupted())
    with pytest.raises(SystemExit, match="7"):
        request(input_hashes=Exited())
    repeated = request(input_hashes=(h("a"), h("a"), h("b")))
    swapped = request(input_hashes=(h("b"), h("a"), h("a")))
    assert repeated.input_hashes == (h("a"), h("a"), h("b"))
    assert repeated.semantic_hash != swapped.semantic_hash


def test_records_are_sealed_hash_stable_and_tamper_evident() -> None:
    first, second = caps(), caps()
    assert (
        first == second
        and hash(first) == hash(second)
        and "semantic_hash" in repr(first)
    )
    req = request()
    object.__setattr__(req, "algorithm", "tampered")
    assert req != request() and "tampered" not in repr(req)
    with pytest.raises(BackendBoundaryError):
        hash(req)
    supported = cast(BackendSupported, negotiate(caps(), request()))
    object.__setattr__(supported.provenance, "name", "tampered")
    with pytest.raises(BackendBoundaryError):
        hash(supported)


def test_optional_import_isolation() -> None:
    root = str(Path(base.__file__).parents[2])
    code = (
        "import sys; import anyalgebra.backends.base; "
        "print(','.join(sorted(x for x in sys.modules "
        "if x.split('.')[0] in {'sympy','numpy','scipy','sage','gap'})))"
    )
    output = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": root},
    )
    assert output.stdout.strip() == ""


def factory(cls: type[object]) -> Any:
    return cast(Any, cls.__dict__["create"]).__func__


@pytest.mark.parametrize(
    "cls",
    (
        CapabilitySpec,
        BackendCapabilities,
        BackendRequest,
        BackendOptions,
        BackendProvenance,
        BackendSupported,
        BackendUnsupported,
        BackendResult,
    ),
)
def test_all_records_are_factory_owned_and_non_subclassable(cls: type[object]) -> None:
    with pytest.raises(BackendBoundaryError):
        cls()
    with pytest.raises(TypeError):
        type("HostileSubclass", (cls,), {})


def test_every_defensive_factory_and_validation_branch_is_observable() -> None:
    cap, req = caps(), request()
    provenance = BackendProvenance.create(req, cap)
    options = BackendOptions.create(convention_hashes=(h("c"),), limits=(("cases", 2),))
    invalid = (
        lambda: caps(name=""),
        lambda: caps(name="x" * 257),
        lambda: caps(name="bad\nname"),
        lambda: caps(name="bad/name"),
        lambda: caps(specifications=object()),
        lambda: caps(specifications=(object(),)),
        lambda: caps(specifications=(spec(), spec())),
        lambda: caps(convention_hashes=(h("d"),)),
        lambda: request(convention_hashes=(h("c"), h("c"))),
        lambda: request(limits=(("cases", 1, 2),)),
        lambda: request(limits=(object(),)),
        lambda: request(limits=(("cases", 1), ("cases", 2))),
        lambda: request(requires_exact=cast(bool, 1)),
        lambda: CapabilitySpec.create(
            operation="rank", domain="QQ", algorithm="direct", exact=cast(bool, 1)
        ),
        lambda: negotiate(object(), req),
        lambda: negotiate(cap, object()),
        lambda: BackendUnsupported.create(req, cap, "missing_operation"),
        lambda: BackendUnsupported.create(req, cap, UnsupportedReason.MISSING_DOMAIN),
        lambda: BackendProvenance.create(request(algorithm="missing"), cap),
        lambda: BackendResult.create(
            request=req, provenance=object(), options=options, value_hash=h("v")
        ),
    )
    for construct in invalid:
        with pytest.raises(BackendBoundaryError):
            construct()
    with pytest.raises(BackendBoundaryError):
        base._digest({"set": {"x"}})
    with pytest.raises(NotImplementedError):
        base._Record()._record()
    tampered = caps()
    object.__setattr__(tampered, "name", "tampered")
    assert tampered != caps()
    assert caps() != request()

    cases: tuple[tuple[type[object], dict[str, object]], ...] = (
        (
            CapabilitySpec,
            {"operation": "rank", "domain": "QQ", "algorithm": "direct", "exact": True},
        ),
        (BackendCapabilities, {"name": "x", "version": "0", "specifications": ()}),
        (
            BackendRequest,
            {
                "operation": "rank",
                "domain": "QQ",
                "algorithm": "direct",
                "input_hashes": (),
                "limits": (),
            },
        ),
        (BackendOptions, {}),
        (BackendProvenance, {"request": req, "capabilities": cap}),
        (BackendSupported, {"request": req, "capabilities": cap}),
        (
            BackendUnsupported,
            {
                "request": req,
                "capabilities": cap,
                "reason": UnsupportedReason.MISSING_DOMAIN,
            },
        ),
        (
            BackendResult,
            {
                "request": req,
                "provenance": provenance,
                "options": options,
                "value_hash": h("v"),
            },
        ),
    )
    for cls, keywords in cases:
        with pytest.raises(BackendBoundaryError):
            factory(cls)(object(), **keywords)
