"""Tests for stable semantic fingerprints of finite free-module parents."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
import subprocess
import sys

import pytest

from anyalgebra.core.domains import DomainElement, QQ, ZZ
from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.core.parents import ParentFingerprintError, SemanticHash


@dataclass(frozen=True)
class _FingerprintDomain:
    """A noncanonical exact domain identified by an explicit semantic hash."""

    domain_hash: SemanticHash

    def normalize(self, value: object) -> object:
        return value

    def element(self, value: object) -> DomainElement[object]:
        raise NotImplementedError

    def fingerprint(self) -> SemanticHash:
        return self.domain_hash


@dataclass(frozen=True)
class _UnknownDomain:
    """A protocol-shaped domain with no stable identity declaration."""

    def normalize(self, value: object) -> object:
        return value

    def element(self, value: object) -> DomainElement[object]:
        raise NotImplementedError


@dataclass(frozen=True)
class _InvalidFingerprintDomain(_UnknownDomain):
    """A domain whose fingerprint method violates the return contract."""

    def fingerprint(self) -> str:
        return "sha256:" + "0" * 64


@dataclass(frozen=True)
class _NonCallableFingerprintDomain(_UnknownDomain):
    """A domain whose fingerprint attribute is not a method."""

    fingerprint = 7


@dataclass(frozen=True)
class _FailingFingerprintDomain(_UnknownDomain):
    """A domain whose declared fingerprint computation fails."""

    def fingerprint(self) -> SemanticHash:
        raise RuntimeError("unstable internal detail at 0xfeed")


@dataclass(frozen=True)
class _FailingLookupDomain(_UnknownDomain):
    """A domain whose fingerprint attribute cannot even be retrieved."""

    @property
    def fingerprint(self) -> SemanticHash:
        raise RuntimeError("unstable lookup detail at 0xfeed")


class _StringSubclass(str):
    """An exact-string boundary fixture."""


def _expected_module_digest(domain_record: dict[str, object], labels: list[str]) -> str:
    """Independently reproduce the documented canonical fingerprint bytes."""
    record = {
        "basis": {"labels": labels},
        "coefficientDomain": domain_record,
        "schemaType": "anyalgebra.parent.free_module",
        "schemaVersion": 1,
    }
    canonical = json.dumps(
        record,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@pytest.mark.parametrize(
    ("domain", "domain_record"),
    [
        (
            ZZ(),
            {"schemaType": "anyalgebra.domain.integer", "schemaVersion": 1},
        ),
        (
            QQ(),
            {"schemaType": "anyalgebra.domain.rational", "schemaVersion": 1},
        ),
    ],
)
def test_canonical_domains_have_independently_reproducible_golden_digests(
    domain: object, domain_record: dict[str, object]
) -> None:
    """ZZ and QQ use stable allow-listed records, not class names or reprs."""
    basis = Basis(("e0", "β"), coefficient_domain=domain)  # type: ignore[arg-type]
    module = FreeModule(domain, basis, name="display only")  # type: ignore[arg-type]

    result = module.fingerprint()

    assert result == SemanticHash(
        algorithm="sha256",
        digest=_expected_module_digest(domain_record, ["e0", "β"]),
    )
    assert str(result) == f"sha256:{result.digest}"
    assert "0x" not in str(result)


def test_name_and_reconstruction_do_not_change_semantic_identity() -> None:
    """Display metadata is excluded from equality and fingerprint material."""
    domain = ZZ()
    left = FreeModule(
        domain,
        Basis(("x", "y"), coefficient_domain=domain),
        name="V",
    )
    right = FreeModule(
        domain,
        Basis(["x", "y"], coefficient_domain=domain),
        name="renamed display",
    )

    assert left is not right
    assert left == right
    assert left.fingerprint() == right.fingerprint()
    assert left.fingerprint() is not left.fingerprint()


def test_basis_labels_order_and_coefficient_domain_are_semantic() -> None:
    """Every defining coordinate choice affects the parent fingerprint."""
    zz = ZZ()
    qq = QQ()
    baseline = FreeModule(zz, Basis(("x", "y"), coefficient_domain=zz))
    relabeled = FreeModule(zz, Basis(("x", "z"), coefficient_domain=zz))
    reordered = FreeModule(zz, Basis(("y", "x"), coefficient_domain=zz))
    rational = FreeModule(qq, Basis(("x", "y"), coefficient_domain=qq))

    fingerprints = {
        parent.fingerprint() for parent in (baseline, relabeled, reordered, rational)
    }
    assert len(fingerprints) == 4


def test_explicit_nested_domain_fingerprint_is_a_canonical_reference() -> None:
    """A noncanonical domain may opt in through a validated semantic hash."""
    nested = SemanticHash("sha256", "a" * 64)
    domain = _FingerprintDomain(nested)
    module = FreeModule(domain, Basis(("u",), coefficient_domain=domain))
    domain_record = {
        "fingerprint": {"algorithm": "sha256", "digest": "a" * 64},
        "schemaType": "anyalgebra.domain.reference",
        "schemaVersion": 1,
    }

    assert module.fingerprint().digest == _expected_module_digest(domain_record, ["u"])


def test_unknown_or_invalid_domain_identity_is_rejected_deterministically() -> None:
    """No repr, address, or import path becomes an accidental domain identity."""
    unknown = _UnknownDomain()
    invalid = _InvalidFingerprintDomain()

    for domain, reason in (
        (unknown, "coefficient domain has no semantic fingerprint"),
        (invalid, "coefficient domain fingerprint must be a SemanticHash"),
    ):
        module = FreeModule(domain, Basis(("e",), coefficient_domain=domain))
        with pytest.raises(ParentFingerprintError) as caught:
            module.fingerprint()
        assert isinstance(caught.value, AnyAlgebraError)
        assert caught.value.parent is module
        assert caught.value.reason == reason
        assert "0x" not in str(caught.value)
        assert "test_parent_fingerprint" not in str(caught.value)


@pytest.mark.parametrize(
    ("domain", "reason"),
    [
        (
            _NonCallableFingerprintDomain(),
            "coefficient domain fingerprint must be callable",
        ),
        (
            _FailingFingerprintDomain(),
            "coefficient domain fingerprint failed",
        ),
        (
            _FailingLookupDomain(),
            "coefficient domain fingerprint lookup failed",
        ),
    ],
)
def test_broken_domain_fingerprint_protocol_has_typed_stable_diagnostics(
    domain: object, reason: str
) -> None:
    """Protocol failures never leak arbitrary domain details into messages."""
    module = FreeModule(
        domain,  # type: ignore[arg-type]
        Basis(("e",), coefficient_domain=domain),  # type: ignore[arg-type]
    )

    with pytest.raises(ParentFingerprintError) as caught:
        module.fingerprint()

    assert caught.value.parent is module
    assert caught.value.reason == reason
    assert str(caught.value) == reason
    assert "0x" not in str(caught.value)
    if type(domain) is _NonCallableFingerprintDomain:
        assert caught.value.__cause__ is None
    else:
        assert isinstance(caught.value.__cause__, RuntimeError)


@pytest.mark.parametrize(
    ("algorithm", "digest"),
    [
        ("SHA256", "0" * 64),
        ("sha1", "0" * 40),
        ("sha256", "0" * 63),
        ("sha256", "A" * 64),
        ("sha256", "g" * 64),
        (1, "0" * 64),
        ("sha256", 1),
        (_StringSubclass("sha256"), "0" * 64),
        ("sha256", _StringSubclass("0" * 64)),
    ],
)
def test_semantic_hash_rejects_invalid_algorithm_or_digest_fields(
    algorithm: object, digest: object
) -> None:
    """Hash references have one exact, canonical v0.0 representation."""
    with pytest.raises(ParentFingerprintError):
        SemanticHash(algorithm, digest)  # type: ignore[arg-type]


def test_semantic_hash_and_parent_definition_are_immutable_snapshots() -> None:
    """Caller collections and later mutation attempts cannot alter the digest."""
    labels = ["x", "y"]
    domain = ZZ()
    basis = Basis(labels, coefficient_domain=domain)
    module = FreeModule(domain, basis, name="V")
    before = module.fingerprint()
    labels[:] = ["changed"]

    assert module.fingerprint() == before
    with pytest.raises(FrozenInstanceError):
        before.digest = "f" * 64  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        module.name = "changed"  # type: ignore[misc]
    assert module.fingerprint() == before


def test_fingerprint_is_stable_across_hash_seeds_and_processes() -> None:
    """Runtime hash randomization has no role in semantic bytes."""
    root = Path(__file__).resolve().parents[2]
    script = (
        f"import sys;sys.path.insert(0,{str(root / 'src')!r});"
        "from anyalgebra.core.domains import ZZ;"
        "from anyalgebra.core.modules import Basis,FreeModule;"
        "d=ZZ();"
        "print(FreeModule(d,Basis(('e0','β'),coefficient_domain=d),"
        "name='ignored').fingerprint())"
    )
    outputs: list[str] = []
    for seed in ("1", "8675309"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-I", "-c", script],
            cwd=root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(completed.stdout.strip())

    assert (
        outputs
        == [
            "sha256:"
            + _expected_module_digest(
                {"schemaType": "anyalgebra.domain.integer", "schemaVersion": 1},
                ["e0", "β"],
            )
        ]
        * 2
    )
