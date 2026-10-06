"""Tests for atomic staged construction of finite free-module parents."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import anyalgebra.core.parents as parent_module
from anyalgebra.core.domains import QQ, ZZ
from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.core.parents import (
    FrozenBuilderError,
    ParentBuildError,
    ParentBuildIssue,
    ParentBuilder,
)


def _issue_codes(error: ParentBuildError) -> tuple[str, ...]:
    return tuple(issue.code for issue in error.issues)


def test_empty_builder_reports_every_missing_required_field() -> None:
    """Freeze collects independent missing-field issues instead of failing fast."""
    builder = ParentBuilder()

    with pytest.raises(ParentBuildError) as caught:
        builder.freeze()

    assert isinstance(caught.value, AnyAlgebraError)
    assert caught.value.issues == (
        ParentBuildIssue(
            code="missing_domain",
            field="domain",
            message="module domain is required",
        ),
        ParentBuildIssue(
            code="missing_basis",
            field="basis",
            message="module basis is required",
        ),
    )
    assert _issue_codes(caught.value) == ("missing_domain", "missing_basis")
    with pytest.raises(FrozenInstanceError):
        caught.value.issues[0].code = "changed"  # type: ignore[misc]


def test_malformed_fields_are_independent_and_mismatch_is_not_speculative() -> None:
    """Invalid prerequisites and an invalid optional name all receive issues."""
    builder = (
        ParentBuilder().with_name(" bad ").with_basis(("e",)).with_domain(object())
    )

    with pytest.raises(ParentBuildError) as caught:
        builder.freeze()

    assert _issue_codes(caught.value) == (
        "invalid_domain",
        "invalid_basis",
        "invalid_name",
    )
    assert all(
        issue.field in {"domain", "basis", "name"} for issue in caught.value.issues
    )
    assert "domain_basis_mismatch" not in _issue_codes(caught.value)


@pytest.mark.parametrize("name", ["", " ", " V", "V ", 1, True])
def test_each_malformed_name_has_one_stable_issue(name: object) -> None:
    """The optional name obeys the same exact-string contract as FreeModule."""
    domain = ZZ()
    builder = (
        ParentBuilder()
        .with_domain(domain)
        .with_basis(Basis(("e",), coefficient_domain=domain))
    )
    builder.with_name(name)

    with pytest.raises(ParentBuildError) as caught:
        builder.freeze()

    assert _issue_codes(caught.value) == ("invalid_name",)


def test_literal_domain_basis_mismatch_is_reported_only_after_validity() -> None:
    """Equal shapes never substitute one literal coefficient parent for another."""
    builder = (
        ParentBuilder()
        .with_domain(ZZ())
        .with_basis(Basis(("e",), coefficient_domain=QQ()))
    )

    with pytest.raises(ParentBuildError) as caught:
        builder.freeze()

    assert _issue_codes(caught.value) == ("domain_basis_mismatch",)


def test_issue_order_does_not_depend_on_mutator_order() -> None:
    """Validation order is domain, basis, name, then cross-field checks."""
    left = ParentBuilder().with_domain(None).with_basis(None).with_name(7)
    right = ParentBuilder().with_name(7).with_basis(None).with_domain(None)

    errors: list[ParentBuildError] = []
    for builder in (left, right):
        with pytest.raises(ParentBuildError) as caught:
            builder.freeze()
        errors.append(caught.value)

    assert errors[0].issues == errors[1].issues
    assert _issue_codes(errors[0]) == (
        "invalid_domain",
        "invalid_basis",
        "invalid_name",
    )


def test_failed_freeze_can_be_corrected_without_leaking_a_parent() -> None:
    """An invalid freeze leaves the builder mutable and publishes no result."""
    builder = ParentBuilder().with_name("V")

    with pytest.raises(ParentBuildError):
        builder.freeze()
    for attribute in ("parent", "result", "partial_parent"):
        assert not hasattr(builder, attribute)

    domain = ZZ()
    module = (
        builder.with_domain(domain)
        .with_basis(Basis(("e0", "e1"), coefficient_domain=domain))
        .freeze()
    )

    assert type(module) is FreeModule
    assert module.domain is domain
    assert module.basis.labels == ("e0", "e1")
    assert module.name == "V"


@pytest.mark.parametrize("labels", [(), ("e",), ("e0", "e1", "e2")])
def test_successful_freeze_supports_zero_and_higher_rank(
    labels: tuple[str, ...],
) -> None:
    """The current builder freezes exactly the available FreeModule parent."""
    domain = QQ()
    module = (
        ParentBuilder()
        .with_basis(Basis(labels, coefficient_domain=domain))
        .with_domain(domain)
        .freeze()
    )

    assert type(module) is FreeModule
    assert module.rank == len(labels)
    assert module.name is None


def test_freeze_uses_an_immutable_basis_snapshot() -> None:
    """Mutable label inputs cannot alter a successfully built parent later."""
    labels = ["x", "y"]
    domain = ZZ()
    basis = Basis(labels, coefficient_domain=domain)
    builder = ParentBuilder().with_domain(domain).with_basis(basis).with_name("V")
    labels.append("z")

    module = builder.freeze()
    labels.clear()

    assert module.basis is basis
    assert module.basis.labels == ("x", "y")


def test_factory_failure_is_atomic_and_builder_remains_correctable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The builder marks success only after the immutable factory returns."""
    domain = ZZ()
    builder = (
        ParentBuilder()
        .with_domain(domain)
        .with_basis(Basis(("e",), coefficient_domain=domain))
    )
    real_factory = FreeModule

    def fail_factory(*args: object, **kwargs: object) -> FreeModule:
        raise RuntimeError("factory failed")

    monkeypatch.setattr(parent_module, "FreeModule", fail_factory)
    with pytest.raises(RuntimeError, match="factory failed"):
        builder.freeze()

    monkeypatch.setattr(parent_module, "FreeModule", real_factory)
    assert type(builder.with_name("recovered").freeze()) is FreeModule


def test_success_consumes_builder_and_all_lifecycle_errors_are_typed() -> None:
    """Every mutation and repeated freeze is rejected after one success."""
    domain = ZZ()
    basis = Basis(("e",), coefficient_domain=domain)
    builder = ParentBuilder().with_domain(domain).with_basis(basis)
    builder.freeze()

    operations = (
        ("with_domain", lambda: builder.with_domain(QQ())),
        ("with_basis", lambda: builder.with_basis(basis)),
        ("with_name", lambda: builder.with_name("V")),
        ("freeze", builder.freeze),
    )
    for operation, action in operations:
        with pytest.raises(FrozenBuilderError) as caught:
            action()
        assert isinstance(caught.value, AnyAlgebraError)
        assert caught.value.operation == operation


def test_builder_exposes_no_fingerprint_cycle_or_element_api() -> None:
    """V00-022 stops before fingerprints, cyclic sessions, and elements."""
    builder = ParentBuilder()

    for attribute in (
        "fingerprint",
        "element",
        "zero",
        "session",
        "with_operation",
        "with_parent_reference",
    ):
        assert not hasattr(builder, attribute)
