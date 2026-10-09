"""Every legacy symbol deferred by the frozen parity ledger has one current owner.

The v0.0 ledger records a release label for each deferred family.  Those labels
are historical: a release can later be given another purpose, which leaves the
family without an owner.  The ownership ledger checked here assigns every
deferred symbol to exactly one work package instead, and the maintainer roadmap
schedules the work packages.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "docs/legacy/generated/algmul-dispositions.json"
LEDGER = ROOT / "docs/legacy/deferred-ownership.json"
PARITY_CONTRACT = ROOT / "docs/legacy/algmul-parity.md"
WORK_PACKAGE_ID = re.compile(r"wp\.[a-z]+(-[a-z]+)*")
RELEASE_LABEL = re.compile(r"\bv\d+\.\d+")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def _deferred_symbols() -> dict[str, set[str]]:
    """Return the symbol names of each deferred family in the frozen ledger."""
    audit = _read(AUDIT)
    deferred: dict[str, set[str]] = {
        rule["partition"]: set()
        for rule in audit["rules"]
        if rule["disposition"] == "defer"
    }
    for item in audit["assignments"]:
        if item["partition"] in deferred and item["kind"] == "symbol":
            deferred[item["partition"]].add(item["targetId"].split(":", 1)[1])
    return deferred


def _work_packages() -> list[dict[str, Any]]:
    packages = _read(LEDGER)["workPackages"]
    assert type(packages) is list
    return packages


def test_ledger_is_canonical_and_bound_to_the_frozen_audit() -> None:
    ledger = _read(LEDGER)
    canonical = json.dumps(ledger, indent=2, ensure_ascii=False).encode() + b"\n"
    assert LEDGER.read_bytes() == canonical
    assert set(ledger) == {
        "schema",
        "purpose",
        "dispositions",
        "defaultWorkPackage",
        "workPackages",
    }
    assert ledger["schema"] == "anyalgebra.algmul-deferred-ownership/v1"
    assert ledger["dispositions"] == {
        "path": AUDIT.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(AUDIT.read_bytes()).hexdigest().upper(),
    }


def test_work_packages_are_well_formed_and_carry_no_release_label() -> None:
    packages = _work_packages()
    identifiers = [package["id"] for package in packages]
    assert len(identifiers) == len(set(identifiers)) == 11
    for package in packages:
        assert set(package) == {"id", "title", "scope", "symbols"}
        assert WORK_PACKAGE_ID.fullmatch(package["id"])
        assert len(package["title"]) >= 10
        assert len(package["scope"]) >= 40
        assert package["symbols"]
    # Release labels are what went stale.  They belong to the roadmap alone.
    text = LEDGER.read_text(encoding="utf-8")
    assert RELEASE_LABEL.findall(text.replace("frozen v0.0 parity ledger", "")) == []


def test_every_deferred_symbol_has_exactly_one_owner() -> None:
    deferred = _deferred_symbols()
    assert {partition: len(names) for partition, names in deferred.items()} == {
        "jordan_matrices": 33,
        "conjugations_norms": 83,
        "clifford_geometric_grassmann": 107,
        "lie_group_experiments": 78,
        "incomplete_stubs_comments": 3,
    }
    owned: dict[str, list[str]] = {partition: [] for partition in deferred}
    for package in _work_packages():
        for group in package["symbols"]:
            assert group["names"] == sorted(group["names"])
            owned[group["partition"]].extend(group["names"])
    for partition, names in deferred.items():
        assert len(owned[partition]) == len(set(owned[partition]))
        assert set(owned[partition]) == names


def test_records_without_a_symbol_fall_to_a_default_owner() -> None:
    defaults = _read(LEDGER)["defaultWorkPackage"]
    assert type(defaults) is dict
    assert set(defaults) == set(_deferred_symbols())
    assert set(defaults.values()) <= {package["id"] for package in _work_packages()}


def test_parity_contract_names_every_work_package() -> None:
    contract = PARITY_CONTRACT.read_text(encoding="utf-8")
    assert "deferred-ownership.json" in contract
    for package in _work_packages():
        assert f"`{package['id']}`" in contract
