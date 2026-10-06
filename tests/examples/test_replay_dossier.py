"""Executable contract for the V00-095 serialization and replay dossier."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from anyalgebra.evidence.run import (
    ResultReceipt,
    result_receipt_record,
    result_receipt_registry,
)
from anyalgebra.persistence.registry import SchemaError
from examples import replay_dossier


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_canonical_safe_round_trips_and_rejects_unsafe_records() -> None:
    """Only registered, canonical JSON records cross the decoding boundary."""
    dossier = replay_dossier.build_dossier()
    positive = dossier["receipts"]["positive"]
    record = result_receipt_record(positive)
    reloaded = replay_dossier.safe_record(record, registry=result_receipt_registry())
    assert type(reloaded) is ResultReceipt
    assert reloaded.canonical_bytes() == positive.canonical_bytes()
    assert hashlib.sha256(positive.canonical_bytes()).hexdigest() == (
        positive.semantic_hash.digest
    )

    unknown = dict(record)
    unknown["schemaType"] = "untrusted.python.object"
    future = dict(record)
    future["schemaVersion"] = 2
    for unsafe, code in (
        (unknown, "unknown_schema_type"),
        (future, "unknown_schema_version"),
    ):
        with pytest.raises(SchemaError, match=code):
            replay_dossier.safe_record(unsafe, registry=result_receipt_registry())

    calls = 0

    class Poison:
        def __call__(self) -> None:
            nonlocal calls
            calls += 1

    executable = dict(record)
    executable["callable"] = Poison()
    with pytest.raises(SchemaError, match="invalid_record_value"):
        replay_dossier.safe_record(executable, registry=result_receipt_registry())
    assert calls == 0


def test_migration_is_registered_directional_and_preserves_its_input() -> None:
    """The fixture uses the public migration registry, never an import hook."""
    original = replay_dossier.migration_fixture()
    original_copy = dict(original)
    migrated = replay_dossier.REPLAY_DOSSIER_MIGRATIONS.migrate_record(
        original, target_version=2
    )
    assert original == original_copy
    assert migrated == {
        "schemaType": "anyalgebra.example.replay_fixture",
        "schemaVersion": 2,
        "migration": "v1-to-v2",
        "payload": "durable-fixture",
    }
    with pytest.raises(SchemaError, match="missing_migration"):
        replay_dossier.REPLAY_DOSSIER_MIGRATIONS.migrate_record(
            original, target_version=3
        )


def test_no_self_promotion_and_receipt_provenance_are_explicit() -> None:
    """Execution, conclusion, and tier remain separate in every dossier case."""
    dossier = replay_dossier.build_dossier()
    contracts = dossier["contracts"]
    receipts = dossier["receipts"]
    source = dossier["source"]
    manifest = dossier["manifest"]
    for name, receipt in receipts.items():
        contract = contracts[name]
        assert receipt.execution_status == "completed"
        assert receipt.evidence_tier == "E1-executed"
        assert receipt.contract_hash == contract.semantic_hash
        assert receipt.input_hashes == contract.input_hashes
        assert receipt.source_anchors == (source.semantic_hash,)
        assert receipt.convention_manifests == (manifest.semantic_hash,)
        assert receipt.algorithms == contract.algorithms
        assert receipt.backend_name == "reference"
        assert receipt.bounds == contract.bounds
        assert receipt.artifacts
        assert hashlib.sha256(receipt.canonical_bytes()).hexdigest() == (
            receipt.semantic_hash.digest
        )
    assert receipts["positive"].mathematical_outcome == "constructed"
    assert receipts["negative"].mathematical_outcome == "counterexample"
    assert receipts["inconclusive"].mathematical_outcome == "inconclusive"
    assert not hasattr(dossier["packets"]["positive"], "evidence_tier")


def test_source_anchor_binds_the_actual_dossier_bytes() -> None:
    """The recorded location and hash identify the source actually executed."""
    source = replay_dossier.build_dossier()["source"]
    source_path = Path(replay_dossier.__file__).resolve()
    assert source.location == "examples/replay_dossier.py"
    assert source.edition_or_commit == "v0.0-working-tree"
    assert source.locator == "functions:_source,_contract,_packet,_run,build_dossier"
    assert source.excerpt_hash is None
    assert (
        source.content_hash.digest
        == hashlib.sha256(source_path.read_bytes()).hexdigest()
    )


def test_replay_and_staleness_preserve_all_outcome_kinds() -> None:
    """Negative and unresolved records replay; a changed bound is stale reuse."""
    dossier = replay_dossier.build_dossier()
    reports = dossier["reports"]
    assert reports["positive"].status == "exact_match"
    assert reports["negative"].status == "exact_match"
    assert reports["inconclusive"].status == "exact_match"
    assert reports["stale"].status == "stale"
    assert reports["stale"].stale_edges == ("contract.bounds",)
    assert dossier["receipts"]["positive"].canonical_bytes()


def test_dossier_cli_is_cwd_independent_deterministic_json(tmp_path: Path) -> None:
    """The compact output omits clock-dependent receipt IDs but retains evidence."""
    result = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "examples" / "replay_dossier.py")],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    expected = replay_dossier.run_example()
    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout == (
        json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n"
    )
    assert replay_dossier.run_example() == expected
