"""Safe evidence serialization and replay dossier; no scientific claim is made.

The example exercises a constructed result, a counterexample, an inconclusive
bounded search, and stale reuse after a changed resource bound.  It composes
the public allow-listed codecs, calculation runner, migration registry, and
read-only replay API.  It does not deserialize imports, callables, or arbitrary
Python objects.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict, cast

if __package__ is None:  # Supports direct execution from any working directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anyalgebra.core.parents import SemanticHash
from anyalgebra.evidence.contracts import CalculationContract
from anyalgebra.evidence.models import ConventionManifest, SourceAnchor
from anyalgebra.evidence.replay import ReplayReport, ReplayResolver, verify_replay
from anyalgebra.evidence.run import (
    CalculationResult,
    ExecutionEnvironment,
    ResultReceipt,
    result_receipt_record,
    result_receipt_registry,
    run_calculation,
)
from anyalgebra.persistence.migrations import MigrationRegistry
from anyalgebra.persistence.registry import JSONValue, SchemaError, SerializerRegistry


_MIGRATION_TAG = "anyalgebra.example.replay_fixture"


class ReplayDossier(TypedDict):
    """The durable records retained for direct test inspection."""

    contracts: dict[str, CalculationContract]
    manifest: ConventionManifest
    packets: dict[str, CalculationResult]
    receipts: dict[str, ResultReceipt]
    reports: dict[str, ReplayReport]
    source: SourceAnchor


def _hash(label: str) -> SemanticHash:
    return SemanticHash("sha256", hashlib.sha256(label.encode("utf-8")).hexdigest())


def safe_record(record: object, *, registry: SerializerRegistry) -> object:
    """Decode only with an explicit allow-list; data never selects executable code."""
    return registry.from_record(record)


def _source() -> SourceAnchor:
    return SourceAnchor.create(
        "examples/replay_dossier.py",
        SemanticHash(
            "sha256", hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
        ),
        "v0.0-working-tree",
        "functions:_source,_contract,_packet,_run,build_dossier",
    )


def _manifest(source: SourceAnchor) -> ConventionManifest:
    return ConventionManifest.create(
        "example.replay.conventions.v1",
        ("1", "e"),
        coefficient_domain="QQ",
        multiplication_fixture=_hash("multiplication-fixture-v1"),
        sources=(source,),
        status="accepted",
        signs=(("orientation", "positive"),),
        involutions=(("conjugation", "identity"),),
        normalization=(("unit", "one"),),
        indexing="zero-based",
        parenthesization="explicit",
        conversions=(("coordinates", "identity"),),
    )


def _contract(
    source: SourceAnchor,
    manifest: ConventionManifest,
    outcome: str,
    *,
    bounds: tuple[tuple[str, int], ...] = (("cases", 4),),
) -> CalculationContract:
    return CalculationContract.create(
        contract_name="example.replay-dossier.v1",
        project_id="anyalgebra.examples",
        question=(
            "Record one bounded infrastructure outcome without a scientific claim."
        ),
        acceptable_outcomes=(outcome,),
        input_hashes=(("fixture", _hash(f"input:{outcome}")),),
        source_anchors=(source,),
        convention_manifests=(manifest,),
        algorithms=(("fixture-runner", "1"),),
        backend_request="exact",
        backend_name="reference",
        backend_version="1",
        assumptions=(("scope", "infrastructure-fixture"),),
        theorem_hypotheses=(("carrier", "finite"),),
        bounds=bounds,
        required_cross_checks=("recorded direct fixture",),
        expected_artifacts=("artifact-hash",),
        acceptance_predicates=("retain the declared outcome without promotion",),
    )


def _packet(outcome: str) -> CalculationResult:
    if outcome == "counterexample":
        return CalculationResult.create(
            outcome=outcome,
            summary="bounded fixture has a retained counterexample",
            artifacts=(_hash("negative-artifact"),),
            witnesses=(_hash("smallest-negative-witness"),),
        )
    if outcome == "inconclusive":
        return CalculationResult.create(
            outcome=outcome,
            summary="bounded fixture leaves named branches unresolved",
            artifacts=(_hash("inconclusive-artifact"),),
            case_counts=(("casesExpected", 4), ("casesRun", 2)),
            remaining_branches=("branch-2", "branch-3"),
        )
    return CalculationResult.create(
        outcome=outcome,
        summary="neutral constructed infrastructure fixture",
        artifacts=(_hash("positive-artifact"),),
    )


def _run(
    contract: CalculationContract, packet: CalculationResult, platform: str
) -> ResultReceipt:
    return run_calculation(
        contract,
        lambda _: packet,
        environment=ExecutionEnvironment.create(platform=platform),
    )


def _migrate_v1_to_v2(record: Mapping[str, object]) -> dict[str, JSONValue]:
    """Pure trusted fixture migration registered by code, not by input data."""
    return {
        "schemaType": _MIGRATION_TAG,
        "schemaVersion": 2,
        "migration": "v1-to-v2",
        "payload": cast(JSONValue, record["payload"]),
    }


REPLAY_DOSSIER_MIGRATIONS = MigrationRegistry().with_migration(
    _MIGRATION_TAG, 1, 2, _migrate_v1_to_v2
)


def migration_fixture() -> dict[str, JSONValue]:
    """Return one preserved v1 record suitable for a pure upgrade demonstration."""
    return {
        "schemaType": _MIGRATION_TAG,
        "schemaVersion": 1,
        "payload": "durable-fixture",
    }


def build_dossier() -> ReplayDossier:
    """Run the bounded cases and retain records, reports, and old evidence."""
    source = _source()
    manifest = _manifest(source)
    contracts = {
        outcome: _contract(source, manifest, outcome)
        for outcome in ("constructed", "counterexample", "inconclusive")
    }
    packets = {outcome: _packet(outcome) for outcome in contracts}
    receipts = {
        "positive": _run(contracts["constructed"], packets["constructed"], "origin"),
        "negative": _run(
            contracts["counterexample"], packets["counterexample"], "origin"
        ),
        "inconclusive": _run(
            contracts["inconclusive"], packets["inconclusive"], "origin"
        ),
    }
    reruns = {
        "positive": _run(contracts["constructed"], packets["constructed"], "rerun"),
        "negative": _run(
            contracts["counterexample"], packets["counterexample"], "rerun"
        ),
        "inconclusive": _run(
            contracts["inconclusive"], packets["inconclusive"], "rerun"
        ),
    }
    changed_contract = _contract(
        source, manifest, "constructed", bounds=(("cases", 5),)
    )
    changed = _run(changed_contract, packets["constructed"], "changed-bound")

    def report_for(name: str) -> ReplayReport:
        original = receipts[name]
        rerun = reruns[name]
        return verify_replay(
            original,
            resolver=ReplayResolver.create(
                receipts=(original, rerun),
                reruns=((original.semantic_hash, rerun.semantic_hash),),
            ),
        )

    reports = {
        "positive": report_for("positive"),
        "negative": report_for("negative"),
        "inconclusive": report_for("inconclusive"),
        "stale": verify_replay(
            receipts["positive"],
            resolver=ReplayResolver.create(
                receipts=(receipts["positive"], changed),
                reruns=((receipts["positive"].semantic_hash, changed.semantic_hash),),
            ),
        ),
    }
    return {
        "contracts": {
            "positive": contracts["constructed"],
            "negative": contracts["counterexample"],
            "inconclusive": contracts["inconclusive"],
        },
        "manifest": manifest,
        "packets": {
            "positive": packets["constructed"],
            "negative": packets["counterexample"],
            "inconclusive": packets["inconclusive"],
        },
        "receipts": receipts,
        "reports": reports,
        "source": source,
    }


def _error_code(record: object) -> str:
    try:
        safe_record(record, registry=result_receipt_registry())
    except SchemaError as error:
        return error.code
    raise RuntimeError("unsafe fixture unexpectedly decoded")


def run_example() -> dict[str, object]:
    """Return a deterministic, compact evidence summary with no promotion claim."""
    dossier = build_dossier()
    positive = dossier["receipts"]["positive"]
    record = result_receipt_record(positive)
    unknown = dict(record)
    unknown["schemaType"] = "untrusted.python.object"
    future = dict(record)
    future["schemaVersion"] = 2
    migrated = REPLAY_DOSSIER_MIGRATIONS.migrate_record(
        migration_fixture(), target_version=2
    )
    reloaded = safe_record(record, registry=result_receipt_registry())
    if type(reloaded) is not ResultReceipt:
        raise RuntimeError("receipt record did not round trip through its registry")
    return {
        "canonical": {
            "contract_hash": str(dossier["contracts"]["positive"].semantic_hash),
            "receipt_hash_covers_canonical_bytes": hashlib.sha256(
                positive.canonical_bytes()
            ).hexdigest()
            == positive.semantic_hash.digest,
            "round_trip": reloaded.canonical_bytes() == positive.canonical_bytes(),
        },
        "migration": {
            "direction": "v1-to-v2",
            "original_version": 1,
            "target_version": migrated["schemaVersion"],
        },
        "outcomes": [
            {
                "case": name,
                "evidence_tier": receipt.evidence_tier,
                "execution_status": receipt.execution_status,
                "mathematical_outcome": receipt.mathematical_outcome,
            }
            for name, receipt in dossier["receipts"].items()
        ],
        "replay": {name: report.status for name, report in dossier["reports"].items()},
        "safe_decode": {
            "future_version": _error_code(future),
            "unknown_tag": _error_code(unknown),
        },
        "scope": (
            "serialization and replay infrastructure only; execution, mathematical "
            "outcome, and evidence tier remain separate; no scientific or legacy "
            "correctness claim is made"
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), sort_keys=True, separators=(",", ":")))
