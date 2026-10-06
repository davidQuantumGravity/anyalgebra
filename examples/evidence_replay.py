"""Minimal immutable receipt serialization and replay demonstration.

This is an infrastructure example.  Its exact hashes and replay statuses make
no mathematical or scientific claim.
"""

from __future__ import annotations

import hashlib
from typing import cast

from anyalgebra.core.parents import SemanticHash
from anyalgebra.evidence.contracts import CalculationContract
from anyalgebra.evidence.replay import ReplayResolver, verify_replay
from anyalgebra.evidence.run import (
    CalculationResult,
    ExecutionEnvironment,
    ResultReceipt,
    result_receipt_record,
    result_receipt_registry,
    run_calculation,
)


def _hash(text: str) -> SemanticHash:
    return SemanticHash("sha256", hashlib.sha256(text.encode("utf-8")).hexdigest())


def _contract(input_hash: SemanticHash) -> CalculationContract:
    return CalculationContract.create(
        contract_name="receipt-replay-example",
        project_id="example.evidence",
        question="record a neutral replay fixture",
        acceptable_outcomes=("constructed",),
        input_hashes=(("input", input_hash),),
        source_anchors=(),
        convention_manifests=(),
        algorithms=(("example", "1"),),
        backend_request="exact",
        backend_name="reference",
        backend_version="1",
        assumptions=(),
        theorem_hypotheses=(),
        bounds=(("cases", 1),),
        required_cross_checks=(),
        expected_artifacts=(),
        acceptance_predicates=("stored",),
    )


def _packet(_: CalculationContract) -> CalculationResult:
    return CalculationResult.create(
        outcome="constructed",
        summary="neutral replay fixture completed",
        artifacts=(_hash("artifact"),),
    )


def run_example() -> dict[str, object]:
    """Serialize/reload one immutable receipt and show exact and stale replay."""
    original = run_calculation(
        _contract(_hash("input")),
        _packet,
        environment=ExecutionEnvironment.create(platform="original"),
    )
    reloaded = cast(
        ResultReceipt,
        result_receipt_registry().from_record(result_receipt_record(original)),
    )
    if reloaded.canonical_bytes() != original.canonical_bytes():
        raise RuntimeError("immutable receipt round trip failed")
    rerun = run_calculation(
        _contract(_hash("input")),
        _packet,
        environment=ExecutionEnvironment.create(platform="rerun"),
    )
    exact = verify_replay(
        reloaded,
        resolver=ReplayResolver.create(
            receipts=(reloaded, rerun),
            reruns=((reloaded.semantic_hash, rerun.semantic_hash),),
        ),
    )
    changed = run_calculation(
        _contract(_hash("changed-input")),
        _packet,
        environment=ExecutionEnvironment.create(platform="changed"),
    )
    stale = verify_replay(
        reloaded,
        resolver=ReplayResolver.create(
            receipts=(reloaded, changed),
            reruns=((reloaded.semantic_hash, changed.semantic_hash),),
        ),
    )
    return {
        "round_trip": True,
        "exact_replay": exact.status,
        "stale_replay": stale.status,
        "receipt_hash": str(reloaded.semantic_hash),
    }


if __name__ == "__main__":
    print(run_example())
