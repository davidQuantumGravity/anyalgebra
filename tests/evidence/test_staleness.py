"""V00-078 tests: dependency changes invalidate only reverse-reachable receipts."""

from __future__ import annotations

from typing import Any, cast

import pytest

from anyalgebra.core.parents import SemanticHash
from anyalgebra.evidence.contracts import CalculationContract
from anyalgebra.evidence.run import (
    CalculationResult,
    ExecutionEnvironment,
    ResultReceipt,
    run_calculation,
)
from anyalgebra.evidence.staleness import (
    DependencyEdge,
    StalenessError,
    StalenessGraph,
    StalenessReport,
    propagate_staleness,
)
import anyalgebra.evidence.staleness as staleness_module
import anyalgebra.evidence.run as run_module
from anyalgebra.persistence.registry import SchemaError


def _hash(character: str) -> SemanticHash:
    return SemanticHash("sha256", character * 64)


def _receipt(
    label: str,
    *,
    dependencies: tuple[SemanticHash, ...] = (),
    outcome: str = "constructed",
    input_hash: SemanticHash | None = None,
) -> ResultReceipt:
    contract = CalculationContract.create(
        contract_name="staleness-test",
        project_id="test.project",
        question="test question",
        acceptable_outcomes=(outcome,),
        input_hashes=(("input", _hash(label) if input_hash is None else input_hash),),
        source_anchors=(),
        convention_manifests=(),
        algorithms=(("reference", "1"),),
        backend_request="exact",
        backend_name="reference",
        backend_version="1",
        assumptions=(),
        theorem_hypotheses=(),
        bounds=(("cases", 8),),
        required_cross_checks=(),
        expected_artifacts=(),
        acceptance_predicates=("stored",),
    )
    return run_calculation(
        contract,
        lambda _: CalculationResult.create(
            outcome=outcome,
            summary="fixture outcome",
            dependency_receipts=dependencies,
            witnesses=(_hash("f"),) if outcome == "counterexample" else (),
            remaining_branches=("unexplored",) if outcome == "inconclusive" else (),
        ),
    )


def test_changed_external_root_marks_only_reverse_reachable_receipts() -> None:
    root = _hash("b")
    left = _receipt("c", dependencies=(root,))
    right = _receipt("d")
    leaf = _receipt("e", dependencies=(left.semantic_hash,))

    report = propagate_staleness(
        StalenessGraph.create(receipts=(leaf, right, left)), changed_hashes=(root,)
    )

    assert report.affected_receipts == tuple(
        sorted((left.semantic_hash, leaf.semantic_hash), key=str)
    )
    assert report.invalidating_paths == (
        (root, left.semantic_hash),
        (root, left.semantic_hash, leaf.semantic_hash),
    )


def test_dag_fan_in_fan_out_uses_shortest_paths_and_only_causal_edges() -> None:
    root = _hash("a")
    first = _receipt("b", dependencies=(root,))
    second = _receipt("c", dependencies=(root,))
    joined = _receipt("d", dependencies=(first.semantic_hash, second.semantic_hash))
    unrelated = _receipt("e")
    graph = StalenessGraph.create(receipts=(unrelated, joined, second, first))

    report = propagate_staleness(graph, changed_hashes=(root,))

    assert report.affected_receipts == tuple(
        sorted(
            (first.semantic_hash, second.semantic_hash, joined.semantic_hash), key=str
        )
    )
    assert set(report.invalidating_edges) == {
        DependencyEdge.create(root, first.semantic_hash),
        DependencyEdge.create(root, second.semantic_hash),
        DependencyEdge.create(first.semantic_hash, joined.semantic_hash),
        DependencyEdge.create(second.semantic_hash, joined.semantic_hash),
    }
    joined_paths = [
        path for path in report.invalidating_paths if path[-1] == joined.semantic_hash
    ]
    assert len(joined_paths) == 1
    assert joined_paths[0] == (
        root,
        min((first.semantic_hash, second.semantic_hash), key=str),
        joined.semantic_hash,
    )
    assert unrelated.semantic_hash not in report.affected_receipts


def test_multiple_changed_roots_converge_with_complete_canonical_witnesses() -> None:
    left_root, right_root = _hash("a"), _hash("b")
    left = _receipt("c", dependencies=(left_root,))
    right = _receipt("d", dependencies=(right_root,))
    joined = _receipt("e", dependencies=(left.semantic_hash, right.semantic_hash))

    report = propagate_staleness(
        StalenessGraph.create(receipts=(joined, right, left)),
        changed_hashes=(right_root, left_root),
    )

    assert len(report.invalidating_paths) == 4
    assert {path[0] for path in report.invalidating_paths} == {left_root, right_root}
    assert {path[-1] for path in report.invalidating_paths} == {
        left.semantic_hash,
        right.semantic_hash,
        joined.semantic_hash,
    }


@pytest.mark.parametrize(
    "outcome", ("counterexample", "inconclusive", "reproduction_only")
)
def test_negative_and_inconclusive_receipts_propagate_without_mutation(
    outcome: str,
) -> None:
    root = _hash("a")
    receipt = _receipt("b", dependencies=(root,), outcome=outcome)
    before = receipt.canonical_bytes()

    report = propagate_staleness(
        StalenessGraph.create(receipts=(receipt,)), changed_hashes=(root,)
    )

    assert report.affected_receipts == (receipt.semantic_hash,)
    assert receipt.canonical_bytes() == before
    assert receipt.mathematical_outcome == outcome


def test_historical_supersedes_is_not_a_dependency_edge() -> None:
    old = _receipt("a")
    contract = _receipt("b").contract
    corrected = run_module._build_receipt(
        contract=contract,
        environment=ExecutionEnvironment.create(),
        started_at="2026-07-24T12:00:00Z",
        finished_at="2026-07-24T12:00:00Z",
        execution_status="completed",
        outcome="constructed",
        summary="corrected fixture",
        supersedes=old.semantic_hash,
    )

    graph = StalenessGraph.create(receipts=(corrected,))
    report = propagate_staleness(graph, changed_hashes=(old.semantic_hash,))

    assert graph.edges == ()
    assert report.affected_receipts == ()
    assert report.invalidating_paths == ()


def test_external_intermediates_and_missing_roots_are_deterministic() -> None:
    root, intermediate = _hash("a"), _hash("b")
    receipt = _receipt("c", dependencies=(intermediate,))
    graph = StalenessGraph.create(
        receipts=(receipt,), edges=(DependencyEdge.create(root, intermediate),)
    )

    first = propagate_staleness(graph, changed_hashes=(root,))
    second = propagate_staleness(graph, changed_hashes=(root,))

    assert first == second
    assert first.semantic_hash == second.semantic_hash
    assert first.invalidating_paths == ((root, intermediate, receipt.semantic_hash),)
    assert first.affected_receipts == (receipt.semantic_hash,)


def test_empty_duplicate_and_changed_receipt_boundaries_are_closed() -> None:
    root = _hash("a")
    empty = StalenessGraph.create(receipts=())
    with pytest.raises(StalenessError, match="must not be empty"):
        propagate_staleness(empty, changed_hashes=())
    receipt = _receipt("b")
    with pytest.raises(StalenessError, match="duplicate"):
        StalenessGraph.create(receipts=(receipt, receipt))
    report = propagate_staleness(
        StalenessGraph.create(receipts=(receipt,)),
        changed_hashes=(receipt.semantic_hash,),
    )
    assert report.changed_hashes == (receipt.semantic_hash,)
    assert report.affected_receipts == ()
    with pytest.raises(StalenessError, match="duplicate"):
        propagate_staleness(empty, changed_hashes=(root, root))
    with pytest.raises(StalenessError, match="resource limit"):
        propagate_staleness(
            empty,
            changed_hashes=tuple(
                SemanticHash("sha256", f"{index:064x}") for index in range(257)
            ),
        )
    with pytest.raises(TypeError):
        type("ChildGraph", (StalenessGraph,), {})


def test_graph_and_report_are_factory_only_without_receipt_mutation() -> None:
    root = _hash("a")
    receipt = _receipt("b", dependencies=(root,))
    before = receipt.canonical_bytes()
    graph = StalenessGraph.create(receipts=(receipt,))
    report = propagate_staleness(graph, changed_hashes=(root,))

    forged = {
        "schemaType": "anyalgebra.evidence.staleness_graph",
        "schemaVersion": 1,
        "receiptHashPages": [
            [{"algorithm": "sha256", "digest": receipt.semantic_hash.digest}]
        ],
        "edgePages": [
            [
                {
                    "prerequisite": {"algorithm": "sha256", "digest": root.digest},
                    "dependent": {
                        "algorithm": "sha256",
                        "digest": receipt.semantic_hash.digest,
                    },
                }
            ]
        ],
    }
    assert not hasattr(staleness_module, "staleness_graph_record")
    assert not hasattr(staleness_module, "staleness_graph_registry")
    with pytest.raises(SchemaError, match="parser_failed"):
        staleness_module._GRAPH_REGISTRY.from_record(forged)
    forged_report = {
        "schemaType": "anyalgebra.evidence.staleness_report",
        "schemaVersion": 1,
        "graphHash": {"algorithm": "sha256", "digest": graph.semantic_hash.digest},
        "changedHashPages": [[{"algorithm": "sha256", "digest": root.digest}]],
        "affectedReceiptPages": [],
        "invalidatingEdgePages": [],
        "invalidatingPathPages": [],
    }
    assert not hasattr(staleness_module, "staleness_report_record")
    assert not hasattr(staleness_module, "staleness_report_registry")
    with pytest.raises(SchemaError, match="parser_failed"):
        staleness_module._REPORT_REGISTRY.from_record(forged_report)
    assert receipt.canonical_bytes() == before
    assert not hasattr(report, "evidence_tier")
    assert not hasattr(report, "mathematical_outcome")


def test_graph_and_report_have_fixed_canonical_serialization_goldens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_module, "_utc_now", lambda: "2026-07-24T12:00:00Z")
    root = _hash("a")
    receipt = _receipt("b", dependencies=(root,))
    graph = StalenessGraph.create(receipts=(receipt,))
    report = propagate_staleness(graph, changed_hashes=(root,))

    assert graph.canonical_bytes() == (
        b'{"edgePages":[[{"dependent":{"algorithm":"sha256","digest":"82119a9224d4a7f14f0cae9ca127ea89ed84af5dcef005081d78c4d3046b92d5"},'
        b'"prerequisite":{"algorithm":"sha256","digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}]],'
        b'"receiptHashPages":[[{"algorithm":"sha256","digest":"82119a9224d4a7f14f0cae9ca127ea89ed84af5dcef005081d78c4d3046b92d5"}]],'
        b'"schemaType":"anyalgebra.evidence.staleness_graph","schemaVersion":1}'
    )
    assert graph.semantic_hash.digest == (
        "b4c3a30e5d7e16ae9036a45cf055b5c5757571a64c9c26a905b6675015d326e4"
    )
    assert report.canonical_bytes() == (
        b'{"affectedReceiptPages":[[{"algorithm":"sha256","digest":"82119a9224d4a7f14f0cae9ca127ea89ed84af5dcef005081d78c4d3046b92d5"}]],'
        b'"changedHashPages":[[{"algorithm":"sha256","digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]],'
        b'"graphHash":{"algorithm":"sha256","digest":"b4c3a30e5d7e16ae9036a45cf055b5c5757571a64c9c26a905b6675015d326e4"},'
        b'"invalidatingEdgePages":[[{"dependent":{"algorithm":"sha256","digest":"82119a9224d4a7f14f0cae9ca127ea89ed84af5dcef005081d78c4d3046b92d5"},'
        b'"prerequisite":{"algorithm":"sha256","digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}]],'
        b'"invalidatingPathPages":[[[[{"algorithm":"sha256","digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},'
        b'{"algorithm":"sha256","digest":"82119a9224d4a7f14f0cae9ca127ea89ed84af5dcef005081d78c4d3046b92d5"}]]]],'
        b'"schemaType":"anyalgebra.evidence.staleness_report","schemaVersion":1}'
    )
    assert report.semantic_hash.digest == (
        "cbf7a861b0304740db921027ba2a4ad6fd2c5490ef985baf9c3f8ef84e2ecd95"
    )


def test_cycle_fabricated_prerequisite_and_tampering_are_rejected() -> None:
    first, second, external = _hash("a"), _hash("b"), _hash("c")
    with pytest.raises(StalenessError, match="cycle"):
        StalenessGraph.create(
            receipts=(),
            edges=(
                DependencyEdge.create(first, second),
                DependencyEdge.create(second, first),
            ),
        )
    receipt = _receipt("d")
    with pytest.raises(StalenessError, match="undeclared"):
        StalenessGraph.create(
            receipts=(receipt,),
            edges=(DependencyEdge.create(external, receipt.semantic_hash),),
        )
    graph = StalenessGraph.create(receipts=(receipt,))
    object.__setattr__(graph, "edges", (DependencyEdge.create(first, second),))
    with pytest.raises(StalenessError, match="integrity"):
        propagate_staleness(graph, changed_hashes=(first,))
    clean = StalenessGraph.create(receipts=(receipt,))
    object.__setattr__(clean, "receipt_hashes", list(clean.receipt_hashes))
    with pytest.raises(StalenessError, match="integrity"):
        propagate_staleness(clean, changed_hashes=(first,))
    report = propagate_staleness(
        StalenessGraph.create(receipts=(receipt,)), changed_hashes=(first,)
    )
    object.__setattr__(report, "changed_hashes", list(report.changed_hashes))
    with pytest.raises(StalenessError, match="integrity"):
        report.canonical_bytes()


def test_limits_fail_closed_and_interrupts_propagate() -> None:
    hashes = tuple(SemanticHash("sha256", f"{index:064x}") for index in range(256))
    edges = tuple(
        DependencyEdge.create(hashes[index], hashes[index + 1])
        for index in range(len(hashes) - 1)
    )
    graph = StalenessGraph.create(receipts=(), edges=edges)
    assert len(graph.edges) == 255
    with pytest.raises(StalenessError, match="node resource"):
        StalenessGraph.create(
            receipts=(), edges=(*edges, DependencyEdge.create(hashes[-1], _hash("f")))
        )
    report_hashes = tuple(
        SemanticHash("sha256", f"{index + 256:064x}") for index in range(257)
    )
    with pytest.raises(StalenessError, match="durable record resource"):
        staleness_module._build_report(
            graph_hash=_hash("e"),
            changed_hashes=(report_hashes[0],),
            affected_receipts=report_hashes[1:],
            invalidating_edges=tuple(
                DependencyEdge.create(report_hashes[0], item)
                for item in report_hashes[1:]
            ),
            invalidating_paths=tuple(
                (report_hashes[0], item) for item in report_hashes[1:]
            ),
        )

    class Interrupted:
        def __iter__(self) -> Interrupted:
            return self

        def __next__(self) -> object:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        StalenessGraph.create(receipts=Interrupted())

    class Broken:
        def __iter__(self) -> Broken:
            return self

        def __next__(self) -> object:
            raise RuntimeError("secret")

    class Exiting(Broken):
        def __next__(self) -> object:
            raise SystemExit

    with pytest.raises(StalenessError) as raised:
        StalenessGraph.create(receipts=Broken())
    assert "secret" not in str(raised.value)
    with pytest.raises(SystemExit):
        StalenessGraph.create(receipts=Exiting())


def test_factories_are_sealed_and_representations_are_sanitized() -> None:
    with pytest.raises(StalenessError):
        DependencyEdge()
    with pytest.raises(StalenessError):
        StalenessGraph()
    with pytest.raises(StalenessError):
        StalenessReport()
    with pytest.raises(TypeError):
        type("ChildEdge", (DependencyEdge,), {})


@pytest.mark.parametrize(
    "receipts,edges",
    (
        ("bad", ()),
        ((object(),), ()),
        ((), (object(),)),
        ((), (DependencyEdge.create(_hash("a"), _hash("b")),) * 2),
    ),
)
def test_public_graph_boundary_rejects_hostile_shapes(
    receipts: object, edges: object
) -> None:
    with pytest.raises(StalenessError):
        StalenessGraph.create(receipts=receipts, edges=edges)


def test_public_identity_equality_hash_and_canonical_interfaces() -> None:
    root = _hash("a")
    receipt = _receipt("b", dependencies=(root,))
    first = StalenessGraph.create(receipts=(receipt,))
    second = StalenessGraph.create(receipts=(receipt,))
    report = propagate_staleness(first, changed_hashes=(root,))

    assert first == second
    assert first != object()
    assert hash(first) == hash(second)
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.graph_id.endswith(first.semantic_hash.digest)
    assert "receipt_count=1" in repr(first)
    assert hash(report)
    assert report.report_id.endswith(report.semantic_hash.digest)
    assert "affected_count=1" in repr(report)


@pytest.mark.parametrize("factory", (DependencyEdge.create,))
def test_edge_hash_and_self_boundary(factory: object) -> None:
    with pytest.raises(StalenessError):
        factory(_hash("a"), _hash("a"))  # type: ignore[operator]
    with pytest.raises(StalenessError):
        factory("bad", _hash("a"))  # type: ignore[operator]


def test_exact_types_and_hostile_iterables_fail_closed() -> None:
    class BadIter:
        def __iter__(self) -> object:
            raise RuntimeError("secret")

    mutated = _hash("a")
    object.__setattr__(mutated, "digest", "not-a-hash")
    with pytest.raises(StalenessError, match="iterable"):
        StalenessGraph.create(receipts=BadIter())
    with pytest.raises(StalenessError, match="invalid semantic hash"):
        staleness_module._hash(mutated, "hash")

    receipt = _receipt("b")
    object.__setattr__(receipt, "semantic_hash", mutated)
    with pytest.raises(StalenessError, match="integrity"):
        StalenessGraph.create(receipts=(receipt,))

    edge = DependencyEdge.create(_hash("c"), _hash("d"))
    assert repr(edge) == "DependencyEdge(<content-addressed endpoints>)"
    object.__setattr__(edge, "dependent", mutated)
    with pytest.raises(StalenessError, match="integrity"):
        staleness_module._edge(edge, "edge")


def test_sealed_factories_and_invalid_representations_are_closed() -> None:
    with pytest.raises(StalenessError, match="exact DependencyEdge"):
        cast(Any, DependencyEdge.create).__func__(object, _hash("a"), _hash("b"))
    with pytest.raises(StalenessError, match="exact StalenessGraph"):
        cast(Any, StalenessGraph.create).__func__(object, receipts=())
    with pytest.raises(TypeError):
        type("ChildReport", (StalenessReport,), {})

    graph = StalenessGraph.create(receipts=())
    object.__setattr__(graph, "semantic_hash", _hash("b"))
    assert repr(graph) == "StalenessGraph(<invalid>)"
    with pytest.raises(StalenessError, match="exact StalenessGraph"):
        staleness_module._ensure_graph(object())

    report = staleness_module._build_report(
        graph_hash=_hash("c"),
        changed_hashes=(_hash("d"),),
        affected_receipts=(),
        invalidating_edges=(),
        invalidating_paths=(),
    )
    object.__setattr__(report, "semantic_hash", _hash("e"))
    assert repr(report) == "StalenessReport(<invalid>)"
    with pytest.raises(StalenessError, match="exact StalenessReport"):
        staleness_module._ensure_report(object())


def test_graph_hash_boundaries_and_private_graph_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hashes = tuple(SemanticHash("sha256", f"{index:064x}") for index in range(257))
    edges = tuple(
        DependencyEdge.create(hashes[index], hashes[index + 1])
        for index in range(len(hashes) - 1)
    )
    with pytest.raises(StalenessError, match="node resource"):
        staleness_module._build_graph(receipt_hashes=(), edges=edges)
    with pytest.raises(StalenessError, match="cycle"):
        staleness_module._build_graph(
            receipt_hashes=(),
            edges=(
                DependencyEdge.create(_hash("a"), _hash("b")),
                DependencyEdge.create(_hash("b"), _hash("a")),
            ),
        )

    graph = StalenessGraph.create(receipts=())
    monkeypatch.setattr(
        staleness_module,
        "canonical_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("secret")),
    )
    with pytest.raises(StalenessError, match="cannot derive"):
        staleness_module._graph_hash(graph)
    monkeypatch.setattr(
        staleness_module,
        "_graph_body",
        lambda value: (_ for _ in ()).throw(StalenessError("graph", "blocked")),
    )
    with pytest.raises(StalenessError, match="blocked"):
        staleness_module._graph_hash(graph)


def test_report_validation_rejects_every_causal_invariant() -> None:
    root, middle, leaf, extra = (_hash(character) for character in "abcd")
    edge_one = DependencyEdge.create(root, middle)
    edge_two = DependencyEdge.create(middle, leaf)
    kwargs = {
        "graph_hash": _hash("e"),
        "changed_hashes": (root,),
        "affected_receipts": (leaf,),
        "invalidating_edges": (edge_one, edge_two),
        "invalidating_paths": ((root, middle, leaf),),
    }
    valid = staleness_module._build_report(**kwargs)
    assert valid.affected_receipts == (leaf,)

    cases = (
        {"invalidating_paths": ((root, root, leaf),)},
        {"invalidating_paths": ((root,),)},
        {"invalidating_paths": ((root, middle, leaf),) * 2},
        {"affected_receipts": (middle,)},
        {"invalidating_edges": (edge_one,)},
        {
            "invalidating_edges": (
                edge_one,
                edge_two,
                DependencyEdge.create(extra, _hash("f")),
            )
        },
        {
            "invalidating_edges": (
                DependencyEdge.create(root, extra),
                DependencyEdge.create(extra, middle),
                edge_two,
            ),
            "invalidating_paths": ((extra, middle, leaf),),
        },
        {
            "invalidating_edges": (
                edge_one,
                edge_two,
                DependencyEdge.create(leaf, root),
            )
        },
        {
            "invalidating_edges": (
                edge_one,
                edge_two,
                DependencyEdge.create(root, extra),
                DependencyEdge.create(extra, leaf),
            ),
            "invalidating_paths": ((root, extra, leaf),),
        },
    )
    for changes in cases:
        with pytest.raises(StalenessError):
            staleness_module._build_report(**{**kwargs, **changes})


def test_report_hash_and_tampered_report_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, leaf = _hash("a"), _hash("b")
    report = staleness_module._build_report(
        graph_hash=_hash("c"),
        changed_hashes=(root,),
        affected_receipts=(leaf,),
        invalidating_edges=(DependencyEdge.create(root, leaf),),
        invalidating_paths=((root, leaf),),
    )
    monkeypatch.setattr(
        staleness_module,
        "canonical_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(TypeError("secret")),
    )
    with pytest.raises(StalenessError, match="cannot derive"):
        staleness_module._report_hash(report)
    monkeypatch.undo()

    for name, value in (
        ("graph_hash", "bad"),
        ("affected_receipts", (root,)),
        ("invalidating_edges", ()),
        ("invalidating_paths", ((root,),)),
    ):
        fresh = staleness_module._build_report(
            graph_hash=_hash("c"),
            changed_hashes=(root,),
            affected_receipts=(leaf,),
            invalidating_edges=(DependencyEdge.create(root, leaf),),
            invalidating_paths=((root, leaf),),
        )
        object.__setattr__(fresh, name, value)
        with pytest.raises(StalenessError, match="integrity"):
            fresh.canonical_bytes()


def test_propagation_ties_dead_edges_path_limit_and_noop_root() -> None:
    root, left, right, dead = (_hash(character) for character in "abcd")
    leaf = _receipt("e", dependencies=(left, right))
    graph = StalenessGraph.create(
        receipts=(leaf,),
        edges=(
            DependencyEdge.create(root, left),
            DependencyEdge.create(root, right),
            DependencyEdge.create(root, dead),
        ),
    )
    report = propagate_staleness(graph, changed_hashes=(root,))
    assert report.invalidating_paths == ((root, left, leaf.semantic_hash),)
    assert report.invalidating_edges == (
        DependencyEdge.create(root, left),
        DependencyEdge.create(root, right),
        DependencyEdge.create(left, leaf.semantic_hash),
        DependencyEdge.create(right, leaf.semantic_hash),
    )
    no_op = propagate_staleness(graph, changed_hashes=(_hash("f"),))
    assert no_op.affected_receipts == ()
    assert no_op.invalidating_edges == ()

    inputs = tuple(SemanticHash("sha256", f"{index:064x}") for index in range(127))
    roots = tuple(SemanticHash("sha256", f"{index + 127:064x}") for index in range(128))
    intermediate = _hash("f")
    leaves = tuple(
        _receipt("a", dependencies=(intermediate,), input_hash=input_hash)
        for input_hash in inputs
    )
    saturated = StalenessGraph.create(
        receipts=leaves,
        edges=tuple(
            DependencyEdge.create(root_hash, intermediate) for root_hash in roots
        ),
    )
    assert len(saturated.edges) == 255
    with pytest.raises(StalenessError, match="path resource"):
        propagate_staleness(saturated, changed_hashes=roots)


def test_graph_edge_resource_caps_preflight_before_node_accounting() -> None:
    hashes = tuple(SemanticHash("sha256", f"{index:064x}") for index in range(773))
    many_dependencies = _receipt("a", dependencies=hashes[:256], input_hash=hashes[256])
    one_more_dependency = _receipt(
        "b", dependencies=(hashes[257],), input_hash=hashes[258]
    )
    with pytest.raises(StalenessError, match="resource limit"):
        StalenessGraph.create(receipts=(many_dependencies, one_more_dependency))

    receipt = _receipt("c", dependencies=(hashes[259],), input_hash=hashes[260])
    supplied = tuple(
        DependencyEdge.create(hashes[index + 261], hashes[index + 517])
        for index in range(256)
    )
    with pytest.raises(StalenessError, match="resource limit"):
        StalenessGraph.create(receipts=(receipt,), edges=supplied)


def test_receipt_integrity_rejects_a_valid_but_nonidentical_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original, replacement = _receipt("a"), _receipt("b")

    class Registry:
        def from_record(self, record: object) -> ResultReceipt:
            del record
            return replacement

    monkeypatch.setattr(
        staleness_module,
        "result_receipt_registry",
        Registry,
    )
    with pytest.raises(StalenessError, match="integrity"):
        staleness_module._receipt(original, "receipt")
