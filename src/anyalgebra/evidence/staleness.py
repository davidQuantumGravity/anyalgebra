"""Read-only transitive dependency staleness for immutable result receipts.

Edges point from a prerequisite content hash to the receipt which declared it
in ``dependency_receipts``.  ``supersedes`` is deliberately not an edge: it is
historical correction metadata, not a semantic prerequisite.  This module
neither executes work nor changes a receipt, outcome, tier, or claim.

Graphs and reports are factory-only live diagnostics in v0.0.  Their canonical
bytes are one-way content fingerprints, not import formats: hashes alone cannot
prove the referenced receipts or graph were actually resolved.  Trusted
deserialization requires a later receipt/graph resolver-binding API.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from itertools import pairwise
from dataclasses import dataclass
from typing import cast

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import SemanticHash
from anyalgebra.evidence.run import (
    ReceiptError,
    ResultReceipt,
    result_receipt_canonical_bytes,
    result_receipt_record,
    result_receipt_registry,
)
from anyalgebra.persistence.canonical import canonical_json
from anyalgebra.persistence.registry import JSONValue, SchemaError, SerializerRegistry


_GRAPH_TAG = "anyalgebra.evidence.staleness_graph"
_REPORT_TAG = "anyalgebra.evidence.staleness_report"
_VERSION = 1
# The shared safe serializer admits at most 4,096 JSON nodes.  A graph edge
# consumes several nodes in a canonical graph fingerprint, so 256 is the
# largest simple v0.0 graph without a separate compressed codec.
_MAX_NODES = 256
_MAX_EDGES = 256
_MAX_PATHS = 256


class StalenessError(AnyAlgebraError, ValueError):
    """A closed diagnostic from the non-executing graph boundary."""

    def __init__(self, field: str, reason: str) -> None:
        self.field, self.reason = field, reason
        super().__init__(f"invalid staleness {field}: {reason}")


def _snapshot(value: object, field: str, maximum: int) -> tuple[object, ...]:
    if type(value) is str:
        raise StalenessError(field, "must be an iterable")
    try:
        iterator = iter(cast(Iterable[object], value))
    except Exception as error:
        raise StalenessError(field, "must be an iterable") from error
    items: list[object] = []
    try:
        for _ in range(maximum + 1):
            items.append(next(iterator))
    except StopIteration:
        return tuple(items)
    except Exception as error:
        raise StalenessError(field, "iterable snapshot failed") from error
    raise StalenessError(field, "resource limit exceeded")


def _hash(value: object, field: str) -> SemanticHash:
    if type(value) is not SemanticHash:
        raise StalenessError(field, "must be an exact SemanticHash")
    try:
        return SemanticHash(value.algorithm, value.digest)
    except Exception as error:
        raise StalenessError(field, "has an invalid semantic hash") from error


def _hashes(
    value: object, field: str, maximum: int = _MAX_NODES
) -> tuple[SemanticHash, ...]:
    result = tuple(_hash(item, field) for item in _snapshot(value, field, maximum))
    if len(set(result)) != len(result):
        raise StalenessError(field, "contains duplicate content addresses")
    return tuple(sorted(result, key=str))


def _receipt(value: object, field: str) -> ResultReceipt:
    if type(value) is not ResultReceipt:
        raise StalenessError(field, "must contain exact ResultReceipt values")
    try:
        parsed = result_receipt_registry().from_record(result_receipt_record(value))
        if type(parsed) is not ResultReceipt or result_receipt_canonical_bytes(
            parsed
        ) != result_receipt_canonical_bytes(value):
            raise StalenessError(field, "content address integrity rejected")
    except (ReceiptError, SchemaError, TypeError, ValueError, AttributeError) as error:
        raise StalenessError(field, "content address integrity rejected") from error
    return parsed


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class DependencyEdge:
    """One exact prerequisite-to-dependent edge, independent of receipt bytes."""

    prerequisite: SemanticHash
    dependent: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise StalenessError("dependency_edge", "is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("DependencyEdge cannot be subclassed")

    @classmethod
    def create(cls, prerequisite: object, dependent: object) -> DependencyEdge:
        if cls is not DependencyEdge:
            raise StalenessError(
                "dependency_edge", "factory requires exact DependencyEdge"
            )
        left, right = _hash(prerequisite, "prerequisite"), _hash(dependent, "dependent")
        if left == right:
            raise StalenessError("dependency_edge", "must not be a self edge")
        value = object.__new__(DependencyEdge)
        object.__setattr__(value, "prerequisite", left)
        object.__setattr__(value, "dependent", right)
        return value

    def __repr__(self) -> str:
        return "DependencyEdge(<content-addressed endpoints>)"

    def __eq__(self, other: object) -> bool:
        return type(other) is DependencyEdge and (
            self.prerequisite,
            self.dependent,
        ) == (other.prerequisite, other.dependent)

    def __hash__(self) -> int:
        return hash((self.prerequisite, self.dependent))


def _edge(value: object, field: str) -> DependencyEdge:
    if type(value) is not DependencyEdge:
        raise StalenessError(field, "must contain exact DependencyEdge values")
    try:
        return DependencyEdge.create(value.prerequisite, value.dependent)
    except (AttributeError, TypeError, ValueError) as error:
        raise StalenessError(field, "edge integrity rejected") from error


def _edges(value: object, field: str) -> tuple[DependencyEdge, ...]:
    result = tuple(_edge(item, field) for item in _snapshot(value, field, _MAX_EDGES))
    pairs = {(item.prerequisite, item.dependent) for item in result}
    if len(pairs) != len(result):
        raise StalenessError(field, "contains duplicate edges")
    return tuple(
        sorted(result, key=lambda item: (str(item.prerequisite), str(item.dependent)))
    )


def _edge_cycle(edges: tuple[DependencyEdge, ...]) -> bool:
    """Use iterative topological elimination; long chains never recurse."""
    successors: dict[SemanticHash, tuple[SemanticHash, ...]] = {}
    indegree: dict[SemanticHash, int] = {}
    for edge in edges:
        successors[edge.prerequisite] = (
            *successors.get(edge.prerequisite, ()),
            edge.dependent,
        )
        indegree[edge.prerequisite] = indegree.get(edge.prerequisite, 0)
        indegree[edge.dependent] = indegree.get(edge.dependent, 0) + 1
    pending = sorted((node for node, count in indegree.items() if not count), key=str)
    visited = 0
    while pending:
        node = pending.pop(0)
        visited += 1
        for child in successors.get(node, ()):
            indegree[child] -= 1
            if not indegree[child]:
                pending.append(child)
    return visited != len(indegree)


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class StalenessGraph:
    """Sealed DAG snapshot of receipt dependencies plus explicit hash edges.

    Explicit edges make external roots and representation-boundary cycle checks
    testable.  They are only graph declarations; they cannot execute or alter
    any receipt.
    """

    receipt_hashes: tuple[SemanticHash, ...]
    edges: tuple[DependencyEdge, ...]
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise StalenessError("staleness_graph", "is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("StalenessGraph cannot be subclassed")

    @classmethod
    def create(cls, *, receipts: object, edges: object = ()) -> StalenessGraph:
        if cls is not StalenessGraph:
            raise StalenessError(
                "staleness_graph", "factory requires exact StalenessGraph"
            )
        checked_receipts = tuple(
            _receipt(item, "receipts")
            for item in _snapshot(receipts, "receipts", _MAX_NODES)
        )
        if len({item.semantic_hash for item in checked_receipts}) != len(
            checked_receipts
        ):
            raise StalenessError("receipts", "contains duplicate content addresses")
        checked_receipts = tuple(
            sorted(checked_receipts, key=lambda item: str(item.semantic_hash))
        )
        direct_pairs: list[tuple[SemanticHash, SemanticHash]] = []
        for receipt in checked_receipts:
            for reference in receipt.dependency_receipts:
                direct_pairs.append((reference, receipt.semantic_hash))
                if len(direct_pairs) > _MAX_EDGES:
                    raise StalenessError("edges", "resource limit exceeded")
        direct = tuple(DependencyEdge.create(*pair) for pair in direct_pairs)
        supplied = _edges(edges, "edges")
        receipt_hashes = tuple(item.semantic_hash for item in checked_receipts)
        direct_set = {(item.prerequisite, item.dependent) for item in direct}
        if any(
            edge.dependent in receipt_hashes
            and (edge.prerequisite, edge.dependent) not in direct_set
            for edge in supplied
        ):
            raise StalenessError(
                "edges", "cannot add an undeclared receipt prerequisite"
            )
        merged = tuple(
            DependencyEdge.create(prerequisite, dependent)
            for prerequisite, dependent in sorted(
                {(item.prerequisite, item.dependent) for item in (*direct, *supplied)},
                key=lambda item: (str(item[0]), str(item[1])),
            )
        )
        if len(merged) > _MAX_EDGES:
            raise StalenessError("edges", "resource limit exceeded")
        nodes = set(receipt_hashes)
        nodes.update(edge.prerequisite for edge in merged)
        nodes.update(edge.dependent for edge in merged)
        if len(nodes) > _MAX_NODES:
            raise StalenessError("edges", "node resource limit exceeded")
        if _edge_cycle(merged):
            raise StalenessError("edges", "contains a dependency cycle")
        value = object.__new__(StalenessGraph)
        object.__setattr__(value, "receipt_hashes", receipt_hashes)
        object.__setattr__(value, "edges", merged)
        object.__setattr__(value, "semantic_hash", SemanticHash("sha256", "0" * 64))
        object.__setattr__(value, "semantic_hash", _graph_hash(value))
        return value

    @property
    def graph_id(self) -> str:
        _ensure_graph(self)
        return f"staleness-graph:sha256:{self.semantic_hash.digest}"

    def canonical_bytes(self) -> bytes:
        _ensure_graph(self)
        return staleness_graph_canonical_bytes(self)

    def __repr__(self) -> str:
        try:
            _ensure_graph(self)
            return (
                f"StalenessGraph(receipt_count={len(self.receipt_hashes)}, "
                f"edge_count={len(self.edges)})"
            )
        except Exception:
            return "StalenessGraph(<invalid>)"

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is StalenessGraph
            and self.canonical_bytes() == other.canonical_bytes()
        )

    def __hash__(self) -> int:
        _ensure_graph(self)
        return int.from_bytes(
            bytes.fromhex(self.semantic_hash.digest[:16]), "big", signed=True
        )


def _hash_record(value: SemanticHash) -> dict[str, str]:
    return {"algorithm": value.algorithm, "digest": value.digest}


def _edge_record(value: DependencyEdge) -> dict[str, object]:
    return {
        "prerequisite": _hash_record(value.prerequisite),
        "dependent": _hash_record(value.dependent),
    }


def _pages(values: tuple[object, ...]) -> list[list[object]]:
    return [list(values[index : index + 256]) for index in range(0, len(values), 256)]


def _record_nodes(value: object) -> int:
    """Count the registry snapshot nodes before canonical serialization.

    The registry's global 4,096-node bound is stricter than its per-container
    page limit.  This iterative preflight gives a stable domain diagnostic
    rather than leaking a persistence-layer failure after partial work.
    """
    pending = [value]
    count = 0
    while pending:
        item = pending.pop()
        count += 1
        if type(item) is list:
            pending.extend(item)
        elif type(item) is dict:
            pending.extend(item.values())
    return count


def _check_record_budget(value: object, field: str) -> None:
    # ``schemaType`` and ``schemaVersion`` are added by SerializerRegistry.
    if _record_nodes(value) + 2 > 4096:
        raise StalenessError(field, "durable record resource limit exceeded")


def _graph_body(value: StalenessGraph) -> dict[str, JSONValue]:
    return cast(
        dict[str, JSONValue],
        {
            "receiptHashPages": _pages(
                tuple(_hash_record(item) for item in value.receipt_hashes)
            ),
            "edgePages": _pages(tuple(_edge_record(item) for item in value.edges)),
        },
    )


def _graph_encode(value: StalenessGraph) -> dict[str, JSONValue]:
    return _graph_body(value)


def _reject_graph_record(record: object) -> StalenessGraph:
    """Keep graph construction receipt-validated and factory-only.

    A graph record contains receipt hashes but not the receipts themselves, so
    it cannot establish that an edge into a receipt was declared by that
    receipt.  The codec exists only for one-way canonical fingerprinting.
    """
    del record
    raise StalenessError("staleness_graph", "cannot be deserialized")


_GRAPH_REGISTRY = SerializerRegistry().with_codec(
    _GRAPH_TAG,
    _VERSION,
    StalenessGraph,
    _graph_encode,
    _reject_graph_record,
)


def _graph_hash(value: StalenessGraph) -> SemanticHash:
    try:
        _check_record_budget(_graph_body(value), "staleness_graph")
        return SemanticHash(
            "sha256",
            hashlib.sha256(canonical_json(value, registry=_GRAPH_REGISTRY)).hexdigest(),
        )
    except StalenessError:
        raise
    except Exception as error:
        raise StalenessError(
            "staleness_graph", "cannot derive canonical content address"
        ) from error


def _build_graph(*, receipt_hashes: object, edges: object) -> StalenessGraph:
    checked_hashes = _hashes(receipt_hashes, "receipt_hashes")
    checked_edges = _edges(edges, "edges")
    nodes = set(checked_hashes)
    nodes.update(edge.prerequisite for edge in checked_edges)
    nodes.update(edge.dependent for edge in checked_edges)
    if len(nodes) > _MAX_NODES:
        raise StalenessError("edges", "node resource limit exceeded")
    if _edge_cycle(checked_edges):
        raise StalenessError("edges", "contains a dependency cycle")
    value = object.__new__(StalenessGraph)
    object.__setattr__(value, "receipt_hashes", checked_hashes)
    object.__setattr__(value, "edges", checked_edges)
    object.__setattr__(value, "semantic_hash", SemanticHash("sha256", "0" * 64))
    object.__setattr__(value, "semantic_hash", _graph_hash(value))
    return value


def _ensure_graph(value: object) -> None:
    if type(value) is not StalenessGraph:
        raise StalenessError("staleness_graph", "must be an exact StalenessGraph")
    try:
        rebuilt = _build_graph(receipt_hashes=value.receipt_hashes, edges=value.edges)
        if (
            type(value.receipt_hashes) is not tuple
            or type(value.edges) is not tuple
            or rebuilt.receipt_hashes != value.receipt_hashes
            or rebuilt.edges != value.edges
            or _hash(value.semantic_hash, "semantic_hash") != _graph_hash(value)
        ):
            raise StalenessError(
                "staleness_graph", "content address integrity rejected"
            )
    except Exception as error:
        raise StalenessError(
            "staleness_graph", "content address integrity rejected"
        ) from error


def staleness_graph_canonical_bytes(value: object) -> bytes:
    _ensure_graph(value)
    return canonical_json(value, registry=_GRAPH_REGISTRY)


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class StalenessReport:
    """Immutable operational invalidation result; it carries no claim or tier."""

    graph_hash: SemanticHash
    changed_hashes: tuple[SemanticHash, ...]
    affected_receipts: tuple[SemanticHash, ...]
    invalidating_edges: tuple[DependencyEdge, ...]
    invalidating_paths: tuple[tuple[SemanticHash, ...], ...]
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise StalenessError("staleness_report", "is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("StalenessReport cannot be subclassed")

    @property
    def report_id(self) -> str:
        _ensure_report(self)
        return f"staleness:sha256:{self.semantic_hash.digest}"

    def canonical_bytes(self) -> bytes:
        _ensure_report(self)
        return staleness_report_canonical_bytes(self)

    def __repr__(self) -> str:
        try:
            _ensure_report(self)
            return f"StalenessReport(affected_count={len(self.affected_receipts)})"
        except Exception:
            return "StalenessReport(<invalid>)"

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is StalenessReport
            and self.canonical_bytes() == other.canonical_bytes()
        )

    def __hash__(self) -> int:
        _ensure_report(self)
        return int.from_bytes(
            bytes.fromhex(self.semantic_hash.digest[:16]), "big", signed=True
        )


def _report_body(value: StalenessReport) -> dict[str, JSONValue]:
    return cast(
        dict[str, JSONValue],
        {
            "graphHash": _hash_record(value.graph_hash),
            "changedHashPages": _pages(
                tuple(_hash_record(item) for item in value.changed_hashes)
            ),
            "affectedReceiptPages": _pages(
                tuple(_hash_record(item) for item in value.affected_receipts)
            ),
            "invalidatingEdgePages": _pages(
                tuple(_edge_record(item) for item in value.invalidating_edges)
            ),
            "invalidatingPathPages": _pages(
                tuple(
                    _pages(tuple(_hash_record(item) for item in path))
                    for path in value.invalidating_paths
                )
            ),
        },
    )


def _report_encode(value: StalenessReport) -> dict[str, JSONValue]:
    return _report_body(value)


def _reject_report_record(record: object) -> StalenessReport:
    """Keep report construction tied to a live, factory-validated graph."""
    del record
    raise StalenessError("staleness_report", "cannot be deserialized")


_REPORT_REGISTRY = SerializerRegistry().with_codec(
    _REPORT_TAG,
    _VERSION,
    StalenessReport,
    _report_encode,
    _reject_report_record,
)


def _report_hash(value: StalenessReport) -> SemanticHash:
    try:
        _check_record_budget(_report_body(value), "staleness_report")
        return SemanticHash(
            "sha256",
            hashlib.sha256(
                canonical_json(value, registry=_REPORT_REGISTRY)
            ).hexdigest(),
        )
    except StalenessError:
        raise
    except (SchemaError, TypeError, ValueError, AttributeError) as error:
        raise StalenessError(
            "staleness_report", "cannot derive canonical content address"
        ) from error


def _build_report(
    *,
    graph_hash: object,
    changed_hashes: object,
    affected_receipts: object,
    invalidating_edges: object,
    invalidating_paths: object,
) -> StalenessReport:
    graph = _hash(graph_hash, "graph_hash")
    changed = _hashes(changed_hashes, "changed_hashes")
    if not changed:
        raise StalenessError("changed_hashes", "must not be empty")
    raw_affected = tuple(
        _hash(item, "affected_receipts")
        for item in _snapshot(affected_receipts, "affected_receipts", _MAX_NODES)
    )
    affected = tuple(sorted(set(raw_affected), key=str))
    raw_edges = tuple(
        _edge(item, "invalidating_edges")
        for item in _snapshot(invalidating_edges, "invalidating_edges", _MAX_EDGES)
    )
    edges = tuple(
        DependencyEdge.create(prerequisite, dependent)
        for prerequisite, dependent in sorted(
            {(item.prerequisite, item.dependent) for item in raw_edges},
            key=lambda item: (str(item[0]), str(item[1])),
        )
    )
    raw_paths = _snapshot(invalidating_paths, "invalidating_paths", _MAX_PATHS)
    paths: list[tuple[SemanticHash, ...]] = []
    for path in raw_paths:
        checked = tuple(
            _hash(item, "invalidating_paths")
            for item in _snapshot(path, "invalidating_paths", _MAX_NODES)
        )
        if len(set(checked)) != len(checked):
            raise StalenessError("invalidating_paths", "path contains a cycle")
        if len(checked) < 2:
            raise StalenessError("invalidating_paths", "each path needs an edge")
        paths.append(checked)
    if len(set(paths)) != len(paths):
        raise StalenessError("invalidating_paths", "contains duplicate paths")
    ordered_paths = tuple(sorted(paths, key=lambda path: tuple(map(str, path))))
    if set(affected) != {path[-1] for path in ordered_paths}:
        raise StalenessError("affected_receipts", "must exactly match path endpoints")
    edge_pairs = {(edge.prerequisite, edge.dependent) for edge in edges}
    path_pairs = {
        (left, right) for path in ordered_paths for left, right in pairwise(path)
    }
    if not path_pairs <= edge_pairs:
        raise StalenessError("invalidating_edges", "must contain every path edge")
    successors: dict[SemanticHash, tuple[SemanticHash, ...]] = {}
    predecessors: dict[SemanticHash, tuple[SemanticHash, ...]] = {}
    for edge in edges:
        successors[edge.prerequisite] = (
            *successors.get(edge.prerequisite, ()),
            edge.dependent,
        )
        predecessors[edge.dependent] = (
            *predecessors.get(edge.dependent, ()),
            edge.prerequisite,
        )
    forward, pending = set(changed), list(changed)
    while pending:
        node = pending.pop(0)
        for child in successors.get(node, ()):
            if child not in forward:
                forward.add(child)
                pending.append(child)
    backward, pending = set(affected), list(affected)
    while pending:
        node = pending.pop(0)
        for parent in predecessors.get(node, ()):
            if parent not in backward:
                backward.add(parent)
                pending.append(parent)
    if edge_pairs != {
        (edge.prerequisite, edge.dependent)
        for edge in edges
        if edge.prerequisite in forward and edge.dependent in backward
    }:
        raise StalenessError("invalidating_edges", "contains an unrelated edge")
    if any(
        path[0] not in changed or path[-1] not in affected for path in ordered_paths
    ):
        raise StalenessError("invalidating_paths", "has invalid endpoints")
    if _edge_cycle(edges):
        raise StalenessError("invalidating_edges", "contains a dependency cycle")
    expected_paths: list[tuple[SemanticHash, ...]] = []
    for root in changed:
        shortest: dict[SemanticHash, tuple[SemanticHash, ...]] = {root: (root,)}
        pending = [root]
        while pending:
            node = pending.pop(0)
            path = shortest[node]
            for child in sorted(successors.get(node, ()), key=str):
                candidate = (*path, child)
                previous = shortest.get(child)
                if previous is None or (
                    len(candidate) == len(previous)
                    and tuple(map(str, candidate)) < tuple(map(str, previous))
                ):
                    shortest[child] = candidate
                    pending.append(child)
        expected_paths.extend(
            path
            for endpoint, path in shortest.items()
            if endpoint in affected and len(path) > 1
        )
    canonical_paths = tuple(
        sorted(expected_paths, key=lambda path: tuple(map(str, path)))
    )
    if ordered_paths != canonical_paths:
        raise StalenessError(
            "invalidating_paths", "must be complete canonical witnesses"
        )
    value = object.__new__(StalenessReport)
    for name, item in (
        ("graph_hash", graph),
        ("changed_hashes", changed),
        ("affected_receipts", affected),
        ("invalidating_edges", edges),
        ("invalidating_paths", ordered_paths),
        ("semantic_hash", SemanticHash("sha256", "0" * 64)),
    ):
        object.__setattr__(value, name, item)
    object.__setattr__(value, "semantic_hash", _report_hash(value))
    return value


def _ensure_report(value: object) -> None:
    if type(value) is not StalenessReport:
        raise StalenessError("staleness_report", "must be an exact StalenessReport")
    try:
        rebuilt = _build_report(
            graph_hash=value.graph_hash,
            changed_hashes=value.changed_hashes,
            affected_receipts=value.affected_receipts,
            invalidating_edges=value.invalidating_edges,
            invalidating_paths=value.invalidating_paths,
        )
        if (
            type(value.graph_hash) is not SemanticHash
            or type(value.changed_hashes) is not tuple
            or type(value.affected_receipts) is not tuple
            or type(value.invalidating_edges) is not tuple
            or type(value.invalidating_paths) is not tuple
            or rebuilt.graph_hash != value.graph_hash
            or rebuilt.changed_hashes != value.changed_hashes
            or rebuilt.affected_receipts != value.affected_receipts
            or rebuilt.invalidating_edges != value.invalidating_edges
            or rebuilt.invalidating_paths != value.invalidating_paths
            or rebuilt.semantic_hash != _hash(value.semantic_hash, "semantic_hash")
        ):
            raise StalenessError(
                "staleness_report", "content address integrity rejected"
            )
    except (SchemaError, TypeError, ValueError, AttributeError) as error:
        raise StalenessError(
            "staleness_report", "content address integrity rejected"
        ) from error


def staleness_report_canonical_bytes(value: object) -> bytes:
    _ensure_report(value)
    return canonical_json(value, registry=_REPORT_REGISTRY)


def _propagation(
    graph: StalenessGraph, changed: tuple[SemanticHash, ...]
) -> tuple[tuple[DependencyEdge, ...], tuple[tuple[SemanticHash, ...], ...]]:
    """Return only causal edges and one canonical shortest path per root/end.

    A diamond can have exponentially many simple paths.  The report therefore
    retains every causal edge but only the lexicographically first shortest
    path for each changed-root/affected-receipt pair.
    """
    successors: dict[SemanticHash, tuple[SemanticHash, ...]] = {}
    predecessors: dict[SemanticHash, tuple[SemanticHash, ...]] = {}
    for edge in graph.edges:
        successors[edge.prerequisite] = (
            *successors.get(edge.prerequisite, ()),
            edge.dependent,
        )
        predecessors[edge.dependent] = (
            *predecessors.get(edge.dependent, ()),
            edge.prerequisite,
        )
    receipt_hashes = set(graph.receipt_hashes)
    can_reach_receipt = set(receipt_hashes)
    pending = sorted(receipt_hashes, key=str)
    while pending:
        node = pending.pop(0)
        for parent in sorted(predecessors.get(node, ()), key=str):
            if parent not in can_reach_receipt:
                can_reach_receipt.add(parent)
                pending.append(parent)
    selected_edges: set[tuple[SemanticHash, SemanticHash]] = set()
    paths: list[tuple[SemanticHash, ...]] = []
    for root in changed:
        paths_for_root: dict[SemanticHash, tuple[SemanticHash, ...]] = {root: (root,)}
        pending = [root]
        while pending:
            node = pending.pop(0)
            path = paths_for_root[node]
            for child in sorted(successors.get(node, ()), key=str):
                if node in can_reach_receipt and child in can_reach_receipt:
                    selected_edges.add((node, child))
                candidate = (*path, child)
                previous = paths_for_root.get(child)
                if previous is None:
                    paths_for_root[child] = candidate
                    pending.append(child)
        for endpoint, path in paths_for_root.items():
            if endpoint in receipt_hashes and len(path) > 1:
                paths.append(path)
                if len(paths) > _MAX_PATHS:
                    raise StalenessError("propagation", "path resource limit exceeded")
    edges = tuple(
        DependencyEdge.create(prerequisite, dependent)
        for prerequisite, dependent in sorted(
            selected_edges, key=lambda item: (str(item[0]), str(item[1]))
        )
    )
    return edges, tuple(sorted(paths, key=lambda path: tuple(map(str, path))))


def propagate_staleness(graph: object, *, changed_hashes: object) -> StalenessReport:
    """Report all reverse-reachable receipt hashes without altering history.

    ``changed_hashes`` may include a missing/external prerequisite.  The report
    identifies it separately from affected receipts; a supplied changed receipt
    is not itself called stale merely for being historical evidence.  The graph
    is operational staleness only: it cannot promote or alter scientific claims.
    """
    _ensure_graph(graph)
    checked_graph = cast(StalenessGraph, graph)
    changed = _hashes(changed_hashes, "changed_hashes")
    used_edges, paths = _propagation(checked_graph, changed)
    return _build_report(
        graph_hash=checked_graph.semantic_hash,
        changed_hashes=changed,
        affected_receipts=(path[-1] for path in paths),
        invalidating_edges=used_edges,
        invalidating_paths=paths,
    )
