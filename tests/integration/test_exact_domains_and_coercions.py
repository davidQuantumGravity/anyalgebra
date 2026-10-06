"""Bounded integration tests for exact domains, coercions, and packaging."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import venv
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path

import pytest

from anyalgebra.core.coercions import (
    CoercionGraph,
    CoercionMap,
    coerce,
    common_parent,
)
from anyalgebra.core.domains import DomainElement, QQ, ZZ
from anyalgebra.core.errors import (
    CoercionAmbiguityError,
    CoercionError,
    LossyCoercionError,
)
from anyalgebra.core.rational import Rational


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = REPOSITORY_ROOT / "examples" / "exact_domains.py"

# This grid is exhaustive only within these declared bounds.  Raw QQ pairs are
# retained as cases even when canonical normalization identifies their values.
INTEGER_MIN = -4
INTEGER_MAX = 4
RATIONAL_NUMERATOR_MIN = -3
RATIONAL_NUMERATOR_MAX = 3
RATIONAL_DENOMINATORS = (1, 2, 3)
OPERAND_ORDERS = ("zz-first", "qq-first")
INTEGER_CASES = tuple(range(INTEGER_MIN, INTEGER_MAX + 1))
RATIONAL_RAW_CASES = tuple(
    product(
        range(RATIONAL_NUMERATOR_MIN, RATIONAL_NUMERATOR_MAX + 1),
        RATIONAL_DENOMINATORS,
    )
)
EXPECTED_EXACT_COMMON_PARENT_CASES = (
    len(INTEGER_CASES) * len(RATIONAL_RAW_CASES) * len(OPERAND_ORDERS)
)
EXPECTED_IDENTITY_CASES = len(INTEGER_CASES) + len(RATIONAL_RAW_CASES)

# Every graph-family count is derived from its finite registration and operand
# order bounds below.  The final assertion prevents accidental under-enumeration.
EXPECTED_GRAPH_COMMON_PARENT_CASES = {
    "unique-chain": 2 * len(OPERAND_ORDERS),
    "parallel-ambiguity": 2 * len(OPERAND_ORDERS),
    "lossy-only": 1 * len(OPERAND_ORDERS),
    "unreachable": 1 * len(OPERAND_ORDERS),
    "distinct-equal-label": 1 * len(OPERAND_ORDERS),
}
EXPECTED_GRAPH_COMMON_PARENT_TOTAL = sum(EXPECTED_GRAPH_COMMON_PARENT_CASES.values())


@dataclass(frozen=True, slots=True, eq=False)
class FixtureDomain:
    """Identity-sensitive exact parent used only by the finite graph family."""

    label: str

    def normalize(self, value: object) -> object:
        """Return an already exact fixture payload."""
        return value

    def element(self, value: object) -> FixtureElement:
        """Construct one immutable element owned by this literal parent."""
        return FixtureElement(parent=self, value=value)


@dataclass(frozen=True, slots=True)
class FixtureElement:
    """Minimal immutable element for graph integration checks."""

    parent: FixtureDomain
    value: object


def _fixture_edge(
    identifier: str,
    source: FixtureDomain,
    target: FixtureDomain,
    calls: list[str],
    *,
    lossless: bool = True,
    cost: int = 1,
) -> CoercionMap[object, object]:
    """Create one observable fixture edge with declared exactness metadata."""

    def forward(value: DomainElement[object]) -> FixtureElement:
        calls.append(identifier)
        return target.element(f"{identifier}({value.value})")

    return CoercionMap(
        id=identifier,
        source=source,
        target=target,
        forward=forward,
        injective=lossless,
        exact=True,
        lossless=lossless,
        cost=cost,
    )


def _graph(edges: Sequence[CoercionMap[object, object]]) -> CoercionGraph:
    """Register one finite edge ordering into a persistent graph."""
    graph = CoercionGraph()
    for edge in edges:
        graph = graph.with_map(edge)
    return graph


def _zz_to_qq(value: DomainElement[int]) -> DomainElement[Rational]:
    """Implement the declared canonical exact integer embedding."""
    return QQ().element(value.value)


def _exact_graph() -> CoercionGraph:
    """Return the one-edge graph used throughout the exact scalar grid."""
    return CoercionGraph().with_map(
        CoercionMap(
            id="core.zz_to_qq.v1",
            source=ZZ(),
            target=QQ(),
            forward=_zz_to_qq,
            injective=True,
            exact=True,
            lossless=True,
        )
    )


@pytest.mark.exhaustive
def test_exact_scalar_grid_is_canonical_lossless_and_order_symmetric() -> None:
    """Enumerate every declared raw scalar pair and both operand orders."""
    graph = _exact_graph()
    common_parent_case_count = 0
    identity_case_count = 0

    assert len(INTEGER_CASES) == INTEGER_MAX - INTEGER_MIN + 1 == 9
    assert (
        len(RATIONAL_RAW_CASES)
        == (
            (RATIONAL_NUMERATOR_MAX - RATIONAL_NUMERATOR_MIN + 1)
            * len(RATIONAL_DENOMINATORS)
        )
        == 21
    )
    assert EXPECTED_EXACT_COMMON_PARENT_CASES == 9 * 21 * 2 == 378
    assert EXPECTED_IDENTITY_CASES == 9 + 21 == 30
    assert ZZ() is ZZ()
    assert QQ() is QQ()

    for integer in INTEGER_CASES:
        zz_value = ZZ().element(integer)
        assert coerce(zz_value, ZZ(), graph=graph) is zz_value
        identity_case_count += 1

        for numerator, denominator in RATIONAL_RAW_CASES:
            qq_value = QQ().element((numerator, denominator))
            expected_qq = Rational(numerator, denominator)
            assert qq_value.value == expected_qq

            for order in OPERAND_ORDERS:
                values: tuple[DomainElement[object], DomainElement[object]]
                expected_route_ids: tuple[tuple[str, ...], tuple[str, ...]]
                expected_values: tuple[Rational, Rational]
                if order == "zz-first":
                    values = (zz_value, qq_value)
                    expected_route_ids = (("core.zz_to_qq.v1",), ())
                    expected_values = (Rational(integer), expected_qq)
                else:
                    values = (qq_value, zz_value)
                    expected_route_ids = ((), ("core.zz_to_qq.v1",))
                    expected_values = (expected_qq, Rational(integer))

                plan = common_parent(values, graph=graph)
                applied = plan.apply(values)

                assert plan.target is QQ()
                assert plan.route_ids == expected_route_ids
                assert tuple(value.parent for value in applied) == (QQ(), QQ())
                assert tuple(value.value for value in applied) == expected_values
                common_parent_case_count += 1

    for numerator, denominator in RATIONAL_RAW_CASES:
        qq_value = QQ().element((numerator, denominator))
        assert coerce(qq_value, QQ(), graph=graph) is qq_value
        identity_case_count += 1

    assert common_parent_case_count == EXPECTED_EXACT_COMMON_PARENT_CASES
    assert identity_case_count == EXPECTED_IDENTITY_CASES


@pytest.mark.exhaustive
def test_declared_small_graph_family_exhausts_success_and_failure_routes() -> None:
    """Enumerate registration/operand orders for five bounded graph families."""
    actual_counts = dict.fromkeys(EXPECTED_GRAPH_COMMON_PARENT_CASES, 0)
    diagnostics: dict[tuple[str, str], set[str]] = {}

    source = FixtureDomain("source")
    middle = FixtureDomain("middle")
    target = FixtureDomain("target")
    calls: list[str] = []
    unique_edges = (
        _fixture_edge("source.middle", source, middle, calls),
        _fixture_edge("middle.target", middle, target, calls),
    )
    for registration in permutations(unique_edges):
        graph = _graph(registration)
        transported = coerce(source.element("x"), target, graph=graph)
        assert transported.parent is target
        assert transported.value == "middle.target(source.middle(x))"
        for order in OPERAND_ORDERS:
            values = (
                (source.element("left"), target.element("right"))
                if order == "zz-first"
                else (target.element("right"), source.element("left"))
            )
            plan = common_parent(values, graph=graph)
            applied = plan.apply(values)
            assert plan.target is target
            assert tuple(value.parent for value in applied) == (target, target)
            expected_routes = (
                (("source.middle", "middle.target"), ())
                if order == "zz-first"
                else ((), ("source.middle", "middle.target"))
            )
            assert plan.route_ids == expected_routes
            actual_counts["unique-chain"] += 1

    ambiguity_calls: list[str] = []
    parallel_edges = (
        _fixture_edge("z.expensive", source, target, ambiguity_calls, cost=100),
        _fixture_edge("a.cheap", source, target, ambiguity_calls, cost=0),
    )
    coerce_ambiguity_reasons: set[str] = set()
    for registration in permutations(parallel_edges):
        graph = _graph(registration)
        with pytest.raises(CoercionAmbiguityError) as coerce_error:
            coerce(source.element("x"), target, graph=graph)
        coerce_ambiguity_reasons.add(coerce_error.value.reason)
        for order in OPERAND_ORDERS:
            values = (
                (source.element("left"), target.element("right"))
                if order == "zz-first"
                else (target.element("right"), source.element("left"))
            )
            with pytest.raises(CoercionAmbiguityError) as common_error:
                common_parent(values, graph=graph)
            diagnostics.setdefault(("parallel-ambiguity", order), set()).add(
                common_error.value.reason
            )
            assert "a.cheap" in common_error.value.reason
            assert "z.expensive" in common_error.value.reason
            actual_counts["parallel-ambiguity"] += 1
    assert len(coerce_ambiguity_reasons) == 1
    assert ambiguity_calls == []

    lossy_calls: list[str] = []
    lossy_edge = _fixture_edge(
        "source.target.lossy",
        source,
        target,
        lossy_calls,
        lossless=False,
    )
    lossy_graph = _graph((lossy_edge,))
    with pytest.raises(LossyCoercionError, match=r"source\.target\.lossy"):
        coerce(source.element("x"), target, graph=lossy_graph)
    for order in OPERAND_ORDERS:
        values = (
            (source.element("left"), target.element("right"))
            if order == "zz-first"
            else (target.element("right"), source.element("left"))
        )
        with pytest.raises(LossyCoercionError) as loss_error:
            common_parent(values, graph=lossy_graph)
        diagnostics.setdefault(("lossy-only", order), set()).add(
            loss_error.value.reason
        )
        assert "source.target.lossy" in loss_error.value.reason
        actual_counts["lossy-only"] += 1
    assert lossy_calls == []

    empty_graph = CoercionGraph()
    for order in OPERAND_ORDERS:
        values = (
            (source.element("left"), target.element("right"))
            if order == "zz-first"
            else (target.element("right"), source.element("left"))
        )
        with pytest.raises(CoercionError) as unreachable_error:
            common_parent(values, graph=empty_graph)
        diagnostics.setdefault(("unreachable", order), set()).add(
            unreachable_error.value.reason
        )
        actual_counts["unreachable"] += 1

    equal_label_distinct_source = FixtureDomain("source")
    unique_graph = _graph(unique_edges)
    for order in OPERAND_ORDERS:
        values = (
            (equal_label_distinct_source.element("left"), target.element("right"))
            if order == "zz-first"
            else (target.element("right"), equal_label_distinct_source.element("left"))
        )
        with pytest.raises(CoercionError) as parent_identity_error:
            common_parent(values, graph=unique_graph)
        diagnostics.setdefault(("distinct-equal-label", order), set()).add(
            parent_identity_error.value.reason
        )
        actual_counts["distinct-equal-label"] += 1

    assert EXPECTED_GRAPH_COMMON_PARENT_TOTAL == 14
    assert actual_counts == EXPECTED_GRAPH_COMMON_PARENT_CASES
    assert sum(actual_counts.values()) == EXPECTED_GRAPH_COMMON_PARENT_TOTAL
    assert all(len(reasons) == 1 for reasons in diagnostics.values())


def _subprocess_environment() -> dict[str, str]:
    """Return an environment without inherited Python installation leakage."""
    environment = os.environ.copy()
    for variable in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        environment.pop(variable, None)
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return environment


def _run(
    command: Sequence[str | Path],
    *,
    cwd: Path,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    """Run one checked subprocess and retain output for assertion diagnostics."""
    return subprocess.run(
        [str(part) for part in command],
        check=False,
        cwd=cwd,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        text=True,
    )


def _venv_python(directory: Path) -> Path:
    """Create a clean venv and return its cross-platform interpreter path."""
    venv.EnvBuilder(with_pip=True).create(directory)
    candidates = (directory / "Scripts" / "python.exe", directory / "bin" / "python")
    return next(candidate for candidate in candidates if candidate.is_file())


def _assert_example_payload(payload: object) -> None:
    """Assert the example's stable semantic output rather than display prose."""
    assert payload == {
        "canonical_parents": {"QQ": True, "ZZ": True},
        "canonical_values": {"QQ": [2, 3], "ZZ": 6},
        "common_parent": {
            "route_ids": [["core.zz_to_qq.v1"], []],
            "target": "QQ",
            "values": [[6, 1], [2, 3]],
        },
        "exact_embedding": {
            "parent": "QQ",
            "route_ids": ["core.zz_to_qq.v1"],
            "value": [6, 1],
        },
        "identity": {"QQ_same_object": True, "ZZ_same_object": True},
        "prohibited_loss": {
            "callable_executed": False,
            "error_type": "LossyCoercionError",
            "map_id_reported": True,
        },
        "scope": "infrastructure example only; no research claim verified",
    }


@pytest.mark.slow
def test_example_runs_from_source_and_an_isolated_installed_wheel(
    tmp_path: Path,
) -> None:
    """Run the same example from source and from only an externally installed wheel."""
    source_environment = _subprocess_environment()
    source_environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    source_result = _run(
        (sys.executable, EXAMPLE_PATH),
        cwd=tmp_path,
        environment=source_environment,
    )
    assert source_result.returncode == 0, source_result.stderr
    source_payload = json.loads(source_result.stdout)
    _assert_example_payload(source_payload)

    build_python = _venv_python(tmp_path / "build-environment")
    wheel_directory = tmp_path / "wheels"
    wheel_directory.mkdir()
    build_result = _run(
        (
            build_python,
            "-m",
            "pip",
            "--isolated",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            wheel_directory,
            REPOSITORY_ROOT,
        ),
        cwd=tmp_path,
        environment=_subprocess_environment(),
    )
    assert build_result.returncode == 0, build_result.stdout + build_result.stderr
    wheels = tuple(wheel_directory.glob("anyalgebra-*.whl"))
    assert len(wheels) == 1

    install_python = _venv_python(tmp_path / "install-environment")
    install_result = _run(
        (
            install_python,
            "-m",
            "pip",
            "--isolated",
            "install",
            "--no-deps",
            wheels[0],
        ),
        cwd=tmp_path,
        environment=_subprocess_environment(),
    )
    assert install_result.returncode == 0, install_result.stdout + install_result.stderr

    external_directory = tmp_path / "external-working-directory"
    external_directory.mkdir()
    copied_example = external_directory / EXAMPLE_PATH.name
    shutil.copy2(EXAMPLE_PATH, copied_example)
    installed_result = _run(
        (install_python, "-I", copied_example),
        cwd=external_directory,
        environment=_subprocess_environment(),
    )
    assert installed_result.returncode == 0, installed_result.stderr
    installed_payload = json.loads(installed_result.stdout)
    _assert_example_payload(installed_payload)
    assert installed_payload == source_payload
