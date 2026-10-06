"""Close presentation adapter, basis-change, graph, and route guard seams."""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast
import pytest

import anyalgebra.presentations.basis_change as basis_change
import anyalgebra.presentations.convert as conversion
from anyalgebra.algebra.multilinear import (
    BasisProductTable,
    FiniteMultilinearStructure,
    StructureConstants,
)
from anyalgebra.core.domains import QQ, ZZ
from anyalgebra.core.elements import SparseElement
from anyalgebra.core.modules import Basis, FreeModule
from anyalgebra.presentations.adapters import (
    BasisExpression,
    PresentationAdapterBundle,
    PresentationAdapterError,
)
from anyalgebra.presentations.basis_change import (
    BasisChangeCertificate,
    BasisChangeError,
    ExactBasisIsomorphism,
)
from anyalgebra.presentations.convert import (
    ConversionPlan,
    ConversionRouteError,
    ConversionSearchLimitError,
    ConvertedValue,
    RouteStep,
    convert,
)
from anyalgebra.presentations.graph import (
    Conversion,
    ConversionDefinitionError,
    ConversionGraph,
    PresentationKind,
)


def _module(rank: int = 1, *, qq: bool = False) -> FreeModule:
    domain = QQ() if qq else ZZ()
    return FreeModule(
        domain,
        Basis(tuple(f"e{i}" for i in range(rank)), coefficient_domain=domain),
    )


def _basis(module: FreeModule, index: int) -> SparseElement:
    return module.element({index: 1})


def _identity(module: FreeModule) -> ExactBasisIsomorphism:
    images = tuple(_basis(module, index) for index in range(module.rank))
    return ExactBasisIsomorphism.from_images(module, module, images, images)


def _kind(name: str) -> PresentationKind:
    return PresentationKind.from_id(name)


def _edge(
    name: str,
    source: PresentationKind,
    target: PresentationKind,
    **kwargs: object,
) -> Conversion:
    return Conversion.from_callable(name, source, target, lambda value: value, **kwargs)


def test_adapter_exact_factory_and_identity_guards() -> None:
    module = _module()

    class ExpressionSubclass(BasisExpression):
        pass

    with pytest.raises(PresentationAdapterError, match="exact BasisExpression"):
        ExpressionSubclass.from_sparse(module, module.zero())
    bundle = PresentationAdapterBundle.for_module(module, "demo", 1)
    assert bundle == bundle


def test_adapter_normalizes_compilation_evaluation_and_table_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    bundle = PresentationAdapterBundle.for_module(module, "demo", 1)
    table = BasisProductTable.from_cells(module, 1, ({},))

    def fail_compile(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("private")

    monkeypatch.setattr(FiniteMultilinearStructure, "from_tables", fail_compile)
    with pytest.raises(PresentationAdapterError, match="RuntimeError"):
        bundle.structure_from_table(table)
    monkeypatch.undo()

    structure = FiniteMultilinearStructure.from_tables(module, table)

    def fail_evaluation(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("private")

    monkeypatch.setattr(FiniteMultilinearStructure, "evaluate_basis", fail_evaluation)
    with pytest.raises(PresentationAdapterError, match="basis evaluation failed"):
        bundle.table_from_structure(structure)
    monkeypatch.undo()

    def invalid_cell(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return object()

    monkeypatch.setattr(FiniteMultilinearStructure, "evaluate_basis", invalid_cell)
    with pytest.raises(PresentationAdapterError, match="SparseElement"):
        bundle.table_from_structure(structure)
    monkeypatch.undo()

    def fail_table(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("private")

    monkeypatch.setattr(BasisProductTable, "from_cells", fail_table)
    with pytest.raises(PresentationAdapterError, match="cells could not form"):
        bundle.table_from_structure(structure)


def test_basis_change_certificate_snapshot_vector_and_apply_guards() -> None:
    module = _module()
    with pytest.raises(BasisChangeError, match="produced"):
        BasisChangeCertificate()
    identity = _identity(module)
    assert identity == identity
    assert identity.certificate == identity.certificate
    with pytest.raises(BasisChangeError, match="ordered image iterable"):
        ExactBasisIsomorphism.from_images(module, module, "bad", ())
    with pytest.raises(BasisChangeError, match="construct exact basis vector"):
        basis_change._basis_vector(module, module.rank)
    with pytest.raises(BasisChangeError, match="exact SparseElement"):
        basis_change._linear_apply(
            cast(SparseElement, object()),
            identity.forward_images,
            module,
            side="forward",
        )
    with pytest.raises(BasisChangeError, match="arithmetic failed"):
        basis_change._linear_apply(module.element({0: 1}), (), module, side="forward")
    with pytest.raises(BasisChangeError, match="inverse_images"):
        ExactBasisIsomorphism.from_images(module, module, (_basis(module, 0),), ())
    with pytest.raises(BasisChangeError, match="inverse value"):
        identity.inverse(cast(SparseElement, object()))


def test_inverse_after_forward_certificate_failure_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    original = basis_change._linear_apply

    def selective(
        element: SparseElement,
        images: tuple[SparseElement, ...],
        target: FreeModule,
        *,
        side: str,
    ) -> SparseElement:
        if side == "inverse-after-forward":
            return target.zero()
        return original(element, images, target, side=side)

    monkeypatch.setattr(basis_change, "_linear_apply", selective)
    with pytest.raises(BasisChangeError, match="inverse-after-forward"):
        _identity(module)


def test_transport_constants_domain_and_construction_failure_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _module()
    foreign = _module(qq=True)
    identity = _identity(source)
    foreign_constants = StructureConstants.from_sparse(
        (foreign.basis,), foreign.basis, ()
    )
    with pytest.raises(BasisChangeError, match="coefficient parent"):
        basis_change._transport_constants(foreign_constants, identity)

    constants = StructureConstants.from_sparse((source.basis,), source.basis, ())

    def fail(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("private")

    monkeypatch.setattr(StructureConstants, "from_sparse", fail)
    with pytest.raises(BasisChangeError, match="RuntimeError"):
        basis_change._transport_constants(constants, identity)


def test_route_record_shapes_identity_and_iterable_failures() -> None:
    source, target = _kind("source"), _kind("target")
    edge = _edge("edge", source, target)
    with pytest.raises(ConversionRouteError, match="exact Conversion"):
        RouteStep.from_conversion(object(), "forward")
    step = RouteStep.from_conversion(edge, "forward")
    assert step == step
    with pytest.raises(ConversionRouteError, match="exact tuple"):
        ConversionPlan.from_steps(source, target, [step])
    with pytest.raises(ConversionRouteError, match="exact RouteStep"):
        ConversionPlan.from_steps(source, target, (object(),))
    plan = ConversionPlan.from_steps(source, target, (step,))
    assert plan == plan
    result = ConvertedValue._from_applied(1, plan)
    assert result == result

    class BrokenStart:
        def __iter__(self) -> Iterator[object]:
            raise RuntimeError("private")

    class BrokenNext:
        def __iter__(self) -> Iterator[object]:
            yield step
            raise RuntimeError("private")

    graph = ConversionGraph.from_conversions((edge,))
    generated = conversion.plan_conversion(
        source, target, graph=graph, route=(value for value in (step,))
    )
    assert generated.steps == (step,)
    for route in (BrokenStart(), BrokenNext()):
        with pytest.raises(ConversionRouteError, match="could not be iterated"):
            conversion.plan_conversion(source, target, graph=graph, route=route)
    with pytest.raises(ConversionRouteError, match="exact RouteStep"):
        conversion.plan_conversion(source, target, graph=graph, route=(object(),))


def test_route_depth_limit_and_admissible_true_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target = _kind("source"), _kind("target")
    monkeypatch.setattr(conversion, "_MAX_ROUTE_STEPS", 0)
    with pytest.raises(ConversionSearchLimitError, match="route steps"):
        conversion.plan_conversion(source, target, graph=ConversionGraph.empty())
    monkeypatch.undo()

    edge = _edge("edge", source, target, admissibility=lambda _value: True)
    graph = ConversionGraph.from_conversions((edge,))
    result = convert(1, target, graph=graph, source_kind=source)
    assert result.value == 1


def test_graph_raw_iterables_mutation_limit_and_kind_guards() -> None:
    source, target = _kind("source"), _kind("target")
    with pytest.raises(ConversionDefinitionError, match="iterable"):
        _edge("edge", source, target, guarantees="bad")
    with pytest.raises(ConversionDefinitionError, match="iterable"):
        ConversionGraph.from_conversions("bad")

    class Broken:
        def __iter__(self) -> Iterator[object]:
            raise RuntimeError("private")

    with pytest.raises(ConversionDefinitionError, match="could not be iterated"):
        ConversionGraph.from_conversions(Broken())
    graph = ConversionGraph.empty()
    with pytest.raises(ConversionDefinitionError, match="exact Conversion"):
        graph.with_conversion(object())

    edges = tuple(_edge(f"edge.{i}", source, target) for i in range(512))
    full = ConversionGraph.from_conversions(edges)
    with pytest.raises(ConversionDefinitionError, match="maximum"):
        full.with_conversion(_edge("edge.513", source, target))
    with pytest.raises(ConversionDefinitionError, match="exact PresentationKind"):
        graph.outgoing(object())
    with pytest.raises(ConversionDefinitionError, match="exact PresentationKind"):
        graph.incoming(object())
