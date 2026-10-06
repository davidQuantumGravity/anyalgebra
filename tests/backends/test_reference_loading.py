"""Adversarial contracts for V00-080's reference route and optional loader."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
from types import FunctionType, ModuleType
from typing import Any, cast

import pytest

from anyalgebra.backends.base import (
    BackendBoundaryError,
    BackendCapabilities,
    BackendOptions,
    BackendRequest,
    BackendResult,
    BackendUnsupported,
)
from anyalgebra.backends import loading
from anyalgebra.backends.loading import (
    AdapterAvailability,
    AdapterLoad,
    AdapterLoadState,
    AdapterSpec,
    OptionalAdapterLoader,
    load_optional_adapter,
)
from anyalgebra.backends.reference import ReferenceBackend
from anyalgebra.core.parents import SemanticHash
from anyalgebra.structures.outcomes import Defined, Failed, UnsupportedCapability


def h(value: str) -> SemanticHash:
    return SemanticHash("sha256", hashlib.sha256(value.encode("utf-8")).hexdigest())


def request(**changes: object) -> BackendRequest:
    values: dict[str, object] = {
        "operation": "hash_identity",
        "domain": "semantic_hash",
        "algorithm": "identity",
        "input_hashes": (h("input"),),
        "limits": (("inputs", 1),),
    }
    values.update(changes)
    return BackendRequest.create(**values)


def options(**changes: object) -> BackendOptions:
    values: dict[str, object] = {"limits": (("inputs", 1),)}
    values.update(changes)
    return BackendOptions.create(**values)


def test_reference_routes_only_an_honestly_bound_hash_request() -> None:
    backend = ReferenceBackend()
    supported = backend.supports(request())
    outcome = backend.execute(request(), options=options())
    assert type(supported).__name__ == "BackendSupported"
    assert type(outcome) is Defined and type(outcome.value) is BackendResult
    assert outcome.value.value_hash == h("input")
    assert not hasattr(outcome.value, "value") and "input" not in repr(outcome)


@pytest.mark.parametrize(
    "bad_request",
    (
        request(input_hashes=(), limits=(("inputs", 0),)),
        request(input_hashes=(h("a"), h("b")), limits=(("inputs", 1),)),
        request(limits=()),
    ),
)
def test_reference_rejects_malformed_cardinality_without_fabricating_provenance(
    bad_request: BackendRequest,
) -> None:
    backend = ReferenceBackend()
    with pytest.raises(BackendBoundaryError, match="hash_identity requires"):
        backend.supports(bad_request)
    outcome = backend.execute(bad_request, options=options())
    assert type(outcome) is Failed
    assert (outcome.error, outcome.stage) == (
        "reference execution rejected request binding",
        "binding",
    )


def test_reference_keeps_actual_capability_provenance_for_genuine_absence() -> None:
    backend = ReferenceBackend()
    unsupported = backend.supports(request(operation="unknown"))
    limited = backend.supports(request(limits=(("inputs", 2),)))
    assert type(unsupported) is BackendUnsupported
    assert type(limited) is BackendUnsupported
    assert unsupported.capabilities_hash == backend.capabilities().semantic_hash
    assert limited.capabilities_hash == backend.capabilities().semantic_hash
    with pytest.raises(UnsupportedCapability) as error:
        backend.execute(request(operation="unknown"), options=options())
    assert error.value.capability == "unknown"
    assert error.value.context == unsupported.reason.value


def test_reference_binds_options_sanitizes_failures_and_preserves_interrupts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = ReferenceBackend()
    assert type(backend.execute(request(), options=BackendOptions.create())) is Failed
    assert (
        type(backend.execute(request(), options=cast(BackendOptions, object())))
        is Failed
    )

    def fail(**kwargs: object) -> BackendResult:
        del kwargs
        raise RuntimeError("native secret")

    monkeypatch.setattr(BackendResult, "create", fail)
    failed = ReferenceBackend().execute(request(), options=options())
    assert type(failed) is Failed and "secret" not in repr(failed)

    def interrupt(self: ReferenceBackend, value: BackendRequest) -> object:
        del self, value
        raise KeyboardInterrupt

    monkeypatch.setattr(ReferenceBackend, "supports", interrupt)
    with pytest.raises(KeyboardInterrupt):
        backend.execute(request(), options=options())
    with pytest.raises(TypeError):
        type("Subclass", (ReferenceBackend,), {})


def spec(name: str = "good", module: str = "v00_080_good") -> AdapterSpec:
    return AdapterSpec(name=name, module=module, factory="build", version="1.2.3")


class GoodAdapter:
    def __init__(self, name: str = "good", version: str = "1.2.3") -> None:
        self.name, self.version = name, version

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities.create(
            name=self.name, version=self.version, specifications=()
        )

    def supports(self, value: object) -> object:
        return value

    def execute(self, value: object, *, options: object) -> object:
        return value, options


def _surface_adapter(**changes: object) -> object:
    members: dict[str, object] = {
        "capabilities": GoodAdapter.capabilities,
        "supports": GoodAdapter.supports,
        "execute": GoodAdapter.execute,
    }
    members.update(changes)
    adapter_type = type("SurfaceAdapter", (), members)
    adapter = adapter_type()
    adapter.name, adapter.version = "good", "1.2.3"
    return adapter


def _capabilities_missing_self() -> BackendCapabilities:
    raise AssertionError("shape validation must occur before invocation")


def _capabilities_default(self: object = None) -> BackendCapabilities:
    del self
    raise AssertionError("shape validation must occur before invocation")


def _supports_extra(self: object, request: object, extra: object) -> object:
    return self, request, extra


def _supports_varargs(self: object, request: object, *extra: object) -> object:
    return self, request, extra


def _execute_positional_options(
    self: object, request: object, options: object
) -> object:
    return self, request, options


def _execute_wrong_keyword(self: object, request: object, *, other: object) -> object:
    return self, request, other


def _execute_default_keyword(
    self: object, request: object, *, options: object = None
) -> object:
    return self, request, options


async def _execute_async(self: object, request: object, *, options: object) -> object:
    return self, request, options


def _execute_generator(self: object, request: object, *, options: object) -> object:
    yield self, request, options


def module_for(adapter: object, name: str = "v00_080_good") -> ModuleType:
    module = ModuleType(name)

    def build() -> object:
        return adapter

    module.__dict__["build"] = build
    return module


@pytest.mark.parametrize(
    "changes",
    (
        {"name": "bad/name"},
        {"module": "1bad"},
        {"factory": "bad-name"},
        {"factory": "9build"},
        {"version": "bad%version"},
    ),
)
def test_specs_and_record_factories_are_sealed_and_validate_invariants(
    changes: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "name": "good",
        "module": "v00_080_good",
        "factory": "build",
        "version": "1.2.3",
    }
    values.update(changes)
    with pytest.raises(BackendBoundaryError):
        AdapterSpec(**values)  # type: ignore[arg-type]
    valid = spec()
    with pytest.raises(BackendBoundaryError):
        AdapterAvailability.create(
            state=AdapterLoadState.LOADED, name="good", specification=valid
        )
    loaded = loading._loaded_availability(valid)
    with pytest.raises(BackendBoundaryError):
        AdapterAvailability()
    with pytest.raises(BackendBoundaryError):
        AdapterLoad()
    with pytest.raises(BackendBoundaryError):
        AdapterLoad.create(availability=loaded)
    unavailable = AdapterAvailability.create(
        state=AdapterLoadState.UNKNOWN,
        name="unknown",
        diagnostic="adapter is not allowlisted",
    )
    with pytest.raises(BackendBoundaryError):
        AdapterLoad.create(availability=unavailable, adapter=object())
    with pytest.raises(BackendBoundaryError):
        AdapterAvailability.create(
            state=AdapterLoadState.INVALID, name="good", specification=valid
        )
    with pytest.raises(BackendBoundaryError):
        AdapterAvailability.create(
            state=AdapterLoadState.UNINSTALLED,
            name="good",
            diagnostic="missing",
        )
    with pytest.raises(BackendBoundaryError):
        AdapterAvailability.create(
            state=AdapterLoadState.INVALID,
            name="good",
            specification=object(),
            diagnostic="invalid",
        )
    with pytest.raises(BackendBoundaryError):
        AdapterAvailability.create(
            state=AdapterLoadState.INVALID,
            name="other",
            specification=valid,
            diagnostic="invalid",
        )
    with pytest.raises(BackendBoundaryError):
        cast(Any, AdapterAvailability.create).__func__(
            cast(type[AdapterAvailability], object),
            state=AdapterLoadState.UNKNOWN,
            name="good",
            diagnostic="invalid",
        )
    with pytest.raises(BackendBoundaryError):
        cast(Any, AdapterLoad.create).__func__(
            cast(type[AdapterLoad], object), availability=unavailable
        )
    with pytest.raises(TypeError):
        type("Subclass", (AdapterAvailability,), {})
    with pytest.raises(TypeError):
        type("Subclass", (AdapterLoad,), {})
    with pytest.raises(TypeError):
        type("Subclass", (AdapterSpec,), {})


def test_record_factories_reject_every_contradictory_public_state() -> None:
    valid = spec()
    unavailable = AdapterAvailability.create(
        state=AdapterLoadState.INVALID,
        name="good",
        specification=valid,
        diagnostic="invalid",
    )
    AdapterLoad.create(availability=unavailable)._assert()
    loaded = loading._loaded_availability(valid)
    for create in (
        lambda: AdapterAvailability.create(
            state=cast(AdapterLoadState, "loaded"), name="good"
        ),
        lambda: AdapterAvailability.create(
            state=AdapterLoadState.UNKNOWN,
            name="good",
            specification=valid,
            diagnostic="bad",
        ),
        lambda: AdapterAvailability.create(
            state=AdapterLoadState.INVALID, name="good", diagnostic=None
        ),
        lambda: AdapterAvailability.create(
            state=AdapterLoadState.LOADED,
            name="good",
            specification=valid,
            diagnostic="bad",
        ),
        lambda: AdapterLoad.create(availability=object()),
        lambda: AdapterLoad.create(availability=loaded),
        lambda: AdapterLoad.create(availability=unavailable, adapter=object()),
    ):
        with pytest.raises(BackendBoundaryError):
            create()
    object.__setattr__(unavailable, "module", "wrong")
    with pytest.raises(BackendBoundaryError):
        unavailable._assert()
    with pytest.raises(BackendBoundaryError):
        AdapterLoad.create(availability=loaded, adapter=object())
    complete = AdapterLoad.create(availability=loaded, adapter=GoodAdapter())
    object.__setattr__(complete, "adapter", None)
    with pytest.raises(BackendBoundaryError):
        complete._assert()


def test_availability_diagnostics_are_bounded_plain_text() -> None:
    assert (
        AdapterAvailability.create(
            state=AdapterLoadState.UNKNOWN,
            name="unknown",
            diagnostic="x" * 256,
        ).diagnostic
        == "x" * 256
    )
    for diagnostic in ("x" * 257, "bad\nvalue", "bad\x7fvalue"):
        with pytest.raises(BackendBoundaryError):
            AdapterAvailability.create(
                state=AdapterLoadState.UNKNOWN,
                name="unknown",
                diagnostic=diagnostic,
            )
    loaded = loading._loaded_availability(spec())
    object.__setattr__(loaded, "diagnostic", "unexpected")
    with pytest.raises(BackendBoundaryError):
        loaded._assert()


def test_private_text_and_snapshot_boundaries_are_bounded() -> None:
    with pytest.raises(BackendBoundaryError):
        loading._text(cast(object, 1), "field", frozenset("x"))
    with pytest.raises(BackendBoundaryError):
        loading._name("bad/")
    with pytest.raises(BackendBoundaryError):
        loading._module("")
    with pytest.raises(BackendBoundaryError):
        loading._factory("1bad")
    assert loading._snapshot(tuple(spec() for _ in range(33))) is None
    assert loading._snapshot(cast(tuple[AdapterSpec, ...], (object(),))) is None
    assert loading._snapshot((spec("duplicate"), spec("duplicate"))) is None
    source = spec()
    object.__setattr__(source, "name", "bad/name")
    assert loading._snapshot((source,)) is None


def test_loader_is_lazy_instance_scoped_and_concurrent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    adapter = GoodAdapter()
    module = module_for(adapter)

    def importer(name: str, package: str | None = None) -> ModuleType:
        del package
        calls.append(name)
        return module

    monkeypatch.setattr(importlib, "import_module", importer)
    loader = OptionalAdapterLoader((spec(),))
    assert loader.load("unknown").availability.state is AdapterLoadState.UNKNOWN
    assert calls == []
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(lambda _: loader.load("good"), range(16)))
    assert all(result.adapter is cast(object, adapter) for result in results)
    assert calls == ["v00_080_good"]
    assert len(loader.specifications) == 1
    assert (
        loader.load(cast(str, object())).availability.state is AdapterLoadState.INVALID
    )
    assert loader.load(" bad").availability.state is AdapterLoadState.INVALID


def test_convenience_loader_is_uncached_and_failures_can_recover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    adapter = GoodAdapter()

    def importer(name: str, package: str | None = None) -> ModuleType:
        del package
        calls.append(name)
        if len(calls) == 1:
            raise ModuleNotFoundError(name=name)
        return module_for(adapter, name)

    monkeypatch.setattr(importlib, "import_module", importer)
    loader = OptionalAdapterLoader((spec(),))
    assert loader.load("good").availability.state is AdapterLoadState.UNINSTALLED
    assert loader.load("good").availability.state is AdapterLoadState.LOADED
    assert load_optional_adapter("good", specifications=(spec(),)).adapter is cast(
        object, adapter
    )
    assert len(calls) == 3


def test_loader_distinguishes_missing_module_from_missing_internal_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = spec("missing", "v00_080_missing")
    internal = spec("internal", "v00_080_internal")

    def importer(name: str, package: str | None = None) -> ModuleType:
        del package
        if name == missing.module:
            raise ModuleNotFoundError(name=name)
        raise ModuleNotFoundError(name="internal_dependency")

    monkeypatch.setattr(importlib, "import_module", importer)
    loader = OptionalAdapterLoader((missing, internal))
    assert loader.load("missing").availability.state is AdapterLoadState.UNINSTALLED
    assert loader.load("internal").availability.state is AdapterLoadState.INVALID
    nested = spec("nested", "v00_080_parent.adapter")

    def nested_importer(name: str, package: str | None = None) -> ModuleType:
        del name, package
        raise ModuleNotFoundError(name="v00_080_parent")

    monkeypatch.setattr(importlib, "import_module", nested_importer)
    assert (
        OptionalAdapterLoader((nested,)).load("nested").availability.state
        is AdapterLoadState.UNINSTALLED
    )


@pytest.mark.parametrize(
    "adapter",
    (
        object(),
        GoodAdapter(name="other"),
        GoodAdapter(version="9.9"),
    ),
)
def test_loader_rejects_invalid_shape_and_capability_provenance(
    monkeypatch: pytest.MonkeyPatch, adapter: object
) -> None:
    monkeypatch.setattr(
        importlib, "import_module", lambda name: module_for(adapter, name)
    )
    result = OptionalAdapterLoader((spec(),)).load("good")
    assert result.availability.state is AdapterLoadState.INVALID
    assert result.adapter is None and "object" not in repr(result)


@pytest.mark.parametrize(
    "changes",
    (
        {"capabilities": _capabilities_missing_self},
        {"capabilities": _capabilities_default},
        {"supports": _supports_extra},
        {"supports": _supports_varargs},
        {"execute": _execute_positional_options},
        {"execute": _execute_wrong_keyword},
        {"execute": _execute_default_keyword},
        {"execute": _execute_async},
        {"execute": _execute_generator},
        {"execute": staticmethod(GoodAdapter.execute)},
        {"execute": classmethod(cast(Any, GoodAdapter.execute))},
        {"execute": property(lambda self: GoodAdapter.execute)},
    ),
)
def test_loader_rejects_adversarial_plain_method_surfaces(
    monkeypatch: pytest.MonkeyPatch, changes: dict[str, object]
) -> None:
    adapter = _surface_adapter(**changes)
    monkeypatch.setattr(
        importlib, "import_module", lambda name: module_for(adapter, name)
    )
    result = OptionalAdapterLoader((spec(),)).load("good")
    assert result.availability.state is AdapterLoadState.INVALID
    with pytest.raises(BackendBoundaryError):
        AdapterLoad.create(
            availability=loading._loaded_availability(spec()), adapter=adapter
        )


class HostileMeta(type):
    def __getattribute__(cls, name: str) -> object:
        if name in {"__mro__", "__dict__"}:
            raise RuntimeError("native metaclass detail")
        return super().__getattribute__(name)


class HostileMetaAdapter(GoodAdapter, metaclass=HostileMeta):
    pass


def test_hostile_metaclass_is_safely_discovered_without_instance_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = HostileMetaAdapter()
    direct = AdapterLoad.create(
        availability=loading._loaded_availability(spec()), adapter=adapter
    )
    assert direct.adapter is cast(object, adapter)
    monkeypatch.setattr(
        importlib, "import_module", lambda name: module_for(adapter, name)
    )
    result = OptionalAdapterLoader((spec(),)).load("good")
    assert result.availability.state is AdapterLoadState.LOADED
    assert "native" not in repr(result)


def test_method_discovery_preserves_interrupts(monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupt(
        adapter: object, name: str, positional: int, keyword_only: tuple[str, ...]
    ) -> FunctionType:
        del adapter, name, positional, keyword_only
        raise KeyboardInterrupt

    monkeypatch.setattr(loading, "_method", interrupt)
    with pytest.raises(KeyboardInterrupt):
        AdapterLoad.create(
            availability=loading._loaded_availability(spec()), adapter=GoodAdapter()
        )


def test_adapter_load_identity_includes_opaque_handle_identity() -> None:
    availability = loading._loaded_availability(spec())
    first = AdapterLoad.create(availability=availability, adapter=GoodAdapter())
    second = AdapterLoad.create(
        availability=loading._loaded_availability(spec()), adapter=GoodAdapter()
    )
    assert first == first and first != second
    assert len({first, second}) == 2


class Hostile:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"must not inspect {name}")


def test_loader_rejects_hostile_registration_and_adapter_without_probing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(BackendBoundaryError):
        OptionalAdapterLoader(cast(tuple[AdapterSpec, ...], (object(),)))
    monkeypatch.setattr(
        importlib, "import_module", lambda name: module_for(Hostile(), name)
    )
    result = OptionalAdapterLoader((spec(),)).load("good")
    assert result.availability.state is AdapterLoadState.INVALID


class BadCapabilitiesAdapter:
    def capabilities(self) -> object:
        return object()

    def supports(self, value: object) -> object:
        return value

    def execute(self, value: object, *, options: object) -> object:
        return value, options


class InheritedAdapter(GoodAdapter):
    """A valid adapter whose protocol methods are inherited from its base class."""


class RuntimeCapabilitiesAdapter:
    def capabilities(self) -> BackendCapabilities:
        raise RuntimeError("native capability detail")

    def supports(self, value: object) -> object:
        return value

    def execute(self, value: object, *, options: object) -> object:
        return value, options


class OneValidThenFailAdapter(GoodAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def capabilities(self) -> BackendCapabilities:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("second capability call")
        return super().capabilities()


class InterruptCapabilitiesAdapter:
    def capabilities(self) -> BackendCapabilities:
        raise KeyboardInterrupt

    def supports(self, value: object) -> object:
        return value

    def execute(self, value: object, *, options: object) -> object:
        return value, options


def test_loader_covers_missing_factory_and_capability_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("v00_080_good")
    monkeypatch.setattr(importlib, "import_module", lambda name: module)
    assert OptionalAdapterLoader((spec(),)).load("good").availability.state is (
        AdapterLoadState.INVALID
    )
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: module_for(RuntimeCapabilitiesAdapter(), name),
    )
    assert OptionalAdapterLoader((spec(),)).load("good").availability.state is (
        AdapterLoadState.INVALID
    )
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: module_for(BadCapabilitiesAdapter(), name),
    )
    assert OptionalAdapterLoader((spec(),)).load("good").availability.state is (
        AdapterLoadState.INVALID
    )
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: module_for(InterruptCapabilitiesAdapter(), name),
    )
    with pytest.raises(KeyboardInterrupt):
        OptionalAdapterLoader((spec(),)).load("good")

    interrupt = ModuleType("v00_080_good")

    def build() -> object:
        raise SystemExit(8)

    interrupt.__dict__["build"] = build
    monkeypatch.setattr(importlib, "import_module", lambda name: interrupt)
    with pytest.raises(SystemExit, match="8"):
        OptionalAdapterLoader((spec(),)).load("good")

    monkeypatch.setattr(importlib, "import_module", lambda name: object())
    invalid_module = OptionalAdapterLoader((spec(),)).load("good")
    assert invalid_module.availability.state is AdapterLoadState.INVALID


def test_initial_load_validates_capabilities_once_and_sanitizes_late_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = OneValidThenFailAdapter()
    monkeypatch.setattr(
        importlib, "import_module", lambda name: module_for(adapter, name)
    )
    loaded = OptionalAdapterLoader((spec(),)).load("good")
    assert loaded.availability.state is AdapterLoadState.LOADED
    assert adapter.calls == 1

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: module_for(RuntimeCapabilitiesAdapter(), name),
    )
    invalid = OptionalAdapterLoader((spec(),)).load("good")
    assert invalid.availability.state is AdapterLoadState.INVALID


def test_loader_accepts_inherited_methods_and_ignores_poisoned_public_wrappers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = InheritedAdapter()
    monkeypatch.setattr(
        importlib, "import_module", lambda name: module_for(adapter, name)
    )
    loader = OptionalAdapterLoader((spec(),))
    first = loader.load("good")
    assert first.adapter is cast(object, adapter)
    object.__setattr__(first, "adapter", object())
    second = loader.load("good")
    assert second is not first
    assert second.adapter is cast(object, adapter)
    second._assert()


def test_loader_evicts_invalid_cached_adapter_and_preserves_cached_interrupts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = GoodAdapter()
    calls: list[str] = []

    def importer(name: str, package: str | None = None) -> ModuleType:
        del package
        calls.append(name)
        return module_for(stale if len(calls) == 1 else GoodAdapter(), name)

    monkeypatch.setattr(importlib, "import_module", importer)
    loader = OptionalAdapterLoader((spec(),))
    assert loader.load("good").adapter is cast(object, stale)
    stale.name = "wrong"
    refreshed = loader.load("good")
    assert refreshed.adapter is not cast(object, stale)
    assert calls == ["v00_080_good", "v00_080_good"]

    interrupting = InterruptCapabilitiesAdapter()
    monkeypatch.setattr(
        importlib, "import_module", lambda name: module_for(interrupting, name)
    )
    with pytest.raises(KeyboardInterrupt):
        OptionalAdapterLoader((spec(),)).load("good")


def test_loader_snapshots_source_specs_and_rejects_corruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = spec()
    loader = OptionalAdapterLoader((source,))
    object.__setattr__(source, "module", "v00_080_changed")
    exposed = loader.specifications[0]
    object.__setattr__(exposed, "module", "v00_080_exposed_changed")
    calls: list[str] = []

    def importer(name: str, package: str | None = None) -> ModuleType:
        del package
        calls.append(name)
        return module_for(GoodAdapter(), name)

    monkeypatch.setattr(importlib, "import_module", importer)
    assert loader.load("good").availability.state is AdapterLoadState.LOADED
    assert calls == ["v00_080_good"]
    broken = spec()
    object.__setattr__(broken, "factory", "bad-name")
    with pytest.raises(BackendBoundaryError):
        OptionalAdapterLoader((broken,))


def test_loader_sanitizes_factory_and_import_failures_and_preserves_interrupts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = spec("failure", "v00_080_failure")
    module = ModuleType(failure.module)

    def build() -> object:
        raise RuntimeError("native secret")

    module.__dict__["build"] = build
    monkeypatch.setattr(importlib, "import_module", lambda name: module)
    failed = OptionalAdapterLoader((failure,)).load("failure")
    assert failed.availability.state is AdapterLoadState.FACTORY_FAILED
    assert "secret" not in repr(failed)

    def interrupt(name: str, package: str | None = None) -> ModuleType:
        del name, package
        raise SystemExit(9)

    monkeypatch.setattr(importlib, "import_module", interrupt)
    with pytest.raises(SystemExit, match="9"):
        OptionalAdapterLoader((spec(),)).load("good")

    def import_failure(name: str, package: str | None = None) -> ModuleType:
        del name, package
        raise RuntimeError("native import detail")

    monkeypatch.setattr(importlib, "import_module", import_failure)
    invalid = OptionalAdapterLoader((spec(),)).load("good")
    assert invalid.availability.state is AdapterLoadState.INVALID
    assert "detail" not in repr(invalid)
    convenience = load_optional_adapter(
        "good", specifications=cast(tuple[AdapterSpec, ...], (object(),))
    )
    assert convenience.availability.state is AdapterLoadState.INVALID
    invalid_name = load_optional_adapter(
        " bad", specifications=cast(tuple[AdapterSpec, ...], (object(),))
    )
    assert invalid_name.availability.name == "invalid"

    def loader_interrupt(specifications: object) -> OptionalAdapterLoader:
        del specifications
        raise KeyboardInterrupt

    monkeypatch.setattr(loading, "OptionalAdapterLoader", loader_interrupt)
    with pytest.raises(KeyboardInterrupt):
        load_optional_adapter("good", specifications=(spec(),))
