"""Explicit, lazy loading for opt-in optional backend adapters.

There is intentionally no entry-point, sibling-directory, or path discovery.
An ``OptionalAdapterLoader`` snapshots one allowlist and owns its own successful
load cache; the module-level convenience function intentionally has no cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import importlib
from inspect import (
    CO_ASYNC_GENERATOR,
    CO_COROUTINE,
    CO_GENERATOR,
    CO_VARARGS,
    CO_VARKEYWORDS,
)
from threading import RLock
from types import FunctionType, ModuleType
from typing import cast

from anyalgebra.backends.base import (
    BackendAdapter,
    BackendBoundaryError,
    BackendCapabilities,
)


_MAX_SPECS = 32
_MAX_DIAGNOSTIC = 256
_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_NAME_CHARS = frozenset(f"{_LETTERS}0123456789_-")
_IDENTIFIER_CHARS = frozenset(f"{_LETTERS}0123456789_")
_MODULE_CHARS = frozenset(f"{_LETTERS}0123456789_.")
_VERSION_CHARS = _MODULE_CHARS | frozenset("-+")


def _text(value: object, field_name: str, allowed: frozenset[str]) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise BackendBoundaryError(
            field_name, "must be a nonempty trimmed built-in str"
        )
    if len(value) > 128 or any(char not in allowed for char in value):
        raise BackendBoundaryError(field_name, "contains unsupported characters")
    return value


def _name(value: object, field_name: str = "name") -> str:
    return _text(value, field_name, _NAME_CHARS)


def _factory(value: object) -> str:
    factory = _text(value, "factory", _IDENTIFIER_CHARS)
    if not (factory[0].isalpha() or factory[0] == "_"):
        raise BackendBoundaryError("factory", "must be an ASCII Python identifier")
    return factory


def _module(value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise BackendBoundaryError("module", "must be a nonempty trimmed built-in str")
    parts = value.split(".")
    if len(value) > 256 or any(
        not part
        or not (part[0].isalpha() or part[0] == "_")
        or any(char not in _MODULE_CHARS or char == "." for char in part)
        for part in parts
    ):
        raise BackendBoundaryError("module", "must be a dotted Python identifier")
    return value


def _version(value: object) -> str:
    return _text(value, "version", _VERSION_CHARS)


def _diagnostic(value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise BackendBoundaryError(
            "diagnostic", "must be a nonempty trimmed built-in str"
        )
    if len(value) > _MAX_DIAGNOSTIC or any(
        ord(char) < 32 or ord(char) == 127 for char in value
    ):
        raise BackendBoundaryError("diagnostic", "must be bounded plain text")
    return value


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    """Immutable provenance for one adapter explicitly allowed by a loader."""

    name: str
    module: str
    factory: str
    version: str

    def __post_init__(self) -> None:
        _name(self.name)
        _module(self.module)
        _factory(self.factory)
        _version(self.version)

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("AdapterSpec cannot be subclassed")


class AdapterLoadState(StrEnum):
    """The exhaustive availability states at the optional import boundary."""

    LOADED = "loaded"
    UNKNOWN = "unknown"
    UNINSTALLED = "uninstalled"
    INVALID = "invalid"
    FACTORY_FAILED = "factory_failed"


def _validate_availability(
    state: object,
    name: object,
    module: object,
    version: object,
    specification: object,
    diagnostic: object,
) -> None:
    if type(state) is not AdapterLoadState:
        raise BackendBoundaryError("state", "must be an exact AdapterLoadState")
    _name(name)
    if state is AdapterLoadState.UNKNOWN:
        if specification is not None or module is not None or version is not None:
            raise BackendBoundaryError(
                "availability", "unknown state has no provenance"
            )
    elif specification is None:
        if (
            state is not AdapterLoadState.INVALID
            or module is not None
            or version is not None
        ):
            raise BackendBoundaryError(
                "availability", "only invalid state may omit provenance"
            )
    else:
        if type(specification) is not AdapterSpec:
            raise BackendBoundaryError("specification", "must be an exact AdapterSpec")
        specification.__post_init__()
        if (
            name != specification.name
            or module != specification.module
            or version != specification.version
        ):
            raise BackendBoundaryError(
                "availability", "provenance must match specification"
            )
    if state is AdapterLoadState.LOADED:
        if diagnostic is not None:
            raise BackendBoundaryError("diagnostic", "must be absent for loaded state")
    else:
        _diagnostic(diagnostic)


@dataclass(frozen=True, slots=True, init=False)
class AdapterAvailability:
    """Sealed neutral availability provenance without adapter-native values."""

    state: AdapterLoadState
    name: str
    module: str | None
    version: str | None
    specification: AdapterSpec | None
    diagnostic: str | None

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise BackendBoundaryError("adapter_availability", "is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("AdapterAvailability cannot be subclassed")

    @classmethod
    def create(
        cls,
        *,
        state: object,
        name: object,
        specification: object = None,
        diagnostic: object = None,
    ) -> AdapterAvailability:
        if cls is not AdapterAvailability:
            raise BackendBoundaryError(
                "adapter_availability", "factory requires exact class"
            )
        if state is AdapterLoadState.LOADED:
            raise BackendBoundaryError("state", "loaded availability is loader-owned")
        module = specification.module if type(specification) is AdapterSpec else None
        version = specification.version if type(specification) is AdapterSpec else None
        _validate_availability(state, name, module, version, specification, diagnostic)
        value = object.__new__(AdapterAvailability)
        for field_name, item in (
            ("state", state),
            ("name", name),
            ("module", module),
            ("version", version),
            ("specification", specification),
            ("diagnostic", diagnostic),
        ):
            object.__setattr__(value, field_name, item)
        return value

    def _assert(self) -> None:
        _validate_availability(
            self.state,
            self.name,
            self.module,
            self.version,
            self.specification,
            self.diagnostic,
        )


@dataclass(frozen=True, slots=True, init=False, eq=False)
class AdapterLoad:
    """A sealed availability record plus an opaque, validated adapter handle."""

    availability: AdapterAvailability
    adapter: BackendAdapter | None = field(repr=False, compare=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise BackendBoundaryError("adapter_load", "is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("AdapterLoad cannot be subclassed")

    @classmethod
    def create(cls, *, availability: object, adapter: object = None) -> AdapterLoad:
        if cls is not AdapterLoad:
            raise BackendBoundaryError("adapter_load", "factory requires exact class")
        if type(availability) is not AdapterAvailability:
            raise BackendBoundaryError(
                "availability", "must be an exact AdapterAvailability"
            )
        availability._assert()
        if (availability.state is AdapterLoadState.LOADED) != (adapter is not None):
            raise BackendBoundaryError(
                "adapter", "must be present exactly for loaded state"
            )
        if availability.state is AdapterLoadState.LOADED:
            assert type(availability.specification) is AdapterSpec
            adapter = _validated_adapter(adapter, availability.specification)
        value = object.__new__(AdapterLoad)
        object.__setattr__(value, "availability", availability)
        object.__setattr__(value, "adapter", adapter)
        return value

    def _assert(self) -> None:
        self.availability._assert()
        if (self.availability.state is AdapterLoadState.LOADED) != (
            self.adapter is not None
        ):
            raise BackendBoundaryError("adapter", "load state and handle disagree")
        if self.availability.state is AdapterLoadState.LOADED:
            assert type(self.availability.specification) is AdapterSpec
            _validated_adapter(self.adapter, self.availability.specification)


OPTIONAL_ADAPTER_SPECS: tuple[AdapterSpec, ...] = ()
"""The core has no optional adapters; applications pass their explicit tuple."""


def _unavailable(
    state: AdapterLoadState,
    *,
    name: str,
    specification: AdapterSpec | None = None,
    diagnostic: str,
) -> AdapterLoad:
    return AdapterLoad.create(
        availability=AdapterAvailability.create(
            state=state,
            name=name,
            specification=specification,
            diagnostic=diagnostic,
        )
    )


def _loaded_availability(specification: AdapterSpec) -> AdapterAvailability:
    """Create loader-owned success provenance only while building a load result."""
    _validate_availability(
        AdapterLoadState.LOADED,
        specification.name,
        specification.module,
        specification.version,
        specification,
        None,
    )
    value = object.__new__(AdapterAvailability)
    for field_name, item in (
        ("state", AdapterLoadState.LOADED),
        ("name", specification.name),
        ("module", specification.module),
        ("version", specification.version),
        ("specification", specification),
        ("diagnostic", None),
    ):
        object.__setattr__(value, field_name, item)
    return value


def _snapshot(specifications: object) -> tuple[AdapterSpec, ...] | None:
    """Copy a bounded exact tuple before lookup so source mutations cannot race."""
    if type(specifications) is not tuple or len(specifications) > _MAX_SPECS:
        return None
    if any(type(specification) is not AdapterSpec for specification in specifications):
        return None
    copied: list[AdapterSpec] = []
    try:
        for specification in cast(tuple[AdapterSpec, ...], specifications):
            copied.append(
                AdapterSpec(
                    name=specification.name,
                    module=specification.module,
                    factory=specification.factory,
                    version=specification.version,
                )
            )
    except Exception:
        return None
    result = tuple(copied)
    if len({specification.name for specification in result}) != len(result):
        return None
    return result


def _method(
    adapter: object, name: str, positional: int, keyword_only: tuple[str, ...]
) -> FunctionType | None:
    """Find and validate one plain instance method without descriptor execution."""
    adapter_type = type(adapter)
    mro = type.__getattribute__(adapter_type, "__mro__")
    for cls in mro:
        namespace = type.__getattribute__(cls, "__dict__")
        candidate = namespace.get(name)
        if candidate is None:
            continue
        if type(candidate) is not FunctionType:
            return None
        code = candidate.__code__
        if (
            code.co_flags
            & (
                CO_VARARGS
                | CO_VARKEYWORDS
                | CO_GENERATOR
                | CO_COROUTINE
                | CO_ASYNC_GENERATOR
            )
            or code.co_posonlyargcount != 0
            or code.co_argcount != positional
            or code.co_kwonlyargcount != len(keyword_only)
            or code.co_varnames[
                code.co_argcount : code.co_argcount + code.co_kwonlyargcount
            ]
            != keyword_only
            or candidate.__defaults__ is not None
            or candidate.__kwdefaults__ is not None
        ):
            return None
        return candidate
    return None


def _validated_adapter(adapter: object, specification: AdapterSpec) -> BackendAdapter:
    """Require an intact neutral adapter surface and matching capabilities."""
    try:
        capabilities = _method(adapter, "capabilities", 1, ())
        if (
            capabilities is None
            or _method(adapter, "supports", 2, ()) is None
            or _method(adapter, "execute", 2, ("options",)) is None
        ):
            raise BackendBoundaryError("adapter", "lacks the declared adapter methods")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        raise BackendBoundaryError("adapter", "method discovery rejected") from error
    try:
        declared = capabilities(adapter)
        if type(declared) is not BackendCapabilities:
            raise BackendBoundaryError("capabilities", "must be exact neutral metadata")
        declared._assert()
        if (
            declared.name != specification.name
            or declared.version != specification.version
        ):
            raise BackendBoundaryError(
                "capabilities", "name or version differs from specification"
            )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BackendBoundaryError:
        raise
    except Exception as error:
        raise BackendBoundaryError(
            "capabilities", "adapter capabilities are invalid"
        ) from error
    return cast(BackendAdapter, adapter)


def _loaded(specification: AdapterSpec, adapter: object) -> AdapterLoad:
    """Build a fresh public wrapper after validating one loaded adapter handle."""
    return AdapterLoad.create(
        availability=_loaded_availability(specification),
        adapter=adapter,
    )


def _adapter_from_module(specification: AdapterSpec, module: object) -> AdapterLoad:
    """Call one trusted factory, then validate a narrow neutral adapter surface."""
    if type(module) is not ModuleType:
        return _unavailable(
            AdapterLoadState.INVALID,
            name=specification.name,
            specification=specification,
            diagnostic="optional import did not return a module",
        )
    factory = module.__dict__.get(specification.factory)
    if type(factory) is not FunctionType:
        return _unavailable(
            AdapterLoadState.INVALID,
            name=specification.name,
            specification=specification,
            diagnostic="declared factory is not a plain Python function",
        )
    try:
        adapter = factory()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return _unavailable(
            AdapterLoadState.FACTORY_FAILED,
            name=specification.name,
            specification=specification,
            diagnostic="optional adapter factory failed",
        )
    try:
        return _loaded(specification, adapter)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return _unavailable(
            AdapterLoadState.INVALID,
            name=specification.name,
            specification=specification,
            diagnostic="adapter capabilities are invalid",
        )


class OptionalAdapterLoader:
    """One explicit immutable allowlist and its bounded successful-load cache."""

    __slots__ = ("_by_name", "_lock", "_specifications", "_successful")

    def __init__(self, specifications: object) -> None:
        snapshot = _snapshot(specifications)
        if snapshot is None:
            raise BackendBoundaryError(
                "specifications", "must be a valid bounded tuple"
            )
        self._specifications = snapshot
        self._by_name = {
            specification.name: specification for specification in snapshot
        }
        self._lock = RLock()
        self._successful: dict[str, BackendAdapter] = {}

    @property
    def specifications(self) -> tuple[AdapterSpec, ...]:
        """Return fresh provenance copies that cannot mutate the loader snapshot."""
        return tuple(
            AdapterSpec(
                name=specification.name,
                module=specification.module,
                factory=specification.factory,
                version=specification.version,
            )
            for specification in self._specifications
        )

    def load(self, name: object) -> AdapterLoad:
        """Lazily load one known name, caching only a validated success."""
        if type(name) is not str:
            return _unavailable(
                AdapterLoadState.INVALID,
                name="invalid",
                diagnostic="adapter name is not a built-in string",
            )
        try:
            requested = _name(name)
        except BackendBoundaryError:
            return _unavailable(
                AdapterLoadState.INVALID,
                name="invalid",
                diagnostic="adapter name is invalid",
            )
        specification = self._by_name.get(requested)
        if specification is None:
            return _unavailable(
                AdapterLoadState.UNKNOWN,
                name=requested,
                diagnostic="adapter is not allowlisted",
            )
        with self._lock:
            cached = self._successful.get(requested)
            if cached is not None:
                try:
                    return _loaded(specification, cached)
                except Exception:
                    del self._successful[requested]
            result = self._load_known(specification)
            if result.availability.state is AdapterLoadState.LOADED:
                assert result.adapter is not None
                self._successful[requested] = result.adapter
            return result

    def _load_known(self, specification: AdapterSpec) -> AdapterLoad:
        try:
            module = importlib.import_module(specification.module)
        except ModuleNotFoundError as error:
            if type(error.name) is str and (
                error.name == specification.module
                or specification.module.startswith(f"{error.name}.")
            ):
                return _unavailable(
                    AdapterLoadState.UNINSTALLED,
                    name=specification.name,
                    specification=specification,
                    diagnostic="optional module is not installed",
                )
            return _unavailable(
                AdapterLoadState.INVALID,
                name=specification.name,
                specification=specification,
                diagnostic="optional module import failed",
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            return _unavailable(
                AdapterLoadState.INVALID,
                name=specification.name,
                specification=specification,
                diagnostic="optional module import failed",
            )
        return _adapter_from_module(specification, module)


def load_optional_adapter(
    name: object, *, specifications: object = OPTIONAL_ADAPTER_SPECS
) -> AdapterLoad:
    """Load once through a fresh explicit loader; convenience has no cache lifetime."""
    try:
        loader = OptionalAdapterLoader(specifications)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        try:
            invalid_name = _name(name)
        except BackendBoundaryError:
            invalid_name = "invalid"
        return _unavailable(
            AdapterLoadState.INVALID,
            name=invalid_name,
            diagnostic="adapter specifications are invalid",
        )
    return loader.load(name)
