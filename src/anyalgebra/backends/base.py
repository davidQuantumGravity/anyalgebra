"""Sealed neutral declarations for optional exact backends (V00-079 only)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
from typing import Protocol, TypeAlias, cast

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.core.parents import SemanticHash
from anyalgebra.structures.outcomes import EvaluationOutcome


MAX_DECLARATIONS = 64
MAX_RESOURCE_LIMIT = 1_000_000
_MAX_TEXT = 256
_ID = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"


class BackendBoundaryError(AnyAlgebraError, ValueError):
    def __init__(self, field: str, reason: str) -> None:
        self.field, self.reason = field, reason
        super().__init__(f"invalid backend boundary {field}: {reason}")


class UnsupportedReason(StrEnum):
    MISSING_OPERATION = "missing_operation"
    MISSING_DOMAIN = "missing_domain"
    MISSING_ALGORITHM = "missing_algorithm"
    INEXACT_CAPABILITY = "inexact_capability"
    RESOURCE_LIMIT = "resource_limit"
    PRECONDITION = "precondition"
    CONVENTION = "convention"


def _text(value: object, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise BackendBoundaryError(field, "must be a nonempty trimmed built-in str")
    if len(value) > _MAX_TEXT:
        raise BackendBoundaryError(field, "text limit exceeded")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise BackendBoundaryError(field, "must not contain control characters")
    return value


def _id(value: object, field: str) -> str:
    text = _text(value, field)
    if any(char not in _ID for char in text):
        raise BackendBoundaryError(field, "must use ASCII identifier characters")
    return text


def _items(value: object, field: str) -> tuple[object, ...]:
    if type(value) is str:
        raise BackendBoundaryError(field, "must be an iterable, not str")
    try:
        iterator = iter(cast(Iterable[object], value))
    except Exception as error:
        raise BackendBoundaryError(field, "must be an iterable") from error
    result: list[object] = []
    try:
        for _ in range(MAX_DECLARATIONS + 1):
            result.append(next(iterator))
    except StopIteration:
        return tuple(result)
    except Exception as error:
        raise BackendBoundaryError(field, "iterable snapshot failed") from error
    raise BackendBoundaryError(field, "declaration limit exceeded")


def _hash(value: object, field: str) -> SemanticHash:
    if type(value) is not SemanticHash:
        raise BackendBoundaryError(field, "must be an exact SemanticHash")
    try:
        return SemanticHash(value.algorithm, value.digest)
    except Exception as error:
        raise BackendBoundaryError(field, "contains an invalid SemanticHash") from error


def _hashes(
    value: object, field: str, *, ordered: bool = False
) -> tuple[SemanticHash, ...]:
    result = tuple(_hash(item, field) for item in _items(value, field))
    if ordered:
        return result
    if len(result) != len(set(result)):
        raise BackendBoundaryError(field, "contains duplicate hashes")
    return tuple(sorted(result, key=str))


def _limits(value: object) -> tuple[tuple[str, int], ...]:
    result: list[tuple[str, int]] = []
    names: set[str] = set()
    for item in _items(value, "limits"):
        if type(item) not in (tuple, list):
            raise BackendBoundaryError(
                "limits", "each declaration must be an exact pair"
            )
        pair = cast(tuple[object, object], item)
        if len(pair) != 2:
            raise BackendBoundaryError(
                "limits", "each declaration must be an exact pair"
            )
        name = _id(pair[0], "limits")
        if name in names:
            raise BackendBoundaryError("limits", "contains duplicate keys")
        if type(pair[1]) is not int or pair[1] < 0 or pair[1] > MAX_RESOURCE_LIMIT:
            raise BackendBoundaryError("limits", "must contain bounded built-in ints")
        names.add(name)
        result.append((name, pair[1]))
    return tuple(sorted(result))


def _digest(record: dict[str, object]) -> SemanticHash:
    try:
        payload = json.dumps(
            record, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise BackendBoundaryError("record", "canonical encoding rejected") from error
    return SemanticHash("sha256", hashlib.sha256(payload).hexdigest())


class _Record:
    __slots__ = ()
    semantic_hash: SemanticHash

    def _record(self) -> dict[str, object]:
        raise NotImplementedError

    def _assert(self) -> None:
        try:
            if (
                type(self.semantic_hash) is not SemanticHash
                or _digest(self._record()) != self.semantic_hash
            ):
                raise BackendBoundaryError(
                    "record", "content address integrity rejected"
                )
        except (AttributeError, TypeError, ValueError) as error:
            raise BackendBoundaryError(
                "record", "content address integrity rejected"
            ) from error

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return False
        other_record = other
        try:
            self._assert()
            other_record._assert()
        except BackendBoundaryError:
            return False
        return self.semantic_hash == other_record.semantic_hash

    def __hash__(self) -> int:
        self._assert()
        return int.from_bytes(
            bytes.fromhex(self.semantic_hash.digest[:16]), "big", signed=True
        )

    def __repr__(self) -> str:
        try:
            self._assert()
            digest = self.semantic_hash.digest
        except BackendBoundaryError:
            digest = "invalid"
        return f"{type(self).__name__}(semantic_hash={digest!r})"


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class CapabilitySpec(_Record):
    operation: str
    domain: str
    algorithm: str
    exact: bool
    precondition_hashes: tuple[SemanticHash, ...]
    convention_hashes: tuple[SemanticHash, ...]
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise BackendBoundaryError("capability_spec", "is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("CapabilitySpec cannot be subclassed")

    @classmethod
    def create(
        cls,
        *,
        operation: object,
        domain: object,
        algorithm: object,
        exact: object,
        precondition_hashes: object = (),
        convention_hashes: object = (),
    ) -> CapabilitySpec:
        if cls is not CapabilitySpec:
            raise BackendBoundaryError(
                "capability_spec", "factory requires exact class"
            )
        if type(exact) is not bool:
            raise BackendBoundaryError("exact", "must be an exact built-in bool")
        value = object.__new__(CapabilitySpec)
        for field, item in (
            ("operation", _id(operation, "operation")),
            ("domain", _id(domain, "domain")),
            ("algorithm", _id(algorithm, "algorithm")),
            ("exact", exact),
            (
                "precondition_hashes",
                _hashes(precondition_hashes, "precondition_hashes"),
            ),
            ("convention_hashes", _hashes(convention_hashes, "convention_hashes")),
        ):
            object.__setattr__(value, field, item)
        object.__setattr__(value, "semantic_hash", _digest(value._record()))
        return value

    def _record(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "domain": self.domain,
            "algorithm": self.algorithm,
            "exact": self.exact,
            "preconditions": tuple(map(str, self.precondition_hashes)),
            "conventions": tuple(map(str, self.convention_hashes)),
        }


def _specs(value: object) -> tuple[CapabilitySpec, ...]:
    result = tuple(item for item in _items(value, "specifications"))
    if any(type(item) is not CapabilitySpec for item in result):
        raise BackendBoundaryError(
            "specifications", "must contain exact CapabilitySpec values"
        )
    specs = cast(tuple[CapabilitySpec, ...], result)
    for item in specs:
        item._assert()
    keys = tuple((item.operation, item.domain, item.algorithm) for item in specs)
    if len(keys) != len(set(keys)):
        raise BackendBoundaryError(
            "specifications",
            "contains duplicate operation/domain/algorithm declarations",
        )
    return tuple(
        sorted(specs, key=lambda item: (item.operation, item.domain, item.algorithm))
    )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class BackendCapabilities(_Record):
    name: str
    version: str
    specifications: tuple[CapabilitySpec, ...]
    convention_hashes: tuple[SemanticHash, ...]
    limits: tuple[tuple[str, int], ...]
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise BackendBoundaryError("backend_capabilities", "is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("BackendCapabilities cannot be subclassed")

    @classmethod
    def create(
        cls,
        *,
        name: object,
        version: object,
        specifications: object,
        convention_hashes: object = (),
        limits: object = (),
    ) -> BackendCapabilities:
        if cls is not BackendCapabilities:
            raise BackendBoundaryError(
                "backend_capabilities", "factory requires exact class"
            )
        checked_specs = _specs(specifications)
        checked_conventions = _hashes(convention_hashes, "convention_hashes")
        derived_conventions = tuple(
            sorted(
                {item for spec in checked_specs for item in spec.convention_hashes},
                key=str,
            )
        )
        if checked_conventions != derived_conventions:
            raise BackendBoundaryError(
                "convention_hashes", "must equal the specifications convention union"
            )
        value = object.__new__(BackendCapabilities)
        for field, item in (
            ("name", _id(name, "name")),
            ("version", _id(version, "version")),
            ("specifications", checked_specs),
            ("convention_hashes", checked_conventions),
            ("limits", _limits(limits)),
        ):
            object.__setattr__(value, field, item)
        object.__setattr__(value, "semantic_hash", _digest(value._record()))
        return value

    def _record(self) -> dict[str, object]:
        for item in self.specifications:
            item._assert()
        return {
            "name": self.name,
            "version": self.version,
            "specifications": tuple(
                str(item.semantic_hash) for item in self.specifications
            ),
            "conventions": tuple(map(str, self.convention_hashes)),
            "limits": self.limits,
        }

    @property
    def operations(self) -> tuple[str, ...]:
        """Return the deterministic operation summary derived from specifications."""
        return tuple(sorted({spec.operation for spec in self.specifications}))

    @property
    def domains(self) -> tuple[str, ...]:
        """Return the deterministic domain summary derived from specifications."""
        return tuple(sorted({spec.domain for spec in self.specifications}))

    @property
    def algorithms(self) -> tuple[str, ...]:
        """Return the deterministic algorithm summary derived from specifications."""
        return tuple(sorted({spec.algorithm for spec in self.specifications}))

    @property
    def exactness(self) -> tuple[tuple[str, str, str], ...]:
        """Return exact operation/domain/algorithm declarations from sealed specs."""
        return tuple(
            sorted(
                {
                    (spec.operation, spec.domain, spec.algorithm)
                    for spec in self.specifications
                    if spec.exact
                }
            )
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class BackendRequest(_Record):
    operation: str
    domain: str
    algorithm: str
    requires_exact: bool
    precondition_hashes: tuple[SemanticHash, ...]
    convention_hashes: tuple[SemanticHash, ...]
    input_hashes: tuple[SemanticHash, ...]
    limits: tuple[tuple[str, int], ...]
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise BackendBoundaryError("backend_request", "is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("BackendRequest cannot be subclassed")

    @classmethod
    def create(
        cls,
        *,
        operation: object,
        domain: object,
        algorithm: object,
        input_hashes: object,
        limits: object,
        requires_exact: object = True,
        precondition_hashes: object = (),
        convention_hashes: object = (),
    ) -> BackendRequest:
        if cls is not BackendRequest:
            raise BackendBoundaryError(
                "backend_request", "factory requires exact class"
            )
        if type(requires_exact) is not bool:
            raise BackendBoundaryError(
                "requires_exact", "must be an exact built-in bool"
            )
        value = object.__new__(BackendRequest)
        for field, item in (
            ("operation", _id(operation, "operation")),
            ("domain", _id(domain, "domain")),
            ("algorithm", _id(algorithm, "algorithm")),
            ("requires_exact", requires_exact),
            (
                "precondition_hashes",
                _hashes(precondition_hashes, "precondition_hashes"),
            ),
            ("convention_hashes", _hashes(convention_hashes, "convention_hashes")),
            ("input_hashes", _hashes(input_hashes, "input_hashes", ordered=True)),
            ("limits", _limits(limits)),
        ):
            object.__setattr__(value, field, item)
        object.__setattr__(value, "semantic_hash", _digest(value._record()))
        return value

    def _record(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "domain": self.domain,
            "algorithm": self.algorithm,
            "exact": self.requires_exact,
            "preconditions": tuple(map(str, self.precondition_hashes)),
            "conventions": tuple(map(str, self.convention_hashes)),
            "inputs": tuple(map(str, self.input_hashes)),
            "limits": self.limits,
        }


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class BackendOptions(_Record):
    convention_hashes: tuple[SemanticHash, ...]
    limits: tuple[tuple[str, int], ...]
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise BackendBoundaryError("backend_options", "is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("BackendOptions cannot be subclassed")

    @classmethod
    def create(
        cls, *, convention_hashes: object = (), limits: object = ()
    ) -> BackendOptions:
        if cls is not BackendOptions:
            raise BackendBoundaryError(
                "backend_options", "factory requires exact class"
            )
        value = object.__new__(BackendOptions)
        object.__setattr__(
            value, "convention_hashes", _hashes(convention_hashes, "convention_hashes")
        )
        object.__setattr__(value, "limits", _limits(limits))
        object.__setattr__(value, "semantic_hash", _digest(value._record()))
        return value

    def _record(self) -> dict[str, object]:
        return {
            "conventions": tuple(map(str, self.convention_hashes)),
            "limits": self.limits,
        }


def _capabilities(value: object) -> BackendCapabilities:
    if type(value) is not BackendCapabilities:
        raise BackendBoundaryError(
            "capabilities", "must be an exact BackendCapabilities"
        )
    value._assert()
    return value


def _request(value: object) -> BackendRequest:
    if type(value) is not BackendRequest:
        raise BackendBoundaryError("request", "must be an exact BackendRequest")
    value._assert()
    return value


def _reason(
    capabilities: BackendCapabilities, request: BackendRequest
) -> UnsupportedReason | None:
    specs = capabilities.specifications
    if not any(item.operation == request.operation for item in specs):
        return UnsupportedReason.MISSING_OPERATION
    if not any(
        item.operation == request.operation and item.domain == request.domain
        for item in specs
    ):
        return UnsupportedReason.MISSING_DOMAIN
    matches = tuple(
        item
        for item in specs
        if (item.operation, item.domain, item.algorithm)
        == (request.operation, request.domain, request.algorithm)
    )
    if not matches:
        return UnsupportedReason.MISSING_ALGORITHM
    spec = matches[0]
    if request.requires_exact and not spec.exact:
        return UnsupportedReason.INEXACT_CAPABILITY
    if request.precondition_hashes != spec.precondition_hashes:
        return UnsupportedReason.PRECONDITION
    if request.convention_hashes != spec.convention_hashes:
        return UnsupportedReason.CONVENTION
    declared = dict(capabilities.limits)
    if any(
        name not in declared or value > declared[name] for name, value in request.limits
    ):
        return UnsupportedReason.RESOURCE_LIMIT
    return None


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class BackendProvenance(_Record):
    request_hash: SemanticHash
    name: str
    version: str
    specification_hash: SemanticHash
    convention_hashes: tuple[SemanticHash, ...]
    precondition_hashes: tuple[SemanticHash, ...]
    capabilities_hash: SemanticHash
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise BackendBoundaryError("backend_provenance", "is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("BackendProvenance cannot be subclassed")

    @classmethod
    def create(cls, request: object, capabilities: object) -> BackendProvenance:
        if cls is not BackendProvenance:
            raise BackendBoundaryError(
                "backend_provenance", "factory requires exact class"
            )
        req, cap = _request(request), _capabilities(capabilities)
        if _reason(cap, req) is not None:
            raise BackendBoundaryError(
                "provenance", "requires a declared supported request"
            )
        spec = next(
            item
            for item in cap.specifications
            if (item.operation, item.domain, item.algorithm)
            == (req.operation, req.domain, req.algorithm)
        )
        value = object.__new__(BackendProvenance)
        for field, item in (
            ("request_hash", req.semantic_hash),
            ("name", cap.name),
            ("version", cap.version),
            ("specification_hash", spec.semantic_hash),
            ("convention_hashes", req.convention_hashes),
            ("precondition_hashes", req.precondition_hashes),
            ("capabilities_hash", cap.semantic_hash),
        ):
            object.__setattr__(value, field, item)
        object.__setattr__(value, "semantic_hash", _digest(value._record()))
        return value

    def _record(self) -> dict[str, object]:
        return {
            "request": str(self.request_hash),
            "name": self.name,
            "version": self.version,
            "specification": str(self.specification_hash),
            "conventions": tuple(map(str, self.convention_hashes)),
            "preconditions": tuple(map(str, self.precondition_hashes)),
            "capabilities": str(self.capabilities_hash),
        }


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class BackendSupported(_Record):
    request_hash: SemanticHash
    provenance: BackendProvenance
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise BackendBoundaryError("backend_supported", "is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("BackendSupported cannot be subclassed")

    @classmethod
    def create(cls, request: object, capabilities: object) -> BackendSupported:
        if cls is not BackendSupported:
            raise BackendBoundaryError(
                "backend_supported", "factory requires exact class"
            )
        req, cap = _request(request), _capabilities(capabilities)
        if _reason(cap, req) is not None:
            raise BackendBoundaryError(
                "backend_supported", "request is not declared supported"
            )
        value = object.__new__(BackendSupported)
        object.__setattr__(value, "request_hash", req.semantic_hash)
        object.__setattr__(value, "provenance", BackendProvenance.create(req, cap))
        object.__setattr__(value, "semantic_hash", _digest(value._record()))
        return value

    def _record(self) -> dict[str, object]:
        self.provenance._assert()
        return {
            "request": str(self.request_hash),
            "provenance": str(self.provenance.semantic_hash),
        }


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class BackendUnsupported(_Record):
    request_hash: SemanticHash
    capabilities_hash: SemanticHash
    reason: UnsupportedReason
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise BackendBoundaryError("backend_unsupported", "is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("BackendUnsupported cannot be subclassed")

    @classmethod
    def create(
        cls, request: object, capabilities: object, reason: object
    ) -> BackendUnsupported:
        if cls is not BackendUnsupported:
            raise BackendBoundaryError(
                "backend_unsupported", "factory requires exact class"
            )
        req, cap = _request(request), _capabilities(capabilities)
        if type(reason) is not UnsupportedReason:
            raise BackendBoundaryError("reason", "must be an exact UnsupportedReason")
        if _reason(cap, req) is None or reason is not _reason(cap, req):
            raise BackendBoundaryError(
                "reason", "must match a declared unsupported request"
            )
        value = object.__new__(BackendUnsupported)
        object.__setattr__(value, "request_hash", req.semantic_hash)
        object.__setattr__(value, "capabilities_hash", cap.semantic_hash)
        object.__setattr__(value, "reason", reason)
        object.__setattr__(value, "semantic_hash", _digest(value._record()))
        return value

    def _record(self) -> dict[str, object]:
        return {
            "request": str(self.request_hash),
            "capabilities": str(self.capabilities_hash),
            "reason": self.reason.value,
        }


CapabilityResult: TypeAlias = BackendSupported | BackendUnsupported


def negotiate(capabilities: object, request: object) -> CapabilityResult:
    cap, req = _capabilities(capabilities), _request(request)
    reason = _reason(cap, req)
    return (
        BackendSupported.create(req, cap)
        if reason is None
        else BackendUnsupported.create(req, cap, reason)
    )


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class BackendResult(_Record):
    request_hash: SemanticHash
    provenance: BackendProvenance
    options_hash: SemanticHash
    value_hash: SemanticHash
    semantic_hash: SemanticHash

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise BackendBoundaryError("backend_result", "is factory-owned")

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("BackendResult cannot be subclassed")

    @classmethod
    def create(
        cls, *, request: object, provenance: object, options: object, value_hash: object
    ) -> BackendResult:
        if cls is not BackendResult:
            raise BackendBoundaryError("backend_result", "factory requires exact class")
        req = _request(request)
        if (
            type(provenance) is not BackendProvenance
            or type(options) is not BackendOptions
        ):
            raise BackendBoundaryError(
                "result", "requires exact neutral provenance and options"
            )
        provenance._assert()
        options._assert()
        if (
            provenance.request_hash != req.semantic_hash
            or provenance.convention_hashes != options.convention_hashes
            or provenance.precondition_hashes != req.precondition_hashes
            or options.limits != req.limits
        ):
            raise BackendBoundaryError(
                "result", "options or provenance do not bind request metadata"
            )
        value = object.__new__(BackendResult)
        object.__setattr__(value, "request_hash", req.semantic_hash)
        object.__setattr__(value, "provenance", provenance)
        object.__setattr__(value, "options_hash", options.semantic_hash)
        object.__setattr__(value, "value_hash", _hash(value_hash, "value_hash"))
        object.__setattr__(value, "semantic_hash", _digest(value._record()))
        return value

    def _record(self) -> dict[str, object]:
        self.provenance._assert()
        return {
            "request": str(self.request_hash),
            "provenance": str(self.provenance.semantic_hash),
            "options": str(self.options_hash),
            "value": str(self.value_hash),
        }


class BackendAdapter(Protocol):
    def capabilities(self) -> BackendCapabilities: ...
    def supports(self, request: BackendRequest) -> CapabilityResult: ...
    def execute(
        self, request: BackendRequest, *, options: BackendOptions
    ) -> EvaluationOutcome[BackendResult]: ...
