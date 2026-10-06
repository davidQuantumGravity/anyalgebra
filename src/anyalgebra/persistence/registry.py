"""Safe, immutable allow-list registry for tagged JSON-shaped records.

The registry is intentionally only a codec dispatch boundary.  It never imports
names from input, resolves callables from input, or constructs an arbitrary
class.  A caller may execute only the exact function objects supplied while
building a registry, and each parser result is checked against its registered
exact value type.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import FunctionType, MappingProxyType
from typing import TypeAlias, cast

from anyalgebra.core.errors import AnyAlgebraError


JSONValue: TypeAlias = (
    bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"] | None
)
RecordParser = FunctionType
RecordEncoder = FunctionType

_MAX_DEPTH = 32
_MAX_CONTAINER_ITEMS = 256
_MAX_NODES = 4096
_RESERVED_KEYS = frozenset(("schemaType", "schemaVersion"))


class SchemaError(AnyAlgebraError, ValueError):
    """A record or codec failed the safe serialization boundary."""

    def __init__(self, code: str) -> None:
        """Expose a deterministic machine-readable code without input rendering."""
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class _Codec:
    """One trusted, in-memory codec registered by application code."""

    schema_type: str
    schema_version: int
    value_type: type[object]
    encoder: RecordEncoder
    parser: RecordParser


def _snapshot_json(
    value: object,
    *,
    depth: int,
    budget: list[int],
    ancestors: tuple[object, ...] = (),
) -> object:
    """Make one bounded immutable JSON snapshot without invoking user protocols."""
    budget[0] -= 1
    if budget[0] < 0:
        raise SchemaError("record_node_limit")
    if depth > _MAX_DEPTH:
        raise SchemaError("record_depth_limit")
    if type(value) in (type(None), bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise SchemaError("invalid_record_value")
        return value
    if type(value) is list:
        if any(parent is value for parent in ancestors):
            raise SchemaError("record_cycle")
        if len(value) > _MAX_CONTAINER_ITEMS:
            raise SchemaError("record_container_limit")
        return tuple(
            _snapshot_json(
                item,
                depth=depth + 1,
                budget=budget,
                ancestors=(*ancestors, value),
            )
            for item in value
        )
    if type(value) is dict:
        if any(parent is value for parent in ancestors):
            raise SchemaError("record_cycle")
        if len(value) > _MAX_CONTAINER_ITEMS:
            raise SchemaError("record_container_limit")
        snapshot: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise SchemaError("invalid_record_key")
            snapshot[key] = _snapshot_json(
                item,
                depth=depth + 1,
                budget=budget,
                ancestors=(*ancestors, value),
            )
        return MappingProxyType(snapshot)
    raise SchemaError("invalid_record_value")


def _record_snapshot(record: object) -> Mapping[str, object]:
    """Require one exact dictionary and freeze a bounded deep copy of it."""
    if type(record) is not dict:
        raise SchemaError("invalid_record")
    snapshot = _snapshot_json(record, depth=0, budget=[_MAX_NODES])
    assert isinstance(snapshot, Mapping)
    return snapshot


def _mutable_json(value: object) -> JSONValue:
    """Copy a trusted immutable snapshot into a caller-owned JSON-shaped value."""
    if isinstance(value, Mapping):
        return {key: _mutable_json(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_mutable_json(item) for item in value]
    assert type(value) in (type(None), bool, int, float, str)
    return cast(JSONValue, value)


def _stable_tag(value: object) -> str:
    """Validate one compact, exact, human-readable stable type tag."""
    if type(value) is not str:
        raise SchemaError("invalid_schema_type")
    if not value or len(value) > 128 or value != value.strip():
        raise SchemaError("invalid_schema_type")
    if value[0] not in "abcdefghijklmnopqrstuvwxyz":
        raise SchemaError("invalid_schema_type")
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789._-"
    if any(character not in allowed for character in value):
        raise SchemaError("invalid_schema_type")
    return value


def _schema_version(value: object) -> int:
    """Validate an exact positive built-in schema version."""
    if type(value) is not int or value < 1 or value > 2_147_483_647:
        raise SchemaError("invalid_schema_version")
    return value


class SerializerRegistry:
    """Persistent allow-list of trusted record codecs.

    ``with_codec`` creates a new registry, retaining the old registry exactly as
    it was.  Multiple versions for one stable tag are permitted only when they
    decode to the same exact type; encoding selects the greatest registered
    version.  Migrations remain a later, separate boundary.
    """

    __slots__ = ("_codecs",)
    _codecs: tuple[_Codec, ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self) -> None:
        """Create an empty immutable registry."""
        object.__setattr__(self, "_codecs", ())

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Keep persistent successor construction confined to this exact class."""
        del cls, kwargs
        raise TypeError("SerializerRegistry cannot be subclassed")

    def __setattr__(self, name: str, value: object) -> None:
        """Reject direct mutation of persistent registry state."""
        del name, value
        raise AttributeError("SerializerRegistry is immutable")

    def __delattr__(self, name: str) -> None:
        """Reject direct deletion of persistent registry state."""
        del name
        raise AttributeError("SerializerRegistry is immutable")

    def __repr__(self) -> str:
        """Render bounded registry metadata without codec callable details."""
        return f"SerializerRegistry(codec_count={len(self._codecs)})"

    @classmethod
    def _with_codecs(cls, codecs: tuple[_Codec, ...]) -> SerializerRegistry:
        """Build an internal persistent successor without exposing mutable state."""
        del cls
        successor = object.__new__(SerializerRegistry)
        object.__setattr__(successor, "_codecs", codecs)
        return successor

    def with_codec(
        self,
        schema_type: object,
        schema_version: object,
        value_type: object,
        encoder: object,
        parser: object,
    ) -> SerializerRegistry:
        """Return a successor containing one explicitly supplied codec pair."""
        tag = _stable_tag(schema_type)
        version = _schema_version(schema_version)
        if type(value_type) is not type:
            raise SchemaError("invalid_value_type")
        if type(encoder) is not FunctionType or type(parser) is not FunctionType:
            raise SchemaError("invalid_codec_function")
        for existing in self._codecs:
            if existing.schema_type == tag and existing.schema_version == version:
                raise SchemaError("duplicate_codec")
            if existing.schema_type == tag and existing.value_type is not value_type:
                raise SchemaError("schema_type_conflict")
            if existing.value_type is value_type and existing.schema_type != tag:
                raise SchemaError("value_type_conflict")
        codec = _Codec(tag, version, value_type, encoder, parser)
        return self._with_codecs((*self._codecs, codec))

    def to_record(self, value: object) -> dict[str, JSONValue]:
        """Encode one registered exact type into a fresh tagged record."""
        candidates = tuple(
            codec for codec in self._codecs if type(value) is codec.value_type
        )
        if not candidates:
            raise SchemaError("unregistered_value_type")
        codec = max(candidates, key=lambda candidate: candidate.schema_version)
        try:
            body = codec.encoder(value)
        except Exception:
            failed = True
        else:
            failed = False
        if failed:
            raise SchemaError("encoder_failed")
        if type(body) is not dict:
            raise SchemaError("invalid_encoder_output")
        snapshot = _record_snapshot(body)
        if any(key in snapshot for key in _RESERVED_KEYS):
            raise SchemaError("reserved_record_key")
        record: dict[str, JSONValue] = {
            "schemaType": codec.schema_type,
            "schemaVersion": codec.schema_version,
        }
        for key, item in snapshot.items():
            record[key] = _mutable_json(item)
        return record

    def from_record(self, record: object) -> object:
        """Parse one exact, tagged snapshot using only a registered parser."""
        snapshot = _record_snapshot(record)
        if "schemaType" not in snapshot:
            raise SchemaError("missing_schema_type")
        if "schemaVersion" not in snapshot:
            raise SchemaError("missing_schema_version")
        tag = _stable_tag(snapshot["schemaType"])
        version = _schema_version(snapshot["schemaVersion"])
        tagged = tuple(codec for codec in self._codecs if codec.schema_type == tag)
        if not tagged:
            raise SchemaError("unknown_schema_type")
        codec = next(
            (candidate for candidate in tagged if candidate.schema_version == version),
            None,
        )
        if codec is None:
            raise SchemaError("unknown_schema_version")
        try:
            parsed = codec.parser(snapshot)
        except Exception:
            failed = True
        else:
            failed = False
        if failed:
            raise SchemaError("parser_failed")
        if type(parsed) is not codec.value_type:
            raise SchemaError("invalid_parser_output")
        return parsed
