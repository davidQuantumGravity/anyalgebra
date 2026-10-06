"""Canonical UTF-8 JSON bytes for one trusted registered value.

The contract is deliberately narrower than general-purpose JSON serialization:

* ``canonical_json`` accepts a value only through an exact
  :class:`SerializerRegistry`; it calls ``registry.to_record(value)`` and never
  accepts a caller-supplied record or invokes a decoder.
* The output is UTF-8 without a byte-order mark, has no insignificant
  whitespace, and uses object keys sorted lexicographically by Unicode code
  point.  Sorting and escaping are independent of the process locale.
* Unicode scalar values are emitted literally; quotation marks, reverse
  solidi, and U+0000 through U+001F use the standard deterministic JSON
  escapes.  Lone UTF-16 surrogate code points are rejected.
* Exact built-in integers are emitted as normalized base-10 JSON numbers.
  Floats, including ``-0.0`` and finite floats, have no accepted exact
  encoding and are rejected as ``nonexact_float``.
* V00-070 owns the bounded, exact JSON-shaped record snapshot.  This module
  preflights text and a measured practical v0.0 integer cap of 500,000 bits
  (about 150,500 decimal digits), renders accepted integers without
  ``str(int)``, caps output at one MiB, and reports only sanitized
  :class:`SchemaError` codes.

Canonical bytes are a serialization boundary only.  Hashing, migrations, and
model-specific codecs intentionally remain outside this module.
"""

from __future__ import annotations

from typing import cast

from anyalgebra.persistence.registry import SchemaError, SerializerRegistry


_MAX_STRING_CODE_POINTS = 65_536
_MAX_OUTPUT_BYTES = 1_048_576
_MAX_INTEGER_BITS = 500_000


class CanonicalJSONError(SchemaError):
    """A sanitized failure in canonical JSON preflight or rendering."""


def _preflight(value: object) -> None:
    """Reject noncanonical numeric and Unicode values before JSON encoding."""
    if type(value) in (type(None), bool):
        return
    if type(value) is int:
        if value.bit_length() > _MAX_INTEGER_BITS:
            raise CanonicalJSONError("canonical_size_limit")
        return
    if type(value) is float:
        raise CanonicalJSONError("nonexact_float")
    if type(value) is str:
        if len(value) > _MAX_STRING_CODE_POINTS:
            raise CanonicalJSONError("string_limit")
        if any("\ud800" <= character <= "\udfff" for character in value):
            raise CanonicalJSONError("invalid_unicode_scalar")
        return
    if type(value) is list:
        for item in value:
            _preflight(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise CanonicalJSONError("invalid_record_key")
            _preflight(key)
            _preflight(item)
        return
    raise CanonicalJSONError("invalid_canonical_value")


def _integer_decimal(value: int) -> bytes:
    """Render one exact integer without using Python's unbounded ``str(int)``."""
    if value == 0:
        return b"0"
    negative = value < 0
    remainder = -value if negative else value
    groups: list[int] = []
    while remainder:
        remainder, group = divmod(remainder, 1_000_000_000)
        groups.append(group)
    leading = str(groups[-1])
    decimal = (
        ("-" if negative else "")
        + leading
        + "".join(f"{group:09d}" for group in reversed(groups[:-1]))
    )
    return decimal.encode("ascii")


def _quoted(value: str) -> bytes:
    """Encode one preflighted JSON string with fixed, locale-free escapes."""
    quoted = bytearray(b'"')
    escapes = {
        '"': b'\\"',
        "\\": b"\\\\",
        "\b": b"\\b",
        "\t": b"\\t",
        "\n": b"\\n",
        "\f": b"\\f",
        "\r": b"\\r",
    }
    for character in value:
        escaped = escapes.get(character)
        if escaped is not None:
            quoted.extend(escaped)
        elif ord(character) <= 0x1F:
            quoted.extend(f"\\u{ord(character):04x}".encode("ascii"))
        else:
            quoted.extend(character.encode("utf-8"))
    quoted.append(ord('"'))
    return bytes(quoted)


def _append(output: bytearray, fragment: bytes) -> None:
    """Append one bounded fragment without ever emitting a partial result."""
    if len(output) + len(fragment) > _MAX_OUTPUT_BYTES:
        raise CanonicalJSONError("canonical_size_limit")
    output.extend(fragment)


def _write(value: object, output: bytearray) -> None:
    """Write one preflighted exact JSON value in canonical form."""
    if value is None:
        _append(output, b"null")
        return
    if value is True:
        _append(output, b"true")
        return
    if value is False:
        _append(output, b"false")
        return
    if type(value) is int:
        _append(output, _integer_decimal(value))
        return
    if type(value) is str:
        _append(output, _quoted(value))
        return
    if type(value) is list:
        _append(output, b"[")
        for index, item in enumerate(value):
            if index:
                _append(output, b",")
            _write(item, output)
        _append(output, b"]")
        return
    record = cast(dict[str, object], value)
    _append(output, b"{")
    for index, key in enumerate(sorted(record)):
        if index:
            _append(output, b",")
        _append(output, _quoted(key))
        _append(output, b":")
        _write(record[key], output)
    _append(output, b"}")


def canonical_json(value: object, *, registry: SerializerRegistry) -> bytes:
    """Encode one registered value as deterministic canonical UTF-8 JSON bytes."""
    if type(registry) is not SerializerRegistry:
        raise CanonicalJSONError("invalid_serializer_registry")
    record = registry.to_record(value)
    _preflight(record)
    output = bytearray()
    _write(record, output)
    return bytes(output)
