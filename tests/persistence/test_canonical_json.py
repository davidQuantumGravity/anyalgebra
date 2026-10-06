"""Behavioral contract for deterministic canonical JSON bytes."""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
import sys

import pytest

from anyalgebra.persistence.canonical import CanonicalJSONError, canonical_json
from anyalgebra.persistence.registry import SchemaError, SerializerRegistry


@dataclass(frozen=True)
class _Payload:
    order: tuple[str, ...] = ("z", "a")


def _encode_payload(value: _Payload) -> dict[str, object]:
    body: dict[str, object] = {}
    fields: dict[str, object] = {
        "a": -123,
        "active": True,
        "text": "Snowman ☃",
        "z": 0,
    }
    for key in value.order:
        body[key] = fields[key]
    for key in ("active", "text"):
        body[key] = fields[key]
    return body


def _registry() -> SerializerRegistry:
    return SerializerRegistry().with_codec(
        "anyalgebra.test.payload", 1, _Payload, _encode_payload, _parse_payload
    )


def _parse_payload(record: object) -> _Payload:
    del record
    return _Payload()


def test_canonical_json_has_golden_utf8_bytes_and_sorted_keys() -> None:
    actual = canonical_json(_Payload(), registry=_registry())

    assert actual == (
        b'{"a":-123,"active":true,"schemaType":"anyalgebra.test.payload",'
        b'"schemaVersion":1,"text":"Snowman \xe2\x98\x83","z":0}'
    )
    assert not actual.startswith(b"\xef\xbb\xbf")
    assert b": " not in actual
    assert not actual.endswith(b"\n")


def test_canonical_json_ignores_encoder_insertion_order() -> None:
    registry = _registry()

    assert canonical_json(_Payload(("z", "a")), registry=registry) == canonical_json(
        _Payload(("a", "z")), registry=registry
    )


def test_canonical_json_uses_literal_unicode_and_deterministic_escaping() -> None:
    @dataclass(frozen=True)
    class Escaped:
        pass

    def encode(value: Escaped) -> dict[str, object]:
        del value
        return {"é": '"\\\b\t\n\f\r\x00/\u2028', "a": "plain"}

    registry = SerializerRegistry().with_codec(
        "anyalgebra.test.escaped", 1, Escaped, encode, _parse_escaped
    )

    assert canonical_json(Escaped(), registry=registry) == (
        b'{"a":"plain","schemaType":"anyalgebra.test.escaped",'
        b'"schemaVersion":1,"\xc3\xa9":"\\"\\\\\\b\\t\\n\\f\\r\\u0000/'
        b'\xe2\x80\xa8"}'
    )


def _parse_escaped(record: object) -> object:
    del record
    return object()


def test_canonical_json_normalizes_very_large_exact_integers() -> None:
    @dataclass(frozen=True)
    class Integer:
        value: int

    def encode(value: Integer) -> dict[str, object]:
        return {"value": value.value}

    registry = SerializerRegistry().with_codec(
        "anyalgebra.test.integer", 1, Integer, encode, _parse_integer
    )
    value = 2**20_000
    original = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(640)
    try:
        actual = canonical_json(Integer(value), registry=registry)
    finally:
        sys.set_int_max_str_digits(original)

    sys.set_int_max_str_digits(0)
    try:
        expected = (
            b'{"schemaType":"anyalgebra.test.integer","schemaVersion":1,'
            b'"value":' + str(value).encode("ascii") + b"}"
        )
    finally:
        sys.set_int_max_str_digits(original)
    assert actual == expected


def test_integer_resource_limit_boundary_is_public_and_exact() -> None:
    @dataclass(frozen=True)
    class Integer:
        value: int

    encoder_calls: list[int] = []
    parser_calls: list[object] = []

    def encode(value: Integer) -> dict[str, object]:
        encoder_calls.append(value.value.bit_length())
        return {"value": value.value}

    def parse(record: object) -> Integer:
        parser_calls.append(record)
        return Integer(0)

    registry = SerializerRegistry().with_codec(
        "anyalgebra.test.integer-boundary", 1, Integer, encode, parse
    )
    accepted = 2**499_999
    rejected = 2**500_000
    assert accepted.bit_length() == 500_000
    assert rejected.bit_length() == 500_001

    original = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(640)
    try:
        actual = canonical_json(Integer(accepted), registry=registry)
    finally:
        sys.set_int_max_str_digits(original)

    sys.set_int_max_str_digits(0)
    try:
        expected = (
            b'{"schemaType":"anyalgebra.test.integer-boundary",'
            b'"schemaVersion":1,"value":' + str(accepted).encode("ascii") + b"}"
        )
        assert json.loads(actual)["value"] == accepted
    finally:
        sys.set_int_max_str_digits(original)
    assert actual == expected
    assert encoder_calls == [500_000]
    assert parser_calls == []

    with pytest.raises(CanonicalJSONError, match="canonical_size_limit") as caught:
        canonical_json(Integer(rejected), registry=registry)
    assert caught.value.code == "canonical_size_limit"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert encoder_calls == [500_000, 500_001]
    assert parser_calls == []


def _parse_integer(record: object) -> object:
    del record
    return object()


@pytest.mark.parametrize("value", [0.0, -0.0, 1.5])
def test_canonical_json_rejects_every_float_as_nonexact(value: float) -> None:
    @dataclass(frozen=True)
    class Float:
        value: float

    def encode(item: Float) -> dict[str, object]:
        return {"value": item.value}

    registry = SerializerRegistry().with_codec(
        "anyalgebra.test.float", 1, Float, encode, _parse_float
    )

    with pytest.raises(CanonicalJSONError, match="nonexact_float") as caught:
        canonical_json(Float(value), registry=registry)
    assert caught.value.code == "nonexact_float"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def _parse_float(record: object) -> object:
    del record
    return object()


def test_canonical_json_preserves_registry_sanitized_malformed_encoder_error() -> None:
    def broken(value: _Payload) -> dict[str, object]:
        del value
        raise RuntimeError("private encoder detail")

    registry = SerializerRegistry().with_codec(
        "anyalgebra.test.payload", 1, _Payload, broken, _parse_payload
    )

    with pytest.raises(SchemaError, match="encoder_failed") as caught:
        canonical_json(_Payload(), registry=registry)
    assert "private encoder detail" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_canonical_json_does_not_render_hostile_unregistered_value() -> None:
    class Hostile:
        def __repr__(self) -> str:
            raise AssertionError("repr must not run")

        def __eq__(self, other: object) -> bool:
            del other
            raise AssertionError("equality must not run")

        def __hash__(self) -> int:
            raise AssertionError("hash must not run")

    with pytest.raises(SchemaError, match="unregistered_value_type"):
        canonical_json(Hostile(), registry=_registry())


def test_canonical_json_uses_a_fresh_registry_record_for_each_call() -> None:
    emitted: list[dict[str, object]] = []
    parser_calls: list[object] = []

    def encode(value: _Payload) -> dict[str, object]:
        del value
        body: dict[str, object] = {"nested": ["fresh"]}
        emitted.append(body)
        return body

    def parse(record: object) -> _Payload:
        parser_calls.append(record)
        return _Payload()

    registry = SerializerRegistry().with_codec(
        "anyalgebra.test.fresh", 1, _Payload, encode, parse
    )

    first = canonical_json(_Payload(), registry=registry)
    assert len(emitted) == 1
    assert parser_calls == []
    emitted[0]["nested"] = ["mutated"]
    second = canonical_json(_Payload(), registry=registry)

    assert len(emitted) == 2
    assert parser_calls == []
    assert first == second
    assert first == (
        b'{"nested":["fresh"],"schemaType":"anyalgebra.test.fresh","schemaVersion":1}'
    )


def _forced_registry_record(
    monkeypatch: pytest.MonkeyPatch, record: object
) -> SerializerRegistry:
    def to_record(registry: SerializerRegistry, value: object) -> object:
        del registry, value
        return record

    monkeypatch.setattr(SerializerRegistry, "to_record", to_record)
    return SerializerRegistry()


@pytest.mark.parametrize(
    "record, code",
    [
        ({"bad": object()}, "invalid_canonical_value"),
        ({1: "bad"}, "invalid_record_key"),
        ({"bad": "\ud800"}, "invalid_unicode_scalar"),
        ({"bad": "x" * 65_537}, "string_limit"),
    ],
)
def test_canonical_json_defensively_preflights_trusted_registry_output(
    monkeypatch: pytest.MonkeyPatch, record: object, code: str
) -> None:
    registry = _forced_registry_record(monkeypatch, record)

    with pytest.raises(CanonicalJSONError, match=code):
        canonical_json(_Payload(), registry=registry)


def test_canonical_json_handles_a_none_value_in_the_trusted_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _forced_registry_record(monkeypatch, {"values": [None, False]})

    assert canonical_json(_Payload(), registry=registry) == b'{"values":[null,false]}'


def test_canonical_json_rejects_invalid_registry_before_any_encoder_call() -> None:
    with pytest.raises(CanonicalJSONError, match="invalid_serializer_registry"):
        canonical_json(_Payload(), registry=object())  # type: ignore[arg-type]


def test_canonical_json_enforces_output_byte_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _forced_registry_record(
        monkeypatch,
        {str(index): "x" * 65_536 for index in range(17)},
    )

    with pytest.raises(CanonicalJSONError, match="canonical_size_limit"):
        canonical_json(_Payload(), registry=registry)


def test_canonical_json_has_an_exact_public_output_byte_boundary() -> None:
    @dataclass(frozen=True)
    class Boundary:
        payload_bytes: int

    def encode(value: Boundary) -> dict[str, object]:
        remaining = value.payload_bytes
        body: dict[str, object] = {}
        for index in range(16):
            width = min(65_536, remaining)
            body[f"p{index:02d}"] = "x" * width
            remaining -= width
        assert remaining == 0
        return body

    registry = SerializerRegistry().with_codec(
        "anyalgebra.test.boundary", 1, Boundary, encode, _parse_boundary
    )
    baseline = canonical_json(Boundary(0), registry=registry)
    payload_bytes = 1_048_576 - len(baseline)

    exact = canonical_json(Boundary(payload_bytes), registry=registry)

    assert len(exact) == 1_048_576
    assert json.loads(exact)["schemaType"] == "anyalgebra.test.boundary"
    with pytest.raises(CanonicalJSONError, match="canonical_size_limit"):
        canonical_json(Boundary(payload_bytes + 1), registry=registry)


def _parse_boundary(record: object) -> object:
    del record
    return object()


def test_canonical_json_matches_json_oracle_for_bounded_generated_records() -> None:
    @dataclass(frozen=True)
    class Generated:
        pairs: tuple[tuple[str, object], ...]

    parser_calls: list[object] = []

    def encode(value: Generated) -> dict[str, object]:
        return dict(value.pairs)

    def parse(record: object) -> Generated:
        parser_calls.append(record)
        return Generated(())

    registry = SerializerRegistry().with_codec(
        "anyalgebra.test.generated", 1, Generated, encode, parse
    )
    fields: dict[str, object] = {
        "plain": "text",
        "unicode": "é☃",
        "escape": '"\\\b\t\n\f\r\x00/\u2028',
        "nested": [None, True, False, {"β": -(2**2000)}],
        "large": 2**2000,
    }

    for seed in range(8):
        keys = list(fields)
        random.Random(seed).shuffle(keys)
        value = Generated(tuple((key, fields[key]) for key in keys))
        actual = canonical_json(value, registry=registry)
        record = registry.to_record(value)
        oracle = json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

        assert actual == oracle
        assert json.loads(actual) == record
    assert parser_calls == []
