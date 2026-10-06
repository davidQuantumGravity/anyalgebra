"""Behavioral contract for the safe, allow-listed record registry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import operator
from types import MappingProxyType
from typing import cast

import pytest

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.persistence.registry import SchemaError, SerializerRegistry


@dataclass(frozen=True)
class _Note:
    text: str


def _encode_note(value: _Note) -> dict[str, object]:
    return {"text": value.text}


def _parse_note(record: object) -> _Note:
    assert isinstance(record, MappingProxyType)
    text = record["text"]
    assert type(text) is str
    return _Note(text)


def _registry() -> SerializerRegistry:
    return SerializerRegistry().with_codec(
        "anyalgebra.test.note", 1, _Note, _encode_note, _parse_note
    )


def test_registered_exact_type_round_trips_through_tagged_schema_record() -> None:
    registry = _registry()

    record = registry.to_record(_Note("stable"))

    assert record == {
        "schemaType": "anyalgebra.test.note",
        "schemaVersion": 1,
        "text": "stable",
    }
    assert type(record) is dict
    assert registry.from_record(record) == _Note("stable")


def test_registry_is_persistent_and_does_not_mutate_prior_instances() -> None:
    empty = SerializerRegistry()
    registered = _registry()

    with pytest.raises(SchemaError, match="unregistered_value_type"):
        empty.to_record(_Note("absent"))
    assert registered.to_record(_Note("present"))["text"] == "present"


def test_unknown_schema_type_and_version_have_explicit_diagnostics() -> None:
    registry = _registry()

    with pytest.raises(SchemaError, match="unknown_schema_type") as unknown_type:
        registry.from_record(
            {"schemaType": "anyalgebra.test.unknown", "schemaVersion": 1}
        )
    assert unknown_type.value.code == "unknown_schema_type"
    with pytest.raises(SchemaError, match="unknown_schema_version") as unknown_version:
        registry.from_record({"schemaType": "anyalgebra.test.note", "schemaVersion": 2})
    assert unknown_version.value.code == "unknown_schema_version"


@pytest.mark.parametrize(
    "record, code",
    [
        (object(), "invalid_record"),
        ({}, "missing_schema_type"),
        ({"schemaType": "anyalgebra.test.note"}, "missing_schema_version"),
        (
            {"schemaType": "anyalgebra.test.note", "schemaVersion": True},
            "invalid_schema_version",
        ),
        (
            {
                "schemaType": "anyalgebra.test.note",
                "schemaVersion": 1,
                "text": {1: "bad"},
            },
            "invalid_record_key",
        ),
    ],
)
def test_record_input_is_strictly_structural_and_json_shaped(
    record: object, code: str
) -> None:
    with pytest.raises(SchemaError, match=code):
        _registry().from_record(record)


def test_encoder_output_is_validated_and_reserved_keys_cannot_be_spoofed() -> None:
    def invalid_encoder(value: _Note) -> dict[str, object]:
        del value
        return {"schemaType": "forged"}

    registry = SerializerRegistry().with_codec(
        "anyalgebra.test.note", 1, _Note, invalid_encoder, _parse_note
    )

    with pytest.raises(SchemaError, match="reserved_record_key"):
        registry.to_record(_Note("forged"))


def test_parser_receives_an_immutable_deep_snapshot_and_output_is_exact_type() -> None:
    observed: list[object] = []

    def parser(record: object) -> _Note:
        assert isinstance(record, MappingProxyType)
        nested = record["nested"]
        assert isinstance(nested, tuple)
        observed.append(record)
        return _Note("ok")

    registry = SerializerRegistry().with_codec(
        "anyalgebra.test.note", 1, _Note, _encode_note, parser
    )
    nested_source = [{"value": "before"}]
    source: dict[str, object] = {
        "schemaType": "anyalgebra.test.note",
        "schemaVersion": 1,
        "nested": nested_source,
    }

    assert registry.from_record(source) == _Note("ok")
    nested_source[0]["value"] = "after"
    snapshot = observed[0]
    snapshot_mapping = cast(Mapping[str, object], snapshot)
    assert snapshot_mapping["nested"] == (MappingProxyType({"value": "before"}),)
    with pytest.raises(TypeError):
        operator.setitem(cast(dict[str, object], snapshot), "extra", "blocked")


def test_duplicate_and_conflicting_registrations_are_rejected() -> None:
    registry = _registry()

    with pytest.raises(SchemaError, match="duplicate_codec"):
        registry.with_codec("anyalgebra.test.note", 1, _Note, _encode_note, _parse_note)

    @dataclass(frozen=True)
    class Other:
        text: str

    def encode_other(value: Other) -> dict[str, object]:
        return {"text": value.text}

    def parse_other(record: object) -> Other:
        del record
        return Other("other")

    with pytest.raises(SchemaError, match="schema_type_conflict"):
        registry.with_codec("anyalgebra.test.note", 2, Other, encode_other, parse_other)
    with pytest.raises(SchemaError, match="value_type_conflict"):
        registry.with_codec(
            "anyalgebra.test.other", 1, _Note, _encode_note, _parse_note
        )


def test_same_stable_type_can_register_a_new_version_and_encoder_uses_latest() -> None:
    registry = _registry().with_codec(
        "anyalgebra.test.note", 2, _Note, _encode_note, _parse_note
    )

    assert registry.to_record(_Note("latest"))["schemaVersion"] == 2
    assert registry.from_record(
        {"schemaType": "anyalgebra.test.note", "schemaVersion": 1, "text": "old"}
    ) == _Note("old")


def test_registered_functions_are_only_execution_path_and_failures_are_sanitized() -> (
    None
):
    calls: list[str] = []

    def exploding_encoder(value: _Note) -> dict[str, object]:
        calls.append(value.text)
        raise RuntimeError("secret")

    registry = SerializerRegistry().with_codec(
        "anyalgebra.test.note", 1, _Note, exploding_encoder, _parse_note
    )

    with pytest.raises(SchemaError, match="encoder_failed") as caught:
        registry.to_record(_Note("called"))
    assert calls == ["called"]
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "secret" not in str(caught.value)


def test_hostile_values_do_not_trigger_repr_equality_or_hash() -> None:
    class Hostile:
        def __repr__(self) -> str:
            raise AssertionError("repr must not run")

        def __eq__(self, other: object) -> bool:
            del other
            raise AssertionError("equality must not run")

        def __hash__(self) -> int:
            raise AssertionError("hash must not run")

    with pytest.raises(SchemaError, match="unregistered_value_type"):
        _registry().to_record(Hostile())
    with pytest.raises(SchemaError, match="invalid_record"):
        _registry().from_record(Hostile())


def test_registry_has_no_mutable_state_or_subclass_extension_boundary() -> None:
    registry = _registry()

    assert not hasattr(registry, "__dict__")
    assert "0x" not in repr(registry)
    with pytest.raises(TypeError, match="unhashable"):
        hash(registry)
    with pytest.raises(AttributeError, match="immutable"):
        registry._codecs = ()
    with pytest.raises(AttributeError, match="immutable"):
        del registry._codecs
    with pytest.raises(TypeError, match="cannot be subclassed"):

        class DerivedRegistry(SerializerRegistry):
            pass


def test_schema_error_is_a_package_error_without_input_rendering() -> None:
    error = SchemaError("unknown_schema_type")

    assert isinstance(error, AnyAlgebraError)
    assert error.code == "unknown_schema_type"
    assert str(error) == "unknown_schema_type"


def test_invalid_registration_arguments_are_rejected_before_registry_change() -> None:
    cases = (
        (" bad", 1, _Note, _encode_note, _parse_note, "invalid_schema_type"),
        ("1bad", 1, _Note, _encode_note, _parse_note, "invalid_schema_type"),
        ("bad!", 1, _Note, _encode_note, _parse_note, "invalid_schema_type"),
        (
            "anyalgebra.test.note",
            0,
            _Note,
            _encode_note,
            _parse_note,
            "invalid_schema_version",
        ),
        (
            "anyalgebra.test.note",
            1,
            object(),
            _encode_note,
            _parse_note,
            "invalid_value_type",
        ),
        (
            "anyalgebra.test.note",
            1,
            _Note,
            object(),
            _parse_note,
            "invalid_codec_function",
        ),
        (
            "anyalgebra.test.note",
            1,
            _Note,
            _encode_note,
            object(),
            "invalid_codec_function",
        ),
    )
    for tag, version, value_type, encoder, parser, code in cases:
        with pytest.raises(SchemaError, match=code):
            SerializerRegistry().with_codec(tag, version, value_type, encoder, parser)


def test_snapshot_limits_nonfinite_values_and_cycles_are_explicit() -> None:
    registry = _registry()
    overlong = list(range(257))
    nested: object = "leaf"
    for _ in range(33):
        nested = [nested]
    cyclic: list[object] = []
    cyclic.append(cyclic)
    cyclic_mapping: dict[str, object] = {}
    cyclic_mapping["self"] = cyclic_mapping
    oversized_mapping = {str(index): index for index in range(257)}
    node_limited = {
        "schemaType": "anyalgebra.test.note",
        "schemaVersion": 1,
        **{f"field{index}": list(range(256)) for index in range(17)},
    }

    for record, code in (
        (
            {
                "schemaType": "anyalgebra.test.note",
                "schemaVersion": 1,
                "text": float("nan"),
            },
            "invalid_record_value",
        ),
        (
            {"schemaType": "anyalgebra.test.note", "schemaVersion": 1, "text": 1.5},
            "parser_failed",
        ),
        (
            {
                "schemaType": "anyalgebra.test.note",
                "schemaVersion": 1,
                "text": overlong,
            },
            "record_container_limit",
        ),
        (
            {
                "schemaType": "anyalgebra.test.note",
                "schemaVersion": 1,
                "text": oversized_mapping,
            },
            "record_container_limit",
        ),
        (
            {
                "schemaType": "anyalgebra.test.note",
                "schemaVersion": 1,
                "text": nested,
            },
            "record_depth_limit",
        ),
        (
            {
                "schemaType": "anyalgebra.test.note",
                "schemaVersion": 1,
                "text": cyclic,
            },
            "record_cycle",
        ),
        (
            {
                "schemaType": "anyalgebra.test.note",
                "schemaVersion": 1,
                "text": cyclic_mapping,
            },
            "record_cycle",
        ),
        (
            {
                "schemaType": "anyalgebra.test.note",
                "schemaVersion": 1,
                "text": set(),
            },
            "invalid_record_value",
        ),
        (node_limited, "record_node_limit"),
    ):
        with pytest.raises(SchemaError, match=code):
            registry.from_record(record)


def test_malformed_or_unknown_metadata_does_not_call_a_registered_parser() -> None:
    parser_calls: list[object] = []

    def parser(record: object) -> _Note:
        parser_calls.append(record)
        return _Note("called")

    registry = SerializerRegistry().with_codec(
        "anyalgebra.test.note", 1, _Note, _encode_note, parser
    )
    for record in (
        {"schemaType": 1, "schemaVersion": 1},
        {"schemaType": "anyalgebra.test.unknown", "schemaVersion": 1},
        {"schemaType": "anyalgebra.test.note", "schemaVersion": 2},
    ):
        with pytest.raises(SchemaError):
            registry.from_record(record)
    assert parser_calls == []


def test_parser_failure_and_wrong_exact_output_are_sanitized_and_rejected() -> None:
    def exploding_parser(record: object) -> _Note:
        del record
        raise RuntimeError("secret parser detail")

    exploding = SerializerRegistry().with_codec(
        "anyalgebra.test.note", 1, _Note, _encode_note, exploding_parser
    )
    with pytest.raises(SchemaError, match="parser_failed") as failure:
        exploding.from_record(
            {"schemaType": "anyalgebra.test.note", "schemaVersion": 1}
        )
    assert failure.value.__cause__ is None
    assert failure.value.__context__ is None

    def wrong_parser(record: object) -> object:
        del record
        return object()

    wrong = SerializerRegistry().with_codec(
        "anyalgebra.test.note", 1, _Note, _encode_note, wrong_parser
    )
    with pytest.raises(SchemaError, match="invalid_parser_output"):
        wrong.from_record({"schemaType": "anyalgebra.test.note", "schemaVersion": 1})


def test_nested_encoder_output_is_copied_and_non_mapping_output_rejected() -> None:
    def nested_encoder(value: _Note) -> dict[str, object]:
        return {"text": value.text, "nested": [{"key": "value"}]}

    registry = SerializerRegistry().with_codec(
        "anyalgebra.test.note", 1, _Note, nested_encoder, _parse_note
    )
    assert registry.to_record(_Note("copied"))["nested"] == [{"key": "value"}]

    def not_a_mapping(value: _Note) -> object:
        del value
        return []

    malformed = SerializerRegistry().with_codec(
        "anyalgebra.test.note", 1, _Note, not_a_mapping, _parse_note
    )
    with pytest.raises(SchemaError, match="invalid_encoder_output"):
        malformed.to_record(_Note("bad"))
