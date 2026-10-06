"""Behavioral contract for safe, directional record migrations."""

from __future__ import annotations

from collections.abc import Mapping
import operator
from types import MappingProxyType
from typing import cast

import pytest

from anyalgebra.persistence.migrations import MigrationRegistry
from anyalgebra.persistence.registry import SchemaError, SerializerRegistry


_TAG = "anyalgebra.test.note"


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_thaw(item) for item in value]
    return value


def _v1_to_v2(record: object) -> dict[str, object]:
    snapshot = record
    assert isinstance(snapshot, MappingProxyType)
    return {
        "schemaType": snapshot["schemaType"],
        "schemaVersion": 2,
        "text": snapshot["text"],
    }


def _v2_to_v3(record: object) -> dict[str, object]:
    snapshot = record
    assert isinstance(snapshot, MappingProxyType)
    return {
        "schemaType": snapshot["schemaType"],
        "schemaVersion": 3,
        "text": snapshot["text"],
    }


def _registry() -> MigrationRegistry:
    return (
        MigrationRegistry()
        .with_migration(_TAG, 1, 2, _v1_to_v2)
        .with_migration(_TAG, 2, 3, _v2_to_v3)
    )


def test_migrations_follow_a_preplanned_adjacent_chain_and_parse_only_afterward() -> (
    None
):
    calls: list[object] = []

    def v1_to_v2(record: object) -> dict[str, object]:
        calls.append(record)
        assert isinstance(record, MappingProxyType)
        nested = record["nested"]
        assert isinstance(nested, tuple)
        return {
            "schemaType": _TAG,
            "schemaVersion": 2,
            "text": record["text"],
            "nested": _thaw(nested),
        }

    def v2_to_v3(record: object) -> dict[str, object]:
        calls.append(record)
        assert isinstance(record, Mapping)
        return {
            "schemaType": _TAG,
            "schemaVersion": 3,
            "text": record["text"],
            "nested": _thaw(record["nested"]),
        }

    migrations = (
        MigrationRegistry()
        .with_migration(_TAG, 1, 2, v1_to_v2)
        .with_migration(_TAG, 2, 3, v2_to_v3)
    )
    source: dict[str, object] = {
        "schemaType": _TAG,
        "schemaVersion": 1,
        "text": "old",
        "nested": [{"value": "before"}],
    }

    migrated = migrations.migrate_record(source, target_version=3)

    assert migrated == {
        "schemaType": _TAG,
        "schemaVersion": 3,
        "text": "old",
        "nested": [{"value": "before"}],
    }
    assert len(calls) == 2
    assert calls[0] is not calls[1]
    serializer_calls: list[object] = []

    def encode(value: str) -> dict[str, object]:
        return {"text": value}

    def parse(record: object) -> str:
        serializer_calls.append(record)
        assert isinstance(record, MappingProxyType)
        assert record["schemaVersion"] == 3
        return str(record["text"])

    serializers = SerializerRegistry().with_codec(_TAG, 3, str, encode, parse)
    assert serializers.from_record(migrated) == "old"
    assert len(serializer_calls) == 1


@pytest.mark.parametrize("version", (1, 2))
def test_registered_endpoint_noop_returns_a_fresh_mutable_copy_without_callback(
    version: int,
) -> None:
    calls: list[object] = []

    def migration(record: object) -> dict[str, object]:
        calls.append(record)
        return {"schemaType": _TAG, "schemaVersion": 2}

    registry = MigrationRegistry().with_migration(_TAG, 1, 2, migration)
    source: dict[str, object] = {
        "schemaType": _TAG,
        "schemaVersion": version,
        "nested": [{"value": "original"}],
    }

    copied = registry.migrate_record(source, target_version=version)
    nested = copied["nested"]
    assert type(nested) is list
    first = nested[0]
    assert type(first) is dict
    first["value"] = "changed"

    assert calls == []
    assert copied is not source
    assert source["nested"] == [{"value": "original"}]


@pytest.mark.parametrize(
    ("record", "target", "code"),
    [
        ({"schemaType": _TAG, "schemaVersion": 2}, 1, "migration_downgrade"),
        (
            {"schemaType": "anyalgebra.test.unknown", "schemaVersion": 1},
            2,
            "unknown_schema_type",
        ),
        ({"schemaType": _TAG, "schemaVersion": 1}, 3, "missing_migration"),
        (
            {"schemaType": _TAG, "schemaVersion": 2_147_483_647},
            2_147_483_647,
            "unknown_schema_version",
        ),
        ({"schemaType": _TAG, "schemaVersion": 3}, 4, "unknown_schema_version"),
        ({"schemaType": _TAG, "schemaVersion": 3}, 2, "unknown_schema_version"),
        ({"schemaVersion": 1}, 2, "missing_schema_type"),
        ({"schemaType": _TAG}, 2, "missing_schema_version"),
        ({"schemaType": _TAG, "schemaVersion": True}, 2, "invalid_schema_version"),
        ({"schemaType": _TAG, "schemaVersion": 1}, True, "invalid_schema_version"),
        (object(), 2, "invalid_record"),
    ],
)
def test_preflight_rejections_never_call_a_migration(
    record: object, target: object, code: str
) -> None:
    calls: list[object] = []

    def migration(snapshot: object) -> dict[str, object]:
        calls.append(snapshot)
        return {"schemaType": _TAG, "schemaVersion": 2}

    registry = MigrationRegistry().with_migration(_TAG, 1, 2, migration)

    with pytest.raises(SchemaError, match=code) as caught:
        registry.migrate_record(record, target_version=target)
    assert caught.value.code == code
    assert calls == []


def test_registration_rejects_unsafe_migration_definitions() -> None:
    registry = MigrationRegistry().with_migration(_TAG, 1, 2, _v1_to_v2)

    with pytest.raises(SchemaError, match="duplicate_migration"):
        registry.with_migration(_TAG, 1, 2, _v2_to_v3)
    for source, target in ((2, 1), (1, 3)):
        with pytest.raises(SchemaError, match="invalid_migration_direction"):
            MigrationRegistry().with_migration(_TAG, source, target, _v1_to_v2)
    for tag, source, target, callback, code in (
        (" bad", 1, 2, _v1_to_v2, "invalid_schema_type"),
        (_TAG, True, 2, _v1_to_v2, "invalid_schema_version"),
        (_TAG, 1, 2, object(), "invalid_migration_function"),
    ):
        with pytest.raises(SchemaError, match=code):
            MigrationRegistry().with_migration(tag, source, target, callback)


@pytest.mark.parametrize(
    ("output", "code"),
    [
        ({"schemaVersion": 2}, "missing_schema_type"),
        ({"schemaType": _TAG}, "missing_schema_version"),
        (
            {"schemaType": "anyalgebra.test.other", "schemaVersion": 2},
            "invalid_migration_output_tag",
        ),
        ({"schemaType": _TAG, "schemaVersion": 3}, "invalid_migration_output_version"),
        ({"schemaType": _TAG, "schemaVersion": True}, "invalid_schema_version"),
        ([], "invalid_migration_output"),
    ],
)
def test_tampered_migration_outputs_are_rejected(output: object, code: str) -> None:
    def migration(record: object) -> object:
        del record
        return output

    registry = MigrationRegistry().with_migration(_TAG, 1, 2, migration)

    with pytest.raises(SchemaError, match=code):
        registry.migrate_record(
            {"schemaType": _TAG, "schemaVersion": 1}, target_version=2
        )


def test_callback_failure_and_snapshot_boundary_are_sanitized_and_preserve_source() -> (
    None
):
    source: dict[str, object] = {
        "schemaType": _TAG,
        "schemaVersion": 1,
        "nested": [{"value": "before"}],
    }
    observed: list[object] = []

    def migration(record: object) -> dict[str, object]:
        observed.append(record)
        assert isinstance(record, MappingProxyType)
        with pytest.raises(TypeError):
            operator.setitem(cast(dict[str, object], record), "blocked", "blocked")
        nested = record["nested"]
        assert isinstance(nested, tuple)
        source["nested"][0]["value"] = "after"  # type: ignore[index]
        return {
            "schemaType": record["schemaType"],
            "schemaVersion": 2,
            "nested": _thaw(nested),
        }

    migrated = (
        MigrationRegistry()
        .with_migration(_TAG, 1, 2, migration)
        .migrate_record(source, target_version=2)
    )
    assert observed
    assert migrated["nested"] == [{"value": "before"}]
    assert source["nested"] == [{"value": "after"}]

    def exploding(record: object) -> dict[str, object]:
        del record
        raise RuntimeError("secret migration detail")

    with pytest.raises(SchemaError, match="migration_failed") as caught:
        MigrationRegistry().with_migration(_TAG, 1, 2, exploding).migrate_record(
            {"schemaType": _TAG, "schemaVersion": 1}, target_version=2
        )
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "secret" not in str(caught.value)


def test_hostile_values_subclasses_and_registry_mutation_cannot_expand_execution() -> (
    None
):
    class Hostile:
        def __repr__(self) -> str:
            raise AssertionError("repr must not run")

        def __eq__(self, other: object) -> bool:
            del other
            raise AssertionError("equality must not run")

        def __hash__(self) -> int:
            raise AssertionError("hash must not run")

    class DictSubclass(dict[str, object]):
        pass

    calls: list[object] = []

    def migration(record: object) -> dict[str, object]:
        calls.append(record)
        return {"schemaType": _TAG, "schemaVersion": 2}

    registry = MigrationRegistry().with_migration(_TAG, 1, 2, migration)
    for value in (Hostile(), DictSubclass(schemaType=_TAG, schemaVersion=1)):
        with pytest.raises(SchemaError, match="invalid_record"):
            registry.migrate_record(value, target_version=2)
    with pytest.raises(SchemaError, match="invalid_migration_function"):
        MigrationRegistry().with_migration(_TAG, 1, 2, Hostile())
    assert calls == []
    assert not hasattr(registry, "__dict__")
    assert "0x" not in repr(registry)
    with pytest.raises(TypeError, match="unhashable"):
        hash(registry)
    with pytest.raises(AttributeError, match="immutable"):
        registry._migrations = ()
    with pytest.raises(AttributeError, match="immutable"):
        del registry._migrations
    with pytest.raises(TypeError, match="cannot be subclassed"):

        class DerivedRegistry(MigrationRegistry):
            pass


def test_chain_limit_is_exact_and_input_never_selects_imports_or_callables() -> None:
    calls: list[int] = []

    def make_step(source: int) -> object:
        def step(record: object) -> dict[str, object]:
            calls.append(source)
            snapshot = record
            assert isinstance(snapshot, Mapping)
            return {
                "schemaType": snapshot["schemaType"],
                "schemaVersion": source + 1,
                "payload": _thaw(snapshot["payload"]),
            }

        return step

    registry = MigrationRegistry()
    for version in range(1, 65):
        registry = registry.with_migration(
            _TAG, version, version + 1, make_step(version)
        )
    source = {
        "schemaType": _TAG,
        "schemaVersion": 1,
        "payload": {"__import__": "os", "callable": "builtins.eval"},
    }

    assert registry.migrate_record(source, target_version=65)["schemaVersion"] == 65
    assert calls == list(range(1, 65))
    calls.clear()
    with pytest.raises(SchemaError, match="migration_chain_limit"):
        registry.migrate_record(source, target_version=66)
    assert calls == []
