"""Safe, persistent directional migrations for tagged JSON records.

This module deliberately contains no input-directed import, callable lookup, or
codec dispatch.  Application code registers exact function objects while
building a registry; untrusted records can only select a prevalidated,
adjacent upgrade path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import FunctionType
from anyalgebra.persistence.registry import (
    JSONValue,
    SchemaError,
    _mutable_json,
    _record_snapshot,
    _schema_version,
    _stable_tag,
)


MigrationFunction = FunctionType
_MAX_MIGRATION_STEPS = 64


@dataclass(frozen=True, slots=True)
class _Migration:
    """One trusted, adjacent record transformation."""

    schema_type: str
    source_version: int
    target_version: int
    callback: MigrationFunction


class MigrationRegistry:
    """Immutable allow-list of pure, directional schema migrations.

    ``with_migration`` returns a new registry and leaves its receiver intact.
    Only adjacent upgrades may be registered, making every preplanned path
    finite, deterministic, and free of downgrade or branch selection.
    """

    __slots__ = ("_migrations",)
    _migrations: tuple[_Migration, ...]
    __hash__ = None  # type: ignore[assignment]

    def __init__(self) -> None:
        """Create an empty immutable migration registry."""
        object.__setattr__(self, "_migrations", ())

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Keep trusted registry construction confined to this exact class."""
        del cls, kwargs
        raise TypeError("MigrationRegistry cannot be subclassed")

    def __setattr__(self, name: str, value: object) -> None:
        """Reject direct mutation of persistent registry state."""
        del name, value
        raise AttributeError("MigrationRegistry is immutable")

    def __delattr__(self, name: str) -> None:
        """Reject direct deletion of persistent registry state."""
        del name
        raise AttributeError("MigrationRegistry is immutable")

    def __repr__(self) -> str:
        """Render bounded metadata without callback details."""
        return f"MigrationRegistry(migration_count={len(self._migrations)})"

    @classmethod
    def _with_migrations(cls, migrations: tuple[_Migration, ...]) -> MigrationRegistry:
        """Build a persistent successor without exposing mutable state."""
        del cls
        successor = object.__new__(MigrationRegistry)
        object.__setattr__(successor, "_migrations", migrations)
        return successor

    def with_migration(
        self,
        schema_type: object,
        source_version: object,
        target_version: object,
        callback: object,
    ) -> MigrationRegistry:
        """Return a successor with one exact trusted adjacent upgrade."""
        tag = _stable_tag(schema_type)
        source = _schema_version(source_version)
        target = _schema_version(target_version)
        if target != source + 1:
            raise SchemaError("invalid_migration_direction")
        if type(callback) is not FunctionType:
            raise SchemaError("invalid_migration_function")
        if any(
            existing.schema_type == tag and existing.source_version == source
            for existing in self._migrations
        ):
            raise SchemaError("duplicate_migration")
        migration = _Migration(tag, source, target, callback)
        return self._with_migrations((*self._migrations, migration))

    def _plan(
        self, schema_type: str, source_version: int, target_version: int
    ) -> tuple[_Migration, ...]:
        """Validate and construct a complete callback-free upgrade plan."""
        tagged = tuple(
            migration
            for migration in self._migrations
            if migration.schema_type == schema_type
        )
        if not tagged:
            raise SchemaError("unknown_schema_type")
        endpoints = frozenset(
            version
            for migration in tagged
            for version in (migration.source_version, migration.target_version)
        )
        if source_version not in endpoints:
            raise SchemaError("unknown_schema_version")
        if target_version < source_version:
            raise SchemaError("migration_downgrade")
        if target_version - source_version > _MAX_MIGRATION_STEPS:
            raise SchemaError("migration_chain_limit")
        planned: list[_Migration] = []
        current = source_version
        while current < target_version:
            step = next(
                (
                    migration
                    for migration in self._migrations
                    if migration.schema_type == schema_type
                    and migration.source_version == current
                ),
                None,
            )
            if step is None:
                raise SchemaError("missing_migration")
            planned.append(step)
            current = step.target_version
        return tuple(planned)

    def migrate_record(
        self, record: object, *, target_version: object
    ) -> dict[str, JSONValue]:
        """Return a fresh record after one explicit, prevalidated upgrade path.

        Every callback receives an immutable deep snapshot.  Its result is
        snapshot and metadata validated before it may become the next step's
        input, so neither caller mutation nor callback-side TOCTOU changes the
        planned path.
        """
        snapshot = _record_snapshot(record)
        if "schemaType" not in snapshot:
            raise SchemaError("missing_schema_type")
        if "schemaVersion" not in snapshot:
            raise SchemaError("missing_schema_version")
        tag = _stable_tag(snapshot["schemaType"])
        source = _schema_version(snapshot["schemaVersion"])
        target = _schema_version(target_version)
        planned = self._plan(tag, source, target)
        current: Mapping[str, object] = snapshot
        for step in planned:
            try:
                output = step.callback(current)
            except Exception:
                failed = True
            else:
                failed = False
            if failed:
                raise SchemaError("migration_failed")
            if type(output) is not dict:
                raise SchemaError("invalid_migration_output")
            current = _record_snapshot(output)
            if "schemaType" not in current:
                raise SchemaError("missing_schema_type")
            if "schemaVersion" not in current:
                raise SchemaError("missing_schema_version")
            output_tag = _stable_tag(current["schemaType"])
            output_version = _schema_version(current["schemaVersion"])
            if output_tag != tag:
                raise SchemaError("invalid_migration_output_tag")
            if output_version != step.target_version:
                raise SchemaError("invalid_migration_output_version")
        result = _mutable_json(current)
        assert type(result) is dict
        return result
