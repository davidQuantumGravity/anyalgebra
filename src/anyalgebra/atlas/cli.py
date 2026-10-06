"""Deterministic JSON CLI for finite-algebra atlas workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import NoReturn, cast

from anyalgebra.census.serialization import census_spec_from_canonical_bytes
from anyalgebra.census.spec import CensusSpec
from anyalgebra.core.parents import SemanticHash

from .build import build_finite_algebra_atlas
from .query import load_finite_algebra_atlas, query_atlas


_MAX_SPEC_BYTES = 1_048_576


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        del message
        raise ValueError("invalid_arguments")


def _parser() -> _ArgumentParser:
    parser = _ArgumentParser(prog="anyalgebra-atlas", exit_on_error=False)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-spec", exit_on_error=False)
    validate.add_argument("spec")
    build = commands.add_parser("build", exit_on_error=False)
    build.add_argument("--spec", action="append", required=True)
    build.add_argument("--output", required=True)
    verify = commands.add_parser("verify", exit_on_error=False)
    verify.add_argument("--atlas", required=True)
    query = commands.add_parser("query", exit_on_error=False)
    query.add_argument("--atlas", required=True)
    query.add_argument("--carrier-size", type=int)
    query.add_argument("--arity", type=int)
    query.add_argument("--canonical-id")
    query.add_argument("--law", action="append", default=[])
    query.add_argument("--invariant", action="append", default=[])
    return parser


def _emit(value: dict[str, object], *, error: bool = False) -> None:
    stream = sys.stderr if error else sys.stdout
    stream.write(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def _spec(path: object) -> CensusSpec:
    if type(path) is not str:
        raise ValueError("spec_rejected")
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("spec_rejected")
    if candidate.stat().st_size > _MAX_SPEC_BYTES:
        raise ValueError("spec_rejected")
    return census_spec_from_canonical_bytes(candidate.read_bytes())


def _identifier(value: object) -> SemanticHash | None:
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError("invalid_query")
    algorithm, separator, digest = value.partition(":")
    if not separator:
        raise ValueError("invalid_query")
    return SemanticHash(algorithm, digest)


def _invariants(values: object) -> tuple[tuple[str, str | int | bool], ...]:
    if type(values) is not list:
        raise ValueError("invalid_query")
    result: list[tuple[str, str | int | bool]] = []
    for raw in cast(list[object], values):
        if type(raw) is not str:
            raise ValueError("invalid_query")
        name, separator, text = raw.partition("=")
        if not separator or not name or not text:
            raise ValueError("invalid_query")
        if text == "true":
            expected: str | int | bool = True
        elif text == "false":
            expected = False
        else:
            try:
                expected = int(text)
            except ValueError:
                expected = text
        result.append((name, expected))
    return tuple(result)


def _validate_command(arguments: argparse.Namespace) -> dict[str, object]:
    spec = _spec(arguments.spec)
    return {
        "arity": spec.arity,
        "candidateCount": spec.core.candidate_count,
        "carrierSize": spec.carrier_size,
        "corpusName": spec.core.corpus_name,
        "specHash": str(spec.semantic_hash),
    }


def _build_command(arguments: argparse.Namespace) -> dict[str, object]:
    specs = tuple(_spec(item) for item in arguments.spec)
    target = Path(arguments.output).resolve(strict=False)
    result = build_finite_algebra_atlas(specs, output_directory=target)
    return {
        "atlasHash": str(result.atlas.semantic_hash),
        "corpusCount": result.atlas.header.corpus_count,
        "objectCount": result.object_count,
        "representativeCount": result.atlas.header.representative_count,
        "status": result.atlas.header.status,
    }


def _verify_command(arguments: argparse.Namespace) -> dict[str, object]:
    loaded = load_finite_algebra_atlas(Path(arguments.atlas).resolve(strict=False))
    return {
        "atlasHash": str(loaded.atlas.semantic_hash),
        "corpusCount": loaded.atlas.header.corpus_count,
        "representativeCount": loaded.atlas.header.representative_count,
        "status": loaded.atlas.header.status,
        "verifiedObjectCount": loaded.verified_object_count,
    }


def _query_command(arguments: argparse.Namespace) -> dict[str, object]:
    loaded = load_finite_algebra_atlas(Path(arguments.atlas).resolve(strict=False))
    result = query_atlas(
        loaded,
        carrier_size=arguments.carrier_size,
        arity=arguments.arity,
        canonical_id=_identifier(arguments.canonical_id),
        laws=tuple(arguments.law),
        invariant_fields=_invariants(arguments.invariant),
    )
    return {
        "examinedRepresentativeCount": result.examined_representative_count,
        "matchCount": len(result.matches),
        "matches": [
            {
                "acceptedMemberCount": item.accepted_member_count,
                "canonicalId": str(item.canonical_id),
                "corpusId": item.corpus_id,
                "fullOrbitSize": item.full_orbit_size,
                "representativeOutputs": list(item.representative_outputs),
            }
            for item in result.matches
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """Run one command, emitting only canonical JSON or one stable error code."""
    try:
        arguments = _parser().parse_args(argv)
    except (ValueError, argparse.ArgumentError):
        _emit({"error": "invalid_arguments"}, error=True)
        return 2
    command = arguments.command
    try:
        if command == "validate-spec":
            output = _validate_command(arguments)
        elif command == "build":
            output = _build_command(arguments)
        elif command == "verify":
            output = _verify_command(arguments)
        elif command == "query":
            output = _query_command(arguments)
        else:  # pragma: no cover - argparse's required subparser prevents this
            raise ValueError("invalid_arguments")
    except Exception:
        codes = {
            "validate-spec": "spec_rejected",
            "build": "build_rejected",
            "verify": "atlas_rejected",
            "query": "invalid_query",
        }
        _emit({"error": codes[command]}, error=True)
        return 2
    _emit(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
