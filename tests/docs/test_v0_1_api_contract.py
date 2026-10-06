from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
API_CONTRACT = ROOT / "docs" / "api" / "api-v0.1.md"

APIS = {
    "api.algebra.evaluate": (
        "evaluate_multilinear(structure, *elements, operation=None, graph=None)"
    ),
    "api.census.spec": (
        "CensusSpec.create(*, carrier_size, arity, constraints, equivalence, "
        "ordering, bounds, convention_refs=())"
    ),
    "api.census.enumerate": "enumerate_reference(spec)",
    "api.census.canonicalize": (
        "canonicalize_operation_table(core, source_outputs, equivalence)"
    ),
    "api.census.compare": (
        "compare_isomorphism(core, left_outputs, right_outputs, equivalence, *, "
        "max_permutations=None)"
    ),
    "api.census.analyze": (
        "assemble_finite_carrier_analysis(core, outputs, equivalence, *, "
        "nilpotence_zero=None, max_nilpotence_index=None, subobject_options=None)"
    ),
    "api.atlas.build": "build_finite_algebra_atlas(specs, *, output_directory)",
    "api.atlas.query": (
        "load_finite_algebra_atlas(directory); query_atlas(loaded, *, "
        "carrier_size=None, arity=None, canonical_id=None, laws=(), "
        "invariant_fields=())"
    ),
}


def _contract() -> str:
    return API_CONTRACT.read_text(encoding="utf-8")


def test_v01_002_contract_freezes_all_and_only_planned_public_apis() -> None:
    text = _contract()

    assert "Status: implemented in the locally complete 0.1.0 source milestone" in text
    assert "V01-060 passed the final readiness decision" in text
    for api_id, signature in APIS.items():
        assert text.count(f"### `{api_id}`") == 1
        assert text.count(f"`{signature}`") == 1

    stable_surface = text.split("<!-- stable-surface:end -->")[0].split(
        "<!-- stable-surface:begin -->", maxsplit=1
    )[1]
    assert stable_surface.count("### `api.") == len(APIS)
    assert "experimental" not in stable_surface.lower()
    assert "E8" not in stable_surface
    assert "Albert" not in stable_surface
    assert "Module: `anyalgebra.algebra.multilinear`." in stable_surface


def test_v01_002_contract_defines_ownership_errors_and_bounded_outcomes() -> None:
    text = _contract()

    for error in (
        "MultilinearEvaluationError",
        "CensusSpecError",
        "ReferenceEnumerationError",
        "CanonicalLabelError",
        "IsomorphismOutcomeError",
        "AnalysisRecordError",
        "AtlasBuildError",
        "AtlasQueryError",
    ):
        assert f"`{error}`" in text

    for outcome in (
        "ReferenceEnumeration",
        "CanonicalLabel",
        "Isomorphic",
        "InvariantNonIsomorphic",
        "ExhaustiveNonIsomorphic",
        "Inconclusive",
    ):
        assert f"`{outcome}`" in text

    for requirement in (
        "literal parent",
        "immutable",
        "canonical byte",
        "checked bijection",
        "fingerprint equality never constructs an isomorphism",
        "checked before consuming",
        "literal parent ownership",
        "coercion",
    ):
        assert requirement in text.lower()


def test_v01_002_contract_has_positive_negative_and_boundary_examples() -> None:
    text = _contract()

    for heading in (
        "## Shared records and errors",
        "## Positive examples",
        "## Negative examples",
        "## Boundary examples",
        "## Determinism contract",
        "## Stability boundary",
        "## Task ownership",
    ):
        assert heading in text

    assert "19,683" in text
    assert "729" in text
    assert "empty carrier" in text
    assert "nullary operation" in text
    assert "permutation" in text.lower()
    assert "path traversal" in text.lower()
    assert "unknown schema" in text.lower()


def test_v01_002_each_api_has_explicit_return_error_and_owner() -> None:
    text = _contract()

    for api_id in APIS:
        section = text.split(f"### `{api_id}`", maxsplit=1)[1].split(
            "### `api.", maxsplit=1
        )[0]
        assert "Module:" in section
        assert "Returns:" in section
        assert "Errors:" in section
        assert "Owned by:" in section
