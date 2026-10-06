from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "docs" / "specifications" / "v0.1-capability-matrix.md"

FINITE_FIELDS = (
    "totality",
    "commutativity",
    "associativity",
    "flexibility",
    "left_alternativity",
    "right_alternativity",
    "equations",
    "left_identity",
    "right_identity",
    "two_sided_identity",
    "left_zero",
    "right_zero",
    "two_sided_zero",
    "idempotents",
    "nilpotents",
    "element_profiles",
    "left_nucleus",
    "middle_nucleus",
    "right_nucleus",
    "nucleus",
    "commutant",
    "center",
    "substructures",
    "congruences",
)

LINEAR_FIELDS = (
    "linear_nucleus",
    "linear_center",
    "linear_ideals",
    "derivations",
    "units",
    "span_data",
    "fingerprint",
)

COMPOSITION_FIELDS = (
    "conjugation",
    "real_part",
    "trace",
    "quadratic_norm",
    "norm_composition",
    "moufang_identities",
    "anisotropic_inverse",
    "isotropy",
    "zero_divisors",
)


def _text() -> str:
    return MATRIX.read_text(encoding="utf-8")


def test_v01_007_defines_domains_without_conflating_carrier_and_basis() -> None:
    text = _text()

    for term in (
        "bare finite magma",
        "pointed finite magma",
        "finite quasigroup",
        "finite loop",
        "finite ring",
        "module-backed algebra",
    ):
        assert f"**{term}**" in text.lower()

    assert "finite-carrier" in text
    assert "finite-basis" in text
    assert "does not make its scalar carrier finite" in text
    assert "one operation table does not imply ring addition" in text.lower()


def test_v01_007_maps_every_planned_analysis_field_once() -> None:
    text = _text()
    rows = [line for line in text.splitlines() if line.startswith("| `analysis.")]
    ids = [row.split("|")[1].strip().strip("`") for row in rows]
    expected = tuple(
        f"analysis.{field}"
        for field in (*FINITE_FIELDS, *LINEAR_FIELDS, *COMPOSITION_FIELDS)
    )

    assert tuple(ids) == expected
    assert len(ids) == len(set(ids))
    for row in rows:
        assert "required structure" not in row.lower()
        assert "`Complete`" in row or "`Bounded`" in row


def test_v01_007_freezes_typed_unsupported_and_evidence_semantics() -> None:
    text = _text()

    for status in (
        "Complete",
        "Bounded",
        "RejectionOnly",
        "Unsupported",
        "Failed",
    ):
        assert f"`{status}`" in text

    for code in (
        "missing_binary_operation",
        "missing_distinguished_element",
        "missing_additive_structure",
        "missing_module_structure",
        "unsupported_scalar_domain",
        "missing_involution",
        "missing_quadratic_form",
        "undefined_power_policy",
    ):
        assert f"`{code}`" in text

    assert "unsupported is not false" in text.lower()
    assert "unsupported is not the number zero" in text.lower()
    assert "fingerprint equality" in text.lower()
    assert "never proves isomorphism" in text.lower()


def test_v01_007_has_positive_negative_boundary_and_record_contracts() -> None:
    text = _text()

    for heading in (
        "## Vocabulary",
        "## Result-field contract",
        "## Capability matrix by structure",
        "## Planned analysis fields",
        "## Positive, negative, and boundary controls",
        "## Determinism and evidence",
        "## Non-goals",
    ):
        assert heading in text

    for field in (
        "field_id",
        "status",
        "value",
        "algorithm",
        "hypotheses",
        "bounds",
        "examined_count",
        "expected_count",
        "witness",
        "unsupported_code",
    ):
        assert f"`{field}`" in text

    assert "empty carrier" in text.lower()
    assert "one-element" in text.lower()
    assert "counterexample" in text.lower()
