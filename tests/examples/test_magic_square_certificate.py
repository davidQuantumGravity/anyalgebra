"""Public-example test for the 2-by-2 and 3-by-3 magic-square receipt."""

from __future__ import annotations

from examples.magic_square_certificate import run_example


def test_magic_square_certificate_example() -> None:
    receipt = run_example()

    assert receipt["catalogue_keys"] == ("R", "C", "Cs", "H", "Hs", "O", "Os")
    assert receipt["n2"]["entry_count"] == 49
    assert receipt["n3"]["entry_count"] == 49

    ladder = receipt["n2"]["requested_ladder"]
    assert [entry["group"] for entry in ladder] == [
        "Spin(3)",
        "Spin(10)",
        "Spin(16)",
    ]
    assert receipt["n2"]["spin16_certificate"] == {
        "signature": (16, 0),
        "generator_count": 120,
        "metric_skew_generators": 120,
        "commutators_checked": 7260,
        "closed": True,
    }

    assert receipt["n3"]["compact_e8"]["decomposition"] == (14, 52, 182)
    assert receipt["n3"]["compact_e8"]["lie_algebra"] == "e8(-248)"
    assert receipt["n3"]["compact_e8"]["source"].endswith("Table 1")
    assert (
        receipt["n3"]["compact_e8"]["group_scope"]
        == "connected global form unspecified"
    )
    assert receipt["n3"]["half_split_e8"]["lie_algebra"] == "e8(-24)"
    assert receipt["n3"]["half_split_e8"]["source"].endswith("Table 2")
    assert receipt["n3"]["split_e8"]["lie_algebra"] == "e8(8)"
    assert receipt["n3"]["split_e8"]["source"].endswith("Table 3")
