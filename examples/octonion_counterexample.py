"""Print a deterministic associativity counterexample for one named table.

This example uses only the generic finite-table and elementary-analysis APIs.
It verifies every ordered basis triple until the first nonzero associator is
found, so the printed witness is minimal in the fixture's declared basis order.
"""

from __future__ import annotations

from typing import Final

from anyalgebra.analysis.elementary import associator
from anyalgebra.core.elements import SparseElement
from anyalgebra.fixtures.composition import octonion_fixture


CONVENTION_ID: Final = "algmul.O.v1"
WITNESS_INDICES: Final = (1, 2, 4)
MULTILINEARITY_REDUCTION_HYPOTHESIS: Final = (
    "The exact ZZ-bilinear table makes the associator ZZ-trilinear; "
    "the complete ordered basis grid is sufficient."
)


def _minimal_witness() -> tuple[tuple[int, int, int], SparseElement, int]:
    """Find the first nonzero associator in lexicographic basis-triple order."""
    algebra = octonion_fixture()
    checked = 0
    for left in range(algebra.module.rank):
        for middle in range(algebra.module.rank):
            for right in range(algebra.module.rank):
                checked += 1
                value = associator(
                    algebra,
                    algebra.module.element({left: 1}),
                    algebra.module.element({middle: 1}),
                    algebra.module.element({right: 1}),
                )
                if value != algebra.module.zero():
                    return (left, middle, right), value, checked
    raise RuntimeError("the declared finite table had no nonzero basis associator")


def main() -> None:
    """Print the exact witness and the finite, convention-scoped evidence."""
    indices, value, checked = _minimal_witness()
    algebra = octonion_fixture()
    labels = tuple(algebra.module.basis.labels[index] for index in indices)
    assert indices == WITNESS_INDICES
    assert value.coordinates() == {7: algebra.module.domain.element(2)}

    print(f"convention: {CONVENTION_ID}")
    print(f"basis order: {algebra.module.basis.labels}")
    print(f"reduction hypothesis: {MULTILINEARITY_REDUCTION_HYPOTHESIS}")
    print("basis grid: all 8^3 ordered triples, lexicographic order, no sampling")
    print(f"first nonzero associator after {checked} checks: {labels}")
    print("associator: (e1*e2)*e4 - e1*(e2*e4) = 2*e7")
    print(
        "scope: this is a deterministic witness for the loaded finite-table "
        "convention only; it does not promote a scientific claim."
    )


if __name__ == "__main__":
    main()
