"""Emit deterministic tables for the 2-by-2 and 3-by-3 magic squares."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from anyalgebra.experimental.magic_square import (
    composition_algebra_keys,
    magic_square,
    n2_magic_square_entry,
    n2_orthogonal_certificate,
    n3_magic_square_entry,
)


def _entry_receipt(matrix_size: int, left: str, right: str) -> dict[str, Any]:
    entry = (
        n2_magic_square_entry(left, right)
        if matrix_size == 2
        else n3_magic_square_entry(left, right)
    )
    return {
        "carrier": f"{left} tensor J{matrix_size}({right})",
        "carrier_dimension": entry.carrier_dimension,
        "construction": entry.construction,
        "decomposition": entry.decomposition,
        "group": entry.group,
        "group_scope": entry.group_scope,
        "lie_algebra": entry.lie_algebra,
        "lie_algebra_dimension": entry.lie_algebra_dimension,
        "signature": entry.signature,
        "source": entry.source,
    }


def run_example() -> dict[str, Any]:
    keys = composition_algebra_keys()
    n2 = magic_square(2)
    n3 = magic_square(3)
    spin16 = n2_orthogonal_certificate("O", "O")
    return {
        "catalogue_keys": keys,
        "n2": {
            "entry_count": sum(len(row) for row in n2),
            "requested_ladder": [
                _entry_receipt(2, "R", "C"),
                _entry_receipt(2, "C", "O"),
                _entry_receipt(2, "O", "O"),
            ],
            "spin16_certificate": asdict(spin16),
            "table": [[entry.lie_algebra for entry in row] for row in n2],
        },
        "n3": {
            "entry_count": sum(len(row) for row in n3),
            "exceptional_column": [
                _entry_receipt(3, left, "O") for left in ("R", "C", "H", "O")
            ],
            "compact_e8": _entry_receipt(3, "O", "O"),
            "half_split_e8": _entry_receipt(3, "Os", "O"),
            "split_e8": _entry_receipt(3, "Os", "Os"),
            "table": [[entry.lie_algebra for entry in row] for row in n3],
        },
        "claim_scope": (
            "catalogued standard real forms and exact n=2 orthogonal closure; "
            "the n=3 entries are labels from the published tables, not "
            "constructed brackets"
        ),
    }


def main() -> None:
    print(json.dumps(run_example(), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
