"""Render the README figures from the package's own data.

    python tools/render_readme_figures.py          # rewrite docs/assets/*.svg
    python tools/render_readme_figures.py --check  # fail if a file is stale

The octonion table is drawn from ``anyalgebra.easy.octonions()``, so the
picture cannot disagree with the code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anyalgebra.easy as aa

ASSETS = Path(__file__).resolve().parents[1] / "docs" / "assets"
INK = "#1f2937"
PAPER = "#f8fafc"
RULE = "#cbd5e1"
# One hue per basis element e1 .. e7; the unit is neutral.
HUES = (
    "#64748b",
    "#2563eb",
    "#0891b2",
    "#059669",
    "#ca8a04",
    "#ea580c",
    "#dc2626",
    "#9333ea",
)
FONT = 'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"'


def wordmark() -> str:
    """Return the project wordmark."""
    width, height = 560, 120
    dots = "".join(
        f'<circle cx="{48 + 22 * (n % 2)}" cy="{38 + 22 * (n // 2)}" r="8" fill="{HUES[n + 1]}"/>'  # noqa: E501
        for n in range(4)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="AnyAlgebra">\n'
        f'<rect width="{width}" height="{height}" rx="16" fill="{PAPER}"/>\n'
        f"{dots}\n"
        f'<text x="104" y="64" {FONT} font-size="40" font-weight="700" fill="{INK}">'
        "AnyAlgebra</text>\n"
        f'<text x="106" y="92" {FONT} font-size="15" fill="{HUES[0]}">'
        "exact arithmetic for arbitrary algebraic structures</text>\n"
        "</svg>\n"
    )


def octonion_table() -> str:
    """Return the octonion multiplication table as a colored grid."""
    octonions = aa.octonions()
    labels = octonions.labels
    cell, margin = 54, 60
    size = margin + cell * len(labels) + 16
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size + 30}" '
        f'width="{size}" height="{size + 30}" role="img" '
        'aria-label="Multiplication table of the octonions">',
        f'<rect width="{size}" height="{size + 30}" rx="16" fill="{PAPER}"/>',
    ]
    for index, label in enumerate(labels):
        middle = margin + cell * index + cell // 2
        for x, y in ((middle, 38), (30, middle + 5)):
            parts.append(
                f'<text x="{x}" y="{y}" {FONT} font-size="16" font-weight="700" '
                f'text-anchor="middle" fill="{HUES[index]}">{label}</text>'
            )
    for row, left in enumerate(octonions.basis):
        for column, right in enumerate(octonions.basis):
            product = left * right
            ((label, value),) = product.coefficients.items()
            hue = HUES[labels.index(label)]
            x, y = margin + cell * column, margin + cell * row
            opacity = "0.16" if value > 0 else "0.36"
            parts.append(
                f'<rect x="{x + 2}" y="{y + 2}" width="{cell - 4}" height="{cell - 4}" '
                f'rx="8" fill="{hue}" fill-opacity="{opacity}" stroke="{RULE}"/>'
            )
            text = label if value > 0 else f"-{label}"
            parts.append(
                f'<text x="{x + cell // 2}" y="{y + cell // 2 + 5}" {FONT} font-size="15" '  # noqa: E501
                f'text-anchor="middle" fill="{INK}">{text}</text>'
            )
    parts.append(
        f'<text x="{size // 2}" y="{size + 12}" {FONT} font-size="13" text-anchor="middle" '  # noqa: E501
        f'fill="{HUES[0]}">row times column; darker cells carry a minus sign</text>'
    )
    return "\n".join(parts) + "\n</svg>\n"


FIGURES = {"wordmark.svg": wordmark, "octonion-table.svg": octonion_table}


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report stale files only")
    arguments = parser.parse_args()
    stale = []
    for name, render in FIGURES.items():
        path = ASSETS / name
        text = render()
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            stale.append(name)
            if not arguments.check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8", newline="\n")
    if arguments.check and stale:
        print("stale figures: " + ", ".join(stale))
        return 1
    print("figures are current" if not stale else "rewrote " + ", ".join(stale))
    return 0


if __name__ == "__main__":
    sys.exit(main())
