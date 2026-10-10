"""Render the README figures from the package's own data.

    python tools/render_readme_figures.py          # rewrite docs/assets/*.svg
    python tools/render_readme_figures.py --check  # fail if a file is stale

The octonion table is drawn from ``anyalgebra.easy.octonions()``, so the
picture cannot disagree with the code.
"""

from __future__ import annotations

import argparse
import html
import math
import sys
from pathlib import Path

import anyalgebra.easy as aa

ASSETS = Path(__file__).resolve().parents[1] / "docs" / "assets"
INK = "#1f2937"
PAPER = "#f8fafc"
RULE = "#cbd5e1"
# One hue per basis element; the unit is neutral.  e1, e2, e3 are red, green
# and blue, and e4 to e7 take the remaining well-separated hues.
HUES = (
    "#64748b",
    "#dc2626",
    "#16a34a",
    "#2563eb",
    "#ca8a04",
    "#0891b2",
    "#db2777",
    "#9333ea",
)
IN_HUE, OUT_HUE = HUES[3], HUES[1]
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


LINE = "#94a3b8"
SERIF = "font-family=\"Georgia, Cambria, 'Times New Roman', serif\""


def fano_lines() -> list[tuple[int, int, int]]:
    """Return the seven lines of the Fano plane read off the octonion table.

    A line is a triple ``(i, j, k)`` of imaginary units with ``ei * ej = ek``.
    """
    octonions = aa.octonions()
    labels = octonions.labels
    lines: dict[frozenset[int], tuple[int, int, int]] = {}
    for i in range(1, 8):
        for j in range(1, 8):
            if i == j:
                continue
            ((label, value),) = (octonions[i] * octonions[j]).coefficients.items()
            k = labels.index(label)
            if value > 0 and frozenset((i, j, k)) not in lines:
                lines[frozenset((i, j, k))] = (i, j, k)
    return list(lines.values())


def _unit(dx: float, dy: float) -> tuple[float, float]:
    length = math.sqrt(dx * dx + dy * dy)
    return dx / length, dy / length


def _arrow(x: float, y: float, dx: float, dy: float) -> str:
    """Return an arrowhead at a point, pointing along a direction."""
    ux, uy = _unit(dx, dy)
    tip = (x + 9 * ux, y + 9 * uy)
    left = (x - 6 * ux - 6 * uy, y - 6 * uy + 6 * ux)
    right = (x - 6 * ux + 6 * uy, y - 6 * uy - 6 * ux)
    points = " ".join(f"{px:.1f},{py:.1f}" for px, py in (tip, left, right))
    return f'<polygon points="{points}" fill="{INK}" fill-opacity="0.7"/>'


def fano_plane() -> str:
    """Return the Fano plane that encodes the octonion products."""
    lines = fano_lines()
    cycles = {frozenset(line): line for line in lines}

    def forward(p: int, q: int, r: int) -> bool:
        i, j, k = cycles[frozenset((p, q, r))]
        return (p, q, r) in ((i, j, k), (j, k, i), (k, i, j))

    circle = lines[0]
    center = max(set(range(1, 8)) - set(circle))
    opposite = {
        middle: next(iter(set(line) - {center, middle}))
        for middle in circle
        for line in lines
        if center in line and middle in line
    }
    width, height = 460, 470
    corners = ((230.0, 50.0), (50.0, 362.0), (410.0, 362.0))
    cx = sum(x for x, _ in corners) / 3
    cy = sum(y for _, y in corners) / 3
    place: dict[int, tuple[float, float]] = {center: (cx, cy)}
    for index, middle in enumerate(circle):
        place[opposite[middle]] = corners[index]
        (ax, ay), (bx, by) = (corners[n] for n in range(3) if n != index)
        place[middle] = ((ax + bx) / 2, (ay + by) / 2)
    radius = cy - corners[0][1]
    radius = radius / 2
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        'aria-label="The Fano plane of the octonions">',
        f'<rect width="{width}" height="{height}" rx="16" fill="{PAPER}"/>',
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="none" '
        f'stroke="{LINE}" stroke-width="2.5"/>',
    ]
    straight: list[tuple[int, int, int]] = []
    for index, middle in enumerate(circle):
        first, second = (opposite[circle[n]] for n in range(3) if n != index)
        straight.append((first, middle, second))
        straight.append((opposite[middle], center, middle))
    for p, _, r in straight:
        (x1, y1), (x2, y2) = place[p], place[r]
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{LINE}" stroke-width="2.5"/>'
        )
    for p, q, r in straight:
        if not forward(p, q, r):
            p, r = r, p
        for start, end in ((p, q), (q, r)):
            (x1, y1), (x2, y2) = place[start], place[end]
            parts.append(_arrow((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    # On the circle, the point between two side midpoints faces a corner.
    for index in range(3):
        before, after = (circle[n] for n in range(3) if n != index)
        ux, uy = _unit(corners[index][0] - cx, corners[index][1] - cy)
        (x1, y1), (x2, y2) = place[before], place[after]
        sign = 1 if forward(before, after, circle[index]) else -1
        parts.append(
            _arrow(
                cx + radius * ux, cy + radius * uy, sign * (x2 - x1), sign * (y2 - y1)
            )
        )
    labels = aa.octonions().labels
    for point, (x, y) in place.items():
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="19" fill="{PAPER}" '
            f'stroke="{HUES[point]}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{y + 5:.1f}" {FONT} font-size="15" font-weight="700" '  # noqa: E501
            f'text-anchor="middle" fill="{HUES[point]}">{labels[point]}</text>'
        )
    i, j, k = circle
    parts.append(
        f'<text x="{width // 2}" y="{height - 44}" {FONT} font-size="13" '
        f'text-anchor="middle" fill="{HUES[0]}">seven points, seven lines, three points '  # noqa: E501
        "on each line</text>"
    )
    parts.append(
        f'<text x="{width // 2}" y="{height - 22}" {FONT} font-size="13" '
        f'text-anchor="middle" fill="{HUES[0]}">along the arrows {labels[i]} {labels[j]} = '  # noqa: E501
        f"{labels[k]}; against them the sign flips</text>"
    )
    return "\n".join(parts) + "\n</svg>\n"


NOTEBOOK_CODE = (
    "import anyalgebra.easy as aa",
    "H = aa.quaternions()",
    "x = (1 + 2 * H.i) * (H.j + H.k) / 2",
)


def notebook_cell() -> str:
    """Return a drawing of two notebook cells with their rendered output.

    The code is executed here and the output is taken from the result, so
    the drawing shows what the package returns.
    """
    namespace: dict[str, object] = {}
    exec("\n".join(NOTEBOOK_CODE), namespace)
    quaternions, x = namespace["H"], namespace["x"]
    assert isinstance(quaternions, aa.Algebra) and isinstance(x, aa.Element)
    basis = quaternions.basis
    cell = 58
    width = 600
    grid_top = 236
    height = grid_top + cell * (len(basis) + 1) + 26
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        'aria-label="Two notebook cells and their rendered output">',
        f'<rect width="{width}" height="{height}" rx="16" fill="{PAPER}"/>',
        f'<rect x="92" y="18" width="{width - 110}" height="98" rx="6" fill="#ffffff" '
        f'stroke="{RULE}"/>',
        f'<text x="84" y="42" {FONT} font-size="13" text-anchor="end" fill="{IN_HUE}">'
        "In [1]:</text>",
    ]
    for row, line in enumerate((*NOTEBOOK_CODE, "x")):
        parts.append(
            f'<text x="104" y="{42 + 21 * row}" {FONT} font-size="14" fill="{INK}" '
            f'xml:space="preserve">{html.escape(line)}</text>'
        )
    parts += [
        f'<text x="84" y="148" {FONT} font-size="13" text-anchor="end" fill="{OUT_HUE}">'  # noqa: E501
        "Out[1]:</text>",
        f'<text x="104" y="150" {SERIF} font-size="21" font-style="italic" fill="{INK}">'  # noqa: E501
        f"{html.escape(format(x, 'pretty'))}</text>",
        f'<rect x="92" y="172" width="{width - 110}" height="34" rx="6" fill="#ffffff" '
        f'stroke="{RULE}"/>',
        f'<text x="84" y="194" {FONT} font-size="13" text-anchor="end" fill="{IN_HUE}">'
        "In [2]:</text>",
        f'<text x="104" y="194" {FONT} font-size="14" fill="{INK}">H</text>',
        f'<text x="84" y="{grid_top + 24}" {FONT} font-size="13" text-anchor="end" '
        f'fill="{OUT_HUE}">Out[2]:</text>',
    ]
    left = 104
    for index, element in enumerate(basis):
        text = html.escape(format(element, "pretty"))
        for x_, y_ in (
            (left + cell * (index + 1) + cell // 2, grid_top + 24),
            (left + cell // 2, grid_top + cell * (index + 1) + 24),
        ):
            parts.append(
                f'<text x="{x_}" y="{y_}" {SERIF} font-size="18" font-weight="700" '
                f'font-style="italic" text-anchor="middle" fill="{INK}">{text}</text>'
            )
    edge = left + cell * (len(basis) + 1)
    parts.append(
        f'<line x1="{left}" y1="{grid_top + 36}" x2="{edge}" y2="{grid_top + 36}" '
        f'stroke="{INK}" stroke-width="1.5"/>'
    )
    for row, a in enumerate(basis):
        if row % 2:
            parts.append(
                f'<rect x="{left}" y="{grid_top + cell * (row + 1) - 12}" '
                f'width="{edge - left}" height="{cell}" fill="{RULE}" fill-opacity="0.35"/>'  # noqa: E501
            )
        for column, b in enumerate(basis):
            parts.append(
                f'<text x="{left + cell * (column + 1) + cell // 2}" '
                f'y="{grid_top + cell * (row + 1) + 24}" {SERIF} font-size="18" '
                f'font-style="italic" text-anchor="middle" fill="{INK}">'
                f"{html.escape(format(a * b, 'pretty'))}</text>"
            )
    return "\n".join(parts) + "\n</svg>\n"


FIGURES = {
    "wordmark.svg": wordmark,
    "octonion-table.svg": octonion_table,
    "fano-plane.svg": fano_plane,
    "notebook-cell.svg": notebook_cell,
}


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
