"""The committed README figures are exactly what the generator produces."""

from __future__ import annotations

from tools import render_readme_figures


def test_committed_figures_are_current() -> None:
    for name, render in render_readme_figures.FIGURES.items():
        path = render_readme_figures.ASSETS / name
        assert path.read_text(encoding="utf-8") == render(), name
        assert path.read_text(encoding="utf-8").startswith("<svg xmlns=")
