"""The committed README figures are exactly what the generator produces."""

from __future__ import annotations

from tools import render_readme_figures


def test_committed_figures_are_current() -> None:
    for name, render in render_readme_figures.FIGURES.items():
        path = render_readme_figures.ASSETS / name
        assert path.read_text(encoding="utf-8") == render(), name
        assert path.read_text(encoding="utf-8").startswith("<svg xmlns=")


def test_fano_lines_read_from_the_table_form_a_projective_plane() -> None:
    lines = render_readme_figures.fano_lines()
    assert len(lines) == 7 and lines[0] == (1, 2, 3)
    for point in range(1, 8):
        assert sum(point in line for line in lines) == 3
    for first in range(1, 8):
        for second in range(first + 1, 8):
            assert sum(first in line and second in line for line in lines) == 1
    figure = render_readme_figures.fano_plane()
    assert figure.count("<polygon") == 15 and "e1 e2 = e3" in figure
    assert "3/2 k" in render_readme_figures.notebook_cell()
