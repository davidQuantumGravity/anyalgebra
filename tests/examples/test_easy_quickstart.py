"""Public-example test for the convenience-layer quickstart."""

from __future__ import annotations

from examples.easy_quickstart import run_example


def test_easy_quickstart_example() -> None:
    result = run_example()
    assert result["i*j"] == "k"
    assert result["(i*j)*k"] == "-1"
    assert result["x"] == "-1/2*j + 3/2*k"
    assert result["x as a vector"] == "[0, 0, -1/2, 3/2]"
    assert result["x in LaTeX"] == r"-\frac{1}{2}\,j + \frac{3}{2}\,k"
    assert result["[e1, e2]"] == "2*e3"
    assert result["(e1, e2, e4)"] == "2*e7"
    assert result["quaternion table"].splitlines()[2] == "i     i  -1   k  -j"
