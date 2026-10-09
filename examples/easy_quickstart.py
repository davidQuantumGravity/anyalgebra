"""The convenience layer in a dozen lines; this example makes no research claim."""

from __future__ import annotations

import anyalgebra.easy as aa


def run_example() -> dict[str, str]:
    """Return a few exact results in two print forms."""
    quaternions, octonions = aa.quaternions(), aa.octonions()
    _, i, j, k = quaternions.basis
    x = (1 + 2 * i) * (j + k) / 2
    e = octonions.basis
    return {
        "i*j": str(i * j),
        "(i*j)*k": str((i * j) * k),
        "x": str(x),
        "x as a vector": f"{x:v}",
        "x in LaTeX": f"{x:l}",
        "[e1, e2]": str(aa.comm(e[1], e[2])),
        "(e1, e2, e4)": str(aa.assoc(e[1], e[2], e[4])),
        "quaternion table": quaternions.table(),
    }


if __name__ == "__main__":
    for title, value in run_example().items():
        if len(value.splitlines()) > 1:
            print(f"{title}:")
            print(value)
        else:
            print(f"{title} = {value}")
