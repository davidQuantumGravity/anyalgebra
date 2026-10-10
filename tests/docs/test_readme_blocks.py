"""Every Python block in the README runs, and prints what the README shows."""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

README = Path(__file__).resolve().parents[2] / "README.md"
BLOCK = re.compile(r"```(python|text)\n(.*?)```", flags=re.S)


def test_readme_code_blocks_print_the_shown_output() -> None:
    namespace: dict[str, object] = {"__name__": "readme"}
    printed: str | None = None
    compared = 0
    for language, body in BLOCK.findall(README.read_text(encoding="utf-8")):
        if language == "python":
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(compile(body, "README.md", "exec"), namespace)
            printed = output.getvalue()
        elif printed is not None:
            assert printed.strip() == body.strip()
            printed = None
            compared += 1
    assert compared == 10


def test_readme_shows_no_kernel_repr_and_links_existing_files() -> None:
    text = README.read_text(encoding="utf-8")
    assert "_IntegerElement" not in text and "SparseElement(" not in text
    for target in re.findall(
        r"\]\((docs/[^)#]+|LICENSE|COMMERCIAL-LICENSING.md)\)", text
    ):
        assert (README.parent / target).exists(), target
    for image in re.findall(r'src="(docs/assets/[^"]+)"', text):
        assert (README.parent / image).is_file(), image


def test_api_guide_code_blocks_run() -> None:
    import anyalgebra.easy as aa

    forms = dict(aa._FORMS)
    try:
        for name in ("easy", "composition", "families"):
            guide = README.parent / "docs" / "api" / f"{name}.md"
            namespace: dict[str, object] = {"__name__": name}
            blocks = [
                body
                for language, body in BLOCK.findall(guide.read_text(encoding="utf-8"))
                if language == "python"
            ]
            assert blocks, name
            for body in blocks:
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(body, guide.name, "exec"), namespace)
    finally:
        aa._FORMS.clear()
        aa._FORMS.update(forms)
