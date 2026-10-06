"""Export a portable static AlgMul manifest from an explicitly supplied source.

This maintainer tool is intentionally outside the test path.  Tests consume the
committed manifest and never require the original research checkout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anyalgebra.legacy.algmul_static import capture_pinned_static
from anyalgebra.legacy.models import AlgMulManifest, SourceMetadata


FIXTURE_LOCATION = "fixture://legacy/algmul-static-manifest.json"


def export(source: Path, output: Path) -> None:
    captured = capture_pinned_static(source)
    portable_source = SourceMetadata.create(
        FIXTURE_LOCATION,
        captured.source.content_hash,
        source_bytes=captured.source.source_bytes,
        contexts=captured.source.contexts,
        load_status=captured.source.load_status,
    )
    portable = AlgMulManifest.create(
        manifest_id=captured.manifest_id,
        source=portable_source,
        symbols=captured.symbols,
        definitions=captured.definitions,
        factories=captured.factories,
        registries=captured.registries,
        behaviors=captured.behaviors,
        messages=captured.messages,
        dispositions=captured.dispositions,
    )
    output.write_bytes(portable.canonical_bytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    export(options.source, options.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
