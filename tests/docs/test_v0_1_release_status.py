from __future__ import annotations

import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.internal_records

ROOT = Path(__file__).resolve().parents[2]
VERSION_DIR = ROOT / ".agents" / "project-process" / "versions" / "v0.1"


def _read(path: str | Path) -> str:
    target = ROOT / path if isinstance(path, str) else path
    return target.read_text(encoding="utf-8")


def test_current_documents_identify_the_released_candidate() -> None:
    paths = (
        "README.md",
        "CHANGELOG.md",
        "docs/status.md",
        "docs/version.md",
        "docs/README.md",
        VERSION_DIR / "release-notes.md",
    )
    for path in paths:
        text = _read(path)
        assert "0.1.0" in text
        assert "V01-060" in text

    config = json.loads(_read(".agents/project-process/process-config.json"))
    assert config["versioning"]["releasedVersion"] == "v0.1"
    assert config["versioning"]["activeVersion"] == "v0.2"


def test_release_evidence_counts_and_hashes_are_synchronized() -> None:
    status = _read("docs/status.md")
    results = _read(VERSION_DIR / "test-results.md")
    release_notes = _read(VERSION_DIR / "release-notes.md")

    for text in (status, results, release_notes):
        assert "2,421 passed" in text
        assert "2,392 passed" in text
        assert "2" in text and "skipped" in text
        assert "302" in text and "deselected" in text
        assert "20,320/21,093" in text
        assert "6,529/7,052" in text
        assert "95.3953" in text
        assert "862" in text and "129" in text
        assert (
            "245384be017e38895d684456272eddb5e81a2974220debf348f41e17561eb843" in text
        )


def test_candidate_status_does_not_claim_tag_publication_or_research_completion() -> (
    None
):
    status = _read("docs/status.md").lower()
    release_notes = _read(VERSION_DIR / "release-notes.md").lower()
    for text in (status, release_notes):
        assert "complete" in text
        assert "not tagged" in text or "no git tag" in text
        assert "not published" in text or "publication is not claimed" in text
        assert "research" in text

    assert "v01-002 is next" not in status
    assert "no v0.1 implementation task has been accepted" not in release_notes


def test_acceptance_checklist_has_no_remaining_gate() -> None:
    checklist = _read(VERSION_DIR / "acceptance-checklist.md")
    unchecked = [line for line in checklist.splitlines() if line.startswith("- [ ]")]
    assert unchecked == []
