"""Build the complete bounded v0.1 commutative-magma reference atlas.

The corpus is deliberately small enough for independent exhaustive replay.  A
binary table on three labeled elements has nine cells and therefore ``3^9``
raw possibilities.  Commutativity leaves six unordered input pairs, giving the
independent accepted count ``3^6`` before any program output is consulted.
The next carrier size would require ``4^16`` raw tables, beyond the reference
enumerator's hard batch ceiling, so order three is its largest complete binary
carrier control.  The commutative subcorpus keeps the published atlas compact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

from anyalgebra.atlas.build import build_finite_algebra_atlas
from anyalgebra.atlas.query import load_finite_algebra_atlas, query_atlas
from anyalgebra.census.bounds import CensusOrdering, EnumerationBounds
from anyalgebra.census.constraints import CensusConstraint, ConstraintSet
from anyalgebra.census.equivalence import EquivalencePolicy
from anyalgebra.census.spec import CensusSpec, CensusSpecCore


class AtlasExampleReport(TypedDict):
    """Deterministic, JSON-safe summary of the published reference atlas."""

    analysis: dict[str, int]
    corpus: dict[str, int | str]
    evidence: dict[str, int | str]
    orbits: dict[str, int | str]
    scope: str


def reference_spec() -> CensusSpec:
    """Return the exact immutable specification for the v0.1 control corpus."""
    core = CensusSpecCore.create(
        carrier_size=3,
        arity=2,
        corpus_name="v01-reference-commutative-magma-n3",
    )
    return CensusSpec.create(
        carrier_size=3,
        arity=2,
        constraints=ConstraintSet.create(
            core,
            (CensusConstraint.commutative(),),
        ),
        equivalence=EquivalencePolicy.relabeling(core),
        ordering=CensusOrdering.reference(),
        bounds=EnumerationBounds.create(
            max_candidates=19_683,
            max_orbits=1_000,
            max_work_units=19_683,
            max_memory_bytes=8_000_000,
        ),
    )


def build_reference_atlas(output_directory: Path) -> AtlasExampleReport:
    """Build, reload, verify, query, and summarize the bounded atlas."""
    spec = reference_spec()
    build = build_finite_algebra_atlas(
        (spec,),
        output_directory=output_directory,
    )
    loaded = load_finite_algebra_atlas(build.output_directory)
    all_matches = query_atlas(loaded)
    corpus = build.atlas.corpora[0]

    def law_count(name: str) -> int:
        return len(query_atlas(loaded, laws=(name,)).matches)

    def invariant_count(name: str, value: str | int | bool) -> int:
        return len(query_atlas(loaded, invariant_fields=((name, value),)).matches)

    return {
        "analysis": {
            "associativeOrbitCount": law_count("associativity"),
            "commutativeOrbitCount": law_count("commutativity"),
            "flexibleOrbitCount": law_count("flexibility"),
            "idempotentOrbitCount": law_count("idempotence"),
            "uniqueIdentityOrbitCount": invariant_count("identity_status", "unique"),
            "uniqueZeroOrbitCount": invariant_count("zero_status", "unique"),
        },
        "corpus": {
            "acceptedLabeledCount": corpus.accepted_labeled_count,
            "commutativeDegreesOfFreedom": 6,
            "examinedCandidateCount": corpus.examined_candidate_count,
            "independentAcceptedCount": "3^6=729",
            "rawTableCount": "3^9=19683",
            "rejectedLabeledCount": corpus.rejected_labeled_count,
            "status": "complete" if corpus.complete else "incomplete",
        },
        "evidence": {
            "atlasSemanticHash": str(build.atlas.semantic_hash),
            "objectCount": build.object_count,
            "specSemanticHash": str(spec.semantic_hash),
            "verifiedObjectCount": loaded.verified_object_count,
        },
        "orbits": {
            "acceptedMemberCertificateCount": sum(
                representative.accepted_member_count
                for representative in corpus.representatives
            ),
            "equivalence": "full carrier relabeling",
            "permutationCount": spec.equivalence.permutation_count,
            "representativeCount": len(all_matches.matches),
        },
        "scope": (
            "complete only for binary commutative operation tables on the labeled "
            "carrier {0,1,2}, modulo full carrier relabeling; no broader algebra "
            "classification or physics claim"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the example with one explicit, fresh output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="fresh absolute atlas output directory",
    )
    arguments = parser.parse_args(argv)
    report = build_reference_atlas(arguments.output)
    print(json.dumps(report, allow_nan=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
