"""Read-only, executable traceability audit for the AnyAlgebra v0.0 scope.

The audit intentionally treats registry links as data, not prose.  In
particular, a test ID is owned only when it names collected pytest node IDs;
an API target is owned only when its documented import can be safely resolved
inside this repository and its callable signature agrees with its contract.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import re
import subprocess
import sys
from typing import get_args, get_origin, get_type_hints
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROCESS = Path(".agents") / "project-process"
SCOPE_PATH = PROCESS / "versions" / "v0.0" / "scope.json"
SNAPSHOT_PATH = PROCESS / "versions" / "v0.0" / "contract-snapshot.json"
QUICKSTART_PATH = Path("docs") / "api" / "v0.0-quickstart.md"
ADR_GATE_PATH = PROCESS / "versions" / "v0.0" / "adr-gate.json"
PROGRESS_PATH = (
    PROCESS / "versions" / "v0.0" / "implementation" / "implementation-progress.json"
)
SKILL_LOG_PATH = PROCESS / "versions" / "v0.0" / "skill-run-log.md"
REGISTRIES = {
    "featureIds": "features.json",
    "actionIds": "actions.json",
    "storyIds": "user-stories.json",
    "testIds": "test-matrix.json",
    "apiIds": "api-contracts.json",
    "componentIds": "components.json",
    "stateIds": "states.json",
}
EXPECTED_COUNTS = {"apiIds": 23, "testIds": 22}
REVIEWED_SCOPE_IDS = {
    "featureIds": frozenset(
        (
            "fe.foundation.exact_domains",
            "fe.foundation.parents_elements",
            "fe.structures.generalized",
            "fe.structures.partiality",
            "fe.presentations.conversions",
            "fe.linear.operator_recovery",
            "fe.validation.laws_analysis",
            "fe.evidence.receipts_serialization",
            "fe.backends.reference_boundary",
            "fe.legacy.algmul_supersession",
        )
    ),
    "actionIds": frozenset(
        (
            "action.api.coerce_values",
            "action.api.define_structure",
            "action.api.evaluate_term",
            "action.api.convert_presentation",
            "action.api.recover_structure_constants",
            "action.api.validate_law",
            "action.api.analyze_algebra",
            "action.api.emit_receipt",
            "action.api.replay_receipt",
            "action.audit.capture_algmul",
        )
    ),
    "storyIds": frozenset(
        (
            "us.foundation.mix_exact_domains",
            "us.foundation.define_coexisting_parents",
            "us.structures.define_unknown_structure",
            "us.structures.handle_partiality",
            "us.presentations.round_trip",
            "us.linear.recover_structure",
            "us.validation.obtain_witness",
            "us.evidence.replay_calculation",
            "us.backends.run_without_optionals",
            "us.legacy.migrate_algmul_capability",
        )
    ),
    "componentIds": frozenset(
        (
            "component.core.domains",
            "component.core.parents",
            "component.core.structures",
            "component.core.presentations",
            "component.core.linear",
            "component.core.validation",
            "component.infrastructure.evidence",
            "component.legacy.algmul",
        )
    ),
    "stateIds": frozenset(
        (
            "state.evaluation.outcome",
            "state.validation.outcome",
            "state.evidence.execution",
            "state.evidence.outcome",
        )
    ),
    "decisionIds": frozenset(f"ADR-{number:04d}" for number in range(1, 9)),
}
QUICKSTART_ROW = re.compile(
    r"^\| `(?P<id>api\.[^`]+)` \| `from (?P<module>[\w.]+) import "
    r"(?P<symbol>\w+)` \| `(?P<call>[^`]+)` \|$",
    re.MULTILINE,
)
# v0.0 has a deliberately small public surface, so a precise executable
# contract is more useful than attempting to infer semantics from prose.  A
# target lists its complete parameter *shape*, including keyword-only markers
# and defaults.  Every callable named by an API contract is listed here; a
# quickstart row is permitted to advertise one, and only one, of these targets.
NO_DEFAULT = object()


@dataclass(frozen=True)
class CallableSpec:
    module: str
    symbol: str
    member: str | None
    parameters: tuple[tuple[str, inspect._ParameterKind, object], ...]

    @property
    def display_name(self) -> str:
        return self.symbol if self.member is None else f"{self.symbol}.{self.member}"


P = inspect.Parameter


def _ps(
    *items: tuple[str, inspect._ParameterKind, object],
) -> tuple[tuple[str, inspect._ParameterKind, object], ...]:
    return items


API_SPECS: dict[str, tuple[CallableSpec, ...]] = {
    "api.domain.construct_exact": (
        CallableSpec("anyalgebra.core.domains", "ZZ", None, _ps()),
        CallableSpec("anyalgebra.core.domains", "QQ", None, _ps()),
        CallableSpec(
            "anyalgebra.core.domains",
            "IntegerDomain",
            "element",
            _ps(
                ("self", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("value", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
            ),
        ),
        CallableSpec(
            "anyalgebra.core.domains",
            "RationalDomain",
            "element",
            _ps(
                ("self", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("value", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
            ),
        ),
    ),
    "api.coercion.plan_apply": (
        CallableSpec(
            "anyalgebra.core.coercions",
            "common_parent",
            None,
            _ps(
                ("values", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("graph", P.KEYWORD_ONLY, NO_DEFAULT),
            ),
        ),
        CallableSpec(
            "anyalgebra.core.coercions",
            "coerce",
            None,
            _ps(
                ("value", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("target", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("graph", P.KEYWORD_ONLY, NO_DEFAULT),
            ),
        ),
    ),
    "api.parent.freeze": (
        CallableSpec(
            "anyalgebra.core.parents",
            "ParentBuilder",
            "freeze",
            _ps(("self", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT)),
        ),
    ),
    "api.module.element": (
        CallableSpec(
            "anyalgebra.core.modules",
            "FreeModule",
            "element",
            _ps(
                ("self", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("coordinates", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("graph", P.KEYWORD_ONLY, None),
            ),
        ),
    ),
    "api.structure.build": (
        CallableSpec(
            "anyalgebra.structures.structure",
            "StructureBuilder",
            None,
            _ps(("signature", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT)),
        ),
        CallableSpec(
            "anyalgebra.structures.structure",
            "StructureBuilder",
            "with_carrier",
            _ps(
                ("self", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("sort", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("carrier", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
            ),
        ),
        CallableSpec(
            "anyalgebra.structures.structure",
            "StructureBuilder",
            "with_operation",
            _ps(
                ("self", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("symbol", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("operation", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
            ),
        ),
        CallableSpec(
            "anyalgebra.structures.structure",
            "StructureBuilder",
            "with_relation",
            _ps(
                ("self", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("symbol", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("relation", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
            ),
        ),
        CallableSpec(
            "anyalgebra.structures.structure",
            "StructureBuilder",
            "freeze",
            _ps(("self", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT)),
        ),
    ),
    "api.structure.evaluate": (
        CallableSpec(
            "anyalgebra.structures.evaluate",
            "evaluate_term",
            None,
            _ps(
                ("structure", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("term", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("environment", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
            ),
        ),
    ),
    "api.structure.deductive_closure": (
        CallableSpec(
            "anyalgebra.structures.deductive",
            "closure",
            None,
            _ps(
                ("premises", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("system", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("max_rounds", P.KEYWORD_ONLY, NO_DEFAULT),
            ),
        ),
    ),
    "api.partiality.totalize": (
        CallableSpec(
            "anyalgebra.structures.partiality",
            "totalize",
            None,
            _ps(
                ("partial_operation", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("bottom", P.KEYWORD_ONLY, NO_DEFAULT),
            ),
        ),
    ),
    "api.algebra.from_structure_constants": (
        CallableSpec(
            "anyalgebra.algebra.multilinear",
            "FiniteMultilinearStructure",
            "from_structure_constants",
            _ps(
                ("module", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("constants", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("name", P.KEYWORD_ONLY, None),
            ),
        ),
    ),
    "api.presentation.convert": (
        CallableSpec(
            "anyalgebra.presentations.convert",
            "convert",
            None,
            _ps(
                ("value", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("target_kind", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("graph", P.KEYWORD_ONLY, NO_DEFAULT),
                ("source_kind", P.KEYWORD_ONLY, NO_DEFAULT),
                ("route", P.KEYWORD_ONLY, None),
            ),
        ),
    ),
    "api.presentation.change_basis": (
        CallableSpec(
            "anyalgebra.presentations.basis_change",
            "change_basis",
            None,
            _ps(
                ("value_or_structure", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("isomorphism", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
            ),
        ),
    ),
    "api.matrix.construct": (
        CallableSpec(
            "anyalgebra.linear.matrix",
            "MatrixSpace",
            None,
            _ps(
                ("rows", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("columns", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("entry_parent", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
            ),
        ),
        CallableSpec(
            "anyalgebra.linear.matrix",
            "MatrixSpace",
            "element",
            _ps(
                ("self", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("entries", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
            ),
        ),
    ),
    "api.linear.recover_structure_constants": (
        CallableSpec(
            "anyalgebra.linear.recovery",
            "recover_structure_constants",
            None,
            _ps(
                ("operators", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("declared_bracket", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
            ),
        ),
    ),
    "api.validation.validate_law": (
        CallableSpec(
            "anyalgebra.validation.validate",
            "validate_law",
            None,
            _ps(
                ("structure", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("law", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
            ),
        ),
    ),
    "api.analysis.algebra_fingerprint": (
        CallableSpec(
            "anyalgebra.analysis.fingerprint",
            "algebra_fingerprint",
            None,
            _ps(
                ("algebra", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("options", P.KEYWORD_ONLY, None),
            ),
        ),
    ),
    "api.serialization.safe_record": (
        CallableSpec(
            "anyalgebra.persistence.registry",
            "SerializerRegistry",
            "to_record",
            _ps(
                ("self", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("value", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
            ),
        ),
        CallableSpec(
            "anyalgebra.persistence.registry",
            "SerializerRegistry",
            "from_record",
            _ps(
                ("self", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("record", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
            ),
        ),
        CallableSpec(
            "anyalgebra.persistence.canonical",
            "canonical_json",
            None,
            _ps(
                ("value", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("registry", P.KEYWORD_ONLY, NO_DEFAULT),
            ),
        ),
    ),
    "api.evidence.contract": (
        CallableSpec(
            "anyalgebra.evidence.contracts",
            "CalculationContract",
            "create",
            _ps(
                *tuple(
                    (name, P.KEYWORD_ONLY, NO_DEFAULT)
                    for name in (
                        "contract_name",
                        "project_id",
                        "question",
                        "acceptable_outcomes",
                        "input_hashes",
                        "source_anchors",
                        "convention_manifests",
                        "algorithms",
                        "backend_request",
                        "backend_name",
                        "backend_version",
                        "assumptions",
                        "theorem_hypotheses",
                        "bounds",
                        "required_cross_checks",
                        "expected_artifacts",
                        "acceptance_predicates",
                    )
                )
            ),
        ),
    ),
    "api.evidence.run": (
        CallableSpec(
            "anyalgebra.evidence.run",
            "run_calculation",
            None,
            _ps(
                ("contract", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("calculation", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("environment", P.KEYWORD_ONLY, None),
            ),
        ),
    ),
    "api.evidence.replay": (
        CallableSpec(
            "anyalgebra.evidence.replay",
            "verify_replay",
            None,
            _ps(
                ("receipt", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("resolver", P.KEYWORD_ONLY, NO_DEFAULT),
                ("environment", P.KEYWORD_ONLY, None),
            ),
        ),
    ),
    "api.backend.capabilities": (
        CallableSpec(
            "anyalgebra.backends.reference",
            "ReferenceBackend",
            "capabilities",
            _ps(("self", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT)),
        ),
        CallableSpec(
            "anyalgebra.backends.reference",
            "ReferenceBackend",
            "supports",
            _ps(
                ("self", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("request", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
            ),
        ),
        CallableSpec(
            "anyalgebra.backends.reference",
            "ReferenceBackend",
            "execute",
            _ps(
                ("self", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("request", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("options", P.KEYWORD_ONLY, NO_DEFAULT),
            ),
        ),
    ),
    "api.legacy.capture_static": (
        CallableSpec(
            "anyalgebra.legacy.algmul_static",
            "capture_static",
            None,
            _ps(("source_path", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT)),
        ),
    ),
    "api.legacy.capture_runtime": (
        CallableSpec(
            "tools.capture_algmul_runtime",
            "capture_runtime",
            None,
            _ps(
                ("source_path", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("adapter", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                (
                    "expected_source_sha256",
                    P.KEYWORD_ONLY,
                    "3A13C9F47092B5B9814AA08873FF4F1DE61D4635DF68FC7F3A039A7DA5B0D00D",
                ),
                ("max_payload_bytes", P.KEYWORD_ONLY, 1048576),
            ),
        ),
        CallableSpec(
            "anyalgebra.legacy.mathematica",
            "MathematicaAdapter",
            "execute",
            _ps(
                ("self", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("script", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
            ),
        ),
    ),
    "api.legacy.validate_complete": (
        CallableSpec(
            "anyalgebra.legacy.models",
            "AlgMulManifest",
            "validate_complete",
            _ps(
                ("self", P.POSITIONAL_OR_KEYWORD, NO_DEFAULT),
                ("owned_partitions", P.KEYWORD_ONLY, NO_DEFAULT),
            ),
        ),
    ),
}

API_IDS = frozenset(API_SPECS)

REGISTRY_SIGNATURES = {
    "api.domain.construct_exact": "ZZ(); QQ(); IntegerDomain.element(value); RationalDomain.element(value)",
    "api.coercion.plan_apply": "common_parent(values, *, graph); coerce(value, target, *, graph)",
    "api.parent.freeze": "ParentBuilder.freeze()",
    "api.module.element": "FreeModule.element(coordinates, *, graph=None)",
    "api.structure.build": "StructureBuilder(signature).with_carrier(sort, carrier).with_operation(symbol, operation).with_relation(symbol, relation).freeze()",
    "api.structure.evaluate": "evaluate_term(structure, term, environment)",
    "api.structure.deductive_closure": "closure(premises, system, *, max_rounds)",
    "api.partiality.totalize": "totalize(partial_operation, *, bottom)",
    "api.algebra.from_structure_constants": "FiniteMultilinearStructure.from_structure_constants(module, constants, *, name=None)",
    "api.presentation.convert": "convert(value, target_kind, *, graph, source_kind, route=None)",
    "api.presentation.change_basis": "change_basis(value_or_structure, isomorphism)",
    "api.matrix.construct": "MatrixSpace(rows, columns, entry_parent).element(entries)",
    "api.linear.recover_structure_constants": "recover_structure_constants(operators, declared_bracket)",
    "api.validation.validate_law": "validate_law(structure, law)",
    "api.analysis.algebra_fingerprint": "algebra_fingerprint(algebra, *, options=None)",
    "api.serialization.safe_record": "SerializerRegistry.to_record(value); SerializerRegistry.from_record(record); canonical_json(value, *, registry)",
    "api.evidence.contract": "CalculationContract.create(*, contract_name, project_id, question, acceptable_outcomes, input_hashes, source_anchors, convention_manifests, algorithms, backend_request, backend_name, backend_version, assumptions, theorem_hypotheses, bounds, required_cross_checks, expected_artifacts, acceptance_predicates)",
    "api.evidence.run": "run_calculation(contract, calculation, *, environment=None)",
    "api.evidence.replay": "verify_replay(receipt, *, resolver, environment=None)",
    "api.backend.capabilities": "ReferenceBackend.capabilities(); ReferenceBackend.supports(request); ReferenceBackend.execute(request, *, options)",
    "api.legacy.capture_static": "capture_static(source_path)",
    "api.legacy.capture_runtime": "capture_runtime(source_path, adapter, *, expected_source_sha256='3A13C9F47092B5B9814AA08873FF4F1DE61D4635DF68FC7F3A039A7DA5B0D00D', max_payload_bytes=1048576); MathematicaAdapter.execute(script)",
    "api.legacy.validate_complete": "AlgMulManifest.validate_complete(*, owned_partitions)",
}

PUBLISHED_CALLS = {
    "api.domain.construct_exact": "ZZ()",
    "api.coercion.plan_apply": "common_parent(values, *, graph)",
    "api.parent.freeze": "ParentBuilder.freeze()",
    "api.module.element": "FreeModule.element(coordinates, *, graph=None)",
    "api.structure.build": "StructureBuilder(signature)",
    "api.structure.evaluate": "evaluate_term(structure, term, environment)",
    "api.structure.deductive_closure": "closure(premises, system, *, max_rounds)",
    "api.partiality.totalize": "totalize(partial_operation, *, bottom)",
    "api.algebra.from_structure_constants": "FiniteMultilinearStructure.from_structure_constants(module, constants, *, name=None)",
    "api.presentation.convert": "convert(value, target_kind, *, graph, source_kind, route=None)",
    "api.presentation.change_basis": "change_basis(value_or_structure, isomorphism)",
    "api.matrix.construct": "MatrixSpace(rows, columns, entry_parent)",
    "api.linear.recover_structure_constants": "recover_structure_constants(operators, declared_bracket)",
    "api.validation.validate_law": "validate_law(structure, law)",
    "api.analysis.algebra_fingerprint": "algebra_fingerprint(algebra, *, options=None)",
    "api.serialization.safe_record": "SerializerRegistry.from_record(record)",
    "api.evidence.contract": "CalculationContract.create(*, contract_name, project_id, question, acceptable_outcomes, input_hashes, source_anchors, convention_manifests, algorithms, backend_request, backend_name, backend_version, assumptions, theorem_hypotheses, bounds, required_cross_checks, expected_artifacts, acceptance_predicates)",
    "api.evidence.run": "run_calculation(contract, calculation, *, environment=None)",
    "api.evidence.replay": "verify_replay(receipt, *, resolver, environment=None)",
    "api.backend.capabilities": "ReferenceBackend.capabilities()",
    "api.legacy.capture_static": "capture_static(source_path)",
    "api.legacy.capture_runtime": "capture_runtime(source_path, adapter, *, expected_source_sha256='3A13C9F47092B5B9814AA08873FF4F1DE61D4635DF68FC7F3A039A7DA5B0D00D', max_payload_bytes=1048576)",
    "api.legacy.validate_complete": "AlgMulManifest.validate_complete(*, owned_partitions)",
}
TEST_OWNERS: dict[str, tuple[str, ...]] = {
    "test.domains.canonical_exact": (
        "tests/core/test_exact_domain_properties.py::test_zz_constructor_equivalence_is_exhaustive_in_the_declared_box",
        "tests/core/test_exact_domain_properties.py::test_qq_constructor_equivalence_and_normal_form_are_exhaustive",
    ),
    "test.coercions.unique_or_ambiguous": (
        "tests/integration/test_exact_domains_and_coercions.py::test_declared_small_graph_family_exhausts_success_and_failure_routes",
    ),
    "test.parents.identity_and_mismatch": (
        "tests/integration/test_coexisting_parents.py::test_independent_equal_looking_parents_require_explicit_transport",
    ),
    "test.elements.sparse_canonical": (
        "tests/core/test_sparse_element.py::test_sparse_canonicalizes_raw_and_same_domain_coefficients",
    ),
    "test.structures.arbitrary_signature": (
        "tests/integration/test_required_structure_examples.py::test_two_sorted_example_assembles_and_exercises_operation_and_relation",
    ),
    "test.structures.relational_and_deductive": (
        "tests/structures/test_deductive.py::test_ground_axioms_joins_order_and_replayable_derivations",
    ),
    "test.partiality.all_outcomes": (
        "tests/structures/test_outcomes.py::test_evaluation_outcome_alias_is_closed_over_exactly_four_branches",
    ),
    "test.partiality.no_silent_totalization": (
        "tests/structures/test_totalize.py::test_totalize_maps_explicit_and_omitted_undefined_cells_to_bottom",
    ),
    "test.presentations.round_trip": (
        "tests/presentations/test_convert.py::test_route_validation_inverse_and_round_trip_metadata",
    ),
    "test.presentations.ambiguity": (
        "tests/presentations/test_convert.py::test_ambiguity_cost_and_explicit_disambiguation_never_call_candidates",
    ),
    "test.linear.unique_recovery": (
        "tests/linear/test_operator_recovery.py::test_abelian_and_nonabelian_recovery",
    ),
    "test.linear.nonclosure_and_dependence": (
        "tests/linear/test_operator_recovery.py::test_dependency_and_nonclosure_do_not_return_constants",
    ),
    "test.validation.proved_disproved_inconclusive": (
        "tests/validation/test_reports.py::test_positive_exhaustive_and_checked_theorem_proofs",
        "tests/validation/test_reports.py::test_deterministic_disproof_and_bounded_inconclusive_reports",
        "tests/validation/test_basis_reduction_gate.py::test_each_required_hypothesis_kind_is_independently_denied[0-missing_multilinearity:product:0]",
    ),
    "test.analysis.basis_change_invariance": (
        "tests/integration/test_validation_basis_invariance.py::test_exact_shear_preserves_fingerprint_and_all_octonion_basis_associators",
    ),
    "test.serialization.canonical_safe": (
        "tests/persistence/test_canonical_json.py::test_canonical_json_has_golden_utf8_bytes_and_sorted_keys",
        "tests/persistence/test_record_registry.py::test_unknown_schema_type_and_version_have_explicit_diagnostics",
        "tests/persistence/test_record_registry.py::test_malformed_or_unknown_metadata_does_not_call_a_registered_parser",
    ),
    "test.evidence.no_self_promotion": (
        "tests/evidence/test_claim_promotion.py::test_each_promotion_is_append_only_next_state_and_preserves_prior",
        "tests/evidence/test_claim_promotion.py::test_promotion_rejects_weak_receipt_and_skips[E2-regression-E1-executed]",
    ),
    "test.evidence.replay_and_staleness": (
        "tests/evidence/test_replay.py::test_direct_changes_are_stale_before_result_comparison",
    ),
    "test.backends.reference_without_optionals": (
        "tests/backends/test_reference_loading.py::test_reference_keeps_actual_capability_provenance_for_genuine_absence",
    ),
    "test.backends.differential_contract": (
        "tests/backends/test_differential.py::test_exact_match_is_e1_receipt_bound_and_backend_order_is_deterministic",
    ),
    "test.legacy.evaluated_surface": (
        "tests/legacy/test_algmul_runtime_capture.py::test_fixture_is_current_evidence_with_complete_registries",
    ),
    "test.legacy.generated_property_family": (
        "tests/legacy/test_generated_property_family.py::test_makeproperty_fixture_enumerates_the_complete_intended_surface",
    ),
    "test.legacy.disposition_complete": (
        "tests/legacy/test_disposition_complete.py::test_every_static_and_evaluated_target_receives_one_typed_disposition",
    ),
}
REQUIRED_DOCUMENTS = (
    "docs/specifications/v0.0-spec.md",
    "docs/testing/testing-strategy.md",
    ".agents/project-process/versions/v0.0/acceptance-checklist.md",
    ".agents/context-curation/context-packs/CTX-009-v0-traceability-pack.md",
    ".agents/context-curation/context-packs/CTX-065-v0-acceptance-closure.md",
    ".agents/project-process/versions/v0.0/carryover.md",
)
REQUIRED_DOCUMENT_TOKENS = {
    "docs/specifications/v0.0-spec.md": ("# AnyAlgebra v0.0", "## 9."),
    "docs/testing/testing-strategy.md": ("#", "##"),
    ".agents/project-process/versions/v0.0/acceptance-checklist.md": (
        "# v0.0 Acceptance Checklist",
        "## Traceability",
    ),
    ".agents/context-curation/context-packs/CTX-009-v0-traceability-pack.md": (
        "# CTX-009",
        "APIs",
    ),
    ".agents/context-curation/context-packs/CTX-065-v0-acceptance-closure.md": (
        "# CTX-065",
        "Gate matrix",
    ),
    ".agents/project-process/versions/v0.0/carryover.md": (
        "# v0.0 Carryover",
        "## Deferred",
    ),
}
RELEASE_EXAMPLE_OWNERS = {
    "examples/two_sorted_release.py": (
        "tests/examples/test_two_sorted_release.py::test_release_summary_and_cli_are_deterministic_and_claim_bounded",
    ),
    "examples/partial_and_deductive_release.py": (
        "tests/examples/test_partial_and_deductive_release.py::test_release_summary_and_cli_are_deterministic_and_claim_bounded",
    ),
    "examples/generic_composition_algebras.py": (
        "tests/examples/test_generic_composition_algebras.py::test_cli_is_deterministic_json_and_claim_bounded",
    ),
    "examples/operator_dossier.py": (
        "tests/examples/test_operator_dossier.py::test_dossier_cli_is_cwd_independent_compact_json",
    ),
    "examples/replay_dossier.py": (
        "tests/examples/test_replay_dossier.py::test_dossier_cli_is_cwd_independent_deterministic_json",
    ),
}
COMPONENT_OWNERS: dict[str, tuple[str, ...]] = {
    "component.core.domains": ("anyalgebra.core.domains", "anyalgebra.core.coercions"),
    "component.core.parents": ("anyalgebra.core.parents", "anyalgebra.core.modules"),
    "component.core.structures": ("anyalgebra.structures.structure",),
    "component.core.presentations": ("anyalgebra.presentations.convert",),
    "component.core.linear": ("anyalgebra.linear.recovery",),
    "component.core.validation": ("anyalgebra.validation.validate",),
    "component.infrastructure.evidence": ("anyalgebra.evidence.run",),
    "component.legacy.algmul": ("anyalgebra.legacy.models",),
}
STATE_OWNERS: dict[str, tuple[str, str, str | None]] = {
    "state.evaluation.outcome": (
        "anyalgebra.structures.outcomes",
        "EvaluationOutcome",
        None,
    ),
    "state.validation.outcome": (
        "anyalgebra.validation.reports",
        "ValidationReport",
        None,
    ),
    "state.evidence.execution": (
        "anyalgebra.evidence.run",
        "ResultReceipt",
        "execution_status",
    ),
    "state.evidence.outcome": (
        "anyalgebra.evidence.run",
        "ResultReceipt",
        "mathematical_outcome",
    ),
}
STATE_VALUES = {
    "state.evaluation.outcome": ("defined", "undefined", "indeterminate", "failed"),
    "state.validation.outcome": ("proved", "disproved", "inconclusive"),
    "state.evidence.execution": (
        "not_run",
        "running",
        "completed",
        "failed",
        "blocked",
    ),
    "state.evidence.outcome": (
        "constructed",
        "verified_within_domain",
        "counterexample",
        "no_go_within_assumptions",
        "inconclusive",
        "not_applicable",
        "reproduction_only",
        "implementation_error",
    ),
}


@dataclass(frozen=True, order=True)
class Finding:
    """A stable machine-readable audit failure."""

    code: str
    subject: str
    detail: str


@dataclass(frozen=True)
class AuditReport:
    """Deterministic result returned by :func:`audit`."""

    findings: tuple[Finding, ...]
    counts: dict[str, int]
    ownership: dict[str, tuple[str, ...]]
    evidence: dict[str, str]

    @property
    def ok(self) -> bool:
        return not self.findings

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "findings": [asdict(finding) for finding in self.findings],
            "counts": self.counts,
            "ownership": self.ownership,
            "evidence": self.evidence,
        }


def _load(path: Path, findings: list[Finding], subject: str) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        findings.append(Finding("MISSING_PATH", subject, str(path)))
    except (OSError, UnicodeError, json.JSONDecodeError):
        findings.append(Finding("MALFORMED_JSON", subject, str(path)))
    return None


def _records(
    payload: Any, subject: str, findings: list[Finding]
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(payload, list):
        findings.append(Finding("INVALID_REGISTRY", subject, "expected a JSON list"))
        return {}
    records: dict[str, Mapping[str, Any]] = {}
    for item in payload:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            findings.append(
                Finding("INVALID_RECORD", subject, "record has no string id")
            )
            continue
        identifier = item["id"]
        if identifier in records:
            findings.append(Finding("DUPLICATE_ID", identifier, subject))
        else:
            records[identifier] = item
    return records


def _string_list(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return tuple(value)


def _safe_module(root: Path, module: str) -> bool:
    relative = Path(*module.split("."))
    if module.startswith("anyalgebra."):
        source = root / "src"
        return (source / relative).with_suffix(".py").is_file() or (
            source / relative / "__init__.py"
        ).is_file()
    return (
        module.startswith("tools.") and (root / relative).with_suffix(".py").is_file()
    )


def _check_api_targets(
    root: Path,
    api_records: Mapping[str, Mapping[str, Any]],
    scope_ids: tuple[str, ...],
    findings: list[Finding],
    ownership: dict[str, tuple[str, ...]],
) -> None:
    quickstart = root / QUICKSTART_PATH
    if not quickstart.is_file():
        findings.append(Finding("MISSING_PATH", "quickstart", str(quickstart)))
        return
    try:
        quickstart_text = quickstart.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        findings.append(
            Finding("API_DOCUMENTATION_UNREADABLE", "quickstart", type(error).__name__)
        )
        return
    rows = {
        match.group("id"): match.groupdict()
        for match in QUICKSTART_ROW.finditer(quickstart_text)
    }
    reviewed_scope = set(scope_ids) == set(API_SPECS)
    source = str((root / "src").resolve())
    root_text = str(root.resolve())
    inserted = source not in sys.path
    inserted_root = root_text not in sys.path
    if inserted:
        sys.path.insert(0, source)
    if inserted_root:
        sys.path.insert(0, root_text)
    importlib.invalidate_caches()

    def is_reviewed_namespace(name: str) -> bool:
        return (
            name == "anyalgebra"
            or name.startswith("anyalgebra.")
            or name == "tools"
            or name.startswith("tools.")
        )

    # The reviewed runtime-capture seam lives in ``tools`` rather than the
    # installable package.  Isolate its parent namespace too: otherwise an
    # already imported repository copy can mask a tampered temporary fixture.
    prior_modules = {
        name: loaded
        for name, loaded in sys.modules.items()
        if is_reviewed_namespace(name)
    }
    prior_module_dicts = {
        name: dict(loaded.__dict__) for name, loaded in prior_modules.items()
    }
    for name in tuple(sys.modules):
        if is_reviewed_namespace(name):
            del sys.modules[name]
    try:
        for identifier in scope_ids:
            row = rows.get(identifier)
            contract = api_records.get(identifier)
            if row is None:
                findings.append(
                    Finding(
                        "API_TARGET_MISSING", identifier, "no quickstart import row"
                    )
                )
                continue
            if contract is None:
                continue
            specs = API_SPECS.get(identifier)
            if specs is None and reviewed_scope:
                findings.append(
                    Finding("API_OWNER_MISSING", identifier, "reviewed map")
                )
                continue
            unreviewed = specs is None
            if unreviewed:
                specs = (CallableSpec(row["module"], row["symbol"], None, _ps()),)
            assert specs is not None
            if (
                identifier in REGISTRY_SIGNATURES
                and contract.get("signature") != REGISTRY_SIGNATURES[identifier]
            ):
                findings.append(
                    Finding(
                        "API_CONTRACT_SIGNATURE_MISMATCH",
                        identifier,
                        "registry signature differs from executable contract",
                    )
                )
                findings.append(
                    Finding(
                        "API_CONTRACT_CALL_MISMATCH",
                        identifier,
                        "registry callable family differs from executable contract",
                    )
                )
            expected_published = PUBLISHED_CALLS.get(identifier)
            if expected_published is not None and row["call"] != expected_published:
                findings.append(
                    Finding("API_PUBLISHED_CALL_MISMATCH", identifier, row["call"])
                )
            documented_target = next(
                (
                    spec
                    for spec in specs
                    if (spec.module, spec.symbol) == (row["module"], row["symbol"])
                ),
                None,
            )
            if documented_target is None:
                findings.append(
                    Finding("API_DOCUMENTATION_OWNER_MISMATCH", identifier, row["call"])
                )
            resolved: list[str] = []
            for spec in specs:
                if not _safe_module(root, spec.module):
                    findings.append(
                        Finding("UNSAFE_API_MODULE", identifier, spec.module)
                    )
                    continue
                try:
                    imported = getattr(
                        importlib.import_module(spec.module), spec.symbol
                    )
                    target = getattr(imported, spec.member) if spec.member else imported
                    signature = inspect.signature(target)
                except (
                    AttributeError,
                    ImportError,
                    TypeError,
                    ValueError,
                    OSError,
                    UnicodeError,
                    Exception,
                ) as error:
                    findings.append(
                        Finding(
                            "API_TARGET_UNRESOLVED",
                            identifier,
                            f"{spec.display_name}:{type(error).__name__}",
                        )
                    )
                    continue
                if not callable(target):
                    findings.append(
                        Finding(
                            "API_TARGET_NOT_CALLABLE", identifier, spec.display_name
                        )
                    )
                    continue
                actual = tuple(
                    (
                        parameter.name,
                        parameter.kind,
                        NO_DEFAULT
                        if parameter.default is inspect.Parameter.empty
                        else parameter.default,
                    )
                    for parameter in signature.parameters.values()
                )
                if not unreviewed and actual != spec.parameters:
                    findings.append(
                        Finding(
                            "API_SIGNATURE_MISMATCH",
                            identifier,
                            f"{spec.display_name}: expected {spec.parameters!r}; found {actual!r}",
                        )
                    )
                resolved.append(f"{spec.module}:{spec.display_name}")
            if documented_target is not None and row["call"] != PUBLISHED_CALLS.get(
                identifier, row["call"]
            ):
                findings.append(
                    Finding("API_PUBLISHED_CALL_MISMATCH", identifier, row["call"])
                )
            ownership[identifier] = (*resolved, row["call"])
    finally:
        # A temporary-fixture audit must not replace the application's imported
        # package graph.  Restore modules *and* parent-package attributes
        # exactly; importing a child otherwise leaves a stale attribute on its
        # parent after sys.modules has been repaired.
        for name in tuple(sys.modules):
            if is_reviewed_namespace(name) and name not in prior_modules:
                del sys.modules[name]
        for name, loaded in prior_modules.items():
            sys.modules[name] = loaded
        for name, restored_values in prior_module_dicts.items():
            module_dict = prior_modules[name].__dict__
            module_dict.clear()
            module_dict.update(restored_values)
        if inserted:
            sys.path.remove(source)
        if inserted_root:
            sys.path.remove(root_text)


def collect_pytest_nodes(root: Path) -> tuple[set[str], Finding | None]:
    """Collect real pytest node IDs without executing arbitrary registry paths."""
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return set(), Finding("PYTEST_COLLECTION_TIMEOUT", "pytest", "120 seconds")
    except OSError as error:
        return set(), Finding(
            "PYTEST_COLLECTION_OSERROR", "pytest", type(error).__name__
        )
    if completed.returncode != 0:
        return set(), Finding(
            "PYTEST_COLLECTION_FAILED", "pytest", completed.stderr.strip()
        )
    return {
        line.strip()
        for line in completed.stdout.splitlines()
        if "::" in line and not line.lstrip().startswith("<")
    }, None


def _run_selected_pytest(
    root: Path, node_ids: Sequence[str], subject: str
) -> Finding | None:
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *node_ids],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=240,
        )
    except subprocess.TimeoutExpired:
        return Finding("FOCUSED_TEST_SELECTION_TIMEOUT", subject, "240 seconds")
    except OSError as error:
        return Finding("FOCUSED_TEST_SELECTION_OSERROR", subject, type(error).__name__)
    if completed.returncode != 0:
        return Finding(
            "FOCUSED_TEST_SELECTION_FAILED", subject, completed.stdout.strip()
        )
    return None


def _check_test_nodes(
    root: Path,
    test_records: Mapping[str, Mapping[str, Any]],
    scope_ids: tuple[str, ...],
    findings: list[Finding],
    ownership: dict[str, tuple[str, ...]],
    collected_node_ids: set[str] | None,
    execute_nodes: bool,
) -> None:
    nodes, collection_error = (
        collect_pytest_nodes(root)
        if collected_node_ids is None
        else (collected_node_ids, None)
    )
    if collection_error is not None:
        findings.append(collection_error)
        return
    claimed: dict[str, str] = {}
    selected: list[str] = []
    for identifier in scope_ids:
        record = test_records.get(identifier)
        if record is None:
            continue
        node_ids = (
            TEST_OWNERS.get(identifier)
            if set(scope_ids) == set(TEST_OWNERS)
            else _string_list(record.get("nodeIds"))
        )
        if not node_ids:
            findings.append(
                Finding(
                    "TEST_NODE_OWNERSHIP_MISSING", identifier, "reviewed map/nodeIds"
                )
            )
            continue
        ownership[identifier] = node_ids
        for node_id in node_ids:
            prior = claimed.setdefault(node_id, identifier)
            if prior != identifier:
                findings.append(
                    Finding(
                        "DUPLICATE_TEST_NODE_OWNER", node_id, f"{prior}, {identifier}"
                    )
                )
            if node_id not in nodes:
                findings.append(Finding("TEST_NODE_UNCOLLECTED", identifier, node_id))
            else:
                selected.append(node_id)
    if (
        execute_nodes
        and selected
        and not any(f.code == "TEST_NODE_UNCOLLECTED" for f in findings)
    ):
        finding = _run_selected_pytest(root, sorted(set(selected)), "pytest")
        if finding is not None:
            findings.append(finding)


def _linked_ids(record: Mapping[str, Any], field: str) -> set[str]:
    values = _string_list(record.get(field))
    return set(values or ())


def _check_registry_graph(
    registries: Mapping[str, Mapping[str, Mapping[str, Any]]],
    scope: Mapping[str, Any],
    findings: list[Finding],
    ownership: dict[str, tuple[str, ...]],
) -> None:
    scoped = {field: set(_string_list(scope.get(field)) or ()) for field in REGISTRIES}
    for field, records in registries.items():
        identifiers = _string_list(scope.get(field)) or ()
        for identifier in identifiers:
            if identifier not in records:
                findings.append(Finding("ORPHAN_SCOPE_ID", identifier, field))
            else:
                ownership[identifier] = (
                    f".agents/project-process/{REGISTRIES[field]}",
                )
    link_kinds = {
        "featureIds": "featureIds",
        "actionIds": "actionIds",
        "storyIds": "storyIds",
        "testIds": "testIds",
        "componentIds": "componentIds",
        "stateIds": "stateIds",
        "apiIds": "apiIds",
    }
    for records in registries.values():
        for identifier, record in records.items():
            for link_field, target_kind in link_kinds.items():
                linked = _string_list(record.get(link_field))
                if linked is None and link_field in record:
                    findings.append(
                        Finding("INVALID_LINK_LIST", identifier, link_field)
                    )
                    continue
                for linked_id in linked or ():
                    if linked_id not in scoped[target_kind]:
                        findings.append(
                            Finding(
                                "UNSCOPED_OR_UNKNOWN_LINK",
                                identifier,
                                f"{link_field}:{linked_id}",
                            )
                        )

    def reciprocal(
        source_kind: str, target_kind: str, forward: str, backward: str
    ) -> None:
        for source_id in scoped[source_kind]:
            source = registries[source_kind].get(source_id)
            if source is None:
                continue
            for target_id in _linked_ids(source, forward):
                target = registries[target_kind].get(target_id)
                if target is not None and source_id not in _linked_ids(
                    target, backward
                ):
                    findings.append(
                        Finding(
                            "RECIPROCAL_LINK_MISSING",
                            source_id,
                            f"{forward}:{target_id} lacks {backward}",
                        )
                    )
        # Also inspect reverse declarations.  Checking only the forward list
        # lets deleting a source edge pass while the target still claims it.
        for target_id in scoped[target_kind]:
            target = registries[target_kind].get(target_id)
            if target is None:
                continue
            for source_id in _linked_ids(target, backward):
                source = registries[source_kind].get(source_id)
                if source is not None and target_id not in _linked_ids(source, forward):
                    findings.append(
                        Finding(
                            "RECIPROCAL_LINK_MISSING",
                            target_id,
                            f"{backward}:{source_id} lacks {forward}",
                        )
                    )

    reciprocal("featureIds", "actionIds", "actionIds", "featureIds")
    reciprocal("featureIds", "storyIds", "storyIds", "featureIds")
    reciprocal("featureIds", "testIds", "testIds", "featureIds")
    reciprocal("actionIds", "storyIds", "storyIds", "actionIds")
    reciprocal("actionIds", "testIds", "testIds", "actionIds")
    reciprocal("storyIds", "testIds", "testIds", "storyIds")
    for component_id in scoped["componentIds"]:
        component = registries["componentIds"].get(component_id)
        if component is not None and not _linked_ids(component, "testIds"):
            findings.append(
                Finding("COMPONENT_TEST_LINK_MISSING", component_id, "testIds")
            )
    for state_id in scoped["stateIds"]:
        state = registries["stateIds"].get(state_id)
        if state is not None and not _linked_ids(state, "featureIds"):
            findings.append(
                Finding("STATE_FEATURE_LINK_MISSING", state_id, "featureIds")
            )
    for feature_id in _string_list(scope.get("featureIds")) or ():
        related = {
            kind: [
                record_id
                for record_id, record in records.items()
                if feature_id in _linked_ids(record, "featureIds")
            ]
            for kind, records in registries.items()
            if kind in {"actionIds", "storyIds", "testIds", "apiIds", "componentIds"}
        }
        for kind, record_ids in related.items():
            if not record_ids:
                findings.append(Finding("FEATURE_LINK_MISSING", feature_id, kind))


def _nodes_for_tests(test_ids: Iterable[str]) -> tuple[str, ...]:
    return tuple(node for test_id in test_ids for node in TEST_OWNERS.get(test_id, ()))


def _check_scoped_owners(
    root: Path,
    registries: Mapping[str, Mapping[str, Mapping[str, Any]]],
    scope: Mapping[str, Any],
    findings: list[Finding],
    ownership: dict[str, tuple[str, ...]],
) -> None:
    """Attach concrete API/test/module seams to every non-API scoped record."""
    release_like = bool(
        set(_string_list(scope.get("apiIds")) or ()) & API_IDS
        or set(_string_list(scope.get("testIds")) or ()) & set(TEST_OWNERS)
    )
    if not release_like:
        return
    component_ids = set(_string_list(scope.get("componentIds")) or ())
    state_ids = set(_string_list(scope.get("stateIds")) or ())
    if component_ids != set(COMPONENT_OWNERS):
        findings.append(Finding("COMPONENT_OWNER_SET_MISMATCH", "componentIds", "map"))
    if state_ids != set(STATE_OWNERS):
        findings.append(Finding("STATE_OWNER_SET_MISMATCH", "stateIds", "map"))
    if component_ids != set(COMPONENT_OWNERS) or state_ids != set(STATE_OWNERS):
        return
    for field in ("featureIds", "actionIds", "storyIds"):
        for identifier in _string_list(scope.get(field)) or ():
            record = registries[field].get(identifier)
            if record is None:
                continue
            feature_ids = (
                {identifier}
                if field == "featureIds"
                else _linked_ids(record, "featureIds")
            )
            test_ids = _linked_ids(record, "testIds")
            api_ids = tuple(
                api_id
                for api_id, api in registries["apiIds"].items()
                if feature_ids & _linked_ids(api, "featureIds")
            )
            if not test_ids or not api_ids:
                findings.append(
                    Finding("CONCRETE_DOWNSTREAM_MISSING", identifier, "api/test")
                )
            ownership[identifier] = (
                f".agents/project-process/{REGISTRIES[field]}",
                *sorted(api_ids),
                *_nodes_for_tests(sorted(test_ids)),
            )
    for identifier in _string_list(scope.get("componentIds")) or ():
        record = registries["componentIds"].get(identifier)
        modules = COMPONENT_OWNERS.get(identifier)
        if record is None or not modules:
            findings.append(Finding("COMPONENT_OWNER_MISSING", identifier, "map"))
            continue
        if not _linked_ids(record, "testIds"):
            findings.append(Finding("CONCRETE_DOWNSTREAM_MISSING", identifier, "test"))
        for module in modules:
            if not _safe_module(root, module):
                findings.append(
                    Finding("COMPONENT_MODULE_UNRESOLVED", identifier, module)
                )
        ownership[identifier] = (
            f".agents/project-process/{REGISTRIES['componentIds']}",
            *modules,
            *_nodes_for_tests(sorted(_linked_ids(record, "testIds"))),
        )
    source = str((root / "src").resolve())
    root_text = str(root.resolve())
    inserted = source not in sys.path
    inserted_root = root_text not in sys.path
    if inserted:
        sys.path.insert(0, source)
    if inserted_root:
        sys.path.insert(0, root_text)
    importlib.invalidate_caches()

    def is_reviewed_namespace(name: str) -> bool:
        return (
            name == "anyalgebra"
            or name.startswith("anyalgebra.")
            or name == "tools"
            or name.startswith("tools.")
        )

    prior_modules = {
        name: loaded
        for name, loaded in sys.modules.items()
        if is_reviewed_namespace(name)
    }
    prior_module_dicts = {
        name: dict(loaded.__dict__) for name, loaded in prior_modules.items()
    }
    for name in tuple(sys.modules):
        if is_reviewed_namespace(name):
            del sys.modules[name]
    try:
        for identifier in _string_list(scope.get("stateIds")) or ():
            record = registries["stateIds"].get(identifier)
            seam = STATE_OWNERS.get(identifier)
            if record is None or seam is None:
                findings.append(Finding("STATE_OWNER_MISSING", identifier, "map"))
                continue
            module, symbol, state_field = seam
            values = _string_list(record.get("values"))
            if values != STATE_VALUES[identifier]:
                findings.append(Finding("STATE_VALUES_MISMATCH", identifier, "values"))
            try:
                owner_module = importlib.import_module(module)
                target = getattr(owner_module, symbol)
                annotations = getattr(target, "__annotations__", {})
                if state_field is not None and state_field not in annotations:
                    raise AttributeError(state_field)
                if identifier == "state.evaluation.outcome":
                    expected = {
                        getattr(owner_module, name)
                        for name in (
                            "Defined",
                            "Undefined",
                            "Indeterminate",
                            "Failed",
                        )
                    }
                    observed = {
                        get_origin(branch) or branch for branch in get_args(target)
                    }
                    if observed != expected:
                        findings.append(
                            Finding(
                                "STATE_CODE_VALUES_MISMATCH",
                                identifier,
                                "EvaluationOutcome union branches",
                            )
                        )
                elif identifier == "state.validation.outcome":
                    expected = {
                        getattr(owner_module, name)
                        for name in ("Proved", "Disproved", "Inconclusive")
                    }
                    if {item.__name__.lower() for item in expected} != set(
                        STATE_VALUES[identifier]
                    ) or not all(issubclass(item, target) for item in expected):
                        findings.append(
                            Finding(
                                "STATE_CODE_VALUES_MISMATCH",
                                identifier,
                                "ValidationReport outcome subclasses",
                            )
                        )
                elif identifier == "state.evidence.outcome":
                    code_values = getattr(owner_module, "_OUTCOMES", None)
                    if not isinstance(code_values, frozenset) or set(
                        code_values
                    ) != set(STATE_VALUES[identifier]):
                        findings.append(
                            Finding(
                                "STATE_CODE_VALUES_MISMATCH",
                                identifier,
                                "ResultReceipt mathematical-outcome allow-list",
                            )
                        )
                elif identifier == "state.evidence.execution":
                    assert state_field is not None
                    lifecycle = owner_module.ExecutionStatus
                    terminal = owner_module.TerminalExecutionStatus
                    lifecycle_values = {item.value for item in lifecycle}
                    terminal_values = set(get_args(terminal))
                    receipt_annotation = get_type_hints(target).get(state_field)
                    if (
                        lifecycle_values != set(STATE_VALUES[identifier])
                        or terminal_values != {"completed", "failed"}
                        or receipt_annotation != terminal
                    ):
                        findings.append(
                            Finding(
                                "STATE_CODE_VALUES_MISMATCH",
                                identifier,
                                "ExecutionStatus lifecycle or terminal receipt annotation",
                            )
                        )
            except (
                AttributeError,
                ImportError,
                OSError,
                UnicodeError,
                Exception,
            ) as error:
                findings.append(
                    Finding("STATE_SEAM_UNRESOLVED", identifier, type(error).__name__)
                )
            state_test_ids = tuple(
                test_id
                for test_id, test in registries["testIds"].items()
                if _linked_ids(record, "featureIds") & _linked_ids(test, "featureIds")
            )
            if not state_test_ids:
                findings.append(
                    Finding("CONCRETE_DOWNSTREAM_MISSING", identifier, "test")
                )
            ownership[identifier] = (
                f".agents/project-process/{REGISTRIES['stateIds']}",
                f"{module}:{symbol}" + (f".{state_field}" if state_field else ""),
                *_nodes_for_tests(state_test_ids),
            )
    finally:
        for name in tuple(sys.modules):
            if is_reviewed_namespace(name) and name not in prior_modules:
                del sys.modules[name]
        for name, loaded in prior_modules.items():
            sys.modules[name] = loaded
        for name, restored_values in prior_module_dicts.items():
            module_dict = prior_modules[name].__dict__
            module_dict.clear()
            module_dict.update(restored_values)
        if inserted:
            sys.path.remove(source)
        if inserted_root:
            sys.path.remove(root_text)


def _check_adrs(
    root: Path,
    scope: Mapping[str, Any],
    findings: list[Finding],
    ownership: dict[str, tuple[str, ...]],
) -> None:
    gate = _load(root / ADR_GATE_PATH, findings, "adr-gate")
    entries = gate.get("entries") if isinstance(gate, Mapping) else None
    by_id = (
        {entry.get("adrId"): entry for entry in entries if isinstance(entry, Mapping)}
        if isinstance(entries, list)
        else {}
    )
    for identifier in _string_list(scope.get("decisionIds")) or ():
        entry = by_id.get(identifier)
        if not isinstance(entry, Mapping):
            findings.append(Finding("ADR_GATE_ENTRY_MISSING", identifier, "adr-gate"))
            continue
        path = entry.get("path")
        decision = entry.get("gateDecision")
        if not isinstance(path, str) or not (root / path).is_file():
            findings.append(Finding("ADR_PATH_MISSING", identifier, str(path)))
        elif decision not in {"accepted", "blocking"}:
            findings.append(
                Finding("ADR_STATUS_UNSUPPORTED", identifier, str(decision))
            )
        else:
            ownership[identifier] = (path, decision)


def _check_release_documents_and_examples(
    root: Path,
    findings: list[Finding],
    ownership: dict[str, tuple[str, ...]],
    collected_node_ids: set[str] | None,
    execute_nodes: bool,
) -> None:
    """Prove release documents exist and examples are executed by direct tests."""
    for path in REQUIRED_DOCUMENTS:
        document = root / path
        if not document.is_file():
            findings.append(
                Finding("RELEASE_DOCUMENT_MISSING", path, "required scope document")
            )
            continue
        try:
            text = document.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            findings.append(Finding("RELEASE_DOCUMENT_UNREADABLE", path, "utf-8"))
            continue
        if not text.strip() or any(
            token not in text for token in REQUIRED_DOCUMENT_TOKENS[path]
        ):
            findings.append(
                Finding("RELEASE_DOCUMENT_CONTENT_MISMATCH", path, "tokens")
            )
    nodes, collection_error = (
        collect_pytest_nodes(root)
        if collected_node_ids is None
        else (collected_node_ids, None)
    )
    if collection_error is not None:
        findings.append(collection_error)
        return
    selected: list[str] = []
    for example, example_nodes in RELEASE_EXAMPLE_OWNERS.items():
        if not (root / example).is_file():
            findings.append(
                Finding("RELEASE_EXAMPLE_MISSING", example, "release example")
            )
            continue
        ownership[example] = example_nodes
        for node_id in example_nodes:
            if node_id not in nodes:
                findings.append(
                    Finding("RELEASE_EXAMPLE_TEST_UNCOLLECTED", example, node_id)
                )
            else:
                selected.append(node_id)
    if (
        execute_nodes
        and selected
        and not any(
            finding.code == "RELEASE_EXAMPLE_TEST_UNCOLLECTED" for finding in findings
        )
    ):
        finding = _run_selected_pytest(root, selected, "release-examples")
        if finding is not None:
            findings.append(finding)


def _accepted_evidence(
    root: Path, findings: list[Finding], *, strict: bool
) -> dict[str, str]:
    progress = _load(root / PROGRESS_PATH, findings, "implementation-progress")
    evidence: dict[str, str] = {}
    accepted = progress.get("accepted") if isinstance(progress, Mapping) else None
    entries = (
        {entry.get("taskId"): entry for entry in accepted if isinstance(entry, Mapping)}
        if isinstance(accepted, list)
        else {}
    )
    for number in range(91, 99):
        task_id = f"V00-{number:03d}"
        entry = entries.get(task_id)
        if not isinstance(entry, Mapping):
            findings.append(
                Finding("ACCEPTED_EVIDENCE_MISSING", task_id, "implementation-progress")
            )
            continue
        if not strict:
            continue
        deliverables = _string_list(entry.get("deliverables"))
        verification = entry.get("verification")
        if not deliverables:
            findings.append(
                Finding("EVIDENCE_DELIVERABLES_MISSING", task_id, "deliverables")
            )
            continue
        assert deliverables is not None
        if not isinstance(verification, Mapping):
            findings.append(
                Finding("EVIDENCE_VERIFICATION_MISSING", task_id, "verification")
            )
            continue
        for field in ("stableFiles", "independentAudit"):
            value = verification.get(field)
            if not isinstance(value, str) or not value.strip():
                findings.append(Finding("EVIDENCE_FIELD_MISSING", task_id, field))
        independent_audit = verification.get("independentAudit")
        if isinstance(independent_audit, str):
            review_text = independent_audit.lower()
            if (
                "sol" not in review_text
                or "accept" not in review_text
                or not (
                    "gate" in review_text
                    or "hash" in review_text
                    or "full regression" in review_text
                )
            ):
                findings.append(
                    Finding(
                        "EVIDENCE_INDEPENDENT_AUDIT_CONTENT_MISMATCH",
                        task_id,
                        "requires named Sol review, ACCEPT, and gate/hash evidence",
                    )
                )
        focused_tests = verification.get("focusedPytest")
        if not isinstance(focused_tests, str) or not (
            "test" in focused_tests.lower() and "pass" in focused_tests.lower()
        ):
            findings.append(
                Finding(
                    "EVIDENCE_TEST_CONTENT_MISMATCH",
                    task_id,
                    "focusedPytest must name passing tests",
                )
            )
        stable_files = verification.get("stableFiles")
        if not isinstance(stable_files, str):
            continue
        for deliverable in deliverables:
            path = root / deliverable
            name = Path(deliverable).name
            match = re.search(
                rf"{re.escape(name)}\s+SHA-256\s+([0-9A-Fa-f]{{64}})",
                stable_files,
            )
            if match is None:
                findings.append(
                    Finding("EVIDENCE_STABLE_FILE_HASH_MISSING", task_id, name)
                )
                continue
            try:
                observed = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                findings.append(
                    Finding("EVIDENCE_DELIVERABLE_MISSING", task_id, deliverable)
                )
                continue
            if observed.lower() != match.group(1).lower():
                findings.append(
                    Finding(
                        "EVIDENCE_STABLE_FILE_HASH_MISMATCH",
                        task_id,
                        deliverable,
                    )
                )
    log = root / SKILL_LOG_PATH
    if log.is_file():
        try:
            log_text = log.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            findings.append(Finding("SKILL_LOG_UNREADABLE", "skill-run-log", "utf-8"))
            return evidence
        if strict:
            for number in range(91, 99):
                task_id = f"V00-{number:03d}"
                match = re.search(
                    rf"^## {re.escape(task_id)}[^\n]*\n(?P<body>.*?)(?=^## |\Z)",
                    log_text,
                    flags=re.MULTILINE | re.DOTALL,
                )
                if match is None:
                    findings.append(
                        Finding("SKILL_LOG_SECTION_MISSING", task_id, "skill-run-log")
                    )
                    continue
                body = match.group("body").lower()
                if (
                    len(body.strip()) < 160
                    or "accept" not in body
                    or not ("test" in body or "pytest" in body)
                    or not ("review" in body or "audit" in body)
                ):
                    findings.append(
                        Finding(
                            "SKILL_LOG_SECTION_CONTENT_MISMATCH",
                            task_id,
                            "requires substantive test, review/audit, and ACCEPT evidence",
                        )
                    )
        evidence["v91_v98_log_sha256"] = hashlib.sha256(log.read_bytes()).hexdigest()
    else:
        findings.append(Finding("MISSING_PATH", "skill-run-log", str(log)))
    return evidence


def audit(
    repository_root: Path | str | None = None,
    *,
    collected_node_ids: set[str] | None = None,
    execute_nodes: bool = True,
) -> AuditReport:
    """Return a deterministic, read-only v0.0 traceability report."""
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    findings: list[Finding] = []
    ownership: dict[str, tuple[str, ...]] = {}
    scope_payload = _load(root / SCOPE_PATH, findings, "scope")
    snapshot = _load(root / SNAPSHOT_PATH, findings, "contract-snapshot")
    if not isinstance(scope_payload, Mapping):
        return AuditReport(tuple(sorted(findings)), {}, {}, {})
    scope = scope_payload
    reviewed_ids = {
        **REVIEWED_SCOPE_IDS,
        "apiIds": API_IDS,
        "testIds": frozenset(TEST_OWNERS),
    }
    # The frozen v0.0 cardinalities distinguish the release registry from
    # small, deliberately unrelated fixture repositories.  Once a scope
    # claims the complete release cardinalities, every reviewed set is exact;
    # an all-ID substitution must not bypass this check merely by overlap=0.
    expected_api_count = EXPECTED_COUNTS["apiIds"]
    expected_test_count = EXPECTED_COUNTS["testIds"]
    is_full_review = (
        len(_string_list(scope.get("apiIds")) or ()) == expected_api_count
        and len(_string_list(scope.get("testIds")) or ()) == expected_test_count
    )
    if is_full_review:
        for field, reviewed_set in reviewed_ids.items():
            actual = frozenset(_string_list(scope.get(field)) or ())
            if actual != reviewed_set:
                findings.append(
                    Finding(
                        "REVIEWED_SCOPE_SET_MISMATCH",
                        field,
                        f"missing={sorted(reviewed_set - actual)!r}; "
                        f"extra={sorted(actual - reviewed_set)!r}",
                    )
                )
    counts = {field: len(_string_list(scope.get(field)) or ()) for field in REGISTRIES}
    expected_counts = dict(EXPECTED_COUNTS)
    if isinstance(snapshot, Mapping) and isinstance(snapshot.get("counts"), Mapping):
        expected_counts.update(
            {
                key: value
                for key, value in snapshot["counts"].items()
                if isinstance(value, int)
            }
        )
    for field, expected in expected_counts.items():
        if counts.get(field) != expected:
            findings.append(
                Finding(
                    "SCOPED_COUNT_MISMATCH",
                    field,
                    f"expected {expected}, found {counts.get(field, 0)}",
                )
            )
    registries = {
        field: _records(
            _load(root / PROCESS / filename, findings, filename), filename, findings
        )
        for field, filename in REGISTRIES.items()
    }
    _check_registry_graph(registries, scope, findings, ownership)
    api_ids = _string_list(scope.get("apiIds")) or ()
    test_ids = _string_list(scope.get("testIds")) or ()
    _check_api_targets(root, registries["apiIds"], api_ids, findings, ownership)
    _check_test_nodes(
        root,
        registries["testIds"],
        test_ids,
        findings,
        ownership,
        collected_node_ids,
        execute_nodes,
    )
    _check_adrs(root, scope, findings, ownership)
    _check_scoped_owners(root, registries, scope, findings, ownership)
    if set(api_ids) == set(API_IDS) and set(test_ids) == set(TEST_OWNERS):
        _check_release_documents_and_examples(
            root, findings, ownership, collected_node_ids, execute_nodes
        )
    evidence = _accepted_evidence(
        root,
        findings,
        strict=set(api_ids) == set(API_IDS) and set(test_ids) == set(TEST_OWNERS),
    )
    return AuditReport(
        tuple(sorted(set(findings))), counts, dict(sorted(ownership.items())), evidence
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repository_root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="collect node IDs but do not run owned nodes",
    )
    arguments = parser.parse_args(argv)
    report = audit(arguments.repository_root, execute_nodes=not arguments.no_execute)
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
