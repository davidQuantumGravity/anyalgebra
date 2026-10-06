"""Stable validation-report evidence contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from itertools import repeat
from typing import Any, cast

import pytest

from anyalgebra.validation.reports import (
    ActiveBound,
    DeterministicWitness,
    Disproved,
    HypothesisEvidence,
    Inconclusive,
    Proved,
    ValidationBounds,
    ValidationCounts,
    ValidationReport,
    ValidationReportDefinitionError,
)


def _counts(
    *,
    expected: int = 4,
    evaluated: int = 4,
    hypotheses: int = 0,
    rejected: int = 0,
    conclusions: int = 4,
    compared: int | None = None,
    vacuous: int = 0,
    skipped: int = 0,
    undecidable: int = 0,
    undefined_left: int = 0,
    undefined_right: int = 0,
    undefined_both: int = 0,
    false: int = 0,
    failed: int = 0,
) -> ValidationCounts:
    if compared is None:
        compared = conclusions
    return ValidationCounts(
        expected,
        evaluated,
        hypotheses_evaluated=hypotheses,
        hypothesis_rejected=rejected,
        conclusions_evaluated=conclusions,
        compared=compared,
        vacuous_assignments=vacuous,
        skipped_assignments=skipped,
        undecidable_assignments=undecidable,
        undefined_left=undefined_left,
        undefined_right=undefined_right,
        undefined_both=undefined_both,
        conclusion_false=false,
        evaluation_failed=failed,
    )


def _bounds(
    route: str = "exhaustive_finite",
    *,
    complete: bool = True,
    assignment_limit: int = 4,
) -> ValidationBounds:
    return ValidationBounds(
        route,
        complete,
        assignment_limit,
        domain_cardinalities=(2, 2),
        active_bounds=(ActiveBound("term_nodes", 512),),
    )


def _counterexample(
    index: int = 2,
    *,
    subtype: str = "defined_unequal",
    evaluated_indices: tuple[int, ...] = (0, 1, 2),
    declared_order: tuple[int, ...] | None = None,
) -> DeterministicWitness:
    if declared_order is None:
        declared_order = evaluated_indices + tuple(
            candidate for candidate in range(4) if candidate not in evaluated_indices
        )
    return DeterministicWitness(
        "counterexample",
        "lexicographic",
        "anyalgebra.lexicographic",
        1,
        "least_under_declared_order",
        index,
        (index // 2, index % 2),
        "conclusion",
        counterexample_subtype=subtype,
        order_size=4,
        declared_order=declared_order,
        evaluated_indices=evaluated_indices,
        earlier_resolved=True,
    )


def _gap(index: int = 1, phase: str = "premise") -> DeterministicWitness:
    evaluated_indices = (0, index)
    declared_order = evaluated_indices + tuple(
        candidate for candidate in range(4) if candidate not in evaluated_indices
    )
    return DeterministicWitness(
        "evaluation_gap",
        "lexicographic",
        "anyalgebra.lexicographic",
        1,
        "not_applicable",
        index,
        (0, 1),
        phase,
        0 if phase.startswith("premise") else None,
        order_size=4,
        declared_order=declared_order,
        evaluated_indices=evaluated_indices,
        earlier_resolved=False,
    )


def _verified(
    name: str = "multilinearity", *, kind: str = "multilinearity"
) -> HypothesisEvidence:
    return HypothesisEvidence(
        name,
        "verified",
        kind=kind,
        evidence_id=f"proof:{name}",
        algorithm="anyalgebra.exhaustive_finite_law",
        algorithm_version=1,
    )


def test_positive_exhaustive_and_checked_theorem_proofs() -> None:
    declared = HypothesisEvidence("x_is_defined")
    exhaustive = Proved(
        (declared,),
        _counts(hypotheses=4),
        _bounds(),
        "anyalgebra.exhaustive_finite_law",
        1,
        proof_route="exhaustive_finite",
    )
    assert exhaustive.status == "proved"
    assert exhaustive.proof_route == exhaustive.bounds.route == "exhaustive_finite"
    assert exhaustive.hypotheses[0] is not declared
    assert exhaustive.hypotheses[0].name == declared.name
    theorem_bounds = ValidationBounds(
        "checked_theorem_reduction",
        True,
        1,
        active_bounds=(ActiveBound("obligations", 2),),
    )
    theorem = Proved(
        (
            _verified("theorem", kind="theorem"),
            _verified("law", kind="law_identity"),
            _verified("closure", kind="closure"),
            _verified("scalar", kind="scalar_domain"),
            _verified(),
        ),
        ValidationCounts(0, 0),
        theorem_bounds,
        "anyalgebra.checked_basis_reduction",
        1,
        proof_route="checked_theorem_reduction",
    )
    assert theorem.status == "proved"
    assert theorem.proof_route == "checked_theorem_reduction"


def test_deterministic_disproof_and_bounded_inconclusive_reports() -> None:
    disproof = Disproved(
        (),
        _counts(evaluated=3, conclusions=3, compared=3, false=1),
        _bounds(),
        "anyalgebra.witness_minimization",
        1,
        _counterexample(),
    )
    assert disproof.status == "disproved"
    assert disproof.witness is not None
    assert disproof.witness.minimality == "least_under_declared_order"
    sampled = Inconclusive(
        (),
        _counts(expected=4, evaluated=2, conclusions=2, compared=2),
        _bounds("sampled", complete=False),
        "anyalgebra.splitmix64_partial_fisher_yates",
        1,
        reasons=("sampled_pass_not_proof",),
    )
    assert sampled.status == "inconclusive"
    assert sampled.reasons == ("sampled_pass_not_proof",)
    gap = Inconclusive(
        (
            HypothesisEvidence(
                "closure",
                "unresolved",
                evidence_id="gap:1",
                algorithm="a",
                algorithm_version=1,
            ),
        ),
        _counts(
            undecidable=1,
            conclusions=3,
            compared=3,
            hypotheses=4,
        ),
        _bounds(),
        "anyalgebra.exhaustive_finite_law",
        1,
        reasons=("earlier_evaluation_gap",),
        witness=_gap(),
    )
    assert gap.witness is not None and gap.witness.kind == "evaluation_gap"


def test_count_coherence_and_status_soundness_rejections() -> None:
    bad_counts: tuple[tuple[dict[str, int], str], ...] = (
        ({"expected": 1, "evaluated": 2}, "assignments_evaluated"),
        (
            {"expected": 65_537, "evaluated": 0, "conclusions": 0, "compared": 0},
            "maximum",
        ),
        ({"vacuous": 5}, "vacuous"),
        ({"vacuous": 1, "conclusions": 4}, "conclusions"),
        ({"rejected": 1}, "hypothesis_rejected"),
        ({"conclusions": 1, "compared": 2}, "compared"),
        ({"conclusions": 1, "skipped": 2}, "skipped"),
        ({"conclusions": 1, "false": 2}, "conclusion_false"),
        (
            {"conclusions": 2, "compared": 2, "skipped": 1},
            "conclusion_outcomes",
        ),
        ({"conclusions": 1, "undefined_left": 1, "undefined_right": 1}, "undefined"),
        ({"conclusions": 1, "compared": 0, "undefined_left": 1}, "undefined"),
        (
            {"conclusions": 1, "compared": 0, "skipped": 1, "false": 1},
            "conclusion_false",
        ),
        ({"undecidable": 0, "failed": 1}, "evaluation_failed"),
        (
            {
                "expected": 1,
                "evaluated": 0,
                "conclusions": 0,
                "compared": 0,
                "hypotheses": 1,
            },
            "nonzero",
        ),
    )
    for kwargs, match in bad_counts:
        del match
        with pytest.raises(ValidationReportDefinitionError):
            _counts(**kwargs)
    with pytest.raises(ValidationReportDefinitionError, match="exhaust"):
        Proved(
            (),
            _counts(evaluated=3, conclusions=3, compared=3),
            _bounds(),
            "a",
            1,
            proof_route="exhaustive_finite",
        )
    for counts in (
        _counts(undecidable=1, conclusions=3, compared=3),
        _counts(undecidable=1, conclusions=3, compared=3, failed=1),
        _counts(false=1),
    ):
        with pytest.raises(
            ValidationReportDefinitionError, match=r"false|failed|unresolved"
        ):
            Proved(
                (),
                counts,
                _bounds(),
                "a",
                1,
                proof_route="exhaustive_finite",
            )
    with pytest.raises(ValidationReportDefinitionError, match="complete proof route"):
        Proved((), _counts(), _bounds(), "a", 1, proof_route="sampled")
    with pytest.raises(ValidationReportDefinitionError, match="complete proof route"):
        Proved(
            (),
            _counts(),
            _bounds("sampled", complete=False),
            "a",
            1,
            proof_route="exhaustive_finite",
        )
    with pytest.raises(ValidationReportDefinitionError, match="mixed-radix"):
        Proved(
            (),
            ValidationCounts(0, 0),
            ValidationBounds("exhaustive_finite", True, 1),
            "a",
            1,
            proof_route="exhaustive_finite",
        )
    for hypotheses in (
        (),
        (HypothesisEvidence("declared"),),
        (
            HypothesisEvidence(
                "x", "unresolved", evidence_id="e", algorithm="a", algorithm_version=1
            ),
        ),
    ):
        with pytest.raises(
            ValidationReportDefinitionError, match=r"verified|failed|unresolved"
        ):
            Proved(
                hypotheses,
                ValidationCounts(0, 0),
                ValidationBounds("checked_theorem_reduction", True, 1),
                "a",
                1,
                proof_route="checked_theorem_reduction",
            )
    with pytest.raises(ValidationReportDefinitionError, match="false coverage"):
        Inconclusive((), _counts(false=1), _bounds(), "a", 1, reasons=("gap",))
    with pytest.raises(ValidationReportDefinitionError, match="clean complete"):
        Inconclusive((), _counts(), _bounds(), "a", 1, reasons=("not_a_gap",))


def test_witness_kind_phase_minimality_and_scope_rejections() -> None:
    cases: tuple[tuple[tuple[object, ...], str], ...] = (
        (("other", "o", "a", 1, "not_applicable", 0, (), "conclusion"), "kind"),
        (
            ("counterexample", "o", "a", 1, "not_applicable", 0, (), "conclusion"),
            "minimality",
        ),
        (
            (
                "counterexample",
                "o",
                "a",
                1,
                "least_under_declared_order",
                0,
                (),
                "premise",
                0,
            ),
            "phase",
        ),
        (
            (
                "evaluation_gap",
                "o",
                "a",
                1,
                "least_under_declared_order",
                0,
                (),
                "conclusion",
            ),
            "minimality",
        ),
        (("evaluation_gap", "o", "a", 1, "not_applicable", 0, (), "unknown"), "phase"),
        (
            ("evaluation_gap", "o", "a", 1, "not_applicable", 0, (), "premise"),
            "premise_index",
        ),
        (
            ("evaluation_gap", "o", "a", 1, "not_applicable", 0, (), "conclusion", 0),
            "premise_index",
        ),
    )
    for args, match in cases:
        with pytest.raises(ValidationReportDefinitionError, match=match):
            DeterministicWitness(
                *cast(Any, args),
                counterexample_subtype=(
                    "defined_unequal" if args[0] == "counterexample" else None
                ),
                order_size=4,
                declared_order=(0, 1, 2, 3),
                evaluated_indices=(0,),
                earlier_resolved=args[0] == "counterexample",
            )
    with pytest.raises(ValidationReportDefinitionError, match="mixed-radix"):
        Disproved(
            (),
            _counts(expected=2, evaluated=1, conclusions=1, compared=1, false=1),
            _bounds(),
            "a",
            1,
            _counterexample(2),
        )
    with pytest.raises(ValidationReportDefinitionError, match="resolved least"):
        Disproved(
            (),
            _counts(evaluated=3, conclusions=3, compared=3),
            _bounds(),
            "a",
            1,
            _counterexample(),
        )
    with pytest.raises(ValidationReportDefinitionError, match="evaluation gap"):
        Inconclusive(
            (),
            _counts(expected=4, evaluated=3, conclusions=3, compared=3),
            _bounds("bounded_search", complete=False),
            "a",
            1,
            reasons=("bounded",),
            witness=_counterexample(),
        )


def test_hypothesis_bound_and_report_definition_rejections() -> None:
    with pytest.raises(ValidationReportDefinitionError, match="evidence"):
        HypothesisEvidence("x", evidence_id="forged")
    for status in ("verified", "failed", "unresolved"):
        with pytest.raises(ValidationReportDefinitionError, match="evidence_id"):
            HypothesisEvidence("x", status)
    with pytest.raises(ValidationReportDefinitionError, match="status"):
        HypothesisEvidence("x", cast(Any, object()))
    with pytest.raises(ValidationReportDefinitionError, match="inclusive"):
        ActiveBound("n", 1, inclusive=cast(Any, 1))
    with pytest.raises(ValidationReportDefinitionError, match="route"):
        ValidationBounds("unknown", False, 1)
    with pytest.raises(ValidationReportDefinitionError, match="cannot claim"):
        ValidationBounds("sampled", True, 1)
    with pytest.raises(ValidationReportDefinitionError, match="maximum"):
        ValidationBounds("exhaustive_finite", True, 65_537)
    with pytest.raises(ValidationReportDefinitionError, match="product"):
        ValidationBounds("exhaustive_finite", True, 3, domain_cardinalities=(2, 2))
    duplicate = ActiveBound("n", 1)
    with pytest.raises(ValidationReportDefinitionError, match="duplicate"):
        ValidationBounds(
            "exhaustive_finite", True, 1, active_bounds=(duplicate, duplicate)
        )
    with pytest.raises(ValidationReportDefinitionError, match="ActiveBound"):
        ValidationBounds(
            "exhaustive_finite", True, 1, active_bounds=(cast(Any, object()),)
        )
    with pytest.raises(ValidationReportDefinitionError, match="duplicate"):
        Proved(
            (HypothesisEvidence("x"), HypothesisEvidence("x")),
            _counts(),
            _bounds(),
            "a",
            1,
            proof_route="exhaustive_finite",
        )
    with pytest.raises(ValidationReportDefinitionError, match="HypothesisEvidence"):
        Proved(
            (cast(Any, object()),),
            _counts(),
            _bounds(),
            "a",
            1,
            proof_route="exhaustive_finite",
        )
    with pytest.raises(ValidationReportDefinitionError, match="concrete"):
        ValidationReport()


def test_bounded_one_pass_inputs_and_exact_limits() -> None:
    class Counter:
        def __init__(self, value: object) -> None:
            self.value = value
            self.pulls = 0

        def __iter__(self) -> Counter:
            return self

        def __next__(self) -> object:
            self.pulls += 1
            return self.value

    class UniqueHypotheses:
        def __init__(self) -> None:
            self.pulls = 0

        def __iter__(self) -> UniqueHypotheses:
            return self

        def __next__(self) -> HypothesisEvidence:
            value = HypothesisEvidence(f"h{self.pulls}")
            self.pulls += 1
            return value

    hypotheses = UniqueHypotheses()
    with pytest.raises(ValidationReportDefinitionError, match="maximum 512"):
        Proved(
            cast(Any, hypotheses),
            _counts(),
            _bounds(),
            "a",
            1,
            proof_route="exhaustive_finite",
        )
    assert hypotheses.pulls == 513
    cardinalities = Counter(1)
    with pytest.raises(ValidationReportDefinitionError, match="maximum 512"):
        ValidationBounds(
            "exhaustive_finite",
            True,
            1,
            domain_cardinalities=cast(Any, cardinalities),
        )
    assert cardinalities.pulls == 513
    with pytest.raises(ValidationReportDefinitionError, match="maximum 65536"):
        DeterministicWitness(
            "evaluation_gap",
            "o",
            "a",
            1,
            "not_applicable",
            0,
            repeat(0, 65_537),
            "conclusion",
            order_size=1,
            declared_order=(0,),
            evaluated_indices=(0,),
            earlier_resolved=False,
        )
    exact = DeterministicWitness(
        "evaluation_gap",
        "o",
        "a",
        1,
        "not_applicable",
        0,
        repeat(0, 65_536),
        "conclusion",
        order_size=1,
        declared_order=(0,),
        evaluated_indices=(0,),
        earlier_resolved=False,
    )
    assert len(exact.carrier_indices) == 65_536
    with pytest.raises(ValidationReportDefinitionError, match="component count"):
        Inconclusive(
            (),
            _counts(expected=4, evaluated=1, conclusions=1, compared=0, undecidable=1),
            ValidationBounds(
                "bounded_search",
                False,
                4,
                domain_cardinalities=(2, 2),
                witness_component_limit=1,
            ),
            "a",
            1,
            reasons=("bounded",),
            witness=DeterministicWitness(
                "evaluation_gap",
                "o",
                "a",
                1,
                "not_applicable",
                0,
                (0, 0),
                "conclusion",
                order_size=4,
                declared_order=(0, 1, 2, 3),
                evaluated_indices=(0,),
                earlier_resolved=False,
            ),
        )


def test_hostile_inputs_are_not_rendered_compared_or_executed() -> None:
    class Hostile:
        repr_calls = 0
        eq_calls = 0

        def __repr__(self) -> str:
            type(self).repr_calls += 1
            raise RuntimeError("repr")

        def __eq__(self, other: object) -> bool:
            del other
            type(self).eq_calls += 1
            raise RuntimeError("eq")

    hostile = Hostile()
    with pytest.raises(ValidationReportDefinitionError, match="phase"):
        DeterministicWitness(
            "evaluation_gap",
            "o",
            "a",
            1,
            "not_applicable",
            0,
            (),
            cast(Any, hostile),
            order_size=1,
            declared_order=(0,),
            evaluated_indices=(0,),
            earlier_resolved=False,
        )
    with pytest.raises(ValidationReportDefinitionError, match="status"):
        HypothesisEvidence("x", cast(Any, hostile))
    assert Hostile.repr_calls == Hostile.eq_calls == 0


def test_records_are_sealed_identity_equal_unhashable_and_safe() -> None:
    values: tuple[object, ...] = (
        HypothesisEvidence("x"),
        ActiveBound("n", 1),
        _bounds(),
        _counts(),
        _counterexample(),
        Proved((), _counts(), _bounds(), "a", 1, proof_route="exhaustive_finite"),
        Disproved(
            (),
            _counts(evaluated=3, conclusions=3, compared=3, false=1),
            _bounds(),
            "a",
            1,
            _counterexample(),
        ),
        Inconclusive(
            (),
            _counts(expected=4, evaluated=2, conclusions=2, compared=2),
            _bounds("sampled", complete=False),
            "a",
            1,
            reasons=("sampled_pass_not_proof",),
        ),
    )
    for value in values:
        assert not hasattr(value, "__dict__")
        with pytest.raises(TypeError):
            hash(value)
        with pytest.raises((FrozenInstanceError, TypeError)):
            cast(Any, value).name = "changed"
        assert type(repr(value)) is str
    assert values[-1] is not Inconclusive(
        (),
        _counts(expected=4, evaluated=2, conclusions=2, compared=2),
        _bounds("sampled", complete=False),
        "a",
        1,
        reasons=("sampled_pass_not_proof",),
    )
    assert ValidationReport.schema_tag == "anyalgebra.validation.report"
    assert ValidationReport.schema_version == 1


def test_primitive_declarations_reject_every_malformed_boundary() -> None:
    for value in (-1, True, object()):
        with pytest.raises(ValidationReportDefinitionError, match="non-negative"):
            ActiveBound("n", cast(Any, value))
    for value in (0, -1, True, object()):
        with pytest.raises(ValidationReportDefinitionError, match="positive"):
            HypothesisEvidence(
                "h",
                "verified",
                evidence_id="e",
                algorithm="a",
                algorithm_version=cast(Any, value),
            )
    for value in ("", " padded", "x" * 513, cast(Any, object())):
        with pytest.raises(ValidationReportDefinitionError, match="trimmed"):
            ActiveBound(cast(Any, value), 1)
    for value in ("text", b"bytes"):
        with pytest.raises(ValidationReportDefinitionError, match="not text"):
            ValidationBounds(
                "exhaustive_finite", True, 1, domain_cardinalities=cast(Any, value)
            )

    class BadIterable:
        def __iter__(self) -> Any:
            raise RuntimeError("iter")

    class BadIterator:
        def __iter__(self) -> BadIterator:
            return self

        def __next__(self) -> object:
            raise RuntimeError("next")

    for values in (BadIterable(), BadIterator()):
        with pytest.raises(ValidationReportDefinitionError, match="snapshotted"):
            ValidationBounds(
                "exhaustive_finite",
                True,
                1,
                domain_cardinalities=cast(Any, values),
            )


def test_exact_record_types_and_remaining_bound_edges_are_sealed() -> None:
    class DerivedHypothesis(HypothesisEvidence):
        pass

    class DerivedBound(ActiveBound):
        pass

    class DerivedBounds(ValidationBounds):
        pass

    class DerivedCounts(ValidationCounts):
        pass

    class DerivedWitness(DeterministicWitness):
        pass

    class DerivedProved(Proved):
        pass

    class DerivedDisproved(Disproved):
        pass

    class DerivedInconclusive(Inconclusive):
        pass

    with pytest.raises(ValidationReportDefinitionError, match="exact Hypothesis"):
        DerivedHypothesis("h")
    with pytest.raises(ValidationReportDefinitionError, match="exact ActiveBound"):
        DerivedBound("n", 1)
    with pytest.raises(ValidationReportDefinitionError, match="exact ValidationBounds"):
        DerivedBounds("exhaustive_finite", True, 1)
    with pytest.raises(ValidationReportDefinitionError, match="exact ValidationCounts"):
        DerivedCounts(0, 0)
    with pytest.raises(ValidationReportDefinitionError, match="exact Deterministic"):
        DerivedWitness(
            "evaluation_gap",
            "o",
            "a",
            1,
            "not_applicable",
            0,
            (),
            "conclusion",
            order_size=1,
            declared_order=(0,),
            evaluated_indices=(0,),
            earlier_resolved=False,
        )
    with pytest.raises(ValidationReportDefinitionError, match="exact Proved"):
        DerivedProved((), _counts(), _bounds(), "a", 1, proof_route="exhaustive_finite")
    with pytest.raises(ValidationReportDefinitionError, match="exact Disproved"):
        DerivedDisproved((), _counts(false=1), _bounds(), "a", 1, _counterexample())
    with pytest.raises(ValidationReportDefinitionError, match="exact Inconclusive"):
        DerivedInconclusive(
            (),
            _counts(expected=4, evaluated=2, conclusions=2),
            _bounds("sampled", complete=False),
            "a",
            1,
            reasons=("sampled",),
        )

    with pytest.raises(ValidationReportDefinitionError, match="built-in bool"):
        ValidationBounds("exhaustive_finite", cast(Any, 1), 1)
    with pytest.raises(ValidationReportDefinitionError, match="positive"):
        ValidationBounds("exhaustive_finite", True, 0)
    with pytest.raises(ValidationReportDefinitionError, match="witness_component"):
        ValidationBounds("exhaustive_finite", True, 1, witness_component_limit=65_537)
    with pytest.raises(ValidationReportDefinitionError, match="positive"):
        ValidationBounds("exhaustive_finite", True, 1, domain_cardinalities=(0,))


def test_report_preflight_rejects_foreign_records_and_invalid_witness_scope() -> None:
    with pytest.raises(ValidationReportDefinitionError, match="exact ValidationCounts"):
        Proved(
            (),
            cast(Any, object()),
            _bounds(),
            "a",
            1,
            proof_route="exhaustive_finite",
        )
    with pytest.raises(ValidationReportDefinitionError, match="exact ValidationBounds"):
        Proved(
            (),
            _counts(),
            cast(Any, object()),
            "a",
            1,
            proof_route="exhaustive_finite",
        )
    with pytest.raises(ValidationReportDefinitionError, match="assignment_limit"):
        Inconclusive(
            (),
            _counts(),
            ValidationBounds("bounded_search", False, 3),
            "a",
            1,
            reasons=("bounded",),
        )
    with pytest.raises(ValidationReportDefinitionError, match="exact Deterministic"):
        Inconclusive(
            (),
            _counts(),
            _bounds("bounded_search", complete=False),
            "a",
            1,
            reasons=("bounded",),
            witness=cast(Any, object()),
        )
    with pytest.raises(ValidationReportDefinitionError, match="evaluated assignment"):
        Inconclusive(
            (),
            _counts(expected=4, evaluated=0, conclusions=0, compared=0),
            _bounds("bounded_search", complete=False),
            "a",
            1,
            reasons=("bounded",),
            witness=_gap(phase="conclusion"),
        )
    with pytest.raises(ValidationReportDefinitionError, match="arity"):
        Inconclusive(
            (),
            _counts(expected=4, evaluated=1, conclusions=1, undecidable=1, compared=0),
            _bounds("bounded_search", complete=False),
            "a",
            1,
            reasons=("gap",),
            witness=DeterministicWitness(
                "evaluation_gap",
                "o",
                "a",
                1,
                "not_applicable",
                0,
                (0,),
                "conclusion",
                order_size=4,
                declared_order=(0, 1, 2, 3),
                evaluated_indices=(0,),
                earlier_resolved=False,
            ),
        )
    with pytest.raises(ValidationReportDefinitionError, match="domain cardinality"):
        Inconclusive(
            (),
            _counts(expected=4, evaluated=1, conclusions=1, undecidable=1, compared=0),
            _bounds("bounded_search", complete=False),
            "a",
            1,
            reasons=("gap",),
            witness=DeterministicWitness(
                "evaluation_gap",
                "o",
                "a",
                1,
                "not_applicable",
                0,
                (2, 0),
                "conclusion",
                order_size=4,
                declared_order=(0, 1, 2, 3),
                evaluated_indices=(0,),
                earlier_resolved=False,
            ),
        )


def test_undecidable_counts_and_inconclusive_completion_are_coherent() -> None:
    with pytest.raises(
        ValidationReportDefinitionError, match="undecidable_assignments"
    ):
        ValidationCounts(4, 1, undecidable_assignments=2)
    interrupted = Inconclusive(
        (),
        _counts(expected=4, evaluated=2, conclusions=2, compared=2),
        _bounds(),
        "a",
        1,
        reasons=("interrupted_before_complete_coverage",),
    )
    assert interrupted.bounds.complete
    with pytest.raises(ValidationReportDefinitionError, match="duplicate reason"):
        Inconclusive(
            (),
            _counts(expected=4, evaluated=2, conclusions=2, compared=2),
            _bounds("sampled", complete=False),
            "a",
            1,
            reasons=("sampled", "sampled"),
        )
    with pytest.raises(ValidationReportDefinitionError, match="requires a reason"):
        Inconclusive(
            (),
            _counts(expected=4, evaluated=2, conclusions=2, compared=2),
            _bounds("sampled", complete=False),
            "a",
            1,
            reasons=(),
        )


def test_strong_undefined_can_be_a_definite_disproof_without_comparison() -> None:
    counts = _counts(
        expected=4,
        evaluated=3,
        conclusions=3,
        compared=2,
        undefined_left=1,
        false=1,
    )
    report = Disproved(
        (), counts, _bounds(), "a", 1, _counterexample(subtype="strong_undefined")
    )
    assert report.counts.conclusion_false == 1
    assert report.counts.undefined_left == 1


def test_remaining_witness_and_status_guards_are_explicit() -> None:
    with pytest.raises(ValidationReportDefinitionError, match="minimality"):
        DeterministicWitness(
            "evaluation_gap",
            "o",
            "a",
            1,
            cast(Any, "unknown"),
            0,
            (),
            "conclusion",
            order_size=1,
            declared_order=(0,),
            evaluated_indices=(0,),
            earlier_resolved=False,
        )
    with pytest.raises(ValidationReportDefinitionError, match="resolved least prefix"):
        Disproved(
            (),
            _counts(false=1),
            _bounds(),
            "a",
            1,
            _gap(phase="conclusion"),
        )

    class UnknownReport(ValidationReport):
        pass

    unknown = object.__new__(UnknownReport)
    with pytest.raises(ValidationReportDefinitionError, match="unknown concrete"):
        _ = unknown.status


def test_terminal_partition_and_partial_semantics_are_exact() -> None:
    bad = (
        (
            dict(
                conclusions_evaluated=4,
                compared=3,
                vacuous_assignments=1,
            ),
            "conclusions_evaluated",
        ),
        (
            dict(
                conclusions_evaluated=3,
                compared=3,
                undefined_left=1,
            ),
            "conclusion_outcomes",
        ),
        (
            dict(
                conclusions_evaluated=4,
                compared=4,
                skipped_assignments=1,
            ),
            "skipped_assignments",
        ),
        (
            dict(
                conclusions_evaluated=4,
                compared=4,
                conclusion_false=5,
            ),
            "conclusion_false",
        ),
    )
    for kwargs, match in bad:
        with pytest.raises(ValidationReportDefinitionError, match=match):
            ValidationCounts(4, 4, **kwargs)

    conditional = ValidationCounts(
        4,
        4,
        conclusions_evaluated=4,
        compared=3,
        skipped_assignments=1,
        undefined_left=1,
    )
    report = Proved(
        (),
        conditional,
        _bounds(),
        "a",
        1,
        proof_route="exhaustive_finite",
        partial_semantics="definedness_conditional",
    )
    assert report.partial_semantics == "definedness_conditional"

    strong_skip = ValidationCounts(
        4,
        4,
        conclusions_evaluated=4,
        compared=3,
        skipped_assignments=1,
        undefined_left=1,
        conclusion_false=1,
    )
    with pytest.raises(ValidationReportDefinitionError, match="cannot skip"):
        Inconclusive(
            (),
            strong_skip,
            _bounds("bounded_search", complete=False),
            "a",
            1,
            reasons=("bad strong coverage",),
        )
    strong_missing_false = ValidationCounts(
        4,
        4,
        conclusions_evaluated=4,
        compared=3,
        undefined_left=1,
    )
    with pytest.raises(ValidationReportDefinitionError, match="contribute to false"):
        Inconclusive(
            (),
            strong_missing_false,
            _bounds("bounded_search", complete=False),
            "a",
            1,
            reasons=("bad strong coverage",),
        )
    with pytest.raises(ValidationReportDefinitionError, match="skip every undefined"):
        Inconclusive(
            (),
            strong_missing_false,
            _bounds("bounded_search", complete=False),
            "a",
            1,
            reasons=("bad conditional coverage",),
            partial_semantics="definedness_conditional",
        )
    conditional_false_without_comparison = ValidationCounts(
        1,
        1,
        conclusions_evaluated=1,
        skipped_assignments=1,
        undefined_left=1,
        conclusion_false=1,
    )
    with pytest.raises(ValidationReportDefinitionError, match="requires comparison"):
        Inconclusive(
            (),
            conditional_false_without_comparison,
            ValidationBounds("bounded_search", False, 1, domain_cardinalities=(1,)),
            "a",
            1,
            reasons=("bad conditional false",),
            partial_semantics="definedness_conditional",
        )
    all_undefined = ValidationCounts(
        4,
        4,
        conclusions_evaluated=4,
        skipped_assignments=4,
        undefined_left=4,
    )
    with pytest.raises(ValidationReportDefinitionError, match="compared conclusion"):
        Proved(
            (),
            all_undefined,
            _bounds(),
            "a",
            1,
            proof_route="exhaustive_finite",
            partial_semantics="definedness_conditional",
        )
    inconclusive = Inconclusive(
        (),
        all_undefined,
        _bounds(),
        "a",
        1,
        reasons=("no comparable accepted assignment",),
        partial_semantics="definedness_conditional",
    )
    assert inconclusive.counts.compared == 0
    vacuous = ValidationCounts(
        4,
        4,
        hypotheses_evaluated=4,
        hypothesis_rejected=4,
        vacuous_assignments=4,
    )
    vacuous_proof = Proved(
        (HypothesisEvidence("premise"),),
        vacuous,
        _bounds(),
        "a",
        1,
        proof_route="exhaustive_finite",
        partial_semantics="definedness_conditional",
    )
    assert vacuous_proof.counts.vacuous_assignments == 4


def test_witness_records_actual_unique_evaluated_assignment_indices() -> None:
    assert _counterexample().evaluated_indices == (0, 1, 2)
    custom_order = _counterexample(evaluated_indices=(3, 0, 2))
    report = Disproved(
        (),
        _counts(evaluated=3, conclusions=3, compared=3, false=1),
        _bounds(),
        "a",
        1,
        custom_order,
    )
    assert report.witness is not None
    assert report.witness.evaluated_indices == (3, 0, 2)
    with pytest.raises(ValidationReportDefinitionError, match="resolved least prefix"):
        Disproved(
            (),
            _counts(evaluated=3, conclusions=3, compared=3, false=1),
            _bounds(),
            "a",
            1,
            _counterexample(evaluated_indices=(2,)),
        )
    base = dict(
        kind="evaluation_gap",
        order_name="o",
        order_algorithm="a",
        order_version=1,
        minimality="not_applicable",
        assignment_index=0,
        carrier_indices=(),
        phase="conclusion",
        order_size=4,
        declared_order=(0, 1, 2, 3),
        earlier_resolved=False,
    )
    for evaluated, match in (
        ((), "at least"),
        ((0, 4), "inside order_size"),
        ((0, 0), "duplicates"),
        ((1,), "final evaluated"),
    ):
        with pytest.raises(ValidationReportDefinitionError, match=match):
            cast(Any, DeterministicWitness)(
                **base, evaluated_indices=cast(Any, evaluated)
            )
    with pytest.raises(ValidationReportDefinitionError, match="inside order_size"):
        cast(Any, DeterministicWitness)(
            **{**base, "assignment_index": 4}, evaluated_indices=(4,)
        )
    with pytest.raises(ValidationReportDefinitionError, match="built-in bool"):
        cast(Any, DeterministicWitness)(
            **{**base, "earlier_resolved": cast(Any, 1)}, evaluated_indices=(0,)
        )
    with pytest.raises(ValidationReportDefinitionError, match="maximum"):
        cast(Any, DeterministicWitness)(
            **{**base, "order_size": 65_537}, evaluated_indices=(0,)
        )
    with pytest.raises(ValidationReportDefinitionError, match="counterexample_subtype"):
        DeterministicWitness(
            "counterexample",
            "o",
            "a",
            1,
            "least_under_declared_order",
            0,
            (),
            "conclusion",
            counterexample_subtype="unknown",
            order_size=1,
            declared_order=(0,),
            evaluated_indices=(0,),
            earlier_resolved=True,
        )
    with pytest.raises(ValidationReportDefinitionError, match="earlier candidate"):
        DeterministicWitness(
            "counterexample",
            "o",
            "a",
            1,
            "least_under_declared_order",
            0,
            (),
            "conclusion",
            counterexample_subtype="defined_unequal",
            order_size=1,
            declared_order=(0,),
            evaluated_indices=(0,),
            earlier_resolved=False,
        )
    with pytest.raises(ValidationReportDefinitionError, match="cannot claim"):
        cast(Any, DeterministicWitness)(
            **base,
            evaluated_indices=(0,),
            counterexample_subtype="defined_unequal",
        )


def test_mixed_radix_domain_and_witness_coordinates_are_bound_together() -> None:
    zero_variable_bounds = ValidationBounds("exhaustive_finite", True, 1)
    zero_variable = Proved(
        (),
        ValidationCounts(
            1,
            1,
            conclusions_evaluated=1,
            compared=1,
        ),
        zero_variable_bounds,
        "a",
        1,
        proof_route="exhaustive_finite",
    )
    assert zero_variable.counts.assignments_expected == 1
    with pytest.raises(ValidationReportDefinitionError, match="mixed-radix"):
        Inconclusive(
            (),
            ValidationCounts(3, 0),
            _bounds("bounded_search", complete=False),
            "a",
            1,
            reasons=("wrong domain size",),
        )
    with pytest.raises(ValidationReportDefinitionError, match="coordinates"):
        Disproved(
            (),
            _counts(evaluated=3, conclusions=3, compared=3, false=1),
            _bounds(),
            "a",
            1,
            DeterministicWitness(
                "counterexample",
                "o",
                "a",
                1,
                "least_under_declared_order",
                1,
                (1, 0),
                "conclusion",
                counterexample_subtype="defined_unequal",
                order_size=4,
                declared_order=(0, 2, 1, 3),
                evaluated_indices=(0, 2, 1),
                earlier_resolved=True,
            ),
        )
    with pytest.raises(ValidationReportDefinitionError, match="order_size"):
        Inconclusive(
            (),
            _counts(evaluated=2, conclusions=2, compared=2),
            _bounds("bounded_search", complete=False),
            "a",
            1,
            reasons=("bad order",),
            witness=DeterministicWitness(
                "evaluation_gap",
                "o",
                "a",
                1,
                "not_applicable",
                1,
                (0, 1),
                "conclusion",
                order_size=3,
                declared_order=(0, 1, 2),
                evaluated_indices=(0, 1),
                earlier_resolved=False,
            ),
        )
    with pytest.raises(ValidationReportDefinitionError, match="evaluated indices"):
        Inconclusive(
            (),
            _counts(evaluated=1, conclusions=1, compared=1),
            _bounds("bounded_search", complete=False),
            "a",
            1,
            reasons=("too much witness coverage",),
            witness=_gap(phase="conclusion"),
        )


def test_checked_theorem_requires_every_independent_obligation_kind() -> None:
    bounds = ValidationBounds("checked_theorem_reduction", True, 1)
    obligations = (
        _verified("theorem", kind="theorem"),
        _verified("law", kind="law_identity"),
        _verified("closure", kind="closure"),
        _verified("scalar", kind="scalar_domain"),
        _verified("linear", kind="multilinearity"),
    )
    Proved(
        obligations,
        ValidationCounts(0, 0),
        bounds,
        "a",
        1,
        proof_route="checked_theorem_reduction",
    )
    with pytest.raises(ValidationReportDefinitionError, match="counts must be zero"):
        Proved(
            obligations,
            ValidationCounts(
                1,
                1,
                conclusions_evaluated=1,
                compared=1,
            ),
            bounds,
            "a",
            1,
            proof_route="checked_theorem_reduction",
        )
    with pytest.raises(ValidationReportDefinitionError, match="finite domain"):
        ValidationBounds(
            "checked_theorem_reduction", True, 1, domain_cardinalities=(1,)
        )
    duplicate = list(obligations)
    object.__setattr__(duplicate[1], "evidence_id", duplicate[0].evidence_id)
    with pytest.raises(ValidationReportDefinitionError, match="distinct"):
        Proved(
            duplicate,
            ValidationCounts(0, 0),
            bounds,
            "a",
            1,
            proof_route="checked_theorem_reduction",
        )
    with pytest.raises(ValidationReportDefinitionError, match="required verified"):
        Proved(
            obligations[1:],
            ValidationCounts(0, 0),
            bounds,
            "a",
            1,
            proof_route="checked_theorem_reduction",
        )
    fresh_without_linearity = (
        _verified("theorem2", kind="theorem"),
        _verified("law2", kind="law_identity"),
        _verified("closure2", kind="closure"),
        _verified("scalar2", kind="scalar_domain"),
    )
    with pytest.raises(ValidationReportDefinitionError, match="multilinearity"):
        Proved(
            fresh_without_linearity,
            ValidationCounts(0, 0),
            bounds,
            "a",
            1,
            proof_route="checked_theorem_reduction",
        )


def test_hypothesis_kind_and_evaluation_count_are_bounded() -> None:
    with pytest.raises(ValidationReportDefinitionError, match="hypothesis kind"):
        HypothesisEvidence("h", kind="unknown")
    with pytest.raises(ValidationReportDefinitionError, match="times declared"):
        Proved(
            (HypothesisEvidence("h"),),
            _counts(hypotheses=5),
            _bounds(),
            "a",
            1,
            proof_route="exhaustive_finite",
        )


def test_sampled_and_bounded_witnesses_cannot_claim_global_leastness() -> None:
    counts = _counts(evaluated=3, conclusions=3, compared=3, false=1)
    for route in ("sampled", "bounded_search"):
        with pytest.raises(ValidationReportDefinitionError, match="least disproof"):
            Disproved(
                (),
                counts,
                _bounds(route, complete=False),
                "a",
                1,
                _counterexample(),
            )
    with pytest.raises(ValidationReportDefinitionError, match="strong_undefined"):
        Disproved(
            (),
            counts,
            _bounds(),
            "a",
            1,
            _counterexample(subtype="strong_undefined"),
        )
    undefined = _counts(
        evaluated=3,
        conclusions=3,
        compared=2,
        undefined_left=1,
        false=1,
    )
    strong = Disproved(
        (),
        undefined,
        _bounds(),
        "a",
        1,
        _counterexample(subtype="strong_undefined"),
    )
    assert strong.witness is not None
    for incoherent_counts in (
        _counts(evaluated=3, conclusions=3, compared=3, false=2),
        _counts(
            evaluated=3,
            conclusions=2,
            compared=2,
            undecidable=1,
            false=1,
        ),
    ):
        with pytest.raises(ValidationReportDefinitionError, match="resolved least"):
            Disproved(
                (),
                incoherent_counts,
                _bounds(),
                "a",
                1,
                _counterexample(),
            )
    with pytest.raises(ValidationReportDefinitionError, match="unambiguous"):
        Disproved(
            (),
            undefined,
            _bounds(),
            "a",
            1,
            _counterexample(subtype="defined_unequal"),
        )
    undefined_only = _counts(
        evaluated=1,
        conclusions=1,
        compared=0,
        undefined_left=1,
        false=1,
    )
    with pytest.raises(ValidationReportDefinitionError, match="unambiguous"):
        Disproved(
            (),
            undefined_only,
            _bounds(),
            "a",
            1,
            _counterexample(0, subtype="defined_unequal", evaluated_indices=(0,)),
        )


def test_nested_records_are_reconstructed_before_hostile_iterables_run() -> None:
    original_counts = _counts()
    original_bounds = _bounds()
    original_witness = _counterexample()

    class MutatingHypotheses:
        def __iter__(self) -> MutatingHypotheses:
            return self

        def __next__(self) -> HypothesisEvidence:
            object.__setattr__(original_counts, "assignments_expected", 0)
            object.__setattr__(original_bounds, "route", "sampled")
            object.__setattr__(original_witness, "assignment_index", 0)
            raise StopIteration

    report = Proved(
        MutatingHypotheses(),
        original_counts,
        original_bounds,
        "a",
        1,
        proof_route="exhaustive_finite",
    )
    assert report.counts.assignments_expected == 4
    assert report.bounds.route == "exhaustive_finite"

    first = HypothesisEvidence("first")

    class MutateAfterYield:
        step = 0

        def __iter__(self) -> MutateAfterYield:
            return self

        def __next__(self) -> HypothesisEvidence:
            if self.step == 0:
                self.step = 1
                return first
            object.__setattr__(first, "name", "mutated")
            raise StopIteration

    report = Proved(
        MutateAfterYield(),
        _counts(hypotheses=4),
        _bounds(),
        "a",
        1,
        proof_route="exhaustive_finite",
    )
    assert report.hypotheses[0].name == "first"


def test_tampered_nested_records_fail_without_calling_hostile_storage() -> None:
    class Hostile:
        calls = 0

        def __iter__(self) -> Hostile:
            type(self).calls += 1
            raise RuntimeError("iter")

        def __hash__(self) -> int:
            type(self).calls += 1
            raise RuntimeError("hash")

        def __eq__(self, other: object) -> bool:
            del other
            type(self).calls += 1
            raise RuntimeError("eq")

    malformed_bounds = _bounds()
    object.__setattr__(malformed_bounds, "active_bounds", Hostile())
    with pytest.raises(ValidationReportDefinitionError, match="nested"):
        Proved(
            (),
            _counts(),
            malformed_bounds,
            "a",
            1,
            proof_route="exhaustive_finite",
        )
    malformed_witness = _counterexample()
    object.__setattr__(malformed_witness, "evaluated_indices", Hostile())
    with pytest.raises(ValidationReportDefinitionError, match="nested"):
        Disproved(
            (),
            _counts(evaluated=3, conclusions=3, compared=3, false=1),
            _bounds(),
            "a",
            1,
            malformed_witness,
        )
    assert Hostile.calls == 0

    for value, field, make in (
        (
            HypothesisEvidence("h"),
            "kind",
            lambda item: Proved(
                (item,),
                _counts(hypotheses=4),
                _bounds(),
                "a",
                1,
                proof_route="exhaustive_finite",
            ),
        ),
        (
            ActiveBound("n", 1),
            "limit",
            lambda item: ValidationBounds(
                "exhaustive_finite", True, 1, active_bounds=(item,)
            ),
        ),
        (
            _counts(),
            "compared",
            lambda item: Proved(
                (),
                item,
                _bounds(),
                "a",
                1,
                proof_route="exhaustive_finite",
            ),
        ),
        (
            _counterexample(),
            "phase",
            lambda item: Disproved(
                (),
                _counts(evaluated=3, conclusions=3, compared=3, false=1),
                _bounds(),
                "a",
                1,
                item,
            ),
        ),
    ):
        object.__delattr__(value, field)
        with pytest.raises(ValidationReportDefinitionError, match="malformed"):
            make(value)


def test_remaining_iterable_and_primitive_preflight_failures_are_typed() -> None:
    class BadIterable:
        def __iter__(self) -> Any:
            raise RuntimeError("iter")

    class BadIterator:
        def __iter__(self) -> BadIterator:
            return self

        def __next__(self) -> object:
            raise RuntimeError("next")

    for values in ("text", BadIterable(), BadIterator()):
        with pytest.raises(
            ValidationReportDefinitionError, match=r"not text|snapshotted"
        ):
            ValidationBounds(
                "exhaustive_finite",
                True,
                1,
                active_bounds=cast(Any, values),
            )
    with pytest.raises(ValidationReportDefinitionError, match="maximum 512"):
        ValidationBounds(
            "exhaustive_finite",
            True,
            1,
            active_bounds=(
                ActiveBound(f"bound_{index}", index) for index in range(513)
            ),
        )
    malformed_bounds = _bounds()
    object.__delattr__(malformed_bounds, "route")
    with pytest.raises(ValidationReportDefinitionError, match="malformed"):
        Proved(
            (),
            _counts(),
            malformed_bounds,
            "a",
            1,
            proof_route="exhaustive_finite",
        )
    with pytest.raises(ValidationReportDefinitionError, match="inside order_size"):
        DeterministicWitness(
            "evaluation_gap",
            "o",
            "a",
            1,
            "not_applicable",
            4,
            (),
            "conclusion",
            order_size=4,
            declared_order=(0, 1, 2, 3),
            evaluated_indices=(0,),
            earlier_resolved=False,
        )
    with pytest.raises(ValidationReportDefinitionError, match="partial_semantics"):
        Proved(
            (),
            _counts(),
            _bounds(),
            "a",
            1,
            proof_route="exhaustive_finite",
            partial_semantics="unknown",
        )
    for values in ("text", BadIterable(), BadIterator()):
        with pytest.raises(
            ValidationReportDefinitionError, match=r"not text|snapshotted"
        ):
            Proved(
                cast(Any, values),
                _counts(),
                _bounds(),
                "a",
                1,
                proof_route="exhaustive_finite",
            )


def test_vacuity_requires_actual_hypothesis_evaluation() -> None:
    with pytest.raises(ValidationReportDefinitionError, match="vacuous_assignments"):
        ValidationCounts(1, 1, vacuous_assignments=1)
    with pytest.raises(ValidationReportDefinitionError, match="hypothesis_rejected"):
        ValidationCounts(
            2,
            2,
            hypotheses_evaluated=1,
            hypothesis_rejected=2,
            vacuous_assignments=2,
        )
    undefined_premise_vacuity = ValidationCounts(
        1,
        1,
        hypotheses_evaluated=1,
        vacuous_assignments=1,
    )
    report = Proved(
        (HypothesisEvidence("partial premise"),),
        undefined_premise_vacuity,
        ValidationBounds("exhaustive_finite", True, 1, domain_cardinalities=(1,)),
        "a",
        1,
        proof_route="exhaustive_finite",
    )
    assert report.counts.hypothesis_rejected == 0


def test_declared_order_is_a_full_permutation_and_controls_the_prefix() -> None:
    common = dict(
        kind="evaluation_gap",
        order_name="custom",
        order_algorithm="custom.order",
        order_version=1,
        minimality="not_applicable",
        assignment_index=1,
        carrier_indices=(0, 1),
        phase="conclusion",
        order_size=4,
        evaluated_indices=(0, 1),
        earlier_resolved=False,
    )
    for order in ((0, 1, 2), (0, 1, 1, 3), (0, 1, 2, 4)):
        with pytest.raises(ValidationReportDefinitionError, match="full permutation"):
            cast(Any, DeterministicWitness)(**common, declared_order=order)
    with pytest.raises(ValidationReportDefinitionError, match="leading prefix"):
        cast(Any, DeterministicWitness)(
            **common,
            declared_order=(2, 0, 1, 3),
        )


def test_gap_and_premise_witnesses_require_execution_evidence() -> None:
    conclusion_gap = DeterministicWitness(
        "evaluation_gap",
        "o",
        "a",
        1,
        "not_applicable",
        0,
        (0, 0),
        "conclusion",
        order_size=4,
        declared_order=(0, 1, 2, 3),
        evaluated_indices=(0,),
        earlier_resolved=False,
    )
    with pytest.raises(ValidationReportDefinitionError, match="undecidable"):
        Inconclusive(
            (),
            _counts(evaluated=1, conclusions=1, compared=1),
            _bounds("bounded_search", complete=False),
            "a",
            1,
            reasons=("fabricated gap",),
            witness=conclusion_gap,
        )
    premise_gap = _gap()
    with pytest.raises(ValidationReportDefinitionError, match="declared hypothesis"):
        Inconclusive(
            (),
            _counts(
                evaluated=2,
                conclusions=1,
                compared=1,
                undecidable=1,
            ),
            _bounds("bounded_search", complete=False),
            "a",
            1,
            reasons=("foreign premise",),
            witness=premise_gap,
        )
    with pytest.raises(ValidationReportDefinitionError, match="premise index"):
        Inconclusive(
            (HypothesisEvidence("premise"),),
            _counts(
                evaluated=2,
                conclusions=1,
                compared=1,
                undecidable=1,
            ),
            _bounds("bounded_search", complete=False),
            "a",
            1,
            reasons=("unevaluated premise",),
            witness=premise_gap,
        )
    with pytest.raises(
        ValidationReportDefinitionError, match="premise-evaluation minimum"
    ):
        Inconclusive(
            (HypothesisEvidence("premise"),),
            _counts(
                evaluated=4,
                conclusions=4,
                compared=4,
                hypotheses=1,
            ),
            _bounds("sampled", complete=False),
            "a",
            1,
            reasons=("under-counted premise evaluations",),
        )
    second_premise_gap = DeterministicWitness(
        "evaluation_gap",
        "o",
        "a",
        1,
        "not_applicable",
        0,
        (0, 0),
        "premise",
        1,
        order_size=4,
        declared_order=(0, 1, 2, 3),
        evaluated_indices=(0,),
        earlier_resolved=False,
    )
    with pytest.raises(ValidationReportDefinitionError, match="premise index"):
        Inconclusive(
            (HypothesisEvidence("first"), HypothesisEvidence("second")),
            _counts(
                evaluated=1,
                conclusions=0,
                compared=0,
                undecidable=1,
                hypotheses=1,
            ),
            _bounds("bounded_search", complete=False),
            "a",
            1,
            reasons=("second premise was not reached",),
            witness=second_premise_gap,
        )
    with pytest.raises(ValidationReportDefinitionError, match="only law-hypothesis"):
        Inconclusive(
            (_verified("closure", kind="closure"), HypothesisEvidence("law")),
            _counts(
                evaluated=1,
                conclusions=0,
                compared=0,
                undecidable=1,
                hypotheses=1,
            ),
            _bounds("bounded_search", complete=False),
            "a",
            1,
            reasons=("global obligation used as a premise",),
            witness=DeterministicWitness(
                "evaluation_gap",
                "o",
                "a",
                1,
                "not_applicable",
                0,
                (0, 0),
                "premise",
                0,
                order_size=4,
                declared_order=(0, 1, 2, 3),
                evaluated_indices=(0,),
                earlier_resolved=False,
            ),
        )
    with pytest.raises(
        ValidationReportDefinitionError, match="premise-evaluation minimum"
    ):
        Inconclusive(
            (HypothesisEvidence("first"), HypothesisEvidence("second")),
            _counts(
                evaluated=4,
                conclusions=4,
                compared=4,
                hypotheses=4,
            ),
            _bounds("sampled", complete=False),
            "a",
            1,
            reasons=("only half the ordered premises were counted",),
        )
    late_second_premise_gap = DeterministicWitness(
        "evaluation_gap",
        "o",
        "a",
        1,
        "not_applicable",
        3,
        (1, 1),
        "premise",
        1,
        order_size=4,
        declared_order=(0, 1, 2, 3),
        evaluated_indices=(0, 1, 2, 3),
        earlier_resolved=False,
    )
    with pytest.raises(
        ValidationReportDefinitionError, match="premise-evaluation minimum"
    ):
        Inconclusive(
            (HypothesisEvidence("first"), HypothesisEvidence("second")),
            _counts(
                evaluated=4,
                conclusions=3,
                compared=3,
                undecidable=1,
                hypotheses=7,
            ),
            _bounds("bounded_search", complete=False),
            "a",
            1,
            reasons=("gap premise depth was not counted",),
            witness=late_second_premise_gap,
        )
