"""Typed replacement inventory for AlgMul's missing ``MakeProperty`` factory.

``MakeProperty`` intended to create 128 Wolfram names at runtime.  The pinned
legacy capture observed neither the factory nor any generated definition, so
this module records the intended surface as data and maps only its eight law
kernels to explicit typed :class:`~anyalgebra.structures.laws.Law` factories.
It deliberately does not recreate 128 Python functions or turn a bounded
validation result into a legacy-style Boolean.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from anyalgebra.core.errors import AnyAlgebraError
from anyalgebra.legacy.models import Disposition
from anyalgebra.structures.laws import Equation, Law
from anyalgebra.structures.signatures import OperationSymbol
from anyalgebra.structures.terms import Term, Variable
from anyalgebra.validation.factories import (
    associative_law,
    commutative_law,
    custom_law,
    flexible_law,
    right_alternativity_law,
)


class PropertyMappingError(AnyAlgebraError, ValueError):
    """A request cannot be mapped to the finite typed legacy inventory."""

    def __init__(self, *, field: str, reason: str) -> None:
        """Keep failures structural and free of arbitrary caller rendering."""
        self.field = field
        self.reason = reason
        super().__init__(f"invalid legacy property mapping {field}: {reason}")


@dataclass(frozen=True, slots=True)
class LegacyPropertyLaw:
    """One source-level ``MakeProperty`` call and its typed replacement route."""

    name: str
    arity: int
    kernel: str
    typed_factory: str
    disposition: Disposition
    replacement_route: str
    oracle: str
    caveat: str


@dataclass(frozen=True, slots=True)
class PropertySurface:
    """One intended prefix, carrier lift, evaluation mode, and result mode."""

    prefix: str
    mode: str
    lift: str
    result: str
    disposition: Disposition
    result_route: str
    oracle: str


@dataclass(frozen=True, slots=True)
class IntendedPropertyMember:
    """One of the 128 intended names, retained as metadata rather than a function."""

    name: str
    law_name: str
    arity: int
    kernel: str
    typed_factory: str
    mode: str
    lift: str
    result: str
    disposition: Disposition
    replacement_route: str
    result_route: str
    oracle: str
    legacy_status: str
    caveat: str


@dataclass(frozen=True, slots=True)
class PropertyFactoryRecord:
    """Canonical disposition and route for the absent legacy factory itself."""

    name: str
    legacy_status: str
    disposition: Disposition
    replacement_route: str
    oracle: str
    note: str


@dataclass(frozen=True, slots=True)
class PropertyValidationPolicy:
    """Evidence boundary retained for all typed property replacement routes."""

    primary_api: str
    outcomes: tuple[str, ...]
    sampled_pass: str
    counterexample_order: str
    basis_reduction: str
    fingerprints: str


_LAWS: tuple[LegacyPropertyLaw, ...] = (
    LegacyPropertyLaw(
        "Commutative",
        2,
        "Comm",
        "commutative_law",
        Disposition.REPLACE,
        "typed_law_factory",
        "definition_plus_finite_validation",
        "The source kernel is an additive commutator; the typed law uses equality.",
    ),
    LegacyPropertyLaw(
        "Associative",
        3,
        "Assoc",
        "associative_law",
        Disposition.REPLACE,
        "typed_law_factory",
        "definition_plus_finite_validation",
        "The exact parenthesization is retained in the Law term tree.",
    ),
    LegacyPropertyLaw(
        "Alternative",
        2,
        "Alternative",
        "right_alternativity_law",
        Disposition.REPLACE,
        "typed_law_factory",
        "definition_plus_finite_validation",
        "Legacy Alternative is only right alternativity, not the two-sided property.",
    ),
    LegacyPropertyLaw(
        "Flexible",
        2,
        "Flexible",
        "flexible_law",
        Disposition.REPLACE,
        "typed_law_factory",
        "definition_plus_finite_validation",
        "No associativity or alternativity premise is silently added.",
    ),
    LegacyPropertyLaw(
        "PowerAssociative1",
        1,
        "PowerAssociative1",
        "legacy_power_associative_1_law",
        Disposition.REPLACE,
        "typed_law_factory",
        "definition_plus_finite_validation",
        "This is only the degree-three parenthesization diagnostic.",
    ),
    LegacyPropertyLaw(
        "PowerAssociative2",
        1,
        "PowerAssociative2",
        "legacy_power_associative_2_law",
        Disposition.REPLACE,
        "typed_law_factory",
        "definition_plus_finite_validation",
        "This is only the degree-four parenthesization diagnostic.",
    ),
    LegacyPropertyLaw(
        "JIdentity",
        2,
        "JIdentity",
        "legacy_j_identity_law",
        Disposition.REPLACE,
        "typed_law_factory",
        "definition_plus_finite_validation",
        "The legacy parenthesization is retained; commutativity is a separate law.",
    ),
    LegacyPropertyLaw(
        "Jacobi",
        3,
        "JacobiIdentity",
        "legacy_jacobi_law",
        Disposition.REPLACE,
        "typed_law_factory",
        "definition_plus_finite_validation",
        "Requires explicit bracket, addition, and zero operations.",
    ),
)

_SURFACES: tuple[PropertySurface, ...] = (
    PropertySurface(
        "A",
        "symbolic",
        "scalar",
        "expression",
        Disposition.REPLACE,
        "typed_law_expression",
        "definition",
    ),
    PropertySurface(
        "AT",
        "symbolic",
        "tensor",
        "expression",
        Disposition.REPLACE,
        "typed_law_expression",
        "definition",
    ),
    PropertySurface(
        "AM",
        "symbolic",
        "matrix",
        "expression",
        Disposition.REPLACE,
        "typed_law_expression",
        "definition",
    ),
    PropertySurface(
        "AMT",
        "symbolic",
        "matrix_over_tensor",
        "expression",
        Disposition.REPLACE,
        "typed_law_expression",
        "definition",
    ),
    PropertySurface(
        "IsA",
        "symbolic",
        "scalar",
        "validation",
        Disposition.REPLACE,
        "validation_report",
        "definition",
    ),
    PropertySurface(
        "IsAT",
        "symbolic",
        "tensor",
        "validation",
        Disposition.REPLACE,
        "validation_report",
        "definition",
    ),
    PropertySurface(
        "IsAM",
        "symbolic",
        "matrix",
        "validation",
        Disposition.REPLACE,
        "validation_report",
        "definition",
    ),
    PropertySurface(
        "IsAMT",
        "symbolic",
        "matrix_over_tensor",
        "validation",
        Disposition.REPLACE,
        "validation_report",
        "definition",
    ),
    PropertySurface(
        "N",
        "bounded_numeric",
        "scalar",
        "expression",
        Disposition.REPLACE,
        "typed_law_expression",
        "bounded_sample_not_proof",
    ),
    PropertySurface(
        "NT",
        "bounded_numeric",
        "tensor",
        "expression",
        Disposition.REPLACE,
        "typed_law_expression",
        "bounded_sample_not_proof",
    ),
    PropertySurface(
        "NM",
        "bounded_numeric",
        "matrix",
        "expression",
        Disposition.REPLACE,
        "typed_law_expression",
        "bounded_sample_not_proof",
    ),
    PropertySurface(
        "NMT",
        "bounded_numeric",
        "matrix_over_tensor",
        "expression",
        Disposition.REPLACE,
        "typed_law_expression",
        "bounded_sample_not_proof",
    ),
    PropertySurface(
        "IsN",
        "bounded_numeric",
        "scalar",
        "validation",
        Disposition.REPLACE,
        "validation_report",
        "bounded_sample_not_proof",
    ),
    PropertySurface(
        "IsNT",
        "bounded_numeric",
        "tensor",
        "validation",
        Disposition.REPLACE,
        "validation_report",
        "bounded_sample_not_proof",
    ),
    PropertySurface(
        "IsNM",
        "bounded_numeric",
        "matrix",
        "validation",
        Disposition.REPLACE,
        "validation_report",
        "bounded_sample_not_proof",
    ),
    PropertySurface(
        "IsNMT",
        "bounded_numeric",
        "matrix_over_tensor",
        "validation",
        Disposition.REPLACE,
        "validation_report",
        "bounded_sample_not_proof",
    ),
)

_FACTORY_RECORD = PropertyFactoryRecord(
    "MakeProperty",
    "absent_in_pinned_runtime_capture",
    Disposition.REPLACE,
    "typed_property_registry",
    "pinned_runtime_surface_receipt",
    (
        "The pinned Wolfram 12.0 clean-kernel receipt observed the factory "
        "absent and all 128 intended symbols absent; this is runtime-surface "
        "evidence, not a correctness oracle."
    ),
)

_VALIDATION_POLICY = PropertyValidationPolicy(
    "anyalgebra.validation.validate.validate_law",
    ("Proved", "Disproved", "Inconclusive"),
    "not_proof",
    "finite_substitution_domain_lexicographic",
    "requires_recorded_multilinearity_hypotheses",
    "state_conventions_and_bounds",
)


def property_laws() -> tuple[LegacyPropertyLaw, ...]:
    """Return the source-ordered, immutable eight-law replacement inventory."""
    return _LAWS


def property_surfaces() -> tuple[PropertySurface, ...]:
    """Return the source-ordered, immutable sixteen-prefix inventory."""
    return _SURFACES


def property_factory_record() -> PropertyFactoryRecord:
    """Return the canonical typed disposition for the missing factory record."""
    return _FACTORY_RECORD


def property_validation_policy() -> PropertyValidationPolicy:
    """Return the immutable evidence policy for every replacement member."""
    return _VALIDATION_POLICY


def _member_caveat(law: LegacyPropertyLaw, surface: PropertySurface) -> str:
    """Record known source defects at the exact affected intended name only."""
    if surface.prefix == "AMT" and law.arity == 2:
        return (
            "legacy_defect: AMT arity-2 branch passes raw fooFoo, not "
            "MT-prefixed kernel"
        )
    if surface.prefix == "NM" and law.arity == 3:
        return (
            "legacy_defect: NM arity-3 branch omits a comma after "
            "ToExpression[M<>fooFoo]"
        )
    return law.caveat


def intended_property_members() -> tuple[IntendedPropertyMember, ...]:
    """Return all 128 intended runtime names in deterministic source-call order.

    Their ``legacy_status`` reports the pinned clean-kernel observation, while
    their dispositions describe the corrected AnyAlgebra replacement.  A
    finite validation consumes a ``Law`` through ``validate_law``; it never
    upgrades a sampled or bounded result merely because a legacy wrapper did.
    """
    return tuple(
        IntendedPropertyMember(
            name=surface.prefix + law.name,
            law_name=law.name,
            arity=law.arity,
            kernel=law.kernel,
            typed_factory=law.typed_factory,
            mode=surface.mode,
            lift=surface.lift,
            result=surface.result,
            disposition=surface.disposition,
            replacement_route=law.replacement_route,
            result_route=surface.result_route,
            oracle=surface.oracle,
            legacy_status="absent_in_pinned_runtime_capture",
            caveat=_member_caveat(law, surface),
        )
        for law in _LAWS
        for surface in _SURFACES
    )


def _require_operation(value: object, *, field: str) -> OperationSymbol:
    """Require one literal closed binary operation without name-based lookup."""
    if type(value) is not OperationSymbol:
        raise PropertyMappingError(
            field=field, reason="must be an exact OperationSymbol"
        )
    if value.arity != 2 or value.inputs != (value.output, value.output):
        raise PropertyMappingError(
            field=field, reason="must be a closed binary OperationSymbol"
        )
    return value


def _require_zero(value: object, *, sort_name: str, sort: object) -> OperationSymbol:
    """Require the explicit additive zero used only by the Jacobi equation."""
    if type(value) is not OperationSymbol:
        raise PropertyMappingError(
            field="zero", reason="must be an exact OperationSymbol"
        )
    if value.arity != 0 or value.output != sort:
        raise PropertyMappingError(
            field="zero",
            reason=f"must be a nullary OperationSymbol with {sort_name} output sort",
        )
    return value


def _power_law(product: OperationSymbol, *, name: str, degree: int) -> Law:
    """Build either literal legacy finite power-parenthesization diagnostic."""
    x = Variable("x", product.output)
    x_term = Term.variable(x)
    square = Term.apply(product, x_term, x_term)
    cube_left = Term.apply(product, square, x_term)
    cube_right = Term.apply(product, x_term, square)
    if degree == 3:
        return custom_law(
            name, (x,), Equation(cube_left, cube_right), operations=(product,)
        )
    return custom_law(
        name,
        (x,),
        Equation(
            Term.apply(product, cube_left, x_term), Term.apply(product, square, square)
        ),
        operations=(product,),
    )


def _j_identity_law(product: OperationSymbol, *, name: str) -> Law:
    """Build the exact legacy ``JIdentity`` tree without adding commutativity."""
    x, y = Variable("x", product.output), Variable("y", product.output)
    x_term, y_term = Term.variable(x), Term.variable(y)
    square = Term.apply(product, x_term, x_term)
    return custom_law(
        name,
        (x, y),
        Equation(
            Term.apply(product, Term.apply(product, x_term, y_term), square),
            Term.apply(product, x_term, Term.apply(product, y_term, square)),
        ),
        operations=(product,),
    )


def _jacobi_law(
    bracket: OperationSymbol,
    addition: OperationSymbol,
    zero: OperationSymbol,
    *,
    name: str,
) -> Law:
    """Build the literal cyclic legacy Jacobi order as one equality to zero."""
    checked_addition = _require_operation(addition, field="addition")
    if checked_addition.output != bracket.output:
        raise PropertyMappingError(
            field="addition", reason="must have the bracket sort as its closed sort"
        )
    checked_zero = _require_zero(zero, sort_name="bracket", sort=bracket.output)
    x, y, z = (Variable(value, bracket.output) for value in ("x", "y", "z"))
    x_term, y_term, z_term = (Term.variable(value) for value in (x, y, z))
    first = Term.apply(bracket, x_term, Term.apply(bracket, y_term, z_term))
    second = Term.apply(bracket, z_term, Term.apply(bracket, x_term, y_term))
    third = Term.apply(bracket, y_term, Term.apply(bracket, z_term, x_term))
    return custom_law(
        name,
        (x, y, z),
        Equation(
            Term.apply(
                checked_addition, Term.apply(checked_addition, first, second), third
            ),
            Term.apply(checked_zero),
        ),
        operations=(bracket, checked_addition, checked_zero),
    )


def _with_partial_semantics(law: Law, partial_semantics: str) -> Law:
    """Keep custom legacy term trees while applying the declared law policy."""
    if partial_semantics == "strong":
        return law
    return Law(
        law.name,
        law.variables,
        law.conclusion,
        hypotheses=law.hypotheses,
        partial_semantics=partial_semantics,
    )


def legacy_power_associative_1_law(
    operation: OperationSymbol,
    *,
    name: str = "legacy:PowerAssociative1",
    partial_semantics: str = "strong",
) -> Law:
    """Return the literal degree-three legacy power diagnostic as a typed law."""
    return _with_partial_semantics(
        _power_law(operation, name=name, degree=3), partial_semantics
    )


def legacy_power_associative_2_law(
    operation: OperationSymbol,
    *,
    name: str = "legacy:PowerAssociative2",
    partial_semantics: str = "strong",
) -> Law:
    """Return the literal degree-four legacy power diagnostic as a typed law."""
    return _with_partial_semantics(
        _power_law(operation, name=name, degree=4), partial_semantics
    )


def legacy_j_identity_law(
    operation: OperationSymbol,
    *,
    name: str = "legacy:JIdentity",
    partial_semantics: str = "strong",
) -> Law:
    """Return the literal legacy Jordan-identity tree without extra premises."""
    return _with_partial_semantics(
        _j_identity_law(operation, name=name), partial_semantics
    )


def legacy_jacobi_law(
    bracket: OperationSymbol,
    *,
    addition: OperationSymbol,
    zero: OperationSymbol,
    name: str = "legacy:Jacobi",
    partial_semantics: str = "strong",
) -> Law:
    """Return the literal legacy cyclic Jacobi sum as one typed equation."""
    return _with_partial_semantics(
        _jacobi_law(bracket, addition, zero, name=name), partial_semantics
    )


_LAW_FACTORY_REGISTRY: Mapping[str, Callable[..., Law]] = MappingProxyType(
    {
        "Commutative": commutative_law,
        "Associative": associative_law,
        "Alternative": right_alternativity_law,
        "Flexible": flexible_law,
        "PowerAssociative1": legacy_power_associative_1_law,
        "PowerAssociative2": legacy_power_associative_2_law,
        "JIdentity": legacy_j_identity_law,
        "Jacobi": legacy_jacobi_law,
    }
)


def _law_record(law_name: str) -> LegacyPropertyLaw:
    """Look up one source call after exact public-name validation."""
    if type(law_name) is not str:
        raise PropertyMappingError(field="law_name", reason="must be a built-in str")
    law = next((item for item in _LAWS if item.name == law_name), None)
    if law is None:
        raise PropertyMappingError(
            field="law_name", reason="is not in the intended eight-law inventory"
        )
    return law


def property_law_factory(law_name: str) -> Callable[..., Law]:
    """Resolve a metadata factory identifier to the callable used for dispatch."""
    law = _law_record(law_name)
    factory = _LAW_FACTORY_REGISTRY[law.name]
    canonical: object = globals().get(law.typed_factory)
    if factory.__name__ != law.typed_factory or canonical is not factory:
        raise PropertyMappingError(
            field="typed_factory",
            reason="does not resolve to the law dispatch callable",
        )
    return factory


def makeproperty_law(
    law_name: str,
    operation: OperationSymbol,
    *,
    addition: OperationSymbol | None = None,
    zero: OperationSymbol | None = None,
    partial_semantics: str = "strong",
) -> Law:
    """Map one intended legacy law name to a typed law with explicit operations.

    For every non-Jacobi mapping, ``operation`` is the selected closed binary
    product.  ``Jacobi`` instead treats it as the bracket and additionally
    requires explicit closed addition and a nullary zero.  The returned law is
    syntax only; callers must pass it to ``validation.validate_law`` with a
    declared finite structure, bounds, and conventions to obtain evidence.
    """
    law = _law_record(law_name)
    product = _require_operation(operation, field="operation")
    if type(partial_semantics) is not str or partial_semantics not in {
        "strong",
        "definedness_conditional",
    }:
        raise PropertyMappingError(
            field="partial_semantics",
            reason="must be 'strong' or 'definedness_conditional'",
        )
    if law.name != "Jacobi" and (addition is not None or zero is not None):
        raise PropertyMappingError(
            field="addition", reason="addition and zero are only accepted for Jacobi"
        )
    factory = property_law_factory(law.name)
    name = f"legacy:{law.name}"
    if law.name == "Jacobi":
        if addition is None or zero is None:
            raise PropertyMappingError(
                field="Jacobi", reason="requires explicit addition and zero operations"
            )
        return factory(
            product,
            addition=addition,
            zero=zero,
            name=name,
            partial_semantics=partial_semantics,
        )
    return factory(product, name=name, partial_semantics=partial_semantics)
