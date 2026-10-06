"""Typed diagnostics for neutral AnyAlgebra foundation boundaries."""

from __future__ import annotations


class AnyAlgebraError(Exception):
    """Base class for package-defined errors with mathematical diagnostics."""


class DomainConstructionError(AnyAlgebraError, ValueError):
    """A domain rejected an input before constructing an exact element."""

    def __init__(self, domain: object, reason: str) -> None:
        """Record the rejecting domain and a deterministic diagnostic reason."""
        self.domain = domain
        self.reason = reason
        super().__init__(f"exact construction rejected by {domain!r}: {reason}")


class CoercionError(AnyAlgebraError):
    """Base class for a prohibited or unresolved coercion."""

    def __init__(self, reason: str) -> None:
        """Record the diagnostic without selecting or approximating a route."""
        self.reason = reason
        super().__init__(reason)


class CoercionAmbiguityError(CoercionError):
    """More than one admissible coercion route remains unselected."""


class LossyCoercionError(CoercionError):
    """A requested conversion would discard information without explicit policy."""
