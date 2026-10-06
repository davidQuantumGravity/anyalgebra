"""Small, exact, standard-library reference route for V00-080.

The neutral backend request contains ordered semantic hashes, not their source
values.  Accordingly this backend deliberately implements only ``hash_identity``:
one input hash is returned as the result hash.  It proves exact routing and
receipt binding without falsely claiming to evaluate an unavailable value.
"""

from __future__ import annotations

from anyalgebra.backends.base import (
    BackendBoundaryError,
    BackendCapabilities,
    BackendOptions,
    BackendRequest,
    BackendResult,
    BackendUnsupported,
    CapabilityResult,
    CapabilitySpec,
    negotiate,
)
from anyalgebra.structures.outcomes import (
    Defined,
    EvaluationOutcome,
    Failed,
    UnsupportedCapability,
)


_OPERATION = "hash_identity"
_DOMAIN = "semantic_hash"
_ALGORITHM = "identity"
_LIMIT_NAME = "inputs"
_LIMIT_VALUE = 1


class ReferenceBackend:
    """The sealed, pure-Python exact backend for the neutral hash boundary."""

    __slots__ = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        del cls, kwargs
        raise TypeError("ReferenceBackend cannot be subclassed")

    def capabilities(self) -> BackendCapabilities:
        """Return a fresh sealed declaration, safe from caller-side tampering."""
        specification = CapabilitySpec.create(
            operation=_OPERATION,
            domain=_DOMAIN,
            algorithm=_ALGORITHM,
            exact=True,
        )
        return BackendCapabilities.create(
            name="reference",
            version="0.0.0",
            specifications=(specification,),
            limits=((_LIMIT_NAME, _LIMIT_VALUE),),
        )

    def supports(self, request: BackendRequest) -> CapabilityResult:
        """Negotiate before execution and report unsupported requests neutrally."""
        capabilities = self.capabilities()
        negotiated = negotiate(capabilities, request)
        if type(negotiated) is BackendUnsupported:
            return negotiated
        if len(request.input_hashes) != _LIMIT_VALUE or request.limits != (
            (_LIMIT_NAME, _LIMIT_VALUE),
        ):
            raise BackendBoundaryError(
                "request",
                "hash_identity requires exactly one input and an inputs=1 limit",
            )
        return negotiated

    def execute(
        self, request: BackendRequest, *, options: BackendOptions
    ) -> EvaluationOutcome[BackendResult]:
        """Return a neutral receipt or a closed evaluation failure.

        Unsupported capability is intentionally outside ``EvaluationOutcome``:
        callers must use ``supports`` to obtain the typed availability result.
        """
        try:
            support = self.supports(request)
            if isinstance(support, BackendUnsupported):
                raise UnsupportedCapability(
                    request.operation, context=support.reason.value
                )
            if type(options) is not BackendOptions:
                return Failed("reference execution rejected invalid options", "binding")
            options._assert()
            if (
                options.convention_hashes != request.convention_hashes
                or options.limits != request.limits
            ):
                return Failed("reference execution rejected option binding", "binding")
            result = BackendResult.create(
                request=request,
                provenance=support.provenance,
                options=options,
                value_hash=request.input_hashes[0],
            )
            return Defined(result)
        except BackendBoundaryError:
            return Failed("reference execution rejected request binding", "binding")
        except (KeyboardInterrupt, SystemExit, UnsupportedCapability):
            raise
        except Exception:
            return Failed("reference execution failed", "execution")
