"""Provider adapters. There is one: OpenRouter.

A package rather than a module because architecture.md 24 names it as one, and because a second
surface — were the subnet ever to add one — must land beside this rather than inside it.
"""

from gateway.adapters.openrouter import (
    AdapterError,
    CallOutcome,
    ModelPin,
    OpenRouterAdapter,
    UndeclaredModel,
)

__all__ = [
    "AdapterError",
    "CallOutcome",
    "ModelPin",
    "OpenRouterAdapter",
    "UndeclaredModel",
]
