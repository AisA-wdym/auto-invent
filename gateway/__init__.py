"""The Research Compute Gateway. architecture.md 3.4.

Brokers every external call a laboratory makes: one provider surface (OpenRouter), two accounts,
and a hard-chained receipt for each call.
"""

from gateway.credentials import (
    CredentialError,
    CredentialSet,
    MinerCredential,
    ValidatorCredential,
)
from gateway.metering import BudgetExceeded, Ledger, PriceTable
from gateway.tokens import SessionToken, TokenError, TokenIssuer

__all__ = [
    "BudgetExceeded",
    "CredentialError",
    "CredentialSet",
    "Ledger",
    "MinerCredential",
    "PriceTable",
    "SessionToken",
    "TokenError",
    "TokenIssuer",
    "ValidatorCredential",
]
