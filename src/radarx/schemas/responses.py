"""API response models."""

from radarx.schemas.token import TokenScore
from radarx.schemas.wallet import WalletReport

# These are aliases for the main response models
TokenScoreResponse = TokenScore
WalletReportResponse = WalletReport

__all__ = ["TokenScoreResponse", "WalletReportResponse"]
