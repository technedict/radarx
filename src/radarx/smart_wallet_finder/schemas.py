"""
Smart Wallet Finder API Schemas

Pydantic models for request/response validation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SmartWalletFindRequest(BaseModel):
    """Request to find smart wallets for a token."""

    token_address: str = Field(..., description="Token contract address")
    chain: str = Field(default="ethereum", description="Blockchain network")
    window_days: int = Field(default=30, ge=1, le=365, description="Analysis window in days")
    min_trade_size_usd: Optional[float] = Field(None, ge=0, description="Minimum trade size (USD)")
    min_holdings_usd: Optional[float] = Field(None, ge=0, description="Minimum holdings (USD)")
    include_internal_transfers: bool = Field(
        default=False, description="Include internal transfers"
    )
    top_k: int = Field(default=100, ge=1, le=1000, description="Number of top wallets to return")
    min_confidence: float = Field(
        default=0.5, ge=0, le=1, description="Minimum confidence threshold"
    )


class WalletProfileRequest(BaseModel):
    """Request for detailed wallet profile."""

    wallet_address: str = Field(..., description="Wallet address")
    token_address: str = Field(..., description="Token contract address")
    chain: str = Field(default="ethereum", description="Blockchain network")
    window_days: int = Field(default=30, ge=1, le=365, description="Analysis window in days")


class BulkScanRequest(BaseModel):
    """Request to scan multiple tokens."""

    token_addresses: List[str] = Field(..., description="List of token addresses")
    chain: str = Field(default="ethereum", description="Blockchain network")
    window_days: int = Field(default=30, ge=1, le=365, description="Analysis window in days")
    top_k_per_token: int = Field(default=10, ge=1, le=100, description="Top wallets per token")
    min_confidence: float = Field(
        default=0.6, ge=0, le=1, description="Minimum confidence threshold"
    )


class KeyMetrics(BaseModel):
    """Key metrics for a wallet."""

    win_rate: float = Field(..., ge=0, le=1, description="Win rate ratio")
    realized_roi: float = Field(..., description="Average realized ROI")
    trades_count: int = Field(..., ge=0, description="Number of trades")
    early_entry_rate: float = Field(..., ge=0, le=1, description="Pre-pump entry rate")
    graph_centrality: float = Field(..., ge=0, le=1, description="Graph centrality score")


class SignalContribution(BaseModel):
    """Signal contribution to score."""

    category: str = Field(..., description="Signal category")
    name: str = Field(..., description="Signal name")
    value: float = Field(..., description="Signal value")
    contribution: float = Field(..., description="Contribution to score")
    direction: str = Field(..., description="Direction (positive/negative)")
    description: str = Field(..., description="Human-readable description")


class TimelineEvent(BaseModel):
    """Timeline event."""

    event_type: str = Field(..., description="Event type")
    description: str = Field(..., description="Event description")
    impact: str = Field(..., description="Impact (positive/negative/neutral)")


class WalletExplanation(BaseModel):
    """Explanation for wallet score."""

    summary: str = Field(..., description="High-level summary")
    top_signals: List[SignalContribution] = Field(..., description="Top contributing signals")
    interpretation: str = Field(..., description="Score interpretation")
    timeline: List[TimelineEvent] = Field(..., description="Timeline of key events")
    confidence_level: str = Field(..., description="Confidence level")


class RankedWallet(BaseModel):
    """A ranked smart wallet result."""

    rank: int = Field(..., ge=1, description="Rank position")
    wallet_address: str = Field(..., description="Wallet address")
    smart_money_score: float = Field(..., ge=0, le=1, description="Smart-money probability score")
    key_metrics: KeyMetrics = Field(..., description="Key performance metrics")
    explanation: WalletExplanation = Field(..., description="Score explanation")
    risk_score: float = Field(..., ge=0, le=1, description="Risk score")


class AnalysisMetadata(BaseModel):
    """Metadata about the analysis."""

    total_wallets_analyzed: int = Field(..., ge=0, description="Total wallets analyzed")
    wallets_passing_filters: int = Field(..., ge=0, description="Wallets passing filters")
    wallets_returned: int = Field(..., ge=0, description="Wallets in result")
    confidence_threshold: float = Field(..., ge=0, le=1, description="Confidence threshold used")


class SummaryStats(BaseModel):
    """Summary statistics."""

    avg_smart_money_score: float = Field(..., ge=0, le=1, description="Average score")
    median_smart_money_score: float = Field(..., ge=0, le=1, description="Median score")
    avg_win_rate: float = Field(..., ge=0, le=1, description="Average win rate")
    total_smart_wallets: int = Field(..., ge=0, description="Total smart wallets found")


class SmartWalletFindResponse(BaseModel):
    """Response from find smart wallets endpoint."""

    token_address: str = Field(..., description="Token address analyzed")
    chain: str = Field(..., description="Blockchain network")
    analysis_window_days: int = Field(..., description="Analysis window in days")
    timestamp: str = Field(..., description="Analysis timestamp (ISO 8601)")
    ranked_wallets: List[RankedWallet] = Field(..., description="Ranked list of smart wallets")
    summary_stats: SummaryStats = Field(..., description="Summary statistics")
    metadata: AnalysisMetadata = Field(..., description="Analysis metadata")


class Trade(BaseModel):
    """Individual trade."""

    timestamp: str = Field(..., description="Trade timestamp")
    side: str = Field(..., description="Trade side (buy/sell)")
    amount_usd: float = Field(..., description="Trade amount in USD")
    price: float = Field(..., description="Token price")
    pnl: Optional[float] = Field(None, description="Realized PnL")
    roi: Optional[float] = Field(None, description="ROI percentage")


class WalletProfileResponse(BaseModel):
    """Response from wallet profile endpoint."""

    wallet_address: str = Field(..., description="Wallet address")
    token_address: str = Field(..., description="Token address")
    chain: str = Field(..., description="Blockchain network")
    score: float = Field(..., ge=0, le=1, description="Smart-money score")
    trades: List[Trade] = Field(..., description="List of trades")
    realized_roi: float = Field(..., description="Realized ROI")
    win_rate: float = Field(..., ge=0, le=1, description="Win rate")
    graph_neighbors: List[str] = Field(..., description="Graph neighbor wallets")
    explanation: WalletExplanation = Field(..., description="Score explanation")
    timestamp: str = Field(..., description="Analysis timestamp")


class TokenWalletSummary(BaseModel):
    """Summary of smart wallets for a token."""

    token_address: str = Field(..., description="Token address")
    top_wallets: List[RankedWallet] = Field(..., description="Top smart wallets")
    avg_score: float = Field(..., ge=0, le=1, description="Average smart-money score")


class BulkScanResponse(BaseModel):
    """Response from bulk scan endpoint."""

    chain: str = Field(..., description="Blockchain network")
    tokens_analyzed: int = Field(..., ge=0, description="Number of tokens analyzed")
    timestamp: str = Field(..., description="Analysis timestamp")
    results: List[TokenWalletSummary] = Field(..., description="Results per token")
    leaderboard: List[RankedWallet] = Field(..., description="Global leaderboard across all tokens")
