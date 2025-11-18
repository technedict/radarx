"""Pydantic models for wallet analytics and reporting."""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Timeframe(BaseModel):
    """Time range for wallet analysis."""

    from_date: datetime = Field(..., alias="from")
    to_date: datetime = Field(..., alias="to")
    period: Literal["1d", "7d", "30d", "all-time"]

    model_config = ConfigDict(populate_by_name=True)


class WinRateByTimeframe(BaseModel):
    """Win rates broken down by timeframe."""

    one_day: Optional[float] = Field(None, alias="1d", ge=0, le=1)
    seven_day: Optional[float] = Field(None, alias="7d", ge=0, le=1)
    thirty_day: Optional[float] = Field(None, alias="30d", ge=0, le=1)
    all_time: Optional[float] = Field(None, ge=0, le=1)

    model_config = ConfigDict(populate_by_name=True)


class WinRate(BaseModel):
    """Win rate statistics."""

    overall: float = Field(..., ge=0, le=1, description="Overall win rate")
    by_timeframe: WinRateByTimeframe
    profitable_trades: int = Field(..., ge=0)
    total_trades: int = Field(..., ge=0)


class RealizedPnL(BaseModel):
    """Realized profit and loss statistics."""

    total_usd: float
    average_per_trade_usd: float
    best_trade_usd: float
    worst_trade_usd: float


class UnrealizedPnLByToken(BaseModel):
    """Unrealized PnL for a specific token position."""

    token_address: str
    token_symbol: str
    unrealized_pnl_usd: float
    entry_value_usd: float
    current_value_usd: float


class UnrealizedPnL(BaseModel):
    """Unrealized profit and loss for open positions."""

    total_usd: float
    by_token: List[UnrealizedPnLByToken] = Field(default_factory=list)


class TotalVolume(BaseModel):
    """Trading volume statistics."""

    buy_volume_usd: float
    sell_volume_usd: float
    total_volume_usd: float


class PnLSummary(BaseModel):
    """Complete PnL summary."""

    realized_pnl: RealizedPnL
    unrealized_pnl: Optional[UnrealizedPnL] = None
    total_volume: TotalVolume


class PerformanceMetrics(BaseModel):
    """Advanced performance metrics."""

    average_trade_duration_hours: float
    average_profit_per_trade_usd: float
    trade_frequency_per_day: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None


class TokenBreakdown(BaseModel):
    """Performance breakdown for a specific token."""

    token_address: str
    token_symbol: str
    chain: str
    trade_count: int
    win_rate: float = Field(..., ge=0, le=1)
    total_pnl_usd: float
    average_roi: float
    volume_usd: float
    first_trade: datetime
    last_trade: datetime


class ChainBreakdown(BaseModel):
    """Performance breakdown by blockchain."""

    chain: str
    trade_count: int
    win_rate: float = Field(..., ge=0, le=1)
    total_pnl_usd: float
    volume_usd: float


class Trade(BaseModel):
    """Individual trade record."""

    trade_id: str
    token_address: str
    token_symbol: str
    chain: str
    entry_timestamp: datetime
    exit_timestamp: Optional[datetime] = None
    entry_price: float
    exit_price: Optional[float] = None
    quantity: str
    entry_value_usd: float
    exit_value_usd: Optional[float] = None
    pnl_usd: Optional[float] = None
    roi: Optional[float] = None
    duration_hours: Optional[float] = None
    status: Literal["open", "closed"]


class BehavioralPatterns(BaseModel):
    """Detected behavioral patterns and flags."""

    pattern_tags: List[str] = Field(
        default_factory=list, description="Tags like 'early_adopter', 'diamond_hands', etc."
    )
    is_bot_like: bool = False
    is_smart_money: bool = False
    copies_wallet: Optional[str] = Field(
        None, description="Address of wallet this wallet frequently copies"
    )
    wash_trading_score: float = Field(0.0, ge=0, le=1)


class RelatedWallet(BaseModel):
    """Related wallet with correlation info."""

    wallet_address: str
    relationship_type: Literal["fund_flow", "similar_pattern", "coordinated"]
    correlation_score: float
    shared_tokens: int


class Ranking(BaseModel):
    """Wallet ranking information."""

    global_rank: Optional[int] = None
    chain_rank: Optional[int] = None
    percentile: float


class WalletMetrics(BaseModel):
    """Core wallet performance metrics."""

    win_rate: WinRate
    pnl_summary: PnLSummary
    performance_metrics: Optional[PerformanceMetrics] = None


class WalletReport(BaseModel):
    """Complete wallet analytics report."""

    wallet_address: str
    chain: str
    timestamp: datetime
    timeframe: Timeframe
    win_rate: WinRate
    pnl_summary: PnLSummary
    performance_metrics: Optional[PerformanceMetrics] = None
    breakdown_by_token: List[TokenBreakdown] = Field(default_factory=list)
    breakdown_by_chain: List[ChainBreakdown] = Field(default_factory=list)
    trades: List[Trade] = Field(default_factory=list)
    behavioral_patterns: Optional[BehavioralPatterns] = None
    related_wallets: List[RelatedWallet] = Field(default_factory=list)
    ranking: Optional[Ranking] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "wallet_address": "0xabcdef1234567890abcdef1234567890abcdef12",
                "chain": "ethereum",
                "timestamp": "2024-01-15T10:30:00Z",
                "timeframe": {
                    "from": "2023-01-01T00:00:00Z",
                    "to": "2024-01-15T10:30:00Z",
                    "period": "all-time",
                },
                "win_rate": {
                    "overall": 0.65,
                    "by_timeframe": {"1d": 0.70, "7d": 0.68, "30d": 0.65, "all_time": 0.65},
                    "profitable_trades": 65,
                    "total_trades": 100,
                },
                "pnl_summary": {
                    "realized_pnl": {
                        "total_usd": 15000,
                        "average_per_trade_usd": 150,
                        "best_trade_usd": 5000,
                        "worst_trade_usd": -1000,
                    },
                    "total_volume": {
                        "buy_volume_usd": 50000,
                        "sell_volume_usd": 65000,
                        "total_volume_usd": 115000,
                    },
                },
            }
        }
    )
