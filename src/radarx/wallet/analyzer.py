"""
Wallet Analyzer

Calculates win rates, PnL metrics, and performance statistics for wallet trading history.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np


class WalletAnalyzer:
    """Analyzes wallet trading performance and calculates metrics."""

    def __init__(self):
        """Initialize wallet analyzer."""
        pass

    def calculate_win_rate(
        self, trades: List[Dict[str, Any]], timeframes: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate win rate across multiple timeframes.

        Args:
            trades: List of trade dictionaries with 'timestamp', 'pnl', 'token'
            timeframes: List of timeframes ['1d', '7d', '30d', 'all']

        Returns:
            Dictionary with win rates per timeframe
        """
        if timeframes is None:
            timeframes = ["1d", "7d", "30d", "all"]

        now = datetime.now(timezone.utc)
        results = {}

        for tf in timeframes:
            # Filter trades by timeframe
            if tf == "all":
                filtered_trades = trades
            else:
                days = int(tf.replace("d", ""))
                cutoff = now - timedelta(days=days)
                filtered_trades = [
                    t
                    for t in trades
                    if datetime.fromisoformat(t["timestamp"].replace("Z", "+00:00")) >= cutoff
                ]

            if not filtered_trades:
                results[tf] = {"win_rate": 0.0, "profitable_trades": 0, "total_trades": 0}
                continue

            profitable = sum(1 for t in filtered_trades if t.get("pnl", 0) > 0)
            total = len(filtered_trades)

            results[tf] = {
                "win_rate": profitable / total if total > 0 else 0.0,
                "profitable_trades": profitable,
                "total_trades": total,
            }

        return results

    def calculate_realized_pnl(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate realized PnL from closed trades.

        Args:
            trades: List of completed trades with 'pnl' field

        Returns:
            Dictionary with PnL statistics
        """
        if not trades:
            return {
                "total": 0.0,
                "average": 0.0,
                "best": 0.0,
                "worst": 0.0,
                "total_volume_buy": 0.0,
                "total_volume_sell": 0.0,
            }

        pnls = [t.get("pnl", 0) for t in trades]

        return {
            "total": sum(pnls),
            "average": np.mean(pnls) if pnls else 0.0,
            "best": max(pnls) if pnls else 0.0,
            "worst": min(pnls) if pnls else 0.0,
            "total_volume_buy": sum(t.get("buy_amount", 0) for t in trades),
            "total_volume_sell": sum(t.get("sell_amount", 0) for t in trades),
        }

    def calculate_unrealized_pnl(
        self, current_positions: List[Dict[str, Any]], current_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate unrealized PnL for open positions.

        Args:
            current_positions: List of open positions with 'token', 'amount', 'avg_buy_price'
            current_prices: Dictionary mapping token addresses to current prices

        Returns:
            Dictionary with unrealized PnL statistics
        """
        if not current_positions:
            return {"total": 0.0, "positions": []}

        position_pnls = []
        total_unrealized = 0.0

        for pos in current_positions:
            token = pos["token"]
            amount = pos.get("amount", 0)
            avg_buy_price = pos.get("avg_buy_price", 0)
            current_price = current_prices.get(token, avg_buy_price)

            unrealized_pnl = (current_price - avg_buy_price) * amount
            total_unrealized += unrealized_pnl

            position_pnls.append(
                {
                    "token": token,
                    "unrealized_pnl": unrealized_pnl,
                    "current_value": current_price * amount,
                    "cost_basis": avg_buy_price * amount,
                }
            )

        return {"total": total_unrealized, "positions": position_pnls}

    def calculate_performance_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate advanced performance metrics.

        Args:
            trades: List of trades with timestamps and PnL

        Returns:
            Dictionary with performance metrics
        """
        if not trades or len(trades) < 2:
            return {
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "avg_hold_time_hours": 0.0,
                "trade_frequency_per_day": 0.0,
            }

        # Calculate returns
        pnls = [t.get("pnl", 0) for t in trades]
        returns = np.array(pnls)

        # Sharpe ratio (annualized)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(365)
        else:
            sharpe = 0.0

        # Max drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = np.max(drawdown) / max(running_max[-1], 1) if len(running_max) > 0 else 0.0

        # Hold time
        hold_times = []
        for trade in trades:
            if "buy_timestamp" in trade and "sell_timestamp" in trade:
                buy_time = datetime.fromisoformat(trade["buy_timestamp"].replace("Z", "+00:00"))
                sell_time = datetime.fromisoformat(trade["sell_timestamp"].replace("Z", "+00:00"))
                hold_hours = (sell_time - buy_time).total_seconds() / 3600
                hold_times.append(hold_hours)

        avg_hold_time = np.mean(hold_times) if hold_times else 0.0

        # Trade frequency
        if trades:
            first_trade = datetime.fromisoformat(trades[0]["timestamp"].replace("Z", "+00:00"))
            last_trade = datetime.fromisoformat(trades[-1]["timestamp"].replace("Z", "+00:00"))
            days = max((last_trade - first_trade).days, 1)
            frequency = len(trades) / days
        else:
            frequency = 0.0

        return {
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "avg_hold_time_hours": avg_hold_time,
            "trade_frequency_per_day": frequency,
        }

    def get_token_breakdown(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get per-token performance breakdown.

        Args:
            trades: List of trades with 'token' and 'pnl'

        Returns:
            List of token performance dictionaries
        """
        token_stats = {}

        for trade in trades:
            token = trade.get("token")
            if not token:
                continue

            if token not in token_stats:
                token_stats[token] = {"token": token, "total_pnl": 0.0, "trades": 0, "wins": 0}

            pnl = trade.get("pnl", 0)
            token_stats[token]["total_pnl"] += pnl
            token_stats[token]["trades"] += 1
            if pnl > 0:
                token_stats[token]["wins"] += 1

        # Calculate win rate and ROI per token
        breakdown = []
        for token, stats in token_stats.items():
            stats["win_rate"] = stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0.0
            breakdown.append(stats)

        # Sort by total PnL descending
        breakdown.sort(key=lambda x: x["total_pnl"], reverse=True)

        return breakdown

    def get_chain_breakdown(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get per-chain performance breakdown.

        Args:
            trades: List of trades with 'chain' and 'pnl'

        Returns:
            List of chain performance dictionaries
        """
        chain_stats = {}

        for trade in trades:
            chain = trade.get("chain", "unknown")

            if chain not in chain_stats:
                chain_stats[chain] = {"chain": chain, "total_pnl": 0.0, "trades": 0, "wins": 0}

            pnl = trade.get("pnl", 0)
            chain_stats[chain]["total_pnl"] += pnl
            chain_stats[chain]["trades"] += 1
            if pnl > 0:
                chain_stats[chain]["wins"] += 1

        # Calculate win rate per chain
        breakdown = []
        for chain, stats in chain_stats.items():
            stats["win_rate"] = stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0.0
            breakdown.append(stats)

        # Sort by total PnL descending
        breakdown.sort(key=lambda x: x["total_pnl"], reverse=True)

        return breakdown
