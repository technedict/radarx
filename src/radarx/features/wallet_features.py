"""Wallet-level feature extraction."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from radarx.data import DataNormalizer

logger = logging.getLogger(__name__)


class WalletFeatureExtractor:
    """Extract wallet behavioral and performance features."""

    async def extract_trading_features(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract trading pattern features.

        Args:
            trades: List of trade data

        Returns:
            Dict of trading features
        """
        features = {}

        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "avg_hold_time_hours": 0,
                "trade_frequency_per_day": 0,
            }

        # Basic stats
        features["total_trades"] = len(trades)

        # Win rate
        profitable = sum(1 for t in trades if DataNormalizer.clean_numeric(t.get("pnl", 0)) > 0)
        features["win_rate"] = profitable / len(trades)

        # Average hold time
        hold_times = []
        for trade in trades:
            if "entry_time" in trade and "exit_time" in trade:
                entry = DataNormalizer.normalize_timestamp(trade["entry_time"])
                exit = DataNormalizer.normalize_timestamp(trade["exit_time"])
                hold_time = (exit - entry).total_seconds() / 3600  # hours
                hold_times.append(hold_time)

        features["avg_hold_time_hours"] = np.mean(hold_times) if hold_times else 0

        # Trade frequency
        if trades:
            first_trade = min(
                DataNormalizer.normalize_timestamp(t.get("timestamp", datetime.utcnow()))
                for t in trades
            )
            last_trade = max(
                DataNormalizer.normalize_timestamp(t.get("timestamp", datetime.utcnow()))
                for t in trades
            )
            days = (last_trade - first_trade).days or 1
            features["trade_frequency_per_day"] = len(trades) / days
        else:
            features["trade_frequency_per_day"] = 0

        return features

    async def extract_pnl_features(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract PnL-related features.

        Args:
            trades: List of trade data

        Returns:
            Dict of PnL features
        """
        features = {}

        if not trades:
            return {
                "total_pnl": 0,
                "avg_pnl_per_trade": 0,
                "best_trade": 0,
                "worst_trade": 0,
            }

        pnls = [DataNormalizer.clean_numeric(t.get("pnl", 0)) for t in trades]

        features["total_pnl"] = sum(pnls)
        features["avg_pnl_per_trade"] = np.mean(pnls)
        features["best_trade"] = max(pnls)
        features["worst_trade"] = min(pnls)

        # Risk metrics
        if len(pnls) > 1:
            features["pnl_std"] = np.std(pnls)
            features["sharpe_ratio"] = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0
        else:
            features["pnl_std"] = 0
            features["sharpe_ratio"] = 0

        return features

    async def extract_all_features(self, wallet_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract all wallet features.

        Args:
            wallet_data: Wallet data including trades

        Returns:
            Complete feature dictionary
        """
        all_features = {}

        trades = wallet_data.get("trades", [])

        trading_features = await self.extract_trading_features(trades)
        all_features.update(trading_features)

        pnl_features = await self.extract_pnl_features(trades)
        all_features.update(pnl_features)

        return all_features
