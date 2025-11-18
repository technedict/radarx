"""
Signal Detection Modules

Implements various detection signals for smart wallet identification:
- Timing signals (pre-pump/pre-dump detection)
- Profitability analysis
- Graph analysis
- Behavioral pattern detection
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TimingSignalDetector:
    """
    Detects event timing signals - wallets that bought before pumps or sold before dumps.
    """

    def __init__(
        self,
        pre_pump_window_minutes: int = 60,
        pre_dump_window_minutes: int = 60,
        pump_threshold_pct: float = 20.0,
        dump_threshold_pct: float = -15.0,
    ):
        """
        Initialize timing signal detector.

        Args:
            pre_pump_window_minutes: Lead time window before price pump
            pre_dump_window_minutes: Lead time window before price dump
            pump_threshold_pct: Minimum price increase to qualify as pump
            dump_threshold_pct: Maximum price decrease to qualify as dump
        """
        self.pre_pump_window = timedelta(minutes=pre_pump_window_minutes)
        self.pre_dump_window = timedelta(minutes=pre_dump_window_minutes)
        self.pump_threshold = pump_threshold_pct / 100.0
        self.dump_threshold = dump_threshold_pct / 100.0

    def detect(
        self,
        trades: List[Dict[str, Any]],
        price_timeline: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Detect timing signals for wallet trades.

        Args:
            trades: List of wallet trades
            price_timeline: Price timeline with timestamps and prices

        Returns:
            Dictionary with timing signals:
                - pre_pump_buys: Number of buys before pumps
                - pre_dump_sells: Number of sells before dumps
                - pre_pump_entry_rate: Ratio of pre-pump buys to total buys
                - pre_dump_exit_rate: Ratio of pre-dump sells to total sells
                - avg_entry_timing_minutes: Average minutes before peak
                - avg_exit_timing_minutes: Average minutes before trough
        """
        if not trades or not price_timeline:
            return self._empty_signals()

        # Identify pumps and dumps in price timeline
        pumps = self._identify_pumps(price_timeline)
        dumps = self._identify_dumps(price_timeline)

        # Analyze buy timing relative to pumps
        buys = [t for t in trades if t.get("side") == "buy"]
        pre_pump_buys = self._count_pre_event_trades(buys, pumps, self.pre_pump_window)

        # Analyze sell timing relative to dumps
        sells = [t for t in trades if t.get("side") == "sell"]
        pre_dump_sells = self._count_pre_event_trades(sells, dumps, self.pre_dump_window)

        # Calculate timing metrics
        entry_timings = self._calculate_time_to_peaks(buys, price_timeline)
        exit_timings = self._calculate_time_to_troughs(sells, price_timeline)

        return {
            "pre_pump_buys": pre_pump_buys,
            "pre_dump_sells": pre_dump_sells,
            "total_buys": len(buys),
            "total_sells": len(sells),
            "pre_pump_entry_rate": pre_pump_buys / len(buys) if buys else 0.0,
            "pre_dump_exit_rate": pre_dump_sells / len(sells) if sells else 0.0,
            "avg_entry_timing_minutes": np.mean(entry_timings) if entry_timings else 0.0,
            "avg_exit_timing_minutes": np.mean(exit_timings) if exit_timings else 0.0,
            "pumps_detected": len(pumps),
            "dumps_detected": len(dumps),
        }

    def _identify_pumps(
        self,
        price_timeline: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Identify pump events in price timeline."""
        pumps = []

        for i in range(1, len(price_timeline)):
            prev_price = price_timeline[i - 1].get("price", 0)
            curr_price = price_timeline[i].get("price", 0)

            if prev_price > 0:
                price_change = (curr_price - prev_price) / prev_price

                if price_change >= self.pump_threshold:
                    pumps.append(
                        {
                            "timestamp": price_timeline[i].get("timestamp"),
                            "price_change_pct": price_change * 100,
                            "price_before": prev_price,
                            "price_after": curr_price,
                        }
                    )

        return pumps

    def _identify_dumps(
        self,
        price_timeline: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Identify dump events in price timeline."""
        dumps = []

        for i in range(1, len(price_timeline)):
            prev_price = price_timeline[i - 1].get("price", 0)
            curr_price = price_timeline[i].get("price", 0)

            if prev_price > 0:
                price_change = (curr_price - prev_price) / prev_price

                if price_change <= self.dump_threshold:
                    dumps.append(
                        {
                            "timestamp": price_timeline[i].get("timestamp"),
                            "price_change_pct": price_change * 100,
                            "price_before": prev_price,
                            "price_after": curr_price,
                        }
                    )

        return dumps

    def _count_pre_event_trades(
        self,
        trades: List[Dict[str, Any]],
        events: List[Dict[str, Any]],
        window: timedelta,
    ) -> int:
        """Count trades that occurred within window before events."""
        count = 0

        for trade in trades:
            trade_time = self._parse_timestamp(trade.get("timestamp"))

            for event in events:
                event_time = self._parse_timestamp(event.get("timestamp"))
                time_diff = event_time - trade_time

                # Trade is before event and within window
                if timedelta(0) <= time_diff <= window:
                    count += 1
                    break  # Only count once per trade

        return count

    def _calculate_time_to_peaks(
        self,
        trades: List[Dict[str, Any]],
        price_timeline: List[Dict[str, Any]],
    ) -> List[float]:
        """Calculate minutes from each buy to next price peak."""
        timings = []

        for trade in trades:
            trade_time = self._parse_timestamp(trade.get("timestamp"))

            # Find next peak after trade
            peak_time = self._find_next_peak(trade_time, price_timeline)

            if peak_time:
                minutes_to_peak = (peak_time - trade_time).total_seconds() / 60
                timings.append(minutes_to_peak)

        return timings

    def _calculate_time_to_troughs(
        self,
        trades: List[Dict[str, Any]],
        price_timeline: List[Dict[str, Any]],
    ) -> List[float]:
        """Calculate minutes from each sell to next price trough."""
        timings = []

        for trade in trades:
            trade_time = self._parse_timestamp(trade.get("timestamp"))

            # Find next trough after trade
            trough_time = self._find_next_trough(trade_time, price_timeline)

            if trough_time:
                minutes_to_trough = (trough_time - trade_time).total_seconds() / 60
                timings.append(minutes_to_trough)

        return timings

    def _find_next_peak(
        self,
        from_time: datetime,
        price_timeline: List[Dict[str, Any]],
    ) -> Optional[datetime]:
        """Find next local price peak after given time."""
        future_prices = [
            p for p in price_timeline if self._parse_timestamp(p.get("timestamp")) > from_time
        ]

        if len(future_prices) < 3:
            return None

        # Simple local maxima detection
        for i in range(1, len(future_prices) - 1):
            if future_prices[i].get("price", 0) > future_prices[i - 1].get(
                "price", 0
            ) and future_prices[i].get("price", 0) > future_prices[i + 1].get("price", 0):
                return self._parse_timestamp(future_prices[i].get("timestamp"))

        return None

    def _find_next_trough(
        self,
        from_time: datetime,
        price_timeline: List[Dict[str, Any]],
    ) -> Optional[datetime]:
        """Find next local price trough after given time."""
        future_prices = [
            p for p in price_timeline if self._parse_timestamp(p.get("timestamp")) > from_time
        ]

        if len(future_prices) < 3:
            return None

        # Simple local minima detection
        for i in range(1, len(future_prices) - 1):
            if future_prices[i].get("price", 0) < future_prices[i - 1].get(
                "price", 0
            ) and future_prices[i].get("price", 0) < future_prices[i + 1].get("price", 0):
                return self._parse_timestamp(future_prices[i].get("timestamp"))

        return None

    def _parse_timestamp(self, timestamp: Any) -> datetime:
        """Parse timestamp to datetime object."""
        if isinstance(timestamp, datetime):
            return timestamp
        if isinstance(timestamp, str):
            # Handle ISO format with Z
            ts_str = timestamp.replace("Z", "+00:00")
            return datetime.fromisoformat(ts_str)
        return datetime.now(timezone.utc)

    def _empty_signals(self) -> Dict[str, Any]:
        """Return empty signals structure."""
        return {
            "pre_pump_buys": 0,
            "pre_dump_sells": 0,
            "total_buys": 0,
            "total_sells": 0,
            "pre_pump_entry_rate": 0.0,
            "pre_dump_exit_rate": 0.0,
            "avg_entry_timing_minutes": 0.0,
            "avg_exit_timing_minutes": 0.0,
            "pumps_detected": 0,
            "dumps_detected": 0,
        }


class ProfitabilityAnalyzer:
    """
    Analyzes wallet profitability and trade behavior metrics.
    """

    def analyze(
        self,
        trades: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Analyze profitability metrics.

        Args:
            trades: List of wallet trades with PnL information

        Returns:
            Dictionary with profitability metrics:
                - win_rate: Percentage of profitable trades
                - avg_roi: Average return on investment
                - median_roi: Median ROI
                - total_trades: Number of trades
                - avg_holding_duration_hours: Average holding time
                - sharpe_ratio: Risk-adjusted returns
        """
        if not trades:
            return self._empty_metrics()

        # Calculate win rate
        profitable_trades = [t for t in trades if t.get("pnl", 0) > 0]
        win_rate = len(profitable_trades) / len(trades)

        # Calculate ROI metrics
        rois = [t.get("roi", 0) for t in trades if "roi" in t]
        avg_roi = np.mean(rois) if rois else 0.0
        median_roi = np.median(rois) if rois else 0.0

        # Calculate holding durations
        durations = []
        for trade in trades:
            if "entry_time" in trade and "exit_time" in trade:
                entry = self._parse_timestamp(trade["entry_time"])
                exit_time = self._parse_timestamp(trade["exit_time"])
                duration_hours = (exit_time - entry).total_seconds() / 3600
                durations.append(duration_hours)

        avg_duration = np.mean(durations) if durations else 0.0

        # Calculate Sharpe ratio (simplified)
        sharpe = self._calculate_sharpe_ratio(rois) if rois else 0.0

        return {
            "win_rate": win_rate,
            "avg_roi": avg_roi,
            "median_roi": median_roi,
            "total_trades": len(trades),
            "profitable_trades": len(profitable_trades),
            "avg_holding_duration_hours": avg_duration,
            "sharpe_ratio": sharpe,
            "best_roi": max(rois) if rois else 0.0,
            "worst_roi": min(rois) if rois else 0.0,
        }

    def _calculate_sharpe_ratio(self, rois: List[float]) -> float:
        """Calculate Sharpe ratio from ROI list."""
        if not rois or len(rois) < 2:
            return 0.0

        mean_roi = np.mean(rois)
        std_roi = np.std(rois)

        if std_roi == 0:
            return 0.0

        # Simplified Sharpe (assuming 0 risk-free rate)
        return mean_roi / std_roi

    def _parse_timestamp(self, timestamp: Any) -> datetime:
        """Parse timestamp to datetime object."""
        if isinstance(timestamp, datetime):
            return timestamp
        if isinstance(timestamp, str):
            ts_str = timestamp.replace("Z", "+00:00")
            return datetime.fromisoformat(ts_str)
        return datetime.now(timezone.utc)

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure."""
        return {
            "win_rate": 0.0,
            "avg_roi": 0.0,
            "median_roi": 0.0,
            "total_trades": 0,
            "profitable_trades": 0,
            "avg_holding_duration_hours": 0.0,
            "sharpe_ratio": 0.0,
            "best_roi": 0.0,
            "worst_roi": 0.0,
        }


class GraphAnalyzer:
    """
    Analyzes transaction graph features - fund flows, clusters, centrality.
    """

    def analyze(
        self,
        wallet_address: str,
        graph_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze graph features for a wallet.

        Args:
            wallet_address: Wallet address
            graph_data: Graph data with edges and node properties

        Returns:
            Dictionary with graph metrics:
                - centrality_score: Wallet centrality in transaction graph
                - clustering_coefficient: Local clustering coefficient
                - common_funding_sources: Number of shared funding sources
                - cluster_id: Community cluster assignment
                - connected_smart_wallets: Count of connections to known smart wallets
        """
        if not graph_data:
            return self._empty_metrics()

        nodes = graph_data.get("nodes", {})
        edges = graph_data.get("edges", [])

        # Calculate centrality (simplified degree centrality)
        centrality = self._calculate_centrality(wallet_address, edges)

        # Calculate clustering coefficient
        clustering = self._calculate_clustering(wallet_address, edges)

        # Find common funding sources
        funding_sources = self._find_funding_sources(wallet_address, edges)

        # Get cluster assignment
        clusters = graph_data.get("clusters", {})
        cluster_id = clusters.get(wallet_address, 0)

        # Count connections to known smart wallets
        smart_wallets = graph_data.get("smart_wallets", set())
        connected_smart = self._count_smart_connections(wallet_address, edges, smart_wallets)

        return {
            "centrality_score": centrality,
            "clustering_coefficient": clustering,
            "common_funding_sources": len(funding_sources),
            "cluster_id": cluster_id,
            "connected_smart_wallets": connected_smart,
            "total_connections": len(self._get_neighbors(wallet_address, edges)),
        }

    def _calculate_centrality(
        self,
        wallet: str,
        edges: List[Dict[str, Any]],
    ) -> float:
        """Calculate degree centrality for wallet."""
        neighbors = self._get_neighbors(wallet, edges)
        total_nodes = len(set([e.get("from") for e in edges] + [e.get("to") for e in edges]))

        if total_nodes <= 1:
            return 0.0

        return len(neighbors) / (total_nodes - 1)

    def _calculate_clustering(
        self,
        wallet: str,
        edges: List[Dict[str, Any]],
    ) -> float:
        """Calculate local clustering coefficient."""
        neighbors = self._get_neighbors(wallet, edges)

        if len(neighbors) < 2:
            return 0.0

        # Count edges between neighbors
        edges_between = 0
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i + 1 :]:
                if self._has_edge(n1, n2, edges):
                    edges_between += 1

        # Maximum possible edges between neighbors
        max_edges = len(neighbors) * (len(neighbors) - 1) / 2

        return edges_between / max_edges if max_edges > 0 else 0.0

    def _find_funding_sources(
        self,
        wallet: str,
        edges: List[Dict[str, Any]],
    ) -> List[str]:
        """Find wallets that sent funds to this wallet."""
        sources = set()

        for edge in edges:
            if edge.get("to") == wallet and edge.get("type") == "fund_flow":
                sources.add(edge.get("from"))

        return list(sources)

    def _count_smart_connections(
        self,
        wallet: str,
        edges: List[Dict[str, Any]],
        smart_wallets: set,
    ) -> int:
        """Count connections to known smart wallets."""
        neighbors = self._get_neighbors(wallet, edges)
        return len([n for n in neighbors if n in smart_wallets])

    def _get_neighbors(
        self,
        wallet: str,
        edges: List[Dict[str, Any]],
    ) -> List[str]:
        """Get all neighboring wallets."""
        neighbors = set()

        for edge in edges:
            if edge.get("from") == wallet:
                neighbors.add(edge.get("to"))
            elif edge.get("to") == wallet:
                neighbors.add(edge.get("from"))

        return list(neighbors)

    def _has_edge(
        self,
        wallet1: str,
        wallet2: str,
        edges: List[Dict[str, Any]],
    ) -> bool:
        """Check if edge exists between two wallets."""
        for edge in edges:
            if (edge.get("from") == wallet1 and edge.get("to") == wallet2) or (
                edge.get("from") == wallet2 and edge.get("to") == wallet1
            ):
                return True
        return False

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure."""
        return {
            "centrality_score": 0.0,
            "clustering_coefficient": 0.0,
            "common_funding_sources": 0,
            "cluster_id": 0,
            "connected_smart_wallets": 0,
            "total_connections": 0,
        }


class BehavioralAnalyzer:
    """
    Analyzes behavioral patterns and fingerprints.
    """

    def analyze(
        self,
        wallet_address: str,
        trades: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Analyze behavioral patterns.

        Args:
            wallet_address: Wallet address
            trades: List of trades

        Returns:
            Dictionary with behavioral metrics:
                - trade_frequency: Trades per day
                - preferred_trading_hours: Most common trading hours (UTC)
                - dex_diversity: Number of unique DEX protocols used
                - gas_pattern_consistency: Consistency in gas usage
                - pattern_tags: List of detected behavioral patterns
        """
        if not trades:
            return self._empty_metrics()

        # Calculate trade frequency
        frequency = self._calculate_trade_frequency(trades)

        # Find preferred trading hours
        hours = self._get_preferred_hours(trades)

        # Count unique DEX protocols
        dex_set = set(t.get("dex", "unknown") for t in trades)

        # Analyze gas patterns
        gas_consistency = self._analyze_gas_patterns(trades)

        # Detect pattern tags
        pattern_tags = self._detect_patterns(trades)

        return {
            "trade_frequency_per_day": frequency,
            "preferred_trading_hours": hours,
            "dex_diversity": len(dex_set),
            "gas_pattern_consistency": gas_consistency,
            "pattern_tags": pattern_tags,
            "unique_tokens_traded": len(set(t.get("token") for t in trades if "token" in t)),
        }

    def _calculate_trade_frequency(
        self,
        trades: List[Dict[str, Any]],
    ) -> float:
        """Calculate average trades per day."""
        if not trades:
            return 0.0

        timestamps = [self._parse_timestamp(t.get("timestamp")) for t in trades if "timestamp" in t]

        if len(timestamps) < 2:
            return 0.0

        min_time = min(timestamps)
        max_time = max(timestamps)
        days = (max_time - min_time).total_seconds() / 86400

        return len(trades) / days if days > 0 else 0.0

    def _get_preferred_hours(
        self,
        trades: List[Dict[str, Any]],
    ) -> List[int]:
        """Get most common trading hours."""
        hours = [self._parse_timestamp(t.get("timestamp")).hour for t in trades if "timestamp" in t]

        if not hours:
            return []

        # Find top 3 most common hours
        hour_counts = {}
        for h in hours:
            hour_counts[h] = hour_counts.get(h, 0) + 1

        sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
        return [h for h, _ in sorted_hours[:3]]

    def _analyze_gas_patterns(
        self,
        trades: List[Dict[str, Any]],
    ) -> float:
        """Analyze consistency in gas usage patterns."""
        gas_prices = [t.get("gas_price", 0) for t in trades if "gas_price" in t]

        if len(gas_prices) < 2:
            return 0.0

        # Calculate coefficient of variation (lower = more consistent)
        mean_gas = np.mean(gas_prices)
        std_gas = np.std(gas_prices)

        if mean_gas == 0:
            return 0.0

        cv = std_gas / mean_gas

        # Convert to consistency score (0-1, higher = more consistent)
        return max(0.0, 1.0 - min(cv, 1.0))

    def _detect_patterns(
        self,
        trades: List[Dict[str, Any]],
    ) -> List[str]:
        """Detect behavioral pattern tags."""
        patterns = []

        if not trades:
            return patterns

        # Early adopter: trades within first 24h of token launch
        early_trades = sum(1 for t in trades if t.get("token_age_hours", 999) < 24)
        if early_trades / len(trades) > 0.5:
            patterns.append("early_adopter")

        # Diamond hands: long average holding duration
        avg_duration = np.mean([t.get("holding_duration_hours", 0) for t in trades])
        if avg_duration > 168:  # 1 week
            patterns.append("diamond_hands")

        # Swing trader: medium holding duration
        if 1 < avg_duration < 168:
            patterns.append("swing_trader")

        # High frequency: many trades per day
        frequency = self._calculate_trade_frequency(trades)
        if frequency > 10:
            patterns.append("high_frequency")

        return patterns

    def _parse_timestamp(self, timestamp: Any) -> datetime:
        """Parse timestamp to datetime object."""
        if isinstance(timestamp, datetime):
            return timestamp
        if isinstance(timestamp, str):
            ts_str = timestamp.replace("Z", "+00:00")
            return datetime.fromisoformat(ts_str)
        return datetime.now(timezone.utc)

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure."""
        return {
            "trade_frequency_per_day": 0.0,
            "preferred_trading_hours": [],
            "dex_diversity": 0,
            "gas_pattern_consistency": 0.0,
            "pattern_tags": [],
            "unique_tokens_traded": 0,
        }
