"""
Outcome labeler for generating training labels from historical data.

Labels tokens that reached specific multiplier targets within time horizons,
handling censored data where targets were not reached.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class OutcomeLabel:
    """Label for a single token at a specific timestamp."""

    token_address: str
    timestamp: datetime
    labels: Dict[str, Dict[str, bool]]  # {horizon: {multiplier: reached}}
    time_to_target: Dict[str, Dict[str, Optional[float]]]  # hours to reach target
    censored: Dict[str, Dict[str, bool]]  # whether observation was censored
    max_price_reached: Dict[str, float]  # max price in each horizon
    final_liquidity: Dict[str, float]  # liquidity at end of horizon


class OutcomeLabeler:
    """
    Labels historical token data with outcome targets.

    For each token at each timestamp, determines if it reached various
    multiplier targets (2x, 5x, 10x, etc.) within specific time horizons
    (24h, 7d, 30d), accounting for liquidity and volume requirements.
    """

    def __init__(
        self,
        multipliers: List[str] = None,
        horizons: List[str] = None,
        min_liquidity_usd: float = 50000.0,
        min_volume_24h: float = 10000.0,
        require_sustained: bool = True,
        sustain_duration_hours: float = 1.0,
    ):
        """
        Initialize outcome labeler.

        Args:
            multipliers: List of multiplier targets (e.g., ['2x', '5x', '10x'])
            horizons: List of time horizons (e.g., ['24h', '7d', '30d'])
            min_liquidity_usd: Minimum liquidity to consider valid
            min_volume_24h: Minimum 24h volume to consider valid
            require_sustained: Whether target must be sustained
            sustain_duration_hours: Hours price must stay above target
        """
        self.multipliers = multipliers or ["2x", "5x", "10x", "20x", "50x"]
        self.horizons = horizons or ["24h", "7d", "30d"]
        self.min_liquidity_usd = min_liquidity_usd
        self.min_volume_24h = min_volume_24h
        self.require_sustained = require_sustained
        self.sustain_duration_hours = sustain_duration_hours

        # Parse horizon durations
        self.horizon_hours = {}
        for horizon in self.horizons:
            if horizon.endswith("h"):
                self.horizon_hours[horizon] = int(horizon[:-1])
            elif horizon.endswith("d"):
                self.horizon_hours[horizon] = int(horizon[:-1]) * 24
            else:
                raise ValueError(f"Invalid horizon format: {horizon}")

    def label_outcomes(
        self,
        token_data: List[Dict],
        start_timestamp: Optional[datetime] = None,
        end_timestamp: Optional[datetime] = None,
    ) -> List[OutcomeLabel]:
        """
        Label outcomes for historical token data.

        Args:
            token_data: List of dicts with token price/liquidity history
                Each dict should have: token_address, timestamp, price,
                liquidity_usd, volume_24h
            start_timestamp: Start of labeling period
            end_timestamp: End of labeling period

        Returns:
            List of OutcomeLabel objects
        """
        # Sort data by timestamp
        sorted_data = sorted(token_data, key=lambda x: x["timestamp"])

        # Group by token
        token_groups = {}
        for data in sorted_data:
            addr = data["token_address"]
            if addr not in token_groups:
                token_groups[addr] = []
            token_groups[addr].append(data)

        # Generate labels
        labels = []
        for token_addr, token_history in token_groups.items():
            token_labels = self._label_token(
                token_addr, token_history, start_timestamp, end_timestamp
            )
            labels.extend(token_labels)

        return labels

    def _label_token(
        self,
        token_address: str,
        price_history: List[Dict],
        start_timestamp: Optional[datetime],
        end_timestamp: Optional[datetime],
    ) -> List[OutcomeLabel]:
        """Label outcomes for a single token."""
        labels = []

        for i, data_point in enumerate(price_history):
            timestamp = data_point["timestamp"]

            # Skip if outside time range
            if start_timestamp and timestamp < start_timestamp:
                continue
            if end_timestamp and timestamp > end_timestamp:
                continue

            # Skip if insufficient liquidity/volume
            if data_point.get("liquidity_usd", 0) < self.min_liquidity_usd:
                continue
            if data_point.get("volume_24h", 0) < self.min_volume_24h:
                continue

            # Get future price action for each horizon
            entry_price = data_point["price"]
            horizon_labels = {}
            time_to_target = {}
            censored = {}
            max_prices = {}
            final_liquidity = {}

            for horizon in self.horizons:
                horizon_hours = self.horizon_hours[horizon]
                end_time = timestamp + timedelta(hours=horizon_hours)

                # Get price history within horizon
                future_prices = [
                    (p["timestamp"], p["price"], p.get("liquidity_usd", 0))
                    for p in price_history[i:]
                    if timestamp <= p["timestamp"] <= end_time
                ]

                # Check each multiplier
                multiplier_labels = {}
                multiplier_times = {}
                multiplier_censored = {}

                for multiplier in self.multipliers:
                    mult_value = float(multiplier.replace("x", ""))
                    target_price = entry_price * mult_value

                    reached, time_hours, is_censored = self._check_target_reached(
                        future_prices, target_price, timestamp
                    )

                    multiplier_labels[multiplier] = reached
                    multiplier_times[multiplier] = time_hours
                    multiplier_censored[multiplier] = is_censored

                horizon_labels[horizon] = multiplier_labels
                time_to_target[horizon] = multiplier_times
                censored[horizon] = multiplier_censored

                # Track max price and final liquidity
                if future_prices:
                    max_prices[horizon] = max(p[1] for p in future_prices)
                    final_liquidity[horizon] = future_prices[-1][2]
                else:
                    max_prices[horizon] = entry_price
                    final_liquidity[horizon] = data_point.get("liquidity_usd", 0)

            # Create label
            label = OutcomeLabel(
                token_address=token_address,
                timestamp=timestamp,
                labels=horizon_labels,
                time_to_target=time_to_target,
                censored=censored,
                max_price_reached=max_prices,
                final_liquidity=final_liquidity,
            )
            labels.append(label)

        return labels

    def _check_target_reached(
        self,
        future_prices: List[Tuple[datetime, float, float]],
        target_price: float,
        entry_time: datetime,
    ) -> Tuple[bool, Optional[float], bool]:
        """
        Check if target price was reached and sustained.

        Returns:
            (reached, time_to_target_hours, is_censored)
        """
        if not future_prices:
            return False, None, True

        # Find first time target was reached
        first_hit_idx = None
        for i, (ts, price, liq) in enumerate(future_prices):
            if price >= target_price:
                # Check liquidity is still sufficient
                if liq < self.min_liquidity_usd:
                    continue
                first_hit_idx = i
                break

        if first_hit_idx is None:
            # Target never reached (or liquidity dried up)
            return False, None, False

        # Check if sustained (if required)
        if self.require_sustained:
            first_hit_time, first_hit_price, _ = future_prices[first_hit_idx]
            sustain_end = first_hit_time + timedelta(hours=self.sustain_duration_hours)

            # Check prices remain above target during sustain period
            sustained = True
            for ts, price, liq in future_prices[first_hit_idx:]:
                if ts > sustain_end:
                    break
                if price < target_price or liq < self.min_liquidity_usd:
                    sustained = False
                    break

            if not sustained:
                # Target was hit but not sustained
                return False, None, False

        # Calculate time to target
        first_hit_time = future_prices[first_hit_idx][0]
        time_hours = (first_hit_time - entry_time).total_seconds() / 3600

        return True, time_hours, False

    def get_label_quality_metrics(self, labels: List[OutcomeLabel]) -> Dict:
        """
        Calculate quality metrics for generated labels.

        Returns dict with hit rates, censoring rates, average time to target,
        etc. for each (horizon, multiplier) combination.
        """
        metrics = {}

        for horizon in self.horizons:
            metrics[horizon] = {}

            for multiplier in self.multipliers:
                # Count outcomes
                total = len(labels)
                if total == 0:
                    continue

                reached_count = sum(
                    1 for label in labels if label.labels.get(horizon, {}).get(multiplier, False)
                )
                censored_count = sum(
                    1 for label in labels if label.censored.get(horizon, {}).get(multiplier, False)
                )

                # Calculate time to target statistics
                times = [
                    label.time_to_target.get(horizon, {}).get(multiplier)
                    for label in labels
                    if label.time_to_target.get(horizon, {}).get(multiplier) is not None
                ]

                metrics[horizon][multiplier] = {
                    "hit_rate": reached_count / total,
                    "num_reached": reached_count,
                    "censoring_rate": censored_count / total,
                    "num_censored": censored_count,
                    "total_samples": total,
                    "avg_time_hours": sum(times) / len(times) if times else None,
                    "min_time_hours": min(times) if times else None,
                    "max_time_hours": max(times) if times else None,
                }

        return metrics

    def to_training_format(
        self, labels: List[OutcomeLabel], features: Dict[str, Dict]  # token_address -> feature dict
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Convert labels to training format.

        Args:
            labels: List of outcome labels
            features: Dict mapping token_address to feature dict

        Returns:
            (X, y) where X is list of feature dicts and y is list of label dicts
        """
        X = []
        y = []

        for label in labels:
            if label.token_address not in features:
                continue

            # Get features at this timestamp
            feature_dict = features[label.token_address]

            # Create label dict
            label_dict = {}
            for horizon in self.horizons:
                for multiplier in self.multipliers:
                    key = f"{horizon}_{multiplier}"
                    label_dict[key] = label.labels.get(horizon, {}).get(multiplier, False)

                    # Add censoring flag for survival analysis
                    label_dict[f"{key}_censored"] = label.censored.get(horizon, {}).get(
                        multiplier, False
                    )

                    # Add time to event for survival analysis
                    label_dict[f"{key}_time"] = label.time_to_target.get(horizon, {}).get(
                        multiplier
                    )

            X.append(feature_dict)
            y.append(label_dict)

        return X, y

    def export_labels(self, labels: List[OutcomeLabel], filepath: str, format: str = "json"):
        """Export labels to file."""
        import json

        if format == "json":
            label_dicts = [
                {
                    "token_address": label.token_address,
                    "timestamp": label.timestamp.isoformat(),
                    "labels": label.labels,
                    "time_to_target": label.time_to_target,
                    "censored": label.censored,
                    "max_price_reached": label.max_price_reached,
                    "final_liquidity": label.final_liquidity,
                }
                for label in labels
            ]

            with open(filepath, "w") as f:
                json.dump(label_dicts, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
