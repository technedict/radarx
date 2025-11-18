"""Time-windowed feature aggregation."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class TimeWindowAggregator:
    """Aggregate features across multiple time windows."""

    STANDARD_WINDOWS = ["1h", "6h", "24h", "7d", "30d"]

    async def aggregate_features(
        self, entity_id: str, feature_history: List[Dict[str, Any]], windows: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate features across time windows.

        Args:
            entity_id: Entity identifier
            feature_history: List of historical feature snapshots with timestamps
            windows: Time windows to aggregate over

        Returns:
            Dict mapping window -> aggregated features
        """
        windows = windows or self.STANDARD_WINDOWS
        aggregated = {}

        now = datetime.now(timezone.utc)

        for window in windows:
            # Parse window string (e.g., "24h", "7d")
            window_seconds = self._parse_window(window)
            cutoff_time = now - timedelta(seconds=window_seconds)

            # Filter features within window
            window_features = [
                f for f in feature_history if f.get("timestamp", datetime.min) >= cutoff_time
            ]

            # Aggregate
            if window_features:
                aggregated[window] = self._aggregate_window_features(window_features)
            else:
                aggregated[window] = {}

        return aggregated

    def _parse_window(self, window: str) -> int:
        """Parse window string to seconds.

        Args:
            window: Window string like "1h", "24h", "7d"

        Returns:
            Number of seconds
        """
        value = int(window[:-1])
        unit = window[-1]

        if unit == "h":
            return value * 3600
        elif unit == "d":
            return value * 86400
        else:
            raise ValueError(f"Unknown time unit: {unit}")

    def _aggregate_window_features(self, features: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate features within a window.

        Args:
            features: List of feature dicts

        Returns:
            Aggregated features
        """
        if not features:
            return {}

        # Extract feature values
        all_keys = set()
        for f in features:
            all_keys.update(f.get("features", {}).keys())

        aggregated = {}
        for key in all_keys:
            values = [
                f.get("features", {}).get(key, 0) for f in features if key in f.get("features", {})
            ]

            if values:
                aggregated[f"{key}_mean"] = sum(values) / len(values)
                aggregated[f"{key}_max"] = max(values)
                aggregated[f"{key}_min"] = min(values)

        return aggregated
