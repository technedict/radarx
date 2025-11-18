"""Feature store with time-travel capability."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FeatureStore:
    """Store and retrieve features with time-travel capability."""

    def __init__(self):
        """Initialize feature store."""
        self.features: Dict[str, List[Dict[str, Any]]] = {}

    async def store_features(
        self,
        entity_id: str,
        features: Dict[str, float],
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Store features for an entity at a specific time.

        Args:
            entity_id: Unique entity identifier
            features: Feature dictionary
            timestamp: Timestamp (defaults to now)
            metadata: Optional metadata
        """
        timestamp = timestamp or datetime.utcnow()

        if entity_id not in self.features:
            self.features[entity_id] = []

        entry = {
            "timestamp": timestamp,
            "features": features,
            "metadata": metadata or {},
        }

        self.features[entity_id].append(entry)

        # Keep sorted by timestamp
        self.features[entity_id].sort(key=lambda x: x["timestamp"])

        logger.debug(f"Stored features for {entity_id} at {timestamp}")

    async def get_features(
        self, entity_id: str, timestamp: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Get features for an entity at a specific point in time.

        Args:
            entity_id: Entity identifier
            timestamp: Point in time (defaults to now)

        Returns:
            Feature dictionary
        """
        timestamp = timestamp or datetime.utcnow()

        if entity_id not in self.features:
            return {}

        # Find most recent features before or at timestamp
        entity_features = self.features[entity_id]

        for entry in reversed(entity_features):
            if entry["timestamp"] <= timestamp:
                return entry["features"]

        # No features found before timestamp
        return {}

    async def get_feature_history(
        self,
        entity_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get feature history for an entity.

        Args:
            entity_id: Entity identifier
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of feature entries
        """
        if entity_id not in self.features:
            return []

        entity_features = self.features[entity_id]

        # Filter by time range
        if start_time or end_time:
            filtered = []
            for entry in entity_features:
                ts = entry["timestamp"]
                if start_time and ts < start_time:
                    continue
                if end_time and ts > end_time:
                    continue
                filtered.append(entry)
            return filtered

        return entity_features

    async def list_entities(self) -> List[str]:
        """List all entity IDs in the feature store.

        Returns:
            List of entity IDs
        """
        return list(self.features.keys())

    async def delete_entity(self, entity_id: str):
        """Delete all features for an entity.

        Args:
            entity_id: Entity identifier
        """
        if entity_id in self.features:
            del self.features[entity_id]
            logger.info(f"Deleted features for {entity_id}")

    def get_stats(self) -> Dict[str, Any]:
        """Get feature store statistics.

        Returns:
            Statistics dictionary
        """
        total_entities = len(self.features)
        total_entries = sum(len(v) for v in self.features.values())

        return {
            "total_entities": total_entities,
            "total_entries": total_entries,
            "avg_entries_per_entity": total_entries / total_entities if total_entities > 0 else 0,
        }
