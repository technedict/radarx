"""
Learning ledger for tracking model versions, experiments, and performance.

Maintains a history of model training runs, backtest results, and
performance metrics for comparison and rollback.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Represents a single model version."""

    version_id: str
    version_name: str
    timestamp: datetime
    model_config: Dict[str, Any]
    training_data_info: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    # Performance metrics
    backtest_results: Optional[Dict] = None
    strategy_results: Optional[Dict] = None

    # Model artifacts
    model_path: Optional[str] = None
    feature_importance: Optional[Dict] = None


class LearningLedger:
    """
    Tracks model versions and their performance for comparison and rollback.

    Maintains a ledger of all model training runs, including configuration,
    performance metrics, and backtest results. Enables comparing versions
    and selecting the best model for deployment.
    """

    def __init__(self, ledger_path: Optional[str] = None):
        """
        Initialize learning ledger.

        Args:
            ledger_path: Path to save ledger data (optional)
        """
        self.ledger_path = ledger_path
        self.versions: Dict[str, ModelVersion] = {}

        # Load existing ledger if path provided
        if ledger_path and Path(ledger_path).exists():
            self.load()

    def log_model_version(
        self,
        model: Any,
        version: str,
        config: Dict[str, Any],
        training_data_info: Optional[Dict] = None,
        notes: str = "",
        tags: Optional[List[str]] = None,
        model_path: Optional[str] = None,
    ) -> str:
        """
        Log a new model version.

        Args:
            model: The trained model object
            version: Version string (e.g., 'v1.0.0')
            config: Model configuration dict
            training_data_info: Info about training data
            notes: Human-readable notes about this version
            tags: Tags for categorization
            model_path: Path where model is saved

        Returns:
            version_id: Unique identifier for this version
        """
        version_id = f"{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Extract feature importance if available
        feature_importance = self._extract_feature_importance(model)

        # Create version record
        model_version = ModelVersion(
            version_id=version_id,
            version_name=version,
            timestamp=datetime.now(),
            model_config=config,
            training_data_info=training_data_info or {},
            notes=notes,
            tags=tags or [],
            model_path=model_path,
            feature_importance=feature_importance,
        )

        self.versions[version_id] = model_version

        logger.info(f"Logged model version: {version_id}")

        # Auto-save if path configured
        if self.ledger_path:
            self.save()

        return version_id

    def log_backtest_result(
        self, version_id: str, backtest_metrics: Dict, strategy_results: Optional[Dict] = None
    ):
        """
        Log backtest results for a model version.

        Args:
            version_id: Model version identifier
            backtest_metrics: Dict of backtest metrics
            strategy_results: Dict of strategy simulation results
        """
        if version_id not in self.versions:
            raise ValueError(f"Unknown version_id: {version_id}")

        self.versions[version_id].backtest_results = backtest_metrics
        self.versions[version_id].strategy_results = strategy_results

        logger.info(f"Logged backtest results for {version_id}")

        # Auto-save if path configured
        if self.ledger_path:
            self.save()

    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        return self.versions.get(version_id)

    def list_versions(
        self, tags: Optional[List[str]] = None, limit: Optional[int] = None
    ) -> List[ModelVersion]:
        """
        List model versions, optionally filtered by tags.

        Args:
            tags: Filter by tags (returns versions with ANY of these tags)
            limit: Maximum number of versions to return

        Returns:
            List of ModelVersion objects, sorted by timestamp (newest first)
        """
        versions = list(self.versions.values())

        # Filter by tags if specified
        if tags:
            versions = [v for v in versions if any(tag in v.tags for tag in tags)]

        # Sort by timestamp (newest first)
        versions.sort(key=lambda v: v.timestamp, reverse=True)

        # Apply limit
        if limit:
            versions = versions[:limit]

        return versions

    def get_best_model(
        self, metric: str = "sharpe_ratio", tags: Optional[List[str]] = None
    ) -> Optional[ModelVersion]:
        """
        Get the best model version by a specific metric.

        Args:
            metric: Metric to optimize (e.g., 'sharpe_ratio', 'accuracy', 'total_return')
            tags: Optional tag filter

        Returns:
            Best ModelVersion or None if no versions with that metric
        """
        versions = self.list_versions(tags=tags)

        # Filter versions that have the requested metric
        valid_versions = []
        for v in versions:
            # Check backtest results
            if v.backtest_results and metric in v.backtest_results:
                valid_versions.append((v, v.backtest_results[metric]))
            # Check strategy results
            elif v.strategy_results and metric in v.strategy_results:
                valid_versions.append((v, v.strategy_results[metric]))

        if not valid_versions:
            return None

        # Return version with highest metric value
        best_version, best_value = max(valid_versions, key=lambda x: x[1])
        return best_version

    def compare_models(self, version_ids: List[str], metrics: Optional[List[str]] = None) -> Dict:
        """
        Compare multiple model versions.

        Args:
            version_ids: List of version IDs to compare
            metrics: List of metrics to compare (or all if None)

        Returns:
            Dict with comparison data
        """
        comparison = {"versions": [], "metrics": {}}

        for version_id in version_ids:
            if version_id not in self.versions:
                logger.warning(f"Version {version_id} not found, skipping")
                continue

            version = self.versions[version_id]
            comparison["versions"].append(
                {
                    "version_id": version_id,
                    "version_name": version.version_name,
                    "timestamp": version.timestamp.isoformat(),
                    "notes": version.notes,
                    "tags": version.tags,
                }
            )

            # Collect metrics
            all_metrics = {}
            if version.backtest_results:
                all_metrics.update(version.backtest_results)
            if version.strategy_results:
                all_metrics.update(version.strategy_results)

            # Filter metrics if specified
            if metrics:
                all_metrics = {k: v for k, v in all_metrics.items() if k in metrics}

            # Add to comparison
            for metric_name, metric_value in all_metrics.items():
                if metric_name not in comparison["metrics"]:
                    comparison["metrics"][metric_name] = {}
                comparison["metrics"][metric_name][version_id] = metric_value

        return comparison

    def get_performance_history(
        self, metric: str, tags: Optional[List[str]] = None
    ) -> List[Tuple[datetime, float, str]]:
        """
        Get performance history for a metric across versions.

        Args:
            metric: Metric name
            tags: Optional tag filter

        Returns:
            List of (timestamp, metric_value, version_id) tuples
        """
        versions = self.list_versions(tags=tags)

        history = []
        for v in versions:
            value = None
            if v.backtest_results and metric in v.backtest_results:
                value = v.backtest_results[metric]
            elif v.strategy_results and metric in v.strategy_results:
                value = v.strategy_results[metric]

            if value is not None:
                history.append((v.timestamp, value, v.version_id))

        # Sort by timestamp
        history.sort(key=lambda x: x[0])

        return history

    def _extract_feature_importance(self, model: Any) -> Optional[Dict]:
        """Extract feature importance from model if available."""
        try:
            # Try XGBoost/LightGBM format
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
                if hasattr(model, "feature_names_"):
                    feature_names = model.feature_names_
                else:
                    feature_names = [f"feature_{i}" for i in range(len(importance))]

                return dict(zip(feature_names, importance.tolist()))

            # Try sklearn format
            elif hasattr(model, "get_booster"):
                booster = model.get_booster()
                if hasattr(booster, "get_score"):
                    return booster.get_score(importance_type="weight")

        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")

        return None

    def save(self, path: Optional[str] = None):
        """Save ledger to file."""
        save_path = path or self.ledger_path
        if not save_path:
            raise ValueError("No save path specified")

        # Convert to serializable format
        ledger_data = {"versions": {}}

        for version_id, version in self.versions.items():
            version_dict = asdict(version)
            # Convert datetime to ISO format
            version_dict["timestamp"] = version.timestamp.isoformat()
            ledger_data["versions"][version_id] = version_dict

        # Save to file
        with open(save_path, "w") as f:
            json.dump(ledger_data, f, indent=2)

        logger.info(f"Saved learning ledger to {save_path}")

    def load(self, path: Optional[str] = None):
        """Load ledger from file."""
        load_path = path or self.ledger_path
        if not load_path or not Path(load_path).exists():
            raise ValueError(f"Ledger file not found: {load_path}")

        with open(load_path, "r") as f:
            ledger_data = json.load(f)

        # Reconstruct versions
        self.versions = {}
        for version_id, version_dict in ledger_data["versions"].items():
            # Convert timestamp back to datetime
            version_dict["timestamp"] = datetime.fromisoformat(version_dict["timestamp"])

            # Reconstruct ModelVersion object
            self.versions[version_id] = ModelVersion(**version_dict)

        logger.info(f"Loaded learning ledger from {load_path} ({len(self.versions)} versions)")

    def export_summary(self, filepath: str):
        """Export a summary of all model versions to JSON."""
        summary = {"total_versions": len(self.versions), "versions": []}

        for version in self.list_versions():
            version_summary = {
                "version_id": version.version_id,
                "version_name": version.version_name,
                "timestamp": version.timestamp.isoformat(),
                "tags": version.tags,
                "notes": version.notes,
                "backtest_metrics": version.backtest_results or {},
                "strategy_metrics": version.strategy_results or {},
            }
            summary["versions"].append(version_summary)

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Exported ledger summary to {filepath}")
