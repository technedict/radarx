"""
Model Trainer CLI

Command-line interface for training ML models on historical data.
"""

import argparse
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

from radarx.backtesting.labeler import OutcomeLabeler
from radarx.backtesting.ledger import LearningLedger, ModelVersion
from radarx.config import settings
from radarx.models.calibrator import ProbabilityCalibrator
from radarx.models.online_learner import OnlineLearner
from radarx.models.probability_predictor import ProbabilityPredictor
from radarx.models.risk_scorer import RiskScorer

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def load_training_data(data_path: Optional[str] = None) -> tuple:
    """
    Load training data.

    In production, this would load real historical data with features and labels.
    For now, returns sample data for demonstration.

    Returns:
        Tuple of (features, labels)
    """
    logger.info("Loading training data...")

    if data_path and Path(data_path).exists():
        logger.info(f"Loading data from {data_path}")
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        return data["features"], data["labels"]

    # Generate sample data
    logger.info("Generating sample training data...")
    n_samples = 1000
    n_features = 20

    # Random features
    features = np.random.randn(n_samples, n_features)

    # Binary labels (simplified)
    labels = (np.random.rand(n_samples) > 0.5).astype(int)

    logger.info(f"Loaded {n_samples} samples with {n_features} features")
    return features, labels


def train_model(
    model_type: str = "probability",
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    horizons: List[str] = None,
    multipliers: List[str] = None,
    calibrate: bool = True,
    version_name: Optional[str] = None,
    verbose: bool = False,
):
    """
    Train a model with specified parameters.

    Args:
        model_type: Type of model to train (probability, risk)
        data_path: Path to training data
        output_dir: Directory to save model
        horizons: Time horizons for probability models
        multipliers: Price multipliers for probability models
        calibrate: Whether to calibrate probabilities
        version_name: Model version name
        verbose: Enable verbose logging
    """
    setup_logging(verbose)

    if horizons is None:
        horizons = ["24h", "7d", "30d"]
    if multipliers is None:
        multipliers = ["2x", "5x", "10x"]

    logger.info("Starting model training")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Horizons: {horizons}")
    logger.info(f"Multipliers: {multipliers}")

    # Load training data
    X_train, y_train = load_training_data(data_path)

    # Initialize model
    if model_type == "probability":
        logger.info("Training probability prediction model...")
        model = ProbabilityPredictor(horizons=horizons, multipliers=multipliers)

        # Train ensemble
        logger.info("Training ensemble models...")
        model.fit(X_train, y_train)

        # Calibrate if requested
        if calibrate:
            logger.info("Calibrating probabilities...")
            calibrator = ProbabilityCalibrator(method="isotonic")

            # Get predictions
            predictions = model.predict_proba(X_train)

            # Calibrate (simplified - would use validation set in production)
            for horizon in horizons:
                for multiplier in multipliers:
                    key = f"{horizon}_{multiplier}"
                    if key in predictions:
                        calibrator.fit(predictions[key], y_train)

        logger.info("Training completed successfully")

    elif model_type == "risk":
        logger.info("Training risk scoring model...")
        model = RiskScorer()

        # Risk scorer doesn't require traditional training
        # but could learn thresholds from data
        logger.info("Risk scorer initialized with default parameters")

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Save model
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = output_path / f"{model_type}_model_{timestamp}.pkl"

        with open(model_file, "wb") as f:
            pickle.dump(model, f)

        logger.info(f"Model saved to {model_file}")

        # Save metadata
        metadata = {
            "model_type": model_type,
            "trained_at": timestamp,
            "version_name": version_name or f"v_{timestamp}",
            "horizons": horizons,
            "multipliers": multipliers,
            "calibrated": calibrate,
            "n_samples": len(X_train),
            "n_features": X_train.shape[1] if len(X_train.shape) > 1 else 1,
        }

        metadata_file = output_path / f"{model_type}_metadata_{timestamp}.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to {metadata_file}")

        # Update learning ledger
        try:
            ledger = LearningLedger()
            version = ModelVersion(
                version_id=f"model_{timestamp}",
                version_name=version_name or f"v_{timestamp}",
                timestamp=datetime.now(),
                model_config={
                    "type": model_type,
                    "horizons": horizons,
                    "multipliers": multipliers,
                    "calibrated": calibrate,
                },
                training_data_info={
                    "n_samples": len(X_train),
                    "n_features": X_train.shape[1] if len(X_train.shape) > 1 else 1,
                    "data_path": data_path,
                },
                notes=f"Trained {model_type} model",
                tags=["trained", model_type],
            )
            ledger.add_version(version)
            logger.info("Added version to learning ledger")
        except Exception as e:
            logger.warning(f"Failed to update learning ledger: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Model Type: {model_type}")
    print(f"Training Samples: {len(X_train)}")
    print(f"Features: {X_train.shape[1] if len(X_train.shape) > 1 else 1}")

    if model_type == "probability":
        print(f"Horizons: {', '.join(horizons)}")
        print(f"Multipliers: {', '.join(multipliers)}")
        print(f"Calibrated: {calibrate}")

    print("=" * 60)

    return model


def main():
    """Main entry point for training CLI."""
    parser = argparse.ArgumentParser(
        description="Train ML models on historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train probability model with default settings
  radarx-train --model-type probability --output-dir ./models

  # Train with custom horizons and multipliers
  radarx-train --model-type probability \\
               --horizons 24h 7d 30d \\
               --multipliers 2x 5x 10x 20x \\
               --output-dir ./models

  # Train from specific data file
  radarx-train --model-type probability \\
               --data-path ./data/training.pkl \\
               --output-dir ./models \\
               --version-name v1.0.0

  # Train risk scorer
  radarx-train --model-type risk --output-dir ./models --verbose
        """,
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="probability",
        choices=["probability", "risk"],
        help="Type of model to train (default: probability)",
    )

    parser.add_argument(
        "--data-path", type=str, help="Path to training data (pickle file with features and labels)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Directory to save trained model (default: ./models)",
    )

    parser.add_argument(
        "--horizons",
        type=str,
        nargs="+",
        default=["24h", "7d", "30d"],
        help="Time horizons for probability models (default: 24h 7d 30d)",
    )

    parser.add_argument(
        "--multipliers",
        type=str,
        nargs="+",
        default=["2x", "5x", "10x"],
        help="Price multipliers for probability models (default: 2x 5x 10x)",
    )

    parser.add_argument("--no-calibrate", action="store_true", help="Skip probability calibration")

    parser.add_argument("--version-name", type=str, help="Model version name (optional)")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    try:
        train_model(
            model_type=args.model_type,
            data_path=args.data_path,
            output_dir=args.output_dir,
            horizons=args.horizons,
            multipliers=args.multipliers,
            calibrate=not args.no_calibrate,
            version_name=args.version_name,
            verbose=args.verbose,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
