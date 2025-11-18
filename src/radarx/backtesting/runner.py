"""
Backtest Runner CLI

Command-line interface for running backtests on historical token data.
"""

import argparse
import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from radarx.backtesting.engine import BacktestEngine
from radarx.backtesting.labeler import OutcomeLabeler
from radarx.backtesting.ledger import LearningLedger
from radarx.backtesting.strategy import StrategySimulator
from radarx.config import settings
from radarx.models.probability_predictor import ProbabilityPredictor

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def load_sample_data(start_date: datetime, end_date: datetime) -> list:
    """
    Load sample data for backtesting.

    In production, this would load real historical data.
    For now, returns sample data for demonstration.
    """
    logger.info(f"Loading sample data from {start_date} to {end_date}")

    # Sample data structure
    sample_data = []
    current_date = start_date

    # Generate some sample data points
    while current_date <= end_date:
        sample_data.append(
            {
                "timestamp": current_date.isoformat() + "Z",
                "token_address": f"0x{random.randint(1000, 9999):04x}" + "0" * 36,
                "chain": "ethereum",
                "price": random.uniform(0.001, 10.0),
                "volume_24h": random.uniform(10000, 1000000),
                "liquidity": random.uniform(50000, 5000000),
                "holders": random.randint(100, 10000),
                "features": {
                    "volume_momentum": random.uniform(-0.5, 0.5),
                    "liquidity_score": random.uniform(0, 100),
                    "holder_concentration": random.uniform(0, 1),
                },
            }
        )
        current_date += timedelta(hours=6)

    logger.info(f"Loaded {len(sample_data)} data points")
    return sample_data


def run_backtest(
    start_date: str,
    end_date: str,
    strategy: str = "threshold",
    fee_rate: float = 0.003,
    slippage_rate: float = 0.001,
    output_dir: Optional[str] = None,
    verbose: bool = False,
):
    """
    Run backtest with specified parameters.

    Args:
        start_date: Start date in ISO format (YYYY-MM-DD)
        end_date: End date in ISO format (YYYY-MM-DD)
        strategy: Trading strategy to use
        fee_rate: Trading fee rate
        slippage_rate: Slippage rate
        output_dir: Directory to save results
        verbose: Enable verbose logging
    """
    setup_logging(verbose)

    logger.info("Starting backtest")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Fee rate: {fee_rate:.2%}, Slippage rate: {slippage_rate:.2%}")

    # Parse dates
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)

    # Load data
    data = load_sample_data(start_dt, end_dt)

    # Initialize components
    engine = BacktestEngine(fee_rate=fee_rate, slippage_rate=slippage_rate)
    simulator = StrategySimulator(fee_rate=fee_rate, slippage_rate=slippage_rate)
    predictor = ProbabilityPredictor()

    logger.info("Running walk-forward backtest...")

    # Run backtest
    try:
        results = engine.run_walk_forward_backtest(
            model=predictor,
            data=data,
            train_window_days=90,
            test_window_days=30,
            start_date=start_dt,
            end_date=end_dt,
        )

        logger.info("Backtest completed successfully")

        # Print summary
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"Period: {start_date} to {end_date}")
        print(f"Total Predictions: {results.get('total_predictions', 0)}")
        print(f"Accuracy: {results.get('accuracy', 0):.2%}")
        print(f"Precision: {results.get('precision', 0):.2%}")
        print(f"Recall: {results.get('recall', 0):.2%}")
        print(f"F1 Score: {results.get('f1_score', 0):.3f}")

        if "calibration_error" in results:
            print(f"Calibration Error: {results['calibration_error']:.4f}")

        print("=" * 60)

        # Save results if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            results_file = output_path / f"backtest_{start_date}_{end_date}.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Results saved to {results_file}")

        return results

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        raise


def main():
    """Main entry point for backtest CLI."""
    parser = argparse.ArgumentParser(
        description="Run backtests on historical token data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run backtest for 2023
  radarx-backtest --start-date 2023-01-01 --end-date 2024-01-01

  # Run with custom fee rates and save results
  radarx-backtest --start-date 2023-01-01 --end-date 2024-01-01 \\
                  --fee-rate 0.005 --slippage-rate 0.002 \\
                  --output-dir ./results

  # Run with verbose logging
  radarx-backtest --start-date 2023-01-01 --end-date 2024-01-01 --verbose
        """,
    )

    parser.add_argument(
        "--start-date", type=str, required=True, help="Start date in ISO format (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end-date", type=str, required=True, help="End date in ISO format (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="threshold",
        choices=["threshold", "proportional", "kelly", "fixed"],
        help="Trading strategy to use (default: threshold)",
    )

    parser.add_argument(
        "--fee-rate", type=float, default=0.003, help="Trading fee rate (default: 0.003 = 0.3%%)"
    )

    parser.add_argument(
        "--slippage-rate", type=float, default=0.001, help="Slippage rate (default: 0.001 = 0.1%%)"
    )

    parser.add_argument("--output-dir", type=str, help="Directory to save results (optional)")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    try:
        run_backtest(
            start_date=args.start_date,
            end_date=args.end_date,
            strategy=args.strategy,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
