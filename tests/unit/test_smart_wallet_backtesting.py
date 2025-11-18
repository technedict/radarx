"""
Tests for Smart Wallet Finder Backtesting Framework
"""

import pytest
from datetime import datetime, timedelta

from radarx.smart_wallet_finder.backtesting import (
    BacktestConfig,
    BacktestResult,
    SmartWalletBacktester,
    WalkForwardValidator,
)


class TestSmartWalletBacktester:
    """Test smart wallet backtesting functionality."""

    @pytest.fixture
    def backtester(self):
        config = BacktestConfig(
            initial_capital=10000.0,
            position_size=0.1,
            top_k=5,
        )
        return SmartWalletBacktester(config)

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for backtesting."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)

        # Smart wallet rankings
        rankings = []
        for i in range(10):
            rankings.append(
                {
                    "wallet_address": f"wallet_{i}",
                    "smart_money_score": 0.9 - i * 0.05,
                    "rank": i + 1,
                }
            )

        # Price data
        price_data = []
        for i in range(31):
            price_data.append(
                {
                    "timestamp": (start_date + timedelta(days=i)).isoformat(),
                    "price": 10.0 + i * 0.5,
                }
            )

        # Trades data
        trades_data = []
        for i in range(50):
            trades_data.append(
                {
                    "timestamp": (start_date + timedelta(days=i % 30, hours=i % 24)).isoformat(),
                    "buyer": f"wallet_{i % 10}",
                    "seller": f"wallet_{(i + 1) % 10}",
                    "side": "buy" if i % 2 == 0 else "sell",
                    "amount_usd": 1000 + i * 10,
                }
            )

        return rankings, start_date, end_date, price_data, trades_data

    def test_run_backtest(self, backtester, sample_data):
        """Test running a backtest."""
        rankings, start_date, end_date, price_data, trades_data = sample_data

        result = backtester.run_backtest(
            rankings,
            "test_token",
            start_date,
            end_date,
            price_data,
            trades_data,
        )

        assert isinstance(result, BacktestResult)
        assert isinstance(result.total_return, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown, float)
        assert isinstance(result.win_rate, float)
        assert isinstance(result.num_trades, int)
        assert isinstance(result.precision_at_k, dict)
        assert len(result.portfolio_value_history) > 0

    def test_precision_at_k(self, backtester, sample_data):
        """Test precision@K calculation."""
        rankings, start_date, end_date, price_data, trades_data = sample_data

        result = backtester.run_backtest(
            rankings,
            "test_token",
            start_date,
            end_date,
            price_data,
            trades_data,
        )

        # Should have precision scores for different K values
        assert len(result.precision_at_k) > 0

        for k, precision in result.precision_at_k.items():
            assert isinstance(k, int)
            assert isinstance(precision, float)
            assert 0 <= precision <= 1


class TestWalkForwardValidator:
    """Test walk-forward validation."""

    @pytest.fixture
    def validator(self):
        return WalkForwardValidator(
            train_window_days=30,
            test_window_days=10,
            step_days=10,
        )

    def test_generate_splits(self, validator):
        """Test generating train/test splits."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 3, 31)

        splits = validator.generate_splits(start_date, end_date)

        assert len(splits) > 0

        for train_start, train_end, test_start, test_end in splits:
            # Train period should be 30 days
            assert (train_end - train_start).days == 30

            # Test period should be 10 days
            assert (test_end - test_start).days == 10

            # Test should immediately follow train
            assert train_end == test_start

            # All dates should be within overall range
            assert start_date <= train_start
            assert test_end <= end_date

    def test_no_splits_if_too_short(self, validator):
        """Test that no splits generated if period too short."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 15)  # Only 15 days

        splits = validator.generate_splits(start_date, end_date)

        # Should be empty since we need 40 days minimum (30 train + 10 test)
        assert len(splits) == 0
