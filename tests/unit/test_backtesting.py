"""
Unit tests for backtesting framework.
"""

import pytest
from datetime import datetime, timedelta
import numpy as np

from radarx.backtesting import (
    BacktestEngine,
    StrategySimulator,
    OutcomeLabeler,
    LearningLedger
)


class TestBacktestEngine:
    """Tests for BacktestEngine."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = BacktestEngine(
            train_window_days=90,
            test_window_days=30,
            step_days=7
        )
        assert engine.train_window_days == 90
        assert engine.test_window_days == 30
        assert engine.step_days == 7
    
    def test_calculate_hit_rate_by_bucket(self):
        """Test hit rate calculation by probability bucket."""
        engine = BacktestEngine()
        
        predictions = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        actuals = [0, 0, 0, 1, 0, 1, 1, 1, 1, 1]
        
        result = engine.calculate_hit_rate_by_bucket(predictions, actuals)
        
        assert 'buckets' in result
        assert len(result['buckets']) > 0
    
    def test_calculate_ece(self):
        """Test Expected Calibration Error calculation."""
        engine = BacktestEngine()
        
        predictions = [0.1, 0.3, 0.5, 0.7, 0.9]
        actuals = [0, 0, 1, 1, 1]
        
        ece = engine.calculate_ece(predictions, actuals)
        
        assert isinstance(ece, float)
        assert 0 <= ece <= 1
    
    def test_simulate_trading(self):
        """Test trading simulation."""
        engine = BacktestEngine()
        
        predictions = [0.2, 0.8, 0.3, 0.7]
        actual_returns = [0.1, 0.5, -0.1, 0.3]
        
        result = engine.simulate_trading(
            predictions,
            actual_returns,
            threshold=0.5,
            capital=10000
        )
        
        assert 'total_return' in result
        assert 'num_trades' in result
        assert 'win_rate' in result


class TestStrategySimulator:
    """Tests for StrategySimulator."""
    
    def test_initialization(self):
        """Test simulator initialization."""
        sim = StrategySimulator(
            initial_capital=10000,
            trading_fee=0.003,
            slippage_factor=0.001
        )
        assert sim.initial_capital == 10000
        assert sim.capital == 10000
        assert sim.trading_fee == 0.003
    
    def test_threshold_strategy(self):
        """Test threshold-based strategy."""
        sim = StrategySimulator(initial_capital=10000)
        
        # Create mock data
        base_time = datetime.now()
        historical_data = [
            {
                'token_address': 'token1',
                'timestamp': base_time + timedelta(hours=i),
                'price': 1.0 + i * 0.1
            }
            for i in range(10)
        ]
        
        predictions = {
            'token1': {
                'heatmap': {
                    '7d': {
                        '10x': 0.2
                    }
                }
            }
        }
        
        result = sim.run_strategy(
            strategy_type='threshold',
            historical_data=historical_data,
            predictions=predictions,
            threshold=0.15
        )
        
        assert 'total_return' in result
        assert 'num_trades' in result
    
    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        sim = StrategySimulator(initial_capital=10000)
        
        # Manually add some closed trades
        from radarx.backtesting.strategy import Trade
        sim.closed_trades = [
            Trade(
                token_address='token1',
                entry_time=datetime.now(),
                entry_price=1.0,
                quantity=100,
                exit_time=datetime.now() + timedelta(days=1),
                exit_price=1.5,
                pnl=50,
                fees_paid=1,
                slippage_cost=0.5
            ),
            Trade(
                token_address='token2',
                entry_time=datetime.now(),
                entry_price=1.0,
                quantity=100,
                exit_time=datetime.now() + timedelta(days=1),
                exit_price=0.8,
                pnl=-20,
                fees_paid=1,
                slippage_cost=0.5
            )
        ]
        
        metrics = sim._calculate_performance_metrics()
        
        assert 'total_pnl' in metrics
        assert 'win_rate' in metrics
        assert 'sharpe_ratio' in metrics
        assert metrics['num_trades'] == 2
        assert metrics['num_wins'] == 1
        assert metrics['num_losses'] == 1
    
    def test_trade_history(self):
        """Test trade history export."""
        sim = StrategySimulator(initial_capital=10000)
        
        from radarx.backtesting.strategy import Trade
        sim.closed_trades = [
            Trade(
                token_address='token1',
                entry_time=datetime.now(),
                entry_price=1.0,
                quantity=100,
                exit_time=datetime.now() + timedelta(days=1),
                exit_price=1.2,
                pnl=20,
                fees_paid=0.5,
                slippage_cost=0.2
            )
        ]
        
        history = sim.get_trade_history()
        
        assert len(history) == 1
        assert history[0]['token'] == 'token1'
        assert history[0]['pnl'] == 20


class TestOutcomeLabeler:
    """Tests for OutcomeLabeler."""
    
    def test_initialization(self):
        """Test labeler initialization."""
        labeler = OutcomeLabeler(
            multipliers=['2x', '5x', '10x'],
            horizons=['24h', '7d'],
            min_liquidity_usd=50000
        )
        assert labeler.multipliers == ['2x', '5x', '10x']
        assert labeler.horizons == ['24h', '7d']
        assert labeler.horizon_hours == {'24h': 24, '7d': 168}
    
    def test_label_outcomes(self):
        """Test outcome labeling."""
        labeler = OutcomeLabeler(
            multipliers=['2x', '5x'],
            horizons=['24h'],
            min_liquidity_usd=10000,
            min_volume_24h=1000
        )
        
        base_time = datetime.now()
        token_data = [
            {
                'token_address': 'token1',
                'timestamp': base_time,
                'price': 1.0,
                'liquidity_usd': 50000,
                'volume_24h': 5000
            },
            {
                'token_address': 'token1',
                'timestamp': base_time + timedelta(hours=12),
                'price': 2.5,  # Reached 2x
                'liquidity_usd': 50000,
                'volume_24h': 5000
            },
            {
                'token_address': 'token1',
                'timestamp': base_time + timedelta(hours=24),
                'price': 6.0,  # Reached 5x
                'liquidity_usd': 50000,
                'volume_24h': 5000
            }
        ]
        
        labels = labeler.label_outcomes(token_data)
        
        assert len(labels) > 0
        assert labels[0].token_address == 'token1'
        assert '24h' in labels[0].labels
        assert '2x' in labels[0].labels['24h']
    
    def test_label_quality_metrics(self):
        """Test label quality metrics."""
        labeler = OutcomeLabeler(
            multipliers=['2x'],
            horizons=['24h']
        )
        
        from radarx.backtesting.labeler import OutcomeLabel
        labels = [
            OutcomeLabel(
                token_address='token1',
                timestamp=datetime.now(),
                labels={'24h': {'2x': True}},
                time_to_target={'24h': {'2x': 12.0}},
                censored={'24h': {'2x': False}},
                max_price_reached={'24h': 2.5},
                final_liquidity={'24h': 50000}
            ),
            OutcomeLabel(
                token_address='token2',
                timestamp=datetime.now(),
                labels={'24h': {'2x': False}},
                time_to_target={'24h': {'2x': None}},
                censored={'24h': {'2x': False}},
                max_price_reached={'24h': 1.5},
                final_liquidity={'24h': 50000}
            )
        ]
        
        metrics = labeler.get_label_quality_metrics(labels)
        
        assert '24h' in metrics
        assert '2x' in metrics['24h']
        assert metrics['24h']['2x']['hit_rate'] == 0.5
        assert metrics['24h']['2x']['total_samples'] == 2
    
    def test_to_training_format(self):
        """Test conversion to training format."""
        labeler = OutcomeLabeler(
            multipliers=['2x'],
            horizons=['24h']
        )
        
        from radarx.backtesting.labeler import OutcomeLabel
        labels = [
            OutcomeLabel(
                token_address='token1',
                timestamp=datetime.now(),
                labels={'24h': {'2x': True}},
                time_to_target={'24h': {'2x': 12.0}},
                censored={'24h': {'2x': False}},
                max_price_reached={'24h': 2.5},
                final_liquidity={'24h': 50000}
            )
        ]
        
        features = {
            'token1': {'feature1': 1.0, 'feature2': 2.0}
        }
        
        X, y = labeler.to_training_format(labels, features)
        
        assert len(X) == 1
        assert len(y) == 1
        assert X[0] == features['token1']
        assert '24h_2x' in y[0]
        assert y[0]['24h_2x'] is True


class TestLearningLedger:
    """Tests for LearningLedger."""
    
    def test_initialization(self):
        """Test ledger initialization."""
        ledger = LearningLedger()
        assert len(ledger.versions) == 0
    
    def test_log_model_version(self):
        """Test logging a model version."""
        ledger = LearningLedger()
        
        class DummyModel:
            pass
        
        model = DummyModel()
        version_id = ledger.log_model_version(
            model=model,
            version='v1.0.0',
            config={'param1': 'value1'},
            notes='Test model'
        )
        
        assert version_id in ledger.versions
        assert ledger.versions[version_id].version_name == 'v1.0.0'
    
    def test_log_backtest_result(self):
        """Test logging backtest results."""
        ledger = LearningLedger()
        
        class DummyModel:
            pass
        
        version_id = ledger.log_model_version(
            model=DummyModel(),
            version='v1.0.0',
            config={}
        )
        
        ledger.log_backtest_result(
            version_id=version_id,
            backtest_metrics={'accuracy': 0.75, 'ece': 0.05},
            strategy_results={'total_return': 0.3, 'sharpe_ratio': 1.5}
        )
        
        version = ledger.get_version(version_id)
        assert version.backtest_results['accuracy'] == 0.75
        assert version.strategy_results['sharpe_ratio'] == 1.5
    
    def test_get_best_model(self):
        """Test getting best model by metric."""
        ledger = LearningLedger()
        
        class DummyModel:
            pass
        
        # Log multiple versions
        v1 = ledger.log_model_version(DummyModel(), 'v1.0.0', {})
        ledger.log_backtest_result(v1, {}, {'sharpe_ratio': 1.0})
        
        v2 = ledger.log_model_version(DummyModel(), 'v2.0.0', {})
        ledger.log_backtest_result(v2, {}, {'sharpe_ratio': 2.0})
        
        v3 = ledger.log_model_version(DummyModel(), 'v3.0.0', {})
        ledger.log_backtest_result(v3, {}, {'sharpe_ratio': 1.5})
        
        best = ledger.get_best_model(metric='sharpe_ratio')
        
        assert best.version_name == 'v2.0.0'
    
    def test_compare_models(self):
        """Test model comparison."""
        ledger = LearningLedger()
        
        class DummyModel:
            pass
        
        v1 = ledger.log_model_version(DummyModel(), 'v1.0.0', {})
        ledger.log_backtest_result(v1, {'accuracy': 0.7}, {'sharpe_ratio': 1.0})
        
        v2 = ledger.log_model_version(DummyModel(), 'v2.0.0', {})
        ledger.log_backtest_result(v2, {'accuracy': 0.75}, {'sharpe_ratio': 1.5})
        
        comparison = ledger.compare_models([v1, v2])
        
        assert len(comparison['versions']) == 2
        assert 'accuracy' in comparison['metrics']
        assert 'sharpe_ratio' in comparison['metrics']
    
    def test_performance_history(self):
        """Test performance history tracking."""
        ledger = LearningLedger()
        
        class DummyModel:
            pass
        
        v1 = ledger.log_model_version(DummyModel(), 'v1.0.0', {})
        ledger.log_backtest_result(v1, {}, {'sharpe_ratio': 1.0})
        
        v2 = ledger.log_model_version(DummyModel(), 'v2.0.0', {})
        ledger.log_backtest_result(v2, {}, {'sharpe_ratio': 1.5})
        
        history = ledger.get_performance_history('sharpe_ratio')
        
        assert len(history) == 2
        assert all(isinstance(h[0], datetime) for h in history)
        assert history[0][1] == 1.0 or history[0][1] == 1.5  # One of the values


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
