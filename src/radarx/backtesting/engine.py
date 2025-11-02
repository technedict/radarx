"""
Backtest Engine

Walk-forward backtesting with realistic fee and slippage simulation.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np


class BacktestEngine:
    """Walk-forward backtesting engine with realistic simulation."""
    
    def __init__(
        self,
        fee_rate: float = 0.003,  # 0.3%
        slippage_rate: float = 0.001,  # 0.1%
    ):
        """
        Initialize backtest engine.
        
        Args:
            fee_rate: Trading fee rate (default 0.3%)
            slippage_rate: Slippage rate (default 0.1%)
        """
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
    
    def run_walk_forward_backtest(
        self,
        model,
        data: List[Dict[str, Any]],
        train_window_days: int = 90,
        test_window_days: int = 30,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Run walk-forward backtest.
        
        Args:
            model: Trained model with predict method
            data: Historical data with timestamps
            train_window_days: Training window size
            test_window_days: Testing window size
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Dictionary with backtest results and metrics
        """
        if not data:
            return self._empty_results()
        
        # Sort data by timestamp
        sorted_data = sorted(data, key=lambda x: x['timestamp'])
        
        # Determine date range
        if not start_date:
            start_date = datetime.fromisoformat(sorted_data[0]['timestamp'].replace('Z', '+00:00'))
        if not end_date:
            end_date = datetime.fromisoformat(sorted_data[-1]['timestamp'].replace('Z', '+00:00'))
        
        all_predictions = []
        all_outcomes = []
        
        # Walk forward
        current_date = start_date + timedelta(days=train_window_days)
        
        while current_date < end_date:
            # Get training window
            train_start = current_date - timedelta(days=train_window_days)
            train_data = [
                d for d in sorted_data
                if train_start <= datetime.fromisoformat(d['timestamp'].replace('Z', '+00:00')) < current_date
            ]
            
            # Get test window
            test_end = current_date + timedelta(days=test_window_days)
            test_data = [
                d for d in sorted_data
                if current_date <= datetime.fromisoformat(d['timestamp'].replace('Z', '+00:00')) < test_end
            ]
            
            if train_data and test_data:
                # Note: In production, would retrain model here
                # For now, just predict on test data
                for test_sample in test_data:
                    pred = test_sample.get('predicted_probability', 0.5)
                    outcome = test_sample.get('actual_outcome', False)
                    
                    all_predictions.append(pred)
                    all_outcomes.append(outcome)
            
            # Move window forward
            current_date += timedelta(days=test_window_days)
        
        # Calculate metrics
        return self._calculate_backtest_metrics(all_predictions, all_outcomes)
    
    def calculate_hit_rates(
        self,
        predictions: List[Dict[str, Any]],
        outcomes: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate hit rates by probability bucket.
        
        Args:
            predictions: List of predictions with probabilities
            outcomes: List of actual outcomes
            
        Returns:
            Hit rates by multiplier and horizon
        """
        # Group by probability buckets
        buckets = defaultdict(lambda: {'predicted': [], 'actual': []})
        
        for pred, outcome in zip(predictions, outcomes):
            for horizon in ['24h', '7d', '30d']:
                for multiplier in ['2x', '5x', '10x', '20x', '50x']:
                    key = f"{horizon}_{multiplier}"
                    prob = pred.get(horizon, {}).get(multiplier, 0.0)
                    actual = outcome.get(horizon, {}).get(multiplier, False)
                    
                    bucket_idx = int(prob * 10)  # 10 buckets
                    buckets[bucket_idx]['predicted'].append(prob)
                    buckets[bucket_idx]['actual'].append(actual)
        
        # Calculate hit rate per bucket
        hit_rates = {}
        for bucket_idx, data in buckets.items():
            if data['actual']:
                hit_rate = sum(data['actual']) / len(data['actual'])
                avg_predicted = np.mean(data['predicted'])
                
                hit_rates[f"bucket_{bucket_idx}"] = {
                    'hit_rate': hit_rate,
                    'avg_predicted': avg_predicted,
                    'count': len(data['actual'])
                }
        
        return hit_rates
    
    def calculate_calibration_error(
        self,
        predictions: List[float],
        outcomes: List[bool],
        n_bins: int = 10
    ) -> Tuple[float, List[Dict[str, float]]]:
        """
        Calculate Expected Calibration Error (ECE).
        
        Args:
            predictions: Predicted probabilities
            outcomes: Actual outcomes (binary)
            n_bins: Number of bins for calibration
            
        Returns:
            Tuple of (ECE, calibration_curve)
        """
        if not predictions or not outcomes:
            return 0.0, []
        
        # Create bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_edges[:-1]
        bin_uppers = bin_edges[1:]
        
        ece = 0.0
        calibration_curve = []
        
        predictions_arr = np.array(predictions)
        outcomes_arr = np.array(outcomes, dtype=float)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Get predictions in this bin
            in_bin = (predictions_arr > bin_lower) & (predictions_arr <= bin_upper)
            
            if np.sum(in_bin) > 0:
                bin_accuracy = np.mean(outcomes_arr[in_bin])
                bin_confidence = np.mean(predictions_arr[in_bin])
                bin_count = np.sum(in_bin)
                
                # ECE contribution
                ece += (bin_count / len(predictions)) * abs(bin_accuracy - bin_confidence)
                
                calibration_curve.append({
                    'bin_lower': float(bin_lower),
                    'bin_upper': float(bin_upper),
                    'predicted': float(bin_confidence),
                    'actual': float(bin_accuracy),
                    'count': int(bin_count)
                })
        
        return ece, calibration_curve
    
    def simulate_trading_with_fees(
        self,
        signals: List[Dict[str, Any]],
        initial_capital: float = 10000.0
    ) -> Dict[str, Any]:
        """
        Simulate trading with realistic fees and slippage.
        
        Args:
            signals: Trading signals with entry/exit points
            initial_capital: Starting capital
            
        Returns:
            Trading simulation results
        """
        capital = initial_capital
        positions = {}
        trades = []
        pnl_history = [0.0]
        
        for signal in signals:
            action = signal.get('action')  # 'buy' or 'sell'
            token = signal.get('token')
            price = signal.get('price', 0)
            amount = signal.get('amount', 0)
            timestamp = signal.get('timestamp')
            
            if action == 'buy' and capital > 0:
                # Apply slippage and fees
                effective_price = price * (1 + self.slippage_rate)
                cost = amount * effective_price
                fee = cost * self.fee_rate
                total_cost = cost + fee
                
                if total_cost <= capital:
                    capital -= total_cost
                    positions[token] = {
                        'amount': amount,
                        'entry_price': effective_price,
                        'timestamp': timestamp
                    }
                    
                    trades.append({
                        'type': 'buy',
                        'token': token,
                        'amount': amount,
                        'price': effective_price,
                        'cost': total_cost,
                        'timestamp': timestamp
                    })
            
            elif action == 'sell' and token in positions:
                position = positions[token]
                
                # Apply slippage and fees
                effective_price = price * (1 - self.slippage_rate)
                proceeds = position['amount'] * effective_price
                fee = proceeds * self.fee_rate
                net_proceeds = proceeds - fee
                
                capital += net_proceeds
                
                # Calculate PnL
                entry_cost = position['amount'] * position['entry_price'] * (1 + self.fee_rate)
                pnl = net_proceeds - entry_cost
                pnl_history.append(pnl_history[-1] + pnl)
                
                trades.append({
                    'type': 'sell',
                    'token': token,
                    'amount': position['amount'],
                    'price': effective_price,
                    'proceeds': net_proceeds,
                    'pnl': pnl,
                    'timestamp': timestamp
                })
                
                del positions[token]
        
        # Calculate metrics
        final_capital = capital + sum(
            pos['amount'] * pos['entry_price'] for pos in positions.values()
        )
        
        total_return = (final_capital - initial_capital) / initial_capital
        
        # Max drawdown
        cumulative_pnl = np.array(pnl_history)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        max_drawdown = np.max(drawdown) / max(running_max[-1], 1) if len(running_max) > 0 else 0.0
        
        return {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'total_pnl': final_capital - initial_capital,
            'num_trades': len(trades),
            'max_drawdown': max_drawdown,
            'trades': trades
        }
    
    def _calculate_backtest_metrics(
        self,
        predictions: List[float],
        outcomes: List[bool]
    ) -> Dict[str, Any]:
        """Calculate comprehensive backtest metrics."""
        if not predictions or not outcomes:
            return self._empty_results()
        
        # Accuracy metrics
        predictions_binary = [1 if p > 0.5 else 0 for p in predictions]
        outcomes_int = [int(o) for o in outcomes]
        
        correct = sum(1 for p, o in zip(predictions_binary, outcomes_int) if p == o)
        accuracy = correct / len(predictions)
        
        # Precision and recall
        tp = sum(1 for p, o in zip(predictions_binary, outcomes_int) if p == 1 and o == 1)
        fp = sum(1 for p, o in zip(predictions_binary, outcomes_int) if p == 1 and o == 0)
        fn = sum(1 for p, o in zip(predictions_binary, outcomes_int) if p == 0 and o == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calibration
        ece, calibration_curve = self.calculate_calibration_error(predictions, outcomes)
        
        return {
            'num_samples': len(predictions),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'ece': ece,
            'calibration_curve': calibration_curve
        }
    
    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure."""
        return {
            'num_samples': 0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'ece': 0.0,
            'calibration_curve': []
        }
