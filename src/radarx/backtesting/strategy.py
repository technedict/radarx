"""
Trading strategy simulator for backtesting.

Provides realistic simulation of trading strategies with position management,
fee/slippage modeling, and performance metric calculation.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Trade:
    """Represents a single trade."""
    token_address: str
    entry_time: datetime
    entry_price: float
    quantity: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    fees_paid: float = 0.0
    slippage_cost: float = 0.0


@dataclass
class Position:
    """Represents an open position."""
    token_address: str
    entry_time: datetime
    entry_price: float
    quantity: float
    cost_basis: float


class StrategySimulator:
    """
    Simulates trading strategies with realistic transaction costs.
    
    Supports multiple strategy types:
    - Threshold: Buy when probability > threshold
    - Top-N: Buy top N tokens by probability
    - Risk-adjusted: Buy based on probability/risk ratio
    - Kelly: Use Kelly criterion for position sizing
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        trading_fee: float = 0.003,  # 0.3%
        slippage_factor: float = 0.001,  # 0.1%
        max_position_size: float = 0.1,  # 10% of capital per position
        max_positions: int = 10
    ):
        """
        Initialize strategy simulator.
        
        Args:
            initial_capital: Starting capital in USD
            trading_fee: Trading fee as decimal (0.003 = 0.3%)
            slippage_factor: Slippage as decimal (0.001 = 0.1%)
            max_position_size: Max position size as fraction of capital
            max_positions: Maximum number of open positions
        """
        self.initial_capital = initial_capital
        self.trading_fee = trading_fee
        self.slippage_factor = slippage_factor
        self.max_position_size = max_position_size
        self.max_positions = max_positions
        
        # Simulation state
        self.capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Trade] = []
        
    def run_strategy(
        self,
        strategy_type: str,
        historical_data: List[Dict],
        predictions: Dict[str, Dict],
        **strategy_params
    ) -> Dict:
        """
        Run a trading strategy simulation.
        
        Args:
            strategy_type: One of 'threshold', 'top_n', 'risk_adjusted', 'kelly'
            historical_data: List of token data dicts with price history
            predictions: Dict mapping token_address to prediction dict
            **strategy_params: Strategy-specific parameters
            
        Returns:
            Dict with performance metrics
        """
        # Reset state
        self.capital = self.initial_capital
        self.positions = {}
        self.closed_trades = []
        
        # Sort data by timestamp
        sorted_data = sorted(historical_data, key=lambda x: x['timestamp'])
        
        for data_point in sorted_data:
            timestamp = data_point['timestamp']
            
            # Check for exits first
            self._check_exits(timestamp, historical_data)
            
            # Generate entry signals
            if strategy_type == 'threshold':
                self._threshold_strategy(data_point, predictions, **strategy_params)
            elif strategy_type == 'top_n':
                self._top_n_strategy(data_point, predictions, **strategy_params)
            elif strategy_type == 'risk_adjusted':
                self._risk_adjusted_strategy(data_point, predictions, **strategy_params)
            elif strategy_type == 'kelly':
                self._kelly_strategy(data_point, predictions, **strategy_params)
            else:
                raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        # Close all remaining positions at final prices
        final_timestamp = sorted_data[-1]['timestamp']
        for token_addr in list(self.positions.keys()):
            final_price = self._get_price_at_time(token_addr, final_timestamp, historical_data)
            if final_price:
                self._close_position(token_addr, final_timestamp, final_price)
        
        # Calculate performance metrics
        return self._calculate_performance_metrics()
    
    def _threshold_strategy(
        self,
        data_point: Dict,
        predictions: Dict,
        threshold: float = 0.15,
        horizon: str = '7d',
        multiplier: str = '10x'
    ):
        """Buy when probability exceeds threshold."""
        token_addr = data_point['token_address']
        
        if token_addr not in predictions:
            return
        
        pred = predictions[token_addr]
        prob = pred.get('heatmap', {}).get(horizon, {}).get(multiplier, 0.0)
        
        if prob > threshold and len(self.positions) < self.max_positions:
            position_size = self.capital * self.max_position_size
            self._open_position(
                token_addr,
                data_point['timestamp'],
                data_point['price'],
                position_size
            )
    
    def _top_n_strategy(
        self,
        data_point: Dict,
        predictions: Dict,
        top_n: int = 5,
        horizon: str = '7d',
        multiplier: str = '10x'
    ):
        """Buy top N tokens by probability."""
        # This would need access to all tokens at current timestamp
        # Simplified implementation
        token_addr = data_point['token_address']
        
        if token_addr not in predictions:
            return
        
        pred = predictions[token_addr]
        prob = pred.get('heatmap', {}).get(horizon, {}).get(multiplier, 0.0)
        
        # In real implementation, would rank all available tokens
        if prob > 0.1 and len(self.positions) < top_n:
            position_size = self.capital * self.max_position_size
            self._open_position(
                token_addr,
                data_point['timestamp'],
                data_point['price'],
                position_size
            )
    
    def _risk_adjusted_strategy(
        self,
        data_point: Dict,
        predictions: Dict,
        min_ratio: float = 0.3,
        horizon: str = '7d',
        multiplier: str = '10x'
    ):
        """Buy when probability/risk ratio exceeds threshold."""
        token_addr = data_point['token_address']
        
        if token_addr not in predictions:
            return
        
        pred = predictions[token_addr]
        prob = pred.get('heatmap', {}).get(horizon, {}).get(multiplier, 0.0)
        risk = pred.get('risk_score', {}).get('composite_score', 50.0) / 100.0
        
        ratio = prob / max(risk, 0.01)  # Avoid division by zero
        
        if ratio > min_ratio and len(self.positions) < self.max_positions:
            position_size = self.capital * self.max_position_size
            self._open_position(
                token_addr,
                data_point['timestamp'],
                data_point['price'],
                position_size
            )
    
    def _kelly_strategy(
        self,
        data_point: Dict,
        predictions: Dict,
        horizon: str = '7d',
        multiplier: str = '10x',
        kelly_fraction: float = 0.25  # Use fractional Kelly
    ):
        """Use Kelly criterion for position sizing."""
        token_addr = data_point['token_address']
        
        if token_addr not in predictions:
            return
        
        pred = predictions[token_addr]
        prob = pred.get('heatmap', {}).get(horizon, {}).get(multiplier, 0.0)
        
        # Kelly formula: f = (bp - q) / b
        # where b is odds, p is win probability, q = 1-p
        target_mult = float(multiplier.replace('x', ''))
        b = target_mult - 1  # Odds
        f = (b * prob - (1 - prob)) / b
        
        if f > 0:
            # Use fractional Kelly for safety
            position_size = min(
                self.capital * f * kelly_fraction,
                self.capital * self.max_position_size
            )
            
            if len(self.positions) < self.max_positions and position_size > 0:
                self._open_position(
                    token_addr,
                    data_point['timestamp'],
                    data_point['price'],
                    position_size
                )
    
    def _open_position(
        self,
        token_address: str,
        timestamp: datetime,
        price: float,
        position_size_usd: float
    ):
        """Open a new position."""
        if token_address in self.positions:
            return  # Already have position
        
        if self.capital < position_size_usd:
            return  # Not enough capital
        
        # Calculate fees and slippage
        fee = position_size_usd * self.trading_fee
        slippage = position_size_usd * self.slippage_factor
        total_cost = position_size_usd + fee + slippage
        
        if total_cost > self.capital:
            return
        
        # Calculate quantity (accounting for slippage in price)
        effective_price = price * (1 + self.slippage_factor)
        quantity = position_size_usd / effective_price
        
        # Update capital
        self.capital -= total_cost
        
        # Create position
        self.positions[token_address] = Position(
            token_address=token_address,
            entry_time=timestamp,
            entry_price=effective_price,
            quantity=quantity,
            cost_basis=total_cost
        )
    
    def _close_position(
        self,
        token_address: str,
        timestamp: datetime,
        price: float
    ):
        """Close an existing position."""
        if token_address not in self.positions:
            return
        
        position = self.positions[token_address]
        
        # Calculate exit value (accounting for slippage)
        effective_price = price * (1 - self.slippage_factor)
        gross_proceeds = position.quantity * effective_price
        
        # Calculate fees
        fee = gross_proceeds * self.trading_fee
        net_proceeds = gross_proceeds - fee
        
        # Calculate total costs (entry + exit)
        total_slippage = (
            position.cost_basis * self.slippage_factor +
            gross_proceeds * self.slippage_factor
        )
        total_fees = position.cost_basis * self.trading_fee + fee
        
        # Calculate P&L
        pnl = net_proceeds - position.cost_basis
        
        # Update capital
        self.capital += net_proceeds
        
        # Record trade
        trade = Trade(
            token_address=token_address,
            entry_time=position.entry_time,
            entry_price=position.entry_price,
            quantity=position.quantity,
            exit_time=timestamp,
            exit_price=effective_price,
            pnl=pnl,
            fees_paid=total_fees,
            slippage_cost=total_slippage
        )
        self.closed_trades.append(trade)
        
        # Remove position
        del self.positions[token_address]
    
    def _check_exits(self, timestamp: datetime, historical_data: List[Dict]):
        """Check if any positions should be exited."""
        for token_addr in list(self.positions.keys()):
            position = self.positions[token_addr]
            
            # Simple exit logic: hold for 7 days or until price target
            hold_time = timestamp - position.entry_time
            
            current_price = self._get_price_at_time(token_addr, timestamp, historical_data)
            if not current_price:
                continue
            
            # Exit conditions
            price_change = (current_price - position.entry_price) / position.entry_price
            
            # Take profit at 10x or stop loss at -50%
            if price_change >= 9.0 or price_change <= -0.5 or hold_time > timedelta(days=7):
                self._close_position(token_addr, timestamp, current_price)
    
    def _get_price_at_time(
        self,
        token_address: str,
        timestamp: datetime,
        historical_data: List[Dict]
    ) -> Optional[float]:
        """Get token price at specific timestamp."""
        for data in historical_data:
            if data['token_address'] == token_address and data['timestamp'] == timestamp:
                return data['price']
        return None
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate strategy performance metrics."""
        if not self.closed_trades:
            return {
                'total_return': 0.0,
                'num_trades': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }
        
        # Calculate returns
        total_pnl = sum(t.pnl for t in self.closed_trades)
        total_return = total_pnl / self.initial_capital
        
        # Win rate
        winning_trades = [t for t in self.closed_trades if t.pnl > 0]
        win_rate = len(winning_trades) / len(self.closed_trades)
        
        # Average P&L
        avg_pnl = total_pnl / len(self.closed_trades)
        
        # Sharpe ratio (simplified)
        returns = [t.pnl / self.initial_capital for t in self.closed_trades]
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown
        cumulative = [self.initial_capital]
        for trade in self.closed_trades:
            cumulative.append(cumulative[-1] + trade.pnl)
        
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (np.array(cumulative) - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Average hold time
        hold_times = [
            (t.exit_time - t.entry_time).total_seconds() / 3600  # hours
            for t in self.closed_trades if t.exit_time
        ]
        avg_hold_time = np.mean(hold_times) if hold_times else 0.0
        
        return {
            'total_return': total_return,
            'total_pnl': total_pnl,
            'final_capital': self.capital,
            'num_trades': len(self.closed_trades),
            'win_rate': win_rate,
            'num_wins': len(winning_trades),
            'num_losses': len(self.closed_trades) - len(winning_trades),
            'avg_pnl': avg_pnl,
            'best_trade': max(t.pnl for t in self.closed_trades),
            'worst_trade': min(t.pnl for t in self.closed_trades),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_hold_time_hours': avg_hold_time,
            'total_fees_paid': sum(t.fees_paid for t in self.closed_trades),
            'total_slippage': sum(t.slippage_cost for t in self.closed_trades)
        }
    
    def get_trade_history(self) -> List[Dict]:
        """Get detailed trade history."""
        return [
            {
                'token': t.token_address,
                'entry_time': t.entry_time.isoformat(),
                'entry_price': t.entry_price,
                'exit_time': t.exit_time.isoformat() if t.exit_time else None,
                'exit_price': t.exit_price,
                'quantity': t.quantity,
                'pnl': t.pnl,
                'pnl_pct': (t.pnl / (t.quantity * t.entry_price)) if t.quantity > 0 else 0,
                'fees': t.fees_paid,
                'slippage': t.slippage_cost
            }
            for t in self.closed_trades
        ]
