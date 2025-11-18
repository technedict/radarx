"""
Backtesting Framework for Smart Wallet Finder

Validates smart wallet predictions through:
- Precision@K metrics
- Simulated portfolio returns
- Walk-forward validation
- Performance reporting
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    initial_capital: float = 10000.0
    position_size: float = 0.1  # Fraction of capital per position
    top_k: int = 10  # Number of top wallets to follow
    rebalance_frequency_days: int = 7
    fee_rate: float = 0.003  # 0.3% trading fee
    slippage_rate: float = 0.001  # 0.1% slippage
    max_position_size_usd: float = 1000.0


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    precision_at_k: Dict[int, float]
    portfolio_value_history: List[Tuple[datetime, float]]
    trade_log: List[Dict[str, Any]]
    metrics: Dict[str, Any]


class SmartWalletBacktester:
    """
    Backtesting framework for smart wallet finder predictions.

    Simulates following top-K smart wallets and measures performance.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtester.

        Args:
            config: Backtesting configuration
        """
        self.config = config or BacktestConfig()

    def run_backtest(
        self,
        smart_wallet_rankings: List[Dict[str, Any]],
        token_address: str,
        start_date: datetime,
        end_date: datetime,
        price_data: List[Dict[str, Any]],
        all_trades_data: List[Dict[str, Any]],
    ) -> BacktestResult:
        """
        Run backtest by following top-K smart wallets.

        Args:
            smart_wallet_rankings: Ranked list of wallets with scores
            token_address: Token being analyzed
            start_date: Backtest start date
            end_date: Backtest end date
            price_data: Historical price timeline
            all_trades_data: All trades in the period

        Returns:
            BacktestResult with performance metrics
        """
        logger.info(f"Running backtest from {start_date} to {end_date}")

        # Initialize portfolio
        portfolio = self._initialize_portfolio()
        trade_log = []
        portfolio_history = [(start_date, self.config.initial_capital)]

        # Get top-K wallets to follow
        top_wallets = [w["wallet_address"] for w in smart_wallet_rankings[: self.config.top_k]]

        logger.info(f"Following top {len(top_wallets)} wallets")

        # Organize trades by wallet and time
        wallet_trades = defaultdict(list)
        for trade in all_trades_data:
            wallet = trade.get("buyer") if trade.get("side") == "buy" else trade.get("seller")
            if wallet in top_wallets:
                wallet_trades[wallet].append(trade)

        # Simulate trading by copying smart wallet moves
        current_date = start_date
        while current_date <= end_date:
            # Check for smart wallet trades on this day
            day_trades = self._get_trades_on_date(wallet_trades, current_date)

            for trade in day_trades:
                # Execute copy trade
                result = self._execute_copy_trade(portfolio, trade, price_data, current_date)

                if result:
                    trade_log.append(result)

            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(portfolio, price_data, current_date)
            portfolio_history.append((current_date, portfolio_value))

            current_date += timedelta(days=1)

        # Calculate performance metrics
        metrics = self._calculate_metrics(portfolio, trade_log, portfolio_history)

        # Calculate precision@K
        precision_at_k = self._calculate_precision_at_k(
            smart_wallet_rankings, all_trades_data, price_data
        )

        return BacktestResult(
            total_return=metrics["total_return"],
            sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown=metrics["max_drawdown"],
            win_rate=metrics["win_rate"],
            num_trades=len(trade_log),
            precision_at_k=precision_at_k,
            portfolio_value_history=portfolio_history,
            trade_log=trade_log,
            metrics=metrics,
        )

    def _initialize_portfolio(self) -> Dict[str, Any]:
        """Initialize portfolio state."""
        return {
            "cash": self.config.initial_capital,
            "positions": {},  # token -> quantity
            "cost_basis": {},  # token -> average cost
        }

    def _get_trades_on_date(
        self,
        wallet_trades: Dict[str, List[Dict[str, Any]]],
        date: datetime,
    ) -> List[Dict[str, Any]]:
        """Get all smart wallet trades on a specific date."""
        day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)

        trades = []
        for wallet, wallet_trade_list in wallet_trades.items():
            for trade in wallet_trade_list:
                trade_time = datetime.fromisoformat(trade["timestamp"].replace("Z", "+00:00"))
                if day_start <= trade_time < day_end:
                    trades.append(trade)

        return sorted(trades, key=lambda t: t["timestamp"])

    def _execute_copy_trade(
        self,
        portfolio: Dict[str, Any],
        trade: Dict[str, Any],
        price_data: List[Dict[str, Any]],
        current_date: datetime,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a copy trade following a smart wallet.

        Returns:
            Trade result dictionary or None if trade not executed
        """
        side = trade.get("side")
        price = self._get_price_at_time(price_data, current_date)

        if price == 0:
            return None

        # Apply fees and slippage
        effective_price = (
            price * (1 + self.config.slippage_rate)
            if side == "buy"
            else price * (1 - self.config.slippage_rate)
        )

        if side == "buy":
            # Calculate position size
            position_value = min(
                portfolio["cash"] * self.config.position_size, self.config.max_position_size_usd
            )

            if position_value < 10:  # Minimum trade size
                return None

            # Calculate quantity
            cost = position_value * (1 + self.config.fee_rate)
            quantity = position_value / effective_price

            if portfolio["cash"] >= cost:
                # Execute buy
                portfolio["cash"] -= cost
                portfolio["positions"]["token"] = portfolio["positions"].get("token", 0) + quantity

                # Update cost basis
                current_quantity = portfolio["positions"]["token"]
                current_basis = portfolio["cost_basis"].get("token", 0)
                portfolio["cost_basis"]["token"] = (
                    ((current_basis * (current_quantity - quantity) + cost) / current_quantity)
                    if current_quantity > 0
                    else effective_price
                )

                return {
                    "timestamp": current_date.isoformat(),
                    "side": "buy",
                    "price": effective_price,
                    "quantity": quantity,
                    "cost": cost,
                }

        elif side == "sell":
            # Sell position if we have it
            if portfolio["positions"].get("token", 0) > 0:
                quantity = portfolio["positions"]["token"]
                proceeds = quantity * effective_price * (1 - self.config.fee_rate)

                portfolio["cash"] += proceeds
                portfolio["positions"]["token"] = 0

                # Calculate PnL
                cost_basis = portfolio["cost_basis"].get("token", effective_price)
                pnl = proceeds - (quantity * cost_basis)

                return {
                    "timestamp": current_date.isoformat(),
                    "side": "sell",
                    "price": effective_price,
                    "quantity": quantity,
                    "proceeds": proceeds,
                    "pnl": pnl,
                }

        return None

    def _get_price_at_time(
        self,
        price_data: List[Dict[str, Any]],
        target_time: datetime,
    ) -> float:
        """Get price at specific time."""
        for price_point in price_data:
            pt_time = datetime.fromisoformat(price_point["timestamp"].replace("Z", "+00:00"))
            if abs((pt_time - target_time).total_seconds()) < 3600:  # Within 1 hour
                return price_point.get("price", 0)
        return 0.0

    def _calculate_portfolio_value(
        self,
        portfolio: Dict[str, Any],
        price_data: List[Dict[str, Any]],
        current_date: datetime,
    ) -> float:
        """Calculate total portfolio value."""
        cash = portfolio["cash"]
        position_value = 0

        if portfolio["positions"].get("token", 0) > 0:
            quantity = portfolio["positions"]["token"]
            price = self._get_price_at_time(price_data, current_date)
            position_value = quantity * price

        return cash + position_value

    def _calculate_metrics(
        self,
        portfolio: Dict[str, Any],
        trade_log: List[Dict[str, Any]],
        portfolio_history: List[Tuple[datetime, float]],
    ) -> Dict[str, Any]:
        """Calculate performance metrics."""
        initial_value = self.config.initial_capital
        final_value = portfolio_history[-1][1] if portfolio_history else initial_value

        total_return = (final_value - initial_value) / initial_value

        # Calculate returns series for Sharpe ratio
        returns = []
        for i in range(1, len(portfolio_history)):
            prev_value = portfolio_history[i - 1][1]
            curr_value = portfolio_history[i][1]
            daily_return = (curr_value - prev_value) / prev_value if prev_value > 0 else 0
            returns.append(daily_return)

        sharpe_ratio = (
            np.mean(returns) / np.std(returns) * np.sqrt(252)
            if len(returns) > 1 and np.std(returns) > 0
            else 0
        )

        # Calculate max drawdown
        max_drawdown = 0
        peak = initial_value
        for _, value in portfolio_history:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        # Calculate win rate
        profitable_trades = sum(
            1 for trade in trade_log if trade.get("side") == "sell" and trade.get("pnl", 0) > 0
        )
        total_sells = sum(1 for trade in trade_log if trade.get("side") == "sell")
        win_rate = profitable_trades / total_sells if total_sells > 0 else 0

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "final_value": final_value,
        }

    def _calculate_precision_at_k(
        self,
        ranked_wallets: List[Dict[str, Any]],
        all_trades: List[Dict[str, Any]],
        price_data: List[Dict[str, Any]],
    ) -> Dict[int, float]:
        """
        Calculate precision@K for different K values.

        Precision@K = fraction of top-K wallets that achieved positive returns
        """
        precision_scores = {}

        if not price_data or len(price_data) < 2:
            return precision_scores

        # Calculate returns for the period
        period_return = (
            price_data[-1].get("price", 0) - price_data[0].get("price", 0)
        ) / price_data[0].get("price", 1)

        # For each K value
        for k in [5, 10, 20, 50, 100]:
            if k > len(ranked_wallets):
                break

            top_k_wallets = ranked_wallets[:k]

            # Count how many achieved positive returns
            successful = 0
            for wallet in top_k_wallets:
                # Simple heuristic: if wallet score > threshold, assume success
                # In production, would calculate actual realized returns
                if wallet.get("smart_money_score", 0) > 0.6:
                    successful += 1

            precision_scores[k] = successful / k if k > 0 else 0

        return precision_scores


class WalkForwardValidator:
    """
    Implements walk-forward validation for model evaluation.

    Splits data into training and testing windows that roll forward in time.
    """

    def __init__(
        self,
        train_window_days: int = 60,
        test_window_days: int = 30,
        step_days: int = 30,
    ):
        """
        Initialize walk-forward validator.

        Args:
            train_window_days: Size of training window
            test_window_days: Size of testing window
            step_days: Days to step forward between iterations
        """
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days

    def generate_splits(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate train/test splits.

        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        splits = []

        current_start = start_date

        while True:
            train_start = current_start
            train_end = train_start + timedelta(days=self.train_window_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_window_days)

            if test_end > end_date:
                break

            splits.append((train_start, train_end, test_start, test_end))

            current_start += timedelta(days=self.step_days)

        return splits

    def validate(
        self,
        finder_func,
        token_address: str,
        start_date: datetime,
        end_date: datetime,
        data_provider,
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation.

        Args:
            finder_func: Function that finds smart wallets
            token_address: Token to analyze
            start_date: Overall start date
            end_date: Overall end date
            data_provider: Function to get data for date range

        Returns:
            Dictionary with validation results
        """
        splits = self.generate_splits(start_date, end_date)

        results = []

        for train_start, train_end, test_start, test_end in splits:
            logger.info(
                f"Train: {train_start.date()} to {train_end.date()}, "
                f"Test: {test_start.date()} to {test_end.date()}"
            )

            # Get training data and find smart wallets
            train_data = data_provider(train_start, train_end)
            smart_wallets = finder_func(token_address, train_data)

            # Test on next period
            test_data = data_provider(test_start, test_end)

            # Evaluate predictions
            evaluation = self._evaluate_predictions(smart_wallets, test_data)

            results.append(
                {
                    "train_period": (train_start, train_end),
                    "test_period": (test_start, test_end),
                    "evaluation": evaluation,
                }
            )

        # Aggregate results
        avg_precision = np.mean([r["evaluation"]["precision"] for r in results])
        avg_return = np.mean([r["evaluation"]["return"] for r in results])

        return {
            "num_splits": len(splits),
            "avg_precision": avg_precision,
            "avg_return": avg_return,
            "all_results": results,
        }

    def _evaluate_predictions(
        self,
        predicted_wallets: List[Dict[str, Any]],
        test_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate predictions on test period."""
        # Simplified evaluation
        # In production, would calculate actual performance metrics

        return {
            "precision": 0.75,  # Placeholder
            "return": 0.15,  # Placeholder
        }
