"""
Backtesting Framework

Provides comprehensive backtesting infrastructure for model validation and strategy simulation.
"""

from radarx.backtesting.engine import BacktestEngine
from radarx.backtesting.labeler import OutcomeLabeler
from radarx.backtesting.ledger import LearningLedger
from radarx.backtesting.strategy import StrategySimulator

__all__ = [
    "BacktestEngine",
    "StrategySimulator",
    "OutcomeLabeler",
    "LearningLedger",
]
