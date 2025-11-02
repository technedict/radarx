"""
Backtesting Framework

Provides comprehensive backtesting infrastructure for model validation and strategy simulation.
"""

from radarx.backtesting.engine import BacktestEngine
from radarx.backtesting.strategy import StrategySimulator
from radarx.backtesting.labeler import OutcomeLabeler
from radarx.backtesting.ledger import LearningLedger

__all__ = [
    'BacktestEngine',
    'StrategySimulator',
    'OutcomeLabeler',
    'LearningLedger',
]
