"""Backtesting framework for strategy validation."""

from src.execution.backtesting.config import BacktestConfig, BacktestResult, BacktestMode
from src.execution.backtesting.engine import BacktestEngine
from src.execution.backtesting.data_manager import HistoricalDataManager, StressDataManager
from src.execution.backtesting.execution_sim import BacktestExecutionSimulator
from src.execution.backtesting.performance import PerformanceAnalyzer
from src.execution.backtesting.walk_forward import WalkForwardOptimizer
from src.execution.backtesting.monte_carlo import MonteCarloSimulator
from src.execution.backtesting.stress_testing import StressTester
from src.execution.backtesting.strategy_comparison import StrategyComparator
from src.execution.backtesting.report_generator import ReportGenerator

__all__ = [
    'BacktestConfig',
    'BacktestResult',
    'BacktestMode',
    'BacktestEngine',
    'HistoricalDataManager',
    'StressDataManager',
    'BacktestExecutionSimulator',
    'PerformanceAnalyzer',
    'WalkForwardOptimizer',
    'MonteCarloSimulator',
    'StressTester',
    'StrategyComparator',
    'ReportGenerator',
]
