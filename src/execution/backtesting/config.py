"""Backtesting configuration."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict
from enum import Enum

from src.core.logging_config import get_logger

logger = get_logger()


class BacktestMode(Enum):
    """Backtest execution mode."""
    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"
    STRESS_TEST = "stress_test"
    WALK_FORWARD = "walk_forward"


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    start_date: datetime
    end_date: datetime
    symbols: List[str]
    venues: List[str]
    initial_capital: float = 1_000_000
    mode: BacktestMode = BacktestMode.HISTORICAL
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    max_position_size: int = 10000
    enable_risk_checks: bool = True
    enable_cost_model: bool = True
    data_frequency: str = "1min"
    warmup_period: int = 100
    training_window_days: int = 60
    testing_window_days: int = 20
    reoptimization_frequency: int = 20

    def validate(self):
        """Validate configuration."""
        if self.end_date <= self.start_date:
            raise ValueError("End date must be after start date")

        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")

        if not self.symbols:
            raise ValueError("Must specify at least one symbol")

        logger.verbose("Backtest config validated", symbols=len(self.symbols))


@dataclass
class BacktestResult:
    """Results from backtest execution."""
    config: BacktestConfig = None
    start_time: datetime = None
    end_time: datetime = None
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_trade_pnl: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    final_capital: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    returns: List[float] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)
    monte_carlo_analysis: Dict = field(default_factory=dict)
    stress_test_summary: Dict = field(default_factory=dict)
    walk_forward_analysis: Dict = field(default_factory=dict)
    strategy_performance: Dict = field(default_factory=dict)
    venue_performance: Dict = field(default_factory=dict)

    def summary(self) -> Dict:
        """Get summary statistics."""
        return {
            'total_pnl': self.total_pnl,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'avg_trade_pnl': self.avg_trade_pnl,
            'final_capital': self.final_capital
        }
