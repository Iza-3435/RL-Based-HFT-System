"""Performance analysis for backtesting."""

import numpy as np
from typing import Dict, List

from src.core.logging_config import get_logger

logger = get_logger()


class PerformanceAnalyzer:
    """Analyzes backtest performance."""

    def calculate_metrics(
        self,
        trades: List[Dict],
        equity_curve: List[float],
        initial_capital: float
    ) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not trades:
            return self._empty_metrics()

        returns = self._calculate_returns(equity_curve)

        metrics = {
            'total_pnl': equity_curve[-1] - initial_capital if equity_curve else 0,
            'winning_trades': sum(1 for t in trades if t['pnl'] > 0),
            'losing_trades': sum(1 for t in trades if t['pnl'] <= 0),
            'win_rate': self._calculate_win_rate(trades),
            'avg_trade_pnl': np.mean([t['pnl'] for t in trades]),
            'sharpe_ratio': self._calculate_sharpe(returns),
            'max_drawdown': self._calculate_max_drawdown(equity_curve),
            'total_commission': sum(t.get('commission', 0) for t in trades),
            'total_slippage': sum(t.get('slippage', 0) for t in trades),
            'sortino_ratio': self._calculate_sortino(returns),
            'calmar_ratio': self._calculate_calmar(returns, equity_curve),
            'profit_factor': self._calculate_profit_factor(trades)
        }

        logger.verbose("Performance metrics calculated", trades=len(trades))
        return metrics

    def _calculate_returns(self, equity_curve: List[float]) -> np.ndarray:
        """Calculate returns from equity curve."""
        if len(equity_curve) < 2:
            return np.array([])

        returns = np.diff(equity_curve) / equity_curve[:-1]
        return returns

    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate."""
        if not trades:
            return 0.0

        winning = sum(1 for t in trades if t['pnl'] > 0)
        return (winning / len(trades)) * 100

    def _calculate_sharpe(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - (risk_free_rate / 252)

        if np.std(excess_returns) == 0:
            return 0.0

        sharpe = np.sqrt(252) * (np.mean(excess_returns) / np.std(excess_returns))
        return sharpe

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown."""
        if not equity_curve:
            return 0.0

        peak = equity_curve[0]
        max_dd = 0.0

        for value in equity_curve:
            if value > peak:
                peak = value

            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return max_dd * 100

    def _calculate_sortino(self, returns: np.ndarray, target_return: float = 0.0) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - target_return
        downside_returns = returns[returns < target_return]

        if len(downside_returns) == 0:
            return 0.0

        downside_std = np.std(downside_returns)

        if downside_std == 0:
            return 0.0

        sortino = np.sqrt(252) * (np.mean(excess_returns) / downside_std)
        return sortino

    def _calculate_calmar(self, returns: np.ndarray, equity_curve: List[float]) -> float:
        """Calculate Calmar ratio."""
        if len(returns) == 0:
            return 0.0

        annual_return = np.mean(returns) * 252
        max_dd = self._calculate_max_drawdown(equity_curve) / 100

        if max_dd == 0:
            return 0.0

        return annual_return / max_dd

    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Calculate profit factor."""
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))

        if gross_loss == 0:
            return 0.0 if gross_profit == 0 else float('inf')

        return gross_profit / gross_loss

    def _empty_metrics(self) -> Dict:
        """Return empty metrics."""
        return {
            'total_pnl': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_trade_pnl': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'total_commission': 0,
            'total_slippage': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'profit_factor': 0
        }
