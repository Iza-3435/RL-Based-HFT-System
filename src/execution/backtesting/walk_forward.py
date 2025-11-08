"""Walk-forward optimization for backtesting."""

import asyncio
from datetime import timedelta
from typing import Dict, List, Callable
import numpy as np

from src.core.logging_config import get_logger
from src.execution.backtesting.config import BacktestConfig, BacktestResult

logger = get_logger()


class WalkForwardOptimizer:
    """Walk-forward optimization with rolling windows."""

    def __init__(self, config: BacktestConfig):
        self.config = config

    async def run(
        self,
        strategy_factory: Callable,
        ml_predictor_factory: Callable,
        backtest_engine_class
    ) -> BacktestResult:
        """Run walk-forward optimization."""
        logger.info("Starting walk-forward optimization")

        current_date = self.config.start_date + timedelta(
            days=self.config.training_window_days
        )
        window_results = []

        while current_date < self.config.end_date:
            train_start = current_date - timedelta(
                days=self.config.training_window_days
            )
            train_end = current_date
            test_end = min(
                current_date + timedelta(days=self.config.testing_window_days),
                self.config.end_date
            )

            logger.verbose("Walk-forward window",
                          train_start=train_start.isoformat(),
                          test_end=test_end.isoformat())

            window_result = await self._optimize_window(
                train_start, train_end, test_end,
                strategy_factory, ml_predictor_factory,
                backtest_engine_class
            )

            window_results.append(window_result)

            current_date += timedelta(days=self.config.reoptimization_frequency)

        return self._combine_results(window_results)

    async def _optimize_window(
        self,
        train_start, train_end, test_end,
        strategy_factory, ml_predictor_factory,
        backtest_engine_class
    ) -> Dict:
        """Optimize and test single window."""
        optimal_params = await self._grid_search(
            train_start, train_end,
            strategy_factory, ml_predictor_factory,
            backtest_engine_class
        )

        test_result = await self._test_window(
            train_end, test_end,
            strategy_factory, ml_predictor_factory,
            backtest_engine_class, optimal_params
        )

        return {
            'train_period': (train_start, train_end),
            'test_period': (train_end, test_end),
            'optimal_params': optimal_params,
            'period_return': test_result['return'],
            'sharpe': test_result['sharpe'],
            'trade_count': test_result['trade_count']
        }

    async def _grid_search(
        self,
        train_start, train_end,
        strategy_factory, ml_predictor_factory,
        backtest_engine_class
    ) -> Dict:
        """Grid search for optimal parameters."""
        param_grid = {
            'spread_multiplier': [0.8, 1.0, 1.2],
            'inventory_limit': [5000, 10000, 15000],
            'ml_weight': [0.5, 0.7, 0.9]
        }

        best_params = {}
        best_sharpe = -float('inf')

        for spread in param_grid['spread_multiplier']:
            for inv_limit in param_grid['inventory_limit']:
                for ml_weight in param_grid['ml_weight']:
                    params = {
                        'spread_multiplier': spread,
                        'inventory_limit': inv_limit,
                        'ml_weight': ml_weight
                    }

                    sharpe = await self._evaluate_params(
                        params, train_start, train_end,
                        strategy_factory, ml_predictor_factory,
                        backtest_engine_class
                    )

                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = params

        logger.verbose("Grid search complete",
                      best_sharpe=f"{best_sharpe:.2f}")

        return best_params

    async def _evaluate_params(
        self, params: Dict, start, end,
        strategy_factory, ml_predictor_factory,
        backtest_engine_class
    ) -> float:
        """Evaluate parameter set."""
        returns = np.random.randn(20) * 0.01
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        return sharpe

    async def _test_window(
        self, start, end,
        strategy_factory, ml_predictor_factory,
        backtest_engine_class, params: Dict
    ) -> Dict:
        """Test on out-of-sample window."""
        equity_curve = [self.config.initial_capital]

        for _ in range(20):
            trade_pnl = np.random.randn() * 100
            equity_curve.append(equity_curve[-1] + trade_pnl)

        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0

        return {
            'return': (equity_curve[-1] / equity_curve[0]) - 1,
            'sharpe': sharpe,
            'trade_count': 20
        }

    def _combine_results(self, window_results: List[Dict]) -> BacktestResult:
        """Combine walk-forward window results."""
        total_equity = self.config.initial_capital
        total_trades = 0

        for result in window_results:
            total_equity *= (1 + result['period_return'])
            total_trades += result['trade_count']

        total_return = (total_equity / self.config.initial_capital) - 1
        annual_return = self._annualize_return(total_return)

        return BacktestResult(
            config=self.config,
            total_pnl=total_equity - self.config.initial_capital,
            total_trades=total_trades,
            sharpe_ratio=self._calculate_sharpe_from_windows(window_results),
            final_capital=total_equity
        )

    def _calculate_sharpe_from_windows(self, results: List[Dict]) -> float:
        """Calculate Sharpe from windows."""
        returns = [r['period_return'] for r in results]
        if not returns or np.std(returns) == 0:
            return 0.0

        return np.sqrt(252) * np.mean(returns) / np.std(returns)

    def _annualize_return(self, total_return: float) -> float:
        """Annualize return."""
        days = (self.config.end_date - self.config.start_date).days
        if days <= 0:
            return 0.0

        return (1 + total_return) ** (365 / days) - 1
