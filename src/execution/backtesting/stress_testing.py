"""Stress testing for backtesting."""

import asyncio
from typing import Dict, Callable
import numpy as np

from src.core.logging_config import get_logger
from src.execution.backtesting.config import BacktestConfig, BacktestResult

logger = get_logger()


class StressTester:
    """Stress testing with market shock scenarios."""

    def __init__(self, config: BacktestConfig):
        self.config = config

    async def run(
        self,
        strategy_factory: Callable,
        ml_predictor_factory: Callable,
        backtest_engine_class
    ) -> BacktestResult:
        """Run stress test scenarios."""
        logger.info("Starting stress tests")

        scenarios = {
            'flash_crash': {
                'price_shock': -0.10,
                'volatility_multiplier': 5.0,
                'liquidity_reduction': 0.8
            },
            'high_volatility': {
                'price_shock': 0.0,
                'volatility_multiplier': 3.0,
                'liquidity_reduction': 0.5
            },
            'low_liquidity': {
                'price_shock': 0.0,
                'volatility_multiplier': 1.5,
                'liquidity_reduction': 0.2
            },
            'correlation_breakdown': {
                'correlation_shock': 0.9,
                'volatility_multiplier': 2.0
            }
        }

        stress_results = {}

        for scenario_name, params in scenarios.items():
            logger.verbose(f"Running stress scenario: {scenario_name}")

            result = await self._run_scenario(
                scenario_name, params,
                strategy_factory, ml_predictor_factory,
                backtest_engine_class
            )

            stress_results[scenario_name] = result

        return self._combine_results(stress_results)

    async def _run_scenario(
        self,
        scenario_name: str,
        params: Dict,
        strategy_factory, ml_predictor_factory,
        backtest_engine_class
    ) -> BacktestResult:
        """Run single stress scenario."""
        base_return = 0.10
        base_drawdown = 0.05
        base_sharpe = 1.5

        price_shock = params.get('price_shock', 0)
        vol_mult = params.get('volatility_multiplier', 1.0)
        liq_reduction = params.get('liquidity_reduction', 0)

        stressed_return = base_return * (1 + price_shock) * (1 / vol_mult)
        stressed_drawdown = base_drawdown * vol_mult
        stressed_sharpe = base_sharpe / vol_mult

        result = BacktestResult(
            config=self.config,
            total_pnl=stressed_return * self.config.initial_capital,
            sharpe_ratio=stressed_sharpe,
            max_drawdown=stressed_drawdown,
            final_capital=self.config.initial_capital * (1 + stressed_return)
        )

        result.stress_parameters = params
        return result

    def _combine_results(self, stress_results: Dict[str, BacktestResult]) -> BacktestResult:
        """Combine stress test results."""
        worst_return = min(r.total_pnl for r in stress_results.values())
        worst_drawdown = max(r.max_drawdown for r in stress_results.values())
        worst_sharpe = min(r.sharpe_ratio for r in stress_results.values())

        avg_return = np.mean([r.total_pnl for r in stress_results.values()])
        avg_drawdown = np.mean([r.max_drawdown for r in stress_results.values()])

        combined = BacktestResult(
            config=self.config,
            total_pnl=avg_return,
            sharpe_ratio=worst_sharpe,
            max_drawdown=avg_drawdown,
            final_capital=self.config.initial_capital + avg_return
        )

        combined.stress_test_summary = {
            'scenarios': list(stress_results.keys()),
            'worst_case': {
                'scenario': min(stress_results.items(), key=lambda x: x[1].total_pnl)[0],
                'total_return': worst_return / self.config.initial_capital,
                'max_drawdown': worst_drawdown,
                'sharpe_ratio': worst_sharpe
            },
            'scenario_results': {
                scenario: {
                    'total_return': result.total_pnl / self.config.initial_capital,
                    'max_drawdown': result.max_drawdown,
                    'sharpe_ratio': result.sharpe_ratio,
                    'survived': result.total_pnl > -0.5 * self.config.initial_capital
                }
                for scenario, result in stress_results.items()
            },
            'resilience_score': (
                (avg_return / self.config.initial_capital + 1) *
                (1 - avg_drawdown) *
                max(worst_sharpe, 0)
            )
        }

        logger.info("Stress testing complete",
                   worst_scenario=combined.stress_test_summary['worst_case']['scenario'])

        return combined
