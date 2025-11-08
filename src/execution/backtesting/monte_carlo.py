"""Monte Carlo simulation for backtesting."""

import asyncio
from typing import Dict, List
import numpy as np

from src.core.logging_config import get_logger
from src.execution.backtesting.config import BacktestConfig, BacktestResult

logger = get_logger()


class MonteCarloSimulator:
    """Monte Carlo simulation for strategy validation."""

    def __init__(self, config: BacktestConfig):
        self.config = config

    async def run(
        self,
        base_result: BacktestResult,
        num_simulations: int = 1000
    ) -> BacktestResult:
        """Run Monte Carlo simulation."""
        logger.info(f"Running Monte Carlo with {num_simulations} paths")

        if not base_result.returns:
            logger.warning("No returns data for Monte Carlo")
            return base_result

        simulation_results = []

        for i in range(num_simulations):
            if i % 100 == 0:
                logger.verbose(f"Monte Carlo progress: {i}/{num_simulations}")

            sim_result = self._simulate_path(base_result.returns)
            simulation_results.append(sim_result)

        base_result.monte_carlo_analysis = self._aggregate_results(
            simulation_results
        )

        logger.info("Monte Carlo simulation complete")
        return base_result

    def _simulate_path(self, returns: List[float]) -> Dict:
        """Simulate single path by bootstrapping returns."""
        simulated_returns = np.random.choice(
            returns,
            size=len(returns),
            replace=True
        )

        equity_curve = self.config.initial_capital * np.cumprod(
            1 + simulated_returns
        )

        return {
            'total_return': (equity_curve[-1] / self.config.initial_capital) - 1,
            'sharpe_ratio': self._calculate_sharpe(simulated_returns),
            'max_drawdown': self._calculate_max_drawdown(equity_curve),
            'final_equity': equity_curve[-1]
        }

    def _aggregate_results(self, simulations: List[Dict]) -> Dict:
        """Aggregate Monte Carlo results."""
        returns = [s['total_return'] for s in simulations]
        sharpes = [s['sharpe_ratio'] for s in simulations]
        drawdowns = [s['max_drawdown'] for s in simulations]

        return {
            'simulation_count': len(simulations),
            'return_distribution': {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'percentiles': {
                    '5th': np.percentile(returns, 5),
                    '25th': np.percentile(returns, 25),
                    'median': np.percentile(returns, 50),
                    '75th': np.percentile(returns, 75),
                    '95th': np.percentile(returns, 95)
                }
            },
            'sharpe_distribution': {
                'mean': np.mean(sharpes),
                'std': np.std(sharpes),
                'percentiles': {
                    '5th': np.percentile(sharpes, 5),
                    'median': np.percentile(sharpes, 50),
                    '95th': np.percentile(sharpes, 95)
                }
            },
            'drawdown_distribution': {
                'mean': np.mean(drawdowns),
                'std': np.std(drawdowns),
                'percentiles': {
                    '5th': np.percentile(drawdowns, 5),
                    'median': np.percentile(drawdowns, 50),
                    '95th': np.percentile(drawdowns, 95)
                }
            },
            'positive_scenarios': sum(1 for r in returns if r > 0) / len(returns),
            'value_at_risk_95': -np.percentile(returns, 5),
            'conditional_var_95': -np.mean([r for r in returns if r <= np.percentile(returns, 5)])
        }

    def _calculate_sharpe(self, returns: np.ndarray, rf_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - (rf_rate / 252)

        if np.std(excess_returns) == 0:
            return 0.0

        return np.sqrt(252) * (np.mean(excess_returns) / np.std(excess_returns))

    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        rolling_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - rolling_max) / rolling_max

        return abs(np.min(drawdown))
