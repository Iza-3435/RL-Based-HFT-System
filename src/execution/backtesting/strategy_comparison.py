"""Strategy comparison framework."""

import asyncio
from typing import Dict, Callable
import numpy as np
from scipy import stats

from src.core.logging_config import get_logger
from src.execution.backtesting.config import BacktestConfig, BacktestMode, BacktestResult

logger = get_logger()


class StrategyComparator:
    """Compare different routing strategies and approaches."""

    def __init__(self):
        self.comparison_results = {}

    async def compare_routing_approaches(
        self,
        backtest_config: BacktestConfig,
        backtest_engine_class
    ) -> Dict:
        """Compare ML routing vs baseline approaches."""
        logger.info("Starting routing comparison")

        approaches = {
            'ml_optimized': self._create_ml_routing,
            'random_routing': self._create_random_routing,
            'static_routing': self._create_static_routing,
            'lowest_fee': self._create_lowest_fee_routing
        }

        results = {}

        for approach_name, approach_factory in approaches.items():
            logger.verbose(f"Testing approach: {approach_name}")

            engine = backtest_engine_class(backtest_config)

            result = await engine.run(
                strategy_factory=self._create_strategies,
                ml_predictor_factory=approach_factory
            )

            results[approach_name] = result

        comparison = self._analyze_comparison(results)
        self.comparison_results['routing_approaches'] = comparison

        logger.info("Routing comparison complete")
        return comparison

    def _create_ml_routing(self):
        """Create ML-optimized routing."""
        class MLRouter:
            async def predict_latency(self, venue, features):
                base_latencies = {
                    'NYSE': 800,
                    'NASDAQ': 900,
                    'CBOE': 1000,
                    'IEX': 1200,
                    'ARCA': 850
                }
                return base_latencies.get(venue, 1000) + np.random.normal(0, 50)

            async def get_best_venue(self, predictions):
                return min(predictions.items(), key=lambda x: x[1])[0]

        return MLRouter()

    def _create_random_routing(self):
        """Create random routing baseline."""
        class RandomRouter:
            def __init__(self):
                self.venues = ['NYSE', 'NASDAQ', 'CBOE', 'IEX', 'ARCA']

            async def predict_latency(self, venue, features):
                return 1000

            async def get_best_venue(self, predictions):
                return np.random.choice(self.venues)

        return RandomRouter()

    def _create_static_routing(self):
        """Create static routing (always same venue)."""
        class StaticRouter:
            async def predict_latency(self, venue, features):
                return 1000

            async def get_best_venue(self, predictions):
                return 'NYSE'

        return StaticRouter()

    def _create_lowest_fee_routing(self):
        """Create lowest-fee routing."""
        class LowestFeeRouter:
            def __init__(self):
                self.fee_schedule = {
                    'IEX': 0.0009,
                    'CBOE': 0.0028,
                    'NYSE': 0.0030,
                    'NASDAQ': 0.0030,
                    'ARCA': 0.0030
                }

            async def predict_latency(self, venue, features):
                return 1000

            async def get_best_venue(self, predictions):
                return min(self.fee_schedule.items(), key=lambda x: x[1])[0]

        return LowestFeeRouter()

    def _create_strategies(self):
        """Create standard strategy set."""
        return {}

    def _analyze_comparison(self, results: Dict[str, BacktestResult]) -> Dict:
        """Analyze routing comparison results."""
        analysis = {
            'performance_summary': {},
            'improvement_vs_baseline': {},
            'statistical_tests': {},
            'cost_analysis': {}
        }

        baseline = results.get('random_routing')

        for approach, result in results.items():
            analysis['performance_summary'][approach] = {
                'total_return': result.total_pnl / result.config.initial_capital,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'total_trades': result.total_trades
            }

            if baseline and approach != 'random_routing':
                baseline_return = baseline.total_pnl / baseline.config.initial_capital
                this_return = result.total_pnl / result.config.initial_capital

                analysis['improvement_vs_baseline'][approach] = {
                    'return_improvement': this_return - baseline_return,
                    'sharpe_improvement': result.sharpe_ratio - baseline.sharpe_ratio,
                    'drawdown_improvement': baseline.max_drawdown - result.max_drawdown
                }

            analysis['cost_analysis'][approach] = {
                'total_commission': result.total_commission,
                'total_slippage': result.total_slippage,
                'cost_per_trade': (
                    (result.total_commission + result.total_slippage) /
                    max(result.total_trades, 1)
                )
            }

        if 'ml_optimized' in results and 'random_routing' in results:
            ml_result = results['ml_optimized']
            random_result = results['random_routing']

            if ml_result.returns and random_result.returns:
                t_stat, p_value = stats.ttest_ind(
                    ml_result.returns,
                    random_result.returns
                )

                analysis['statistical_tests']['ml_vs_random'] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }

        return analysis
