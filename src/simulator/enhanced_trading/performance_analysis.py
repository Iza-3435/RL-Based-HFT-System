"""Performance analysis and attribution with latency impact."""

import logging
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from simulator.trading_simulator import Fill, OrderSide

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Comprehensive performance analysis with latency attribution."""

    def __init__(self, execution_engine, strategies: Dict, fill_history: List[Fill]):
        self.execution_engine = execution_engine
        self.strategies = strategies
        self.fill_history = fill_history

    def calculate_enhanced_pnl_attribution(
        self, total_pnl: float
    ) -> Dict[str, Any]:
        """Calculate P&L attribution including latency costs."""
        attribution = {
            "total_pnl": total_pnl,
            "gross_pnl": 0.0,
            "latency_costs": 0.0,
            "execution_costs": 0.0,
            "by_venue": defaultdict(float),
            "by_strategy": defaultdict(float),
        }

        if hasattr(self.execution_engine, "get_latency_cost_analysis"):
            latency_costs = self.execution_engine.get_latency_cost_analysis()
            attribution["latency_costs"] = latency_costs.get(
                "total_latency_cost_usd", 0
            )

        for fill in self.fill_history:
            venue_pnl = 0
            if hasattr(fill, "latency_cost_bps"):
                latency_cost = fill.latency_cost_bps * fill.price * fill.quantity / 10000
                venue_pnl -= latency_cost

            attribution["by_venue"][fill.venue] += venue_pnl

        for strategy_name, strategy in self.strategies.items():
            strategy_pnl = strategy.get_total_pnl()

            strategy_fills = [
                f
                for f in self.fill_history
                if getattr(f, "order_id", "").startswith(strategy_name.upper()[:3])
            ]

            latency_impact = sum(
                getattr(f, "latency_cost_bps", 0) * f.price * f.quantity / 10000
                for f in strategy_fills
            )

            attribution["by_strategy"][strategy_name] = {
                "gross_pnl": strategy_pnl["total_pnl"],
                "latency_impact": -latency_impact,
                "net_pnl": strategy_pnl["total_pnl"] - latency_impact,
            }

        return attribution

    def analyze_strategy_latency_impact(self) -> Dict[str, Any]:
        """Analyze how latency affects each trading strategy."""
        impact_analysis = {}

        for strategy_name, strategy in self.strategies.items():
            strategy_fills = [
                f
                for f in self.fill_history
                if hasattr(f, "order_id") and strategy_name.upper()[:3] in f.order_id
            ]

            if not strategy_fills:
                continue

            latencies = [getattr(f, "latency_us", 1000) for f in strategy_fills]
            slippages = [getattr(f, "slippage_bps", 0) for f in strategy_fills]

            if strategy_name == "arbitrage":
                impact_analysis[strategy_name] = {
                    "avg_latency_us": np.mean(latencies),
                    "opportunity_capture_rate": getattr(
                        strategy, "opportunities_captured", 0
                    )
                    / max(getattr(strategy, "opportunities_detected", 1), 1),
                    "latency_sensitivity": self._calculate_latency_sensitivity(
                        latencies, slippages
                    ),
                    "optimal_latency_threshold_us": 300,
                }

            elif strategy_name == "market_making":
                impact_analysis[strategy_name] = {
                    "avg_latency_us": np.mean(latencies),
                    "spread_capture_efficiency": getattr(strategy, "spread_captured", 0)
                    / max(len(strategy_fills), 1),
                    "maker_ratio": self._calculate_maker_ratio(strategy_fills),
                    "optimal_latency_threshold_us": 800,
                }

            elif strategy_name == "momentum":
                impact_analysis[strategy_name] = {
                    "avg_latency_us": np.mean(latencies),
                    "signal_decay_impact": self._estimate_signal_decay(latencies),
                    "execution_efficiency": self._calculate_execution_efficiency(
                        strategy_fills
                    ),
                    "optimal_latency_threshold_us": 1500,
                }

        return impact_analysis

    def generate_optimization_recommendations(
        self, strategy_impact: Dict
    ) -> List[str]:
        """Generate actionable optimization recommendations."""
        recommendations = []

        if hasattr(self.execution_engine, "latency_analytics"):
            if hasattr(self.execution_engine.latency_analytics, "_generate_recommendations"):
                recommendations.extend(
                    self.execution_engine.latency_analytics._generate_recommendations()
                )

        for strategy_name, impact in strategy_impact.items():
            avg_latency = impact.get("avg_latency_us", 0)
            optimal_threshold = impact.get("optimal_latency_threshold_us", 1000)

            if avg_latency > optimal_threshold * 1.5:
                recommendations.append(
                    f"{strategy_name.title()} strategy experiencing high latency "
                    f"({avg_latency:.0f}μs vs optimal {optimal_threshold}μs). "
                    f"Consider venue optimization or strategy parameters."
                )

        if hasattr(self.execution_engine, "get_venue_latency_rankings"):
            rankings = self.execution_engine.get_venue_latency_rankings()
            if len(rankings) > 1:
                best_latency = rankings[0][1]
                worst_latency = rankings[-1][1]

                if worst_latency > best_latency * 2:
                    recommendations.append(
                        f"Consider rebalancing order flow: {rankings[-1][0]} showing "
                        f"{worst_latency:.0f}μs vs best venue {rankings[0][0]} at {best_latency:.0f}μs"
                    )

        if hasattr(self.execution_engine, "get_latency_cost_analysis"):
            cost_analysis = self.execution_engine.get_latency_cost_analysis()
            potential_savings = cost_analysis.get("potential_savings_analysis", {})

            if potential_savings.get("potential_improvement_pct", 0) > 20:
                recommendations.append(
                    f"Significant latency optimization opportunity: "
                    f"{potential_savings['potential_improvement_pct']:.1f}% improvement possible, "
                    f"estimated savings: {potential_savings.get('estimated_savings_bps', 0):.2f} bps"
                )

        return recommendations

    def _calculate_latency_sensitivity(
        self, latencies: List[float], slippages: List[float]
    ) -> float:
        """Calculate how sensitive slippage is to latency changes."""
        if len(latencies) < 2 or len(slippages) < 2:
            return 0.0

        latency_array = np.array(latencies)
        slippage_array = np.array(slippages)

        if np.std(latency_array) == 0 or np.std(slippage_array) == 0:
            return 0.0

        correlation = np.corrcoef(latency_array, slippage_array)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0

    def _calculate_maker_ratio(self, fills: List[Fill]) -> float:
        """Calculate ratio of maker vs taker fills."""
        if not fills:
            return 0.0

        maker_fills = sum(1 for f in fills if getattr(f, "rebate", 0) > 0)
        return maker_fills / len(fills)

    def _estimate_signal_decay(self, latencies: List[float]) -> float:
        """Estimate signal decay impact from latency."""
        if not latencies:
            return 0.0

        avg_latency = np.mean(latencies)
        decay_factor = np.exp(-avg_latency / 500)
        signal_strength_loss = 1.0 - decay_factor

        return signal_strength_loss

    def _calculate_execution_efficiency(self, fills: List[Fill]) -> float:
        """Calculate execution efficiency metric."""
        if not fills:
            return 0.0

        total_efficiency = 0.0

        for fill in fills:
            slippage = getattr(fill, "slippage_bps", 0)
            latency = getattr(fill, "latency_us", 1000)

            slippage_efficiency = max(0, 1.0 - slippage / 10)
            latency_efficiency = max(0, 1.0 - latency / 2000)

            fill_efficiency = (slippage_efficiency + latency_efficiency) / 2
            total_efficiency += fill_efficiency

        return total_efficiency / len(fills)
