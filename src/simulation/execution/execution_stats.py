"""Execution statistics and performance tracking."""

import numpy as np
from typing import Dict, Any, List, Tuple
from collections import defaultdict


class ExecutionStatistics:
    """Track and analyze execution performance."""

    def __init__(self) -> None:
        self.fill_count = 0
        self.total_latency_us = 0.0
        self.total_slippage_bps = 0.0
        self.latency_cost_bps = 0.0
        self.venue_performance = defaultdict(
            lambda: {
                "fills": 0,
                "total_latency": 0.0,
                "total_slippage": 0.0,
                "prediction_errors": [],
            }
        )

    def update(self, fill: Any, latency_breakdown: Any) -> None:
        """Update statistics with new fill."""
        self.fill_count += 1
        self.total_latency_us += latency_breakdown.total_latency_us
        self.total_slippage_bps += fill.slippage_bps
        self.latency_cost_bps += fill.latency_cost_bps

        venue_stats = self.venue_performance[fill.venue]
        venue_stats["fills"] += 1
        venue_stats["total_latency"] += latency_breakdown.total_latency_us
        venue_stats["total_slippage"] += fill.slippage_bps

        if latency_breakdown.prediction_error_us is not None:
            venue_stats["prediction_errors"].append(latency_breakdown.prediction_error_us)

    def get_execution_stats(self, latency_simulator: Any) -> Dict[str, Any]:
        """Get comprehensive execution statistics."""
        base_stats = {
            "total_fills": self.fill_count,
            "avg_latency_us": self.total_latency_us / max(self.fill_count, 1),
            "avg_slippage_bps": self.total_slippage_bps / max(self.fill_count, 1),
            "avg_latency_cost_bps": self.latency_cost_bps / max(self.fill_count, 1),
            "fill_rate": 1.0,
        }

        venue_stats = {}
        for venue, stats in self.venue_performance.items():
            if stats["fills"] > 0:
                venue_stats[venue] = {
                    "fills": stats["fills"],
                    "avg_latency_us": stats["total_latency"] / stats["fills"],
                    "avg_slippage_bps": stats["total_slippage"] / stats["fills"],
                    "prediction_accuracy": self._calculate_prediction_accuracy(stats),
                }

        latency_stats = {
            "prediction_accuracy": latency_simulator.get_prediction_accuracy_stats(),
            "congestion_analysis": latency_simulator.get_congestion_analysis(),
        }

        return {
            "execution_stats": base_stats,
            "venue_performance": venue_stats,
            "latency_analysis": latency_stats,
        }

    def _calculate_prediction_accuracy(self, venue_stats: Dict) -> Dict[str, float]:
        """Calculate prediction accuracy for a venue."""
        errors = venue_stats["prediction_errors"]
        if not errors:
            return {}

        avg_latency = venue_stats["total_latency"] / venue_stats["fills"]

        return {
            "mean_error_us": np.mean(errors),
            "rmse_us": np.sqrt(np.mean(np.array(errors) ** 2)),
            "mape_pct": np.mean(errors) / avg_latency * 100,
            "within_10pct": (np.sum(np.array(errors) < avg_latency * 0.1) / len(errors) * 100),
        }

    def get_venue_rankings(self, latency_simulator: Any) -> List[Tuple[str, float]]:
        """Get venues ranked by average latency performance."""
        rankings = []

        for venue in latency_simulator.venues:
            stats = latency_simulator.get_venue_latency_stats(venue, window_minutes=10)
            if stats:
                rankings.append((venue, stats["mean_us"]))

        rankings.sort(key=lambda x: x[1])
        return rankings

    def get_cost_analysis(self) -> Dict[str, Any]:
        """Analyze costs specifically attributed to latency."""
        total_fills = max(self.fill_count, 1)
        avg_latency_cost_bps = self.latency_cost_bps / total_fills

        avg_trade_value = 50000
        latency_cost_per_trade = avg_trade_value * avg_latency_cost_bps / 10000
        total_latency_cost = latency_cost_per_trade * total_fills

        venue_costs = {}
        for venue, stats in self.venue_performance.items():
            if stats["fills"] > 0:
                venue_avg_latency = stats["total_latency"] / stats["fills"]
                venue_cost_bps = venue_avg_latency / 100 * 0.1
                venue_costs[venue] = {
                    "avg_latency_us": venue_avg_latency,
                    "estimated_cost_bps": venue_cost_bps,
                    "fills": stats["fills"],
                }

        return {
            "total_latency_cost_usd": total_latency_cost,
            "avg_latency_cost_per_trade_usd": latency_cost_per_trade,
            "avg_latency_cost_bps": avg_latency_cost_bps,
            "venue_cost_breakdown": venue_costs,
            "potential_savings": self._calculate_potential_savings(),
        }

    def _calculate_potential_savings(self) -> Dict[str, float]:
        """Calculate potential savings from latency optimization."""
        if not self.venue_performance:
            return {}

        best_venue_latency = min(
            stats["total_latency"] / stats["fills"]
            for stats in self.venue_performance.values()
            if stats["fills"] > 0
        )

        current_avg_latency = self.total_latency_us / max(self.fill_count, 1)
        latency_improvement_us = current_avg_latency - best_venue_latency
        improvement_pct = latency_improvement_us / current_avg_latency * 100
        cost_savings_bps = latency_improvement_us / 100 * 0.1

        return {
            "current_avg_latency_us": current_avg_latency,
            "best_venue_latency_us": best_venue_latency,
            "potential_improvement_us": latency_improvement_us,
            "potential_improvement_pct": improvement_pct,
            "estimated_savings_bps": cost_savings_bps,
        }
