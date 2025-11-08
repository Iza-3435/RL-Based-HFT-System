"""Comprehensive latency analytics and reporting."""

import time
import logging
import json
import numpy as np
from typing import Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class LatencyAnalytics:
    """Comprehensive latency analytics and reporting."""

    def __init__(self, execution_engine: Any) -> None:
        self.execution_engine = execution_engine
        self.latency_simulator = execution_engine.latency_simulator

    def generate_latency_report(self, timeframe_minutes: int = 60) -> Dict[str, Any]:
        """Generate comprehensive latency performance report."""
        return {
            "report_timestamp": time.time(),
            "timeframe_minutes": timeframe_minutes,
            "summary": self._generate_summary_stats(timeframe_minutes),
            "venue_analysis": self._generate_venue_analysis(timeframe_minutes),
            "component_breakdown": self._generate_component_breakdown(timeframe_minutes),
            "prediction_performance": self._generate_prediction_analysis(),
            "cost_impact": self._generate_cost_impact_analysis(),
            "recommendations": self._generate_recommendations(),
        }

    def _generate_summary_stats(self, timeframe_minutes: int) -> Dict[str, Any]:
        """Generate summary statistics."""
        execution_stats = self.execution_engine.get_enhanced_execution_stats()
        pred_accuracy = execution_stats["latency_analysis"]["prediction_accuracy"]
        congestion = execution_stats["latency_analysis"]["congestion_analysis"]

        return {
            "total_orders_executed": execution_stats["execution_stats"]["total_fills"],
            "avg_total_latency_us": execution_stats["execution_stats"]["avg_latency_us"],
            "avg_slippage_bps": execution_stats["execution_stats"]["avg_slippage_bps"],
            "avg_latency_cost_bps": execution_stats["execution_stats"]["avg_latency_cost_bps"],
            "prediction_accuracy_pct": pred_accuracy.get("prediction_within_10pct", 0),
            "current_congestion_level": congestion["current_congestion_level"],
        }

    def _generate_venue_analysis(self, timeframe_minutes: int) -> Dict[str, Any]:
        """Generate per-venue analysis."""
        venue_analysis = {}

        for venue in self.latency_simulator.venues:
            stats = self.latency_simulator.get_venue_latency_stats(venue, timeframe_minutes)

            if stats:
                venue_analysis[venue] = {
                    "performance_metrics": {
                        "mean_latency_us": stats["mean_us"],
                        "p95_latency_us": stats["p95_us"],
                        "p99_latency_us": stats["p99_us"],
                        "latency_std_us": stats["std_us"],
                    },
                    "component_breakdown": {
                        "network_contribution_pct": stats.get(
                            "network_latency_us_contribution_pct", 0
                        ),
                        "queue_contribution_pct": stats.get("queue_delay_us_contribution_pct", 0),
                        "exchange_contribution_pct": stats.get(
                            "exchange_delay_us_contribution_pct", 0
                        ),
                    },
                    "queue_health": {
                        "utilization_pct": stats.get("queue_capacity_utilization", 0) * 100,
                        "current_delay_ms": stats.get("queue_current_delay_ms", 0),
                        "queue_full_events": stats.get("queue_queue_full_events", 0),
                    },
                }

        return venue_analysis

    def _generate_component_breakdown(self, timeframe_minutes: int) -> Dict[str, Any]:
        """Analyze latency by component."""
        recent_latencies = [
            b
            for b in self.latency_simulator.latency_history
            if time.time() - b.timestamp < timeframe_minutes * 60
        ]

        if not recent_latencies:
            return {}

        components = [
            "network_latency_us",
            "queue_delay_us",
            "exchange_delay_us",
            "processing_delay_us",
        ]
        breakdown = {}

        total_latencies = [b.total_latency_us for b in recent_latencies]
        mean_total = np.mean(total_latencies)

        for component in components:
            values = [getattr(b, component) for b in recent_latencies]
            breakdown[component] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "p95": np.percentile(values, 95),
                "contribution_pct": np.mean(values) / mean_total * 100,
            }

        return breakdown

    def _generate_prediction_analysis(self) -> Dict[str, Any]:
        """Analyze ML prediction performance."""
        return self.latency_simulator.get_prediction_accuracy_stats()

    def _generate_cost_impact_analysis(self) -> Dict[str, Any]:
        """Analyze cost impact of latency."""
        return self.execution_engine.get_latency_cost_analysis()

    def _generate_recommendations(self) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        pred_stats = self.latency_simulator.get_prediction_accuracy_stats()
        if pred_stats.get("prediction_within_10pct", 0) < 75:
            recommendations.append(
                "ML latency prediction accuracy is below 75%. Consider retraining models."
            )

        congestion = self.latency_simulator.get_congestion_analysis()
        if congestion["current_congestion_level"] in ["high", "critical"]:
            recommendations.append(
                "Current network congestion is high. Consider routing to less congested venues."
            )

        rankings = self.execution_engine.get_venue_latency_rankings()
        if len(rankings) > 1:
            best_venue, best_latency = rankings[0]
            worst_venue, worst_latency = rankings[-1]

            if worst_latency > best_latency * 1.5:
                recommendations.append(
                    f"Consider avoiding {worst_venue} (avg {worst_latency:.0f}μs) "
                    f"in favor of {best_venue} (avg {best_latency:.0f}μs)"
                )

        for venue in self.latency_simulator.venues:
            queue_stats = self.latency_simulator.message_queues[venue].get_queue_stats()
            if queue_stats["capacity_utilization"] > 0.8:
                recommendations.append(
                    f"Queue utilization at {venue} is {queue_stats['capacity_utilization']:.1%}. "
                    f"Consider reducing order flow."
                )

        return recommendations

    def export_detailed_data(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """Export detailed latency data for further analysis."""
        data = {
            "latency_breakdown_data": self.latency_simulator.export_latency_data(),
            "execution_statistics": self.execution_engine.get_enhanced_execution_stats(),
            "venue_rankings": self.execution_engine.get_venue_latency_rankings(),
            "cost_analysis": self.execution_engine.get_latency_cost_analysis(),
        }

        if filename:
            with open(filename, "w") as f:
                json.dump(data, f, indent=2, default=str)

        return data
