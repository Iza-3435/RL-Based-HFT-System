"""Latency statistics and analytics."""

import time
import numpy as np
from typing import Dict, List, Any, Deque
from collections import defaultdict
from .types import LatencyBreakdown, CongestionLevel
from .message_queue import MessageQueue


class LatencyStatistics:
    """Calculate latency statistics and analytics."""

    def __init__(
        self,
        latency_history: Deque[LatencyBreakdown],
        prediction_errors: List[float],
        message_queues: Dict[str, MessageQueue],
        congestion_level: CongestionLevel,
        base_congestion: float,
    ) -> None:
        self.latency_history = latency_history
        self.prediction_errors = prediction_errors
        self.message_queues = message_queues
        self.congestion_level = congestion_level
        self.base_congestion = base_congestion

    def get_venue_latency_stats(self, venue: str, window_minutes: int = 5) -> Dict[str, float]:
        """Get latency statistics for a venue over recent window."""
        current_time = time.time()
        cutoff_time = current_time - (window_minutes * 60)

        recent_latencies = [
            breakdown
            for breakdown in self.latency_history
            if breakdown.venue == venue and breakdown.timestamp >= cutoff_time
        ]

        if not recent_latencies:
            return {}

        total_latencies = [b.total_latency_us for b in recent_latencies]

        stats = {
            "count": len(total_latencies),
            "mean_us": np.mean(total_latencies),
            "median_us": np.median(total_latencies),
            "std_us": np.std(total_latencies),
            "min_us": np.min(total_latencies),
            "max_us": np.max(total_latencies),
            "p95_us": np.percentile(total_latencies, 95),
            "p99_us": np.percentile(total_latencies, 99),
        }

        component_stats = {}
        for component in [
            "network_latency_us",
            "queue_delay_us",
            "exchange_delay_us",
            "processing_delay_us",
        ]:
            values = [getattr(b, component) for b in recent_latencies]
            component_stats[f"{component}_mean"] = np.mean(values)
            component_stats[f"{component}_contribution_pct"] = (
                np.mean(values) / stats["mean_us"] * 100
            )

        stats.update(component_stats)

        if venue in self.message_queues:
            queue_stats = self.message_queues[venue].get_queue_stats()
            stats.update({f"queue_{k}": v for k, v in queue_stats.items()})

        return stats

    def get_prediction_accuracy_stats(self) -> Dict[str, float]:
        """Get ML prediction accuracy statistics."""
        if not self.prediction_errors:
            return {}

        errors = np.array(self.prediction_errors)
        recent_errors = errors[-100:] if len(errors) > 100 else errors

        recent_latencies = [
            b.total_latency_us for b in list(self.latency_history)[-len(recent_errors) :]
        ]

        return {
            "total_predictions": len(errors),
            "mean_error_us": np.mean(errors),
            "median_error_us": np.median(errors),
            "rmse_us": np.sqrt(np.mean(errors**2)),
            "mean_absolute_percentage_error": (
                np.mean(recent_errors) / np.mean(recent_latencies) * 100 if recent_latencies else 0
            ),
            "prediction_within_10pct": (
                np.sum(recent_errors < np.mean(recent_errors) * 0.1) / len(recent_errors) * 100
            ),
        }

    def get_congestion_analysis(self) -> Dict[str, Any]:
        """Analyze current congestion and impact."""
        current_time = time.time()

        recent_latencies = [b for b in self.latency_history if current_time - b.timestamp < 300]

        if not recent_latencies:
            return {}

        latency_by_congestion = defaultdict(list)
        for breakdown in recent_latencies:
            latency_by_congestion[breakdown.congestion_level].append(breakdown.total_latency_us)

        congestion_stats = {}
        all_latencies_mean = np.mean([b.total_latency_us for b in recent_latencies])

        for level, latencies in latency_by_congestion.items():
            if latencies:
                congestion_stats[level.value] = {
                    "count": len(latencies),
                    "mean_latency_us": np.mean(latencies),
                    "latency_increase_pct": (np.mean(latencies) / all_latencies_mean - 1) * 100,
                }

        return {
            "current_congestion_level": self.congestion_level.value,
            "base_congestion_factor": self.base_congestion,
            "latency_by_congestion": congestion_stats,
            "queue_states": {
                venue: queue.get_queue_stats() for venue, queue in self.message_queues.items()
            },
        }
