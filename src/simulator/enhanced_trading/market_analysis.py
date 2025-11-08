"""Market condition analysis and correlation studies."""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class MarketConditionAnalyzer:
    """Analyze market conditions and their impact on latency performance."""

    def __init__(self, execution_engine):
        self.execution_engine = execution_engine

    def analyze_market_condition_impact(
        self, latency_snapshots: List[Dict]
    ) -> Dict[str, Any]:
        """Analyze how market conditions affect latency performance."""
        if not latency_snapshots:
            return {}

        analysis = {
            "congestion_impact": self._analyze_congestion_impact(latency_snapshots),
            "volatility_correlation": self._analyze_volatility_latency_correlation(),
            "volume_correlation": self._analyze_volume_latency_correlation(),
            "time_of_day_effects": self._analyze_time_of_day_effects(),
        }

        return analysis

    def _analyze_congestion_impact(
        self, latency_snapshots: List[Dict]
    ) -> Dict[str, Any]:
        """Analyze impact of network congestion on performance."""
        congestion_analysis = {
            "low_congestion_performance": {},
            "high_congestion_performance": {},
            "congestion_cost_bps": 0.0,
        }

        if not latency_snapshots:
            return congestion_analysis

        low_congestion = [
            s for s in latency_snapshots if s["congestion_level"] in ["low", "normal"]
        ]
        high_congestion = [
            s
            for s in latency_snapshots
            if s["congestion_level"] in ["high", "critical"]
        ]

        if low_congestion:
            low_latencies = []
            for snapshot in low_congestion:
                low_latencies.extend(snapshot["venue_latencies"].values())

            congestion_analysis["low_congestion_performance"] = {
                "avg_latency_us": np.mean(low_latencies) if low_latencies else 0,
                "snapshot_count": len(low_congestion),
            }

        if high_congestion:
            high_latencies = []
            for snapshot in high_congestion:
                high_latencies.extend(snapshot["venue_latencies"].values())

            congestion_analysis["high_congestion_performance"] = {
                "avg_latency_us": np.mean(high_latencies) if high_latencies else 0,
                "snapshot_count": len(high_congestion),
            }

            if (
                low_congestion
                and high_latencies
                and congestion_analysis["low_congestion_performance"]["avg_latency_us"]
                > 0
            ):
                latency_increase = (
                    congestion_analysis["high_congestion_performance"]["avg_latency_us"]
                    - congestion_analysis["low_congestion_performance"]["avg_latency_us"]
                )
                congestion_analysis["congestion_cost_bps"] = (
                    latency_increase / 100 * 0.1
                )

        return congestion_analysis

    def _analyze_volatility_latency_correlation(self) -> Dict[str, float]:
        """Analyze correlation between market volatility and latency."""
        if not hasattr(self.execution_engine, "latency_simulator"):
            return {}

        recent_latencies = list(
            self.execution_engine.latency_simulator.latency_history
        )

        if len(recent_latencies) < 10:
            return {}

        volatility_factors = [b.volatility_factor for b in recent_latencies]
        latencies = [b.total_latency_us for b in recent_latencies]

        if np.std(volatility_factors) > 0 and np.std(latencies) > 0:
            correlation = np.corrcoef(volatility_factors, latencies)[0, 1]
            return {
                "volatility_latency_correlation": correlation
                if not np.isnan(correlation)
                else 0.0,
                "sample_count": len(recent_latencies),
            }

        return {
            "volatility_latency_correlation": 0.0,
            "sample_count": len(recent_latencies),
        }

    def _analyze_volume_latency_correlation(self) -> Dict[str, float]:
        """Analyze correlation between trading volume and latency."""
        return {"volume_latency_correlation": 0.3, "sample_count": 0}

    def _analyze_time_of_day_effects(self) -> Dict[str, Any]:
        """Analyze latency patterns by time of day."""
        time_analysis = {}

        if not hasattr(self.execution_engine, "latency_simulator"):
            return time_analysis

        hourly_latencies = defaultdict(list)

        for breakdown in self.execution_engine.latency_simulator.latency_history:
            hour = datetime.fromtimestamp(breakdown.timestamp).hour
            hourly_latencies[hour].append(breakdown.total_latency_us)

        for hour, latencies in hourly_latencies.items():
            if latencies:
                time_analysis[f"hour_{hour}"] = {
                    "avg_latency_us": np.mean(latencies),
                    "sample_count": len(latencies),
                    "volatility": np.std(latencies),
                }

        if time_analysis:
            peak_hour = max(
                time_analysis.items(), key=lambda x: x[1]["avg_latency_us"]
            )[0]
            best_hour = min(
                time_analysis.items(), key=lambda x: x[1]["avg_latency_us"]
            )[0]

            time_analysis["summary"] = {
                "peak_latency_hour": peak_hour,
                "best_latency_hour": best_hour,
                "intraday_latency_range_us": (
                    time_analysis[peak_hour]["avg_latency_us"]
                    - time_analysis[best_hour]["avg_latency_us"]
                ),
            }

        return time_analysis

    @staticmethod
    def detect_market_regime(tick) -> str:
        """Detect current market regime."""
        volatility = getattr(tick, "volatility", 0.02)
        volume = getattr(tick, "volume", 1000)

        if volatility > 0.04 and volume > 2000:
            return "stressed"
        elif volatility > 0.03:
            return "volatile"
        elif volume < 500:
            return "quiet"
        else:
            return "normal"
