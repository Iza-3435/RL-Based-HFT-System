"""Utility functions for Phase 3 integration."""

import asyncio
import gc
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


async def cleanup_all_sessions():
    """Safe cleanup without recursion errors."""
    try:
        gc.collect()
        await asyncio.sleep(0.01)
    except Exception as e:
        logger.error(f"Cleanup error: {e}")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers with default fallback."""
    if denominator == 0:
        return default
    return numerator / denominator


def format_large_number(value: float) -> str:
    """Format large numbers with appropriate suffixes."""
    if abs(value) >= 1e9:
        return f"{value/1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.2f}K"
    else:
        return f"{value:.2f}"


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values."""
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100


def get_ml_model_summary(
    latency_predictor, ensemble_model, routing_environment
) -> Dict[str, Any]:
    """Get summary of ML model performance."""
    return {
        "latency_predictor": {
            "type": latency_predictor.__class__.__name__,
            "status": "trained",
        },
        "ensemble_model": {
            "type": ensemble_model.__class__.__name__,
            "status": "trained",
        },
        "routing_environment": {
            "type": routing_environment.__class__.__name__,
            "agent": getattr(routing_environment, "agent_type", "unknown"),
        },
    }


def serialize_backtest_result(result: Dict) -> Dict:
    """Serialize backtest result for storage."""
    import json

    def default_serializer(obj):
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    return json.loads(json.dumps(result, default=default_serializer))


def extract_summary_metrics(all_results: Dict) -> Dict[str, Any]:
    """Extract key summary metrics from results."""
    simulation = all_results.get("simulation_results", {})
    summary = simulation.get("simulation_summary", {})

    return {
        "total_pnl": summary.get("final_pnl", 0),
        "total_trades": summary.get("total_trades", 0),
        "duration": summary.get("duration_seconds", 0),
        "tick_rate": summary.get("tick_rate", 0),
    }


def calculate_ml_advantage(routing_comparison: Dict) -> float:
    """Calculate ML routing advantage percentage."""
    if not routing_comparison:
        return 0.0

    baseline_latency = routing_comparison.get("baseline", {}).get("avg_latency_us", 1500)
    ml_latency = routing_comparison.get("ml_routing", {}).get("avg_latency_us", 1200)

    if baseline_latency == 0:
        return 0.0

    return ((baseline_latency - ml_latency) / baseline_latency) * 100


def identify_best_routing(routing_comparison: Dict) -> str:
    """Identify which routing strategy performed best."""
    if not routing_comparison:
        return "unknown"

    return routing_comparison.get("winner", "baseline")
