"""Utility functions for enhanced trading simulator."""

import logging
from typing import Any, Dict, List

import numpy as np

from simulator.enhanced_latency_simulation import (
    EnhancedOrderExecutionEngine,
    LatencyAnalytics,
    LatencySimulator,
    create_latency_configuration,
    integrate_latency_simulation,
)
from simulator.trading_simulator import TradingSimulator

from .enhanced_simulator import EnhancedTradingSimulator

logger = logging.getLogger(__name__)


def create_enhanced_trading_simulator(
    symbols: List[str], venues: List[str], config: Dict[str, Any] = None
) -> EnhancedTradingSimulator:
    """Factory function to create enhanced trading simulator with optimal configuration."""
    default_config = {
        "enable_latency_simulation": True,
        "latency_prediction_enabled": True,
        "congestion_modeling": True,
        "venue_optimization": True,
        "strategy_latency_optimization": True,
    }

    final_config = {**default_config, **(config or {})}

    simulator = EnhancedTradingSimulator(
        venues=venues,
        symbols=symbols,
        enable_latency_simulation=final_config["enable_latency_simulation"],
    )

    if final_config["venue_optimization"]:
        _configure_venue_optimization(simulator)

    if final_config["strategy_latency_optimization"]:
        _configure_strategy_latency_parameters(simulator)

    logger.info(
        f"Enhanced trading simulator created with {len(symbols)} symbols, "
        f"{len(venues)} venues, config: {final_config}"
    )

    return simulator


def _configure_venue_optimization(simulator: EnhancedTradingSimulator):
    """Configure venue-specific optimization parameters."""
    if hasattr(simulator.execution_engine, "latency_simulator"):
        latency_sim = simulator.execution_engine.latency_simulator

        for venue, profile in latency_sim.venue_profiles.items():
            if venue == "IEX":
                profile.network_latency_std_us *= 0.5
                profile.spike_probability *= 0.3
            elif venue == "NASDAQ":
                profile.processing_rate_msg_per_sec *= 1.2
            elif venue == "CBOE":
                profile.volatility_sensitivity *= 1.1


def _configure_strategy_latency_parameters(simulator: EnhancedTradingSimulator):
    """Configure strategy-specific latency optimization parameters."""
    if "market_making" in simulator.strategies:
        mm_strategy = simulator.strategies["market_making"]
        mm_strategy.params["latency_weight"] = 0.3
        mm_strategy.params["consistency_weight"] = 0.7

    if "arbitrage" in simulator.strategies:
        arb_strategy = simulator.strategies["arbitrage"]
        arb_strategy.params["latency_weight"] = 0.8
        arb_strategy.params["max_acceptable_latency_us"] = 300

    if "momentum" in simulator.strategies:
        mom_strategy = simulator.strategies["momentum"]
        mom_strategy.params["latency_weight"] = 0.5
        mom_strategy.params["signal_decay_factor"] = 0.95


def patch_existing_simulator(
    existing_simulator: TradingSimulator, enable_enhanced_latency: bool = True
) -> TradingSimulator:
    """Patch an existing TradingSimulator instance with enhanced latency capabilities."""
    logger.info("Patching existing TradingSimulator with enhanced latency capabilities")

    original_execution_engine = existing_simulator.execution_engine

    if enable_enhanced_latency:
        existing_simulator.execution_engine = EnhancedOrderExecutionEngine(
            existing_simulator.venues
        )
    else:
        latency_simulator = LatencySimulator(existing_simulator.venues)
        integrate_latency_simulation(original_execution_engine, latency_simulator)

    existing_simulator.latency_analytics = LatencyAnalytics(
        existing_simulator.execution_engine
    )
    existing_simulator.latency_config = create_latency_configuration()

    logger.info("Existing simulator successfully patched with latency capabilities")
    return existing_simulator


def quick_latency_test(
    simulator: EnhancedTradingSimulator, venue: str, num_samples: int = 100
) -> Dict[str, Any]:
    """Quick latency test for a specific venue."""
    if not hasattr(simulator.execution_engine, "latency_simulator"):
        return {"error": "Latency simulator not available"}

    latency_samples = []

    for _ in range(num_samples):
        breakdown = simulator.execution_engine.latency_simulator.simulate_latency(
            venue=venue, symbol="TEST", order_type="limit"
        )
        latency_samples.append(breakdown.total_latency_us)

    return {
        "venue": venue,
        "samples": num_samples,
        "mean_latency_us": np.mean(latency_samples),
        "median_latency_us": np.median(latency_samples),
        "p95_latency_us": np.percentile(latency_samples, 95),
        "p99_latency_us": np.percentile(latency_samples, 99),
        "std_latency_us": np.std(latency_samples),
        "min_latency_us": np.min(latency_samples),
        "max_latency_us": np.max(latency_samples),
    }
