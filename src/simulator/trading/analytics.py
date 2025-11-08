"""Performance measurement and P&L attribution analysis."""

from collections import defaultdict
from typing import Any, Dict, List

from .dataclasses import Fill
from .enums import OrderSide


def calculate_pnl_attribution(simulator) -> Dict[str, Any]:
    """
    Detailed P&L attribution analysis.

    Breaks down performance by:
    - Strategy contribution
    - Venue efficiency
    - Timing/latency impact
    - ML routing benefit
    """
    attribution = {
        "total_pnl": simulator.total_pnl,
        "strategy_attribution": {},
        "venue_attribution": {},
        "latency_cost_analysis": {},
        "ml_routing_benefit": {},
    }

    for strategy_name, strategy in simulator.strategies.items():
        pnl_data = strategy.get_total_pnl()
        attribution["strategy_attribution"][strategy_name] = {
            "gross_pnl": pnl_data["realized_pnl"] + pnl_data["unrealized_pnl"],
            "fees_paid": pnl_data["fees_paid"],
            "net_pnl": pnl_data["total_pnl"],
            "pnl_contribution_pct": pnl_data["total_pnl"] / simulator.total_pnl * 100
            if simulator.total_pnl != 0
            else 0,
        }

    venue_pnl = defaultdict(float)
    venue_volume = defaultdict(int)

    for strategy in simulator.strategies.values():
        for fill in strategy.fills:
            if fill.side == OrderSide.SELL:
                position = strategy.positions[fill.symbol]
                pnl = fill.quantity * (fill.price - position.average_cost)
                venue_pnl[fill.venue] += pnl
            venue_volume[fill.venue] += fill.quantity

    for venue in venue_pnl:
        attribution["venue_attribution"][venue] = {
            "pnl": venue_pnl[venue],
            "volume": venue_volume[venue],
            "pnl_per_share": venue_pnl[venue] / venue_volume[venue]
            if venue_volume[venue] > 0
            else 0,
        }

    latency_costs = calculate_latency_costs(simulator.fill_history)
    attribution["latency_cost_analysis"] = latency_costs

    ml_benefit = simulator._analyze_ml_impact()
    attribution["ml_routing_benefit"] = ml_benefit

    return attribution


def calculate_latency_costs(fills: List[Fill]) -> Dict[str, float]:
    """Calculate opportunity costs due to latency."""
    total_latency_cost = 0
    latency_by_strategy = defaultdict(float)

    for fill in fills:
        latency_cost = fill.slippage_bps * 0.5 * fill.price * fill.quantity / 10000
        total_latency_cost += latency_cost

        if "MM" in fill.order_id:
            latency_by_strategy["market_making"] += latency_cost
        elif "ARB" in fill.order_id:
            latency_by_strategy["arbitrage"] += latency_cost
        elif "MOM" in fill.order_id:
            latency_by_strategy["momentum"] += latency_cost

    return {
        "total_latency_cost": total_latency_cost,
        "cost_by_strategy": dict(latency_by_strategy),
        "avg_cost_per_trade": total_latency_cost / len(fills) if fills else 0,
    }
