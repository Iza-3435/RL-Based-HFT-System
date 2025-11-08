"""Cost modeling for trading."""

from typing import Dict, Any, List
from collections import defaultdict
import numpy as np

from src.risk.fee_tracker import FeeTracker
from src.trading.types import Fill, Order, TradingStrategyType


class LatencyCostModel:
    """Model opportunity costs due to latency."""

    def __init__(self):
        self.base_decay_rate = 0.0001
        self.volatility_multiplier = 2.0
        self.competition_factor = 1.5

    def calculate_cost(self, fill: Fill, order: Order) -> float:
        """Calculate opportunity cost from latency."""
        latency_ms = fill.latency_us / 1000

        base_cost = (
            self.base_decay_rate * latency_ms *
            fill.price * fill.quantity
        )

        if hasattr(order, 'market_regime') and order.market_regime == 'volatile':
            base_cost *= self.volatility_multiplier

        if order.strategy == TradingStrategyType.ARBITRAGE:
            base_cost *= self.competition_factor

        actual_cost = min(
            base_cost,
            fill.slippage_bps * fill.price * fill.quantity / 10000 * 0.5
        )

        return actual_cost

    def estimate_latency_alpha(
        self,
        avg_latency_us: float,
        baseline_latency_us: float,
        daily_volume: int,
        avg_price: float
    ) -> float:
        """Estimate alpha from latency improvement."""
        latency_improvement = baseline_latency_us - avg_latency_us

        if latency_improvement <= 0:
            return 0

        sensitive_volume = daily_volume * 0.20
        bp_improvement = (latency_improvement / 100) * 1.0

        daily_alpha = sensitive_volume * avg_price * bp_improvement / 10000

        return daily_alpha


class CostAnalysis:
    """Comprehensive cost analysis across all trading."""

    def __init__(self):
        self.fee_tracker = FeeTracker()
        self.latency_cost_model = LatencyCostModel()
        self.cost_history = []

    def analyze_costs(
        self,
        fills: List[Fill],
        orders: Dict[str, Order]
    ) -> Dict[str, Any]:
        """Analyze all trading costs."""
        analysis = {
            'total_costs': 0,
            'by_type': {
                'fees': 0,
                'market_impact': 0,
                'latency_cost': 0,
                'opportunity_cost': 0
            },
            'by_venue': defaultdict(float),
            'by_strategy': defaultdict(float),
            'cost_per_share': 0,
            'cost_as_pct_of_volume': 0,
            'potential_savings': {}
        }

        total_volume = 0
        total_notional = 0

        for fill in fills:
            order = orders.get(fill.order_id)
            if not order:
                continue

            net_fee = fill.fees - fill.rebate
            analysis['by_type']['fees'] += net_fee

            impact_cost = (
                fill.market_impact_bps * fill.price * fill.quantity / 10000
            )
            analysis['by_type']['market_impact'] += impact_cost

            latency_cost = self.latency_cost_model.calculate_cost(fill, order)
            analysis['by_type']['latency_cost'] += latency_cost

            total_cost = net_fee + impact_cost + latency_cost
            analysis['total_costs'] += total_cost

            analysis['by_venue'][fill.venue] += total_cost
            analysis['by_strategy'][order.strategy.value] += total_cost

            total_volume += fill.quantity
            total_notional += fill.quantity * fill.price

        if total_volume > 0:
            analysis['cost_per_share'] = analysis['total_costs'] / total_volume

        if total_notional > 0:
            analysis['cost_as_pct_of_volume'] = (
                analysis['total_costs'] / total_notional * 100
            )

        analysis['potential_savings'] = self._identify_savings(analysis, fills)

        return analysis

    def _identify_savings(
        self,
        analysis: Dict,
        fills: List[Fill]
    ) -> Dict[str, float]:
        """Identify potential cost savings."""
        savings = {}

        current_taker_pct = (
            sum(1 for f in fills if f.fees > 0) / len(fills) if fills else 0
        )
        if current_taker_pct > 0.5:
            potential_maker_fees = analysis['by_type']['fees'] * -0.5
            savings['increase_maker_percentage'] = abs(
                potential_maker_fees - analysis['by_type']['fees']
            )

        avg_latency = np.mean([f.latency_us for f in fills]) if fills else 1000
        if avg_latency > 500:
            latency_improvement_pct = (avg_latency - 500) / avg_latency
            savings['reduce_latency_to_500us'] = (
                analysis['by_type']['latency_cost'] * latency_improvement_pct
            )

        venue_costs = analysis['by_venue']
        if venue_costs:
            cheapest_venue_cost = min(venue_costs.values())
            total_venue_cost = sum(venue_costs.values())
            if len(venue_costs) > 1:
                savings['optimize_venue_selection'] = (
                    total_venue_cost - (cheapest_venue_cost * len(venue_costs))
                )

        return savings
