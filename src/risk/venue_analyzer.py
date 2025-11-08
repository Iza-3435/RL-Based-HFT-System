"""Venue performance analysis."""

from typing import Dict, Optional
from collections import defaultdict
import numpy as np

from src.trading.types import Order, Fill


class VenueAnalyzer:
    """Analyze trading performance by venue."""

    def __init__(self):
        self.venue_metrics = defaultdict(lambda: {
            'orders_sent': 0,
            'orders_filled': 0,
            'total_volume': 0,
            'total_fees': 0,
            'total_rebates': 0,
            'latencies': [],
            'slippages': [],
            'ml_routing_score': 0
        })

    def update_metrics(self, order: Order, fill: Optional[Fill] = None):
        """Update venue metrics with order/fill data."""
        metrics = self.venue_metrics[order.venue]
        metrics['orders_sent'] += 1

        if fill:
            metrics['orders_filled'] += 1
            metrics['total_volume'] += fill.quantity
            metrics['total_fees'] += fill.fees
            metrics['total_rebates'] += fill.rebate
            metrics['latencies'].append(fill.latency_us)
            metrics['slippages'].append(fill.slippage_bps)

            if hasattr(order, 'predicted_latency_us') and order.predicted_latency_us:
                prediction_error = abs(
                    fill.latency_us - order.predicted_latency_us
                ) / order.predicted_latency_us
                routing_score = max(0, 1 - prediction_error)
                n = metrics['orders_filled']
                metrics['ml_routing_score'] = (
                    (metrics['ml_routing_score'] * (n - 1) + routing_score) / n
                )

    def analyze_venue_performance(self) -> Dict[str, Dict]:
        """Generate venue performance analysis."""
        analysis = {}

        for venue, metrics in self.venue_metrics.items():
            fill_rate = metrics['orders_filled'] / max(metrics['orders_sent'], 1)

            if metrics['latencies']:
                avg_latency = np.mean(metrics['latencies'])
                p50_latency = np.percentile(metrics['latencies'], 50)
                p99_latency = np.percentile(metrics['latencies'], 99)
            else:
                avg_latency = p50_latency = p99_latency = 0

            if metrics['slippages']:
                avg_slippage = np.mean(metrics['slippages'])
            else:
                avg_slippage = 0

            net_fees = metrics['total_fees'] - metrics['total_rebates']

            analysis[venue] = {
                'fill_rate': fill_rate,
                'total_volume': metrics['total_volume'],
                'net_fees': net_fees,
                'fee_per_share': net_fees / max(metrics['total_volume'], 1),
                'latency_stats': {
                    'mean': avg_latency,
                    'p50': p50_latency,
                    'p99': p99_latency
                },
                'avg_slippage_bps': avg_slippage,
                'ml_routing_score': metrics['ml_routing_score'],
                'efficiency_score': self._calculate_efficiency_score(
                    metrics, fill_rate, avg_latency, avg_slippage
                )
            }

        return analysis

    def _calculate_efficiency_score(
        self,
        metrics: Dict,
        fill_rate: float,
        avg_latency: float,
        avg_slippage: float
    ) -> float:
        """Calculate overall venue efficiency score (0-100)."""
        weights = {
            'fill_rate': 0.3,
            'latency': 0.3,
            'cost': 0.2,
            'slippage': 0.2
        }

        fill_score = fill_rate * 100

        latency_score = max(0, min(100, (2000 - avg_latency) / 18))

        net_fee_per_share = (
            (metrics['total_fees'] - metrics['total_rebates']) /
            max(metrics['total_volume'], 1)
        )
        cost_score = max(0, min(100, (0.005 + net_fee_per_share) / 0.005 * 100))

        slippage_score = max(0, min(100, (10 - avg_slippage) * 10))

        efficiency_score = (
            weights['fill_rate'] * fill_score +
            weights['latency'] * latency_score +
            weights['cost'] * cost_score +
            weights['slippage'] * slippage_score
        )

        return efficiency_score

    def recommend_venue_allocation(self) -> Dict[str, float]:
        """Recommend optimal venue allocation based on performance."""
        venue_analysis = self.analyze_venue_performance()

        if not venue_analysis:
            return {}

        scores = {
            venue: data['efficiency_score']
            for venue, data in venue_analysis.items()
        }

        total_score = sum(scores.values())
        if total_score == 0:
            return {venue: 1.0 / len(scores) for venue in scores}

        allocations = {}
        for venue, score in scores.items():
            raw_allocation = score / total_score
            allocations[venue] = max(0.1, raw_allocation)

        total_allocation = sum(allocations.values())
        for venue in allocations:
            allocations[venue] /= total_allocation

        return allocations
