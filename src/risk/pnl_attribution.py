"""P&L attribution system."""

from typing import Dict, Any
from collections import defaultdict
from datetime import datetime

from src.risk.types import PnLComponent
from src.risk.fee_tracker import FeeTracker
from src.risk.cost_models import LatencyCostModel
from src.trading.types import Fill, Order, OrderSide, TradingStrategyType


class PnLAttribution:
    """Detailed P&L attribution system."""

    def __init__(self):
        self.pnl_components = defaultdict(
            lambda: defaultdict(lambda: PnLComponent())
        )
        self.attribution_history = []
        self.fee_tracker = FeeTracker()
        self.latency_cost_model = LatencyCostModel()

    def attribute_fill(self, fill: Fill, order: Order, market_state: Dict):
        """Attribute P&L from a fill."""
        strategy = order.strategy.value
        venue = fill.venue
        hour = datetime.fromtimestamp(fill.timestamp).hour
        regime = market_state.get('regime', 'normal')

        if order.strategy == TradingStrategyType.MARKET_MAKING:
            revenue_source = 'spread_capture'
            gross_pnl = self._calculate_spread_capture(fill, market_state)
        elif order.strategy == TradingStrategyType.ARBITRAGE:
            revenue_source = 'arbitrage'
            gross_pnl = self._calculate_arbitrage_pnl(fill, market_state)
        elif order.strategy == TradingStrategyType.MOMENTUM:
            revenue_source = 'momentum'
            gross_pnl = 0
        else:
            revenue_source = 'other'
            gross_pnl = 0

        fees = fill.fees
        rebates = fill.rebate
        market_impact = fill.market_impact_bps * fill.price * fill.quantity / 10000
        latency_cost = self.latency_cost_model.calculate_cost(fill, order)

        for key in [
            (strategy, revenue_source),
            (venue, 'venue'),
            (hour, 'hour'),
            (regime, 'regime')
        ]:
            component = self.pnl_components[key[0]][key[1]]
            component.source = key[1]
            component.gross_pnl += gross_pnl
            component.fees += fees
            component.rebates += rebates
            component.market_impact += market_impact
            component.latency_cost += latency_cost
            component.net_pnl += (
                gross_pnl - fees + rebates - market_impact - latency_cost
            )
            component.trade_count += 1
            component.volume += fill.quantity

    def _calculate_spread_capture(self, fill: Fill, market_state: Dict) -> float:
        """Calculate spread capture for market making."""
        mid_price = market_state.get('mid_price', fill.price)

        if fill.side == OrderSide.BUY:
            spread_capture = (mid_price - fill.price) * fill.quantity
        else:
            spread_capture = (fill.price - mid_price) * fill.quantity

        return max(0, spread_capture)

    def _calculate_arbitrage_pnl(self, fill: Fill, market_state: Dict) -> float:
        """Calculate arbitrage P&L."""
        return fill.quantity * 0.05

    def close_position(
        self,
        symbol: str,
        strategy: str,
        avg_entry_price: float,
        exit_price: float,
        quantity: int
    ):
        """Calculate P&L on position close."""
        if strategy == TradingStrategyType.MOMENTUM.value:
            pnl = (exit_price - avg_entry_price) * quantity
            component = self.pnl_components[strategy]['momentum']
            component.gross_pnl += pnl
            component.net_pnl += pnl

    def get_attribution_report(self) -> Dict[str, Any]:
        """Generate P&L attribution report."""
        report = {
            'total_pnl': 0,
            'by_strategy': {},
            'by_venue': {},
            'by_hour': {},
            'by_regime': {},
            'by_source': {},
            'cost_breakdown': {
                'total_fees': 0,
                'total_rebates': 0,
                'total_market_impact': 0,
                'total_latency_cost': 0
            }
        }

        for primary_key, components in self.pnl_components.items():
            for secondary_key, component in components.items():
                if secondary_key == 'venue':
                    report['by_venue'][primary_key] = self._component_to_dict(component)
                elif secondary_key == 'hour':
                    report['by_hour'][primary_key] = self._component_to_dict(component)
                elif secondary_key == 'regime':
                    report['by_regime'][primary_key] = self._component_to_dict(component)
                elif secondary_key in ['spread_capture', 'arbitrage', 'momentum']:
                    report['by_source'][secondary_key] = self._component_to_dict(component)
                    if primary_key not in report['by_strategy']:
                        report['by_strategy'][primary_key] = self._component_to_dict(component)

                report['total_pnl'] += component.net_pnl
                report['cost_breakdown']['total_fees'] += component.fees
                report['cost_breakdown']['total_rebates'] += component.rebates
                report['cost_breakdown']['total_market_impact'] += component.market_impact
                report['cost_breakdown']['total_latency_cost'] += component.latency_cost

        return report

    def _component_to_dict(self, component: PnLComponent) -> Dict:
        """Convert P&L component to dictionary."""
        return {
            'gross_pnl': component.gross_pnl,
            'fees': component.fees,
            'rebates': component.rebates,
            'market_impact': component.market_impact,
            'latency_cost': component.latency_cost,
            'net_pnl': component.net_pnl,
            'trade_count': component.trade_count,
            'volume': component.volume,
            'pnl_per_trade': component.pnl_per_trade,
            'pnl_per_share': component.pnl_per_share
        }
