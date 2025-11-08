"""Market making strategy."""

import time
from typing import Dict, List

from src.trading.strategies.base import TradingStrategy
from src.trading.types import (
    TradingStrategyType, Order, OrderSide, OrderType, Fill
)


class MarketMakingStrategy(TradingStrategy):
    """Market making with inventory management."""

    def __init__(self, params: Dict = None):
        default_params = {
            'spread_multiplier': 1.0,
            'inventory_limit': 10000,
            'skew_factor': 0.1,
            'min_edge_bps': 2.0,
            'quote_size': 100,
            'urgency_threshold': 0.7
        }
        params = {**default_params, **(params or {})}
        super().__init__(TradingStrategyType.MARKET_MAKING, params)

        self.quote_count = 0
        self.spread_captured = 0

    async def generate_signals(
        self,
        market_data: Dict,
        ml_predictions: Dict
    ) -> List[Order]:
        """Generate market making orders."""
        orders = []

        for symbol in market_data.get('symbols', []):
            market_state = market_data[symbol]
            current_position = self.positions[symbol].quantity

            if abs(current_position) >= self.params['inventory_limit']:
                continue

            fair_value = market_state['mid_price']
            fair_spread_bps = market_state.get('spread_bps', 2.0)

            if ml_predictions.get('volatility_forecast', 0.01) > 0.02:
                spread_multiplier = self.params['spread_multiplier'] * 1.5
            else:
                spread_multiplier = self.params['spread_multiplier']

            inventory_skew_bps = (
                (current_position / 1000) * self.params['skew_factor']
            )

            half_spread_bps = fair_spread_bps * spread_multiplier / 2

            bid_price = fair_value * (
                1 - (half_spread_bps + inventory_skew_bps) / 10000
            )
            ask_price = fair_value * (
                1 + (half_spread_bps - inventory_skew_bps) / 10000
            )

            bid_edge_bps = (fair_value - bid_price) / fair_value * 10000
            ask_edge_bps = (ask_price - fair_value) / fair_value * 10000

            routing_decision = ml_predictions.get('routing', {})
            best_venue = routing_decision.get('venue', 'NYSE')
            predicted_latency = routing_decision.get('predicted_latency_us', 1000)

            if bid_edge_bps >= self.params['min_edge_bps']:
                bid_order = Order(
                    order_id=f"MM_B_{self.quote_count:08d}",
                    symbol=symbol,
                    venue=best_venue,
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    quantity=self.params['quote_size'],
                    price=bid_price,
                    timestamp=time.time(),
                    strategy=TradingStrategyType.MARKET_MAKING,
                    predicted_latency_us=predicted_latency,
                    routing_confidence=routing_decision.get('confidence', 0.5),
                    market_regime=ml_predictions.get('regime', 'normal')
                )
                orders.append(bid_order)

            if ask_edge_bps >= self.params['min_edge_bps']:
                ask_order = Order(
                    order_id=f"MM_S_{self.quote_count:08d}",
                    symbol=symbol,
                    venue=best_venue,
                    side=OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    quantity=self.params['quote_size'],
                    price=ask_price,
                    timestamp=time.time(),
                    strategy=TradingStrategyType.MARKET_MAKING,
                    predicted_latency_us=predicted_latency,
                    routing_confidence=routing_decision.get('confidence', 0.5),
                    market_regime=ml_predictions.get('regime', 'normal')
                )
                orders.append(ask_order)

            self.quote_count += 1

        return orders

    def update_spread_capture(self, buy_fill: Fill, sell_fill: Fill):
        """Track spread capture from round-trip trades."""
        if buy_fill.symbol == sell_fill.symbol:
            spread = sell_fill.price - buy_fill.price
            self.spread_captured += spread * min(
                buy_fill.quantity, sell_fill.quantity
            )
