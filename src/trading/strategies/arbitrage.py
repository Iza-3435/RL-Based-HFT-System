"""Cross-venue arbitrage strategy."""

import time
from typing import Dict, List
from collections import defaultdict
import numpy as np

from src.trading.strategies.base import TradingStrategy
from src.trading.types import (
    TradingStrategyType, Order, OrderSide, OrderType, Fill
)


class ArbitrageStrategy(TradingStrategy):
    """Cross-venue arbitrage strategy."""

    def __init__(self, params: Dict = None):
        default_params = {
            'min_arb_bps': 5.0,
            'max_position': 5000,
            'latency_threshold_us': 500,
            'confidence_threshold': 0.8,
            'competition_factor': 0.7
        }
        params = {**default_params, **(params or {})}
        super().__init__(TradingStrategyType.ARBITRAGE, params)

        self.opportunities_detected = 0
        self.opportunities_captured = 0
        self.arb_pnl = 0

    async def generate_signals(
        self,
        market_data: Dict,
        ml_predictions: Dict
    ) -> List[Order]:
        """Generate arbitrage orders."""
        orders = []

        venue_prices = defaultdict(dict)
        for symbol in market_data.get('symbols', []):
            for venue in market_data.get('venues', []):
                key = f"{symbol}_{venue}"
                if key in market_data:
                    venue_prices[symbol][venue] = market_data[key]

        for symbol, prices_by_venue in venue_prices.items():
            if len(prices_by_venue) < 2:
                continue

            best_bid = max(
                prices_by_venue.items(),
                key=lambda x: x[1].get('bid_price', 0)
            )
            best_ask = min(
                prices_by_venue.items(),
                key=lambda x: x[1].get('ask_price', float('inf'))
            )

            bid_venue, bid_data = best_bid
            ask_venue, ask_data = best_ask

            if bid_data['bid_price'] > ask_data['ask_price']:
                arb_bps = (
                    (bid_data['bid_price'] - ask_data['ask_price']) /
                    ask_data['ask_price'] * 10000
                )

                if arb_bps >= self.params['min_arb_bps']:
                    self.opportunities_detected += 1

                    buy_routing = ml_predictions.get(
                        f'routing_{symbol}_{ask_venue}', {}
                    )
                    sell_routing = ml_predictions.get(
                        f'routing_{symbol}_{bid_venue}', {}
                    )

                    buy_latency = buy_routing.get('predicted_latency_us', 1000)
                    sell_latency = sell_routing.get('predicted_latency_us', 1000)

                    total_latency = buy_latency + sell_latency
                    if total_latency > self.params['latency_threshold_us']:
                        continue

                    min_confidence = min(
                        buy_routing.get('confidence', 0),
                        sell_routing.get('confidence', 0)
                    )
                    if min_confidence < self.params['confidence_threshold']:
                        continue

                    if np.random.random() > self.params['competition_factor']:
                        continue

                    buy_size = min(
                        ask_data.get('ask_size', 100),
                        self.params['max_position']
                    )
                    sell_size = min(
                        bid_data.get('bid_size', 100),
                        self.params['max_position']
                    )
                    arb_size = min(buy_size, sell_size)

                    buy_order = Order(
                        order_id=f"ARB_B_{self.opportunities_captured:08d}",
                        symbol=symbol,
                        venue=ask_venue,
                        side=OrderSide.BUY,
                        order_type=OrderType.IOC,
                        quantity=arb_size,
                        price=ask_data['ask_price'],
                        timestamp=time.time(),
                        strategy=TradingStrategyType.ARBITRAGE,
                        predicted_latency_us=buy_latency,
                        routing_confidence=buy_routing.get('confidence', 0.5),
                        market_regime=ml_predictions.get('regime', 'normal')
                    )

                    sell_order = Order(
                        order_id=f"ARB_S_{self.opportunities_captured:08d}",
                        symbol=symbol,
                        venue=bid_venue,
                        side=OrderSide.SELL,
                        order_type=OrderType.IOC,
                        quantity=arb_size,
                        price=bid_data['bid_price'],
                        timestamp=time.time(),
                        strategy=TradingStrategyType.ARBITRAGE,
                        predicted_latency_us=sell_latency,
                        routing_confidence=sell_routing.get('confidence', 0.5),
                        market_regime=ml_predictions.get('regime', 'normal')
                    )

                    orders.extend([buy_order, sell_order])
                    self.opportunities_captured += 1

        return orders

    def calculate_arbitrage_pnl(self, buy_fill: Fill, sell_fill: Fill):
        """Calculate P&L from arbitrage trade."""
        if buy_fill.symbol == sell_fill.symbol:
            quantity = min(buy_fill.quantity, sell_fill.quantity)
            gross_pnl = (sell_fill.price - buy_fill.price) * quantity

            net_pnl = (
                gross_pnl - buy_fill.fees + buy_fill.rebate -
                sell_fill.fees + sell_fill.rebate
            )

            self.arb_pnl += net_pnl
            return net_pnl
        return 0
