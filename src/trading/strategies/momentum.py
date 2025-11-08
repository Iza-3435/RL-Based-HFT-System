"""Momentum trading strategy."""

import time
from typing import Dict, List
from collections import defaultdict, deque
import numpy as np

from src.trading.strategies.base import TradingStrategy
from src.trading.types import (
    TradingStrategyType, Order, OrderSide, OrderType
)


class MomentumStrategy(TradingStrategy):
    """Momentum trading with ML-optimized entry/exit."""

    def __init__(self, params: Dict = None):
        default_params = {
            'lookback_period': 20,
            'entry_threshold': 2.0,
            'exit_threshold': 0.5,
            'stop_loss_bps': 50,
            'take_profit_bps': 100,
            'max_position': 1000,
            'hold_time': 300,
            'ml_signal_weight': 0.7
        }
        params = {**default_params, **(params or {})}
        super().__init__(TradingStrategyType.MOMENTUM, params)

        self.signal_history = defaultdict(deque)
        self.entry_prices = {}
        self.entry_times = {}

    async def generate_signals(
        self,
        market_data: Dict,
        ml_predictions: Dict
    ) -> List[Order]:
        """Generate momentum trading signals."""
        orders = []

        for symbol in market_data.get('symbols', []):
            market_state = market_data[symbol]
            current_position = self.positions[symbol].quantity

            self.signal_history[symbol].append({
                'price': market_state['mid_price'],
                'volume': market_state['volume'],
                'timestamp': time.time()
            })

            if len(self.signal_history[symbol]) > self.params['lookback_period']:
                self.signal_history[symbol].popleft()

            if len(self.signal_history[symbol]) < self.params['lookback_period']:
                continue

            prices = [s['price'] for s in self.signal_history[symbol]]
            returns = np.diff(np.log(prices))

            recent_return = returns[-1]
            avg_return = np.mean(returns[:-1])
            std_return = np.std(returns[:-1])

            if std_return > 0:
                z_score = (recent_return - avg_return) / std_return
            else:
                z_score = 0

            ml_signal = ml_predictions.get(f'momentum_signal_{symbol}', 0)
            combined_signal = (
                self.params['ml_signal_weight'] * ml_signal +
                (1 - self.params['ml_signal_weight']) * z_score
            )

            routing = ml_predictions.get(f'routing_{symbol}', {})
            best_venue = routing.get('venue', 'NYSE')
            predicted_latency = routing.get('predicted_latency_us', 1000)

            if current_position == 0:
                if combined_signal > self.params['entry_threshold']:
                    order = Order(
                        order_id=f"MOM_B_{len(self.orders):08d}",
                        symbol=symbol,
                        venue=best_venue,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=self.params['max_position'],
                        price=market_state['ask_price'],
                        timestamp=time.time(),
                        strategy=TradingStrategyType.MOMENTUM,
                        predicted_latency_us=predicted_latency,
                        routing_confidence=routing.get('confidence', 0.5),
                        market_regime=ml_predictions.get('regime', 'normal')
                    )
                    orders.append(order)
                    self.entry_prices[symbol] = market_state['mid_price']
                    self.entry_times[symbol] = time.time()

                elif combined_signal < -self.params['entry_threshold']:
                    order = Order(
                        order_id=f"MOM_S_{len(self.orders):08d}",
                        symbol=symbol,
                        venue=best_venue,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        quantity=self.params['max_position'],
                        price=market_state['bid_price'],
                        timestamp=time.time(),
                        strategy=TradingStrategyType.MOMENTUM,
                        predicted_latency_us=predicted_latency,
                        routing_confidence=routing.get('confidence', 0.5),
                        market_regime=ml_predictions.get('regime', 'normal')
                    )
                    orders.append(order)
                    self.entry_prices[symbol] = market_state['mid_price']
                    self.entry_times[symbol] = time.time()

            else:
                entry_price = self.entry_prices.get(
                    symbol, market_state['mid_price']
                )
                time_in_position = time.time() - self.entry_times.get(
                    symbol, time.time()
                )
                current_price = market_state['mid_price']

                if current_position > 0:
                    pnl_bps = (
                        (current_price - entry_price) / entry_price * 10000
                    )

                    exit_signal = (
                        combined_signal < self.params['exit_threshold'] or
                        pnl_bps <= -self.params['stop_loss_bps'] or
                        pnl_bps >= self.params['take_profit_bps'] or
                        time_in_position > self.params['hold_time']
                    )

                    if exit_signal:
                        order = Order(
                            order_id=f"MOM_EXIT_S_{len(self.orders):08d}",
                            symbol=symbol,
                            venue=best_venue,
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            quantity=current_position,
                            price=market_state['bid_price'],
                            timestamp=time.time(),
                            strategy=TradingStrategyType.MOMENTUM,
                            predicted_latency_us=predicted_latency,
                            routing_confidence=routing.get('confidence', 0.5),
                            market_regime=ml_predictions.get('regime', 'normal')
                        )
                        orders.append(order)

                else:
                    pnl_bps = (
                        (entry_price - current_price) / entry_price * 10000
                    )

                    exit_signal = (
                        combined_signal > -self.params['exit_threshold'] or
                        pnl_bps <= -self.params['stop_loss_bps'] or
                        pnl_bps >= self.params['take_profit_bps'] or
                        time_in_position > self.params['hold_time']
                    )

                    if exit_signal:
                        order = Order(
                            order_id=f"MOM_EXIT_B_{len(self.orders):08d}",
                            symbol=symbol,
                            venue=best_venue,
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            quantity=abs(current_position),
                            price=market_state['ask_price'],
                            timestamp=time.time(),
                            strategy=TradingStrategyType.MOMENTUM,
                            predicted_latency_us=predicted_latency,
                            routing_confidence=routing.get('confidence', 0.5),
                            market_regime=ml_predictions.get('regime', 'normal')
                        )
                        orders.append(order)

        return orders
