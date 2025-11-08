"""Order execution engine with latency and market impact."""

import logging
from collections import defaultdict
from typing import Dict, Optional

import numpy as np

from .dataclasses import Fill, Order
from .enums import OrderSide, OrderStatus, OrderType
from .market_impact import MarketImpactModel

logger = logging.getLogger(__name__)


class OrderExecutionEngine:
    """Realistic order execution with latency, market impact, and slippage."""

    def __init__(self, fee_schedule: Dict[str, Dict] = None):
        self.market_impact_model = MarketImpactModel()
        self.execution_queue = []
        self.order_book = defaultdict(lambda: {"bids": [], "asks": []})

        self.fee_schedule = fee_schedule or {
            "NYSE": {"maker_fee": -0.20, "taker_fee": 0.30},
            "NASDAQ": {"maker_fee": -0.25, "taker_fee": 0.30},
            "CBOE": {"maker_fee": -0.23, "taker_fee": 0.28},
            "IEX": {"maker_fee": 0.0, "taker_fee": 0.09},
            "ARCA": {"maker_fee": -0.20, "taker_fee": 0.30},
        }

        self.fill_count = 0
        self.total_latency_us = 0
        self.total_slippage_bps = 0

    async def execute_order(
        self, order: Order, market_state: Dict, actual_latency_us: float
    ) -> Optional[Fill]:
        """Execute order with realistic market mechanics."""
        arrival_price = market_state["mid_price"]

        price_drift = self._simulate_price_drift(
            arrival_price, actual_latency_us, market_state["volatility"]
        )
        execution_price = arrival_price + price_drift

        permanent_impact, temporary_impact = self.market_impact_model.calculate_impact(
            order, market_state, actual_latency_us
        )

        if order.order_type == OrderType.MARKET:
            if order.side == OrderSide.BUY:
                fill_price = market_state["ask_price"] * (1 + temporary_impact / 10000)
            else:
                fill_price = market_state["bid_price"] * (1 - temporary_impact / 10000)

        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                if order.price >= market_state["ask_price"]:
                    fill_price = market_state["ask_price"] * (
                        1 + temporary_impact / 10000
                    )
                else:
                    if self._check_queue_position(order, market_state):
                        fill_price = order.price
                    else:
                        return None
            else:
                if order.price <= market_state["bid_price"]:
                    fill_price = market_state["bid_price"] * (
                        1 - temporary_impact / 10000
                    )
                else:
                    if self._check_queue_position(order, market_state):
                        fill_price = order.price
                    else:
                        return None

        is_maker = order.order_type == OrderType.LIMIT and fill_price == order.price
        fee_structure = self.fee_schedule[order.venue]

        if is_maker:
            fee_bps = fee_structure["maker_fee"]
        else:
            fee_bps = fee_structure["taker_fee"]

        fees = (
            abs(fee_bps) * fill_price * order.quantity / 10000 if fee_bps > 0 else 0
        )
        rebate = (
            abs(fee_bps) * fill_price * order.quantity / 10000 if fee_bps < 0 else 0
        )

        if order.order_type == OrderType.MARKET:
            expected_price = arrival_price
        else:
            expected_price = order.price

        slippage_bps = abs(fill_price - expected_price) / expected_price * 10000

        fill = Fill(
            fill_id=f"F{self.fill_count:08d}",
            order_id=order.order_id,
            symbol=order.symbol,
            venue=order.venue,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            timestamp=order.timestamp + actual_latency_us / 1e6,
            fees=fees,
            rebate=rebate,
            latency_us=actual_latency_us,
            slippage_bps=slippage_bps,
            market_impact_bps=temporary_impact,
        )

        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_price = fill_price
        order.fill_timestamp = fill.timestamp
        order.latency_us = actual_latency_us

        self.fill_count += 1
        self.total_latency_us += actual_latency_us
        self.total_slippage_bps += slippage_bps

        self._apply_permanent_impact(market_state, order.side, permanent_impact)

        return fill

    def _simulate_price_drift(
        self, price: float, latency_us: float, volatility: float
    ) -> float:
        """Simulate price movement during latency period."""
        latency_days = latency_us / (1e6 * 60 * 60 * 6.5)

        drift = 0
        diffusion = volatility * np.sqrt(latency_days) * np.random.randn()

        return price * (drift + diffusion)

    def _check_queue_position(self, order: Order, market_state: Dict) -> bool:
        """Check if limit order would get filled based on queue position."""
        queue_position_factor = 0.3

        size_factor = 1.0 - min(
            order.quantity / market_state.get("average_trade_size", 100), 0.5
        )

        fill_probability = queue_position_factor * size_factor

        return np.random.random() < fill_probability

    def _apply_permanent_impact(
        self, market_state: Dict, side: OrderSide, impact_bps: float
    ):
        """Apply permanent market impact to prices."""
        impact_pct = impact_bps / 10000

        if side == OrderSide.BUY:
            market_state["bid_price"] *= 1 + impact_pct * 0.5
            market_state["ask_price"] *= 1 + impact_pct * 0.5
            market_state["mid_price"] *= 1 + impact_pct * 0.5
        else:
            market_state["bid_price"] *= 1 - impact_pct * 0.5
            market_state["ask_price"] *= 1 - impact_pct * 0.5
            market_state["mid_price"] *= 1 - impact_pct * 0.5

    def get_execution_stats(self) -> Dict[str, float]:
        """Get execution statistics."""
        return {
            "total_fills": self.fill_count,
            "avg_latency_us": self.total_latency_us / max(self.fill_count, 1),
            "avg_slippage_bps": self.total_slippage_bps / max(self.fill_count, 1),
            "fill_rate": 1.0,
        }
