"""Trading simulator dataclasses."""

from dataclasses import dataclass
from typing import Optional

from .enums import OrderSide, OrderStatus, OrderType, TradingStrategyType


@dataclass
class Order:
    """Order representation."""

    order_id: str
    symbol: str
    venue: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: float
    timestamp: float
    strategy: TradingStrategyType

    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_price: float = 0.0
    fill_timestamp: Optional[float] = None
    latency_us: Optional[float] = None

    predicted_latency_us: Optional[float] = None
    routing_confidence: Optional[float] = None
    market_regime: Optional[str] = None


@dataclass
class Fill:
    """Trade execution fill."""

    fill_id: str
    order_id: str
    symbol: str
    venue: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: float
    fees: float
    rebate: float

    latency_us: float
    slippage_bps: float
    market_impact_bps: float


@dataclass
class Position:
    """Position tracking."""

    symbol: str
    quantity: int = 0
    average_cost: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_volume: int = 0

    def update_position(self, fill: Fill, current_price: float):
        """Update position with new fill."""
        if fill.side == OrderSide.BUY:
            total_cost = self.quantity * self.average_cost + fill.quantity * fill.price
            self.quantity += fill.quantity
            self.average_cost = total_cost / self.quantity if self.quantity > 0 else 0
        else:
            if self.quantity > 0:
                realized = fill.quantity * (fill.price - self.average_cost)
                self.realized_pnl += realized
            self.quantity -= fill.quantity

        if self.quantity != 0:
            self.unrealized_pnl = self.quantity * (current_price - self.average_cost)
        else:
            self.unrealized_pnl = 0.0

        self.total_volume += fill.quantity
