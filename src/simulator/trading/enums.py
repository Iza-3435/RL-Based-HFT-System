"""Trading simulator enums."""

from enum import Enum


class OrderType(Enum):
    """Order types supported by the simulator."""

    MARKET = "market"
    LIMIT = "limit"
    IOC = "immediate_or_cancel"
    FOK = "fill_or_kill"
    HIDDEN = "hidden"


class OrderSide(Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order execution status."""

    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class TradingStrategyType(Enum):
    """Trading strategy types."""

    MARKET_MAKING = "market_making"
    ARBITRAGE = "arbitrage"
    MOMENTUM = "momentum"
