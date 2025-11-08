"""Type definitions for production trading system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TradingMode(Enum):
    FAST = "fast"
    BALANCED = "balanced"
    PRODUCTION = "production"


class MarketRegime(Enum):
    LOW_VOLATILITY = "low_volatility"
    HIGH_VOLATILITY = "high_volatility"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"


@dataclass
class VenueConfig:
    """Configuration for trading venue."""

    name: str
    base_latency_us: int
    latency_range_us: tuple[int, int]
    fee_bps: float
    slippage_bps: float


@dataclass
class OrderRequest:
    """Order request."""

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    venue: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Trade:
    """Executed trade."""

    id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    venue: str
    timestamp: datetime
    latency_us: int
    fees: float


@dataclass
class Position:
    """Trading position."""

    symbol: str
    quantity: int
    avg_price: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class MarketTick:
    """Market data tick."""

    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last_price: float
    volume: int
    venue: str


@dataclass
class PerformanceMetrics:
    """System performance metrics."""

    total_pnl: float
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    avg_latency_us: float
    p95_latency_us: float
    p99_latency_us: float
