"""Cost modeling types."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class LiquidityTier(Enum):
    """Liquidity classification for symbols."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MarketRegime(Enum):
    """Market regime states affecting costs."""
    QUIET = "quiet"
    NORMAL = "normal"
    VOLATILE = "volatile"
    STRESSED = "stressed"


@dataclass
class SlippageParameters:
    """Slippage model parameters by liquidity tier."""
    base_slippage_bps: float
    size_impact_factor: float
    volatility_multiplier: float
    spread_sensitivity: float
    time_of_day_factor: Dict[int, float]


@dataclass
class MarketImpactParameters:
    """Market impact model parameters."""
    temporary_impact_base: float
    permanent_impact_base: float
    volatility_scaling: float
    adv_scaling: float
    sqrt_scaling: bool = True

    venue_multipliers: Dict[str, float] = field(default_factory=dict)

    temporary_half_life_seconds: float = 300
    recovery_rate: float = 0.1


@dataclass
class VenueCostProfile:
    """Comprehensive venue cost profile."""
    name: str

    maker_fee_bps: float
    taker_fee_bps: float
    rebate_bps: float

    impact_multiplier: float
    liquidity_factor: float
    latency_sensitivity: float

    fill_probability: float
    adverse_selection_factor: float

    peak_hours: List[int] = field(default_factory=lambda: [9, 10, 15, 16])
    peak_cost_multiplier: float = 1.3


@dataclass
class ExecutionCostBreakdown:
    """Detailed breakdown of execution costs."""
    order_id: str
    symbol: str
    venue: str
    timestamp: float

    side: str
    quantity: int
    order_price: float
    execution_price: float

    slippage_cost: float
    temporary_impact_cost: float
    permanent_impact_cost: float
    market_impact_cost: float
    latency_cost: float
    fees_paid: float
    rebates_received: float
    opportunity_cost: float

    gross_execution_cost: float
    net_execution_cost: float
    total_transaction_cost: float

    cost_per_share: float
    cost_bps: float
    implementation_shortfall_bps: float
