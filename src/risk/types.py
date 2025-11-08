"""Risk management types."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class RiskMetric(Enum):
    """Types of risk metrics tracked."""
    POSITION = "position"
    EXPOSURE = "exposure"
    CONCENTRATION = "concentration"
    DRAWDOWN = "drawdown"
    VAR = "value_at_risk"
    STRESS = "stress_test"
    OPERATIONAL = "operational"


class RiskLevel(Enum):
    """Risk severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskLimit:
    """Risk limit configuration."""
    metric: RiskMetric
    limit_type: str
    threshold: float
    current_value: float = 0.0
    breach_count: int = 0
    last_breach_time: Optional[float] = None
    action: str = "alert"


@dataclass
class RiskAlert:
    """Risk alert notification."""
    timestamp: float
    metric: RiskMetric
    level: RiskLevel
    message: str
    current_value: float
    limit_value: float
    action_taken: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class PnLComponent:
    """P&L attribution component."""
    source: str = ""
    gross_pnl: float = 0.0
    fees: float = 0.0
    rebates: float = 0.0
    market_impact: float = 0.0
    latency_cost: float = 0.0
    net_pnl: float = 0.0
    trade_count: int = 0
    volume: int = 0

    @property
    def pnl_per_trade(self) -> float:
        """Calculate P&L per trade."""
        return self.net_pnl / self.trade_count if self.trade_count > 0 else 0

    @property
    def pnl_per_share(self) -> float:
        """Calculate P&L per share."""
        return self.net_pnl / self.volume if self.volume > 0 else 0
