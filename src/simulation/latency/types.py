"""Latency simulation types and data structures."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class LatencyComponent(Enum):
    """Components of total latency."""

    NETWORK = "network"
    QUEUE = "queue"
    EXCHANGE = "exchange"
    PROCESSING = "processing"


class CongestionLevel(Enum):
    """Network congestion levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LatencyBreakdown:
    """Detailed latency component breakdown."""

    timestamp: float
    venue: str
    symbol: str
    order_type: str
    network_latency_us: float
    queue_delay_us: float
    exchange_delay_us: float
    processing_delay_us: float
    total_latency_us: float
    predicted_latency_us: Optional[float] = None
    congestion_level: CongestionLevel = CongestionLevel.NORMAL
    time_of_day_factor: float = 1.0
    volatility_factor: float = 1.0

    @property
    def prediction_error_us(self) -> Optional[float]:
        """Calculate prediction error if available."""
        if self.predicted_latency_us is not None:
            return abs(self.total_latency_us - self.predicted_latency_us)
        return None

    @property
    def prediction_accuracy_pct(self) -> Optional[float]:
        """Calculate prediction accuracy percentage."""
        if self.predicted_latency_us is not None and self.predicted_latency_us > 0:
            error_pct = (
                abs(self.total_latency_us - self.predicted_latency_us) / self.predicted_latency_us
            )
            return max(0, 100 - error_pct * 100)
        return None


@dataclass
class VenueLatencyProfile:
    """Latency characteristics for each venue."""

    venue: str
    base_network_latency_us: float = 800.0
    network_latency_std_us: float = 150.0
    spike_probability: float = 0.001
    spike_multiplier: float = 4.0
    base_queue_capacity: int = 1000
    processing_rate_msg_per_sec: int = 100000
    queue_buildup_factor: float = 1.5
    base_exchange_delay_us: float = 75.0
    exchange_delay_std_us: float = 25.0
    market_order_delay_multiplier: float = 0.8
    limit_order_delay_multiplier: float = 1.2
    market_open_multiplier: float = 2.0
    market_close_multiplier: float = 2.5
    lunch_multiplier: float = 0.7
    after_hours_multiplier: float = 1.2
    volatility_sensitivity: float = 1.8
    reliability_factor: float = 0.99
    degraded_mode_multiplier: float = 3.0
