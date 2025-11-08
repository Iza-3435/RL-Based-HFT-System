"""Market condition factors affecting latency."""

import numpy as np
from datetime import datetime
from typing import Dict
from collections import defaultdict
from .types import VenueLatencyProfile, CongestionLevel


class MarketFactorCalculator:
    """Calculate latency multipliers based on market conditions."""

    def __init__(self) -> None:
        self.current_volatility: Dict[str, float] = defaultdict(lambda: 0.02)
        self.current_volume: Dict[str, float] = defaultdict(lambda: 1.0)
        self.congestion_level = CongestionLevel.NORMAL
        self.base_congestion = 0.0

    def update_market_conditions(
        self, symbol: str, volatility: float, volume_factor: float
    ) -> None:
        """Update market conditions affecting latency."""
        self.current_volatility[symbol] = volatility
        self.current_volume[symbol] = volume_factor

        volatility_stress = min(volatility / 0.05, 2.0)
        volume_stress = min(volume_factor / 2.0, 2.0)
        combined_stress = (volatility_stress + volume_stress) / 2.0

        if combined_stress > 1.5:
            self.congestion_level = CongestionLevel.CRITICAL
            self.base_congestion = 0.8
        elif combined_stress > 1.2:
            self.congestion_level = CongestionLevel.HIGH
            self.base_congestion = 0.6
        elif combined_stress > 0.8:
            self.congestion_level = CongestionLevel.NORMAL
            self.base_congestion = 0.3
        else:
            self.congestion_level = CongestionLevel.LOW
            self.base_congestion = 0.1

    def get_time_of_day_factor(self, timestamp: float) -> float:
        """Calculate time-of-day latency multiplier."""
        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour
        minute = dt.minute

        if hour == 9 and minute >= 30:
            return 2.0
        elif hour == 10:
            return 1.6
        elif 11 <= hour <= 12:
            return 1.2
        elif hour == 12:
            return 0.8
        elif 13 <= hour <= 14:
            return 1.1
        elif hour == 15:
            return 1.8
        elif hour == 16 and minute <= 30:
            return 2.5
        elif 17 <= hour <= 19:
            return 1.3
        else:
            return 0.9

    def get_volatility_factor(self, symbol: str, profile: VenueLatencyProfile) -> float:
        """Calculate volatility-based latency multiplier."""
        volatility = self.current_volatility[symbol]
        vol_ratio = volatility / 0.02
        factor = 1.0 + (vol_ratio - 1.0) * (profile.volatility_sensitivity - 1.0)
        return max(0.5, min(3.0, factor))

    def get_congestion_factor(self) -> float:
        """Get current network congestion factor."""
        random_variation = np.random.normal(0, 0.1)
        congestion = self.base_congestion + random_variation
        return max(0.0, min(1.0, congestion))
