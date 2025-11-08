"""Main latency simulator orchestration."""

import time
import logging
from typing import Dict, List, Optional, Deque
from collections import deque

from .types import VenueLatencyProfile, LatencyBreakdown
from .message_queue import MessageQueue
from .venue_config import initialize_venue_profiles, initialize_message_queues
from .market_factors import MarketFactorCalculator
from .latency_calculator import LatencyCalculator

logger = logging.getLogger(__name__)


class LatencySimulator:
    """Comprehensive latency simulation system."""

    def __init__(self, venues: List[str]) -> None:
        self.venues = venues
        self.venue_profiles = initialize_venue_profiles()
        self.message_queues = initialize_message_queues(self.venue_profiles)

        self.market_factors = MarketFactorCalculator()
        self.calculator = LatencyCalculator(self.message_queues)

        self.latency_history: Deque[LatencyBreakdown] = deque(maxlen=10000)
        self.prediction_errors: List[float] = []
        self.congestion_events: Deque = deque(maxlen=100)

        logger.info(f"LatencySimulator initialized for {len(venues)} venues")

    def update_market_conditions(
        self, symbol: str, volatility: float, volume_factor: float
    ) -> None:
        """Update market conditions affecting latency."""
        self.market_factors.update_market_conditions(symbol, volatility, volume_factor)

    def simulate_latency(
        self,
        venue: str,
        symbol: str,
        order_type: str = "limit",
        predicted_latency_us: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> LatencyBreakdown:
        """Simulate complete latency for an order."""
        if timestamp is None:
            timestamp = time.time()

        if venue not in self.venue_profiles:
            raise ValueError(f"Unknown venue: {venue}")

        profile = self.venue_profiles[venue]

        time_factor = self.market_factors.get_time_of_day_factor(timestamp)
        volatility_factor = self.market_factors.get_volatility_factor(symbol, profile)
        congestion_factor = self.market_factors.get_congestion_factor()

        network_latency = self.calculator.calculate_network_latency(
            profile, time_factor, congestion_factor
        )
        queue_delay = self.calculator.calculate_queue_delay(venue, congestion_factor)
        exchange_delay = self.calculator.calculate_exchange_delay(
            profile, order_type, volatility_factor
        )
        processing_delay = self.calculator.calculate_processing_delay(profile, congestion_factor)

        breakdown = self.calculator.create_latency_breakdown(
            timestamp=timestamp,
            venue=venue,
            symbol=symbol,
            order_type=order_type,
            network_latency=network_latency,
            queue_delay=queue_delay,
            exchange_delay=exchange_delay,
            processing_delay=processing_delay,
            predicted_latency_us=predicted_latency_us,
            congestion_level=self.market_factors.congestion_level,
            time_factor=time_factor,
            volatility_factor=volatility_factor,
        )

        self.latency_history.append(breakdown)

        if predicted_latency_us is not None:
            error = abs(breakdown.total_latency_us - predicted_latency_us)
            self.prediction_errors.append(error)

        return breakdown

    def reset_statistics(self) -> None:
        """Reset performance tracking."""
        self.latency_history.clear()
        self.prediction_errors.clear()
        self.congestion_events.clear()

        for queue in self.message_queues.values():
            queue.queue_full_events = 0
