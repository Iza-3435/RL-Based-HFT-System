"""Core latency calculation engine."""

import time
import numpy as np
from typing import Dict, Optional
from .types import VenueLatencyProfile, LatencyBreakdown, CongestionLevel
from .message_queue import MessageQueue


class LatencyCalculator:
    """Calculate latency components with realistic distributions."""

    def __init__(self, message_queues: Dict[str, MessageQueue]) -> None:
        self.message_queues = message_queues

    def calculate_network_latency(
        self, profile: VenueLatencyProfile, time_factor: float, congestion_factor: float
    ) -> float:
        """Simulate network latency with realistic distribution."""
        base_latency = profile.base_network_latency_us * time_factor

        mu = np.log(base_latency)
        sigma = profile.network_latency_std_us / base_latency
        network_latency = np.random.lognormal(mu, sigma)

        network_latency *= 1.0 + congestion_factor * 0.8

        if np.random.random() < profile.spike_probability * congestion_factor:
            network_latency *= profile.spike_multiplier

        if np.random.random() > profile.reliability_factor:
            network_latency *= profile.degraded_mode_multiplier

        return max(50.0, network_latency)

    def calculate_queue_delay(self, venue: str, congestion_factor: float) -> float:
        """Simulate message queue delay."""
        queue = self.message_queues[venue]

        if congestion_factor > 0.5:
            num_additional = int(congestion_factor * 50)
            for i in range(num_additional):
                queue.add_message(f"congestion_msg_{i}")

        success, queue_delay_seconds = queue.add_message(f"order_{time.time()}")
        queue_delay_us = queue_delay_seconds * 1e6
        queue_delay_us += np.random.exponential(20.0)

        return max(0.0, queue_delay_us)

    def calculate_exchange_delay(
        self, profile: VenueLatencyProfile, order_type: str, volatility_factor: float
    ) -> float:
        """Simulate exchange matching engine delay."""
        base_delay = profile.base_exchange_delay_us

        if order_type.lower() == "market":
            base_delay *= profile.market_order_delay_multiplier
        elif order_type.lower() in ["limit", "stop_limit"]:
            base_delay *= profile.limit_order_delay_multiplier

        exchange_delay = np.random.normal(base_delay, profile.exchange_delay_std_us)
        exchange_delay *= 1.0 + (volatility_factor - 1.0) * 0.3

        return max(10.0, exchange_delay)

    def calculate_processing_delay(
        self, profile: VenueLatencyProfile, congestion_factor: float
    ) -> float:
        """Simulate additional processing delays."""
        base_processing = 25.0
        processing_delay = np.random.exponential(base_processing)
        processing_delay *= 1.0 + congestion_factor * 0.5
        return processing_delay

    def create_latency_breakdown(
        self,
        timestamp: float,
        venue: str,
        symbol: str,
        order_type: str,
        network_latency: float,
        queue_delay: float,
        exchange_delay: float,
        processing_delay: float,
        predicted_latency_us: Optional[float],
        congestion_level: CongestionLevel,
        time_factor: float,
        volatility_factor: float,
    ) -> LatencyBreakdown:
        """Create detailed latency breakdown."""
        total_latency = network_latency + queue_delay + exchange_delay + processing_delay

        return LatencyBreakdown(
            timestamp=timestamp,
            venue=venue,
            symbol=symbol,
            order_type=order_type,
            network_latency_us=network_latency,
            queue_delay_us=queue_delay,
            exchange_delay_us=exchange_delay,
            processing_delay_us=processing_delay,
            total_latency_us=total_latency,
            predicted_latency_us=predicted_latency_us,
            congestion_level=congestion_level,
            time_of_day_factor=time_factor,
            volatility_factor=volatility_factor,
        )
