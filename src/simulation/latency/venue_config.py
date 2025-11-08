"""Venue latency profile configuration."""

from typing import Dict
from .types import VenueLatencyProfile
from .message_queue import MessageQueue


def initialize_venue_profiles() -> Dict[str, VenueLatencyProfile]:
    """Initialize realistic latency profiles for each venue."""
    return {
        "NYSE": VenueLatencyProfile(
            venue="NYSE",
            base_network_latency_us=850.0,
            network_latency_std_us=120.0,
            spike_probability=0.0008,
            spike_multiplier=3.5,
            base_queue_capacity=1500,
            processing_rate_msg_per_sec=120000,
            base_exchange_delay_us=65.0,
            exchange_delay_std_us=20.0,
            market_open_multiplier=1.8,
            market_close_multiplier=2.2,
            volatility_sensitivity=1.5,
            reliability_factor=0.995,
        ),
        "NASDAQ": VenueLatencyProfile(
            venue="NASDAQ",
            base_network_latency_us=750.0,
            network_latency_std_us=100.0,
            spike_probability=0.0005,
            spike_multiplier=3.0,
            base_queue_capacity=2000,
            processing_rate_msg_per_sec=150000,
            base_exchange_delay_us=55.0,
            exchange_delay_std_us=15.0,
            market_open_multiplier=1.6,
            market_close_multiplier=2.0,
            volatility_sensitivity=1.3,
            reliability_factor=0.997,
        ),
        "CBOE": VenueLatencyProfile(
            venue="CBOE",
            base_network_latency_us=950.0,
            network_latency_std_us=180.0,
            spike_probability=0.0012,
            spike_multiplier=4.0,
            base_queue_capacity=1200,
            processing_rate_msg_per_sec=90000,
            base_exchange_delay_us=85.0,
            exchange_delay_std_us=30.0,
            market_open_multiplier=2.0,
            market_close_multiplier=2.5,
            volatility_sensitivity=1.8,
            reliability_factor=0.992,
        ),
        "IEX": VenueLatencyProfile(
            venue="IEX",
            base_network_latency_us=1200.0,
            network_latency_std_us=50.0,
            spike_probability=0.0002,
            spike_multiplier=2.0,
            base_queue_capacity=1000,
            processing_rate_msg_per_sec=80000,
            base_exchange_delay_us=90.0,
            exchange_delay_std_us=10.0,
            market_open_multiplier=1.2,
            market_close_multiplier=1.3,
            volatility_sensitivity=0.8,
            reliability_factor=0.998,
        ),
        "ARCA": VenueLatencyProfile(
            venue="ARCA",
            base_network_latency_us=800.0,
            network_latency_std_us=140.0,
            spike_probability=0.0010,
            spike_multiplier=3.8,
            base_queue_capacity=1300,
            processing_rate_msg_per_sec=110000,
            base_exchange_delay_us=70.0,
            exchange_delay_std_us=25.0,
            market_open_multiplier=1.9,
            market_close_multiplier=2.3,
            volatility_sensitivity=1.6,
            reliability_factor=0.993,
        ),
    }


def initialize_message_queues(
    venue_profiles: Dict[str, VenueLatencyProfile],
) -> Dict[str, MessageQueue]:
    """Initialize message queues for each venue."""
    return {
        venue: MessageQueue(
            capacity=profile.base_queue_capacity,
            processing_rate=profile.processing_rate_msg_per_sec,
        )
        for venue, profile in venue_profiles.items()
    }
