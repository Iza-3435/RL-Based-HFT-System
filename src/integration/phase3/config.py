"""Configuration and constants for Phase 3 integration."""

import logging
from typing import List
from data.real_market_data_generator import VenueConfig

EXPANDED_STOCK_LIST = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "TSLA",
    "NVDA",
    "META",
    "AMZN",
    "NFLX",
    "JPM",
    "BAC",
    "WFC",
    "GS",
    "C",
    "JNJ",
    "PFE",
    "UNH",
    "ABBV",
    "PG",
    "KO",
    "XOM",
    "CVX",
    "DIS",
    "SPY",
    "QQQ",
    "IWM",
    "GLD",
    "TLT",
]

FAST_MODE = False
BALANCED_MODE = False
PRODUCTION_MODE = True


def get_venue_configs() -> dict[str, VenueConfig]:
    """Get venue configuration for all trading venues."""
    return {
        "NYSE": VenueConfig("NYSE", 850, (50, 200), 0.001, 1.5),
        "NASDAQ": VenueConfig("NASDAQ", 920, (60, 180), 0.0008, 1.3),
        "CBOE": VenueConfig("CBOE", 1100, (80, 250), 0.0015, 1.8),
        "IEX": VenueConfig("IEX", 870, (55, 190), 0.0012, 1.6),
        "ARCA": VenueConfig("ARCA", 880, (60, 210), 0.0009, 1.4),
    }


def configure_logging() -> logging.Logger:
    """Configure production logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def get_training_config(mode: str = "production") -> dict:
    """Get training configuration based on mode."""
    if mode == "fast":
        return {
            "epochs": 5,
            "batch_size": 32,
            "routing_episodes": 50,
            "sequence_length": 10,
            "update_threshold": 10,
        }
    elif mode == "balanced":
        return {
            "epochs": 25,
            "batch_size": 64,
            "routing_episodes": 500,
            "sequence_length": 30,
            "update_threshold": 25,
        }
    else:
        return {
            "epochs": 100,
            "batch_size": 128,
            "routing_episodes": 2000,
            "sequence_length": 50,
            "update_threshold": 100,
        }


def get_risk_limits() -> dict:
    """Get risk management limits."""
    return {
        "max_loss": -50000,
        "pnl_volatility_threshold": 10000,
        "max_position_size": 1000,
        "max_exposure": 100000,
    }
