"""Execution fee schedules and configuration."""

from typing import Dict

DEFAULT_FEE_SCHEDULE: Dict[str, Dict[str, float]] = {
    "NYSE": {"maker_fee": -0.20, "taker_fee": 0.30},
    "NASDAQ": {"maker_fee": -0.25, "taker_fee": 0.30},
    "CBOE": {"maker_fee": -0.23, "taker_fee": 0.28},
    "IEX": {"maker_fee": 0.0, "taker_fee": 0.09},
    "ARCA": {"maker_fee": -0.20, "taker_fee": 0.30},
}
