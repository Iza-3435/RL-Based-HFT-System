"""
Routing types and data structures.

This module defines the core types used throughout the RL routing system,
including action enumerations and routing decision structures.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class RoutingAction(Enum):
    """Available routing actions for venue selection.

    Attributes:
        ROUTE_NYSE: Route order to NYSE venue
        ROUTE_NASDAQ: Route order to NASDAQ venue
        ROUTE_CBOE: Route order to CBOE venue
        ROUTE_IEX: Route order to IEX venue
        ROUTE_ARCA: Route order to ARCA venue
        HOLD: Hold order and wait for better conditions
        CANCEL: Cancel the pending order
    """

    ROUTE_NYSE = 0
    ROUTE_NASDAQ = 1
    ROUTE_CBOE = 2
    ROUTE_IEX = 3
    ROUTE_ARCA = 4
    HOLD = 5
    CANCEL = 6


@dataclass
class RoutingDecision:
    """Structured routing decision output from RL agents.

    Attributes:
        timestamp: Unix timestamp of decision
        symbol: Trading symbol
        action: Selected routing action
        venue: Target venue name (None for HOLD/CANCEL)
        confidence: Agent confidence in decision (0-1)
        expected_latency_us: Predicted latency in microseconds
        expected_reward: Expected reward value
        exploration_rate: Epsilon value for exploration
        state_features: State feature vector as dictionary
    """

    timestamp: float
    symbol: str
    action: RoutingAction
    venue: Optional[str]
    confidence: float
    expected_latency_us: float
    expected_reward: float
    exploration_rate: float
    state_features: Dict[str, float]
