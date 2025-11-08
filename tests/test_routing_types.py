"""Unit tests for routing types."""

import pytest
from src.ml.routing.types import RoutingAction, RoutingDecision


class TestRoutingAction:
    """Tests for RoutingAction enum."""

    def test_all_actions_defined(self):
        """Test all routing actions are defined."""
        assert RoutingAction.ROUTE_NYSE.value == 0
        assert RoutingAction.ROUTE_NASDAQ.value == 1
        assert RoutingAction.ROUTE_CBOE.value == 2
        assert RoutingAction.ROUTE_IEX.value == 3
        assert RoutingAction.ROUTE_ARCA.value == 4
        assert RoutingAction.HOLD.value == 5
        assert RoutingAction.CANCEL.value == 6

    def test_enum_values_unique(self):
        """Test all enum values are unique."""
        values = [action.value for action in RoutingAction]
        assert len(values) == len(set(values))


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""

    def test_routing_decision_creation(self):
        """Test creating a routing decision."""
        decision = RoutingDecision(
            timestamp=1234567890.0,
            symbol="AAPL",
            action=RoutingAction.ROUTE_NYSE,
            venue="NYSE",
            confidence=0.95,
            expected_latency_us=500.0,
            expected_reward=10.0,
            exploration_rate=0.1,
            state_features={"feature_0": 0.5, "feature_1": 0.8},
        )

        assert decision.timestamp == 1234567890.0
        assert decision.symbol == "AAPL"
        assert decision.action == RoutingAction.ROUTE_NYSE
        assert decision.venue == "NYSE"
        assert decision.confidence == 0.95
        assert decision.expected_latency_us == 500.0
        assert decision.expected_reward == 10.0
        assert decision.exploration_rate == 0.1
        assert len(decision.state_features) == 2

    def test_routing_decision_with_none_venue(self):
        """Test routing decision with None venue (HOLD/CANCEL)."""
        decision = RoutingDecision(
            timestamp=1234567890.0,
            symbol="AAPL",
            action=RoutingAction.HOLD,
            venue=None,
            confidence=0.5,
            expected_latency_us=0.0,
            expected_reward=-0.1,
            exploration_rate=0.0,
            state_features={},
        )

        assert decision.venue is None
        assert decision.action == RoutingAction.HOLD
