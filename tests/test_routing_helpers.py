"""Unit tests for routing helper functions."""

import pytest
import numpy as np
from src.ml.routing.types import RoutingAction, RoutingDecision
from src.ml.routing.routing_helpers import (
    calculate_expected_reward,
    calculate_actual_reward,
)


class TestCalculateExpectedReward:
    """Tests for expected reward calculation."""

    def test_hold_action(self):
        """Test expected reward for HOLD action."""
        reward = calculate_expected_reward(
            action=RoutingAction.HOLD, expected_latency=500.0, urgency=0.5
        )
        assert reward < 0
        assert reward == -0.05

    def test_cancel_action(self):
        """Test expected reward for CANCEL action."""
        reward = calculate_expected_reward(
            action=RoutingAction.CANCEL, expected_latency=0.0, urgency=0.5
        )
        assert reward == -0.5

    def test_routing_action_low_latency(self):
        """Test expected reward for routing with low latency."""
        reward = calculate_expected_reward(
            action=RoutingAction.ROUTE_NYSE, expected_latency=200.0, urgency=1.0
        )
        assert reward > 0

    def test_routing_action_high_latency(self):
        """Test expected reward for routing with high latency."""
        reward = calculate_expected_reward(
            action=RoutingAction.ROUTE_NYSE, expected_latency=5000.0, urgency=1.0
        )
        assert reward > 0
        low_latency_reward = calculate_expected_reward(RoutingAction.ROUTE_NYSE, 200.0, 1.0)
        assert reward < low_latency_reward


class TestCalculateActualReward:
    """Tests for actual reward calculation."""

    def test_failed_execution(self):
        """Test reward for failed execution."""
        decision = RoutingDecision(
            timestamp=0.0,
            symbol="AAPL",
            action=RoutingAction.ROUTE_NYSE,
            venue="NYSE",
            confidence=0.9,
            expected_latency_us=500.0,
            expected_reward=10.0,
            exploration_rate=0.1,
            state_features={},
        )

        reward = calculate_actual_reward(decision, actual_latency=1000.0, fill_success=False)
        assert reward == -5.0

    def test_excellent_latency(self):
        """Test reward for excellent latency (<500us)."""
        decision = RoutingDecision(
            timestamp=0.0,
            symbol="AAPL",
            action=RoutingAction.ROUTE_NYSE,
            venue="NYSE",
            confidence=0.9,
            expected_latency_us=400.0,
            expected_reward=10.0,
            exploration_rate=0.1,
            state_features={},
        )

        reward = calculate_actual_reward(decision, actual_latency=450.0, fill_success=True)
        assert reward > 10.0

    def test_good_latency(self):
        """Test reward for good latency (500-1000us)."""
        decision = RoutingDecision(
            timestamp=0.0,
            symbol="AAPL",
            action=RoutingAction.ROUTE_NYSE,
            venue="NYSE",
            confidence=0.9,
            expected_latency_us=800.0,
            expected_reward=10.0,
            exploration_rate=0.1,
            state_features={},
        )

        reward = calculate_actual_reward(decision, actual_latency=900.0, fill_success=True)
        assert reward > 10.0

    def test_poor_latency(self):
        """Test reward for poor latency (>2000us)."""
        decision = RoutingDecision(
            timestamp=0.0,
            symbol="AAPL",
            action=RoutingAction.ROUTE_NYSE,
            venue="NYSE",
            confidence=0.9,
            expected_latency_us=3000.0,
            expected_reward=10.0,
            exploration_rate=0.1,
            state_features={},
        )

        reward = calculate_actual_reward(decision, actual_latency=3500.0, fill_success=True)
        assert reward < 15.0

    def test_prediction_accuracy_bonus(self):
        """Test prediction accuracy bonus."""
        decision = RoutingDecision(
            timestamp=0.0,
            symbol="AAPL",
            action=RoutingAction.ROUTE_NYSE,
            venue="NYSE",
            confidence=0.9,
            expected_latency_us=500.0,
            expected_reward=10.0,
            exploration_rate=0.1,
            state_features={},
        )

        accurate_reward = calculate_actual_reward(decision, actual_latency=500.0, fill_success=True)
        inaccurate_reward = calculate_actual_reward(
            decision, actual_latency=2000.0, fill_success=True
        )

        assert accurate_reward > inaccurate_reward
