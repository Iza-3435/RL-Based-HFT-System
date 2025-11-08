"""Unit tests for PPO agent."""

import pytest
import numpy as np
from src.ml.routing.ppo_agent import PPORouter


class TestPPORouter:
    """Tests for PPO routing agent."""

    def test_initialization(self):
        """Test PPO router initializes correctly."""
        router = PPORouter(state_size=25, action_size=7)
        assert router.state_size == 25
        assert router.action_size == 7
        assert router.gamma == 0.99
        assert router.clip_param == 0.2

    def test_act(self):
        """Test action selection."""
        router = PPORouter(state_size=25, action_size=7)
        state = np.random.randn(25)

        action, confidence, value, log_prob = router.act(state)
        assert isinstance(action, int)
        assert 0 <= action < 7
        assert 0.0 <= confidence <= 1.0
        assert isinstance(value, float)
        assert isinstance(log_prob, float)

    def test_store_transition(self):
        """Test storing transitions."""
        router = PPORouter(state_size=25, action_size=7)
        state = np.random.randn(25)

        router.store_transition(
            state=state,
            action=0,
            reward=1.0,
            log_prob=-1.5,
            value=5.0,
            done=False,
        )

        assert len(router.states) == 1
        assert len(router.actions) == 1
        assert len(router.rewards) == 1

    def test_train_empty_buffer(self):
        """Test training with empty buffer."""
        router = PPORouter(state_size=25, action_size=7)
        router.train()

    def test_train_with_experiences(self):
        """Test training with collected experiences."""
        router = PPORouter(state_size=25, action_size=7)
        state = np.random.randn(25)

        for i in range(20):
            action, confidence, value, log_prob = router.act(state)
            router.store_transition(state, action, float(i), log_prob, value, False)

        router.train(next_value=0.0)
        assert len(router.states) == 0

    def test_calculate_returns(self):
        """Test return calculation."""
        router = PPORouter(state_size=25, action_size=7, gamma=0.99)
        state = np.random.randn(25)

        for i in range(10):
            router.store_transition(state, 0, 1.0, -1.5, 5.0, False)

        returns = router._calculate_returns(next_value=0.0)
        assert len(returns) == 10
        assert returns[0] > returns[-1]
