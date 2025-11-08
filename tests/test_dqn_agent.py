"""Unit tests for DQN agent."""

import pytest
import numpy as np
import torch
from src.ml.routing.dqn_agent import DQNRouter


class TestDQNRouter:
    """Tests for DQN routing agent."""

    def test_initialization(self):
        """Test DQN router initializes correctly."""
        router = DQNRouter(state_size=25, action_size=7)
        assert router.state_size == 25
        assert router.action_size == 7
        assert router.gamma == 0.99
        assert router.tau == 0.001
        assert isinstance(router.q_network, torch.nn.Module)

    def test_act_exploitation(self):
        """Test action selection in exploitation mode."""
        router = DQNRouter(state_size=25, action_size=7)
        state = np.random.randn(25)

        action, confidence = router.act(state, epsilon=0.0)
        assert isinstance(action, int)
        assert 0 <= action < 7
        assert 0.0 <= confidence <= 1.0

    def test_act_exploration(self):
        """Test action selection in exploration mode."""
        router = DQNRouter(state_size=25, action_size=7)
        state = np.random.randn(25)

        action, confidence = router.act(state, epsilon=1.0)
        assert isinstance(action, int)
        assert 0 <= action < 7

    def test_step_and_learn(self):
        """Test stepping through experiences and learning."""
        router = DQNRouter(state_size=25, action_size=7, buffer_size=1000)
        state = np.random.randn(25)
        next_state = np.random.randn(25)

        for i in range(100):
            router.step(state, action=i % 7, reward=1.0, next_state=next_state, done=False)

        assert len(router.memory) == 100

    def test_training_loss_tracking(self):
        """Test training losses are tracked."""
        router = DQNRouter(state_size=25, action_size=7)
        state = np.random.randn(25)
        next_state = np.random.randn(25)

        for i in range(200):
            router.step(state, i % 7, 1.0, next_state, False)

        assert len(router.training_losses) > 0

    def test_save_and_load(self, tmp_path):
        """Test saving and loading model."""
        router = DQNRouter(state_size=25, action_size=7)
        state = np.random.randn(25)

        original_action, _ = router.act(state, epsilon=0.0)

        filepath = tmp_path / "dqn_test.pt"
        router.save(str(filepath))

        router2 = DQNRouter(state_size=25, action_size=7)
        router2.load(str(filepath))

        loaded_action, _ = router2.act(state, epsilon=0.0)
        assert original_action == loaded_action
