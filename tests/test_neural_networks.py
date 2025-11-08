"""Unit tests for neural network architectures."""

import pytest
import torch
import numpy as np
from src.ml.routing.neural_networks import DQNNetwork, NoisyLinear, PolicyNetwork


class TestNoisyLinear:
    """Tests for NoisyLinear layer."""

    def test_noisy_linear_initialization(self):
        """Test NoisyLinear layer initializes correctly."""
        layer = NoisyLinear(10, 5)
        assert layer.in_features == 10
        assert layer.out_features == 5
        assert layer.weight_mu.shape == (5, 10)
        assert layer.weight_sigma.shape == (5, 10)

    def test_noisy_linear_forward(self):
        """Test forward pass."""
        layer = NoisyLinear(10, 5)
        x = torch.randn(32, 10)
        output = layer(x)
        assert output.shape == (32, 5)

    def test_reset_noise(self):
        """Test noise reset."""
        layer = NoisyLinear(10, 5)
        epsilon_before = layer.weight_epsilon.clone()
        layer.reset_noise()
        epsilon_after = layer.weight_epsilon
        assert not torch.equal(epsilon_before, epsilon_after)


class TestDQNNetwork:
    """Tests for DQN network."""

    def test_dqn_network_initialization(self):
        """Test DQN network initializes correctly."""
        net = DQNNetwork(state_size=25, action_size=7, hidden_size=128)
        assert net.fc1.in_features == 25
        assert net.advantage_stream.out_features == 7

    def test_dqn_network_forward(self):
        """Test forward pass."""
        net = DQNNetwork(state_size=25, action_size=7)
        state = torch.randn(32, 25)
        q_values = net(state)
        assert q_values.shape == (32, 7)

    def test_dqn_network_dueling_architecture(self):
        """Test dueling architecture produces correct Q-values."""
        net = DQNNetwork(state_size=25, action_size=7)
        state = torch.randn(1, 25)
        q_values = net(state)
        assert q_values.shape == (1, 7)
        assert not torch.isnan(q_values).any()

    def test_reset_noise(self):
        """Test noise reset."""
        net = DQNNetwork(state_size=25, action_size=7)
        net.reset_noise()


class TestPolicyNetwork:
    """Tests for PPO policy network."""

    def test_policy_network_initialization(self):
        """Test policy network initializes correctly."""
        net = PolicyNetwork(state_size=25, action_size=7, hidden_size=128)
        assert net.fc1.in_features == 25
        assert net.actor.out_features == 7
        assert net.critic.out_features == 1

    def test_policy_network_forward(self):
        """Test forward pass returns action probs and value."""
        net = PolicyNetwork(state_size=25, action_size=7)
        state = torch.randn(32, 25)
        action_probs, value = net(state)

        assert action_probs.shape == (32, 7)
        assert value.shape == (32, 1)
        assert torch.allclose(action_probs.sum(dim=1), torch.ones(32), atol=1e-5)
        assert (action_probs >= 0).all() and (action_probs <= 1).all()

    def test_policy_network_value_estimation(self):
        """Test value estimation."""
        net = PolicyNetwork(state_size=25, action_size=7)
        state = torch.randn(1, 25)
        _, value = net(state)
        assert value.shape == (1, 1)
        assert not torch.isnan(value).any()
