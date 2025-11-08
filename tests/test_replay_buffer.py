"""Unit tests for experience replay buffer."""

import pytest
import numpy as np
from src.ml.routing.replay_buffer import PrioritizedReplayBuffer, Experience


class TestPrioritizedReplayBuffer:
    """Tests for prioritized replay buffer."""

    def test_buffer_initialization(self):
        """Test buffer initializes correctly."""
        buffer = PrioritizedReplayBuffer(capacity=1000, alpha=0.6, beta=0.4)
        assert buffer.capacity == 1000
        assert buffer.alpha == 0.6
        assert buffer.beta == 0.4
        assert len(buffer) == 0

    def test_add_experience(self):
        """Test adding experiences."""
        buffer = PrioritizedReplayBuffer(capacity=100)
        state = np.random.randn(25)
        next_state = np.random.randn(25)

        buffer.add(state, action=0, reward=1.0, next_state=next_state, done=False)
        assert len(buffer) == 1

        for i in range(50):
            buffer.add(state, i % 7, float(i), next_state, False)
        assert len(buffer) == 51

    def test_buffer_wraparound(self):
        """Test buffer wraps around at capacity."""
        buffer = PrioritizedReplayBuffer(capacity=10)
        state = np.random.randn(25)
        next_state = np.random.randn(25)

        for i in range(15):
            buffer.add(state, i % 7, float(i), next_state, False)

        assert len(buffer) == 10
        assert buffer.position == 5

    def test_sample_experiences(self):
        """Test sampling experiences."""
        buffer = PrioritizedReplayBuffer(capacity=100)
        state = np.random.randn(25)
        next_state = np.random.randn(25)

        for i in range(50):
            buffer.add(state, i % 7, float(i), next_state, False)

        batch = buffer.sample(batch_size=32)
        assert len(batch) == 32
        assert all(isinstance(exp, Experience) for exp in batch)

    def test_prioritization(self):
        """Test prioritized sampling."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6)
        state = np.random.randn(25)
        next_state = np.random.randn(25)

        for i in range(50):
            buffer.add(state, i % 7, float(i), next_state, False)

        batch = buffer.sample(batch_size=10)
        assert len(batch) == 10

    def test_beta_annealing(self):
        """Test beta annealing over time."""
        buffer = PrioritizedReplayBuffer(capacity=100, beta=0.4)
        state = np.random.randn(25)
        next_state = np.random.randn(25)

        for i in range(50):
            buffer.add(state, i % 7, float(i), next_state, False)

        initial_beta = buffer.beta
        for _ in range(100):
            buffer.sample(batch_size=10)

        assert buffer.beta > initial_beta
        assert buffer.beta <= 1.0
