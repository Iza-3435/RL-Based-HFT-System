"""Unit tests for multi-armed bandit agent."""

import pytest
from src.ml.routing.bandit_agent import MultiArmedBanditRouter


class TestMultiArmedBanditRouter:
    """Tests for bandit routing agent."""

    def test_initialization(self):
        """Test bandit initializes correctly."""
        venues = ["NYSE", "NASDAQ", "ARCA"]
        bandit = MultiArmedBanditRouter(venues, exploration_rate=0.1)

        assert bandit.venues == venues
        assert bandit.exploration_rate == 0.1
        assert all(v in bandit.alpha for v in venues)
        assert all(v in bandit.beta for v in venues)

    def test_thompson_sampling(self):
        """Test Thompson sampling selection."""
        venues = ["NYSE", "NASDAQ", "ARCA"]
        bandit = MultiArmedBanditRouter(venues)

        venue, confidence = bandit.select_venue(method="thompson")
        assert venue in venues
        assert 0.0 <= confidence <= 1.0

    def test_ucb_selection(self):
        """Test UCB selection."""
        venues = ["NYSE", "NASDAQ", "ARCA"]
        bandit = MultiArmedBanditRouter(venues)

        venue, confidence = bandit.select_venue(method="ucb")
        assert venue in venues

    def test_epsilon_greedy(self):
        """Test epsilon-greedy selection."""
        venues = ["NYSE", "NASDAQ", "ARCA"]
        bandit = MultiArmedBanditRouter(venues, exploration_rate=0.1)

        venue, confidence = bandit.select_venue(method="epsilon_greedy")
        assert venue in venues

    def test_update_with_positive_reward(self):
        """Test updating with positive reward."""
        venues = ["NYSE", "NASDAQ"]
        bandit = MultiArmedBanditRouter(venues)

        initial_alpha = bandit.alpha["NYSE"]
        bandit.update("NYSE", reward=5.0)

        assert bandit.counts["NYSE"] == 1
        assert bandit.alpha["NYSE"] > initial_alpha
        assert len(bandit.selection_history) == 1

    def test_update_with_negative_reward(self):
        """Test updating with negative reward."""
        venues = ["NYSE", "NASDAQ"]
        bandit = MultiArmedBanditRouter(venues)

        initial_beta = bandit.beta["NYSE"]
        bandit.update("NYSE", reward=-2.0)

        assert bandit.counts["NYSE"] == 1
        assert bandit.beta["NYSE"] > initial_beta

    def test_get_statistics(self):
        """Test statistics generation."""
        venues = ["NYSE", "NASDAQ", "ARCA"]
        bandit = MultiArmedBanditRouter(venues)

        for _ in range(10):
            venue, _ = bandit.select_venue(method="thompson")
            bandit.update(venue, reward=1.0)

        stats = bandit.get_statistics()
        assert "venue_counts" in stats
        assert "venue_values" in stats
        assert "total_rewards" in stats
        assert "thompson_params" in stats
        assert "selection_percentages" in stats

    def test_convergence(self):
        """Test bandit converges to best venue."""
        venues = ["NYSE", "NASDAQ", "ARCA"]
        bandit = MultiArmedBanditRouter(venues)

        for _ in range(100):
            venue, _ = bandit.select_venue(method="thompson")
            reward = 5.0 if venue == "NYSE" else 1.0
            bandit.update(venue, reward)

        stats = bandit.get_statistics()
        assert stats["venue_values"]["NYSE"] > stats["venue_values"]["NASDAQ"]
