"""Main routing environment manager orchestrating all RL agents."""

import time
import json
import logging
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional

from .types import RoutingAction, RoutingDecision
from .trading_env import TradingEnvironment
from .dqn_agent import DQNRouter
from .ppo_agent import PPORouter
from .bandit_agent import MultiArmedBanditRouter
from .routing_helpers import (
    extract_state_features,
    extract_venue_features,
    calculate_expected_reward,
    calculate_actual_reward,
)

logger = logging.getLogger(__name__)


class RoutingEnvironment:
    """Complete routing environment with RL agents."""

    def __init__(
        self,
        latency_predictor: Any,
        market_generator: Any,
        network_simulator: Any,
        order_book_manager: Any,
        feature_extractor: Any,
        venue_list: Optional[List[str]] = None,
    ) -> None:
        self.latency_predictor = latency_predictor
        self.market_generator = market_generator
        self.network_simulator = network_simulator
        self.order_book_manager = order_book_manager
        self.feature_extractor = feature_extractor

        if venue_list:
            self.venues = list(venue_list)
        elif hasattr(market_generator, "venues"):
            self.venues = list(market_generator.venues)
        else:
            self.venues = ["NYSE", "NASDAQ", "ARCA", "IEX", "CBOE"]
            self._setup_fallback_venues()

        self.trading_env = TradingEnvironment(
            self.venues, latency_predictor, market_generator, network_simulator
        )

        state_size = self.trading_env.observation_space.shape[0]
        action_size = self.trading_env.action_space.n

        self.dqn_router = DQNRouter(state_size, action_size)
        self.ppo_router = PPORouter(state_size, action_size)
        self.bandit_router = MultiArmedBanditRouter(self.venues)

        self.training_mode = False
        self.current_agent = "dqn"

        self.routing_decisions: List[RoutingDecision] = []
        self.performance_metrics = {
            "total_decisions": 0,
            "successful_routes": 0,
            "average_latency": 0.0,
            "total_reward": 0.0,
        }

        logger.info(f"RoutingEnvironment initialized with venues: {self.venues}")

    def _setup_fallback_venues(self) -> None:
        """Setup fallback venue configurations."""
        from ...data_layer.market_data import VenueConfig

        venue_configs = {
            "NYSE": VenueConfig("NYSE", 850, (50, 200), 0.001, 1.5),
            "NASDAQ": VenueConfig("NASDAQ", 920, (60, 180), 0.0008, 1.3),
            "ARCA": VenueConfig("ARCA", 880, (60, 210), 0.0009, 1.4),
            "IEX": VenueConfig("IEX", 870, (55, 190), 0.0012, 1.6),
            "CBOE": VenueConfig("CBOE", 1100, (80, 250), 0.0015, 1.8),
        }
        self.market_generator.venues = venue_configs

    def make_routing_decision(self, symbol: str, urgency: float = 0.5) -> RoutingDecision:
        """Make routing decision for given symbol."""
        state = self._get_current_state(symbol)

        if self.current_agent == "dqn":
            epsilon = 0.01 if not self.training_mode else 0.1
            action, confidence = self.dqn_router.act(state, epsilon)
        elif self.current_agent == "ppo":
            action, confidence, value, log_prob = self.ppo_router.act(state)
        else:
            venue, confidence = self.bandit_router.select_venue()
            action = self.venues.index(venue)

        routing_action = RoutingAction(action)

        if routing_action.value < len(self.venues):
            venue = self.venues[routing_action.value]
            features = extract_venue_features(
                symbol, venue, self.feature_extractor, self.latency_predictor
            )
            latency_pred = self.latency_predictor.predict(venue, features)
            expected_latency = latency_pred.predicted_latency_us
        else:
            venue = None
            expected_latency = 0.0

        expected_reward = calculate_expected_reward(routing_action, expected_latency, urgency)

        decision = RoutingDecision(
            timestamp=time.time(),
            symbol=symbol,
            action=routing_action,
            venue=venue,
            confidence=confidence,
            expected_latency_us=expected_latency,
            expected_reward=expected_reward,
            exploration_rate=epsilon if self.current_agent == "dqn" else 0.0,
            state_features={f"feature_{i}": state[i] for i in range(len(state))},
        )

        self.routing_decisions.append(decision)
        self.performance_metrics["total_decisions"] += 1

        return decision

    def update_with_result(
        self, decision: RoutingDecision, actual_latency_us: float, fill_success: bool
    ) -> None:
        """Update agents with actual results."""
        reward = calculate_actual_reward(decision, actual_latency_us, fill_success)

        if fill_success:
            self.performance_metrics["successful_routes"] += 1

        running_avg = self.performance_metrics["average_latency"]
        n = self.performance_metrics["successful_routes"]
        self.performance_metrics["average_latency"] = (
            (running_avg * (n - 1) + actual_latency_us) / n if n > 0 else actual_latency_us
        )
        self.performance_metrics["total_reward"] += reward

        if decision.venue:
            self.bandit_router.update(decision.venue, reward)
            features = np.array(list(decision.state_features.values()))
            self.latency_predictor.update_online(decision.venue, features, actual_latency_us)

        if self.training_mode:
            next_state = self._get_current_state(decision.symbol)
            done = False

            if self.current_agent == "dqn":
                self.dqn_router.step(
                    np.array(list(decision.state_features.values())),
                    decision.action.value,
                    reward,
                    next_state,
                    done,
                )

    def _get_current_state(self, symbol: str) -> np.ndarray:
        """Extract current state for decision making."""
        return extract_state_features(
            self.venues,
            self.network_simulator,
            self.latency_predictor,
            self.feature_extractor,
            self.order_book_manager,
            symbol,
        )

    def train_agents(self, episodes: int = 1000) -> None:
        """Train RL agents."""
        logger.info(f"Training agents for {episodes} episodes...")
        self.training_mode = True

        for episode in range(episodes):
            state = self.trading_env.reset()
            episode_reward = 0
            done = False

            while not done:
                if self.current_agent == "dqn":
                    epsilon = max(0.02, 0.6 * (0.998**episode))
                    action, _ = self.dqn_router.act(state, epsilon)
                    next_state, reward, done, info = self.trading_env.step(action)
                    self.dqn_router.step(state, action, reward, next_state, done)
                    state = next_state
                    episode_reward += reward
                elif self.current_agent == "ppo":
                    action, confidence, value, log_prob = self.ppo_router.act(state)
                    next_state, reward, done, info = self.trading_env.step(action)
                    self.ppo_router.store_transition(state, action, reward, log_prob, value, done)
                    state = next_state
                    episode_reward += reward
                    if done:
                        self.ppo_router.train()

            if episode % 100 == 0:
                logger.info(f"Episode {episode}: Reward = {episode_reward:.2f}")
                self.dqn_router.reward_history.append(episode_reward)

        self.training_mode = False
        logger.info("Training complete")

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        report = {
            "summary": self.performance_metrics.copy(),
            "agent_performance": {},
            "venue_statistics": {},
        }

        report["agent_performance"]["dqn"] = {
            "training_losses": self.dqn_router.training_losses[-100:],
            "reward_history": self.dqn_router.reward_history[-100:],
        }

        report["agent_performance"]["bandit"] = self.bandit_router.get_statistics()

        for venue in self.venues:
            venue_decisions = [d for d in self.routing_decisions if d.venue == venue]
            if venue_decisions:
                latencies = [d.expected_latency_us for d in venue_decisions]
                report["venue_statistics"][venue] = {
                    "selection_count": len(venue_decisions),
                    "avg_expected_latency": np.mean(latencies),
                    "min_expected_latency": np.min(latencies),
                    "max_expected_latency": np.max(latencies),
                }

        return report

    def save_models(self, directory: str) -> None:
        """Save all trained models."""
        import os

        os.makedirs(directory, exist_ok=True)
        self.dqn_router.save(os.path.join(directory, "dqn_router.pt"))

        with open(os.path.join(directory, "bandit_stats.json"), "w") as f:
            json.dump(self.bandit_router.get_statistics(), f, indent=2)

        with open(os.path.join(directory, "performance_report.json"), "w") as f:
            json.dump(self.get_performance_report(), f, indent=2, default=str)

        logger.info(f"Models saved to {directory}")

    def load_models(self, directory: str) -> None:
        """Load trained models."""
        import os

        dqn_path = os.path.join(directory, "dqn_router.pt")
        if os.path.exists(dqn_path):
            self.dqn_router.load(dqn_path)

        bandit_path = os.path.join(directory, "bandit_stats.json")
        if os.path.exists(bandit_path):
            with open(bandit_path, "r") as f:
                stats = json.load(f)
                self.bandit_router.alpha = stats["thompson_params"]["alpha"]
                self.bandit_router.beta = stats["thompson_params"]["beta"]
                self.bandit_router.counts = stats["venue_counts"]
                self.bandit_router.values = stats["venue_values"]

        logger.info(f"Models loaded from {directory}")
