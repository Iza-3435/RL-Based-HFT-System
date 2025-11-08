"""OpenAI Gym trading environment for routing."""

import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Dict, List, Tuple, Any, Optional

from .types import RoutingAction


class TradingEnvironment(gym.Env):
    """Gym environment for HFT routing decisions."""

    def __init__(
        self,
        venues: List[str],
        latency_predictor: Any,
        market_generator: Any,
        network_simulator: Any,
    ) -> None:
        super().__init__()
        self.venues = venues
        self.latency_predictor = latency_predictor
        self.market_generator = market_generator
        self.network_simulator = network_simulator

        self.action_space = spaces.Discrete(len(RoutingAction))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)

        self.current_tick: Optional[Any] = None
        self.portfolio_exposure = 0.0
        self.recent_latencies = {venue: deque(maxlen=100) for venue in venues}
        self.recent_rewards: deque = deque(maxlen=100)
        self.execution_count = 0
        self.missed_opportunities = 0

        self.episode_rewards: List[float] = []
        self.episode_latencies: List[float] = []
        self.episode_fills: List[bool] = []

    def reset(self) -> np.ndarray:
        self.portfolio_exposure = 0.0
        self.execution_count = 0
        self.missed_opportunities = 0
        self.episode_rewards = []
        self.episode_latencies = []
        self.episode_fills = []

        for venue in self.venues:
            self.recent_latencies[venue].clear()
        self.recent_rewards.clear()

        self.current_tick = self._get_next_tick()
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        routing_action = RoutingAction(action)

        if routing_action == RoutingAction.HOLD:
            reward = self._execute_hold()
        elif routing_action == RoutingAction.CANCEL:
            reward = self._execute_cancel()
        else:
            venue_idx = routing_action.value
            if venue_idx < len(self.venues):
                venue = self.venues[venue_idx]
                reward = self._execute_route(venue)
            else:
                reward = -1.0

        self.current_tick = self._get_next_tick()
        next_state = self._get_state()
        done = self.execution_count >= 1000 or self.portfolio_exposure > 100000

        info = {
            "execution_count": self.execution_count,
            "portfolio_exposure": self.portfolio_exposure,
            "missed_opportunities": self.missed_opportunities,
            "avg_latency": (np.mean(self.episode_latencies) if self.episode_latencies else 0),
        }

        return next_state, reward, done, info

    def _get_state(self) -> np.ndarray:
        state = []

        for venue in self.venues:
            latency = self.network_simulator.get_current_latency(venue)
            state.append(latency / 10000.0)

        if self.current_tick:
            state.extend(
                [
                    self.current_tick.mid_price / 1000.0,
                    np.log1p(self.current_tick.volume) / 10.0,
                    self.current_tick.spread / 100.0,
                    self.current_tick.volatility * 100.0,
                    float(self.current_tick.imbalance),
                    self.current_tick.trade_intensity,
                    float(self.current_tick.hour) / 24.0,
                    float(self.current_tick.minute) / 60.0,
                    float(self.current_tick.is_market_hours),
                    float(self.current_tick.is_auction),
                ]
            )
        else:
            state.extend([0.0] * 10)

        state.extend(
            [
                self.portfolio_exposure / 100000.0,
                self.execution_count / 1000.0,
                self.missed_opportunities / 100.0,
                np.mean(list(self.recent_rewards)) if self.recent_rewards else 0.0,
                len(self.recent_rewards) / 100.0,
            ]
        )

        avg_latencies = []
        for venue in self.venues[:5]:
            if self.recent_latencies[venue]:
                avg_latencies.append(np.mean(list(self.recent_latencies[venue])) / 10000.0)
            else:
                avg_latencies.append(0.1)
        state.extend(avg_latencies)

        return np.array(state, dtype=np.float32)

    def _execute_route(self, venue: str) -> float:
        latency_measurement = self.network_simulator.measure_latency(
            venue, self.current_tick.timestamp
        )
        actual_latency = latency_measurement.latency_us

        if latency_measurement.packet_loss:
            self.missed_opportunities += 1
            return -10.0

        self.recent_latencies[venue].append(actual_latency)
        self.episode_latencies.append(actual_latency)

        latency_reward = self._calculate_latency_reward(actual_latency)
        opportunity_reward = self._calculate_opportunity_reward(venue)
        risk_penalty = self._calculate_risk_penalty()

        reward = latency_reward + opportunity_reward - risk_penalty

        self.execution_count += 1
        self.portfolio_exposure += self.current_tick.volume * self.current_tick.mid_price
        self.recent_rewards.append(reward)
        self.episode_rewards.append(reward)

        return reward

    def _execute_hold(self) -> float:
        opportunity_cost = -0.1
        best_venue_latency = min(self.network_simulator.get_current_latency(v) for v in self.venues)
        if best_venue_latency < 500:
            opportunity_cost = -1.0
            self.missed_opportunities += 1
        return opportunity_cost

    def _execute_cancel(self) -> float:
        return -0.5

    def _calculate_latency_reward(self, latency_us: float) -> float:
        if latency_us < 500:
            return 5.0
        elif latency_us < 1000:
            return 2.0
        elif latency_us < 2000:
            return 0.5
        else:
            return -2.0 * (latency_us / 2000.0)

    def _calculate_opportunity_reward(self, venue: str) -> float:
        if self.current_tick:
            volume_score = np.log1p(self.current_tick.volume) / 10.0
            volatility_score = self.current_tick.volatility * 100.0
            spread_score = 1.0 / (1.0 + self.current_tick.spread)
            opportunity = volume_score * volatility_score * spread_score
            return min(opportunity, 5.0)
        return 0.0

    def _calculate_risk_penalty(self) -> float:
        exposure_ratio = self.portfolio_exposure / 100000.0
        if exposure_ratio > 0.8:
            return 2.0 * exposure_ratio
        return 0.0

    def _get_next_tick(self) -> Any:
        class DummyTick:
            def __init__(self) -> None:
                self.timestamp = time.time()
                self.mid_price = 100.0 + np.random.randn() * 0.1
                self.volume = np.random.randint(100, 10000)
                self.spread = 0.01 + np.random.exponential(0.01)
                self.volatility = 0.01 + np.random.exponential(0.005)
                self.imbalance = np.random.randn() * 0.5
                self.trade_intensity = np.random.random()
                self.hour = np.random.randint(9, 16)
                self.minute = np.random.randint(0, 60)
                self.is_market_hours = True
                self.is_auction = False

        return DummyTick()
