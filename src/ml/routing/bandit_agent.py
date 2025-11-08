"""Multi-armed bandit routing agent."""

import time
import random
import numpy as np
from typing import Dict, List, Tuple, Any


class MultiArmedBanditRouter:
    """Multi-armed bandit for venue selection with Thompson sampling."""

    def __init__(self, venues: List[str], exploration_rate: float = 0.1) -> None:
        self.venues = venues
        self.exploration_rate = exploration_rate

        self.alpha = {venue: 1.0 for venue in venues}
        self.beta = {venue: 1.0 for venue in venues}
        self.counts = {venue: 0 for venue in venues}
        self.values = {venue: 0.0 for venue in venues}
        self.total_rewards = {venue: 0.0 for venue in venues}
        self.selection_history: List[Dict[str, Any]] = []

    def select_venue(self, method: str = "thompson") -> Tuple[str, float]:
        if method == "thompson":
            return self._thompson_sampling()
        elif method == "ucb":
            return self._upper_confidence_bound()
        elif method == "epsilon_greedy":
            return self._epsilon_greedy()
        else:
            return random.choice(self.venues), 1.0 / len(self.venues)

    def _thompson_sampling(self) -> Tuple[str, float]:
        samples = {}
        for venue in self.venues:
            samples[venue] = np.random.beta(self.alpha[venue], self.beta[venue])

        best_venue = max(samples, key=samples.get)
        confidence = samples[best_venue]
        return best_venue, confidence

    def _upper_confidence_bound(self) -> Tuple[str, float]:
        total_counts = sum(self.counts.values())
        if total_counts == 0:
            return random.choice(self.venues), 0.0

        ucb_values = {}
        for venue in self.venues:
            if self.counts[venue] == 0:
                ucb_values[venue] = float("inf")
            else:
                exploration_term = np.sqrt(2 * np.log(total_counts) / self.counts[venue])
                ucb_values[venue] = self.values[venue] + exploration_term

        best_venue = max(ucb_values, key=ucb_values.get)
        confidence = self.values[best_venue] if self.counts[best_venue] > 0 else 0.0
        return best_venue, confidence

    def _epsilon_greedy(self) -> Tuple[str, float]:
        if random.random() < self.exploration_rate:
            return random.choice(self.venues), self.exploration_rate
        else:
            if any(self.counts[v] > 0 for v in self.venues):
                best_venue = max(self.venues, key=lambda v: self.values[v])
                return best_venue, 1.0 - self.exploration_rate
            else:
                return random.choice(self.venues), 0.5

    def update(self, venue: str, reward: float) -> None:
        self.counts[venue] += 1
        self.values[venue] += (reward - self.values[venue]) / self.counts[venue]
        self.total_rewards[venue] += reward

        if reward > 0:
            self.alpha[venue] += reward
        else:
            self.beta[venue] += abs(reward)

        self.selection_history.append(
            {
                "venue": venue,
                "reward": reward,
                "timestamp": time.time(),
            }
        )

    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            "venue_counts": self.counts.copy(),
            "venue_values": self.values.copy(),
            "total_rewards": self.total_rewards.copy(),
            "thompson_params": {
                "alpha": self.alpha.copy(),
                "beta": self.beta.copy(),
            },
        }

        total_selections = sum(self.counts.values())
        if total_selections > 0:
            stats["selection_percentages"] = {
                venue: count / total_selections * 100 for venue, count in self.counts.items()
            }

        return stats
