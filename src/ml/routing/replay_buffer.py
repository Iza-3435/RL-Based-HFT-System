"""Experience replay buffer with prioritization."""

import numpy as np
from collections import namedtuple
from typing import List


Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])


class PrioritizedReplayBuffer:
    """Prioritized experience replay for DQN."""

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001

        self.buffer: List[Experience] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = Experience(state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Experience]:
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[: self.position]

        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]

        self.beta = min(1.0, self.beta + self.beta_increment)
        return experiences

    def __len__(self) -> int:
        return len(self.buffer)
