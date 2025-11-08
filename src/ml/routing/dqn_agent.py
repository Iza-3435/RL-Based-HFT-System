"""DQN routing agent with double DQN and prioritized replay."""

import random
import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Tuple, List

from .neural_networks import DQNNetwork
from .replay_buffer import Experience

logger = logging.getLogger(__name__)


class DQNRouter:
    """Deep Q-Network router with double DQN and dueling architecture."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        tau: float = 0.001,
        buffer_size: int = 100000,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        try:
            from hft_rl_integration import HFTReplayBufferCpp

            self.memory = HFTReplayBufferCpp(buffer_size, alpha=0.6, beta=0.4)
            logger.info("Using C++ accelerated replay buffer")
        except ImportError:
            self.memory = deque(maxlen=buffer_size)
            logger.info("Using Python replay buffer")

        self.batch_size = 64
        self.update_every = 4
        self.t_step = 0

        self.training_losses: List[float] = []
        self.epsilon_history: List[float] = []
        self.reward_history: List[float] = []

        logger.info(f"DQNRouter initialized on {self.device}")

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> Tuple[int, float]:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        self.q_network.train()

        if random.random() > epsilon:
            action = q_values.argmax().item()
            confidence = torch.softmax(q_values, dim=1).max().item()
        else:
            action = random.randrange(self.action_size)
            confidence = 1.0 / self.action_size

        return action, confidence

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        if isinstance(self.memory, deque):
            self.memory.append(Experience(state, action, reward, next_state, done))
        else:
            self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            if isinstance(self.memory, deque):
                experiences = random.sample(list(self.memory), self.batch_size)
            else:
                experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

    def learn(self, experiences: List[Experience]) -> None:
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
        next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = F.mse_loss(current_q_values.squeeze(), target_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.soft_update()
        self.q_network.reset_noise()
        self.target_network.reset_noise()
        self.training_losses.append(loss.item())

    def soft_update(self) -> None:
        for target_param, local_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, filepath: str) -> None:
        checkpoint = {
            "q_network_state": self.q_network.state_dict(),
            "target_network_state": self.target_network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "training_losses": self.training_losses,
            "reward_history": self.reward_history,
        }
        torch.save(checkpoint, filepath)
        logger.info(f"DQNRouter saved to {filepath}")

    def load(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state"])
        self.target_network.load_state_dict(checkpoint["target_network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.training_losses = checkpoint.get("training_losses", [])
        self.reward_history = checkpoint.get("reward_history", [])
        logger.info(f"DQNRouter loaded from {filepath}")
