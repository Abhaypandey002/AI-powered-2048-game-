"""DQN training loop using PyTorch."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ai_2048_engine.agents.dqn_agent import DQNAgent, DQNAgentConfig
from ai_2048_engine.env.game_2048_env import Game2048Env
from ai_2048_engine.env.utils import board_to_features


@dataclass
class Transition:
    """Single experience tuple for replay."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Simple circular replay buffer."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[int(i)] for i in indices]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.buffer)


def compute_epsilon(episode: int, eps_start: float, eps_end: float, decay_episodes: int) -> float:
    if episode >= decay_episodes:
        return eps_end
    slope = (eps_end - eps_start) / decay_episodes
    return eps_start + slope * episode


def train_dqn(
    num_episodes: int = 500,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 1e-3,
    buffer_capacity: int = 50000,
    target_update: int = 1000,
    checkpoint_path: str = "checkpoints/dqn_2048.pt",
    device: str = "cpu",
) -> DQNAgent:
    """Train a DQN agent on the 2048 environment."""
    env = Game2048Env()
    agent = DQNAgent(DQNAgentConfig(epsilon=1.0, device=device))
    target_agent = DQNAgent(DQNAgentConfig(epsilon=0.0, device=device))
    target_agent.policy_net.load_state_dict(agent.policy_net.state_dict())
    buffer = ReplayBuffer(capacity=buffer_capacity)

    optimizer = optim.Adam(agent.policy_net.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    step_count = 0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        epsilon = compute_epsilon(episode, 1.0, 0.1, max(1, num_episodes // 2))
        agent.config.epsilon = epsilon
        while not done:
            action = agent.select_action(env)
            next_state, reward, done, _ = env.step(action)
            transition = Transition(board_to_features(state), action, reward, board_to_features(next_state), done)
            buffer.add(transition)
            state = next_state
            step_count += 1

            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                states = torch.tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=device)
                actions = torch.tensor([t.action for t in batch], dtype=torch.int64, device=device)
                rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
                next_states = torch.tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32, device=device)
                dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)

                q_values = agent.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_agent.policy_net(next_states).max(dim=1).values
                targets = rewards + gamma * next_q_values * (1 - dones)

                loss = mse_loss(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if step_count % target_update == 0 and step_count > 0 and len(buffer) > 0:
                target_agent.policy_net.load_state_dict(agent.policy_net.state_dict())

        if (episode + 1) % 50 == 0:
            agent.save_checkpoint(checkpoint_file)

    agent.save_checkpoint(checkpoint_file)
    return agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent for 2048.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to train.")
    parser.add_argument(
        "--checkpoint-path", type=str, default="checkpoints/dqn_2048.pt", help="Path to save the trained model."
    )
    parser.add_argument("--device", type=str, default="cpu", help="Compute device for training.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_dqn(num_episodes=args.episodes, checkpoint_path=args.checkpoint_path, device=args.device)
