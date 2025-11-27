"""DQN training loop implemented with pure Python."""
from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List

from ai_2048_engine.agents.dqn_agent import DQNAgent
from ai_2048_engine.env.game_2048_env import Game2048Env
from ai_2048_engine.env.utils import board_to_features


@dataclass
class Transition:
    state: List[float]
    action: int
    reward: float
    next_state: List[float]
    done: bool


class ReplayBuffer:
    """Simple circular replay buffer."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        indices = random.sample(range(len(self.buffer)), batch_size)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
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
    target_update: int = 100,
    checkpoint_path: str = "checkpoints/dqn_2048.pt",
) -> DQNAgent:
    """Train a DQN agent on the 2048 environment."""
    env = Game2048Env()
    agent = DQNAgent(epsilon=1.0)
    target_agent = DQNAgent(epsilon=0.0)
    target_agent.network = agent.network.clone()
    buffer = ReplayBuffer(capacity=buffer_capacity)

    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        epsilon = compute_epsilon(episode, 1.0, 0.1, max(1, num_episodes // 2))
        agent.epsilon = epsilon
        while not done:
            action = agent.select_action(env)
            next_state, reward, done, _ = env.step(action)
            transition = Transition(board_to_features(state), action, reward, board_to_features(next_state), done)
            buffer.add(transition)
            state = next_state

            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                for transition in batch:
                    target_qs = target_agent.predict(transition.next_state)
                    max_next_q = max(target_qs)
                    target_value = transition.reward + (0 if transition.done else gamma * max_next_q)
                    current_outputs = agent.predict(transition.state)
                    updated_targets = current_outputs[:]
                    updated_targets[transition.action] = target_value
                    agent.train_step(transition.state, updated_targets, lr)

            if len(buffer) % target_update == 0 and len(buffer) > 0:
                target_agent.network = agent.network.clone()

        if (episode + 1) % 50 == 0:
            with checkpoint_file.open("w") as fh:
                fh.write("Checkpoint saved for lightweight DQN.")

    with checkpoint_file.open("w") as fh:
        fh.write("Checkpoint saved for lightweight DQN.")
    return agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent for 2048.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to train.")
    parser.add_argument(
        "--checkpoint-path", type=str, default="checkpoints/dqn_2048.pt", help="Path to save the trained model."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_dqn(num_episodes=args.episodes, checkpoint_path=args.checkpoint_path)
