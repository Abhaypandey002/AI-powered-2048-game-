"""PyTorch Deep Q-Network agent."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn

from ai_2048_engine.agents import BaseAgent
from ai_2048_engine.env.game_2048_env import Game2048Env
from ai_2048_engine.env.utils import board_to_features


class DQN(nn.Module):
    """Simple feed-forward network mapping board features to Q-values."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


@dataclass
class DQNAgentConfig:
    """Configuration for the DQN agent."""

    epsilon: float = 0.1
    device: str = "cpu"


class DQNAgent(BaseAgent):
    """Agent backed by a Deep Q-Network with epsilon-greedy policy."""

    def __init__(self, config: Optional[DQNAgentConfig] = None, seed: int | None = None) -> None:
        self.config = config or DQNAgentConfig()
        self.device = torch.device(self.config.device)
        self.rng = random.Random(seed)
        torch.manual_seed(seed or 0)
        self.policy_net = DQN().to(self.device)

    def select_action(self, env: Game2048Env) -> int:
        valid_actions = env.valid_moves()
        if not valid_actions:
            return 0
        if self.rng.random() < self.config.epsilon:
            return self.rng.choice(valid_actions)
        features = board_to_features(env.board)
        state_tensor = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
        best_action = max(valid_actions, key=lambda a: q_values[a])
        return int(best_action)

    def load_checkpoint(self, path: str | Path, map_location: str | torch.device | None = None) -> None:
        """Load model weights from a checkpoint."""
        checkpoint = torch.load(Path(path), map_location=map_location or self.device)
        self.policy_net.load_state_dict(checkpoint)

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model weights to a checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), Path(path))

    def q_values(self, features: Iterable[float]) -> torch.Tensor:
        """Compute Q-values for provided features."""
        state_tensor = torch.tensor(list(features), dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.policy_net(state_tensor)


__all__ = ["DQNAgent", "DQN", "DQNAgentConfig"]
