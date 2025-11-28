"""Random action agent."""
from __future__ import annotations

import random

from ai_2048_engine.agents import BaseAgent
from ai_2048_engine.env.game_2048_env import Game2048Env


class RandomAgent(BaseAgent):
    """Agent that selects moves uniformly at random among valid actions."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def select_action(self, env: Game2048Env) -> int:
        valid = env.valid_moves()
        if not valid:
            return self.rng.randrange(0, 4)
        return self.rng.choice(valid)


__all__ = ["RandomAgent"]
