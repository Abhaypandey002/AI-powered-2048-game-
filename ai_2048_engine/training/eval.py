"""Agent evaluation utilities."""
from __future__ import annotations

from typing import Dict

import numpy as np

from ai_2048_engine.agents import BaseAgent
from ai_2048_engine.env.game_2048_env import Game2048Env


def evaluate_agent(agent: BaseAgent, num_episodes: int = 20) -> Dict[str, float]:
    """Evaluate an agent over multiple episodes."""
    scores = []
    max_tiles = []
    for _ in range(num_episodes):
        env = Game2048Env()
        env.reset()
        done = False
        while not done:
            action = agent.select_action(env)
            _, _, done, info = env.step(action)
        scores.append(info["score"])
        max_tiles.append(int(env.board.max(initial=0)))
    return {
        "average_score": float(sum(scores) / len(scores)) if scores else 0.0,
        "max_score": float(max(scores) if scores else 0),
        "average_max_tile": float(sum(max_tiles) / len(max_tiles)) if max_tiles else 0.0,
    }


__all__ = ["evaluate_agent"]
