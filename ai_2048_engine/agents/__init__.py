"""Agent interfaces and utilities."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ai_2048_engine.env.game_2048_env import Game2048Env


class BaseAgent(ABC):
    """Abstract base class for 2048 agents."""

    @abstractmethod
    def select_action(self, env: Game2048Env) -> int:
        """Return an action index from {0,1,2,3}."""
        raise NotImplementedError


__all__ = ["BaseAgent"]
