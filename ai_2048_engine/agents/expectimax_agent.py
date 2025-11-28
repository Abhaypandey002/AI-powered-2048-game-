"""Expectimax search agent for 2048."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ai_2048_engine.agents import BaseAgent
from ai_2048_engine.agents.heuristic_agent import evaluate_board
from ai_2048_engine.env.game_2048_env import Game2048Env
from ai_2048_engine.env.utils import apply_move, get_empty_cells


@dataclass
class ExpectimaxConfig:
    """Configuration for the expectimax agent."""

    max_depth: int = 3


class ExpectimaxAgent(BaseAgent):
    """Depth-limited expectimax search agent."""

    def __init__(self, config: Optional[ExpectimaxConfig] = None) -> None:
        self.config = config or ExpectimaxConfig()

    def select_action(self, env: Game2048Env) -> int:
        best_value = -float("inf")
        best_action = 0
        for action in env.valid_moves():
            new_board, moved, _ = apply_move(env.board, action)
            if not moved:
                continue
            value = self._expectimax_value(new_board, self.config.max_depth - 1, False)
            if value > best_value:
                best_value = value
                best_action = action
        return int(best_action)

    def _expectimax_value(self, board: np.ndarray, depth: int, is_max_node: bool) -> float:
        if depth == 0:
            return evaluate_board(board)
        valid_actions = [a for a in (0, 1, 2, 3) if apply_move(board, a)[1]]
        empty_cells = get_empty_cells(board)
        if not valid_actions and not empty_cells:
            return evaluate_board(board)
        if is_max_node:
            values = []
            for action in valid_actions:
                next_board, moved, reward = apply_move(board, action)
                if not moved:
                    continue
                values.append(reward + self._expectimax_value(next_board, depth - 1, False))
            return max(values) if values else evaluate_board(board)
        total_value = 0.0
        for (r, c) in empty_cells:
            for value, prob in ((2, 0.9), (4, 0.1)):
                next_board = board.copy()
                next_board[r, c] = value
                total_value += prob * self._expectimax_value(next_board, depth - 1, True)
        return total_value / float(len(empty_cells)) if empty_cells else evaluate_board(board)


__all__ = ["ExpectimaxAgent", "ExpectimaxConfig"]
