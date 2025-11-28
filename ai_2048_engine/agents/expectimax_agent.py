"""Expectimax search agent."""
from __future__ import annotations

import random
from typing import Optional

from ai_2048_engine.agents import BaseAgent
from ai_2048_engine.agents.heuristic_agent import evaluate_board
from ai_2048_engine.env.game_2048_env import Game2048Env
from ai_2048_engine.env.utils import apply_move, get_empty_cells


class ExpectimaxAgent(BaseAgent):
    """Agent using an expectimax search to choose actions."""

    def __init__(self, max_depth: int = 3, seed: int | None = None) -> None:
        self.max_depth = max_depth
        self.rng = random.Random(seed)

    def select_action(self, env: Game2048Env) -> int:
        best_value = -float("inf")
        best_action: Optional[int] = None
        valid_moves = env.valid_moves()
        for action in valid_moves:
            new_board, moved, reward = apply_move(env.board, action)
            if not moved:
                continue
            value = reward + self._expectimax_value(new_board, self.max_depth - 1, False)
            if value > best_value:
                best_value = value
                best_action = action
        if best_action is None:
            return self.rng.randrange(0, 4)
        return int(best_action)

    def _expectimax_value(self, board, depth: int, is_max_node: bool) -> float:
        if depth == 0:
            return evaluate_board(board)
        valid_moves = []
        for action in (0, 1, 2, 3):
            _, moved, _ = apply_move(board, action)
            if moved:
                valid_moves.append(action)
        if not valid_moves:
            return evaluate_board(board)

        if is_max_node:
            values = []
            for action in valid_moves:
                new_board, moved, reward = apply_move(board, action)
                if not moved:
                    continue
                child_value = reward + self._expectimax_value(new_board, depth - 1, False)
                values.append(child_value)
            return max(values) if values else evaluate_board(board)

        empty_cells = get_empty_cells(board)
        if not empty_cells:
            return evaluate_board(board)
        expected_value = 0.0
        prob_per_cell = 1.0 / len(empty_cells)
        for (i, j) in empty_cells:
            for value, prob_tile in ((2, 0.9), (4, 0.1)):
                new_board = [row[:] for row in board]
                new_board[i][j] = value
                child_value = self._expectimax_value(new_board, depth - 1, True)
                expected_value += prob_per_cell * prob_tile * child_value
        return expected_value


__all__ = ["ExpectimaxAgent"]
