"""Heuristic-based 2048 agent."""
from __future__ import annotations

import math
import random
from typing import List

from ai_2048_engine.agents import BaseAgent
from ai_2048_engine.env.game_2048_env import Game2048Env
from ai_2048_engine.env.utils import apply_move


Corner_positions = [(0, 0), (0, 3), (3, 0), (3, 3)]


def evaluate_board(board: List[List[int]]) -> float:
    """Compute a heuristic score for the given board."""
    empty_cells = sum(1 for row in board for val in row if val == 0)
    smoothness = _calculate_smoothness(board)
    monotonicity = _calculate_monotonicity(board)
    max_tile = max(max(row) for row in board) if board else 0
    corner_bonus = 1.5 if any(board[i][j] == max_tile for i, j in Corner_positions) else 1.0
    score = 2.0 * empty_cells + 1.0 * monotonicity - 0.1 * smoothness
    if max_tile > 0:
        score += corner_bonus * math.log2(max_tile)
    return score


def _calculate_smoothness(board: List[List[int]]) -> float:
    diff = 0.0
    for i in range(4):
        for j in range(4):
            if board[i][j] == 0:
                continue
            for dx, dy in ((1, 0), (0, 1)):
                nx, ny = i + dx, j + dy
                if 0 <= nx < 4 and 0 <= ny < 4 and board[nx][ny] != 0:
                    diff += abs(math.log2(board[i][j]) - math.log2(board[nx][ny]))
    return diff


def _calculate_monotonicity(board: List[List[int]]) -> float:
    score = 0.0
    for i in range(4):
        row = [val for val in board[i] if val != 0]
        score += _line_monotonicity(row)
    for j in range(4):
        col = [board[i][j] for i in range(4) if board[i][j] != 0]
        score += _line_monotonicity(col)
    return score


def _line_monotonicity(line: List[int]) -> float:
    if len(line) < 2:
        return 0.0
    diffs = [math.log2(line[idx + 1]) - math.log2(line[idx]) for idx in range(len(line) - 1)]
    inc = sum(1 for d in diffs if d >= 0)
    dec = sum(1 for d in diffs if d <= 0)
    return float(max(inc, dec))


class HeuristicAgent(BaseAgent):
    """Agent that evaluates moves using a handcrafted heuristic."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def select_action(self, env: Game2048Env) -> int:
        best_score = -float("inf")
        best_action = None
        for action in (0, 1, 2, 3):
            new_board, moved, reward = apply_move(env.board, action)
            if not moved:
                continue
            score = reward + evaluate_board(new_board)
            if score > best_score:
                best_score = score
                best_action = action
        if best_action is None:
            valid = env.valid_moves()
            if not valid:
                return 0
            return self.rng.choice(valid)
        return int(best_action)


__all__ = ["HeuristicAgent", "evaluate_board"]
