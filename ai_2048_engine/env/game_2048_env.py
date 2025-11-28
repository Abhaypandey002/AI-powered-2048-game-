"""2048 game environment with a minimal Gym-like API."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ai_2048_engine.env.utils import Action, Board, apply_move, board_to_features, spawn_random_tile


class Game2048Env:
    """Game environment implementing 2048 mechanics."""

    def __init__(self, seed: int | None = None) -> None:
        self.board: Board = np.zeros((4, 4), dtype=np.int64)
        self.score: int = 0
        self.rng = np.random.default_rng(seed)

    def reset(self) -> np.ndarray:
        """Reset the environment to the initial state."""
        self.board = np.zeros((4, 4), dtype=np.int64)
        self.score = 0
        self.board = spawn_random_tile(self.board, self.rng)
        self.board = spawn_random_tile(self.board, self.rng)
        return self.board.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, Dict[str, int]]:
        """Execute a move in the given direction."""
        if action not in (0, 1, 2, 3):
            raise ValueError("Action must be an integer in {0,1,2,3}.")

        new_board, moved, reward = apply_move(self.board, action)
        if moved:
            self.board = spawn_random_tile(new_board, self.rng)
            self.score += reward
        done = self.is_terminal()
        info: Dict[str, int] = {"score": self.score}
        return self.board.copy(), float(reward), done, info

    def valid_moves(self) -> List[int]:
        """Return a list of actions that change the board."""
        valid: List[int] = []
        for action in (0, 1, 2, 3):
            _, moved, _ = apply_move(self.board, action)
            if moved:
                valid.append(action)
        return valid

    def is_terminal(self) -> bool:
        """Return True if no valid moves remain."""
        return len(self.valid_moves()) == 0

    def render(self) -> None:
        """Print the current board to the console."""
        print("Score:", self.score)
        print("+----" * 4 + "+")
        for row in self.board:
            row_str = "|" + "|".join(f"{int(val):^4}" if val > 0 else "    " for val in row) + "|"
            print(row_str)
            print("+----" * 4 + "+")


__all__ = ["Game2048Env", "Action", "board_to_features"]
