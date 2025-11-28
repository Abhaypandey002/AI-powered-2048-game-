"""Utility functions for 2048 environment logic using NumPy arrays."""
from __future__ import annotations

from typing import List

import numpy as np

Action = int
Board = np.ndarray


def _compress_and_merge(line: np.ndarray) -> tuple[np.ndarray, int]:
    """Slide non-zero tiles left, merging equal adjacent tiles.

    Args:
        line: One-dimensional NumPy array representing a row or column.

    Returns:
        A tuple of (merged_line, reward) where merged_line has the same length
        as input and reward is the sum of merged tile values.
    """
    non_zero = line[line != 0]
    merged: List[int] = []
    reward = 0
    skip_next = False
    for idx, value in enumerate(non_zero):
        if skip_next:
            skip_next = False
            continue
        if idx + 1 < len(non_zero) and non_zero[idx + 1] == value:
            merged_value = int(value * 2)
            merged.append(merged_value)
            reward += merged_value
            skip_next = True
        else:
            merged.append(int(value))
    merged_array = np.array(merged, dtype=np.int64)
    padding = np.zeros(len(line) - len(merged_array), dtype=np.int64)
    return np.concatenate([merged_array, padding]), reward


def apply_move(board: Board, action: Action) -> tuple[Board, bool, int]:
    """Apply a move to the board.

    Args:
        board: Current game board as a (4, 4) ndarray.
        action: Integer action in {0: up, 1: down, 2: left, 3: right}.

    Returns:
        Tuple of (new_board, moved, reward) where `moved` indicates whether the
        board changed.
    """
    if action not in (0, 1, 2, 3):
        raise ValueError("Invalid action; expected one of {0,1,2,3}.")

    board = board.astype(np.int64, copy=False)
    new_board = np.copy(board)
    reward_total = 0

    if action in (0, 1):
        for col in range(4):
            column = board[:, col]
            if action == 1:
                column = column[::-1]
            moved_col, reward = _compress_and_merge(column)
            if action == 1:
                moved_col = moved_col[::-1]
            new_board[:, col] = moved_col
            reward_total += reward
    else:
        for row in range(4):
            row_data = board[row, :]
            if action == 3:
                row_data = row_data[::-1]
            moved_row, reward = _compress_and_merge(row_data)
            if action == 3:
                moved_row = moved_row[::-1]
            new_board[row, :] = moved_row
            reward_total += reward

    moved = not np.array_equal(new_board, board)
    return new_board, moved, reward_total


def get_empty_cells(board: Board) -> list[tuple[int, int]]:
    """Return coordinates of empty cells on the board."""
    coords = list(zip(*np.where(board == 0)))
    return [(int(r), int(c)) for r, c in coords]


def spawn_random_tile(board: Board, rng: np.random.Generator) -> Board:
    """Spawn a random tile (2 with prob 0.9, 4 with prob 0.1) in an empty cell."""
    empty_cells = get_empty_cells(board)
    if not empty_cells:
        return board
    row, col = empty_cells[rng.integers(len(empty_cells))]
    value = 4 if rng.random() < 0.1 else 2
    new_board = np.copy(board)
    new_board[row, col] = value
    return new_board


def board_to_features(board: Board) -> np.ndarray:
    """Convert the board into a flattened log2 feature vector."""
    flat = board.flatten().astype(np.float32)
    with np.errstate(divide="ignore"):
        features = np.where(flat > 0, np.log2(flat), 0.0)
    return features


__all__ = [
    "Action",
    "Board",
    "apply_move",
    "spawn_random_tile",
    "get_empty_cells",
    "board_to_features",
]
