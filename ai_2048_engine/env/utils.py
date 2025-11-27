"""Utility functions for 2048 environment logic without external dependencies."""
from __future__ import annotations

import math
import random
from typing import List, Tuple


Action = int
Board = List[List[int]]


def _move_line_left(line: List[int]) -> tuple[List[int], int]:
    """Slide and merge a single line to the left following 2048 rules."""
    non_zero = [value for value in line if value != 0]
    merged_line: List[int] = []
    reward = 0
    idx = 0
    while idx < len(non_zero):
        current = non_zero[idx]
        if idx + 1 < len(non_zero) and non_zero[idx + 1] == current:
            merged_value = current * 2
            merged_line.append(merged_value)
            reward += merged_value
            idx += 2
        else:
            merged_line.append(current)
            idx += 1
    while len(merged_line) < len(line):
        merged_line.append(0)
    return merged_line, reward


def apply_move(board: Board, action: Action) -> tuple[Board, bool, int]:
    """Apply a move to the board."""
    if action not in (0, 1, 2, 3):
        raise ValueError("Invalid action; expected one of {0,1,2,3}.")

    new_board: Board = [row[:] for row in board]
    reward_total = 0

    if action == 0:  # up
        for col in range(4):
            column = [board[row][col] for row in range(4)]
            moved_col, reward = _move_line_left(column)
            for row in range(4):
                new_board[row][col] = moved_col[row]
            reward_total += reward
    elif action == 1:  # down
        for col in range(4):
            column = [board[row][col] for row in range(3, -1, -1)]
            moved_col, reward = _move_line_left(column)
            for idx, row in enumerate(range(3, -1, -1)):
                new_board[row][col] = moved_col[idx]
            reward_total += reward
    elif action == 2:  # left
        for row in range(4):
            moved_row, reward = _move_line_left(board[row])
            new_board[row] = moved_row
            reward_total += reward
    else:  # right
        for row in range(4):
            reversed_row = list(reversed(board[row]))
            moved_row, reward = _move_line_left(reversed_row)
            new_board[row] = list(reversed(moved_row))
            reward_total += reward

    moved = new_board != board
    return new_board, moved, reward_total


def get_empty_cells(board: Board) -> List[Tuple[int, int]]:
    """Return coordinates of empty cells on the board."""
    return [(i, j) for i in range(4) for j in range(4) if board[i][j] == 0]


def spawn_random_tile(board: Board, rng: random.Random) -> Board:
    """Spawn a random tile (2 with prob 0.9, 4 with prob 0.1) in an empty cell."""
    empty_cells = get_empty_cells(board)
    if not empty_cells:
        return board
    row, col = rng.choice(empty_cells)
    value = 4 if rng.random() < 0.1 else 2
    new_board: Board = [r[:] for r in board]
    new_board[row][col] = value
    return new_board


def board_to_features(board: Board) -> List[float]:
    """Convert the board into a flattened feature vector using log2 encoding."""
    features: List[float] = []
    for row in board:
        for value in row:
            features.append(math.log2(value) if value > 0 else 0.0)
    return features
