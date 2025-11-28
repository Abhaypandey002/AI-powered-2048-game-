import numpy as np

from ai_2048_engine.env.game_2048_env import Game2048Env
from ai_2048_engine.env.utils import apply_move


def test_reset_spawns_two_tiles() -> None:
    env = Game2048Env(seed=42)
    board = env.reset()
    assert board.shape == (4, 4)
    assert np.count_nonzero(board) == 2


def test_step_merges_tiles_and_updates_score() -> None:
    env = Game2048Env(seed=0)
    env.board = np.array([[2, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    _, reward, _, info = env.step(0)
    assert reward == 4
    assert info["score"] == 4
    assert env.board[0, 0] == 4 or env.board[1, 0] == 4


def test_terminal_detection() -> None:
    env = Game2048Env(seed=1)
    env.board = np.array(
        [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]
    )
    assert env.is_terminal()


def test_apply_move_merges_correctly() -> None:
    board = np.array([[2, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    new_board, moved, reward = apply_move(board, 2)
    assert moved
    assert reward == 4
    assert new_board[0, 0] == 4
