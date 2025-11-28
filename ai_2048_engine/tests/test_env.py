from ai_2048_engine.env.game_2048_env import Game2048Env


def test_reset_initial_tiles() -> None:
    env = Game2048Env(seed=123)
    board = env.reset()
    assert len(board) == 4 and len(board[0]) == 4
    non_zero = sum(1 for row in board for val in row if val != 0)
    assert non_zero == 2


def test_step_merging_and_score() -> None:
    env = Game2048Env()
    env.board = [
        [2, 0, 0, 0],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    _, reward, _, _ = env.step(0)
    assert reward == 4
    assert env.board[0][0] == 4
    assert env.score == 4


def test_terminal_state_detection() -> None:
    env = Game2048Env()
    env.board = [
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 4, 2],
    ]
    assert env.is_terminal()
