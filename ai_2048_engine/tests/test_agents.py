from ai_2048_engine.agents.expectimax_agent import ExpectimaxAgent
from ai_2048_engine.agents.heuristic_agent import HeuristicAgent
from ai_2048_engine.agents.random_agent import RandomAgent
from ai_2048_engine.env.game_2048_env import Game2048Env


def test_random_agent_action() -> None:
    env = Game2048Env(seed=0)
    env.reset()
    agent = RandomAgent(seed=0)
    action = agent.select_action(env)
    assert action in {0, 1, 2, 3}


def test_heuristic_agent_prefers_merge() -> None:
    env = Game2048Env()
    env.board = [
        [2, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    agent = HeuristicAgent()
    action = agent.select_action(env)
    assert action in {0, 2}


def test_expectimax_agent_returns_valid_move() -> None:
    env = Game2048Env(seed=1)
    env.reset()
    agent = ExpectimaxAgent(max_depth=2, seed=1)
    action = agent.select_action(env)
    assert action in {0, 1, 2, 3}
