"""Command-line interface for interacting with 2048 AI agents."""
from __future__ import annotations

import argparse

from ai_2048_engine.agents.dqn_agent import DQNAgent
from ai_2048_engine.agents.expectimax_agent import ExpectimaxAgent
from ai_2048_engine.agents.heuristic_agent import HeuristicAgent
from ai_2048_engine.agents.random_agent import RandomAgent
from ai_2048_engine.env.game_2048_env import Game2048Env
from ai_2048_engine.training.dqn_trainer import train_dqn
from ai_2048_engine.training.eval import evaluate_agent


AGENT_BUILDERS = {
    "random": lambda: RandomAgent(),
    "heuristic": lambda: HeuristicAgent(),
    "expectimax": lambda: ExpectimaxAgent(),
    "dqn": lambda: DQNAgent(),
}


def play(agent_name: str, episodes: int, render: bool) -> None:
    agent = AGENT_BUILDERS[agent_name]()
    env = Game2048Env()
    for episode in range(episodes):
        env.reset()
        done = False
        while not done:
            action = agent.select_action(env)
            _, _, done, info = env.step(action)
            if render:
                env.render()
        max_tile = max(max(row) for row in env.board)
        print(f"Episode {episode + 1} finished with score {info['score']} and max tile {max_tile}")


def eval_agent(agent_name: str, episodes: int) -> None:
    agent = AGENT_BUILDERS[agent_name]()
    results = evaluate_agent(agent, num_episodes=episodes)
    print("Evaluation results:", results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="2048 AI engine CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    play_parser = subparsers.add_parser("play", help="Play games with a selected agent")
    play_parser.add_argument("--agent", choices=list(AGENT_BUILDERS.keys()), default="heuristic")
    play_parser.add_argument("--episodes", type=int, default=1)
    play_parser.add_argument("--render", action="store_true")

    train_parser = subparsers.add_parser("train-dqn", help="Train the DQN agent")
    train_parser.add_argument("--episodes", type=int, default=500)
    train_parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/dqn_2048.pt",
        help="Path to save the trained checkpoint",
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate an agent")
    eval_parser.add_argument("--agent", choices=list(AGENT_BUILDERS.keys()), default="heuristic")
    eval_parser.add_argument("--episodes", type=int, default=20)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "play":
        play(args.agent, args.episodes, args.render)
    elif args.command == "train-dqn":
        train_dqn(num_episodes=args.episodes, checkpoint_path=args.checkpoint_path)
    elif args.command == "eval":
        eval_agent(args.agent, args.episodes)


if __name__ == "__main__":
    main()
