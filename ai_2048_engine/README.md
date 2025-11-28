# 2048 AI Engine

A modular 2048 engine with multiple AI agents (Random, Heuristic, Expectimax, DQN), a NumPy-based environment, training loop, and CLI tools.

## Structure
- `env/` – `Game2048Env` and helpers for moves, spawning, and features.
- `agents/` – Random, Heuristic, Expectimax, and PyTorch DQN agents.
- `training/` – DQN trainer with replay buffer/target network and evaluation helpers.
- `ui/` – CLI runner for play, training, and evaluation.
- `tests/` – Pytest suite for environment and agents.

## Usage
```bash
python -m ai_2048_engine.ui.cli_runner play --agent heuristic --episodes 1 --render
python -m ai_2048_engine.ui.cli_runner train-dqn --episodes 500 --checkpoint-path checkpoints/dqn_2048.pt
python -m ai_2048_engine.ui.cli_runner eval --agent expectimax --episodes 20
```

## Development
Install dependencies with `pip install -r requirements.txt` from the project root, then run `python -m pytest` to execute the tests.
