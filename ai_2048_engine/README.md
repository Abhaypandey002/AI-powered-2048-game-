# 2048 AI Engine

## Overview
This project provides a production-ready 2048 AI engine with multiple agents ranging from simple random play to a neural network-driven DQN. It is organized with a clean, modular architecture so you can train, evaluate, and interact with the game easily.

## Tech Stack
- **Python** 3.11+
- Pure standard library implementation (no external dependencies required)

## Project Structure
- `env/` – Core game environment and board utilities.
- `agents/` – Implementations of Random, Heuristic, Expectimax, and a lightweight DQN agent.
- `training/` – DQN training loop and evaluation utilities.
- `ui/` – Command-line runner to play, train, and evaluate agents.
- `tests/` – Pytest-based unit tests for the environment and agents.

## Setup & Installation
1. Ensure Python 3.11+ is installed.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies (none required, but the command is included for consistency):
   ```bash
   pip install -r requirements.txt
   ```

## How to Run
### Play with an Agent
```bash
python -m ai_2048_engine.ui.cli_runner play --agent heuristic --episodes 5 --render
```
Available agents: `random`, `heuristic`, `expectimax`, `dqn`.

### Train the DQN Agent
```bash
python -m ai_2048_engine.ui.cli_runner train-dqn --episodes 1000 --checkpoint-path checkpoints/dqn_2048.pt
```

### Evaluate an Agent
```bash
python -m ai_2048_engine.ui.cli_runner eval --agent heuristic --episodes 20
```

## Core Logic
- **Game2048Env**: Holds the board state, score, and RNG. Supports `reset`, `step`, `valid_moves`, `is_terminal`, and `render` in `env/game_2048_env.py`.
- **Utilities**: `env/utils.py` contains pure functions for applying moves, spawning tiles, and extracting features via log2 encoding.
- **Agents**:
  - *RandomAgent*: Chooses random valid moves.
  - *HeuristicAgent*: Scores boards using empty cells, monotonicity, smoothness, and corner max-tile bonuses.
  - *ExpectimaxAgent*: Depth-limited expectimax search that accounts for tile spawn probabilities.
  - *DQNAgent*: Uses a lightweight feed-forward neural network with epsilon-greedy exploration.
- **Training**: `training/dqn_trainer.py` implements experience replay, epsilon scheduling, and periodic target updates. Checkpoints are stored under `checkpoints/`.

## Extensibility
- Add new agents by subclassing `BaseAgent` in `agents/__init__.py` and implementing `select_action`.
- Replace or augment heuristics by editing `agents/heuristic_agent.py`.
- Integrate additional UIs by reusing `Game2048Env` and the agent interfaces.

## Testing
Run the test suite with:
```bash
python -m pytest
```
